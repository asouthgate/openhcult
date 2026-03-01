from __future__ import annotations

import json
import logging
import time
from typing import Optional
from fastapi import Depends, HTTPException, Query, APIRouter

from pydantic import BaseModel

from hcultctrl import config
from hcultctrl.utils import parse_utc_ms, normalize_metadata, get_db_conn
from hcultdb import queries as database

logger = logging.getLogger(__name__)
router = APIRouter()


class ObservationIn(BaseModel):
    note: str
    observed_at: Optional[str] = None
    plant_id: Optional[int] = None


class ObservationUpdate(BaseModel):
    note: Optional[str] = None
    observed_at: Optional[str] = None
    plant_id: Optional[int] = None


@router.post("/observations")
def create_observation(payload: ObservationIn, conn=Depends(get_db_conn)):
    logger.info(
        "POST /observations observed_at=%s note_length=%s",
        payload.observed_at,
        len(payload.note or ""),
    )
    note = payload.note.strip()
    if not note:
        raise HTTPException(status_code=400, detail="note must be non-empty")
    observed_at_ms = (
        parse_utc_ms(payload.observed_at, "observed_at")
        if payload.observed_at
        else int(time.time() * 1000)
    )
    obs_id = database.insert_observation(
        conn,
        note=note,
        observed_at_ms=observed_at_ms,
        plant_id=payload.plant_id,
    )
    return {
        "id": obs_id,
        "observed_at": observed_at_ms,
        "note": note,
        "plant_id": payload.plant_id,
    }


@router.get("/observations")
def list_observations(
    start_ms: Optional[int] = Query(default=None, ge=0),
    end_ms: Optional[int] = Query(default=None, ge=0),
    start_utc: Optional[str] = None,
    end_utc: Optional[str] = None,
    limit: int = Query(default=1000, ge=1, le=100000),
    conn=Depends(get_db_conn),
):
    logger.info(
        "GET /observations start_ms=%s end_ms=%s start_utc=%s end_utc=%s limit=%s",
        start_ms,
        end_ms,
        start_utc,
        end_utc,
        limit,
    )
    if start_utc and start_ms is not None:
        raise HTTPException(status_code=400, detail="Use start_ms or start_utc, not both")
    if end_utc and end_ms is not None:
        raise HTTPException(status_code=400, detail="Use end_ms or end_utc, not both")

    if start_utc:
        start_ms = parse_utc_ms(start_utc, "start_utc")
    if end_utc:
        end_ms = parse_utc_ms(end_utc, "end_utc")

    if start_ms is not None and end_ms is not None and start_ms > end_ms:
        raise HTTPException(status_code=400, detail="start_ms must be <= end_ms")

    rows = database.fetch_observations(conn, start_ms=start_ms, end_ms=end_ms, limit=limit)
    data = [
        {
            "id": row["id"],
            "observed_at": row["observed_at"],
            "note": row["note"],
            "plant_id": row.get("plant_id"),
        }
        for row in rows
    ]
    return {"count": len(data), "data": data}


@router.patch("/observations/{obs_id}")
def update_observation(obs_id: int, payload: ObservationUpdate, conn=Depends(get_db_conn)):
    logger.info(
        "PATCH /observations/%s observed_at=%s note_set=%s",
        obs_id,
        payload.observed_at,
        payload.note is not None,
    )
    note = payload.note.strip() if payload.note is not None else None
    if payload.note is not None and not note:
        raise HTTPException(status_code=400, detail="note must be non-empty")
    observed_at_ms = (
        parse_utc_ms(payload.observed_at, "observed_at")
        if payload.observed_at
        else None
    )
    try:
        database.update_observation(
            conn,
            obs_id=obs_id,
            observed_at_ms=observed_at_ms,
            note=note,
            plant_id=payload.plant_id,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Observation not found")
    return {
        "id": obs_id,
        "observed_at": observed_at_ms,
        "note": note,
        "plant_id": payload.plant_id,
    }
