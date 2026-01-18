"""FastAPI app exposing time-sliced sensor data."""

from __future__ import annotations

import csv
import io
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from . import config
from . import database


app = FastAPI(title="hcultctrl", version="0.1.0")

_db_conn = None


def _get_db():
    global _db_conn
    if _db_conn is None:
        db_path = config.get_db_path()
        _db_conn = database.connect(str(db_path))
    return _db_conn


@app.get("/")
def root():
    return {"service": "hcultctrl", "status": "ok"}


@app.get("/timeseries")
def timeseries(
    sensor: Optional[str] = None,
    device: Optional[str] = None,
    start_ms: Optional[int] = Query(default=None, ge=0),
    end_ms: Optional[int] = Query(default=None, ge=0),
    start_utc: Optional[str] = None,
    end_utc: Optional[str] = None,
    limit: int = Query(default=10000, ge=1, le=100000),
    format: str = Query(default="json", pattern="^(json|csv)$"),
):
    if start_utc and start_ms is not None:
        raise HTTPException(status_code=400, detail="Use start_ms or start_utc, not both")
    if end_utc and end_ms is not None:
        raise HTTPException(status_code=400, detail="Use end_ms or end_utc, not both")

    if start_utc:
        start_ms = _parse_utc_ms(start_utc, "start_utc")
    if end_utc:
        end_ms = _parse_utc_ms(end_utc, "end_utc")

    if start_ms is not None and end_ms is not None and start_ms > end_ms:
        raise HTTPException(status_code=400, detail="start_ms must be <= end_ms")

    rows = database.fetch_timeseries(
        _get_db(),
        sensor=sensor,
        device=device,
        start_ms=start_ms,
        end_ms=end_ms,
        limit=limit,
    )

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "device_name",
                "device_address",
                "sensor",
                "measurement",
                "measurement_time_us",
                "adjusted_time_ms",
                "collection_time_ms",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["device_name"],
                    row["device_address"],
                    row["sensor"],
                    row["measurement"],
                    row["measurement_time_us"],
                    row["adjusted_time_ms"],
                    row["collection_time_ms"],
                ]
            )
        return PlainTextResponse(output.getvalue(), media_type="text/csv")

    data = [
        {
            "device_name": row["device_name"],
            "device_address": row["device_address"],
            "sensor": row["sensor"],
            "measurement": row["measurement"],
            "measurement_time_us": row["measurement_time_us"],
            "adjusted_time_ms": row["adjusted_time_ms"],
            "collection_time_ms": row["collection_time_ms"],
        }
        for row in rows
    ]
    return {"count": len(data), "data": data}


class ObservationIn(BaseModel):
    note: str
    observed_at: Optional[str] = None


class ObservationUpdate(BaseModel):
    note: Optional[str] = None
    observed_at: Optional[str] = None


@app.post("/observations")
def create_observation(payload: ObservationIn):
    note = payload.note.strip()
    if not note:
        raise HTTPException(status_code=400, detail="note must be non-empty")
    observed_at_ms = (
        _parse_utc_ms(payload.observed_at, "observed_at")
        if payload.observed_at
        else int(time.time() * 1000)
    )
    obs_id = database.insert_observation(
        _get_db(), note=note, observed_at_ms=observed_at_ms
    )
    return {"id": obs_id, "observed_at": observed_at_ms, "note": note}


@app.get("/observations")
def list_observations(
    start_ms: Optional[int] = Query(default=None, ge=0),
    end_ms: Optional[int] = Query(default=None, ge=0),
    start_utc: Optional[str] = None,
    end_utc: Optional[str] = None,
    limit: int = Query(default=1000, ge=1, le=100000),
):
    if start_utc and start_ms is not None:
        raise HTTPException(status_code=400, detail="Use start_ms or start_utc, not both")
    if end_utc and end_ms is not None:
        raise HTTPException(status_code=400, detail="Use end_ms or end_utc, not both")

    if start_utc:
        start_ms = _parse_utc_ms(start_utc, "start_utc")
    if end_utc:
        end_ms = _parse_utc_ms(end_utc, "end_utc")

    if start_ms is not None and end_ms is not None and start_ms > end_ms:
        raise HTTPException(status_code=400, detail="start_ms must be <= end_ms")

    rows = database.fetch_observations(
        _get_db(), start_ms=start_ms, end_ms=end_ms, limit=limit
    )
    data = [
        {"id": row["id"], "observed_at": row["observed_at"], "note": row["note"]}
        for row in rows
    ]
    return {"count": len(data), "data": data}


@app.patch("/observations/{obs_id}")
def update_observation(obs_id: int, payload: ObservationUpdate):
    note = payload.note.strip() if payload.note is not None else None
    if payload.note is not None and not note:
        raise HTTPException(status_code=400, detail="note must be non-empty")
    observed_at_ms = (
        _parse_utc_ms(payload.observed_at, "observed_at")
        if payload.observed_at
        else None
    )
    try:
        database.update_observation(
            _get_db(), obs_id=obs_id, observed_at_ms=observed_at_ms, note=note
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Observation not found")
    return {"id": obs_id, "observed_at": observed_at_ms, "note": note}


def _parse_utc_ms(value: str, field: str) -> int:
    try:
        if value.endswith("Z"):
            parsed = datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
        else:
            parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {field}: {value}") from exc
