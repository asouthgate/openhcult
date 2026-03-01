from __future__ import annotations

import json
import logging
from typing import Optional
from fastapi import Depends, HTTPException, Query, APIRouter

from pydantic import BaseModel

from hcultctrl import config
from hcultctrl.utils import parse_utc_ms, normalize_metadata, get_db_conn
from hcultdb import queries as database

logger = logging.getLogger(__name__)
router = APIRouter()

class SpeciesIn(BaseModel):
    name: str
    common_name: Optional[str] = None
    metadata: Optional[dict] = None


class SpeciesUpdate(BaseModel):
    name: Optional[str] = None
    common_name: Optional[str] = None
    metadata: Optional[dict] = None





@router.get("/species")
def list_species(limit: int = Query(default=1000, ge=1, le=100000), conn=Depends(get_db_conn)):
    logger.info("GET /species limit=%s", limit)
    rows = database.fetch_species(conn, limit=limit)
    data = [
        {
            "id": row["id"],
            "name": row["name"],
            "common_name": row["common_name"],
            "metadata": normalize_metadata(row["metadata"]),
        }
        for row in rows
    ]
    return {"count": len(data), "data": data}


@router.post("/species")
def create_species(payload: SpeciesIn, conn=Depends(get_db_conn)):
    logger.info(
        "POST /species name=%s common_name=%s metadata_set=%s",
        payload.name,
        payload.common_name,
        payload.metadata is not None,
    )
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must be non-empty")
    metadata = json.dumps(payload.metadata) if payload.metadata is not None else None
    try:
        species_id = database.insert_species(
            conn,
            name=name,
            common_name=payload.common_name,
            metadata=metadata,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "unique" in message or "duplicate" in message:
            raise HTTPException(status_code=409, detail="Species already exists") from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": species_id, "name": name, "common_name": payload.common_name, "metadata": payload.metadata}


@router.patch("/species/{species_name}")
def update_species(species_name: str, payload: SpeciesUpdate, conn=Depends(get_db_conn)):
    logger.info(
        "PATCH /species/%s name_set=%s common_name_set=%s metadata_set=%s",
        species_name,
        payload.name is not None,
        payload.common_name is not None,
        payload.metadata is not None,
    )
    name = payload.name.strip() if payload.name is not None else None
    if payload.name is not None and not name:
        raise HTTPException(status_code=400, detail="name must be non-empty")
    metadata = json.dumps(payload.metadata) if payload.metadata is not None else None
    try:
        database.update_species(
            conn,
            name=species_name,
            common_name=payload.common_name,
            metadata=metadata,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Species not found")
    return {"species_name": species_name, "name": name, "common_name": payload.common_name, "metadata": payload.metadata}


@router.delete("/species/{species_name}")
def delete_species(species_name: str, conn=Depends(get_db_conn)):
    logger.info("DELETE /species/%s", species_name)
    try:
        database.delete_species_by_name(conn, name=species_name)
    except ValueError:
        raise HTTPException(status_code=404, detail="Species not found")
    except Exception as exc:
        message = str(exc).lower()
        if "foreign key" in message or "violates" in message:
            raise HTTPException(
                status_code=409, detail="Species has plants; delete plants first"
            ) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": species_name}
