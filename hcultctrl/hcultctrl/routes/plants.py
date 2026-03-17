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


class PlantIn(BaseModel):
    plant_name: str
    species_id: Optional[int] = None
    species_name: Optional[str] = None
    tag: Optional[str] = None
    metadata: Optional[dict] = None


class PlantUpdate(BaseModel):
    species_id: Optional[int] = None
    species_name: Optional[str] = None
    tag: Optional[str] = None
    metadata: Optional[dict] = None


class PlantSensorAssign(BaseModel):
    device: str
    sensor: str


class PlantStatusAssign(BaseModel):
    status_code: str
    note: Optional[str] = None
    observed_at: Optional[str] = None


@router.get("/plants")
def list_plants(
    limit: int = Query(default=1000, ge=1, le=100000),
    include: str | None = Query(
        default=None,
        description="Comma-separated list of relations to include (e.g. 'sensors')",
    ),
    conn=Depends(get_db_conn),
):
    logger.info("GET /plants limit=%s include=%s", limit, include)

    # Parse the include string
    include_list = include.split(",") if include else []
    include_sensors = "sensors" in include_list

    rows = database.fetch_plants(conn, limit=limit, include_sensors=include_sensors)

    data = []
    for row in rows:
        plant = {
            "id": row["id"],
            "plant_name": row["plant_name"],
            "species_id": row["species_id"],
            "species_name": row["species_name"],
            "tag": row["tag"],
            "metadata": normalize_metadata(row["metadata"]),
        }
        # Only add the key if it was requested
        if include_sensors:
            plant["sensors"] = row["sensors"]

        data.append(plant)

    return {"count": len(data), "data": data}


@router.post("/plants")
def create_plant(payload: PlantIn, conn=Depends(get_db_conn)):
    logger.info(
        "POST /plants plant_name=%s species_id=%s species_name=%s tag=%s metadata_set=%s",
        payload.plant_name,
        payload.species_id,
        payload.species_name,
        payload.tag,
        payload.metadata is not None,
    )
    metadata = json.dumps(payload.metadata) if payload.metadata is not None else None
    plant_name = payload.plant_name
    species_id = payload.species_id
    if payload.species_name:
        species_id = database.fetch_species_id(conn, name=payload.species_name)
        if species_id is None:
            raise HTTPException(status_code=404, detail="Species not found")
    try:
        plant_id = database.insert_plant(
            conn,
            plant_name=plant_name,
            species_id=species_id,
            tag=payload.tag,
            metadata=metadata,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "id": plant_id,
        "species_id": species_id,
        "tag": payload.tag,
        "metadata": payload.metadata,
    }


@router.post("/plants/{plant_name}/assign")
def assign_plant_sensor(
    plant_name: str, payload: PlantSensorAssign, conn=Depends(get_db_conn)
):
    logger.info(
        "POST /plants/%s/assign device=%s sensor=%s",
        plant_name,
        payload.device,
        payload.sensor,
    )
    plant = database.fetch_plant_by_name(conn, plant_name=plant_name)
    if plant is None:
        raise HTTPException(status_code=404, detail="Plant not found")
    device = database.fetch_device_by_name_or_address(conn, device=payload.device)
    if device is None:
        raise HTTPException(status_code=404, detail="Device not found")
    sensor = payload.sensor.strip()
    if not sensor:
        raise HTTPException(status_code=400, detail="sensor must be non-empty")
    database.assign_plant_sensor(
        conn,
        plant_id=plant["id"],
        device_id=device["id"],
        sensor=sensor,
    )
    return {
        "plant_id": plant["id"],
        "plant_name": plant["plant_name"],
        "device_id": device["id"],
        "device_name": device.get("name"),
        "device_address": device.get("address"),
        "sensor": sensor,
    }


@router.patch("/plants/{plant_id}")
def update_plant(plant_id: int, payload: PlantUpdate, conn=Depends(get_db_conn)):
    logger.info(
        "PATCH /plants/%s species_id=%s species_name=%s tag=%s metadata_set=%s",
        plant_id,
        payload.species_id,
        payload.species_name,
        payload.tag,
        payload.metadata is not None,
    )
    metadata = json.dumps(payload.metadata) if payload.metadata is not None else None
    species_id = payload.species_id
    if payload.species_name:
        species_id = database.fetch_species_id(conn, name=payload.species_name)
        if species_id is None:
            raise HTTPException(status_code=404, detail="Species not found")
    try:
        database.update_plant(
            conn,
            plant_id=plant_id,
            species_id=species_id,
            tag=payload.tag,
            metadata=metadata,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Plant not found")
    return {
        "id": plant_id,
        "species_id": species_id,
        "tag": payload.tag,
        "metadata": payload.metadata,
    }


@router.post("/plants/{plant_name}/status")
def assign_plant_status(
    plant_name: str, payload: PlantStatusAssign, conn=Depends(get_db_conn)
):
    logger.info("POST /plants/%s/status status=%s", plant_name, payload.status_code)
    plant = database.fetch_plant_by_name(conn, plant_name=plant_name)
    if plant is None:
        raise HTTPException(status_code=404, detail="Plant not found")
    status_code = payload.status_code.strip()
    if not status_code:
        raise HTTPException(status_code=400, detail="status_code must be non-empty")
    status_type = database.fetch_status_type_by_code(conn, code=status_code)
    if status_type is None:
        raise HTTPException(status_code=404, detail="Status type not found")
    observed_at = (
        _parse_utc_ms(payload.observed_at, "observed_at")
        if payload.observed_at
        else int(time.time() * 1000)
    )
    status_id = database.insert_plant_status(
        conn,
        plant_id=plant["id"],
        status_type_id=status_type["id"],
        observed_at=observed_at,
        note=payload.note.strip() if payload.note else None,
    )
    return {
        "id": status_id,
        "plant_id": plant["id"],
        "plant_name": plant["plant_name"],
        "status_code": status_type["code"],
        "status_label": status_type["label"],
        "observed_at": observed_at,
        "note": payload.note.strip() if payload.note else None,
    }


@router.get("/plants/{plant_name}/status")
def list_plant_statuses(
    plant_name: str,
    limit: int = Query(default=100, ge=1, le=1000),
    conn=Depends(get_db_conn),
):
    logger.info("GET /plants/%s/status limit=%s", plant_name, limit)
    plant = database.fetch_plant_by_name(conn, plant_name=plant_name)
    if plant is None:
        raise HTTPException(status_code=404, detail="Plant not found")
    rows = database.fetch_plant_statuses(conn, plant_id=plant["id"], limit=limit)
    data = [
        {
            "id": row["id"],
            "plant_id": row["plant_id"],
            "status_code": row["status_code"],
            "status_label": row["status_label"],
            "status_description": row.get("status_description"),
            "observed_at": row["observed_at"],
            "note": row.get("note"),
            "cleared_at": row.get("cleared_at"),
        }
        for row in rows
    ]
    return {"count": len(data), "data": data}


@router.delete("/plants/{plant_name}")
def delete_plant(plant_name: str, conn=Depends(get_db_conn)):
    logger.info("DELETE /plants/%s", plant_name)
    try:
        database.delete_plant_by_name(conn, plant_name=plant_name)
    except ValueError:
        raise HTTPException(status_code=404, detail="Plant not found")
    return {"plant_name": plant_name}


@router.get("/plants/{plant_name}/health")
def plant_health(plant_name: str, conn=Depends(get_db_conn)):
    logger.info("GET /plants/%s/health", plant_name)
    plant = database.fetch_plant_by_name(conn, plant_name=plant_name)
    if plant is None:
        raise HTTPException(status_code=404, detail="Plant not found")
    return _build_health_payload(conn, plant)


@router.post("/plants/{plant_name}/health")
def plant_health_post(plant_name: str, conn=Depends(get_db_conn)):
    return plant_health(plant_name, conn=conn)


@router.get("/plants/health")
def plant_health_all(conn=Depends(get_db_conn)):
    logger.info("GET /plants/health")
    rows = database.fetch_plants(conn, limit=10000)
    data = [_build_health_payload(conn, row) for row in rows]
    return {"count": len(data), "data": data}


@router.post("/plants/health")
def plant_health_all_post(conn=Depends(get_db_conn)):
    return plant_health_all(conn=conn)


def _build_health_payload(conn, plant: dict) -> dict:
    recent = list(
        database.fetch_observations_for_plant(conn, plant_id=plant["id"], limit=10)
    )
    statuses = list(database.fetch_plant_statuses(conn, plant_id=plant["id"], limit=50))
    return {
        "plant_name": plant["plant_name"],
        "recent_observations": recent,
        "statuses": [
            {
                "id": row["id"],
                "status_code": row["status_code"],
                "status_label": row["status_label"],
                "status_description": row.get("status_description"),
                "observed_at": row["observed_at"],
                "note": row.get("note"),
                "cleared_at": row.get("cleared_at"),
            }
            for row in statuses
        ],
    }
