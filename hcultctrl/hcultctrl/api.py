"""FastAPI app exposing time-sliced sensor data."""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from . import config
from hcultdb import queries as database


app = FastAPI(title="hcultctrl", version="0.1.0")
logger = logging.getLogger(__name__)


def _get_db_conn():
    db_url = config.get_db_url()
    conn = database.connect(db_url)
    try:
        yield conn
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


def _normalize_metadata(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


@app.get("/")
def root():
    logger.info("GET /")
    return {"service": "hcultctrl", "status": "ok"}


@app.get("/timeseries")
def timeseries(
    sensor: Optional[str] = None,
    device: Optional[str] = None,
    plant: Optional[str] = None,
    start_ms: Optional[int] = Query(default=None, ge=0),
    end_ms: Optional[int] = Query(default=None, ge=0),
    start_utc: Optional[str] = None,
    end_utc: Optional[str] = None,
    limit: int = Query(default=10000, ge=1, le=100000),
    format: str = Query(default="json", pattern="^(json|csv)$"),
    conn=Depends(_get_db_conn),
):
    logger.info(
        "GET /timeseries sensor=%s device=%s plant=%s start_ms=%s end_ms=%s start_utc=%s end_utc=%s limit=%s format=%s",
        sensor,
        device,
        plant,
        start_ms,
        end_ms,
        start_utc,
        end_utc,
        limit,
        format,
    )
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
        conn,
        sensor=sensor,
        device=device,
        plant=plant,
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
    plant_id: Optional[int] = None


class ObservationUpdate(BaseModel):
    note: Optional[str] = None
    observed_at: Optional[str] = None
    plant_id: Optional[int] = None


class DeviceNameUpdate(BaseModel):
    name: Optional[str] = None


class PlantSensorAssign(BaseModel):
    device: str
    sensor: str


class PlantStatusAssign(BaseModel):
    status_code: str
    note: Optional[str] = None


@app.post("/observations")
def create_observation(payload: ObservationIn, conn=Depends(_get_db_conn)):
    logger.info(
        "POST /observations observed_at=%s note_length=%s",
        payload.observed_at,
        len(payload.note or ""),
    )
    note = payload.note.strip()
    if not note:
        raise HTTPException(status_code=400, detail="note must be non-empty")
    observed_at_ms = (
        _parse_utc_ms(payload.observed_at, "observed_at")
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


@app.get("/observations")
def list_observations(
    start_ms: Optional[int] = Query(default=None, ge=0),
    end_ms: Optional[int] = Query(default=None, ge=0),
    start_utc: Optional[str] = None,
    end_utc: Optional[str] = None,
    limit: int = Query(default=1000, ge=1, le=100000),
    conn=Depends(_get_db_conn),
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
        start_ms = _parse_utc_ms(start_utc, "start_utc")
    if end_utc:
        end_ms = _parse_utc_ms(end_utc, "end_utc")

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


@app.get("/devices")
def list_devices(limit: int = Query(default=1000, ge=1, le=100000), conn=Depends(_get_db_conn)):
    logger.info("GET /devices limit=%s", limit)
    rows = database.fetch_devices(conn, limit=limit)
    data = [
        {
            "id": row["id"],
            "name": row["name"],
            "tag": row["tag"],
            "address": row["address"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
        }
        for row in rows
    ]
    return {"count": len(data), "data": data}


@app.patch("/observations/{obs_id}")
def update_observation(obs_id: int, payload: ObservationUpdate, conn=Depends(_get_db_conn)):
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
        _parse_utc_ms(payload.observed_at, "observed_at")
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


@app.patch("/devices/{device_address}")
def update_device(device_address: str, payload: DeviceNameUpdate, conn=Depends(_get_db_conn)):
    logger.info(
        "PATCH /devices/%s name_set=%s name_value=%s",
        device_address,
        payload.name is not None,
        payload.name,
    )
    if payload.name is None:
        raise HTTPException(status_code=400, detail="name must be non-empty")
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must be non-empty")
    try:
        database.update_device_name(conn, address=device_address, name=name)
    except ValueError:
        raise HTTPException(status_code=404, detail="Device not found")
    return {"address": device_address, "name": name}


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


class SpeciesIn(BaseModel):
    name: str
    common_name: Optional[str] = None
    metadata: Optional[dict] = None


class SpeciesUpdate(BaseModel):
    name: Optional[str] = None
    common_name: Optional[str] = None
    metadata: Optional[dict] = None


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


@app.get("/species")
def list_species(limit: int = Query(default=1000, ge=1, le=100000), conn=Depends(_get_db_conn)):
    logger.info("GET /species limit=%s", limit)
    rows = database.fetch_species(conn, limit=limit)
    data = [
        {
            "id": row["id"],
            "name": row["name"],
            "common_name": row["common_name"],
            "metadata": _normalize_metadata(row["metadata"]),
        }
        for row in rows
    ]
    return {"count": len(data), "data": data}


@app.post("/species")
def create_species(payload: SpeciesIn, conn=Depends(_get_db_conn)):
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


@app.patch("/species/{species_name}")
def update_species(species_name: str, payload: SpeciesUpdate, conn=Depends(_get_db_conn)):
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


@app.delete("/species/{species_name}")
def delete_species(species_name: str, conn=Depends(_get_db_conn)):
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


@app.get("/plants")
def list_plants(limit: int = Query(default=1000, ge=1, le=100000), conn=Depends(_get_db_conn)):
    logger.info("GET /plants limit=%s", limit)
    rows = database.fetch_plants(conn, limit=limit)
    data = [
        {
            "id": row["id"],
            "plant_name": row["plant_name"],
            "species_id": row["species_id"],
            "species_name": row["species_name"],
            "tag": row["tag"],
            "metadata": _normalize_metadata(row["metadata"]),
        }
        for row in rows
    ]
    return {"count": len(data), "data": data}


@app.post("/plants")
def create_plant(payload: PlantIn, conn=Depends(_get_db_conn)):
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


@app.post("/plants/{plant_name}/assign")
def assign_plant_sensor(
    plant_name: str, payload: PlantSensorAssign, conn=Depends(_get_db_conn)
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


@app.patch("/plants/{plant_id}")
def update_plant(plant_id: int, payload: PlantUpdate, conn=Depends(_get_db_conn)):
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


@app.post("/plants/{plant_name}/status")
def assign_plant_status(
    plant_name: str, payload: PlantStatusAssign, conn=Depends(_get_db_conn)
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
    observed_at = int(time.time() * 1000)
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


@app.get("/plants/{plant_name}/status")
def list_plant_statuses(
    plant_name: str,
    limit: int = Query(default=100, ge=1, le=1000),
    conn=Depends(_get_db_conn),
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


@app.delete("/plants/{plant_name}")
def delete_plant(plant_name: str, conn=Depends(_get_db_conn)):
    logger.info("DELETE /plants/%s", plant_name)
    try:
        database.delete_plant_by_name(conn, plant_name=plant_name)
    except ValueError:
        raise HTTPException(status_code=404, detail="Plant not found")
    return {"plant_name": plant_name}


@app.get("/plants/{plant_name}/health")
def plant_health(plant_name: str, conn=Depends(_get_db_conn)):
    logger.info("GET /plants/%s/health", plant_name)
    plant = database.fetch_plant_by_name(conn, plant_name=plant_name)
    if plant is None:
        raise HTTPException(status_code=404, detail="Plant not found")
    return _build_health_payload(conn, plant)


@app.post("/plants/{plant_name}/health")
def plant_health_post(plant_name: str, conn=Depends(_get_db_conn)):
    return plant_health(plant_name, conn=conn)


@app.get("/plants/health")
def plant_health_all(conn=Depends(_get_db_conn)):
    logger.info("GET /plants/health")
    rows = database.fetch_plants(conn, limit=10000)
    data = [_build_health_payload(conn, row) for row in rows]
    return {"count": len(data), "data": data}


@app.post("/plants/health")
def plant_health_all_post(conn=Depends(_get_db_conn)):
    return plant_health_all(conn=conn)


def _build_health_payload(conn, plant: dict) -> dict:
    recent = list(
        database.fetch_observations_for_plant(conn, plant_id=plant["id"], limit=10)
    )
    statuses = list(
        database.fetch_plant_statuses(conn, plant_id=plant["id"], limit=50)
    )
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
