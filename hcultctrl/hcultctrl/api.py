"""FastAPI app exposing time-sliced sensor data."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import config
from hcultctrl.utils import parse_utc_ms, get_db_conn
from hcultctrl.routes import plants, species
from hcultdb import queries as database


app = FastAPI(title="hcultctrl", version="0.1.0")
app.include_router(plants.router)
app.include_router(species.router)
logger = logging.getLogger(__name__)

from .routes import plants


_frontend_dir = Path(__file__).resolve().parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/frontend", StaticFiles(directory=_frontend_dir, html=True), name="frontend")



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
    conn=Depends(get_db_conn),
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
        start_ms = parse_utc_ms(start_utc, "start_utc")
    if end_utc:
        end_ms = parse_utc_ms(end_utc, "end_utc")

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


@app.post("/observations")
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


@app.get("/observations")
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


@app.get("/devices")
def list_devices(limit: int = Query(default=1000, ge=1, le=100000), conn=Depends(get_db_conn)):
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


@app.patch("/devices/{device_address}")
def update_device(device_address: str, payload: DeviceNameUpdate, conn=Depends(get_db_conn)):
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

