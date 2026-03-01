from __future__ import annotations

import logging
from typing import Optional
from fastapi import Depends, HTTPException, Query, APIRouter

from pydantic import BaseModel

from hcultctrl import config
from hcultctrl.utils import parse_utc_ms, normalize_metadata, get_db_conn
from hcultdb import queries as database

logger = logging.getLogger(__name__)
router = APIRouter()



class DeviceNameUpdate(BaseModel):
    name: Optional[str] = None
@router.get("/devices")
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
@router.patch("/devices/{device_address}")
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

