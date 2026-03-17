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


@router.get("/plant_sensors")
def list_plant_sensors(
    limit: int = Query(default=1000, ge=1, le=100000), conn=Depends(get_db_conn)
):
    logger.info("GET /plant_sensors limit=%s", limit)

    rows = database.fetch_plant_sensors(conn, limit=limit)

    data = [
        {
            "id": row["id"],
            "plant_name": row["plant_name"],
            "device_address": row["device_address"],
            "sensor": row["sensor"],
        }
        for row in rows
    ]

    return {"count": len(data), "data": data}
