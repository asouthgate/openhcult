from __future__ import annotations

import csv
import io
import logging
import time
from typing import Optional, List
from fastapi import Depends, HTTPException, Query, APIRouter

from pydantic import BaseModel


from hcultctrl import config
from hcultctrl.utils import parse_utc_ms, normalize_metadata, get_db_conn
from hcultdb import queries as database

logger = logging.getLogger(__name__)
router = APIRouter()


class ResponseCurveData(BaseModel):
    swc: List[float]
    predicted_sensor_val: List[float]
    ci_lower: List[float]
    ci_upper: List[float]
    version: str
    created_at: str


@router.post("/calibration/response_curve_lookup")
def response_curve_lookup(
    payload: ResponseCurveData, conn=Depends(get_db_conn)
):

    logger.info("POST /calibration/response_curve_lookup version=%s", payload.version)

    ins_id = database.insert_response_curve_lookup(
        conn,
        payload.swc,
        payload.predicted_sensor_val,
        payload.ci_lower,
        payload.ci_upper,
        payload.version,
        payload.created_at,
    )
    return {
        "id": ins_id
    }

