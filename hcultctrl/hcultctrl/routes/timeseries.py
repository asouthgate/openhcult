from __future__ import annotations

import csv
import io
import logging
from typing import Optional
from fastapi import Depends, HTTPException, Query, APIRouter


from hcultctrl.utils import parse_utc_ms, get_db_conn
from hcultdb import queries as database

logger = logging.getLogger(__name__)
router = APIRouter()



@router.get("/timeseries")
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

