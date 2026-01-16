"""SQLite helpers for serving sensor data."""

from __future__ import annotations

import sqlite3
from typing import Iterable, Optional


def connect(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection for read queries."""
    return sqlite3.connect(db_path, check_same_thread=False)


def fetch_timeseries(
    conn: sqlite3.Connection,
    *,
    sensor: Optional[str] = None,
    device: Optional[str] = None,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: int = 10000,
) -> Iterable[sqlite3.Row]:
    """Return sensor readings matching the filter criteria."""
    conn.row_factory = sqlite3.Row
    clauses = []
    params = []
    if sensor:
        clauses.append("sensor_readings.sensor = ?")
        params.append(sensor)
    if device:
        clauses.append("(devices.name = ? OR devices.address = ?)")
        params.extend([device, device])
    if start_ms is not None:
        clauses.append("sensor_readings.adjusted_time_ms >= ?")
        params.append(start_ms)
    if end_ms is not None:
        clauses.append("sensor_readings.adjusted_time_ms <= ?")
        params.append(end_ms)

    where = ""
    if clauses:
        where = "WHERE " + " AND ".join(clauses)

    query = f"""
        SELECT
            devices.name AS device_name,
            devices.address AS device_address,
            sensor_readings.sensor,
            sensor_readings.measurement,
            sensor_readings.measurement_time_us,
            sensor_readings.adjusted_time_ms,
            sensor_readings.collection_time_ms
        FROM sensor_readings
        JOIN devices ON devices.id = sensor_readings.device_id
        {where}
        ORDER BY sensor_readings.adjusted_time_ms ASC
        LIMIT ?
    """
    params.append(limit)
    cursor = conn.execute(query, params)
    return cursor.fetchall()
