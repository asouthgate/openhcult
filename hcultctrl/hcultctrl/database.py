"""Database helpers for serving sensor data."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse, unquote


def _is_postgres(conn) -> bool:
    module = conn.__class__.__module__
    return "psycopg" in module or "psycopg2" in module


def _placeholder(conn) -> str:
    return "%s" if _is_postgres(conn) else "?"


def _connect(db_url: str):
    parsed = urlparse(db_url)
    if parsed.scheme in ("", "file", "sqlite"):
        if parsed.scheme in ("file", "sqlite"):
            db_path = Path(unquote(parsed.path))
        else:
            db_path = Path(db_url)
        return sqlite3.connect(str(db_path), check_same_thread=False)
    if parsed.scheme.startswith("postgres"):
        import psycopg

        return psycopg.connect(db_url)
    raise ValueError(f"Unsupported database URL: {db_url}")


def _fetchall_dicts(cursor) -> list[dict]:
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def connect(db_url: str):
    """Open a database connection for read queries."""
    return _connect(db_url)


def fetch_timeseries(
    conn: sqlite3.Connection,
    *,
    sensor: Optional[str] = None,
    device: Optional[str] = None,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: int = 10000,
) -> Iterable[dict]:
    """Return sensor readings matching the filter criteria."""
    clauses = []
    params = []
    placeholder = _placeholder(conn)
    if sensor:
        clauses.append(f"sensor_readings.sensor = {placeholder}")
        params.append(sensor)
    if device:
        clauses.append(f"(devices.name = {placeholder} OR devices.address = {placeholder})")
        params.extend([device, device])
    if start_ms is not None:
        clauses.append(f"sensor_readings.adjusted_time_ms >= {placeholder}")
        params.append(start_ms)
    if end_ms is not None:
        clauses.append(f"sensor_readings.adjusted_time_ms <= {placeholder}")
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
        LIMIT {placeholder}
    """
    params.append(limit)
    cursor = conn.cursor()
    cursor.execute(query, params)
    return _fetchall_dicts(cursor)


def insert_observation(conn, *, note: str, observed_at_ms: int) -> int:
    """Insert an observation and return its id."""
    cursor = conn.cursor()
    if _is_postgres(conn):
        cursor.execute(
            "INSERT INTO observations (observed_at, note) VALUES (%s, %s) RETURNING id",
            (observed_at_ms, note),
        )
        obs_id = cursor.fetchone()[0]
    else:
        placeholder = _placeholder(conn)
        cursor.execute(
            f"INSERT INTO observations (observed_at, note) VALUES ({placeholder}, {placeholder})",
            (observed_at_ms, note),
        )
        obs_id = cursor.lastrowid
    conn.commit()
    return obs_id


def fetch_observations(
    conn: sqlite3.Connection,
    *,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: int = 1000,
) -> Iterable[dict]:
    """Return observations ordered by observed_at."""
    clauses = []
    params = []
    placeholder = _placeholder(conn)
    if start_ms is not None:
        clauses.append(f"observed_at >= {placeholder}")
        params.append(start_ms)
    if end_ms is not None:
        clauses.append(f"observed_at <= {placeholder}")
        params.append(end_ms)
    where = ""
    if clauses:
        where = "WHERE " + " AND ".join(clauses)
    query = f"""
        SELECT id, observed_at, note
        FROM observations
        {where}
        ORDER BY observed_at ASC, id ASC
        LIMIT {placeholder}
    """
    params.append(limit)
    cursor = conn.cursor()
    cursor.execute(query, params)
    return _fetchall_dicts(cursor)


def update_observation(
    conn, *, obs_id: int, observed_at_ms: int | None, note: str | None
) -> None:
    """Update an observation in place."""
    fields = []
    params = []
    placeholder = _placeholder(conn)
    if observed_at_ms is not None:
        fields.append(f"observed_at = {placeholder}")
        params.append(observed_at_ms)
    if note is not None:
        fields.append(f"note = {placeholder}")
        params.append(note)
    if not fields:
        return
    params.append(obs_id)
    query = f"UPDATE observations SET {', '.join(fields)} WHERE id = {placeholder}"
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError("Observation not found")
