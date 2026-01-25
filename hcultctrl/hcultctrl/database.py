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


def fetch_devices(
    conn,
    *,
    limit: int = 1000,
) -> Iterable[dict]:
    """Return device rows ordered by id."""
    placeholder = _placeholder(conn)
    query = f"""
        SELECT id, name, tag, address, first_seen, last_seen
        FROM devices
        ORDER BY id ASC
        LIMIT {placeholder}
    """
    cursor = conn.cursor()
    cursor.execute(query, [limit])
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


def update_device_name(conn, *, address: str, name: str) -> None:
    """Update a device name by BLE address."""
    placeholder = _placeholder(conn)
    cursor = conn.cursor()
    cursor.execute(
        f"UPDATE devices SET name = {placeholder} WHERE address = {placeholder}",
        (name, address),
    )
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError("Device not found")


def fetch_species(
    conn,
    *,
    limit: int = 1000,
) -> Iterable[dict]:
    """Return species rows ordered by id."""
    placeholder = _placeholder(conn)
    query = f"""
        SELECT id, name, common_name, metadata
        FROM species
        ORDER BY id ASC
        LIMIT {placeholder}
    """
    cursor = conn.cursor()
    cursor.execute(query, [limit])
    return _fetchall_dicts(cursor)


def insert_species(
    conn, *, name: str, common_name: str | None, metadata: str | None
) -> int:
    """Insert a species and return its id."""
    cursor = conn.cursor()
    if _is_postgres(conn):
        cursor.execute(
            "INSERT INTO species (name, common_name, metadata) VALUES (%s, %s, %s) RETURNING id",
            (name, common_name, metadata),
        )
        species_id = cursor.fetchone()[0]
    else:
        placeholder = _placeholder(conn)
        cursor.execute(
            f"INSERT INTO species (name, common_name, metadata) VALUES ({placeholder}, {placeholder}, {placeholder})",
            (name, common_name, metadata),
        )
        species_id = cursor.lastrowid
    conn.commit()
    return species_id


def update_species(
    conn,
    *,
    species_id: int,
    name: str | None,
    common_name: str | None,
    metadata: str | None,
) -> None:
    """Update a species row."""
    placeholder = _placeholder(conn)
    fields = []
    params = []
    if name is not None:
        fields.append(f"name = {placeholder}")
        params.append(name)
    if common_name is not None:
        fields.append(f"common_name = {placeholder}")
        params.append(common_name)
    if metadata is not None:
        fields.append(f"metadata = {placeholder}")
        params.append(metadata)
    if not fields:
        return
    params.append(species_id)
    query = f"UPDATE species SET {', '.join(fields)} WHERE id = {placeholder}"
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError("Species not found")


def delete_species(conn, *, species_id: int) -> None:
    """Delete a species row."""
    placeholder = _placeholder(conn)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM species WHERE id = {placeholder}", (species_id,))
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError("Species not found")


def delete_species_by_name(conn, *, name: str) -> None:
    """Delete a species row."""
    placeholder = _placeholder(conn)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM species WHERE name = {placeholder}", (name,))
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError("Species not found")

def fetch_species_id(conn, *, name: str) -> int | None:
    """Return a species id for a given name."""
    placeholder = _placeholder(conn)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id FROM species WHERE name = {placeholder}", (name,))
    row = cursor.fetchone()
    if row is None:
        return None
    return row[0]


def fetch_plants(
    conn,
    *,
    limit: int = 1000,
) -> Iterable[dict]:
    """Return plant rows ordered by id."""
    placeholder = _placeholder(conn)
    query = f"""
        SELECT p.id, p.plant_name, p.species_id, s.name AS species_name, p.tag, p.metadata
        FROM plants p
        LEFT JOIN species s ON s.id = p.species_id
        ORDER BY p.id ASC
        LIMIT {placeholder}
    """
    cursor = conn.cursor()
    cursor.execute(query, [limit])
    return _fetchall_dicts(cursor)


def insert_plant(
    conn, *, plant_name: str, species_id: int | None, tag: str | None, metadata: str | None
) -> int:
    """Insert a plant and return its id."""
    cursor = conn.cursor()
    if _is_postgres(conn):
        cursor.execute(
            "INSERT INTO plants (plant_name, species_id, tag, metadata) VALUES (%s, %s, %s, %s) RETURNING id",
            (plant_name, species_id, tag, metadata),
        )
        plant_id = cursor.fetchone()[0]
    else:
        placeholder = _placeholder(conn)
        cursor.execute(
            f"INSERT INTO plants (plant_name, species_id, tag, metadata) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder})",
            (plant_name, species_id, tag, metadata),
        )
        plant_id = cursor.lastrowid
    conn.commit()
    return plant_id


def update_plant(
    conn,
    *,
    plant_id: int,
    species_id: int | None,
    tag: str | None,
    metadata: str | None,
) -> None:
    """Update a plant row."""
    placeholder = _placeholder(conn)
    fields = []
    params = []
    if species_id is not None:
        fields.append(f"species_id = {placeholder}")
        params.append(species_id)
    if tag is not None:
        fields.append(f"tag = {placeholder}")
        params.append(tag)
    if metadata is not None:
        fields.append(f"metadata = {placeholder}")
        params.append(metadata)
    if not fields:
        return
    params.append(plant_id)
    query = f"UPDATE plants SET {', '.join(fields)} WHERE id = {placeholder}"
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError("Plant not found")


def delete_plant(conn, *, plant_id: int) -> None:
    """Delete a plant row."""
    placeholder = _placeholder(conn)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM plants WHERE id = {placeholder}", (plant_id,))
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError("Plant not found")


def delete_plant_by_name(conn, *, plant_name: int) -> None:
    """Delete a plant row."""
    placeholder = _placeholder(conn)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM plants WHERE plant_name = {placeholder}", (plant_name,))
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError("Plant not found")
