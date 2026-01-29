"""Database helpers for device registry and sensor readings."""

from datetime import datetime, timezone
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import sqlite3


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
        return sqlite3.connect(str(db_path))
    if parsed.scheme.startswith("postgres"):
        import psycopg

        return psycopg.connect(db_url)
    raise ValueError(f"Unsupported database URL: {db_url}")


def setup_db(db_url: str):
    """Create or migrate the database schema and return an open connection."""
    conn = _connect(db_url)
    cursor = conn.cursor()
    if _is_postgres(conn):
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS devices (
                id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                name TEXT,
                tag TEXT,
                address TEXT UNIQUE,
                first_seen TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                device_id INTEGER NOT NULL,
                sensor TEXT NOT NULL,
                measurement INTEGER NOT NULL,
                measurement_time_us BIGINT,
                collection_time_ms BIGINT,
                adjusted_time_ms BIGINT,
                FOREIGN KEY (device_id) REFERENCES devices(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                observed_at BIGINT NOT NULL,
                note TEXT NOT NULL,
                plant_id INTEGER REFERENCES plants(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS species (
                id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                name TEXT NOT NULL,
                common_name TEXT,
                metadata JSONB
            )
            """
        )
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS species_name_unique ON species (name)"
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS plants (
                id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                plant_name TEXT NOT NULL,
                species_id INTEGER REFERENCES species(id),
                tag TEXT,
                metadata JSONB
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS plant_sensors (
                id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                plant_id INTEGER NOT NULL REFERENCES plants(id) ON DELETE CASCADE,
                device_id INTEGER NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
                sensor TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS plant_sensors_unique
            ON plant_sensors (plant_id, device_id, sensor)
            """
        )
    else:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS devices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                tag TEXT,
                address TEXT UNIQUE,
                first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id INTEGER NOT NULL,
                sensor TEXT NOT NULL,
                measurement INTEGER NOT NULL,
                measurement_time_us INTEGER,
                collection_time_ms INTEGER,
                adjusted_time_ms INTEGER,
                FOREIGN KEY (device_id) REFERENCES devices(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at INTEGER NOT NULL,
                note TEXT NOT NULL,
                plant_id INTEGER
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS species (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                common_name TEXT,
                metadata TEXT
            )
            """
        )
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS species_name_unique ON species (name)"
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS plants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species_id INTEGER REFERENCES species(id),
                tag TEXT,
                metadata TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS plant_sensors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plant_id INTEGER NOT NULL REFERENCES plants(id) ON DELETE CASCADE,
                device_id INTEGER NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
                sensor TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS plant_sensors_unique
            ON plant_sensors (plant_id, device_id, sensor)
            """
        )
    conn.commit()
    return conn


def register_device(conn, name, address):
    """Insert or update a device row and return its device_id."""
    cursor = conn.cursor()
    placeholder = _placeholder(conn)
    cursor.execute(f"SELECT id FROM devices WHERE address = {placeholder}", (address,))
    row = cursor.fetchone()
    if row:
        device_id = row[0]
        cursor.execute(
            f"UPDATE devices SET last_seen = CURRENT_TIMESTAMP WHERE id = {placeholder}",
            (device_id,),
        )
    else:
        if _is_postgres(conn):
            cursor.execute(
                "INSERT INTO devices (name, address) VALUES (%s, %s) RETURNING id",
                (name, address),
            )
            device_id = cursor.fetchone()[0]
        else:
            cursor.execute(
                f"INSERT INTO devices (name, address) VALUES ({placeholder}, {placeholder})",
                (name, address),
            )
            device_id = cursor.lastrowid
    conn.commit()
    return device_id


def write_sensor_readings(conn, device_id, readings):
    """Insert one row per sensor reading for the given device."""
    cursor = conn.cursor()
    placeholder = _placeholder(conn)
    if isinstance(readings, dict):
        rows = [(device_id, key, value) for key, value in readings.items()]
        cursor.executemany(
            "INSERT INTO sensor_readings (device_id, sensor, measurement) "
            f"VALUES ({placeholder}, {placeholder}, {placeholder})",
            rows,
        )
    else:
        rows = [
            (
                device_id,
                sensor,
                measurement,
                timestamp_us,
                adjusted_time_ms,
                collection_time_ms,
            )
            for (
                sensor,
                measurement,
                timestamp_us,
                adjusted_time_ms,
                collection_time_ms,
            ) in readings
        ]
        cursor.executemany(
            "INSERT INTO sensor_readings "
            "(device_id, sensor, measurement, measurement_time_us, "
            "adjusted_time_ms, collection_time_ms) "
            f"VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})",
            rows,
        )
    conn.commit()


def _parse_observed_at_ms(value):
    if value.endswith("Z"):
        parsed = datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
    else:
        parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def add_observation(conn, note, observed_at=None):
    """Insert an observation and return its id."""
    cursor = conn.cursor()
    observed_at_ms = (
        _parse_observed_at_ms(observed_at)
        if observed_at is not None
        else int(time.time() * 1000)
    )
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
