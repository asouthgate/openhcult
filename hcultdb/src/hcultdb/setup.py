"""Database setup helpers."""
from pathlib import Path

import sqlite3

from .connection import connect, is_postgres, placeholder as placeholder_for


def _connect(db_url: str):
    return connect(db_url)


def setup_db(db_url: str):
    """Create or migrate the database schema and return an open connection."""
    conn = _connect(db_url)
    cursor = conn.cursor()
    if is_postgres(conn):
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
            CREATE UNIQUE INDEX IF NOT EXISTS observations_unique
            ON observations (observed_at, note, COALESCE(plant_id, -1))
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS observation_types (
                id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                code TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                description TEXT
            )
            """
        )
        cursor.execute(
            """
            ALTER TABLE observations
            ADD COLUMN IF NOT EXISTS observation_type_id INTEGER
            REFERENCES observation_types(id)
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS status_types (
                id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                code TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                description TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS plant_statuses (
                id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                plant_id INTEGER NOT NULL REFERENCES plants(id) ON DELETE CASCADE,
                status_type_id INTEGER NOT NULL REFERENCES status_types(id),
                observed_at BIGINT NOT NULL,
                note TEXT,
                cleared_at BIGINT
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
                plant_name TEXT NOT NULL,
                species_id INTEGER REFERENCES species(id),
                tag TEXT,
                metadata TEXT
            )
            """
        )
        try:
            cursor.execute(
                """
                ALTER TABLE plants
                ADD COLUMN plant_name TEXT
                """
            )
        except sqlite3.OperationalError:
            pass
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
            CREATE UNIQUE INDEX IF NOT EXISTS observations_unique
            ON observations (observed_at, note, COALESCE(plant_id, -1))
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS observation_types (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                description TEXT
            )
            """
        )
        try:
            cursor.execute(
                """
                ALTER TABLE observations
                ADD COLUMN observation_type_id INTEGER
                REFERENCES observation_types(id)
                """
            )
        except sqlite3.OperationalError:
            pass
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS status_types (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                description TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS plant_statuses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plant_id INTEGER NOT NULL REFERENCES plants(id) ON DELETE CASCADE,
                status_type_id INTEGER NOT NULL REFERENCES status_types(id),
                observed_at INTEGER NOT NULL,
                note TEXT,
                cleared_at INTEGER
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
    _load_status_types(conn)
    _load_observation_types(conn)
    return conn


def _load_status_types(conn) -> None:
    status_path = Path(__file__).resolve().parent / "status_types.txt"
    if not status_path.exists():
        return
    placeholder = placeholder_for(conn)
    rows = []
    for raw in status_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|", maxsplit=2)]
        code = parts[0]
        label = parts[1] if len(parts) > 1 and parts[1] else code
        description = parts[2] if len(parts) > 2 and parts[2] else None
        rows.append((code, label, description))
    if not rows:
        return
    cursor = conn.cursor()
    if is_postgres(conn):
        cursor.executemany(
            "INSERT INTO status_types (code, label, description) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (code) DO UPDATE "
            "SET label = EXCLUDED.label, description = EXCLUDED.description",
            rows,
        )
    else:
        cursor.executemany(
            f"INSERT INTO status_types (code, label, description) "
            f"VALUES ({placeholder}, {placeholder}, {placeholder}) "
            "ON CONFLICT(code) DO UPDATE "
            "SET label = excluded.label, description = excluded.description",
            rows,
        )
    conn.commit()


def _load_observation_types(conn) -> None:
    types_path = Path(__file__).resolve().parent / "observation_types.txt"
    if not types_path.exists():
        return
    placeholder = placeholder_for(conn)
    rows = []
    for raw in types_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|", maxsplit=2)]
        code = parts[0]
        label = parts[1] if len(parts) > 1 and parts[1] else code
        description = parts[2] if len(parts) > 2 and parts[2] else None
        rows.append((code, label, description))
    if not rows:
        return
    cursor = conn.cursor()
    if is_postgres(conn):
        cursor.executemany(
            "INSERT INTO observation_types (code, label, description) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (code) DO UPDATE "
            "SET label = EXCLUDED.label, description = EXCLUDED.description",
            rows,
        )
    else:
        cursor.executemany(
            f"INSERT INTO observation_types (code, label, description) "
            f"VALUES ({placeholder}, {placeholder}, {placeholder}) "
            "ON CONFLICT(code) DO UPDATE "
            "SET label = excluded.label, description = excluded.description",
            rows,
        )
    conn.commit()
