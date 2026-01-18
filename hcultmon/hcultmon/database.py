"""SQLite helpers for device registry and sensor readings."""

from datetime import datetime, timezone
import sqlite3
import time


def setup_db(db_path):
    """Create or migrate the database schema and return an open connection."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
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
            note TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def register_device(conn, name, address):
    """Insert or update a device row and return its device_id."""
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM devices WHERE address = ?", (address,))
    row = cursor.fetchone()
    if row:
        device_id = row[0]
        cursor.execute(
            "UPDATE devices SET name = ?, last_seen = CURRENT_TIMESTAMP WHERE id = ?",
            (name, device_id),
        )
    else:
        cursor.execute(
            "INSERT INTO devices (name, address) VALUES (?, ?)",
            (name, address),
        )
        device_id = cursor.lastrowid
    conn.commit()
    return device_id


def write_sensor_readings(conn, device_id, readings):
    """Insert one row per sensor reading for the given device."""
    cursor = conn.cursor()
    if isinstance(readings, dict):
        rows = [(device_id, key, value) for key, value in readings.items()]
        cursor.executemany(
            "INSERT INTO sensor_readings (device_id, sensor, measurement) VALUES (?, ?, ?)",
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
            "VALUES (?, ?, ?, ?, ?, ?)",
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
    cursor.execute(
        "INSERT INTO observations (observed_at, note) VALUES (?, ?)",
        (observed_at_ms, note),
    )
    conn.commit()
    return cursor.lastrowid
