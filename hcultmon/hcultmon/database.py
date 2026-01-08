"""SQLite helpers for device registry and sensor readings."""

import sqlite3


def setup_db(db_path):
    """Create or migrate the database schema and return an open connection."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
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
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            sensor TEXT NOT NULL,
            measurement INTEGER NOT NULL,
            measurement_time_us INTEGER,
            FOREIGN KEY (device_id) REFERENCES devices(id)
        )
        """
    )
    cursor.execute("PRAGMA table_info(sensor_readings)")
    columns = {row[1] for row in cursor.fetchall()}
    if "measurement_time_us" not in columns:
        cursor.execute(
            "ALTER TABLE sensor_readings ADD COLUMN measurement_time_us INTEGER"
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
        rows = [(device_id, sensor, measurement, timestamp_us)
                for sensor, measurement, timestamp_us in readings]
        cursor.executemany(
            "INSERT INTO sensor_readings "
            "(device_id, sensor, measurement, measurement_time_us) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
    conn.commit()
