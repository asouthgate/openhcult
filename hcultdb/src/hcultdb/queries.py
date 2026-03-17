"""Database helpers for serving sensor data."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import time
from typing import Iterable, Optional

from .connection import (
    fetchall_dicts,
    placeholder as placeholder_for,
)

logger = logging.getLogger(__name__)


def register_device(conn, name, address):
    """Insert or update a device row and return its device_id."""
    logger.debug("Registering or updating device %s at %s", name, address)
    cursor = conn.cursor()
    placeholder = placeholder_for(conn)
    cursor.execute(f"SELECT id FROM devices WHERE address = {placeholder}", (address,))
    row = cursor.fetchone()
    if row:
        device_id = row[0]
        logger.debug("Updating last_seen for device %s at %s", name, address)
        cursor.execute(
            f"UPDATE devices SET last_seen = CURRENT_TIMESTAMP WHERE id = {placeholder}",
            (device_id,),
        )
    else:
        logger.debug("Registering new device %s at %s", name, address)
        cursor.execute(
            "INSERT INTO devices (name, address) VALUES (%s, %s) RETURNING id",
            (name, address),
        )
        device_id = cursor.fetchone()[0]
    conn.commit()
    return device_id


def write_sensor_readings(conn, device_id, readings):
    """Insert one row per sensor reading for the given device."""
    cursor = conn.cursor()
    placeholder = placeholder_for(conn)
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
    cursor.execute(
        "INSERT INTO observations (observed_at, note) VALUES (%s, %s) RETURNING id",
        (observed_at_ms, note),
    )
    obs_id = cursor.fetchone()[0]
    conn.commit()
    return obs_id


def fetch_timeseries(
    conn,
    *,
    sensor: Optional[str] = None,
    device: Optional[str] = None,
    plant: Optional[str] = None,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: int = 10000,
) -> Iterable[dict]:
    """Return sensor readings matching the filter criteria."""
    clauses = []
    params = []
    placeholder = placeholder_for(conn)
    if sensor:
        clauses.append(f"sensor_readings.sensor = {placeholder}")
        params.append(sensor)
    if device:
        clauses.append(
            f"(devices.name = {placeholder} OR devices.address = {placeholder})"
        )
        params.extend([device, device])
    if plant:
        clauses.append(f"plants.plant_name = {placeholder}")
        params.append(plant)
    if start_ms is not None:
        clauses.append(f"sensor_readings.adjusted_time_ms >= {placeholder}")
        params.append(start_ms)
    if end_ms is not None:
        clauses.append(f"sensor_readings.adjusted_time_ms <= {placeholder}")
        params.append(end_ms)

    where = ""
    if clauses:
        where = "WHERE " + " AND ".join(clauses)

    plant_join = ""
    if plant:
        plant_join = """
        JOIN plant_sensors ON plant_sensors.device_id = devices.id
            AND plant_sensors.sensor = sensor_readings.sensor
        JOIN plants ON plants.id = plant_sensors.plant_id
        """

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
        {plant_join}
        {where}
        ORDER BY sensor_readings.adjusted_time_ms ASC
        LIMIT {placeholder}
    """
    params.append(limit)
    cursor = conn.cursor()
    cursor.execute(query, params)
    return fetchall_dicts(cursor)


def insert_observation(
    conn, *, note: str, observed_at_ms: int, plant_id: int | None
) -> int:
    """Insert an observation and return its id."""
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO observations (observed_at, note, plant_id) VALUES (%s, %s, %s) RETURNING id",
        (observed_at_ms, note, plant_id),
    )
    obs_id = cursor.fetchone()[0]
    conn.commit()
    return obs_id


def fetch_observations(
    conn,
    *,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: int = 1000,
) -> Iterable[dict]:
    """Return observations ordered by observed_at."""
    clauses = []
    params = []
    placeholder = placeholder_for(conn)
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
        SELECT id, observed_at, note, plant_id
        FROM observations
        {where}
        ORDER BY observed_at ASC, id ASC
        LIMIT {placeholder}
    """
    params.append(limit)
    cursor = conn.cursor()
    cursor.execute(query, params)
    return fetchall_dicts(cursor)


def fetch_observations_for_plant(
    conn,
    *,
    plant_id: int,
    limit: int = 100,
) -> Iterable[dict]:
    """Return recent observations for a plant, newest first."""
    placeholder = placeholder_for(conn)
    query = f"""
        SELECT id, observed_at, note, plant_id
        FROM observations
        WHERE plant_id = {placeholder}
        ORDER BY observed_at DESC, id DESC
        LIMIT {placeholder}
    """
    cursor = conn.cursor()
    cursor.execute(query, [plant_id, limit])
    return fetchall_dicts(cursor)


def fetch_devices(
    conn,
    *,
    limit: int = 1000,
) -> Iterable[dict]:
    """Return device rows ordered by id."""
    placeholder = placeholder_for(conn)
    query = f"""
        SELECT id, name, tag, address, first_seen, last_seen
        FROM devices
        ORDER BY id ASC
        LIMIT {placeholder}
    """
    cursor = conn.cursor()
    cursor.execute(query, [limit])
    return fetchall_dicts(cursor)


def update_observation(
    conn,
    *,
    obs_id: int,
    observed_at_ms: int | None,
    note: str | None,
    plant_id: int | None,
) -> None:
    """Update an observation in place."""
    fields = []
    params = []
    placeholder = placeholder_for(conn)
    if observed_at_ms is not None:
        fields.append(f"observed_at = {placeholder}")
        params.append(observed_at_ms)
    if note is not None:
        fields.append(f"note = {placeholder}")
        params.append(note)
    if plant_id is not None:
        fields.append(f"plant_id = {placeholder}")
        params.append(plant_id)
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
    placeholder = placeholder_for(conn)
    cursor = conn.cursor()
    logger.info(
        "Updating device name",
        address,
        name,
    )
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
    placeholder = placeholder_for(conn)
    query = f"""
        SELECT id, name, common_name, metadata
        FROM species
        ORDER BY id ASC
        LIMIT {placeholder}
    """
    cursor = conn.cursor()
    cursor.execute(query, [limit])
    return fetchall_dicts(cursor)


def insert_species(
    conn, *, name: str, common_name: str | None, metadata: str | None
) -> int:
    """Insert a species and return its id."""
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO species (name, common_name, metadata) VALUES (%s, %s, %s) RETURNING id",
        (name, common_name, metadata),
    )
    species_id = cursor.fetchone()[0]
    conn.commit()
    return species_id


def update_species(
    conn,
    *,
    name: str | None,
    common_name: str | None,
    metadata: str | None,
) -> None:
    """Update a species row."""
    placeholder = placeholder_for(conn)
    fields = []
    params = []
    if common_name is not None:
        fields.append(f"common_name = {placeholder}")
        params.append(common_name)
    if metadata is not None:
        fields.append(f"metadata = {placeholder}")
        params.append(metadata)
    if not fields:
        return
    params.append(name)
    query = f"UPDATE species SET {', '.join(fields)} WHERE name = {placeholder}"
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError("Species not found")


def delete_species(conn, *, species_id: int) -> None:
    """Delete a species row."""
    placeholder = placeholder_for(conn)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM species WHERE id = {placeholder}", (species_id,))
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError("Species not found")


def delete_species_by_name(conn, *, name: str) -> None:
    """Delete a species row."""
    placeholder = placeholder_for(conn)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM species WHERE name = {placeholder}", (name,))
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError("Species not found")


def fetch_species_id(conn, *, name: str) -> int | None:
    """Return a species id for a given name."""
    placeholder = placeholder_for(conn)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id FROM species WHERE name = {placeholder}", (name,))
    row = cursor.fetchone()
    if row is None:
        return None
    return row[0]


def fetch_plants(
    conn, *, limit: int = 1000, include_sensors: bool = False
) -> Iterable[dict]:
    placeholder = placeholder_for(conn)

    # We use a LEFT JOIN + JSON_AGG so we don't lose plants that have 0 sensors
    if include_sensors:
        query = f"""
            SELECT 
                p.id, p.plant_name, p.species_id, s.name AS species_name, p.tag, p.metadata,
                COALESCE(json_agg(json_build_object(
                    'id', ps.id,
                    'device_address', d.address,
                    'sensor', ps.sensor
                )) FILTER (WHERE ps.id IS NOT NULL), '[]') AS sensors
            FROM plants p
            LEFT JOIN species s ON s.id = p.species_id
            LEFT JOIN plant_sensors ps ON ps.plant_id = p.id
            LEFT JOIN devices d ON ps.device_id = d.id
            GROUP BY p.id, s.name
            ORDER BY p.id ASC
            LIMIT {placeholder}
        """
    else:
        query = f"""
            SELECT p.id, p.plant_name, p.species_id, s.name AS species_name, p.tag, p.metadata
            FROM plants p
            LEFT JOIN species s ON s.id = p.species_id
            ORDER BY p.id ASC
            LIMIT {placeholder}
        """

    cursor = conn.cursor()
    cursor.execute(query, [limit])
    return fetchall_dicts(cursor)


def fetch_plant_by_name(conn, *, plant_name: str) -> dict | None:
    """Return a plant row for a given plant_name."""
    placeholder = placeholder_for(conn)
    query = f"""
        SELECT p.id, p.plant_name, p.species_id, s.name AS species_name, p.tag, p.metadata
        FROM plants p
        LEFT JOIN species s ON s.id = p.species_id
        WHERE p.plant_name = {placeholder}
    """
    cursor = conn.cursor()
    cursor.execute(query, [plant_name])
    rows = fetchall_dicts(cursor)
    if not rows:
        return None
    return rows[0]


def fetch_status_type_by_code(conn, *, code: str) -> dict | None:
    """Return a status type row for a given code."""
    placeholder = placeholder_for(conn)
    query = f"""
        SELECT id, code, label, description
        FROM status_types
        WHERE code = {placeholder}
    """
    cursor = conn.cursor()
    cursor.execute(query, [code])
    rows = fetchall_dicts(cursor)
    if not rows:
        return None
    return rows[0]


def fetch_device_by_name_or_address(conn, *, device: str) -> dict | None:
    """Return a device row for a given name or address."""
    placeholder = placeholder_for(conn)
    query = f"""
        SELECT id, name, tag, address, first_seen, last_seen
        FROM devices
        WHERE name = {placeholder} OR address = {placeholder}
    """
    cursor = conn.cursor()
    cursor.execute(query, [device, device])
    rows = fetchall_dicts(cursor)
    if not rows:
        return None
    return rows[0]


def insert_plant(
    conn,
    *,
    plant_name: str,
    species_id: int | None,
    tag: str | None,
    metadata: str | None,
) -> int:
    """Insert a plant and return its id."""
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO plants (plant_name, species_id, tag, metadata) VALUES (%s, %s, %s, %s) RETURNING id",
        (plant_name, species_id, tag, metadata),
    )
    plant_id = cursor.fetchone()[0]
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
    placeholder = placeholder_for(conn)
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
    placeholder = placeholder_for(conn)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM plants WHERE id = {placeholder}", (plant_id,))
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError("Plant not found")


def delete_plant_by_name(conn, *, plant_name: int) -> None:
    """Delete a plant row."""
    placeholder = placeholder_for(conn)
    cursor = conn.cursor()
    cursor.execute(
        f"DELETE FROM plants WHERE plant_name = {placeholder}", (plant_name,)
    )
    conn.commit()
    if cursor.rowcount == 0:
        raise ValueError("Plant not found")


def assign_plant_sensor(conn, *, plant_id: int, device_id: int, sensor: str) -> None:
    """Insert a plant-to-device sensor mapping."""
    placeholder = placeholder_for(conn)
    cursor = conn.cursor()
    query = (
        "INSERT INTO plant_sensors (plant_id, device_id, sensor) "
        "VALUES (%s, %s, %s) "
        "ON CONFLICT (plant_id, device_id, sensor) DO NOTHING"
    )
    cursor.execute(query, (plant_id, device_id, sensor))
    conn.commit()


def fetch_plant_sensors(
    conn,
    *,
    limit: int = 1000,
) -> Iterable[dict]:
    """Return plant sensor mappings with resolved plant names and device addresses."""
    placeholder = placeholder_for(conn)
    query = f"""
        SELECT 
            ps.id, 
            p.plant_name, 
            d.address AS device_address, 
            ps.sensor
        FROM plant_sensors ps
        JOIN plants p ON ps.plant_id = p.id
        JOIN devices d ON ps.device_id = d.id
        ORDER BY ps.id ASC
        LIMIT {placeholder}
    """
    cursor = conn.cursor()
    cursor.execute(query, [limit])
    return fetchall_dicts(cursor)


def insert_plant_status(
    conn,
    *,
    plant_id: int,
    status_type_id: int,
    observed_at: int,
    note: str | None,
) -> int:
    """Insert a plant status entry and return its id."""
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO plant_statuses (plant_id, status_type_id, observed_at, note) "
        "VALUES (%s, %s, %s, %s) RETURNING id",
        (plant_id, status_type_id, observed_at, note),
    )
    status_id = cursor.fetchone()[0]
    conn.commit()
    return status_id


def fetch_plant_statuses(conn, *, plant_id: int, limit: int = 100) -> Iterable[dict]:
    """Return plant statuses with status type details."""
    placeholder = placeholder_for(conn)
    query = f"""
        SELECT
            ps.id,
            ps.plant_id,
            ps.status_type_id,
            st.code AS status_code,
            st.label AS status_label,
            st.description AS status_description,
            ps.observed_at,
            ps.note,
            ps.cleared_at
        FROM plant_statuses ps
        JOIN status_types st ON st.id = ps.status_type_id
        WHERE ps.plant_id = {placeholder}
        ORDER BY ps.observed_at DESC, ps.id DESC
        LIMIT {placeholder}
    """
    cursor = conn.cursor()
    cursor.execute(query, [plant_id, limit])
    return fetchall_dicts(cursor)


def insert_response_curve_lookup(
    conn,
    swc,
    sensor_vals,
    swc_std,
    version,
    created_at,
):
    cursor = conn.cursor()
    last_id = None

    for j in range(len(swc)):
        cursor.execute(
            """INSERT INTO response_curve_lookup 
               (swc, sensor_val, swc_std, version, created_at) 
               VALUES (%s, %s, %s, %s, %s) 
               RETURNING id""",
            (swc[j], sensor_vals[j], swc_std[j], version, created_at),
        )
        last_id = cursor.fetchone()[0]

    conn.commit()
    return last_id
