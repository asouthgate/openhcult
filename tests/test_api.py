import json
import os
import uuid
from urllib import parse, error

from test_observations import (
    test_observation_without_plant,
    test_observation_with_plant,
    test_observation_unknown_plant_returns_404,
    test_patch_observation_plant_name,
)

from test_utils import request_json

SEED_DSN = os.environ.get(
    "OPENHCULT_SEED_DSN",
    "postgresql://hcult:hcult@127.0.0.1:5432/hcult",
)


def _seed_postgres():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("psycopg is required for OPENHCULT_SEED_DSN") from exc

    with psycopg.connect(SEED_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO devices (name, address) VALUES (%s, %s) "
                "ON CONFLICT (address) DO UPDATE SET name = EXCLUDED.name "
                "RETURNING id",
                ("pytest-device", "AA:BB:CC:DD:EE:FF"),
            )
            device_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO sensor_readings
                    (device_id, sensor, measurement, voltage_mv, measurement_time_us, collection_time_ms, adjusted_time_ms)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s)
                """,
                (device_id, "sensor1", 123, 117, 0, 0, 0),
            )
            cur.execute(
                "INSERT INTO observations (observed_at, note) VALUES (%s, %s)",
                (0, "pytest seed"),
            )
        conn.commit()


def test_root_ok():
    _seed_postgres()
    payload = request_json("/status")
    assert payload["service"] == "hcultctrl"
    assert payload["status"] == "ok"


def test_timeseries_limit():
    payload = request_json("/timeseries?limit=5")
    assert "count" in payload
    assert "data" in payload
    assert isinstance(payload["data"], list)


def test_create_and_list_observations():
    note = "pytest observation"
    created = request_json("/observations", method="POST", payload={"note": note})
    assert "id" in created
    assert created["note"] == note

    listed = request_json("/observations?limit=10")
    notes = [item["note"] for item in listed.get("data", [])]
    assert note in notes


def test_postgres_sensor_readings_seeded():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("psycopg is required for OPENHCULT_SEED_DSN") from exc

    with psycopg.connect(SEED_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sensor_readings")
            count = cur.fetchone()[0]
    assert count > 0


def test_species_smoke_flow():
    name = f"pytest-species-{uuid.uuid4().hex[:8]}"
    created = request_json("/species", method="POST", payload={"name": name})
    assert created["name"] == name

    listed = request_json("/species?limit=10000")
    names = [item["name"] for item in listed.get("data", [])]
    assert name in names

    deleted = request_json(f"/species/{name}", method="DELETE")
    assert deleted["id"] == name


def test_plants_smoke_flow():
    species_name = f"pytest-species-{uuid.uuid4().hex[:8]}"
    plant_name = f"pytest-plant-{uuid.uuid4().hex[:8]}"
    created_species = request_json(
        "/species", method="POST", payload={"name": species_name}
    )
    assert created_species["name"] == species_name
    print(species_name)
    created_plant = request_json(
        "/plants",
        method="POST",
        payload={"plant_name": plant_name, "species_name": species_name},
    )
    plant_id = created_plant["id"]

    status_payload = request_json(
        f"/plants/{plant_name}/status",
        method="POST",
        payload={"status_code": "DROOPING_LEAVES", "note": "pytest"},
    )
    assert status_payload["plant_name"] == plant_name
    assert status_payload["status_code"] == "DROOPING_LEAVES"

    listed_statuses = request_json(f"/plants/{plant_name}/status?limit=10")
    codes = [item["status_code"] for item in listed_statuses.get("data", [])]
    assert "DROOPING_LEAVES" in codes

    listed = request_json("/plants?limit=10000")
    names = [item["plant_name"] for item in listed.get("data", [])]
    assert plant_name in names

    deleted = request_json(f"/plants/{plant_name}", method="DELETE")
    assert deleted["plant_name"] == plant_name

    request_json(f"/species/{species_name}", method="DELETE")


def test_devices_smoke_flow():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("psycopg is required for OPENHCULT_SEED_DSN") from exc

    address = f"AA:BB:CC:DD:{uuid.uuid4().hex[:4].upper()}"
    name = f"pytest-device-{uuid.uuid4().hex[:8]}"
    with psycopg.connect(SEED_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO devices (name, address) VALUES (%s, %s) "
                "ON CONFLICT (address) DO UPDATE SET name = EXCLUDED.name",
                (name, address),
            )
        conn.commit()

    listed = request_json("/devices?limit=10000")
    addresses = [item["address"] for item in listed.get("data", [])]
    assert address in addresses


def test_calibration_smoke_flow():

    data = {
        "swc": [0.0, 0.1, 0.5, 1.0],
        "sensor_val": [100, 200, 300, 500],
        "swc_std": [0.0, 0.1, 0.5, 1.0],
        "version": "bazbar",
        "created_at": "2026-01-01T11:20:23Z",
    }

    inserted_id = request_json(
        "/calibration/response_curve_lookup",
        method="POST",
        payload=data,
    )
    assert inserted_id
