import json
import os
import uuid
from urllib import parse, request


BASE_URL = os.environ.get("OPENHCULT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
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
                    (device_id, sensor, measurement, measurement_time_us, collection_time_ms, adjusted_time_ms)
                VALUES
                    (%s, %s, %s, %s, %s, %s)
                """,
                (device_id, "sensor1", 123, 0, 0, 0),
            )
            cur.execute(
                "INSERT INTO observations (observed_at, note) VALUES (%s, %s)",
                (0, "pytest seed"),
            )
        conn.commit()


def _request_json(path, method="GET", payload=None):
    url = f"{BASE_URL}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, method=method, headers=headers)
    with request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def test_root_ok():
    _seed_postgres()
    payload = _request_json("/")
    assert payload["service"] == "hcultctrl"
    assert payload["status"] == "ok"


def test_timeseries_limit():
    payload = _request_json("/timeseries?limit=5")
    assert "count" in payload
    assert "data" in payload
    assert isinstance(payload["data"], list)


def test_create_and_list_observations():
    note = "pytest observation"
    created = _request_json("/observations", method="POST", payload={"note": note})
    assert "id" in created
    assert created["note"] == note

    listed = _request_json("/observations?limit=10")
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
    created = _request_json("/species", method="POST", payload={"name": name})
    assert created["name"] == name

    listed = _request_json("/species?limit=10000")
    names = [item["name"] for item in listed.get("data", [])]
    assert name in names

    deleted = _request_json(f"/species/{name}", method="DELETE")
    assert deleted["id"] == name


def test_plants_smoke_flow():
    species_name = f"pytest-species-{uuid.uuid4().hex[:8]}"
    plant_name = f"pytest-plant-{uuid.uuid4().hex[:8]}"
    created_species = _request_json("/species", method="POST", payload={"name": species_name})
    assert created_species["name"] == species_name
    print(species_name)
    created_plant = _request_json(
        "/plants",
        method="POST",
        payload={"plant_name": plant_name, "species_name": species_name},
    )
    plant_id = created_plant["id"]

    status_payload = _request_json(
        f"/plants/{plant_name}/status",
        method="POST",
        payload={"status_code": "DROOPING_LEAVES", "note": "pytest"},
    )
    assert status_payload["plant_name"] == plant_name
    assert status_payload["status_code"] == "DROOPING_LEAVES"

    listed_statuses = _request_json(f"/plants/{plant_name}/status?limit=10")
    codes = [item["status_code"] for item in listed_statuses.get("data", [])]
    assert "DROOPING_LEAVES" in codes

    listed = _request_json("/plants?limit=10000")
    names = [item["plant_name"] for item in listed.get("data", [])]
    assert plant_name in names

    deleted = _request_json(f"/plants/{plant_name}", method="DELETE")
    assert deleted["plant_name"] == plant_name

    _request_json(f"/species/{species_name}", method="DELETE")


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

    listed = _request_json("/devices?limit=10000")
    addresses = [item["address"] for item in listed.get("data", [])]
    assert address in addresses
