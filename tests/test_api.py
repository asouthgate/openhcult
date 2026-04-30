import uuid

from hcultdb import queries as database

from test_observations import (
    test_observation_without_plant,
    test_observation_with_plant,
    test_observation_unknown_plant_returns_404,
    test_patch_observation_plant_name,
    test_delete_observation,
)
from test_water_calibration import (
    test_water_calibration_no_data,
    test_water_calibration_with_waterings,
)

from test_utils import request_json


def _seed_db(conn):
    device_id = database.register_device(conn, "pytest-device", "AA:BB:CC:DD:EE:FF")
    database.write_sensor_readings(
        conn,
        device_id=device_id,
        readings=[("sensor1", 123, 117, 0, 0, 0)],
    )
    database.insert_observation(
        conn, note="pytest seed", observed_at_ms=0, plant_name=None
    )


def test_root_ok(db_conn):
    _seed_db(db_conn)
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

    listed = request_json("/observations?limit=10000")
    notes = [item["note"] for item in listed.get("data", [])]
    assert note in notes


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
    request_json(
        "/plants",
        method="POST",
        payload={"plant_name": plant_name, "species_name": species_name},
    )
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


def test_timeseries_respects_assignment_window(db_conn):
    """Readings outside the plant_sensors assignment window must not appear in timeseries."""
    species_name = f"pytest-species-{uuid.uuid4().hex[:8]}"
    plant_a = f"pytest-plant-{uuid.uuid4().hex[:8]}"
    plant_b = f"pytest-plant-{uuid.uuid4().hex[:8]}"
    device_addr = f"CC:EE:{uuid.uuid4().hex[:8].upper()}"

    for name in [species_name]:
        request_json("/species", method="POST", payload={"name": name})
    for name in [plant_a, plant_b]:
        request_json(
            "/plants",
            method="POST",
            payload={"plant_name": name, "species_name": species_name},
        )

    t_pre = 1_000_000
    t_assign_a = 2_000_000
    t_during_a = 3_000_000
    t_reassign = 4_000_000
    t_during_b = 5_000_000

    device_id = database.register_device(
        db_conn, f"pytest-dev-{uuid.uuid4().hex[:6]}", device_addr
    )
    plant_id_a = database.fetch_plant_by_name(db_conn, plant_name=plant_a)["id"]
    plant_id_b = database.fetch_plant_by_name(db_conn, plant_name=plant_b)["id"]

    database.assign_plant_sensor(
        db_conn,
        plant_id=plant_id_a,
        device_id=device_id,
        sensor="cap1",
        assigned_at=t_assign_a,
    )
    database.assign_plant_sensor(
        db_conn,
        plant_id=plant_id_b,
        device_id=device_id,
        sensor="cap1",
        assigned_at=t_reassign,
    )

    database.write_sensor_readings(
        db_conn,
        device_id=device_id,
        readings=[
            ("cap1", 100, 100, 0, t_pre, t_pre),
            ("cap1", 200, 200, 0, t_during_a, t_during_a),
            ("cap1", 300, 300, 0, t_during_b, t_during_b),
        ],
    )

    ts_a = request_json(
        f"/timeseries?plant={plant_a}&sensor=cap1&device={device_addr}&limit=1000"
    )
    ts_b = request_json(
        f"/timeseries?plant={plant_b}&sensor=cap1&device={device_addr}&limit=1000"
    )

    times_a = {r["adjusted_time_ms"] for r in ts_a["data"]}
    times_b = {r["adjusted_time_ms"] for r in ts_b["data"]}

    assert t_during_a in times_a
    assert (
        t_pre not in times_a
    ), "reading before assignment must be excluded from plant A"
    assert (
        t_during_b not in times_a
    ), "reading after unassignment must be excluded from plant A"

    assert t_during_b in times_b
    assert (
        t_during_a not in times_b
    ), "reading before reassignment must be excluded from plant B"
    assert t_pre not in times_b

    for name in [plant_a, plant_b]:
        request_json(f"/plants/{name}", method="DELETE")
    request_json(f"/species/{species_name}", method="DELETE")


def test_devices_smoke_flow(db_conn):
    address = f"AA:BB:CC:DD:{uuid.uuid4().hex[:4].upper()}"
    name = f"pytest-device-{uuid.uuid4().hex[:8]}"
    database.register_device(db_conn, name, address)

    listed = request_json("/devices?limit=10000")
    addresses = [item["address"] for item in listed.get("data", [])]
    assert address in addresses
