import pytest

from hcultdb import queries, setup


def _setup_sqlite(tmp_path):
    db_path = tmp_path / "hcultdb_queries.sqlite"
    conn = setup.setup_db(str(db_path))
    return conn, db_path


def test_connect_register_device_and_timeseries(tmp_path):
    conn, db_path = _setup_sqlite(tmp_path)
    conn.close()

    conn = queries.connect(str(db_path))
    device_id = queries.register_device(conn, "dev-1", "AA:BB:CC:DD:EE:FF")
    queries.write_sensor_readings(
        conn,
        device_id,
        {
            "sensor1": 101,
            "sensor2": 202,
        },
    )
    rows = queries.fetch_timeseries(conn, limit=10)
    assert len(rows) == 2
    sensors = {row["sensor"] for row in rows}
    assert sensors == {"sensor1", "sensor2"}

    devices = queries.fetch_devices(conn, limit=10)
    assert devices[0]["address"] == "AA:BB:CC:DD:EE:FF"
    by_name = queries.fetch_device_by_name_or_address(conn, device="dev-1")
    by_addr = queries.fetch_device_by_name_or_address(conn, device="AA:BB:CC:DD:EE:FF")
    assert by_name["id"] == device_id
    assert by_addr["id"] == device_id

    queries.update_device_name(conn, address="AA:BB:CC:DD:EE:FF", name="dev-1-renamed")
    renamed = queries.fetch_device_by_name_or_address(conn, device="dev-1-renamed")
    assert renamed["id"] == device_id
    with pytest.raises(ValueError):
        queries.update_device_name(conn, address="missing", name="nope")
    conn.close()


def test_species_and_plants(tmp_path):
    conn, _ = _setup_sqlite(tmp_path)
    species_id = queries.insert_species(
        conn,
        name="pothos",
        common_name="Golden Pothos",
        metadata='{"light": "low"}',
    )
    assert queries.fetch_species_id(conn, name="pothos") == species_id
    species_rows = queries.fetch_species(conn, limit=10)
    assert species_rows[0]["name"] == "pothos"

    queries.update_species(
        conn,
        name="pothos",
        common_name="Pothos",
        metadata='{"light": "medium"}',
    )

    plant_id = queries.insert_plant(
        conn,
        plant_name="kitchen-herb",
        species_id=species_id,
        tag="window",
        metadata='{"pot": "clay"}',
    )
    plants = queries.fetch_plants(conn, limit=10)
    assert plants[0]["plant_name"] == "kitchen-herb"
    plant = queries.fetch_plant_by_name(conn, plant_name="kitchen-herb")
    assert plant["id"] == plant_id

    queries.update_plant(
        conn,
        plant_id=plant_id,
        species_id=species_id,
        tag="shelf",
        metadata='{"pot": "plastic"}',
    )
    updated = queries.fetch_plant_by_name(conn, plant_name="kitchen-herb")
    assert updated["tag"] == "shelf"

    plant_id2 = queries.insert_plant(
        conn,
        plant_name="temp-plant",
        species_id=None,
        tag=None,
        metadata=None,
    )
    queries.delete_plant(conn, plant_id=plant_id2)
    with pytest.raises(ValueError):
        queries.delete_plant(conn, plant_id=plant_id2)

    plant_id3 = queries.insert_plant(
        conn,
        plant_name="temp-plant-2",
        species_id=None,
        tag=None,
        metadata=None,
    )
    queries.delete_plant_by_name(conn, plant_name="temp-plant-2")
    with pytest.raises(ValueError):
        queries.delete_plant_by_name(conn, plant_name="temp-plant-2")

    extra_species_id = queries.insert_species(conn, name="fern", common_name=None, metadata=None)
    queries.delete_species(conn, species_id=extra_species_id)
    with pytest.raises(ValueError):
        queries.delete_species(conn, species_id=extra_species_id)

    extra_species_id2 = queries.insert_species(conn, name="ivy", common_name=None, metadata=None)
    queries.delete_species_by_name(conn, name="ivy")
    with pytest.raises(ValueError):
        queries.delete_species_by_name(conn, name="ivy")
    conn.close()


def test_observations_and_statuses(tmp_path):
    conn, _ = _setup_sqlite(tmp_path)
    plant_id = queries.insert_plant(
        conn,
        plant_name="purple-queen",
        species_id=None,
        tag=None,
        metadata=None,
    )
    obs_id = queries.add_observation(conn, note="Note 1", observed_at="2026-01-01T00:00:00Z")
    obs_id2 = queries.insert_observation(
        conn,
        note="Note 2",
        observed_at_ms=123456,
        plant_id=plant_id,
    )
    assert isinstance(obs_id, int)
    assert isinstance(obs_id2, int)

    observations = queries.fetch_observations(conn, limit=10)
    assert len(observations) >= 2
    obs_for_plant = queries.fetch_observations_for_plant(conn, plant_id=plant_id, limit=10)
    assert len(obs_for_plant) == 1

    queries.update_observation(
        conn,
        obs_id=obs_id2,
        observed_at_ms=123457,
        note="Updated",
        plant_id=plant_id,
    )
    updated = queries.fetch_observations_for_plant(conn, plant_id=plant_id, limit=10)[0]
    assert updated["note"] == "Updated"
    with pytest.raises(ValueError):
        queries.update_observation(
            conn,
            obs_id=9999,
            observed_at_ms=1,
            note="Missing",
            plant_id=plant_id,
        )

    status_type = queries.fetch_status_type_by_code(conn, code="CRISPY_LEAVES")
    assert status_type is not None
    status_id = queries.insert_plant_status(
        conn,
        plant_id=plant_id,
        status_type_id=status_type["id"],
        observed_at=123456,
        note="Crispy edges",
    )
    assert isinstance(status_id, int)
    statuses = queries.fetch_plant_statuses(conn, plant_id=plant_id, limit=10)
    assert statuses[0]["status_code"] == "CRISPY_LEAVES"
    conn.close()


def test_assign_plant_sensor(tmp_path):
    conn, _ = _setup_sqlite(tmp_path)
    plant_id = queries.insert_plant(
        conn,
        plant_name="sensor-plant",
        species_id=None,
        tag=None,
        metadata=None,
    )
    device_id = queries.register_device(conn, "dev-2", "11:22:33:44:55:66")
    queries.assign_plant_sensor(conn, plant_id=plant_id, device_id=device_id, sensor="sensor1")
    queries.write_sensor_readings(conn, device_id, {"sensor1": 42})
    rows = queries.fetch_timeseries(conn, plant="sensor-plant", limit=10)
    assert len(rows) == 1
    conn.close()
