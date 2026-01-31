import sqlite3

import pytest

from hcultdb import queries, setup


def _setup_sqlite(tmp_path):
    db_path = tmp_path / "hcultdb_test.sqlite"
    conn = setup.setup_db(str(db_path))
    return conn


def test_setup_loads_status_and_observation_types(tmp_path):
    conn = _setup_sqlite(tmp_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM status_types WHERE code = ?", ("CRISPY_LEAVES",)
    )
    assert cursor.fetchone() is not None
    cursor.execute(
        "SELECT 1 FROM observation_types WHERE code = ?", ("WATERING",)
    )
    assert cursor.fetchone() is not None
    conn.close()


def test_observations_unique_index(tmp_path):
    conn = _setup_sqlite(tmp_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO observations (observed_at, note) VALUES (?, ?)",
        (123, "AUTO: test"),
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute(
            "INSERT INTO observations (observed_at, note) VALUES (?, ?)",
            (123, "AUTO: test"),
        )
    conn.close()


def test_register_device_and_write_readings(tmp_path):
    conn = _setup_sqlite(tmp_path)
    device_id = queries.register_device(conn, "dev-1", "AA:BB:CC:DD:EE:FF")
    assert isinstance(device_id, int)
    queries.write_sensor_readings(
        conn,
        device_id,
        {
            "sensor1": 100,
            "sensor2": 200,
        },
    )
    rows = queries.fetch_timeseries(conn, limit=10)
    assert len(rows) == 2
    sensors = {row["sensor"] for row in rows}
    assert sensors == {"sensor1", "sensor2"}
    conn.close()
