from hcultdb import queries as database
from test_utils import request_json


def test_seed_visualisation_data(seeded_db):
    plants_rows = list(database.fetch_plants(seeded_db.conn, include_sensors=False))
    plant_names_in_db = {row["plant_name"] for row in plants_rows}
    for p in seeded_db.plants:
        assert p["plant"] in plant_names_in_db

    mappings = list(database.fetch_plant_sensors(seeded_db.conn))
    expected_keys = {(s["address"], s["sensor"]) for s in seeded_db.sensors}
    actual_keys = {(m["device_address"], m["sensor"]) for m in mappings}
    assert expected_keys.issubset(actual_keys)

    for p in seeded_db.plants:
        series = request_json(f"/timeseries?limit=100&plant={p['plant']}")
        assert series["count"] > 0

    for s in seeded_db.sensors:
        series = request_json(
            f"/timeseries?limit=10&sensor={s['sensor']}&device={s['address']}"
        )
        assert series["count"] > 0
