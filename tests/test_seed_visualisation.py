from hcultinf.simulation import simulate_plant_moisture
from hcultdb import queries as database
from test_utils import request_json, SENSOR_DRY_MV, SENSOR_WET_MV

_PLANTS = [
    {
        "species": "Monstera deliciosa",
        "plant": "monstera-living-room",
        "address": "AA:11:22:33:44:01",
        "sensor": "capacitive1",
        "base": SENSOR_DRY_MV,
        "wet": SENSOR_WET_MV,
        "max_swc_ml": 250.0,
        "dose_frac_range": (0.6, 0.9),
        "target_wc_range": (0.05, 0.15),
        "drain_per_day": 0.12,
    },
    {
        "species": "Ficus lyrata",
        "plant": "fiddle-leaf-bedroom",
        "address": "AA:11:22:33:44:02",
        "sensor": "capacitive1",
        "base": SENSOR_DRY_MV,
        "wet": SENSOR_WET_MV,
        "max_swc_ml": 120.0,
        "dose_frac_range": (0.2, 0.4),
        "target_wc_range": (0.2, 0.5),
        "drain_per_day": 0.05,
    },
]
_5MIN_MS = 5 * 60 * 1000


def test_seed_visualisation_data(db_conn):
    for p in _PLANTS:
        request_json("/species", method="POST", payload={"name": p["species"]})
        request_json(
            "/plants",
            method="POST",
            payload={"plant_name": p["plant"], "species_name": p["species"]},
        )

    for p in _PLANTS:
        device_id = database.register_device(db_conn, p["plant"], p["address"])
        plant_id = database.fetch_plant_by_name(db_conn, plant_name=p["plant"])["id"]

        database.assign_plant_sensor(
            db_conn,
            plant_id=plant_id,
            device_id=device_id,
            sensor=p["sensor"],
            assigned_at=0,
        )

        base, wet = p["base"], p["wet"]

        readings, watering_times, ml_amounts, before_vals, after_vals = (
            simulate_plant_moisture(
                p["max_swc_ml"],
                base,
                wet,
                dose_frac_range=p["dose_frac_range"],
                target_fc_range=p["target_wc_range"],
                drain_per_day=p["drain_per_day"],
            )
        )

        window_readings = [
            (t - _5MIN_MS, bv) for t, bv in zip(watering_times, before_vals)
        ] + [(t + _5MIN_MS, av) for t, av in zip(watering_times, after_vals)]

        database.write_sensor_readings(
            db_conn,
            device_id=device_id,
            readings=[(p["sensor"], v, mv, t * 1000, t, t) for t, v, mv in readings]
            + [(p["sensor"], v, v, t, t, t) for t, v in window_readings],
        )

        for t, ml in zip(watering_times, ml_amounts):
            database.insert_observation(
                db_conn,
                note=f"WATER manual ml={round(ml)}",
                observed_at_ms=int(t),
                plant_name=p["plant"],
            )

    series = request_json(f"/timeseries?limit=100&plant={_PLANTS[0]['plant']}")
    assert series["count"] > 0
