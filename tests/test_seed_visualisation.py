import os
import random
import psycopg
from hcultinf.simulation import simulate_plant_moisture
from test_utils import request_json, SENSOR_DRY_MV, SENSOR_WET_MV

SEED_DSN = os.environ.get(
    "OPENHCULT_SEED_DSN",
    "postgresql://hcult:hcult@127.0.0.1:5432/hcult",
)

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


def test_seed_visualisation_data():
    for p in _PLANTS:
        request_json("/species", method="POST", payload={"name": p["species"]})
        request_json(
            "/plants",
            method="POST",
            payload={"plant_name": p["plant"], "species_name": p["species"]},
        )

    with psycopg.connect(SEED_DSN) as conn:
        with conn.cursor() as cur:
            for p in _PLANTS:
                cur.execute(
                    "INSERT INTO devices (name, address) VALUES (%s, %s) "
                    "ON CONFLICT (address) DO UPDATE SET name = EXCLUDED.name RETURNING id",
                    (p["plant"], p["address"]),
                )
                device_id = cur.fetchone()[0]

                cur.execute(
                    "SELECT id FROM plants WHERE plant_name = %s", (p["plant"],)
                )
                plant_id = cur.fetchone()[0]

                cur.execute(
                    "INSERT INTO plant_sensors (plant_id, device_id, sensor) VALUES (%s, %s, %s) "
                    "ON CONFLICT (plant_id, device_id, sensor) DO NOTHING",
                    (plant_id, device_id, p["sensor"]),
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

                cur.executemany(
                    "INSERT INTO sensor_readings "
                    "(device_id, sensor, measurement, voltage_mv, measurement_time_us, collection_time_ms, adjusted_time_ms) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    [
                        (device_id, p["sensor"], v, mv, t * 1000, t, t)
                        for t, v, mv in readings
                    ]
                    + [
                        (device_id, p["sensor"], v, v, t, t, t)
                        for t, v in window_readings
                    ],
                )
                cur.executemany(
                    "INSERT INTO observations (observed_at, note, plant_id) VALUES (%s, %s, %s)",
                    [
                        (t, f"WATER manual ml={round(ml)}", plant_id)
                        for t, ml in zip(watering_times, ml_amounts)
                    ],
                )

        conn.commit()

    series = request_json(f"/timeseries?limit=100&plant={_PLANTS[0]['plant']}")
    assert series["count"] > 0
