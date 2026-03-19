import os
import time
import psycopg
from test_utils import request_json, simulate_moisture

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
        "base": 2200,
    },
    {
        "species": "Ficus lyrata",
        "plant": "fiddle-leaf-bedroom",
        "address": "AA:11:22:33:44:02",
        "sensor": "capacitive1",
        "base": 1800,
    },
]

_HOURS = 48


def test_seed_visualisation_data():
    n = _HOURS * 60

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

                readings = simulate_moisture(
                    n, _HOURS, base=p["base"], noise_seed=hash(p["plant"]) % 1000
                )
                cur.executemany(
                    "INSERT INTO sensor_readings "
                    "(device_id, sensor, measurement, measurement_time_us, collection_time_ms, adjusted_time_ms) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [(device_id, p["sensor"], v, t * 1000, t, t) for t, v in readings],
                )

                watering_t = readings[int(n * 0.55)][0]
                cur.execute(
                    "INSERT INTO observations (observed_at, note, plant_id) VALUES (%s, %s, %s)",
                    (watering_t, "WATER manual seed", plant_id),
                )

        conn.commit()

    series = request_json(f"/timeseries?limit=100&plant={_PLANTS[0]['plant']}")
    assert series["count"] > 0
