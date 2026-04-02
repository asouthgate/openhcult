import os
import random
import time
import psycopg
from test_utils import request_json, simulate_moisture_multi, sensor_from_wc

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
        "wet": 900,
    },
    {
        "species": "Ficus lyrata",
        "plant": "fiddle-leaf-bedroom",
        "address": "AA:11:22:33:44:02",
        "sensor": "capacitive1",
        "base": 1800,
        "wet": 900,
        "varied": True,
    },
]

_N_EVENTS = 100
_DAYS_PER_CYCLE = 3
_5MIN_MS = 5 * 60 * 1000
_FC_ML = 75.0
_DRAIN_PER_DAY = 0.05  # fraction of FC lost per day


def _varied_events(base, wet, rng, n=22):
    """Irregular watering events using a linear water-content model.

    Water level hovers ~50-90% FC. Drying is linear between events.
    Returns (readings, watering_times, ml_amounts, before_sensor, after_sensor).
    """
    intervals_days = [rng.uniform(2, 7) for _ in range(n)]
    now_ms = int(time.time() * 1000)
    total_ms = int(sum(intervals_days) * 24 * 3600 * 1000)
    start_ms = now_ms - total_ms

    watering_times, before_wcs, after_wcs, ml_amounts = [], [], [], []
    wc = 0.70
    t = start_ms
    for days in intervals_days:
        t += int(days * 24 * 3600 * 1000)
        wc = max(0.20, wc - _DRAIN_PER_DAY * days)
        before_wcs.append(wc)
        watering_times.append(t)
        dose = rng.uniform(0.20, 0.40)
        wc = min(0.95, wc + dose)
        after_wcs.append(wc)
        ml_amounts.append(round(dose * _FC_ML))

    # Background readings: linear wc between events, sensor via response fn
    def wc_at(ts):
        for i in range(len(watering_times) - 1):
            if watering_times[i] <= ts < watering_times[i + 1]:
                frac = (ts - watering_times[i]) / (
                    watering_times[i + 1] - watering_times[i]
                )
                return after_wcs[i] + frac * (before_wcs[i + 1] - after_wcs[i])
        return after_wcs[-1] if ts >= watering_times[-1] else before_wcs[0]

    step_ms = 30 * 60 * 1000
    readings = []
    for i in range(total_ms // step_ms):
        ts = start_ms + i * step_ms
        raw = max(
            600, min(3000, sensor_from_wc(wc_at(ts), base, wet) + rng.randint(-25, 25))
        )
        readings.append((ts, raw, int(raw * 3900 / 4095)))

    before_sensor = [sensor_from_wc(wc, base, wet) for wc in before_wcs]
    after_sensor = [sensor_from_wc(wc, base, wet) for wc in after_wcs]
    return readings, watering_times, ml_amounts, before_sensor, after_sensor


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

                rng = random.Random(hash(p["plant"]) % 1000)
                wet = p["wet"]

                if p.get("varied"):
                    readings, watering_times, ml_amounts, before_vals, after_vals = (
                        _varied_events(p["base"], wet, rng)
                    )
                    window_readings = [
                        (t - _5MIN_MS, bv) for t, bv in zip(watering_times, before_vals)
                    ] + [
                        (t + _5MIN_MS, av) for t, av in zip(watering_times, after_vals)
                    ]
                else:
                    readings, watering_times = simulate_moisture_multi(
                        _N_EVENTS,
                        days_per_cycle=_DAYS_PER_CYCLE,
                        base=p["base"],
                        wet=wet,
                        noise_seed=hash(p["plant"]) % 1000,
                    )
                    ml_amounts = [200] * len(watering_times)
                    window_readings = [
                        (t - _5MIN_MS, int(p["base"] * 0.75)) for t in watering_times
                    ] + [(t + _5MIN_MS, wet + 50) for t in watering_times]

                cur.executemany(
                    "INSERT INTO sensor_readings "
                    "(device_id, sensor, measurement, voltage_mv, measurement_time_us, collection_time_ms, adjusted_time_ms) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    [
                        (device_id, p["sensor"], v, mv, t * 1000, t, t)
                        for t, v, mv in readings
                    ]
                    + [
                        (device_id, p["sensor"], v, int(v * 0.95), t, t, t)
                        for t, v in window_readings
                    ],
                )
                cur.executemany(
                    "INSERT INTO observations (observed_at, note, plant_id) VALUES (%s, %s, %s)",
                    [
                        (t, f"WATER manual ml={ml}", plant_id)
                        for t, ml in zip(watering_times, ml_amounts)
                    ],
                )

        conn.commit()

    series = request_json(f"/timeseries?limit=100&plant={_PLANTS[0]['plant']}")
    assert series["count"] > 0
