import os
import random
from types import SimpleNamespace

import pytest

from hcultinf.simulation import simulate_plant_moisture, linear_response
from hcultdb import setup
from hcultdb import queries as database
from test_utils import request_json, SENSOR_DRY_MV, SENSOR_WET_MV

SEED_DSN = os.environ.get(
    "OPENHCULT_SEED_DSN",
    "postgresql://hcult:hcult@127.0.0.1:5432/hcult",
)

_5MIN_MS = 5 * 60 * 1000

_SENSOR2_DRY_MV = 2200
_SENSOR2_WET_MV = 800

PLANTS = [
    {
        "species": "Monstera deliciosa",
        "plant": "monstera-living-room",
        "max_swc_ml": 250.0,
        "dose_frac_range": (0.6, 0.9),
        "target_wc_range": (0.05, 0.15),
        "drain_per_day": 0.12,
    },
    {
        "species": "Ficus lyrata",
        "plant": "fiddle-leaf-bedroom",
        "max_swc_ml": 120.0,
        "dose_frac_range": (0.2, 0.4),
        "target_wc_range": (0.2, 0.5),
        "drain_per_day": 0.05,
    },
]

SENSORS = [
    {
        "plant": "monstera-living-room",
        "address": "AA:11:22:33:44:01",
        "sensor": "capacitive1",
        "base": SENSOR_DRY_MV,
        "wet": SENSOR_WET_MV,
        "noise": 25,
    },
    {
        "plant": "fiddle-leaf-bedroom",
        "address": "AA:11:22:33:44:02",
        "sensor": "capacitive1",
        "base": SENSOR_DRY_MV,
        "wet": SENSOR_WET_MV,
        "noise": 25,
    },
    {
        "plant": "fiddle-leaf-bedroom",
        "address": "AA:11:22:33:44:02",
        "sensor": "capacitive2",
        "base": _SENSOR2_DRY_MV,
        "wet": _SENSOR2_WET_MV,
        "noise": 20,
    },
]


def _simulate_plant_fc(p, n_days=30):
    result = simulate_plant_moisture(
        p["max_swc_ml"],
        base=1,
        wet=0,
        dose_frac_range=p["dose_frac_range"],
        target_fc_range=p["target_wc_range"],
        drain_per_day=p["drain_per_day"],
        noise=0,
        response_fn=lambda fc: fc,
        rng=random.Random(42),
        n_days=n_days,
    )
    readings, watering_times, ml_amounts, before_fcs, after_fcs = result
    fc_times = [(t, fc_val) for t, fc_val, _ in readings]
    return fc_times, watering_times, ml_amounts, before_fcs, after_fcs


def _sensor_readings_from_fc(fc_times, sensor_cfg, rng):
    base, wet, noise = sensor_cfg["base"], sensor_cfg["wet"], sensor_cfg["noise"]
    readings = []
    for t, fc in fc_times:
        mv = linear_response(fc, base, wet) + rng.randint(-noise, noise)
        mv = max(wet, min(base, mv))
        readings.append((t, mv, mv))
    return readings


@pytest.fixture(scope="session")
def db_conn():
    conn = setup.setup_db(SEED_DSN)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def seeded_db(db_conn):
    for p in PLANTS:
        request_json("/species", method="POST", payload={"name": p["species"]})
        request_json(
            "/plants",
            method="POST",
            payload={"plant_name": p["plant"], "species_name": p["species"]},
        )

    for s in SENSORS:
        device_id = database.register_device(db_conn, s["plant"], s["address"])
        plant = database.fetch_plant_by_name(db_conn, plant_name=s["plant"])
        database.assign_plant_sensor(
            db_conn,
            plant_id=plant["id"],
            device_id=device_id,
            sensor=s["sensor"],
            assigned_at=0,
        )

    plant_fc_data = {}
    for p in PLANTS:
        fc_times, watering_times, ml_amounts, before_fcs, after_fcs = (
            _simulate_plant_fc(p)
        )
        plant_fc_data[p["plant"]] = (
            fc_times,
            watering_times,
            ml_amounts,
            before_fcs,
            after_fcs,
        )

    device_cache = {}
    for s in SENSORS:
        addr = s["address"]
        if addr not in device_cache:
            dev = database.fetch_device_by_name_or_address(db_conn, device=addr)
            device_cache[addr] = dev["id"]
        device_id = device_cache[addr]

        fc_times, watering_times, ml_amounts, before_fcs, after_fcs = plant_fc_data[
            s["plant"]
        ]
        rng = random.Random(42)
        readings = _sensor_readings_from_fc(fc_times, s, rng)

        before_vals = [linear_response(fc, s["base"], s["wet"]) for fc in before_fcs]
        after_vals = [linear_response(fc, s["base"], s["wet"]) for fc in after_fcs]
        window_readings = [
            (t - _5MIN_MS, bv) for t, bv in zip(watering_times, before_vals)
        ] + [(t + _5MIN_MS, av) for t, av in zip(watering_times, after_vals)]

        database.write_sensor_readings(
            db_conn,
            device_id=device_id,
            readings=[(s["sensor"], v, mv, t * 1000, t, t) for t, v, mv in readings]
            + [(s["sensor"], v, v, t, t, t) for t, v in window_readings],
        )

    inserted_plants = set()
    for s in SENSORS:
        if s["plant"] in inserted_plants:
            continue
        inserted_plants.add(s["plant"])
        _, watering_times, ml_amounts, _, _ = plant_fc_data[s["plant"]]
        for t, ml in zip(watering_times, ml_amounts):
            database.insert_observation(
                db_conn,
                note=f"WATER manual ml={round(ml)}",
                observed_at_ms=int(t),
                plant_name=s["plant"],
            )

    return SimpleNamespace(conn=db_conn, plants=PLANTS, sensors=SENSORS)
