import os
import uuid
from datetime import datetime, timezone
from urllib.error import HTTPError

import pytest

from hcultinf.simulation import simulate_plant_moisture
from test_utils import request_json, SENSOR_DRY_MV, SENSOR_WET_MV

SEED_DSN = os.environ.get(
    "OPENHCULT_SEED_DSN", "postgresql://hcult:hcult@127.0.0.1:5432/hcult"
)

_CALIB_CSV = os.path.join(os.path.dirname(__file__), "..", "calib", "calibration.csv")
_WINDOW_MS = 5 * 60 * 1000  # 5 min — inside the endpoint's 10-min window


def test_water_calibration_no_data():
    species_name = f"pytest-species-{uuid.uuid4().hex[:8]}"
    plant_name = f"pytest-plant-{uuid.uuid4().hex[:8]}"
    request_json("/species", method="POST", payload={"name": species_name})
    request_json(
        "/plants",
        method="POST",
        payload={"plant_name": plant_name, "species_name": species_name},
    )

    with pytest.raises(HTTPError) as exc:
        request_json(f"/water_calibration?plant={plant_name}")
    assert exc.value.code == 400

    request_json(f"/plants/{plant_name}", method="DELETE")
    request_json(f"/species/{species_name}", method="DELETE")


def test_water_calibration_with_waterings():
    if not os.path.exists(_CALIB_CSV):
        pytest.skip("calib/calibration.csv not found")

    try:
        import psycopg
    except ImportError:
        pytest.skip("psycopg not available")

    species_name = f"pytest-species-{uuid.uuid4().hex[:8]}"
    plant_name = f"pytest-plant-{uuid.uuid4().hex[:8]}"
    device_addr = f"CC:DD:{uuid.uuid4().hex[:8].upper()}"

    request_json("/species", method="POST", payload={"name": species_name})
    request_json(
        "/plants",
        method="POST",
        payload={"plant_name": plant_name, "species_name": species_name},
    )

    with psycopg.connect(SEED_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO devices (name, address) VALUES (%s, %s) RETURNING id",
                (f"pytest-dev-{uuid.uuid4().hex[:6]}", device_addr),
            )
            device_id = cur.fetchone()[0]
            cur.execute("SELECT id FROM plants WHERE plant_name = %s", (plant_name,))
            plant_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO plant_sensors (plant_id, device_id, sensor, assigned_at) VALUES (%s, %s, %s, 0)",
                (plant_id, device_id, "cap1"),
            )
        conn.commit()

    readings_bg, watering_times, ml_amounts, before_vals, after_vals = (
        simulate_plant_moisture(
            max_swc_ml=75.0,
            base=SENSOR_DRY_MV,
            wet=SENSOR_WET_MV,
            dose_frac_range=(0.4, 0.8),
            target_fc_range=(0.05, 0.2),
            drain_per_day=0.1,
            noise=0,
        )
    )

    window_readings = [
        (t - _WINDOW_MS, bv) for t, bv in zip(watering_times, before_vals)
    ] + [(t + _WINDOW_MS, av) for t, av in zip(watering_times, after_vals)]

    with psycopg.connect(SEED_DSN) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO sensor_readings "
                "(device_id, sensor, measurement, voltage_mv, measurement_time_us, collection_time_ms, adjusted_time_ms) "
                "VALUES (%s, %s, %s, %s, 0, %s, %s)",
                [
                    (device_id, "cap1", raw, int(raw * 0.95), t, t)
                    for t, raw, mv in readings_bg
                ]
                + [(device_id, "cap1", raw, raw, t, t) for t, raw in window_readings],
            )
        conn.commit()

    for t_water, ml in zip(watering_times, ml_amounts):
        observed_at = (
            datetime.fromtimestamp(t_water / 1000, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        request_json(
            "/observations",
            method="POST",
            payload={
                "note": f"WATER manual ml={round(ml)}",
                "observed_at": observed_at,
                "plant_name": plant_name,
            },
        )

    result = request_json(f"/water_calibration?plant={plant_name}")
    assert "prior_x" in result
    assert "mean" in result
    assert "std" in result
    assert len(result["prior_x"]) == len(result["mean"]) == len(result["std"])
    assert len(result["prior_x"]) > 0

    request_json(f"/plants/{plant_name}", method="DELETE")
    request_json(f"/species/{species_name}", method="DELETE")
