import os
import uuid
from datetime import datetime, timezone
from urllib.error import HTTPError

import pytest

from test_utils import request_json, simulate_moisture_multi

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
        conn.commit()

    request_json(
        f"/plants/{plant_name}/assign",
        method="POST",
        payload={"device": device_addr, "sensor": "cap1"},
    )

    # Sensor A from calibration: dry ~2310, wet ~895
    readings_bg, watering_times = simulate_moisture_multi(
        n_events=10, days_per_cycle=2, base=2310, wet=895
    )

    # Explicit before/after readings within the ±10-min inference window
    window_readings = []
    for t_water in watering_times:
        window_readings.append((t_water - _WINDOW_MS, 2100))  # drying, before watering
        window_readings.append((t_water + _WINDOW_MS, 920))  # freshly watered

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
                + [
                    (device_id, "cap1", raw, int(raw * 0.95), t, t)
                    for t, raw in window_readings
                ],
            )
        conn.commit()

    ml_per_watering = 200
    for t_water in watering_times:
        observed_at = (
            datetime.fromtimestamp(t_water / 1000, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        request_json(
            "/observations",
            method="POST",
            payload={
                "note": f"WATER manual ml={ml_per_watering}",
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
