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


def test_water_calibration_sensor_without_device_rejected():
    with pytest.raises(HTTPError) as exc:
        request_json("/water_calibration?plant=anyplant&sensor=cap1")
    assert exc.value.code == 400


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
    assert "ci_low" in result
    assert "ci_high" in result
    assert (
        len(result["prior_x"])
        == len(result["mean"])
        == len(result["ci_low"])
        == len(result["ci_high"])
    )
    assert len(result["prior_x"]) > 0

    request_json(f"/plants/{plant_name}", method="DELETE")
    request_json(f"/species/{species_name}", method="DELETE")


def test_water_calibration_device_address_filters_correctly():
    """Two devices with the same sensor name on the same plant must return different calibration data."""
    if not os.path.exists(_CALIB_CSV):
        pytest.skip("calib/calibration.csv not found")

    try:
        import psycopg
    except ImportError:
        pytest.skip("psycopg not available")

    species_name = f"pytest-species-{uuid.uuid4().hex[:8]}"
    plant_name = f"pytest-plant-{uuid.uuid4().hex[:8]}"
    addr_a = f"AA:11:{uuid.uuid4().hex[:8].upper()}"
    addr_b = f"BB:22:{uuid.uuid4().hex[:8].upper()}"

    request_json("/species", method="POST", payload={"name": species_name})
    request_json(
        "/plants",
        method="POST",
        payload={"plant_name": plant_name, "species_name": species_name},
    )

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
            cur.execute("SELECT id FROM plants WHERE plant_name = %s", (plant_name,))
            plant_id = cur.fetchone()[0]

            cur.execute(
                "INSERT INTO devices (name, address) VALUES (%s, %s) RETURNING id",
                (f"pytest-dev-a-{uuid.uuid4().hex[:6]}", addr_a),
            )
            device_id_a = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO devices (name, address) VALUES (%s, %s) RETURNING id",
                (f"pytest-dev-b-{uuid.uuid4().hex[:6]}", addr_b),
            )
            device_id_b = cur.fetchone()[0]

            cur.execute(
                "INSERT INTO plant_sensors (plant_id, device_id, sensor, assigned_at) VALUES (%s, %s, %s, 0)",
                (plant_id, device_id_a, "cap1"),
            )
            cur.execute(
                "INSERT INTO plant_sensors (plant_id, device_id, sensor, assigned_at) VALUES (%s, %s, %s, 0)",
                (plant_id, device_id_b, "cap1"),
            )

            cur.executemany(
                "INSERT INTO sensor_readings (device_id, sensor, measurement, voltage_mv, measurement_time_us, collection_time_ms, adjusted_time_ms) VALUES (%s, %s, %s, %s, 0, %s, %s)",
                [
                    (device_id_a, "cap1", raw, int(raw * 0.95), t, t)
                    for t, raw, mv in readings_bg
                ]
                + [(device_id_a, "cap1", raw, raw, t, t) for t, raw in window_readings],
            )
            # Place device B readings inside the calibration windows used in the request below.
            # With offset_ms=0 and width_ms=_WINDOW_MS the windows are [t-_WINDOW_MS, t] and [t, t+_WINDOW_MS].
            first_t = watering_times[0]
            cur.executemany(
                "INSERT INTO sensor_readings (device_id, sensor, measurement, voltage_mv, measurement_time_us, collection_time_ms, adjusted_time_ms) VALUES (%s, %s, %s, %s, 0, %s, %s)",
                [
                    (
                        device_id_b,
                        "cap1",
                        before_vals[0],
                        before_vals[0],
                        first_t - _WINDOW_MS // 2,
                        first_t - _WINDOW_MS // 2,
                    ),
                    (
                        device_id_b,
                        "cap1",
                        after_vals[0],
                        after_vals[0],
                        first_t + _WINDOW_MS // 2,
                        first_t + _WINDOW_MS // 2,
                    ),
                ],
            )
        conn.commit()

    _calib_params = f"offset_ms=0&width_ms={_WINDOW_MS}"
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

    result_a = request_json(
        f"/water_calibration?plant={plant_name}&sensor=cap1&device_address={addr_a}&{_calib_params}"
    )
    result_b = request_json(
        f"/water_calibration?plant={plant_name}&sensor=cap1&device_address={addr_b}&{_calib_params}"
    )

    assert len(result_a["chord_times"]) > len(
        result_b["chord_times"]
    ), "device A has readings for all waterings so should yield more chords than device B"
    assert len(result_b["chord_times"]) == 1

    request_json(f"/plants/{plant_name}", method="DELETE")
    request_json(f"/species/{species_name}", method="DELETE")
