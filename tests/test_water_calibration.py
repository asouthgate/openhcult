import os
import uuid
from datetime import datetime, timezone
from urllib.error import HTTPError

import numpy as np
import pytest

from hcultinf.simulation import simulate_plant_moisture
from hcultdb import queries as database
from test_utils import request_json, SENSOR_DRY_MV, SENSOR_WET_MV

_CALIB_CSV = os.path.join(os.path.dirname(__file__), "..", "calib", "calibration.csv")
_WINDOW_MS = 5 * 60 * 1000


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


def test_water_calibration_with_waterings(db_conn):
    if not os.path.exists(_CALIB_CSV):
        pytest.skip("calib/calibration.csv not found")

    species_name = f"pytest-species-{uuid.uuid4().hex[:8]}"
    plant_name = f"pytest-plant-{uuid.uuid4().hex[:8]}"
    device_addr = f"CC:DD:{uuid.uuid4().hex[:8].upper()}"

    request_json("/species", method="POST", payload={"name": species_name})
    request_json(
        "/plants",
        method="POST",
        payload={"plant_name": plant_name, "species_name": species_name},
    )

    device_id = database.register_device(
        db_conn, f"pytest-dev-{uuid.uuid4().hex[:6]}", device_addr
    )
    plant_id = database.fetch_plant_by_name(db_conn, plant_name=plant_name)["id"]
    database.assign_plant_sensor(
        db_conn,
        plant_id=plant_id,
        device_id=device_id,
        sensor="cap1",
        assigned_at=0,
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

    database.write_sensor_readings(
        db_conn,
        device_id=device_id,
        readings=[("cap1", raw, int(raw * 0.95), 0, t, t) for t, raw, mv in readings_bg]
        + [("cap1", raw, raw, 0, t, t) for t, raw in window_readings],
    )

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

    result = request_json(
        f"/water_calibration?plant={plant_name}&prior_max={SENSOR_DRY_MV}"
    )
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


_FIDDLE_LEAF = "fiddle-leaf-bedroom"
_FIDDLE_DEVICE = "AA:11:22:33:44:02"


def test_combined_water_between_individual_sensors(seeded_db):
    if not os.path.exists(_CALIB_CSV):
        pytest.skip("calib/calibration.csv not found")

    import time

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 7 * 24 * 3600 * 1000

    result = request_json(
        f"/swc_timeseries?plant={_FIDDLE_LEAF}&prior=calibrated"
        f"&start_ms={start_ms}&end_ms={now_ms}"
    )

    assert "times_ms" in result
    assert "mean_swc" in result
    assert "ci_low" in result
    assert "ci_high" in result
    assert "scale" in result

    fused_swc = np.array([v if v is not None else np.nan for v in result["mean_swc"]])
    finite = fused_swc[np.isfinite(fused_swc)]
    assert len(finite) > 0, "Should have some valid SWC predictions"
    assert np.all(finite >= 0), "SWC should be non-negative"
