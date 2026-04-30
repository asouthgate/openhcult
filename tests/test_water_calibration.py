import os
import uuid
from datetime import datetime, timezone
from urllib.error import HTTPError

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

    cap1 = request_json(
        f"/swc_timeseries?plant={_FIDDLE_LEAF}&sensor=capacitive1"
        f"&device_address={_FIDDLE_DEVICE}&prior=calibrated"
        f"&start_ms={start_ms}&end_ms={now_ms}"
    )
    cap2 = request_json(
        f"/swc_timeseries?plant={_FIDDLE_LEAF}&sensor=capacitive2"
        f"&device_address={_FIDDLE_DEVICE}&prior=calibrated"
        f"&start_ms={start_ms}&end_ms={now_ms}"
    )
    combined = request_json(
        f"/combined_swc_timeseries?plant={_FIDDLE_LEAF}&prior=calibrated"
        f"&start_ms={start_ms}&end_ms={now_ms}"
    )

    cap1_vals = [m for m in cap1["mean_swc"] if m is not None]
    cap2_vals = [m for m in cap2["mean_swc"] if m is not None]
    combined_vals = [m for m in combined["mean_swc"] if m is not None]

    for label, vals in [
        ("cap1", cap1_vals),
        ("cap2", cap2_vals),
        ("combined", combined_vals),
    ]:
        arr = __import__("numpy").array(vals)
        print(
            f"{label}: n={len(arr)} mean={arr.mean():.2f} std={arr.std():.2f}"
            f" min={arr.min():.2f} max={arr.max():.2f}"
        )

    cap1_by_time = {
        t: m for t, m in zip(cap1["times_ms"], cap1["mean_swc"]) if m is not None
    }
    cap2_by_time = {
        t: m for t, m in zip(cap2["times_ms"], cap2["mean_swc"]) if m is not None
    }

    between_count = 0
    for t, c_swc in zip(combined["times_ms"], combined["mean_swc"]):
        if c_swc is None:
            continue
        s1 = cap1_by_time.get(t)
        s2 = cap2_by_time.get(t)
        if s1 is None or s2 is None:
            continue
        lo, hi = min(s1, s2), max(s1, s2)
        if lo <= c_swc <= hi:
            between_count += 1

    assert (
        between_count > 0
    ), "combined SWC should fall between individual sensor SWC at overlapping times"
