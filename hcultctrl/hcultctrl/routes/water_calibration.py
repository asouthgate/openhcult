from __future__ import annotations

import csv
import logging
import math
import os
import re

import numpy as np
from fastapi import Depends, HTTPException, APIRouter
from pydantic import BaseModel

from hcultctrl.utils import get_db_conn
from hcultdb import queries as database
from hcultinf.exp import ExponentialCordCalibrator
from hcultinf.exp_mcmc import ExponentialCordCalibratorMCMC
from hcultinf.combiner import fuse_swc
from hcultinf.drying import linear_drying_rate

logger = logging.getLogger(__name__)
router = APIRouter()

_ML_RE = re.compile(r"\bml=(\d+(?:\.\d+)?)\b")
_DEFAULT_OFFSET_MS = 10 * 60 * 1000
_DEFAULT_WIDTH_MS = 50 * 60 * 1000


class WindowParams(BaseModel):
    plant: str
    offset_ms: int = _DEFAULT_OFFSET_MS
    width_ms: int = _DEFAULT_WIDTH_MS
    prior: str = "calibrated"
    prior_min: float | None = None
    prior_max: float | None = None


class TuningParams(BaseModel):
    prior_weight: float = 1.0
    n_burn: int = 10
    n_steps: int = 30
    xmin_low: float = 800.0
    xmin_high: float = 1100.0


class CordDataParams(WindowParams):
    sensor: str | None = None
    device_address: str | None = None


class CalibrationParams(CordDataParams, TuningParams):
    estimator: str = "exp_mcmc"


def _volume_ml(note: str) -> float | None:
    m = _ML_RE.search(note or "")
    return float(m.group(1)) if m else None


def _load_calibration(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    sensor_vals, swc_vals = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if "sensor_val" in fieldnames:
            for row in reader:
                sensor_vals.append(float(row["sensor_val"]))
                swc_vals.append(float(row["swc"]))
        else:
            by_ml: dict[float, list[float]] = {}
            for row in reader:
                ml = float(row["ml"])
                if ml < 0:
                    continue
                if "voltage_mV" in fieldnames:
                    by_ml.setdefault(ml, []).append(float(row["voltage_mV"]))
                else:
                    for col in ("sensor1_voltage", "sensor2_voltage"):
                        if col in fieldnames:
                            by_ml.setdefault(ml, []).append(float(row[col]))
            for ml, readings in by_ml.items():
                sensor_vals.append(float(np.mean(readings)))
                swc_vals.append(ml)
    order = np.argsort(sensor_vals)
    return np.array(sensor_vals)[order], np.array(swc_vals)[order]


def _fetch_cord_data(conn, p: CordDataParams):
    if p.sensor and not p.device_address:
        raise HTTPException(
            status_code=400,
            detail="device_address is required when sensor is specified",
        )
    if p.prior not in ("calibrated", "linear"):
        raise HTTPException(
            status_code=400, detail="prior must be 'calibrated' or 'linear'"
        )

    obs = list(
        database.fetch_observations_for_plant(conn, plant_name=p.plant, limit=1000)
    )
    waterings = [
        (o["observed_at"], _volume_ml(o["note"]))
        for o in obs
        if o.get("note")
        and "WATER" in o["note"]
        and "AUTO" not in o["note"]
        and _volume_ml(o["note"]) is not None
    ]

    if not waterings:
        raise HTTPException(
            status_code=400,
            detail="No watering observations with ml data for this plant",
        )

    chords = []
    chord_times = []
    for t_ms, ml in waterings:
        before = list(
            database.fetch_timeseries(
                conn,
                plant=p.plant,
                sensor=p.sensor,
                device=p.device_address,
                start_ms=t_ms - p.offset_ms - p.width_ms,
                end_ms=t_ms - p.offset_ms,
                limit=5000,
            )
        )
        after = list(
            database.fetch_timeseries(
                conn,
                plant=p.plant,
                sensor=p.sensor,
                device=p.device_address,
                start_ms=t_ms + p.offset_ms,
                end_ms=t_ms + p.offset_ms + p.width_ms,
                limit=5000,
            )
        )
        if not before or not after:
            continue
        x = float(np.median([r["voltage_mv"] for r in before]))
        x_after = float(np.median([r["voltage_mv"] for r in after]))
        chords.append((x, x_after - x, ml))
        chord_times.append(t_ms)

    if not chords:
        raise HTTPException(
            status_code=400,
            detail=f"No sensor data found in windows around waterings (offset={p.offset_ms//60000}min, width={p.width_ms//60000}min)",
        )

    x_arr = np.array([c[0] for c in chords])
    dx_arr = np.array([c[1] for c in chords])
    dy_arr = np.array([c[2] for c in chords])

    if p.prior == "linear":
        prior_x = np.linspace(p.prior_min, p.prior_max, 500)
        t = (prior_x - p.prior_min) / (p.prior_max - p.prior_min)
        prior_y = 1.0 - t
        x_anchor = np.array([p.prior_max])
    else:
        csv_path = os.environ.get("HCULT_CALIBRATION_CSV")
        if not csv_path:
            logger.error("HCULT_CALIBRATION_CSV is not configured")
            raise HTTPException(status_code=500, detail="Calibration data unavailable")
        sensor_vals, swc_vals = _load_calibration(csv_path)
        swc_min, swc_max = swc_vals.min(), swc_vals.max()
        prior_x = sensor_vals
        prior_y = (swc_vals - swc_min) / (swc_max - swc_min)
        x_anchor = np.array([sensor_vals.max()])

    swc_anchor = np.array([0.0])

    return dict(
        x_arr=x_arr,
        dx_arr=dx_arr,
        dy_arr=dy_arr,
        chord_times=chord_times,
        prior_x=prior_x,
        prior_y=prior_y,
        x_anchor=x_anchor,
        swc_anchor=swc_anchor,
        offset_ms=p.offset_ms,
        width_ms=p.width_ms,
    )


def _to_json_safe(arr):
    return [None if not math.isfinite(v) else v for v in np.asarray(arr).flat]


def _validate_estimator(estimator):
    if estimator not in ("exponential", "exp_mcmc"):
        raise HTTPException(
            status_code=400, detail="estimator must be 'exponential' or 'exp_mcmc'"
        )


def _validate_linear_prior(prior, prior_min, prior_max):
    if prior == "linear" and (prior_min is None or prior_max is None):
        raise HTTPException(
            status_code=400,
            detail="prior_min and prior_max are required for linear prior",
        )


def _calibrate(conn, p: CalibrationParams):
    d = _fetch_cord_data(conn, p)
    exp_xmax = p.prior_max if p.prior_max is not None else float(d["prior_x"].max())
    if p.estimator == "exponential":
        exp_xmin = p.prior_min if p.prior_min is not None else float(d["prior_x"].min())
        cal = ExponentialCordCalibrator(
            xmin=exp_xmin,
            xmax=exp_xmax,
            prior_weight=p.prior_weight,
        ).fit(
            d["x_anchor"],
            d["swc_anchor"],
            d["x_arr"],
            d["dx_arr"],
            d["dy_arr"],
            d["prior_x"],
            d["prior_y"],
        )
    else:
        if p.xmin_high <= p.xmin_low:
            raise HTTPException(
                status_code=400, detail="xmin_high must be greater than xmin_low"
            )
        cal = ExponentialCordCalibratorMCMC(
            xmin_low=p.xmin_low,
            xmin_high=p.xmin_high,
            xmax=exp_xmax,
            prior_weight=p.prior_weight,
            n_burn=p.n_burn,
            n_steps=p.n_steps,
        ).fit(
            d["x_anchor"],
            d["swc_anchor"],
            d["x_arr"],
            d["dx_arr"],
            d["dy_arr"],
            d["prior_x"],
            d["prior_y"],
        )
    return d, cal


@router.get("/water_cord_data")
def water_cord_data(
    p: CordDataParams = Depends(),
    prior_alpha: float = 0.5,
    conn=Depends(get_db_conn),
):
    d = _fetch_cord_data(conn, p)
    return {
        "chords_x": d["x_arr"].tolist(),
        "chords_dx": d["dx_arr"].tolist(),
        "chords_dy": d["dy_arr"].tolist(),
        "chord_times": d["chord_times"],
        "prior_x": d["prior_x"].tolist(),
        "prior_y": d["prior_y"].tolist(),
        "offset_ms": d["offset_ms"],
        "width_ms": d["width_ms"],
    }


@router.get("/water_calibration")
def water_calibration(p: CalibrationParams = Depends(), conn=Depends(get_db_conn)):
    _validate_estimator(p.estimator)
    _validate_linear_prior(p.prior, p.prior_min, p.prior_max)

    logger.info(
        "GET /water_calibration plant=%s sensor=%s device_address=%s offset_ms=%d width_ms=%d estimator=%s",
        p.plant,
        p.sensor,
        p.device_address,
        p.offset_ms,
        p.width_ms,
        p.estimator,
    )

    d, cal = _calibrate(conn, p)
    plot_x = np.linspace(d["prior_x"].min(), d["prior_x"].max(), 500)
    plot_prior_y = np.interp(plot_x, d["prior_x"], d["prior_y"])
    mean, ci_low, ci_high = cal.predict(plot_x)
    mean_at_chord_starts = cal(d["x_arr"])

    swc_after = mean_at_chord_starts + d["dy_arr"]
    estimated_mv_after = np.interp(swc_after, mean[::-1], plot_x[::-1])
    estimated_dx_arr = estimated_mv_after - d["x_arr"]

    return {
        "prior_x": _to_json_safe(plot_x),
        "prior_y": plot_prior_y.tolist(),
        "mean": _to_json_safe(mean),
        "ci_low": _to_json_safe(ci_low),
        "ci_high": _to_json_safe(ci_high),
        "scale": float(cal.scale),
        "nlml": float(cal.nlml),
        "anchors_x": d["x_anchor"].tolist(),
        "anchors_y": d["swc_anchor"].tolist(),
        "chords_x": d["x_arr"].tolist(),
        "chords_dx": d["dx_arr"].tolist(),
        "chords_dy": d["dy_arr"].tolist(),
        "mean_at_chord_starts": _to_json_safe(mean_at_chord_starts),
        "estimated_chords_dx": _to_json_safe(estimated_dx_arr),
        "chord_times": d["chord_times"],
        "offset_ms": p.offset_ms,
        "width_ms": p.width_ms,
    }


@router.get("/swc_timeseries")
def swc_timeseries(
    p: CalibrationParams = Depends(),
    start_ms: int | None = None,
    end_ms: int | None = None,
    conn=Depends(get_db_conn),
):
    _validate_estimator(p.estimator)
    _validate_linear_prior(p.prior, p.prior_min, p.prior_max)

    if not p.sensor or not p.device_address:
        raise HTTPException(
            status_code=400, detail="sensor and device_address are required"
        )

    d, cal = _calibrate(conn, p)

    end_time = end_ms if end_ms is not None else int(__import__("time").time() * 1000)
    start_time = start_ms if start_ms is not None else (end_time - 48 * 3600 * 1000)

    readings = list(
        database.fetch_timeseries(
            conn,
            plant=p.plant,
            sensor=p.sensor,
            device=p.device_address,
            start_ms=start_time,
            end_ms=end_time,
            limit=50000,
        )
    )
    if len(readings) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough sensor readings in the specified time range",
        )

    times_ms = np.array([r["adjusted_time_ms"] for r in readings])
    voltages_mv = np.array([r["voltage_mv"] for r in readings])
    order = np.argsort(times_ms)
    times_ms = times_ms[order]
    voltages_mv = voltages_mv[order]

    mean_swc = cal(voltages_mv)
    std_swc = cal.std(voltages_mv)
    ci_low = mean_swc - 1.96 * std_swc
    ci_high = mean_swc + 1.96 * std_swc

    return {
        "times_ms": [int(v) for v in times_ms],
        "mean_swc": _to_json_safe(mean_swc),
        "ci_low": _to_json_safe(ci_low),
        "ci_high": _to_json_safe(ci_high),
        "scale": float(cal.scale),
    }


@router.get("/drying_rate")
def drying_rate(
    p: CalibrationParams = Depends(),
    start_utc: str | None = None,
    end_utc: str | None = None,
    combined: bool = False,
    combine_method: str = "bayesian",
    sigma_bias: float = 0.0,
    conn=Depends(get_db_conn),
):
    from datetime import datetime, timezone

    _validate_estimator(p.estimator)

    end_dt = (
        datetime.now(timezone.utc)
        if end_utc is None
        else datetime.fromisoformat(end_utc.replace("Z", "+00:00"))
    )
    start_dt = (
        end_dt - __import__("datetime").timedelta(hours=48)
        if start_utc is None
        else datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
    )
    start_ms_time = int(start_dt.timestamp() * 1000)
    end_ms_time = int(end_dt.timestamp() * 1000)

    d, cal = _calibrate(conn, p)

    readings = list(
        database.fetch_timeseries(
            conn,
            plant=p.plant,
            sensor=p.sensor,
            device=p.device_address,
            start_ms=start_ms_time,
            end_ms=end_ms_time,
            limit=50000,
        )
    )
    if len(readings) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough sensor readings in the specified time range",
        )

    times_ms = np.array([r["adjusted_time_ms"] for r in readings])
    voltages_mv = np.array([r["voltage_mv"] for r in readings])
    order = np.argsort(times_ms)
    times_ms = times_ms[order]
    voltages_mv = voltages_mv[order]

    result = linear_drying_rate(cal, times_ms, voltages_mv)
    if result is None:
        raise HTTPException(
            status_code=400, detail="Could not estimate drying rate from the data"
        )

    return result


@router.get("/combined_swc_timeseries")
def combined_swc_timeseries(
    wp: WindowParams = Depends(),
    tp: TuningParams = Depends(),
    sigma_bias: float = 0.0,
    start_ms: int | None = None,
    end_ms: int | None = None,
    conn=Depends(get_db_conn),
):
    _validate_linear_prior(wp.prior, wp.prior_min, wp.prior_max)

    sensors = list(database.fetch_plant_sensors(conn, limit=1000))
    plant_sensors = [s for s in sensors if s["plant_name"] == wp.plant]
    if not plant_sensors:
        raise HTTPException(status_code=400, detail="No sensors found for this plant")

    calibrators = []
    sensor_keys = []
    for ps in plant_sensors:
        try:
            p_cal = CalibrationParams(
                **wp.model_dump(),
                **tp.model_dump(),
                sensor=ps["sensor"],
                device_address=ps["device_address"],
                estimator="exp_mcmc",
            )
            _, cal = _calibrate(conn, p_cal)
        except HTTPException as e:
            logger.warning(
                "combined_swc: calibration failed for %s/%s: %s",
                ps["device_address"],
                ps["sensor"],
                e.detail,
            )
            continue
        calibrators.append(cal)
        sensor_keys.append((ps["device_address"], ps["sensor"]))

    if not calibrators:
        raise HTTPException(
            status_code=400, detail="No sensors produced calibration data"
        )

    end_time = end_ms if end_ms is not None else int(__import__("time").time() * 1000)
    start_time = start_ms if start_ms is not None else (end_time - 48 * 3600 * 1000)

    all_readings = list(
        database.fetch_timeseries(
            conn,
            plant=wp.plant,
            start_ms=start_time,
            end_ms=end_time,
            limit=50000,
        )
    )

    logger.info(
        "combined_swc: plant=%s calibrators=%d sensor_keys=%s readings=%d start_ms=%s end_ms=%s",
        wp.plant,
        len(calibrators),
        sensor_keys,
        len(all_readings),
        start_time,
        end_time,
    )

    times_by_sensor = {}
    volts_by_sensor = {}
    for r in all_readings:
        key = (r["device_address"], r["sensor"])
        if key not in sensor_keys:
            continue
        times_by_sensor.setdefault(key, []).append(r["adjusted_time_ms"])
        volts_by_sensor.setdefault(key, []).append(r["voltage_mv"])

    all_times = (
        sorted(set().union(*[set(v) for v in times_by_sensor.values()]))
        if times_by_sensor
        else []
    )
    n = len(all_times)
    time_idx = {t: i for i, t in enumerate(all_times)}

    voltages_per_sensor = []
    for key in sensor_keys:
        ts = times_by_sensor.get(key, [])
        vs = volts_by_sensor.get(key, [])
        arr = np.full(n, np.nan)
        for t, v in zip(ts, vs):
            if t in time_idx:
                arr[time_idx[t]] = v
        logger.info(
            "combined_swc: key=%s ts=%d vs=%d arr_non_nan=%d",
            key,
            len(ts),
            len(vs),
            int(np.isfinite(arr).sum()),
        )
        voltages_per_sensor.append(arr)

    fused = fuse_swc(calibrators, voltages_per_sensor, sigma_bias=sigma_bias)

    logger.info(
        "combined_swc: fused mean_non_nan=%d ci_low_non_nan=%d times=%d",
        int(np.isfinite(fused["mean"]).sum()),
        int(np.isfinite(fused["ci_low"]).sum()),
        len(all_times),
    )

    return {
        "times_ms": all_times,
        "mean_swc": _to_json_safe(fused["mean"]),
        "ci_low": _to_json_safe(fused["ci_low"]),
        "ci_high": _to_json_safe(fused["ci_high"]),
        "n_sensors": fused["n_sensors"].tolist(),
        "scale": float(calibrators[0].scale),
    }
