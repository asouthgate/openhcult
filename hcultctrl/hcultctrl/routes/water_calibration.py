from __future__ import annotations

import csv
import logging
import math
import os
import time

import numpy as np
from fastapi import Depends, HTTPException, APIRouter
from pydantic import BaseModel

from hcultctrl.utils import get_db_conn
from hcultdb import queries as database
from hcultinf.exp_mcmc import ExponentialCordCalibratorMCMC
from hcultinf.drying import drying_rate as compute_drying_rate

logger = logging.getLogger(__name__)
router = APIRouter()

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
    system_capacity_mean: float | None = None
    system_capacity_std: float | None = None
    return_fractional: bool = False


class CordDataParams(WindowParams):
    sensor: str | None = None
    device_address: str | None = None


class CalibrationParams(CordDataParams, TuningParams):
    estimator: str = "exp_mcmc"


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


def _resolve_sensors(conn, p: CalibrationParams):
    if p.sensor and not p.device_address:
        raise HTTPException(
            status_code=400,
            detail="device_address is required when sensor is specified",
        )
    if p.sensor and p.device_address:
        return [{"device_address": p.device_address, "sensor": p.sensor}]
    sensors = list(database.fetch_plant_sensors(conn, limit=1000))
    plant_sensors = [s for s in sensors if s["plant_name"] == p.plant]
    if not plant_sensors:
        raise HTTPException(status_code=400, detail="No sensors found for this plant")
    return plant_sensors


def _fetch_cord_data(conn, plant_name, sensor_specs, p):
    if p.prior not in ("calibrated", "linear"):
        raise HTTPException(
            status_code=400, detail="prior must be 'calibrated' or 'linear'"
        )

    all_x, all_dx, all_dy = [], [], []
    all_chord_times = []
    sensor_labels = []
    active_sensors = []

    for ps in sensor_specs:
        chords, chord_times = database.fetch_chords(
            conn,
            offset_ms=p.offset_ms,
            width_ms=p.width_ms,
            plant_name=plant_name,
            sensor=ps["sensor"],
            device_address=ps["device_address"],
        )
        if not chords:
            logger.warning(
                "No chords for %s/%s, skipping",
                ps["device_address"],
                ps["sensor"],
            )
            continue
        sensor_idx = len(active_sensors)
        active_sensors.append(ps)
        for (x, dx, dy), t in zip(chords, chord_times):
            all_x.append(x)
            all_dx.append(dx)
            all_dy.append(dy)
            all_chord_times.append(t)
            sensor_labels.append(sensor_idx)

    if not all_x:
        raise HTTPException(
            status_code=400,
            detail=f"No sensor data found in windows around waterings (offset={p.offset_ms//60000}min, width={p.width_ms//60000}min)",
        )

    if p.prior == "linear":
        prior_x = np.linspace(p.prior_min, p.prior_max, 500)
        t = (prior_x - p.prior_min) / (prior_x.max() - prior_x.min())
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
        x_arr=np.array(all_x),
        dx_arr=np.array(all_dx),
        dy_arr=np.array(all_dy),
        chord_times=all_chord_times,
        prior_x=prior_x,
        prior_y=prior_y,
        x_anchor=x_anchor,
        swc_anchor=swc_anchor,
        offset_ms=p.offset_ms,
        width_ms=p.width_ms,
        sensor_chord_labels=np.array(sensor_labels),
        active_sensors=active_sensors,
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


def _calibrate(conn, d, p: CalibrationParams):
    active_sensors = d["active_sensors"]
    n_sensors = len(active_sensors)
    exp_xmax = max(
        p.prior_max if p.prior_max is not None else float(d["prior_x"].max()),
        float(d["x_arr"].max()),
    )
    logger.info(
        "Calibrating n_sensors=%d xmax=%.1f",
        n_sensors,
        exp_xmax,
    )
    try:
        cal = ExponentialCordCalibratorMCMC(
            n_sensors=n_sensors,
            xmax=exp_xmax,
            prior_weight=p.prior_weight,
            n_burn=p.n_burn,
            n_steps=p.n_steps,
            system_capacity_mean=p.system_capacity_mean,
            system_capacity_std=p.system_capacity_std,
        ).fit(
            d["x_anchor"],
            d["swc_anchor"],
            d["x_arr"],
            d["dx_arr"],
            d["dy_arr"],
            d["prior_x"],
            d["prior_y"],
            sensor_chord_labels=d["sensor_chord_labels"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return cal


def _align_readings_by_sensor(all_readings, sensor_keys):
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
        voltages_per_sensor.append(arr)

    return all_times, voltages_per_sensor


def _compute_drying_result(times_ms_arr, swc_samples, scale, lambda_tv=None):
    n_samples = swc_samples.shape[0]
    n_times = swc_samples.shape[1]
    rates_per_sample = np.empty((n_samples, n_times))
    valid_per_sample = np.empty((n_samples, n_times), dtype=bool)

    for i in range(n_samples):
        result = compute_drying_rate(
            times_ms_arr,
            swc_samples[i],
            lambda_tv=lambda_tv,
        )
        rates_per_sample[i] = result["rate"]
        valid_per_sample[i] = result["valid"]

    rate_ml_per_day = rates_per_sample * 1440.0
    mean_rate = np.nanmean(rate_ml_per_day, axis=0)
    ci_low = np.nanpercentile(rate_ml_per_day, 2.5, axis=0)
    ci_high = np.nanpercentile(rate_ml_per_day, 97.5, axis=0)
    valid = valid_per_sample.sum(axis=0) > n_samples // 2

    return {
        "times_ms": [int(v) for v in times_ms_arr],
        "rate_ml_per_day": _to_json_safe(mean_rate),
        "rate_ml_per_day_ci_low": _to_json_safe(ci_low),
        "rate_ml_per_day_ci_high": _to_json_safe(ci_high),
        "valid": valid.tolist(),
        "scale": scale,
    }


def _predict_swc_timeseries(
    conn, cal, sensor_keys, plant, start_time, end_time, fractional=False
):
    all_readings = list(
        database.fetch_timeseries(
            conn,
            plant=plant,
            start_ms=start_time,
            end_ms=end_time,
            limit=50000,
        )
    )
    all_times, voltages_per_sensor = _align_readings_by_sensor(
        all_readings, sensor_keys
    )
    if len(all_times) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough sensor readings in the specified time range",
        )
    X = np.column_stack(voltages_per_sensor)
    if fractional:
        mean_swc, ci_low, ci_high = cal.fractional_water_content(X)
    else:
        mean_swc, ci_low, ci_high = cal.predict(X)
    times_ms = np.array(all_times)
    return times_ms, mean_swc, ci_low, ci_high


@router.get("/water_cord_data")
def water_cord_data(
    p: CordDataParams = Depends(),
    prior_alpha: float = 0.5,
    conn=Depends(get_db_conn),
):
    sensor_specs = _resolve_sensors(conn, p)
    d = _fetch_cord_data(conn, p.plant, sensor_specs, p)
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

    sensor_specs = _resolve_sensors(conn, p)
    d = _fetch_cord_data(conn, p.plant, sensor_specs, p)
    cal = _calibrate(conn, d, p)
    plot_x = np.linspace(d["prior_x"].min(), d["prior_x"].max(), 500)
    plot_prior_y = np.interp(plot_x, d["prior_x"], d["prior_y"])

    if (
        p.return_fractional
        and p.system_capacity_mean is not None
        and p.system_capacity_std is not None
    ):
        mean, ci_low, ci_high = cal.fractional_water_content(plot_x)
        fractional = True
    else:
        mean, ci_low, ci_high = cal.predict(plot_x)
        fractional = False

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
        "fractional": fractional,
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

    end_time = end_ms if end_ms is not None else int(time.time() * 1000)
    start_time = start_ms if start_ms is not None else (end_time - 48 * 3600 * 1000)

    sensor_specs = _resolve_sensors(conn, p)
    d = _fetch_cord_data(conn, p.plant, sensor_specs, p)
    cal = _calibrate(conn, d, p)

    sensor_keys = [(ps["device_address"], ps["sensor"]) for ps in d["active_sensors"]]
    times_ms, mean_swc, ci_low, ci_high = _predict_swc_timeseries(
        conn,
        cal,
        sensor_keys,
        p.plant,
        start_time,
        end_time,
        fractional=p.return_fractional
        and p.system_capacity_mean is not None
        and p.system_capacity_std is not None,
    )

    return {
        "times_ms": [int(v) for v in times_ms],
        "mean_swc": _to_json_safe(mean_swc),
        "ci_low": _to_json_safe(ci_low),
        "ci_high": _to_json_safe(ci_high),
        "scale": float(cal.scale),
        "fractional": p.return_fractional
        and p.system_capacity_mean is not None
        and p.system_capacity_std is not None,
    }


@router.get("/drying_rate")
def drying_rate(
    p: CalibrationParams = Depends(),
    start_utc: str | None = None,
    end_utc: str | None = None,
    combined: bool = False,
    combine_method: str = "bayesian",
    sigma_bias: float = 0.0,
    lambda_tv: float | None = None,
    conn=Depends(get_db_conn),
):
    from datetime import datetime, timezone, timedelta

    _validate_estimator(p.estimator)

    end_dt = (
        datetime.now(timezone.utc)
        if end_utc is None
        else datetime.fromisoformat(end_utc.replace("Z", "+00:00"))
    )
    start_dt = (
        end_dt - timedelta(hours=48)
        if start_utc is None
        else datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
    )
    start_ms_time = int(start_dt.timestamp() * 1000)
    end_ms_time = int(end_dt.timestamp() * 1000)

    sensor_specs = _resolve_sensors(conn, p)
    d = _fetch_cord_data(conn, p.plant, sensor_specs, p)
    cal = _calibrate(conn, d, p)

    sensor_keys = [(ps["device_address"], ps["sensor"]) for ps in d["active_sensors"]]
    all_readings = list(
        database.fetch_timeseries(
            conn,
            plant=p.plant,
            start_ms=start_ms_time,
            end_ms=end_ms_time,
            limit=50000,
        )
    )
    all_times, voltages_per_sensor = _align_readings_by_sensor(
        all_readings, sensor_keys
    )
    X = np.column_stack(voltages_per_sensor)
    times_ms = np.array(all_times)

    use_fractional = (
        p.return_fractional
        and p.system_capacity_mean is not None
        and p.system_capacity_std is not None
    )

    if use_fractional:
        swc_samples = cal.fractional_water_content_samples(X)
    else:
        swc_samples = cal.posterior_samples_swc_at(X)

    return _compute_drying_result(
        times_ms,
        swc_samples,
        float(cal.scale),
        lambda_tv=lambda_tv,
    )
