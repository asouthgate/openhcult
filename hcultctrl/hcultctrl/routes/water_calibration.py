from __future__ import annotations

import csv
from datetime import datetime, timezone, timedelta
import math
import os
import time

import numpy as np
from fastapi import Depends, HTTPException, APIRouter
from pydantic import BaseModel

from hcultctrl.utils import get_db_conn
from hcultdb import queries as database
from hcultinf.exp_mcmc import ExponentialCordCalibratorMCMC
from hcultinf.detection import smooth_and_downsample
from hcultinf.drying import compute_drying_rate
from hcultctrl.logging import (
    get_logger,
    timed_func,
    logged_timed_func_call,
)

logger = get_logger(__name__)
router = APIRouter()

_DEFAULT_OFFSET_MS = 10 * 60 * 1000
_DEFAULT_WIDTH_MS = 50 * 60 * 1000
_MIN_CHORDS = 5


class WindowParams(BaseModel):
    plant: str
    offset_ms: int = _DEFAULT_OFFSET_MS
    width_ms: int = _DEFAULT_WIDTH_MS


class TuningParams(BaseModel):
    prior_weight: float = 1.0
    n_burn: int = 100
    n_steps: int = 200
    system_capacity_mean: float | None = None
    system_capacity_std: float | None = None
    return_fractional: bool = False


class CordDataParams(WindowParams):
    sensor: str | None = None
    device_address: str | None = None
    n_recent: int | None = None
    within_ms: int | None = None


class CalibrationParams(CordDataParams, TuningParams):
    estimator: str = "exp_mcmc"
    smoothed: bool = True


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


def _resolve_sensors(conn, p: CordDataParams):
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
    all_x, all_dx, all_dy = [], [], []
    all_chord_times = []
    sensor_labels = []
    active_sensors = []

    chord_start_ms = (
        int(time.time() * 1000) - p.within_ms if p.within_ms is not None else None
    )

    for ps in sensor_specs:
        chords, chord_times = database.fetch_chords(
            conn,
            offset_ms=p.offset_ms,
            width_ms=p.width_ms,
            plant_name=plant_name,
            sensor=ps["sensor"],
            device_address=ps["device_address"],
            limit=p.n_recent if p.n_recent is not None else 1000,
            start_ms=chord_start_ms,
        )
        if len(chords) < _MIN_CHORDS:
            logger.warning(
                "Sensor %s/%s has %d chords (minimum %d), discarding from calibration",
                ps["device_address"],
                ps["sensor"],
                len(chords),
                _MIN_CHORDS,
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

    csv_path = os.environ.get("HCULT_CALIBRATION_CSV")
    if not csv_path:
        logger.error("HCULT_CALIBRATION_CSV is not configured")
        raise HTTPException(status_code=500, detail="Calibration data unavailable")
    sensor_vals, swc_vals = _load_calibration(csv_path)
    swc_min, swc_max = swc_vals.min(), swc_vals.max()
    prior_x = sensor_vals
    prior_y = (swc_vals - swc_min) / (swc_max - swc_min)
    x_anchors = np.array([sensor_vals.max()])
    swc_anchors = np.array([0.0])

    return dict(
        x_arr=np.array(all_x),
        dx_arr=np.array(all_dx),
        dy_arr=np.array(all_dy),
        chord_times=all_chord_times,
        prior_x=prior_x,
        prior_y=prior_y,
        x_anchors=x_anchors,
        swc_anchors=swc_anchors,
        offset_ms=p.offset_ms,
        width_ms=p.width_ms,
        sensor_chord_labels=np.array(sensor_labels),
        active_sensors=active_sensors,
    )


def _to_json_safe(arr):
    return [None if not math.isfinite(v) else v for v in np.asarray(arr).flat]


def _build_chord_voltage_matrix(chords_x, sensor_chord_labels, n_sensors):
    n = len(chords_x)
    X = np.full((n, n_sensors), np.nan)
    labels = np.asarray(sensor_chord_labels)
    for j in range(n_sensors):
        mask = labels == j
        X[mask, j] = chords_x[mask]
    return X


def _predict_per_sensor_curves(
    cal, domain_min, domain_max, active_sensors, return_fractional
):
    n_sensors = len(active_sensors)
    if n_sensors <= 1:
        return []
    n_plot = 200
    curves = []
    for j in range(n_sensors):
        x_grid = np.linspace(float(cal.data_xmin[j]), domain_max, n_plot)
        X = np.full((n_plot, n_sensors), np.nan)
        X[:, j] = x_grid
        if return_fractional:
            mean, ci_low, ci_high = cal.fractional_water_content(X)
        else:
            mean, ci_low, ci_high = cal.predict(X)
        sensor_spec = active_sensors[j]
        curves.append(
            {
                "sensor": sensor_spec["sensor"],
                "device_address": sensor_spec["device_address"],
                "x": x_grid,
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return curves


def _predict_joint_diagonal(cal, n_points, return_fractional):
    if cal.n_sensors <= 1:
        return None
    t = np.linspace(0, 1, n_points)[:, None]
    diag = cal.xmax - t * (cal.xmax - cal.data_xmin[None, :])
    diag_x = np.nanmean(diag, axis=1)
    if return_fractional:
        diag_mean, diag_ci_low, diag_ci_high = cal.fractional_water_content(diag)
    else:
        diag_mean, diag_ci_low, diag_ci_high = cal.predict(diag)
    return {
        "x": diag_x,
        "mean": diag_mean,
        "ci_low": diag_ci_low,
        "ci_high": diag_ci_high,
    }


def _validate_estimator(estimator):
    if estimator not in ("exponential", "exp_mcmc"):
        raise HTTPException(
            status_code=400, detail="estimator must be 'exponential' or 'exp_mcmc'"
        )


def _calibrate(d, p: CalibrationParams):
    active_sensors = d["active_sensors"]
    n_sensors = len(active_sensors)
    exp_xmax = max(
        float(d["prior_x"].max()),
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
            sigma_prior=p.prior_weight,
            n_burn=p.n_burn,
            n_steps=p.n_steps,
            system_capacity_mean=p.system_capacity_mean,
            system_capacity_std=p.system_capacity_std,
        ).fit(
            d["x_anchors"],
            d["swc_anchors"],
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


def _readings_to_input_data_array(all_readings, sensor_keys):
    if not sensor_keys:
        return [], np.empty((0, 0)), set()

    keys_set = set(sensor_keys)
    times_by_sensor = {k: [] for k in sensor_keys}
    volts_by_sensor = {k: [] for k in sensor_keys}

    for r in all_readings:
        key = (r["device_address"], r["sensor"])
        if key not in keys_set:
            continue
        times_by_sensor[key].append(r["adjusted_time_ms"])
        volts_by_sensor[key].append(r["voltage_mv"])

    sensors_with_data = {k for k in sensor_keys if times_by_sensor[k]}
    sensors_without_data = {k for k in sensor_keys if not times_by_sensor[k]}

    if sensors_without_data:
        for k in sorted(sensors_without_data):
            logger.warning(
                "Sensor %s/%s has no readings in time range, column will be NaN",
                k[0],
                k[1],
            )

    if not sensors_with_data:
        return [], np.empty((0, 0)), sensors_with_data

    all_times = sorted(
        set().union(
            *(set(times_by_sensor[k]) for k in sensor_keys if times_by_sensor[k])
        )
    )
    n = len(all_times)
    time_idx = {t: i for i, t in enumerate(all_times)}

    same_timestamps = all(
        set(times_by_sensor[k])
        == set(times_by_sensor[next(k for k in sensor_keys if times_by_sensor[k])])
        for k in sensor_keys
        if times_by_sensor[k]
    )
    if not same_timestamps and len(sensors_with_data) > 1:
        logger.warning(
            "Sensors have different timestamps — no interpolation is performed, "
            "NaNs will appear in gaps. all_times=%d",
            n,
        )

    voltages_per_sensor = []
    for key in sensor_keys:
        ts = times_by_sensor[key]
        vs = volts_by_sensor[key]
        arr = np.full(n, np.nan)
        for t, v in zip(ts, vs):
            if t in time_idx:
                arr[time_idx[t]] = v
        voltages_per_sensor.append(arr)

    X = np.column_stack(voltages_per_sensor)
    n_nan = int(np.sum(~np.isfinite(X)))
    if n_nan > 0:
        total = X.size
        logger.warning(
            "Aligned voltage matrix has %d NaN values out of %d (%.1f%%)",
            n_nan,
            total,
            100.0 * n_nan / total if total else 0,
        )

    return all_times, X, sensors_with_data


def _compute_drying_result(
    times_ms_arr, swc_samples, scale, lambda_tv=None, max_samples=100
):
    n_samples = swc_samples.shape[0]
    n_times = swc_samples.shape[1]

    n_result_samples = min(n_samples, max_samples)
    rates_per_sample = np.empty((n_result_samples, n_times))

    logger.info(
        "Beginning drying rate computation for n_samples=%d n_times=%d",
        n_samples,
        n_times,
    )
    for j in range(n_result_samples):
        i = np.random.randint(n_samples)
        result = compute_drying_rate(
            times_ms_arr,
            swc_samples[i],
            lambda_tv=lambda_tv,
        )
        rates_per_sample[j] = result["rate"]
        logger.info(
            "Completed drying rate computation for sample %d of %d",
            j + 1,
            n_result_samples,
        )

    rate_ml_per_day = rates_per_sample * (24 * 60 * 60 * 1000)
    mean_rate = np.nanmean(rate_ml_per_day, axis=0)
    ci_low = np.nanpercentile(rate_ml_per_day, 2.5, axis=0)
    ci_high = np.nanpercentile(rate_ml_per_day, 97.5, axis=0)

    return {
        "times_ms": [int(v) for v in times_ms_arr],
        "rate_ml_per_day": _to_json_safe(mean_rate),
        "rate_ml_per_day_ci_low": _to_json_safe(ci_low),
        "rate_ml_per_day_ci_high": _to_json_safe(ci_high),
        "scale": scale,
    }


@timed_func
def _smooth_voltage_matrix(times_ms, X):
    ts_dt = times_ms.astype("datetime64[ms]")
    smoothed_cols = []
    for i in range(X.shape[1]):
        _, col_smooth = smooth_and_downsample(ts_dt, X[:, i])
        smoothed_cols.append(col_smooth)
    ds_ts, _ = smooth_and_downsample(ts_dt, X[:, 0])
    X_smooth = np.column_stack(smoothed_cols)
    return ds_ts.astype("datetime64[ms]").astype("int64"), X_smooth


def _predict_swc_timeseries(
    conn, cal, sensor_keys, plant, start_time, end_time, fractional=False, smoothed=True
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
    all_times, X, sensors_with_data = _readings_to_input_data_array(
        all_readings, sensor_keys
    )
    n_active_sensors = len(sensors_with_data)
    if not n_active_sensors:
        raise HTTPException(
            status_code=400,
            detail="No sensor data available in the specified time range",
        )
    if len(all_times) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough sensor readings in the specified time range",
        )

    times_ms = np.array(all_times)

    if smoothed:
        times_ms, X = _smooth_voltage_matrix(times_ms, X)

    logger.info(
        "_predict_swc_timeseries: n_readings=%d n_times=%d n_active_sensors=%d "
        "voltage_min=%s voltage_max=%s n_nan_in_X=%s "
        "cal_data_xmin=%s cal_xmax=%s",
        len(all_readings),
        len(times_ms),
        n_active_sensors,
        float(np.nanmin(X)),
        float(np.nanmax(X)),
        int(np.sum(~np.isfinite(X))),
        cal._data_xmin.tolist() if cal._data_xmin is not None else "none",
        cal._xmax,
    )

    if fractional:
        mean_swc, ci_low, ci_high = cal.fractional_water_content(X)
    else:
        mean_swc, ci_low, ci_high = cal.predict(X)
    return times_ms, mean_swc, ci_low, ci_high


@router.get("/prior")
def prior(conn=Depends(get_db_conn)):
    csv_path = os.environ.get("HCULT_CALIBRATION_CSV")
    if not csv_path:
        raise HTTPException(status_code=500, detail="Calibration data unavailable")
    sensor_vals, swc_vals = _load_calibration(csv_path)
    swc_min, swc_max = swc_vals.min(), swc_vals.max()
    prior_x = sensor_vals.tolist()
    prior_y = ((swc_vals - swc_min) / (swc_max - swc_min)).tolist()
    return {"prior_x": prior_x, "prior_y": prior_y}


@router.get("/chords")
def chords(p: CordDataParams = Depends(), conn=Depends(get_db_conn)):
    sensor_specs = _resolve_sensors(conn, p)
    all_x, all_dx, all_dy, all_chord_times = [], [], [], []
    sensor_chord_labels = []
    active_sensors = []
    chord_start_ms = (
        int(time.time() * 1000) - p.within_ms if p.within_ms is not None else None
    )

    for ps in sensor_specs:
        chords, chord_times = database.fetch_chords(
            conn,
            offset_ms=p.offset_ms,
            width_ms=p.width_ms,
            plant_name=p.plant,
            sensor=ps["sensor"],
            device_address=ps["device_address"],
            limit=p.n_recent if p.n_recent is not None else 1000,
            start_ms=chord_start_ms,
        )
        if len(chords) < _MIN_CHORDS:
            logger.warning(
                "Sensor %s/%s has %d chords (minimum %d), discarding",
                ps["device_address"],
                ps["sensor"],
                len(chords),
                _MIN_CHORDS,
            )
            continue
        sensor_idx = len(active_sensors)
        active_sensors.append(ps)
        for (x, dx, dy), t in zip(chords, chord_times):
            all_x.append(x)
            all_dx.append(dx)
            all_dy.append(dy)
            all_chord_times.append(t)
            sensor_chord_labels.append(sensor_idx)
    return {
        "chords_x": all_x,
        "chords_dx": all_dx,
        "chords_dy": all_dy,
        "chord_times": all_chord_times,
        "offset_ms": p.offset_ms,
        "width_ms": p.width_ms,
        "sensor_chord_labels": sensor_chord_labels,
        "active_sensors": active_sensors,
    }


@router.get("/water_calibration")
def water_calibration(p: CalibrationParams = Depends(), conn=Depends(get_db_conn)):
    _validate_estimator(p.estimator)

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
    cal = _calibrate(d, p)
    n_sensors = len(d["active_sensors"])
    domain_min = min(
        d["x_arr"].min(), (d["x_arr"] + d["dx_arr"]).min(), d["prior_x"].min()
    )
    domain_max = max(
        d["x_arr"].max(), (d["x_arr"] + d["dx_arr"]).max(), d["prior_x"].max()
    )

    use_fractional = (
        p.return_fractional
        and p.system_capacity_mean is not None
        and p.system_capacity_std is not None
    )

    if n_sensors > 1:
        chord_X = _build_chord_voltage_matrix(
            d["x_arr"], d["sensor_chord_labels"], n_sensors
        )
        mean_at_chord_starts = cal(chord_X)
        per_sensor_curves = _predict_per_sensor_curves(
            cal, domain_min, domain_max, d["active_sensors"], use_fractional
        )
        joint_diagonal = _predict_joint_diagonal(cal, 500, use_fractional)
        plot_x = np.asarray(joint_diagonal["x"])
        mean = np.asarray(joint_diagonal["mean"])
        ci_low = np.asarray(joint_diagonal["ci_low"])
        ci_high = np.asarray(joint_diagonal["ci_high"])
        plot_prior_x = np.linspace(domain_min, domain_max, 500)
        plot_prior_y = np.interp(plot_prior_x, d["prior_x"], d["prior_y"])
    else:
        chord_X = d["x_arr"]
        mean_at_chord_starts = cal(chord_X)
        plot_x = np.linspace(domain_min, domain_max, 500)
        if use_fractional:
            mean, ci_low, ci_high = cal.fractional_water_content(plot_x)
        else:
            mean, ci_low, ci_high = cal.predict(plot_x)
        plot_prior_x = plot_x
        plot_prior_y = np.interp(plot_x, d["prior_x"], d["prior_y"])
        per_sensor_curves = []
        joint_diagonal = None

    fractional = use_fractional

    swc_after = mean_at_chord_starts + d["dy_arr"]
    estimated_mv_after = np.interp(swc_after, mean[::-1], plot_x[::-1])
    estimated_dx_arr = estimated_mv_after - d["x_arr"]

    return {
        "curve_x": _to_json_safe(plot_x),
        "prior_x": _to_json_safe(plot_prior_x),
        "prior_y": _to_json_safe(plot_prior_y),
        "mean": _to_json_safe(mean),
        "ci_low": _to_json_safe(ci_low),
        "ci_high": _to_json_safe(ci_high),
        "scale": float(cal.scale) if math.isfinite(cal.scale) else None,
        "nlml": float(cal.nlml) if math.isfinite(cal.nlml) else None,
        "fractional": fractional,
        "chords_x": _to_json_safe(d["x_arr"]),
        "chords_dx": _to_json_safe(d["dx_arr"]),
        "chords_dy": _to_json_safe(d["dy_arr"]),
        "mean_at_chord_starts": _to_json_safe(mean_at_chord_starts),
        "estimated_chords_dx": _to_json_safe(estimated_dx_arr),
        "chord_times": d["chord_times"],
        "offset_ms": p.offset_ms,
        "width_ms": p.width_ms,
        "n_sensors": n_sensors,
        "sensor_chord_labels": d["sensor_chord_labels"].tolist(),
        "active_sensors": d["active_sensors"],
        "per_sensor_curves": [
            {
                "sensor": c["sensor"],
                "device_address": c["device_address"],
                "x": _to_json_safe(c["x"]),
                "mean": _to_json_safe(c["mean"]),
                "ci_low": _to_json_safe(c["ci_low"]),
                "ci_high": _to_json_safe(c["ci_high"]),
            }
            for c in per_sensor_curves
        ],
    }


@router.get("/swc_timeseries")
def swc_timeseries(
    p: CalibrationParams = Depends(),
    start_ms: int | None = None,
    end_ms: int | None = None,
    conn=Depends(get_db_conn),
):
    _validate_estimator(p.estimator)

    end_time = end_ms if end_ms is not None else int(time.time() * 1000)
    start_time = start_ms if start_ms is not None else (end_time - 48 * 3600 * 1000)

    sensor_specs = _resolve_sensors(conn, p)
    d = _fetch_cord_data(conn, p.plant, sensor_specs, p)
    cal = _calibrate(d, p)

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
        smoothed=p.smoothed,
    )

    logger.info(
        "swc_timeseries: n_times=%d n_nan_swc=%d n_nan_ci_low=%d n_nan_ci_high=%d "
        "swc_min=%s swc_max=%s data_xmin=%s xmax=%s",
        len(times_ms),
        int(np.sum(~np.isfinite(mean_swc))),
        int(np.sum(~np.isfinite(ci_low))),
        int(np.sum(~np.isfinite(ci_high))),
        float(np.nanmin(mean_swc)) if np.any(np.isfinite(mean_swc)) else "all_nan",
        float(np.nanmax(mean_swc)) if np.any(np.isfinite(mean_swc)) else "all_nan",
        cal._data_xmin.tolist() if cal._data_xmin is not None else "none",
        cal._xmax,
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
    lambda_tv: float | None = None,
    conn=Depends(get_db_conn),
):

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
    cal = _calibrate(d, p)

    sensor_keys = [(ps["device_address"], ps["sensor"]) for ps in d["active_sensors"]]
    all_readings = logged_timed_func_call(
        logger,
        database.fetch_timeseries,
        conn,
        plant=p.plant,
        start_ms=start_ms_time,
        end_ms=end_ms_time,
        limit=50000,
    )

    all_times, X, sensors_with_data = _readings_to_input_data_array(
        all_readings, sensor_keys
    )
    n_active_sensors = len(sensors_with_data)
    if not n_active_sensors:
        raise HTTPException(
            status_code=400,
            detail="No sensor data available in the specified time range",
        )
    times_ms = np.array(all_times)

    logger.info(f"Retrieved n_times_ms={len(times_ms)} for drying_rate calculation")

    if p.smoothed:
        times_ms, X = _smooth_voltage_matrix(times_ms, X)
        logger.info(
            f"Smoothed and resampled to to n_times_ms={len(times_ms)} for drying_rate calculation"
        )

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
