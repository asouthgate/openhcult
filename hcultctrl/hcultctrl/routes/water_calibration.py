from __future__ import annotations

import csv
import logging
import math
import os
import re

import numpy as np
from fastapi import Depends, HTTPException, APIRouter

from hcultctrl.utils import get_db_conn
from hcultdb import queries as database
from hcultinf.exp import ExponentialCordCalibrator
from hcultinf.exp_mcmc import ExponentialCordCalibratorMCMC
from hcultinf.combiner import combine_posteriors
from hcultinf.drying import linear_drying_rate

logger = logging.getLogger(__name__)
router = APIRouter()

_ML_RE = re.compile(r"\bml=(\d+(?:\.\d+)?)\b")
_DEFAULT_OFFSET_MS = 10 * 60 * 1000
_DEFAULT_WIDTH_MS = 50 * 60 * 1000


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


def _fetch_cord_data(
    conn,
    plant,
    sensor,
    device_address,
    offset_ms,
    width_ms,
    prior,
    prior_min,
    prior_max,
    prior_alpha,
):
    if sensor and not device_address:
        raise HTTPException(
            status_code=400,
            detail="device_address is required when sensor is specified",
        )
    if prior not in ("calibrated", "linear", "power"):
        raise HTTPException(
            status_code=400, detail="prior must be 'calibrated', 'linear', or 'power'"
        )

    obs = list(
        database.fetch_observations_for_plant(conn, plant_name=plant, limit=1000)
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
                plant=plant,
                sensor=sensor,
                device=device_address,
                start_ms=t_ms - offset_ms - width_ms,
                end_ms=t_ms - offset_ms,
                limit=5000,
            )
        )
        after = list(
            database.fetch_timeseries(
                conn,
                plant=plant,
                sensor=sensor,
                device=device_address,
                start_ms=t_ms + offset_ms,
                end_ms=t_ms + offset_ms + width_ms,
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
            detail=f"No sensor data found in windows around waterings (offset={offset_ms//60000}min, width={width_ms//60000}min)",
        )

    x_arr = np.array([c[0] for c in chords])
    dx_arr = np.array([c[1] for c in chords])
    dy_arr = np.array([c[2] for c in chords])

    if prior in ("linear", "power"):
        prior_x = np.linspace(prior_min, prior_max, 500)
        t = (prior_x - prior_min) / (prior_max - prior_min)
        prior_y = (1 - t) ** (1.0 if prior == "linear" else prior_alpha)
        x_anchor = np.array([prior_max])
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
        offset_ms=offset_ms,
        width_ms=width_ms,
    )


def _to_json_safe(arr):
    return [None if not math.isfinite(v) else v for v in np.asarray(arr).flat]


def _build_calibrator(
    estimator,
    prior_x,
    prior_y,
    x_anchor,
    swc_anchor,
    x_arr,
    dx_arr,
    dy_arr,
    prior_min,
    prior_max,
    prior_weight,
    n_burn,
    n_steps,
    xmin_low,
    xmin_high,
):
    exp_xmax = prior_max if prior_max is not None else float(prior_x.max())
    if estimator == "exponential":
        exp_xmin = prior_min if prior_min is not None else float(prior_x.min())
        return ExponentialCordCalibrator(
            xmin=exp_xmin,
            xmax=exp_xmax,
            prior_weight=prior_weight,
        ).fit(x_anchor, swc_anchor, x_arr, dx_arr, dy_arr, prior_x, prior_y)
    if xmin_high <= xmin_low:
        raise HTTPException(
            status_code=400,
            detail="xmin_high must be greater than xmin_low",
        )
    return ExponentialCordCalibratorMCMC(
        xmin_low=xmin_low,
        xmin_high=xmin_high,
        xmax=exp_xmax,
        prior_weight=prior_weight,
        n_burn=n_burn,
        n_steps=n_steps,
    ).fit(x_anchor, swc_anchor, x_arr, dx_arr, dy_arr, prior_x, prior_y)


@router.get("/water_cord_data")
def water_cord_data(
    plant: str,
    sensor: str | None = None,
    device_address: str | None = None,
    offset_ms: int = _DEFAULT_OFFSET_MS,
    width_ms: int = _DEFAULT_WIDTH_MS,
    prior: str = "calibrated",
    prior_min: float | None = None,
    prior_max: float | None = None,
    prior_alpha: float = 0.5,
    conn=Depends(get_db_conn),
):
    d = _fetch_cord_data(
        conn,
        plant,
        sensor,
        device_address,
        offset_ms,
        width_ms,
        prior,
        prior_min,
        prior_max,
        prior_alpha,
    )
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
def water_calibration(
    plant: str,
    sensor: str | None = None,
    device_address: str | None = None,
    offset_ms: int = _DEFAULT_OFFSET_MS,
    width_ms: int = _DEFAULT_WIDTH_MS,
    prior: str = "calibrated",
    prior_min: float | None = None,
    prior_max: float | None = None,
    prior_alpha: float = 0.5,
    estimator: str = "exp_mcmc",
    prior_weight: float = 1.0,
    n_burn: int = 10,
    n_steps: int = 30,
    xmin_low: float = 800.0,
    xmin_high: float = 1100.0,
    combined: bool = False,
    combine_method: str = "bayesian",
    sigma_bias: float = 0.0,
    conn=Depends(get_db_conn),
):
    if estimator not in ("exponential", "exp_mcmc"):
        raise HTTPException(
            status_code=400,
            detail="estimator must be 'exponential' or 'exp_mcmc'",
        )
    if prior in ("linear", "power") and (prior_min is None or prior_max is None):
        raise HTTPException(
            status_code=400,
            detail="prior_min and prior_max are required for linear and power priors",
        )
    if combined and estimator != "exp_mcmc":
        raise HTTPException(
            status_code=400,
            detail="combined mode requires estimator='exp_mcmc'",
        )
    if combine_method not in ("bayesian", "empirical_bayes"):
        raise HTTPException(
            status_code=400,
            detail="combine_method must be 'bayesian' or 'empirical_bayes'",
        )

    logger.info(
        "GET /water_calibration plant=%s sensor=%s device_address=%s offset_ms=%d width_ms=%d estimator=%s combined=%s",
        plant,
        sensor,
        device_address,
        offset_ms,
        width_ms,
        estimator,
        combined,
    )

    if combined:
        sensors = list(database.fetch_plant_sensors(conn, limit=1000))
        plant_sensors = [s for s in sensors if s["plant_name"] == plant]
        if not plant_sensors:
            raise HTTPException(
                status_code=400, detail="No sensors found for this plant"
            )
        calibrators = []
        all_x = []
        all_dx = []
        all_dy = []
        all_chord_times = []
        last_d = None
        for ps in plant_sensors:
            try:
                d = _fetch_cord_data(
                    conn,
                    plant,
                    ps["sensor"],
                    ps["device_address"],
                    offset_ms,
                    width_ms,
                    prior,
                    prior_min,
                    prior_max,
                    prior_alpha,
                )
            except HTTPException:
                continue
            cal = _build_calibrator(
                estimator,
                d["prior_x"],
                d["prior_y"],
                d["x_anchor"],
                d["swc_anchor"],
                d["x_arr"],
                d["dx_arr"],
                d["dy_arr"],
                prior_min,
                prior_max,
                prior_weight,
                n_burn,
                n_steps,
                xmin_low,
                xmin_high,
            )
            calibrators.append(cal)
            all_x.append(d["x_arr"])
            all_dx.append(d["dx_arr"])
            all_dy.append(d["dy_arr"])
            all_chord_times.append(np.array(d["chord_times"]))
            last_d = d
        if not calibrators:
            raise HTTPException(
                status_code=400, detail="No sensors produced calibration data"
            )
        grid_min = float(last_d["prior_x"].min())
        grid_max = (
            float(last_d["prior_y"].max())
            if len(last_d["prior_y"]) > 1
            else float(last_d["prior_x"].max())
        )
        grid_max = max(grid_max, float(last_d["prior_x"].max()))
        x_grid = np.linspace(grid_min, grid_max, 500)
        result = combine_posteriors(
            calibrators, x_grid, sigma_bias=sigma_bias, method=combine_method
        )
        if combine_method == "empirical_bayes":
            mean, ci_low, ci_high, est_sigma_bias = result
        else:
            mean, ci_low, ci_high = result
            est_sigma_bias = None
        first = calibrators[0]
        x_arr = np.concatenate(all_x)
        dx_arr = np.concatenate(all_dx)
        dy_arr = np.concatenate(all_dy)
        chord_times = np.concatenate(all_chord_times).tolist()
        plot_prior_y = np.interp(x_grid, last_d["prior_x"], last_d["prior_y"])
        mean_at_chord_starts = first(x_arr)
        swc_after = mean_at_chord_starts + dy_arr
        estimated_mv_after = np.interp(swc_after, mean[::-1], x_grid[::-1])
        estimated_dx_arr = estimated_mv_after - x_arr
        return {
            "prior_x": _to_json_safe(x_grid),
            "prior_y": plot_prior_y.tolist(),
            "mean": _to_json_safe(mean),
            "ci_low": _to_json_safe(ci_low),
            "ci_high": _to_json_safe(ci_high),
            "scale": float(first.scale),
            "nlml": float(first.nlml),
            "anchors_x": last_d["x_anchor"].tolist(),
            "anchors_y": last_d["swc_anchor"].tolist(),
            "chords_x": x_arr.tolist(),
            "chords_dx": dx_arr.tolist(),
            "chords_dy": dy_arr.tolist(),
            "mean_at_chord_starts": _to_json_safe(mean_at_chord_starts),
            "estimated_chords_dx": _to_json_safe(estimated_dx_arr),
            "chord_times": chord_times,
            "combined": True,
            "n_sensors": len(calibrators),
            **(
                {"estimated_sigma_bias": float(est_sigma_bias)}
                if est_sigma_bias is not None
                else {}
            ),
            "offset_ms": offset_ms,
            "width_ms": width_ms,
        }

    d = _fetch_cord_data(
        conn,
        plant,
        sensor,
        device_address,
        offset_ms,
        width_ms,
        prior,
        prior_min,
        prior_max,
        prior_alpha,
    )
    cal = _build_calibrator(
        estimator,
        d["prior_x"],
        d["prior_y"],
        d["x_anchor"],
        d["swc_anchor"],
        d["x_arr"],
        d["dx_arr"],
        d["dy_arr"],
        prior_min,
        prior_max,
        prior_weight,
        n_burn,
        n_steps,
        xmin_low,
        xmin_high,
    )
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
        "offset_ms": offset_ms,
        "width_ms": width_ms,
    }


@router.get("/drying_rate")
def drying_rate(
    plant: str,
    sensor: str | None = None,
    device_address: str | None = None,
    offset_ms: int = _DEFAULT_OFFSET_MS,
    width_ms: int = _DEFAULT_WIDTH_MS,
    prior: str = "calibrated",
    prior_min: float | None = None,
    prior_max: float | None = None,
    prior_alpha: float = 0.5,
    estimator: str = "exp_mcmc",
    prior_weight: float = 1.0,
    n_burn: int = 10,
    n_steps: int = 30,
    xmin_low: float = 800.0,
    xmin_high: float = 1100.0,
    start_utc: str | None = None,
    end_utc: str | None = None,
    combined: bool = False,
    combine_method: str = "bayesian",
    sigma_bias: float = 0.0,
    conn=Depends(get_db_conn),
):
    from datetime import datetime, timezone

    if estimator not in ("exponential", "exp_mcmc"):
        raise HTTPException(
            status_code=400, detail="estimator must be 'exponential' or 'exp_mcmc'"
        )

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
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    d = _fetch_cord_data(
        conn,
        plant,
        sensor,
        device_address,
        offset_ms,
        width_ms,
        prior,
        prior_min,
        prior_max,
        prior_alpha,
    )
    cal = _build_calibrator(
        estimator,
        d["prior_x"],
        d["prior_y"],
        d["x_anchor"],
        d["swc_anchor"],
        d["x_arr"],
        d["dx_arr"],
        d["dy_arr"],
        prior_min,
        prior_max,
        prior_weight,
        n_burn,
        n_steps,
        xmin_low,
        xmin_high,
    )

    readings = list(
        database.fetch_timeseries(
            conn,
            plant=plant,
            sensor=sensor,
            device=device_address,
            start_ms=start_ms,
            end_ms=end_ms,
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
