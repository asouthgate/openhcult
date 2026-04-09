from __future__ import annotations

import csv
import logging
import os
import re

import numpy as np
from fastapi import Depends, HTTPException, APIRouter

from hcultctrl.utils import get_db_conn
from hcultdb import queries as database
from hcultinf.inference import GPWithPriorShape

logger = logging.getLogger(__name__)
router = APIRouter()

_ML_RE = re.compile(r"\bml=(\d+(?:\.\d+)?)\b")
_DEFAULT_OFFSET_MS = 10 * 60 * 1000
_DEFAULT_WIDTH_MS = 50 * 60 * 1000


def _volume_ml(note: str) -> float | None:
    m = _ML_RE.search(note or "")
    return float(m.group(1)) if m else None


def _load_calibration(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (sensor_vals, swc_vals) sorted by sensor_val ascending.

    Handles two formats:
    - Legacy: sensor_val, swc columns
    - Experimental: ml, sensor1_raw, sensor2_raw columns — averages readings per ml level
    """
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
                for col in ("sensor1_voltage", "sensor2_voltage"):
                    if col in fieldnames:
                        by_ml.setdefault(ml, []).append(float(row[col]))
            for ml, readings in by_ml.items():
                sensor_vals.append(float(np.mean(readings)))
                swc_vals.append(ml)
    order = np.argsort(sensor_vals)
    return np.array(sensor_vals)[order], np.array(swc_vals)[order]


@router.get("/water_calibration")
def water_calibration(
    plant: str,
    offset_ms: int = _DEFAULT_OFFSET_MS,
    width_ms: int = _DEFAULT_WIDTH_MS,
    conn=Depends(get_db_conn),
):
    logger.info(
        "GET /water_calibration plant=%s offset_ms=%d width_ms=%d",
        plant,
        offset_ms,
        width_ms,
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
    for t_ms, ml in waterings:
        before = list(
            database.fetch_timeseries(
                conn,
                plant=plant,
                start_ms=t_ms - offset_ms - width_ms,
                end_ms=t_ms - offset_ms,
                limit=5000,
            )
        )
        after = list(
            database.fetch_timeseries(
                conn,
                plant=plant,
                start_ms=t_ms + offset_ms,
                end_ms=t_ms + offset_ms + width_ms,
                limit=5000,
            )
        )
        if not before or not after:
            continue
        x = float(np.mean([r["voltage_mv"] for r in before]))
        x_after = float(np.mean([r["voltage_mv"] for r in after]))
        chords.append((x, x_after - x, ml))

    if not chords:
        raise HTTPException(
            status_code=400,
            detail="No sensor data found in ±10 min windows around waterings",
        )

    x_arr = np.array([c[0] for c in chords])
    dx_arr = np.array([c[1] for c in chords])
    dy_arr = np.array([c[2] for c in chords])

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

    gp = GPWithPriorShape().fit(
        x_anchor, swc_anchor, x_arr, dx_arr, dy_arr, prior_x, prior_y
    )
    plot_x = np.linspace(prior_x.min(), prior_x.max(), 500)
    plot_prior_y = np.interp(plot_x, prior_x, prior_y)
    mean, std = gp.predict(plot_x)
    mean_at_chord_starts = gp(x_arr)

    return {
        "prior_x": plot_x.tolist(),
        "prior_y": plot_prior_y.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "scale": float(gp.scale),
        "anchors_x": x_anchor.tolist(),
        "anchors_y": swc_anchor.tolist(),
        "chords_x": x_arr.tolist(),
        "chords_dx": dx_arr.tolist(),
        "chords_dy": dy_arr.tolist(),
        "mean_at_chord_starts": mean_at_chord_starts.tolist(),
        "offset_ms": offset_ms,
        "width_ms": width_ms,
    }
