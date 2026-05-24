from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

SENSORS_PKL = "sensor-data-LIMETREE001-2026-04-21T00:00:00Z_2026-04-28T00:00:00Z.pkl"
CHORDS_PKL = "chords-LIMETREE001-2025-04-21T00:00:00Z_2026-04-28T00:00:00Z.pkl"
PRIOR_PKL = "prior-2026-05-16T13:56:42.635558Z.pkl"

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"

N_BURN = 100
N_STEPS = 200
MIN_CHORDS_PER_SENSOR = 5


def load_calibration_data(
    data_dir: Path | None = None, min_chords_per_sensor: int = MIN_CHORDS_PER_SENSOR
):
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    with open(data_dir / SENSORS_PKL, "rb") as f:
        sensor_data = pickle.load(f)
    with open(data_dir / CHORDS_PKL, "rb") as f:
        chords = pickle.load(f)
    with open(data_dir / PRIOR_PKL, "rb") as f:
        prior = pickle.load(f)

    keys = sorted(sensor_data.keys())
    series = []
    for k in keys:
        ts = np.array([v[0].astype("datetime64[ms]") for v in sensor_data[k]])
        volts = np.array([float(v[2]) for v in sensor_data[k]], dtype=float)
        series.append((ts, volts))
    common_ts = series[0][0]
    volt_matrix = np.column_stack([s[1] for s in series])

    chords_x = np.array(chords["chords_x"])
    chords_dx = np.array(chords["chords_dx"])
    chords_dy = np.array(chords["chords_dy"])
    labels = np.array(chords["sensor_chord_labels"])

    keep = np.zeros(len(labels), dtype=bool)
    for lbl in np.unique(labels):
        if np.sum(labels == lbl) >= min_chords_per_sensor:
            keep |= labels == lbl
    labels = labels[keep]
    chords_x = chords_x[keep]
    chords_dx = chords_dx[keep]
    chords_dy = chords_dy[keep]

    n_sensors = len(np.unique(labels))
    remap = {old: new for new, old in enumerate(sorted(np.unique(labels)))}
    labels = np.array([remap[l] for l in labels])

    prior_x = np.array(prior["prior_x"])
    prior_y = np.array(prior["prior_y"])
    xmax = float(max(prior_x.max(), chords_x.max()))
    anchor_x = np.array([xmax])
    anchor_swc = np.array([0.0])

    return {
        "common_ts": common_ts,
        "volt_matrix": volt_matrix,
        "chords_x": chords_x,
        "chords_dx": chords_dx,
        "chords_dy": chords_dy,
        "labels": labels,
        "n_sensors": n_sensors,
        "prior_x": prior_x,
        "prior_y": prior_y,
        "xmax": xmax,
        "anchor_x": anchor_x,
        "anchor_swc": anchor_swc,
    }


def fit_calibrators(d, n_burn=N_BURN, n_steps=N_STEPS):
    from hcultinf.exp_mcmc import ExponentialCordCalibratorMCMC

    single_cals = []
    for s in range(d["n_sensors"]):
        mask = d["labels"] == s
        cal = ExponentialCordCalibratorMCMC(
            xmax=d["xmax"],
            n_burn=n_burn,
            n_steps=n_steps,
        ).fit(
            d["anchor_x"],
            d["anchor_swc"],
            d["chords_x"][mask],
            d["chords_dx"][mask],
            d["chords_dy"][mask],
            d["prior_x"],
            d["prior_y"],
        )
        single_cals.append(cal)

    joint_cal = ExponentialCordCalibratorMCMC(
        xmax=d["xmax"],
        n_burn=n_burn,
        n_steps=n_steps,
        n_sensors=d["n_sensors"],
    ).fit(
        d["anchor_x"],
        d["anchor_swc"],
        d["chords_x"],
        d["chords_dx"],
        d["chords_dy"],
        d["prior_x"],
        d["prior_y"],
        sensor_chord_labels=d["labels"],
    )

    return single_cals, joint_cal
