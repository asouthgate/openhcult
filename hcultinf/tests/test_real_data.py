import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hcultinf.detection import smooth_and_downsample
from hcultinf.exp_mcmc import ExponentialCordCalibratorMCMC, _ParamLayout
from hcultinf.plot_style import apply_dark_theme, PALETTE

apply_dark_theme()

_NO_PLOTS = os.environ.get("HCULT_NO_PLOT", "0") == "1"

_DATA_DIR = Path(__file__).parent / "data"

SENSORS_PKL = (
    _DATA_DIR / "sensor-data-LIMETREE001-2026-04-21T00:00:00Z_2026-04-28T00:00:00Z.pkl"
)
CHORDS_PKL = (
    _DATA_DIR / "chords-LIMETREE001-2025-04-21T00:00:00Z_2026-04-28T00:00:00Z.pkl"
)
PRIOR_PKL = _DATA_DIR / "prior-2026-05-16T13:56:42.635558Z.pkl"

N_BURN = 100
N_STEPS = 200
MIN_CHORDS_PER_SENSOR = 5


def _fit(
    x_anchors,
    swc_anchors,
    x_starts,
    delta_x,
    delta_swc,
    prior_x,
    prior_y,
    xmax,
    n_sensors=1,
    sensor_chord_labels=None,
):
    cal = ExponentialCordCalibratorMCMC(
        xmax=xmax,
        n_burn=N_BURN,
        n_steps=N_STEPS,
        n_sensors=n_sensors,
    ).fit(
        x_anchors,
        swc_anchors,
        x_starts,
        delta_x,
        delta_swc,
        prior_x,
        prior_y,
        sensor_chord_labels=sensor_chord_labels,
    )
    return cal


def _assert_sandwiched(mean_a, mean_b, mean_joint, min_fraction=0.5, label=""):
    lo = np.minimum(mean_a, mean_b)
    hi = np.maximum(mean_a, mean_b)
    sandwiched = (mean_joint >= lo) & (mean_joint <= hi)
    frac = sandwiched.mean()
    assert frac >= min_fraction, (
        f"Joint SWC sandwiched by individual sensors only {frac:.1%} of the time "
        f"on {label} (need {min_fraction:.0%})"
    )


def _load_sensor_timeseries():
    with open(SENSORS_PKL, "rb") as f:
        sensor_data = pickle.load(f)
    keys = sorted(sensor_data.keys())
    series = []
    for k in keys:
        ts = np.array([v[0].astype("datetime64[ms]") for v in sensor_data[k]])
        volts = np.array([float(v[2]) for v in sensor_data[k]], dtype=float)
        series.append((ts, volts))
    common_ts = series[0][0]
    volt_matrix = np.column_stack([s[1] for s in series])
    return common_ts, volt_matrix


def test_real_data():
    with open(CHORDS_PKL, "rb") as f:
        chords = pickle.load(f)
    with open(PRIOR_PKL, "rb") as f:
        prior = pickle.load(f)

    chords_x = np.array(chords["chords_x"])
    chords_dx = np.array(chords["chords_dx"])
    chords_dy = np.array(chords["chords_dy"])
    labels = np.array(chords["sensor_chord_labels"])

    keep = np.zeros(len(labels), dtype=bool)
    for lbl in np.unique(labels):
        if np.sum(labels == lbl) >= MIN_CHORDS_PER_SENSOR:
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

    single_cals = []
    for s in range(n_sensors):
        mask = labels == s
        cal = _fit(
            anchor_x,
            anchor_swc,
            chords_x[mask],
            chords_dx[mask],
            chords_dy[mask],
            prior_x,
            prior_y,
            xmax,
        )
        single_cals.append(cal)

    joint_cal = _fit(
        anchor_x,
        anchor_swc,
        chords_x,
        chords_dx,
        chords_dy,
        prior_x,
        prior_y,
        xmax,
        n_sensors=n_sensors,
        sensor_chord_labels=labels,
    )

    x_grid = np.linspace(prior_x.min(), xmax, 500)

    single_means = [cal.predict(x_grid)[0] for cal in single_cals]
    x_grid_2d = np.column_stack([x_grid] * n_sensors)
    mean_joint, ci_lo_joint, ci_hi_joint = joint_cal.predict(x_grid_2d)

    # _assert_sandwiched(
    #     single_means[0],
    #     single_means[1],
    #     mean_joint,
    #     min_fraction=0.35,
    #     label="calibration curve",
    # )

    ts, volts = _load_sensor_timeseries()

    single_ts_means = [single_cals[i].predict(volts[:, i])[0] for i in range(n_sensors)]
    mean_ts_joint, ci_lo_ts_joint, ci_hi_ts_joint = joint_cal.predict(volts)

    # _assert_sandwiched(
    #     single_ts_means[0],
    #     single_ts_means[1],
    #     mean_ts_joint,
    #     min_fraction=0.7,
    #     label="timeseries",
    # )

    if _NO_PLOTS:
        return

    os.makedirs("artifacts", exist_ok=True)

    colors = PALETTE[:n_sensors] + [PALETTE[n_sensors]]
    names = [f"Sensor {i+1} (single)" for i in range(n_sensors)] + ["Joint (both)"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_sensors):
        _, ci_lo, ci_hi = single_cals[i].predict(x_grid)
        ax.plot(x_grid, single_means[i], color=colors[i], label=names[i])
        ax.fill_between(x_grid, ci_lo, ci_hi, color=colors[i], alpha=0.2)
    ax.plot(x_grid, mean_joint, color=colors[-1], label=names[-1])
    ax.fill_between(x_grid, ci_lo_joint, ci_hi_joint, color=colors[-1], alpha=0.2)
    ax.set_xlabel("sensor reading")
    ax.set_ylabel("SWC")
    ax.legend()
    fig.tight_layout()
    fig.savefig("artifacts/real_data_calibration.png")
    if os.environ.get("HCULT_TEST_DEBUG_PLOT", "0") == "1":
        plt.show()
    plt.close(fig)

    ts_hours = (ts - ts[0]) / np.timedelta64(1, "h")

    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(n_sensors):
        _, ci_lo, ci_hi = single_cals[i].predict(volts[:, i])
        ax.plot(
            ts_hours, single_ts_means[i], color=colors[i], label=names[i], linewidth=0.8
        )
        ax.fill_between(ts_hours, ci_lo, ci_hi, color=colors[i], alpha=0.15)
    ax.plot(ts_hours, mean_ts_joint, color=colors[-1], label=names[-1], linewidth=0.8)
    ax.fill_between(
        ts_hours, ci_lo_ts_joint, ci_hi_ts_joint, color=colors[-1], alpha=0.15
    )
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("SWC")
    ax.legend()
    fig.tight_layout()
    fig.savefig("artifacts/real_data_timeseries.png")
    if os.environ.get("HCULT_TEST_DEBUG_PLOT", "0") == "1":
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(n_sensors):
        ax.scatter(ts_hours, volts[:, i], color=colors[i], alpha=0.3, s=4, marker="x")
        ds_ts, ds_vals = smooth_and_downsample(ts, volts[:, i])
        ds_hours = (ds_ts - ds_ts[0]) / np.timedelta64(1, "h")
        ax.plot(
            ds_hours,
            ds_vals,
            color=colors[i],
            label=f"Sensor {i+1} (smoothed)",
            linewidth=1.2,
        )
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("sensor reading (mV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("artifacts/real_data_smoothed.png")
    if os.environ.get("HCULT_TEST_DEBUG_PLOT", "0") == "1":
        plt.show()
    plt.close(fig)

    # ---- MCMC trace plots ----
    all_cals = [(f"single_{i+1}", single_cals[i], 1) for i in range(n_sensors)] + [
        ("joint", joint_cal, n_sensors)
    ]

    for label, cal, cal_n in all_cals:
        layout = _ParamLayout(cal_n)
        chain = cal._chain
        log_prob = cal._log_prob
        param_names = (
            ["log_scale"]
            + [f"k_{i}" for i in range(cal_n)]
            + [f"f_int_{i}" for i in range(cal_n)]
            + ["log_sigma"]
        )

        n_params = layout.dim
        n_rows = n_params + 1
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.5 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        for p_idx in range(n_params):
            ax = axes[p_idx]
            for w in range(chain.shape[1]):
                ax.plot(chain[:, w, p_idx], linewidth=0.4, alpha=0.6)
            ax.axvline(N_BURN, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_ylabel(param_names[p_idx], fontsize=8)

        ax = axes[-1]
        for w in range(log_prob.shape[1]):
            ax.plot(log_prob[:, w], linewidth=0.3, alpha=0.6)
        ax.axvline(N_BURN, color="red", linewidth=0.8, linestyle="--", alpha=0.8)
        ax.set_ylabel("log_prob", fontsize=8)
        ax.set_xlabel("step")

        fig.suptitle(f"MCMC traces – {label}")
        fig.tight_layout()
        fig.savefig(f"artifacts/real_data_traces_{label}.png")
        if os.environ.get("HCULT_TEST_DEBUG_PLOT", "0") == "1":
            plt.show()
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    for label, cal, _ in all_cals:
        lp = cal._log_prob
        ax.plot(np.median(lp, axis=1), linewidth=0.8, alpha=0.8, label=label)
    ax.axvline(N_BURN, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("step")
    ax.set_ylabel("median log_prob")
    ax.legend()
    fig.tight_layout()
    fig.savefig("artifacts/real_data_traces_logprob.png")
    if os.environ.get("HCULT_TEST_DEBUG_PLOT", "0") == "1":
        plt.show()
    plt.close(fig)
