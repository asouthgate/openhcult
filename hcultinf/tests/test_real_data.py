import os

import matplotlib.pyplot as plt
import numpy as np

from hcultinf.data import load_calibration_data, fit_calibrators, N_BURN, N_STEPS
from hcultinf.exp_mcmc import _ParamLayout
from hcultinf.plot import (
    plot_calibration_curves,
    plot_timeseries,
    plot_smoothed_voltage,
    plot_corner,
)
from hcultinf.plot_style import apply_dark_theme, PALETTE

apply_dark_theme()

_NO_PLOTS = os.environ.get("HCULT_NO_PLOT", "0") == "1"


def test_real_data():
    d = load_calibration_data()
    single_cals, joint_cal = fit_calibrators(d)

    n_sensors = d["n_sensors"]

    if _NO_PLOTS:
        return

    os.makedirs("artifacts", exist_ok=True)

    plot_calibration_curves(
        single_cals,
        joint_cal,
        d,
        out="artifacts/real_data_calibration.png",
    )

    plot_timeseries(
        single_cals,
        joint_cal,
        d,
        out="artifacts/real_data_timeseries.png",
    )

    plot_smoothed_voltage(d, out="artifacts/real_data_smoothed.png")

    plot_corner(
        joint_cal,
        n_sensors=n_sensors,
        out="artifacts/real_data_corner.png",
        title="Real data MCMC posterior",
    )

    colors = PALETTE[:n_sensors] + [PALETTE[n_sensors]]
    names = [f"Sensor {i+1} (single)" for i in range(n_sensors)] + ["Joint (both)"]

    x_grid = np.linspace(float(d["data_xmin"].min()), d["xmax"], 500)

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
        ax.set_ylabel("logprob", fontsize=8)
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
