#!/usr/bin/env python3
"""Generate simulation validation figures for the paper.

Produces figures showing recovery of known calibration curve parameters
from simulated chord data using the multi-sensor MCMC calibrator.

Outputs to docs/images/:
  simulation_calibration.png    - calibration curves with true curves
  simulation_corner.png         - posterior corner plot with true values
  simulation_convergence.png    - curve error vs N
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_hcultinf_root = Path(__file__).resolve().parent.parent.parent / "hcultinf"
sys.path.insert(0, str(_hcultinf_root))

os.environ["HCULT_NO_PLOT"] = "1"

from hcultinf.exp_mcmc import ExponentialCordCalibratorMCMC
from hcultinf.exp import exponential_target
from hcultinf.simulation import simulate_calibration_data_samples
from hcultinf.plot import (
    plot_calibration_curve_simulated,
    plot_chord_noise,
    plot_convergence_error,
    plot_corner,
    plot_traces,
)

SCALE = 270.0
K_TRUTH = [5.0, 10.0]
F_INT_TRUTH = [0.05, 0.1]
XMINS = [3.0, 2.5]
XMAX = 8.5
CHORD_XMAXS = [4.0, 3.5]
DXMIN = 0.2
DXMAX = 0.5
NOISE = 0.2

# N_VALUES = [4, 8, 16, 32, 64]
N_VALUES = [64]
N_TRIALS = 8
N_BURN = 400
N_STEPS = 1000

RUN_CONFIGS = [
    {
        "name": "with_priors",
        "prior_u": np.array([1.0, 0.0]),
        "prior_y": np.array([1.0, 0.0]),
        "x_anchors": np.array([XMAX]),
        "swc_anchors": np.array([0.0]),
    },
    {
        "name": "no_priors",
        "prior_u": np.array([]),
        "prior_y": np.array([]),
        "x_anchors": np.array([]),
        "swc_anchors": np.array([]),
    },
]

TRUE_FNS = [
    lambda x: SCALE * exponential_target(x, K_TRUTH[0], F_INT_TRUTH[0], XMINS[0], XMAX),
    lambda x: SCALE * exponential_target(x, K_TRUTH[1], F_INT_TRUTH[1], XMINS[1], XMAX),
]


def _sim_chords(sensor_idx, n, seed):
    return simulate_calibration_data_samples(
        XMINS[sensor_idx],
        CHORD_XMAXS[sensor_idx],
        DXMIN,
        DXMAX,
        NOISE,
        n,
        TRUE_FNS[sensor_idx],
        uniform=False,
    )


def _fit_single(sensor_idx, x, dx, dy, cfg):
    cal = ExponentialCordCalibratorMCMC(
        xmax=XMAX,
        n_burn=N_BURN,
        n_steps=N_STEPS,
        n_sensors=1,
    )
    cal.fit(
        cfg["x_anchors"].copy(),
        cfg["swc_anchors"].copy(),
        x,
        dx,
        dy,
        prior_u=cfg["prior_u"].copy(),
        prior_y=cfg["prior_y"].copy(),
        data_xmin=np.array([XMINS[sensor_idx]]),
    )
    return cal


def _fit_joint(x0, dx0, dy0, x1, dx1, dy1, cfg):
    x_all = np.concatenate([x0, x1])
    dx_all = np.concatenate([dx0, dx1])
    dy_all = np.concatenate([dy0, dy1])
    labels = np.array([0] * len(x0) + [1] * len(x1))

    cal = ExponentialCordCalibratorMCMC(
        xmax=XMAX,
        n_burn=N_BURN,
        n_steps=N_STEPS,
        n_sensors=2,
    )
    cal.fit(
        cfg["x_anchors"].copy(),
        cfg["swc_anchors"].copy(),
        x_all,
        dx_all,
        dy_all,
        prior_u=cfg["prior_u"].copy(),
        prior_y=cfg["prior_y"].copy(),
        data_xmin=np.array(XMINS),
        sensor_chord_labels=labels,
    )
    return cal


def _curve_error(cal, sensor_idx):
    xmin = XMINS[sensor_idx]
    true_fn = TRUE_FNS[sensor_idx]
    x_grid = np.linspace(xmin, XMAX, 200)
    mean = cal(x_grid)
    true_y = true_fn(x_grid)
    return float(np.mean(np.abs(mean - true_y)))


def _joint_curve_error(cal):
    errs = []
    for s in range(2):
        xmin = XMINS[s]
        true_fn = TRUE_FNS[s]
        x_grid = np.linspace(xmin, XMAX, 200)
        x_2d = np.full((len(x_grid), 2), np.nan)
        x_2d[:, s] = x_grid
        mean = cal(x_2d)
        true_y = true_fn(x_grid)
        errs.append(float(np.mean(np.abs(mean - true_y))))
    return float(np.mean(errs))


def generate_figures(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_convergence = {}

    for cfg in RUN_CONFIGS:
        cfg_name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"Config: {cfg_name}")
        print(f"{'='*60}")

        convergence = {
            "Sensor 1 alone": ([], []),
            "Sensor 2 alone": ([], []),
            "Joint": ([], []),
        }

        last_max_n_result = None

        for n in N_VALUES:
            print(f"  N={n} ...")
            errors = {"Sensor 1 alone": [], "Sensor 2 alone": [], "Joint": []}

            for trial in range(N_TRIALS):
                seed0 = n * 1000 + trial * 2
                seed1 = n * 1000 + trial * 2 + 1

                x0, dx0, dy0 = _sim_chords(0, n, seed0)
                x1, dx1, dy1 = _sim_chords(1, n, seed1)

                if len(x0) < 2 or len(x1) < 2:
                    print(f"    trial {trial}: too few chords (s0={len(x0)}, s1={len(x1)}), skipping")
                    continue

                cal_s0 = _fit_single(0, x0, dx0, dy0, cfg)
                cal_s1 = _fit_single(1, x1, dx1, dy1, cfg)
                cal_joint = _fit_joint(x0, dx0, dy0, x1, dx1, dy1, cfg)

                errors["Sensor 1 alone"].append(_curve_error(cal_s0, 0))
                errors["Sensor 2 alone"].append(_curve_error(cal_s1, 1))
                errors["Joint"].append(_joint_curve_error(cal_joint))

                if n == N_VALUES[-1]:
                    last_max_n_result = (
                        cal_s0, cal_s1, cal_joint, x0, dx0, dy0, x1, dx1, dy1
                    )

            for key in convergence:
                arr = np.array(errors[key])
                if len(arr) > 0:
                    convergence[key][0].append(float(np.mean(arr)))
                    convergence[key][1].append(float(np.std(arr) / np.sqrt(len(arr))))
                else:
                    convergence[key][0].append(np.nan)
                    convergence[key][1].append(np.nan)

            print(
                f"    s0={convergence['Sensor 1 alone'][0][-1]:.2f} "
                f"s1={convergence['Sensor 2 alone'][0][-1]:.2f} "
                f"joint={convergence['Joint'][0][-1]:.2f}"
            )

        all_convergence[cfg_name] = convergence

        cal_s0, cal_s1, cal_joint, x0, dx0, dy0, x1, dx1, dy1 = last_max_n_result

        all_x = np.concatenate([x0, x1])
        all_dx = np.concatenate([dx0, dx1])
        all_dy = np.concatenate([dy0, dy1])
        all_labels = np.array([0] * len(x0) + [1] * len(x1))

        prefix = cfg_name

        print(f"  Generating calibration curve panel ...")
        plot_calibration_curve_simulated(
            [cal_s0, cal_s1],
            cal_joint,
            TRUE_FNS,
            XMINS,
            XMAX,
            all_x,
            all_dx,
            all_dy,
            all_labels,
            out=output_dir / f"{prefix}_simulation_calibration.png",
        )

        print(f"  Generating corner plot ...")
        true_values = {
            "scale": SCALE,
            "k": K_TRUTH,
            "f_int": F_INT_TRUTH,
            "sigma2": NOISE ** 2,
        }
        plot_corner(
            cal_joint,
            n_sensors=2,
            out=output_dir / f"{prefix}_simulation_corner.png",
            true_values=true_values,
        )

        print(f"  Generating chord noise plot ...")
        plot_chord_noise(
            all_x,
            all_dx,
            all_dy,
            all_labels,
            TRUE_FNS,
            NOISE,
            out=output_dir / f"{prefix}_simulation_chord_noise.png",
        )

        print(f"  Generating convergence plot ...")
        plot_convergence_error(
            N_VALUES,
            convergence,
            out=output_dir / f"{prefix}_simulation_convergence.png",
        )

        print(f"  Generating trace plots ...")
        for label, cal_obj, ns in [
            ("single_1", cal_s0, 1),
            ("single_2", cal_s1, 1),
            ("joint", cal_joint, 2),
        ]:
            plot_traces(
                cal_obj,
                n_sensors=ns,
                n_burn=N_BURN,
                out=output_dir / f"{prefix}_simulation_traces_{label}.png",
            )

        params = cal_joint.posterior_params()
        print(f"\n  Joint posterior means vs true values ({cfg_name}):")
        print(f"    scale:   {params['scale'].mean():.1f}  (true {SCALE})")
        for i in range(2):
            print(f"    k_{i+1}:     {params['k'][:, i].mean():.2f}  (true {K_TRUTH[i]})")
            print(f"    f_int_{i+1}: {params['f_int'][:, i].mean():.3f}  (true {F_INT_TRUTH[i]})")
        print(f"    sigma2:  {params['sigma2'].mean():.4f}  (true {NOISE**2:.4f})")

    print(f"\nDone. Figures saved to {output_dir}")


if __name__ == "__main__":
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "images"
    )
    generate_figures(out)
