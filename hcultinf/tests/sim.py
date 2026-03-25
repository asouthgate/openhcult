from __future__ import annotations

import numpy as np


def logistic_x_of_swc(swc):
    return 1.0 / (1.0 + np.exp(-12.0 * (swc - 0.5)))


def invlogistic_swc_of_x(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return 0.5 + np.log(x / (1.0 - x)) / 12.0


def simulate_calibration(
    n_chords=60, x0_noise=0.005, x1_noise=0.005, x_noise=0.2, swc_max=1.0, rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    x_0 = logistic_x_of_swc(0.0) + rng.normal(0, x0_noise)
    x_1 = logistic_x_of_swc(1.0) + rng.normal(0, x1_noise)

    swc_ends_norm = rng.uniform(0.05, 0.99, n_chords)
    delta_swc_norm = np.clip(
        rng.uniform(0.01, 0.30, n_chords), 0.005, swc_ends_norm - 0.01
    )
    swc_starts_norm = swc_ends_norm - delta_swc_norm
    x_starts = logistic_x_of_swc(swc_starts_norm)
    delta_x_true = logistic_x_of_swc(swc_ends_norm) - x_starts
    delta_x = delta_x_true * (1.0 + rng.normal(0, x_noise, n_chords))

    return (
        x_0,
        x_1,
        x_starts,
        swc_starts_norm * swc_max,
        delta_x,
        delta_swc_norm * swc_max,
    )


def plot_calibration(
    x_0,
    x_1,
    x_starts,
    swc_starts,
    delta_x,
    delta_swc,
    swc_max=1.0,
    spline=None,
    title=None,
):
    """Plot response curve, boundary anchors, chord segments, and optional fit."""
    import matplotlib.pyplot as plt

    swc_curve = np.linspace(0.01, 0.99, 500)
    x_curve = logistic_x_of_swc(swc_curve)

    fig, ax = plt.subplots()
    ax.plot(x_curve, swc_curve * swc_max, "k-", lw=2, alpha=0.6, label="true")
    ax.scatter([x_0, x_1], [0.0, swc_max], zorder=5, label="boundary anchors")

    for xs, fs, dx, df in zip(x_starts, swc_starts, delta_x, delta_swc):
        ax.plot([xs, xs + dx], [fs, fs + df], "r-", alpha=0.35, lw=0.8)
    ax.scatter(x_starts, swc_starts, s=8, color="r", alpha=0.5, zorder=4)
    ax.plot([], [], "r-", alpha=0.5, label="chords")

    if spline is not None:
        x_fine = np.linspace(x_curve.min(), x_curve.max(), 500)
        ax.plot(x_fine, spline(x_fine), "b-", lw=2, label="fit")

    ax.set_xlabel("sensor reading (x)")
    ax.set_ylabel(f"swc (0–{swc_max})")
    ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return fig, ax
