from __future__ import annotations

import numpy as np


def _x_of_fc(fc):
    return 1.0 / (1.0 + np.exp(-12.0 * (fc - 0.5)))


def _fc_of_x(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return 0.5 + np.log(x / (1.0 - x)) / 12.0


def simulate_calibration(
    n_chords=60, x0_noise=0.005, x1_noise=0.005, fc_max=1.0, rng=None
):
    """Simulate a logistic FC calibration dataset.

    Boundary calibration: x_0 = sensor reading at FC=0, x_1 = sensor reading at FC=fc_max,
    both observed with Gaussian noise.

    Chord observations: intervals where delta_fc (1-30% of full range) is known exactly,
    with the corresponding delta_x derived from the true curve.

    Returns
    -------
    x_0, x_1             : noisy sensor readings at FC=0 and FC=fc_max
    x_starts, fc_starts  : start positions of each chord observation
    delta_x, delta_fc    : chord extents (fc values in [0, fc_max] scale)
    """
    if rng is None:
        rng = np.random.default_rng()

    x_0 = _x_of_fc(0.0) + rng.normal(0, x0_noise)
    x_1 = _x_of_fc(1.0) + rng.normal(0, x1_noise)

    # Sample chord ENDS uniformly so coverage is even across the fc range.
    # Sampling starts instead would clip delta_fc near fc=1, leaving the high-fc
    # region under-constrained and biasing the fit downward there.
    fc_ends_norm = rng.uniform(0.05, 0.99, n_chords)
    delta_fc_norm = np.clip(
        rng.uniform(0.01, 0.30, n_chords), 0.005, fc_ends_norm - 0.01
    )
    fc_starts_norm = fc_ends_norm - delta_fc_norm
    x_starts = _x_of_fc(fc_starts_norm)
    x_ends = _x_of_fc(fc_ends_norm)
    delta_x = x_ends - x_starts

    return x_0, x_1, x_starts, fc_starts_norm * fc_max, delta_x, delta_fc_norm * fc_max


def plot_calibration(
    x_0,
    x_1,
    x_starts,
    fc_starts,
    delta_x,
    delta_fc,
    fc_max=1.0,
    spline=None,
    title=None,
):
    """Plot response curve, boundary anchors, chord segments, and optional fit."""
    import matplotlib.pyplot as plt

    fc_curve = np.linspace(0.01, 0.99, 500)
    x_curve = _x_of_fc(fc_curve)

    fig, ax = plt.subplots()
    ax.plot(x_curve, fc_curve * fc_max, "k-", lw=2, alpha=0.6, label="true")
    ax.scatter([x_0, x_1], [0.0, fc_max], zorder=5, label="boundary anchors")

    for xs, fs, dx, df in zip(x_starts, fc_starts, delta_x, delta_fc):
        ax.plot([xs, xs + dx], [fs, fs + df], "r-", alpha=0.35, lw=0.8)
    ax.scatter(x_starts, fc_starts, s=8, color="r", alpha=0.5, zorder=4)
    ax.plot([], [], "r-", alpha=0.5, label="chords")

    if spline is not None:
        x_fine = np.linspace(x_curve.min(), x_curve.max(), 500)
        ax.plot(x_fine, spline(x_fine), "b-", lw=2, label="fit")

    ax.set_xlabel("sensor reading (x)")
    ax.set_ylabel(f"FC (0–{fc_max})")
    ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return fig, ax
