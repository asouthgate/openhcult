from __future__ import annotations

import numpy as np


def logistic_x_of_swc(swc):
    return 1.0 / (1.0 + np.exp(-12.0 * (swc - 0.5)))


def invlogistic_swc_of_x(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return 0.5 + np.log(x / (1.0 - x)) / 12.0


def simulate_calibration(
    n_chords=60,
    x0=0.0,
    xend=1.0,
    x0_noise=0.005,
    x1_noise=0.005,
    x_noise=0.2,
    swc_max=1.0,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    x_0 = logistic_x_of_swc(0.0) + rng.normal(0, x0_noise)
    x_1 = logistic_x_of_swc(1.0) + rng.normal(0, x1_noise)

    swc_starts_norm = rng.uniform(x0, xend, n_chords)
    delta_swc_norm = rng.uniform(0.01, 0.30, n_chords)

    swc_ends_norm = swc_starts_norm + delta_swc_norm

    clipped_swc_ends_norm = np.clip(swc_ends_norm, 0.0, 1.0)

    x_starts = logistic_x_of_swc(swc_starts_norm)
    delta_x_noerr = logistic_x_of_swc(clipped_swc_ends_norm) - x_starts
    delta_x = delta_x_noerr * (1.0 + rng.normal(0, x_noise, n_chords))

    scaled_swc_ends = clipped_swc_ends_norm * swc_max
    scaled_swc_starts = swc_starts_norm * swc_max
    scaled_swc_deltas = scaled_swc_ends - scaled_swc_starts

    assert (
        max(scaled_swc_starts) <= swc_max
    ), f"{max(scaled_swc_starts)} should be less than {swc_max}"
    assert all(scaled_swc_starts) <= swc_max
    assert all(delta_swc_norm) >= 0
    assert all(scaled_swc_starts + scaled_swc_deltas <= swc_max)
    return (
        x_0,
        x_1,
        x_starts,
        scaled_swc_starts,
        delta_x,
        scaled_swc_deltas,
    )


def plot_calibration(
    x_0,
    x_1,
    x_starts,
    swc_starts,
    delta_x,
    delta_swc,
    swc_max=1.0,
    title=None,
    swc_max_est=None,
    inferred=None,
    save_path=None,
    ax=None,
    legend=True,
):
    """Plot response curve, boundary anchors, chord segments, and optional fit."""
    assert (
        max(swc_starts) <= swc_max
    ), f"{max(swc_starts)} should be less than {swc_max}"
    import matplotlib.pyplot as plt

    swc_curve = np.linspace(0.01, 0.99, 500)
    x_curve = logistic_x_of_swc(swc_curve)

    caller_owns_ax = ax is not None
    if caller_owns_ax:
        fig = ax.figure
    else:
        fig, ax = plt.subplots()
    ax.plot([], [], "r-", alpha=0.1, label="chords")
    ax.plot(x_curve, swc_curve * swc_max, "k-", lw=2, alpha=0.6, label="true")
    ax.scatter([x_0, x_1], [0.0, swc_max], zorder=5, label="boundary anchors")

    for xs, fs, dx, df in zip(x_starts, swc_starts, delta_x, delta_swc):
        ax.plot([xs, xs + dx], [fs, fs + df], "r-", alpha=0.15, lw=0.8)

    if inferred is not None:
        label = type(inferred).__name__
        if hasattr(inferred, "predict"):
            mean, std = inferred.predict(x_curve)
            ax.plot(x_curve, mean, "g-", lw=2, label=label)
            ax.fill_between(
                x_curve,
                mean - 2 * std,
                mean + 2 * std,
                alpha=0.3,
                color="g",
                label=f"{label} ±2σ",
            )
        else:
            ax.plot(x_curve, inferred(x_curve), "b-", lw=2, label=label)
            if hasattr(inferred, "knots"):
                ax.scatter(
                    inferred.knots,
                    inferred(inferred.knots),
                    s=40,
                    color="b",
                    zorder=6,
                    label=f"{label} knots",
                )

    if swc_max_est is not None:
        ax.axhline(
            swc_max_est,
            color="g",
            lw=1.5,
            linestyle="--",
            label=f"swc_max_est={swc_max_est:.1f}",
        )
        ax.axhline(swc_max, color="k", lw=1, linestyle=":", label=f"true={swc_max}")

    ax.set_xlabel("sensor reading (x)")
    ax.set_ylabel(f"swc max={swc_max}")
    if legend:
        ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    elif not caller_owns_ax:
        plt.show()
    return fig, ax
