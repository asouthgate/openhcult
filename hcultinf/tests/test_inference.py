import numpy as np

from hcultinf.inference import fit_linear, MonotonicSpline
from tests.sim import (
    simulate_calibration,
    plot_calibration,
    invlogistic_swc_of_x,
    logistic_x_of_swc,
)


def test_noiseless_accuracy():
    """All methods recover the true curve within tight tolerance when there is no noise."""
    swc_max = 447.219
    rng = np.random.default_rng(42)
    x_0, x_1, x_starts, swc_starts, delta_x, delta_swc = simulate_calibration(
        n_chords=500, x0_noise=0.0, x1_noise=0.0, x_noise=0.0, swc_max=swc_max, rng=rng
    )
    assert all(swc_starts + delta_swc) <= swc_max
    x_anchors = np.array([x_0])
    swc_anchors = np.array([0.0])
    n_knots = 30
    ms = MonotonicSpline(n_knots, k=4).fit(
        x_anchors, swc_anchors, x_starts, delta_x, delta_swc
    )
    import matplotlib.pyplot as plt

    swc_curve = np.linspace(0.01, 0.99, 500)
    x_curve = logistic_x_of_swc(swc_curve)

    assert (
        max(swc_starts) <= swc_max
    ), f"{max(swc_starts)} should be less than {swc_max}"
    fig, ax = plot_calibration(
        x_0,
        x_1,
        x_starts,
        swc_starts,
        delta_x,
        delta_swc,
        swc_max=swc_max,
        title="noiseless accuracy",
    )
    ax.plot(x_curve, ms(x_curve), "b-", lw=2, label="monotonic spline")
    ax.scatter(ms.knots, ms(ms.knots), s=40, color="b", zorder=6, label="ms knots")
    ax.legend()
    plt.show()

    x_eval = np.linspace(x_anchors.min(), x_anchors.max(), 200)
    swc_true = invlogistic_swc_of_x(x_eval) * swc_max

    assert abs(ms(x_0) - 0.0) < 0.02
    assert np.sqrt(np.mean((ms(x_eval) - swc_true) ** 2)) < 0.015
