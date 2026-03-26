import numpy as np

from hcultinf.inference import (
    fit_linear,
    MonotonicSpline,
    MonotonicPWL,
    GP,
    estimate_swc_max,
)
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


def test_noiseless_accuracy_gp():
    """GP recovers the true curve within tight tolerance when there is no noise."""
    swc_max = 447.219
    rng = np.random.default_rng(42)
    x_0, x_1, x_starts, swc_starts, delta_x, delta_swc = simulate_calibration(
        n_chords=500, x0_noise=0.0, x1_noise=0.0, x_noise=0.0, swc_max=swc_max, rng=rng
    )
    assert all(swc_starts + delta_swc) <= swc_max
    x_anchors = np.array([x_0, x_1])
    swc_anchors = np.array([0.0, swc_max])
    gp = GP().fit(x_anchors, swc_anchors, x_starts, delta_x, delta_swc)

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
        title="noiseless accuracy (GP)",
    )
    slope = np.mean(delta_swc) / np.mean(delta_x)
    linear_est = slope * (x_curve - x_0)

    mean, std = gp.predict(x_curve)
    ax.plot(x_curve, linear_est, "m--", lw=1.5, label="linear estimate")
    ax.plot(x_curve, mean, "g-", lw=2, label="GP")
    ax.fill_between(
        x_curve, mean - 2 * std, mean + 2 * std, alpha=0.3, color="g", label="GP ±2σ"
    )
    ax.legend()
    plt.show()

    x_eval = np.linspace(x_anchors.min(), x_anchors.max(), 200)
    swc_true = invlogistic_swc_of_x(x_eval) * swc_max

    assert abs(gp(x_0) - 0.0) < 0.02
    assert np.sqrt(np.mean((gp(x_eval) - swc_true) ** 2)) < 5.0


def test_noiseless_accuracy_pwl():
    """MonotonicPWL recovers the true curve within tolerance when there is no noise."""
    import matplotlib.pyplot as plt

    swc_max = 447.219
    rng = np.random.default_rng(42)
    x_0, x_1, x_starts, swc_starts, delta_x, delta_swc = simulate_calibration(
        n_chords=500, x0_noise=0.0, x1_noise=0.0, x_noise=0.0, swc_max=swc_max, rng=rng
    )
    x_anchors = np.array([x_0, x_1])
    swc_anchors = np.array([0.0, swc_max])
    pwl = MonotonicPWL(n_nodes=100).fit(
        x_anchors, swc_anchors, x_starts, delta_x, delta_swc
    )

    swc_curve = np.linspace(0.01, 0.99, 500)
    x_curve = logistic_x_of_swc(swc_curve)
    swc_true = invlogistic_swc_of_x(x_curve) * swc_max

    fig, ax = plot_calibration(
        x_0,
        x_1,
        x_starts,
        swc_starts,
        delta_x,
        delta_swc,
        swc_max=swc_max,
        title="noiseless accuracy (MonotonicPWL)",
    )
    ax.plot(x_curve, pwl(x_curve), "b-", lw=2, label="MonotonicPWL")
    ax.legend()
    plt.show()

    assert abs(pwl(x_0) - 0.0) < 0.5
    assert np.sqrt(np.mean((pwl(x_curve) - swc_true) ** 2)) < 10.0


def test_estimate_swc_max_good_prior_partial_coverage():
    """Good prior + 50% x-coverage: estimate is accurate despite partial data."""
    import matplotlib.pyplot as plt

    swc_max = 447.219
    rng = np.random.default_rng(42)
    x_0, x_1, x_starts, swc_starts, delta_x, delta_swc = simulate_calibration(
        n_chords=500, x0_noise=0.0, x1_noise=0.0, x_noise=0.0, swc_max=swc_max, rng=rng
    )
    mask = (x_starts + delta_x) <= 0.5
    x_starts, delta_x, delta_swc, swc_starts = (
        x_starts[mask],
        delta_x[mask],
        delta_swc[mask],
        swc_starts[mask],
    )

    swc_max_est = estimate_swc_max(x_starts, delta_x, delta_swc, invlogistic_swc_of_x)

    fig, ax = plot_calibration(
        x_0,
        x_1,
        x_starts,
        swc_starts,
        delta_x,
        delta_swc,
        swc_max=swc_max,
        title=f"good prior, 50% coverage — est={swc_max_est:.1f} true={swc_max}",
    )
    ax.axhline(
        swc_max_est,
        color="g",
        lw=1.5,
        linestyle="--",
        label=f"swc_max_est={swc_max_est:.1f}",
    )
    ax.axhline(swc_max, color="k", lw=1, linestyle=":", label=f"true={swc_max}")
    ax.legend()
    plt.show()

    assert abs(swc_max_est - swc_max) < 30.0


def test_estimate_swc_max_bad_prior_full_coverage():
    """Bad prior (linear) + 100% coverage: estimate is data-driven, prior shape irrelevant."""
    import matplotlib.pyplot as plt

    swc_max = 447.219
    rng = np.random.default_rng(42)
    x_0, x_1, x_starts, swc_starts, delta_x, delta_swc = simulate_calibration(
        n_chords=500, x0_noise=0.0, x1_noise=0.0, x_noise=0.0, swc_max=swc_max, rng=rng
    )
    linear_prior = lambda x: (x - x_0) / (x_1 - x_0)
    swc_max_est = estimate_swc_max(x_starts, delta_x, delta_swc, linear_prior)

    fig, ax = plot_calibration(
        x_0,
        x_1,
        x_starts,
        swc_starts,
        delta_x,
        delta_swc,
        swc_max=swc_max,
        title=f"bad prior, 100% coverage — est={swc_max_est:.1f} true={swc_max}",
    )
    ax.axhline(
        swc_max_est,
        color="r",
        lw=1.5,
        linestyle="--",
        label=f"swc_max_est={swc_max_est:.1f}",
    )
    ax.axhline(swc_max, color="k", lw=1, linestyle=":", label=f"true={swc_max}")
    ax.legend()
    plt.show()

    assert abs(swc_max_est - swc_max) < 20.0


def test_estimate_swc_max_bad_prior_partial_coverage():
    """Bad prior + partial coverage: estimate fails — both conditions must not hold simultaneously."""
    import matplotlib.pyplot as plt

    swc_max = 447.219
    rng = np.random.default_rng(42)
    x_0, x_1, x_starts, swc_starts, delta_x, delta_swc = simulate_calibration(
        n_chords=500, x0_noise=0.0, x1_noise=0.0, x_noise=0.0, swc_max=swc_max, rng=rng
    )
    mask = (x_starts + delta_x) <= 0.3
    x_starts, delta_x, delta_swc, swc_starts = (
        x_starts[mask],
        delta_x[mask],
        delta_swc[mask],
        swc_starts[mask],
    )
    linear_prior = lambda x: (x - x_0) / (x_1 - x_0)
    swc_max_est = estimate_swc_max(x_starts, delta_x, delta_swc, linear_prior)

    fig, ax = plot_calibration(
        x_0,
        x_1,
        x_starts,
        swc_starts,
        delta_x,
        delta_swc,
        swc_max=swc_max,
        title=f"bad prior, 30% coverage — est={swc_max_est:.1f} true={swc_max}",
    )
    ax.axhline(
        swc_max_est,
        color="r",
        lw=1.5,
        linestyle="--",
        label=f"swc_max_est={swc_max_est:.1f}",
    )
    ax.axhline(swc_max, color="k", lw=1, linestyle=":", label=f"true={swc_max}")
    ax.legend()
    plt.show()

    assert abs(swc_max_est - swc_max) > 100.0
