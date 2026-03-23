import numpy as np

from hcultinf.inference import (
    fit_linear,
    fit_monotonic_spline,
    fit_monotonic_spline_with_chords,
    fit_parametric_monotonic_spline,
)
from tests.sim import simulate_calibration, plot_calibration


def test_parametric_spline_fits_logistic():
    # True relationship: x(z) is logistic, so z(x) goes near-vertical at the extremes.
    # This is the hard case that motivates arc-length reparameterisation.
    z_true = np.linspace(0.05, 0.95, 60)
    x_true = 1.0 / (1.0 + np.exp(-12.0 * (z_true - 0.5)))

    x_mid = 0.5 * (x_true[:-1] + x_true[1:])
    dz_dx = np.diff(z_true) / np.diff(x_true)

    anchor_idx = np.linspace(0, len(x_true) - 1, 8).astype(int)
    x_anchors = x_true[anchor_idx]
    z_anchors = z_true[anchor_idx]

    sx, sz = fit_parametric_monotonic_spline(
        x_anchors, z_anchors, x_mid, dz_dx, knots=6, k=3, w_der=1.0
    )

    s_fine = np.linspace(0, 1, 1000)
    x_fit = sx(s_fine)
    z_fit = sz(s_fine)

    import matplotlib.pyplot as plt

    plt.plot(x_true, z_true)
    plt.plot(x_fit, z_fit)
    plt.show()

    # z must be monotonically decreasing along the curve
    assert np.all(np.diff(z_fit) <= 0.01)

    # Fitted curve must pass close to each interior anchor in (x, z) space
    for xa, za in zip(x_anchors[1:-1], z_anchors[1:-1]):
        dist = np.sqrt((x_fit - xa) ** 2 + (z_fit - za) ** 2)
        assert dist.min() < 0.05, f"Curve too far from anchor ({xa:.3f}, {za:.3f})"


def test_monotonic_spline_with_chords():
    fc_max = 45
    rng = np.random.default_rng(1)
    x_0, x_1, x_starts, fc_starts, delta_x, delta_fc = simulate_calibration(
        n_chords=80, x0_noise=0.005, x1_noise=0.005, fc_max=fc_max, rng=rng
    )

    x_anchors = np.array([x_0, x_1])
    fc_anchors = np.array([0.0, fc_max])
    inner_knots = np.percentile(x_starts, [25, 50, 75])

    spline_no_chords = fit_monotonic_spline(x_anchors, fc_anchors, inner_knots)
    spline_with_chords = fit_monotonic_spline_with_chords(
        x_anchors, fc_anchors, inner_knots, x_starts, delta_x, delta_fc, w_chord=1.0
    )
    spline_x0_only = fit_monotonic_spline_with_chords(
        np.array([x_0]),
        np.array([0.0]),
        inner_knots,
        x_starts,
        delta_x,
        delta_fc,
        w_chord=1.0,
    )

    import matplotlib.pyplot as plt
    from tests.sim import _x_of_fc, _fc_of_x

    fc_curve = np.linspace(0.01, 0.99, 500)
    x_curve = _x_of_fc(fc_curve)

    linear = fit_linear(x_anchors, fc_anchors)

    fig, ax = plot_calibration(
        x_0,
        x_1,
        x_starts,
        fc_starts,
        delta_x,
        delta_fc,
        fc_max=fc_max,
        title="with vs without chords",
    )
    ax.plot(x_curve, linear(x_curve), "m--", lw=1.5, label="linear")
    ax.plot(x_curve, spline_no_chords(x_curve), "g--", lw=1.5, label="no chords")
    ax.plot(x_curve, spline_with_chords(x_curve), "b-", lw=2, label="with chords")
    ax.plot(x_curve, spline_x0_only(x_curve), "c-", lw=1.5, label="x0 only + chords")
    ax.legend()
    plt.show()

    x_eval = np.linspace(x_anchors.min(), x_anchors.max(), 200)
    fc_true = _fc_of_x(x_eval) * fc_max

    err_no_chords = np.mean((spline_no_chords(x_eval) - fc_true) ** 2)
    err_with_chords = np.mean((spline_with_chords(x_eval) - fc_true) ** 2)
    assert err_with_chords <= err_no_chords, "chord data should improve fit"


def test_parametric_spline_sparse_anchors_dense_derivatives(n_anchors=4, n_der=100):
    # Few noisy anchors, many derivative points — derivative data should carry the shape.
    rng = np.random.default_rng(0)

    z_true = np.linspace(0.05, 0.95, 200)
    x_true = 1.0 / (1.0 + np.exp(-12.0 * (z_true - 0.5)))

    anchor_idx = np.linspace(0, len(x_true) - 1, n_anchors).astype(int)
    x_anchors = x_true[anchor_idx] + rng.normal(0, 0.10, n_anchors)
    z_anchors = z_true[anchor_idx] + rng.normal(0, 0.01, n_anchors)

    der_idx = np.linspace(0, len(x_true) - 2, n_der).astype(int)
    x_mid = 0.5 * (x_true[:-1] + x_true[1:])[der_idx]
    dz_dx = (np.diff(z_true) / np.diff(x_true))[der_idx] + rng.normal(0, 0.0, n_der)

    sx, sz = fit_parametric_monotonic_spline(
        x_anchors, z_anchors, x_mid, dz_dx, knots=6, k=3, w_der=1.0
    )

    s_fine = np.linspace(0, 1, 1000)
    x_fit = sx(s_fine)
    z_fit = sz(s_fine)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x_true, z_true, label="true")
    plt.scatter(x_anchors, z_anchors, label="anchors", zorder=5)
    plt.plot(x_fit, z_fit, label="fit")
    plt.legend()
    plt.title(f"{n_anchors} anchors, {n_der} derivative points")
    plt.show()

    assert np.all(np.diff(z_fit) <= 0.01)
    assert x_fit.min() <= x_anchors.min() + 0.02, "curve doesn't reach left anchor"
    assert x_fit.max() >= x_anchors.max() - 0.02, "curve doesn't reach right anchor"
