import numpy as np

from hcultinf.inference import (
    fit_linear,
    fit_monotonic_spline,
    fit_monotonic_spline_with_chords,
    fit_parametric_monotonic_spline_with_chords,
)
from tests.sim import simulate_calibration, plot_calibration


def test_monotonic_spline_with_chords():
    fc_max = 45
    rng = np.random.default_rng(1)
    x_0, x_1, x_starts, fc_starts, delta_x, delta_fc = simulate_calibration(
        n_chords=80, x0_noise=0.005, x1_noise=0.005, fc_max=fc_max, rng=rng
    )

    x_anchors = np.array([x_0, x_1])
    fc_anchors = np.array([0.0, fc_max])
    inner_knots = np.percentile(x_starts, [25, 50, 75])

    linear = fit_linear(x_anchors, fc_anchors)
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

    sx_par, sfc_par = fit_parametric_monotonic_spline_with_chords(
        x_anchors, fc_anchors, x_starts, delta_x, delta_fc, w_chord=1.0
    )
    sx_par_x0, sfc_par_x0 = fit_parametric_monotonic_spline_with_chords(
        np.array([x_0]), np.array([0.0]), x_starts, delta_x, delta_fc, w_chord=1.0
    )

    import matplotlib.pyplot as plt
    from tests.sim import _x_of_fc, _fc_of_x

    fc_curve = np.linspace(0.01, 0.99, 500)
    x_curve = _x_of_fc(fc_curve)

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
    s_fine = np.linspace(0, 1, 1000)
    ax.plot(sx_par(s_fine), sfc_par(s_fine), "r-", lw=2, label="parametric + chords")
    ax.plot(
        sx_par_x0(s_fine),
        sfc_par_x0(s_fine),
        "orange",
        lw=1.5,
        label="parametric x0 only + chords",
    )
    ax.legend()
    plt.show()

    x_eval = np.linspace(x_anchors.min(), x_anchors.max(), 200)
    fc_true = _fc_of_x(x_eval) * fc_max

    err_no_chords = np.mean((spline_no_chords(x_eval) - fc_true) ** 2)
    err_with_chords = np.mean((spline_with_chords(x_eval) - fc_true) ** 2)
    assert err_with_chords <= err_no_chords, "chord data should improve fit"
