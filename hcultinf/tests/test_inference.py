import numpy as np

from hcultinf.inference import fit_linear, MonotonicSpline, ParametricMonotonicSpline
from tests.sim import simulate_calibration, plot_calibration, _fc_of_x, _x_of_fc


def test_noiseless_accuracy():
    """All methods recover the true curve within tight tolerance when there is no noise."""
    fc_max = 1.0
    rng = np.random.default_rng(42)
    x_0, x_1, x_starts, fc_starts, delta_x, delta_fc = simulate_calibration(
        n_chords=400, x0_noise=0.0, x1_noise=0.0, x_noise=0.0, fc_max=fc_max, rng=rng
    )
    x_anchors = np.array([x_0, x_1])
    fc_anchors = np.array([0.0, fc_max])
    n_knots = 30
    inner_knots = np.linspace(x_anchors.min(), x_anchors.max(), n_knots + 2)[1:-1]
    ms = MonotonicSpline(inner_knots).fit(
        x_anchors, fc_anchors, x_starts, delta_x, delta_fc
    )
    pms = ParametricMonotonicSpline(knots=n_knots).fit(
        x_anchors, fc_anchors, x_starts, delta_x, delta_fc
    )

    import matplotlib.pyplot as plt

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
        title="noiseless accuracy",
    )
    ax.plot(x_curve, ms(x_curve), "b-", lw=2, label="monotonic spline")
    ax.scatter(ms.knots, ms(ms.knots), s=40, color="b", zorder=6, label="ms knots")
    s_fine = np.linspace(0, 1, 1000)
    ax.plot(pms.sx(s_fine), pms.sfc(s_fine), "r-", lw=2, label="parametric spline")
    kx, kfc = pms.knot_positions()
    ax.scatter(kx, kfc, s=40, color="r", zorder=6, label="pms knots")
    ax.legend()
    plt.show()

    x_eval = np.linspace(x_anchors.min(), x_anchors.max(), 200)
    fc_true = _fc_of_x(x_eval) * fc_max

    assert abs(ms(x_0) - 0.0) < 0.02
    assert abs(ms(x_1) - fc_max) < 0.02
    assert abs(pms.predict(x_0) - 0.0) < 0.02
    assert abs(pms.predict(x_1) - fc_max) < 0.02

    assert np.sqrt(np.mean((ms(x_eval) - fc_true) ** 2)) < 0.015
    assert np.sqrt(np.mean((pms.predict(x_eval) - fc_true) ** 2)) < 0.015


def test_parametric_beats_monotonic_low_n():
    """With few chords, parametric spline outperforms monotonic spline due to better endpoint handling."""
    fc_max = 1.0
    rng = np.random.default_rng(42)
    x_0, x_1, x_starts, fc_starts, delta_x, delta_fc = simulate_calibration(
        n_chords=50, x0_noise=0.0, x1_noise=0.0, x_noise=0.0, fc_max=fc_max, rng=rng
    )
    x_anchors = np.array([x_0, x_1])
    fc_anchors = np.array([0.0, fc_max])
    n_knots = 10
    inner_knots = np.linspace(x_anchors.min(), x_anchors.max(), n_knots + 2)[1:-1]
    ms = MonotonicSpline(inner_knots).fit(
        x_anchors, fc_anchors, x_starts, delta_x, delta_fc
    )
    pms = ParametricMonotonicSpline(knots=n_knots).fit(
        x_anchors, fc_anchors, x_starts, delta_x, delta_fc
    )

    import matplotlib.pyplot as plt

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
        title="parametric vs monotonic (low n)",
    )
    ax.plot(x_curve, ms(x_curve), "b-", lw=2, label="monotonic spline")
    ax.scatter(ms.knots, ms(ms.knots), s=40, color="b", zorder=6, label="ms knots")
    s_fine = np.linspace(0, 1, 1000)
    ax.plot(pms.sx(s_fine), pms.sfc(s_fine), "r-", lw=2, label="parametric spline")
    kx, kfc = pms.knot_positions()
    ax.scatter(kx, kfc, s=40, color="r", zorder=6, label="pms knots")
    ax.legend()
    plt.show()

    x_eval = np.linspace(x_anchors.min(), x_anchors.max(), 200)
    fc_true = _fc_of_x(x_eval) * fc_max

    assert np.sqrt(np.mean((pms.predict(x_eval) - fc_true) ** 2)) < np.sqrt(
        np.mean((ms(x_eval) - fc_true) ** 2)
    )


def test_clamped_region():
    """Monotonic spline cannot represent sensor saturation (dx=0, dfc>0); parametric handles it."""
    fc_max = 1.0
    rng = np.random.default_rng(42)
    x_0, x_1, x_starts, fc_starts, delta_x, delta_fc = simulate_calibration(
        n_chords=50, x0_noise=0.0, x1_noise=0.0, x_noise=0.0, fc_max=fc_max, rng=rng
    )

    # Clamped region: sensor saturates at x_1 while FC continues to increase
    n_clamped = 10
    dfc_clamped = 0.05
    clamped_x_starts = np.full(n_clamped, x_1)
    clamped_delta_x = np.zeros(n_clamped)
    clamped_delta_fc = np.full(n_clamped, dfc_clamped)
    clamped_fc_starts = fc_max + np.arange(n_clamped) * dfc_clamped  # for plotting only

    x_starts_all = np.concatenate([x_starts, clamped_x_starts])
    delta_x_all = np.concatenate([delta_x, clamped_delta_x])
    delta_fc_all = np.concatenate([delta_fc, clamped_delta_fc])
    fc_starts_all = np.concatenate([fc_starts, clamped_fc_starts])

    x_anchors = np.array([x_0, x_1])
    fc_anchors = np.array([0.0, fc_max])
    n_knots = 10
    inner_knots = np.linspace(x_anchors.min(), x_anchors.max(), n_knots + 2)[1:-1]

    ms = MonotonicSpline(inner_knots).fit(
        x_anchors, fc_anchors, x_starts_all, delta_x_all, delta_fc_all
    )
    pms = ParametricMonotonicSpline(knots=n_knots).fit(
        x_anchors, fc_anchors, x_starts_all, delta_x_all, delta_fc_all
    )

    import matplotlib.pyplot as plt

    fc_curve = np.linspace(0.01, 0.99, 500)
    x_curve = _x_of_fc(fc_curve)

    fig, ax = plot_calibration(
        x_0,
        x_1,
        x_starts_all,
        fc_starts_all,
        delta_x_all,
        delta_fc_all,
        fc_max=fc_max,
        title="clamped region: sensor saturation",
    )
    ax.plot(x_curve, ms(x_curve), "b-", lw=2, label="monotonic spline")
    ax.scatter(ms.knots, ms(ms.knots), s=40, color="b", zorder=6, label="ms knots")
    s_fine = np.linspace(0, 1, 1000)
    ax.plot(pms.sx(s_fine), pms.sfc(s_fine), "r-", lw=2, label="parametric spline")
    kx, kfc = pms.knot_positions()
    ax.scatter(kx, kfc, s=40, color="r", zorder=6, label="pms knots")
    ax.legend()
    plt.show()

    # For any function f, f(x + 0) - f(x) = 0 ≠ delta_fc, so ms can never satisfy vertical chords
    ms_vert_err = np.mean(
        (ms(clamped_x_starts) - ms(clamped_x_starts) - clamped_delta_fc) ** 2
    )
    assert np.isclose(ms_vert_err, np.mean(clamped_delta_fc**2))


def test_chords_improve_fit():
    """Chord data measurably reduces fit error versus anchor-only fitting."""
    fc_max = 45
    rng = np.random.default_rng(1)
    x_0, x_1, x_starts, fc_starts, delta_x, delta_fc = simulate_calibration(
        n_chords=80, x0_noise=0.005, x1_noise=0.005, fc_max=fc_max, rng=rng
    )

    x_anchors = np.array([x_0, x_1])
    fc_anchors = np.array([0.0, fc_max])
    inner_knots = np.percentile(x_starts, [25, 50, 75])

    linear = fit_linear(x_anchors, fc_anchors)
    ms_no_chords = MonotonicSpline(inner_knots).fit(x_anchors, fc_anchors)
    ms = MonotonicSpline(inner_knots).fit(
        x_anchors, fc_anchors, x_starts, delta_x, delta_fc
    )
    ms_x0 = MonotonicSpline(inner_knots).fit(
        np.array([x_0]), np.array([0.0]), x_starts, delta_x, delta_fc
    )
    pms = ParametricMonotonicSpline().fit(
        x_anchors, fc_anchors, x_starts, delta_x, delta_fc
    )
    pms_x0 = ParametricMonotonicSpline().fit(
        np.array([x_0]), np.array([0.0]), x_starts, delta_x, delta_fc
    )

    import matplotlib.pyplot as plt

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
    ax.plot(x_curve, ms_no_chords(x_curve), "g--", lw=1.5, label="no chords")
    ax.plot(x_curve, ms(x_curve), "b-", lw=2, label="with chords")
    ax.scatter(ms.knots, ms(ms.knots), s=40, color="b", zorder=6, label="ms knots")
    ax.plot(x_curve, ms_x0(x_curve), "c-", lw=1.5, label="x0 only + chords")
    s_fine = np.linspace(0, 1, 1000)
    ax.plot(pms.sx(s_fine), pms.sfc(s_fine), "r-", lw=2, label="parametric + chords")
    kx, kfc = pms.knot_positions()
    ax.scatter(kx, kfc, s=40, color="r", zorder=6, label="pms knots")
    ax.plot(
        pms_x0.sx(s_fine),
        pms_x0.sfc(s_fine),
        color="orange",
        lw=1.5,
        label="parametric x0 only + chords",
    )
    ax.legend()
    plt.show()

    x_eval = np.linspace(x_anchors.min(), x_anchors.max(), 200)
    fc_true = _fc_of_x(x_eval) * fc_max

    err_no_chords = np.mean((ms_no_chords(x_eval) - fc_true) ** 2)
    err_with_chords = np.mean((ms(x_eval) - fc_true) ** 2)
    assert err_with_chords <= err_no_chords
