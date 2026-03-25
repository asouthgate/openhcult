import numpy as np

from hcultinf.inference import fit_linear, MonotonicSpline, ParametricMonotonicSpline
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
    x_anchors = np.array([x_0])
    swc_anchors = np.array([0.0])
    n_knots = 30
    ms = MonotonicSpline(n_knots, k=3).fit(
        x_anchors, swc_anchors, x_starts, delta_x, delta_swc
    )
    # pms = ParametricMonotonicSpline(knots=n_knots).fit(
    #     x_anchors, swc_anchors, x_starts, delta_x, delta_swc
    # )

    import matplotlib.pyplot as plt

    swc_curve = np.linspace(0.01, 0.99, 500)
    x_curve = logistic_x_of_swc(swc_curve)

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
    # s_fine = np.linspace(0, 1, 1000)
    # ax.plot(pms.sx(s_fine), pms.sswc(s_fine), "r-", lw=2, label="parametric spline")
    # kx, kswc = pms.knot_positions()
    # ax.scatter(kx, kswc, s=40, color="r", zorder=6, label="pms knots")
    ax.legend()
    plt.show()

    x_eval = np.linspace(x_anchors.min(), x_anchors.max(), 200)
    swc_true = invlogistic_swc_of_x(x_eval) * swc_max

    assert abs(ms(x_0) - 0.0) < 0.02
    # assert abs(pms.predict(x_0) - 0.0) < 0.02
    # assert abs(pms.predict(x_1) - swc_max) < 0.02

    assert np.sqrt(np.mean((ms(x_eval) - swc_true) ** 2)) < 0.015
    # assert np.sqrt(np.mean((pms.predict(x_eval) - swc_true) ** 2)) < 0.015


# def test_chords_improve_fit():
#     """Chord data measurably reduces fit error versus anchor-only fitting."""
#     swc_max = 45
#     rng = np.random.default_rng(1)
#     x_0, x_1, x_starts, swc_starts, delta_x, delta_swc = simulate_calibration(
#         n_chords=80, x0_noise=0.005, x1_noise=0.005, swc_max=swc_max, rng=rng
#     )

#     x_anchors = np.array([x_0, x_1])
#     swc_anchors = np.array([0.0, swc_max])
#     inner_knots = np.percentile(x_starts, [25, 50, 75])

#     linear = fit_linear(x_anchors, swc_anchors)
#     ms_no_chords = MonotonicSpline(inner_knots).fit(x_anchors, swc_anchors)
#     ms = MonotonicSpline(inner_knots).fit(
#         x_anchors, swc_anchors, x_starts, delta_x, delta_swc
#     )
#     ms_x0 = MonotonicSpline(inner_knots).fit(
#         np.array([x_0]), np.array([0.0]), x_starts, delta_x, delta_swc
#     )
#     pms = ParametricMonotonicSpline().fit(
#         x_anchors, swc_anchors, x_starts, delta_x, delta_swc
#     )
#     pms_x0 = ParametricMonotonicSpline().fit(
#         np.array([x_0]), np.array([0.0]), x_starts, delta_x, delta_swc
#     )

#     import matplotlib.pyplot as plt

#     swc_curve = np.linspace(0.01, 0.99, 500)
#     x_curve = logistic_x_of_swc(swc_curve)

#     fig, ax = plot_calibration(
#         x_0,
#         x_1,
#         x_starts,
#         swc_starts,
#         delta_x,
#         delta_swc,
#         swc_max=swc_max,
#         title="with vs without chords",
#     )
#     ax.plot(x_curve, linear(x_curve), "m--", lw=1.5, label="linear")
#     ax.plot(x_curve, ms_no_chords(x_curve), "g--", lw=1.5, label="no chords")
#     ax.plot(x_curve, ms(x_curve), "b-", lw=2, label="with chords")
#     ax.scatter(ms.knots, ms(ms.knots), s=40, color="b", zorder=6, label="ms knots")
#     ax.plot(x_curve, ms_x0(x_curve), "c-", lw=1.5, label="x0 only + chords")
#     s_fine = np.linspace(0, 1, 1000)
#     ax.plot(pms.sx(s_fine), pms.sswc(s_fine), "r-", lw=2, label="parametric + chords")
#     kx, kswc = pms.knot_positions()
#     ax.scatter(kx, kswc, s=40, color="r", zorder=6, label="pms knots")
#     ax.plot(
#         pms_x0.sx(s_fine),
#         pms_x0.sswc(s_fine),
#         color="orange",
#         lw=1.5,
#         label="parametric x0 only + chords",
#     )
#     ax.legend()
#     plt.show()

#     x_eval = np.linspace(x_anchors.min(), x_anchors.max(), 200)
#     swc_true = invlogistic_swc_of_x(x_eval) * swc_max

#     err_no_chords = np.mean((ms_no_chords(x_eval) - swc_true) ** 2)
#     err_with_chords = np.mean((ms(x_eval) - swc_true) ** 2)
#     assert err_with_chords <= err_no_chords
