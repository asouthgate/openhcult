from __future__ import annotations

import numpy as np

from scipy.interpolate import BSpline
from scipy.optimize import minimize


def _make_knots(inner_knots, x_min, x_max, k):
    return np.concatenate([[x_min] * (k + 1), inner_knots, [x_max] * (k + 1)])


def fit_linear(x, y):
    m, c = np.polyfit(x, y, 1)
    return lambda v: m * v + c


def fit_monotonic_spline(x, y, inner_knots, k=3):
    """Regress y against x with specified inner_knots and spline order k."""
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    x_min, x_max = xs.min(), xs.max()

    y_dir = np.sign(ys[-1] - ys[0])
    t = _make_knots(inner_knots, x_min, x_max, k)
    c0 = np.zeros(len(t) - k - 1)

    x_check = np.linspace(x_min, x_max, 50)

    def objective(coeffs):
        return np.sum((BSpline(t, coeffs, k)(x) - y) ** 2)

    def monotonic_constraint(coeffs):
        return y_dir * BSpline(t, coeffs, k)(x_check, nu=1)

    res = minimize(
        objective, c0, constraints={"type": "ineq", "fun": monotonic_constraint}
    )
    if not res.success:
        raise RuntimeError(f"Spline optimization failed: {res.message}")

    return BSpline(t, res.x, k)


def fit_monotonic_spline_with_chords(
    x, y, inner_knots, x_starts, delta_x, delta_z, k=3, w_chord=1.0
):
    """Like fit_monotonic_spline but also fits chord observations.

    Each chord constrains: spline(x_start + delta_x) - spline(x_start) = delta_z.
    Monotone direction is inferred from the data.
    """
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    x_ends = x_starts + delta_x
    x_min = min(xs.min(), x_starts.min(), x_ends.min())
    x_max = max(xs.max(), x_starts.max(), x_ends.max())
    y_dir = np.sign(ys[-1] - ys[0])

    t = _make_knots(inner_knots, x_min, x_max, k)
    n_c = len(t) - k - 1
    greville = np.array([t[i + 1 : i + k + 1].mean() for i in range(n_c)])
    if x_max > x_min:
        c0 = ys[0] + (ys[-1] - ys[0]) * (greville - x_min) / (x_max - x_min)
    else:
        c0 = np.full(n_c, ys[0])

    x_check = np.linspace(x_min, x_max, 50)

    def objective(coeffs):
        spl = BSpline(t, coeffs, k)
        err_fit = np.sum((spl(x) - y) ** 2)
        err_chord = np.sum((spl(x_ends) - spl(x_starts) - delta_z) ** 2)
        return err_fit + w_chord * err_chord

    def monotonic_constraint(coeffs):
        return y_dir * BSpline(t, coeffs, k)(x_check, nu=1)

    res = minimize(
        objective, c0, constraints={"type": "ineq", "fun": monotonic_constraint}
    )
    if not res.success:
        raise RuntimeError(f"Spline optimization failed: {res.message}")

    return BSpline(t, res.x, k)


def _arclen_reparameterise_chords(
    sx, sfc, x_anchors, fc_anchors, x_starts, x_ends, n_fine=500
):
    s_fine = np.linspace(0, 1, n_fine)
    x_fine = sx(s_fine)
    fc_fine = sfc(s_fine)

    x_range = np.ptp(x_fine) or 1.0
    fc_range = np.ptp(fc_fine) or 1.0
    xn = x_fine / x_range
    fcn = fc_fine / fc_range

    seg_len = np.sqrt(np.diff(xn) ** 2 + np.diff(fcn) ** 2)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    arc /= arc[-1]

    s_anc = np.array(
        [
            arc[np.argmin((xn - xa / x_range) ** 2 + (fcn - fa / fc_range) ** 2)]
            for xa, fa in zip(x_anchors, fc_anchors)
        ]
    )
    s_starts = np.array([arc[np.argmin((xn - xs / x_range) ** 2)] for xs in x_starts])
    s_ends = np.array([arc[np.argmin((xn - xe / x_range) ** 2)] for xe in x_ends])

    return s_anc, s_starts, s_ends


def fit_parametric_monotonic_spline_with_chords(
    x_anchors,
    fc_anchors,
    x_starts,
    delta_x,
    delta_fc,
    knots=6,
    k=3,
    w_chord=1.0,
    n_iter=4,
):
    """Parametric monotonic spline fit using chord observations.

    Fits (sx(s), sfc(s)) jointly, iteratively updating the arc-length parameterisation.
    Chord constraints: sfc(s_end) - sfc(s_start) = delta_fc, with x alignment enforced
    via (sx(s_start) - x_start)² and (sx(s_end) - x_end)² terms.
    """
    inner = np.linspace(0, 1, knots + 2)[1:-1]
    t = np.concatenate(([0.0] * (k + 1), inner, [1.0] * (k + 1)))
    n_c = len(t) - k - 1

    x_ends = x_starts + delta_x
    sort_idx = np.argsort(x_anchors)
    x_sorted, fc_sorted = x_anchors[sort_idx], fc_anchors[sort_idx]
    fc_dir = np.sign(fc_sorted[-1] - fc_sorted[0])

    x_min = min(x_sorted.min(), x_starts.min(), x_ends.min())
    x_max = max(x_sorted.max(), x_starts.max(), x_ends.max())

    s_anc = (x_sorted - x_min) / (x_max - x_min)
    s_starts = (x_starts - x_min) / (x_max - x_min)
    s_ends = (x_ends - x_min) / (x_max - x_min)

    s_init = np.linspace(0, 1, n_c)
    c0 = np.concatenate(
        [
            np.interp(s_init, np.linspace(0, 1, len(x_sorted)), x_sorted),
            np.interp(s_init, np.linspace(0, 1, len(fc_sorted)), fc_sorted),
        ]
    )

    for _ in range(n_iter):

        def objective(coeffs, s_anc=s_anc, s_starts=s_starts, s_ends=s_ends):
            cx, cfc = coeffs[:n_c], coeffs[n_c:]
            sx = BSpline(t, cx, k)
            sfc = BSpline(t, cfc, k)
            err_anc = np.sum(
                (sx(s_anc) - x_sorted) ** 2 + (sfc(s_anc) - fc_sorted) ** 2
            )
            err_chord = np.sum(
                (sx(s_starts) - x_starts) ** 2
                + (sx(s_ends) - x_ends) ** 2
                + (sfc(s_ends) - sfc(s_starts) - delta_fc) ** 2
            )
            return err_anc + w_chord * err_chord

        def monotonic_con(coeffs):
            return fc_dir * BSpline(t, coeffs[n_c:], k)(np.linspace(0, 1, 50), nu=1)

        constraints = [{"type": "ineq", "fun": monotonic_con}]
        # Only pin an endpoint if we have an anchor there — with a single anchor
        # x_sorted[0] == x_sorted[-1], which would force both ends to the same
        # point and cause the curve to loop back on itself.
        if len(x_sorted) > 1:
            constraints += [
                {"type": "eq", "fun": lambda c, v=x_sorted[-1]: c[n_c - 1] - v},
                {"type": "eq", "fun": lambda c, v=fc_sorted[-1]: c[-1] - v},
            ]
        constraints += [
            {"type": "eq", "fun": lambda c, v=x_sorted[0]: c[0] - v},
            {"type": "eq", "fun": lambda c, v=fc_sorted[0]: c[n_c] - v},
        ]

        res = minimize(objective, c0, constraints=constraints)
        if not res.success:
            raise RuntimeError(f"Spline optimization failed: {res.message}")

        c0 = res.x
        s_anc, s_starts, s_ends = _arclen_reparameterise_chords(
            BSpline(t, res.x[:n_c], k),
            BSpline(t, res.x[n_c:], k),
            x_sorted,
            fc_sorted,
            x_starts,
            x_ends,
        )

    return BSpline(t, res.x[:n_c], k), BSpline(t, res.x[n_c:], k)
