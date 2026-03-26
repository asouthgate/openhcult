from __future__ import annotations

import numpy as np

from scipy.interpolate import BSpline
from scipy.optimize import minimize


def _make_knots(inner_knots, x_min, x_max, k):
    return np.concatenate([[x_min] * (k + 1), inner_knots, [x_max] * (k + 1)])


def fit_linear(x, y):
    m, c = np.polyfit(x, y, 1)
    return lambda v: m * v + c


class MonotonicSpline:
    def __init__(self, n_knots, k=3, w_chord=1.0):
        self._n_knots = n_knots
        self._inner_knots = None
        self._k = k
        self._w_chord = w_chord
        self._spline = None

    def fit(
        self,
        x_anchors,
        swc_anchors,
        x_starts=None,
        delta_x=None,
        delta_swc=None,
        monotonic_dir=1,
    ):
        x_min = min(list(x_anchors) + list(x_starts))
        x_ends = x_starts + delta_x
        x_max = max(list(x_anchors) + list(x_ends))
        self._inner_knots = np.linspace(x_anchors.min(), x_max, self._n_knots + 2)[1:-1]

        t = _make_knots(self._inner_knots, x_min, x_max, self._k)
        n_c = len(t) - self._k - 1
        c0 = np.full(n_c, swc_anchors[0])

        def monotonic_con(coeffs):
            # Constraint the difference between adjacent coefficients
            # For monotonic, the diffs should be positive (or negative)
            return monotonic_dir * np.diff(coeffs)

        def objective(coeffs):
            spl = BSpline(t, coeffs, self._k)
            return np.sum((spl(x_anchors) - swc_anchors) ** 2) + self._w_chord * np.sum(
                (spl(x_ends) - spl(x_starts) - delta_swc) ** 2
            )

        res = minimize(
            objective, c0, constraints={"type": "ineq", "fun": monotonic_con}
        )

        self._spline = BSpline(t, res.x, self._k)
        return self

    def __call__(self, x):
        return self._spline(x)

    @property
    def knots(self):
        return self._inner_knots
