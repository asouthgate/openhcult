from __future__ import annotations

import numpy as np

from scipy.interpolate import BSpline
from scipy.optimize import minimize


def _make_knots(inner_knots, x_min, x_max, k):
    return np.concatenate([[x_min] * (k + 1), inner_knots, [x_max] * (k + 1)])


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


def fit_linear(x, y):
    m, c = np.polyfit(x, y, 1)
    return lambda v: m * v + c


class MonotonicSpline:
    def __init__(self, inner_knots, k=3, w_chord=1.0):
        self._inner_knots = np.asarray(inner_knots)
        self._k = k
        self._w_chord = w_chord
        self._spline = None

    def fit(self, x_anchors, fc_anchors, x_starts=None, delta_x=None, delta_fc=None):
        x, y = np.asarray(x_anchors), np.asarray(fc_anchors)
        idx = np.argsort(x)
        xs, ys = x[idx], y[idx]

        has_chords = x_starts is not None
        if has_chords:
            x_ends = x_starts + delta_x
            x_min = min(xs.min(), x_starts.min(), x_ends.min())
            x_max = max(xs.max(), x_starts.max(), x_ends.max())
        else:
            x_min, x_max = xs.min(), xs.max()

        y_dir = np.sign(ys[-1] - ys[0])
        t = _make_knots(self._inner_knots, x_min, x_max, self._k)
        n_c = len(t) - self._k - 1
        greville = np.array([t[i + 1 : i + self._k + 1].mean() for i in range(n_c)])
        if x_max > x_min:
            c0 = ys[0] + (ys[-1] - ys[0]) * (greville - x_min) / (x_max - x_min)
        else:
            c0 = np.full(n_c, ys[0])

        x_check = np.linspace(x_min, x_max, 50)
        k, w = self._k, self._w_chord

        def monotonic_con(coeffs):
            return y_dir * BSpline(t, coeffs, k)(x_check, nu=1)

        if has_chords:

            def objective(coeffs):
                spl = BSpline(t, coeffs, k)
                return np.sum((spl(x) - y) ** 2) + w * np.sum(
                    (spl(x_ends) - spl(x_starts) - delta_fc) ** 2
                )

        else:

            def objective(coeffs):
                return np.sum((BSpline(t, coeffs, k)(x) - y) ** 2)

        res = minimize(
            objective, c0, constraints={"type": "ineq", "fun": monotonic_con}
        )
        if not res.success:
            raise RuntimeError(f"Spline optimization failed: {res.message}")

        self._spline = BSpline(t, res.x, k)
        return self

    def __call__(self, x):
        return self._spline(x)

    @property
    def knots(self):
        return self._inner_knots

    def bootstrap_residuals(
        self,
        x_anchors,
        fc_anchors,
        x_starts,
        delta_x,
        delta_fc,
        n_boot=200,
        rng=None,
    ):
        """Resample anchor residuals to estimate fit sensitivity."""
        if rng is None:
            rng = np.random.default_rng()
        self.fit(x_anchors, fc_anchors, x_starts, delta_x, delta_fc)
        x_anchors, fc_anchors = np.asarray(x_anchors), np.asarray(fc_anchors)
        residuals = fc_anchors - self(x_anchors)

        boots = []
        for _ in range(n_boot):
            y_boot = self(x_anchors) + rng.choice(
                residuals, size=len(residuals), replace=True
            )
            b = MonotonicSpline(self._inner_knots, self._k, self._w_chord)
            try:
                b.fit(x_anchors, y_boot, x_starts, delta_x, delta_fc)
                boots.append(b)
            except RuntimeError:
                pass
        return boots

    def bootstrap_data(
        self,
        x_anchors,
        fc_anchors,
        x_starts,
        delta_x,
        delta_fc,
        n_boot=200,
        rng=None,
    ):
        """Resample chord observations to estimate epistemic uncertainty."""
        if rng is None:
            rng = np.random.default_rng()
        self.fit(x_anchors, fc_anchors, x_starts, delta_x, delta_fc)
        n_chords = len(x_starts)

        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n_chords, n_chords)
            b = MonotonicSpline(self._inner_knots, self._k, self._w_chord)
            try:
                b.fit(x_anchors, fc_anchors, x_starts[idx], delta_x[idx], delta_fc[idx])
                boots.append(b)
            except RuntimeError:
                pass
        return boots


class ParametricMonotonicSpline:
    def __init__(self, knots=6, k=3, w_chord=1.0, n_iter=4):
        self.knots = knots
        self._k = k
        self._w_chord = w_chord
        self._n_iter = n_iter
        self.sx = None
        self.sfc = None

    def fit(self, x_anchors, fc_anchors, x_starts, delta_x, delta_fc):
        inner = np.linspace(0, 1, self.knots + 2)[1:-1]
        k = self._k
        t = np.concatenate(([0.0] * (k + 1), inner, [1.0] * (k + 1)))
        n_c = len(t) - k - 1

        x_anchors = np.asarray(x_anchors)
        fc_anchors = np.asarray(fc_anchors)
        x_ends = x_starts + delta_x
        sort_idx = np.argsort(x_anchors)
        x_sorted, fc_sorted = x_anchors[sort_idx], fc_anchors[sort_idx]
        fc_dir = np.sign(fc_sorted[-1] - fc_sorted[0])
        if fc_dir == 0:
            fc_dir = 1

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

        w = self._w_chord
        for _ in range(self._n_iter):

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
                return err_anc + w * err_chord

            def monotonic_con(coeffs):
                return fc_dir * BSpline(t, coeffs[n_c:], k)(np.linspace(0, 1, 50), nu=1)

            constraints = [{"type": "ineq", "fun": monotonic_con}]

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

        self.sx = BSpline(t, res.x[:n_c], k)
        self.sfc = BSpline(t, res.x[n_c:], k)
        return self

    def predict(self, x, n_fine=2000):
        s_fine = np.linspace(0, 1, n_fine)
        return np.interp(x, self.sx(s_fine), self.sfc(s_fine))

    def knot_positions(self):
        inner_s = np.linspace(0, 1, self.knots + 2)[1:-1]
        return self.sx(inner_s), self.sfc(inner_s)

    def bootstrap_residuals(
        self,
        x_anchors,
        fc_anchors,
        x_starts,
        delta_x,
        delta_fc,
        n_boot=200,
        rng=None,
    ):
        """Resample anchor residuals to estimate fit sensitivity."""
        if rng is None:
            rng = np.random.default_rng()
        self.fit(x_anchors, fc_anchors, x_starts, delta_x, delta_fc)
        x_anchors, fc_anchors = np.asarray(x_anchors), np.asarray(fc_anchors)
        residuals = fc_anchors - self.predict(x_anchors)

        boots = []
        for _ in range(n_boot):
            fc_boot = self.predict(x_anchors) + rng.choice(
                residuals, size=len(residuals), replace=True
            )
            b = ParametricMonotonicSpline(
                self.knots, self._k, self._w_chord, self._n_iter
            )
            try:
                b.fit(x_anchors, fc_boot, x_starts, delta_x, delta_fc)
                boots.append(b)
            except RuntimeError:
                pass
        return boots

    def bootstrap_data(
        self,
        x_anchors,
        fc_anchors,
        x_starts,
        delta_x,
        delta_fc,
        n_boot=200,
        rng=None,
    ):
        """Resample chord observations to estimate epistemic uncertainty."""
        if rng is None:
            rng = np.random.default_rng()
        self.fit(x_anchors, fc_anchors, x_starts, delta_x, delta_fc)
        n_chords = len(x_starts)

        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n_chords, n_chords)
            b = ParametricMonotonicSpline(
                self.knots, self._k, self._w_chord, self._n_iter
            )
            try:
                b.fit(x_anchors, fc_anchors, x_starts[idx], delta_x[idx], delta_fc[idx])
                boots.append(b)
            except RuntimeError:
                pass
        return boots
