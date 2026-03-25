from __future__ import annotations

import numpy as np

from scipy.interpolate import BSpline
from scipy.optimize import minimize


def _make_knots(inner_knots, x_min, x_max, k):
    return np.concatenate([[x_min] * (k + 1), inner_knots, [x_max] * (k + 1)])


def _arclen_reparameterise_chords(
    sx, sswc, x_anchors, swc_anchors, x_starts, x_ends, n_fine=500
):
    s_fine = np.linspace(0, 1, n_fine)
    x_fine = sx(s_fine)
    swc_fine = sswc(s_fine)

    x_range = np.ptp(x_fine) or 1.0
    swc_range = np.ptp(swc_fine) or 1.0
    xn = x_fine / x_range
    swcn = swc_fine / swc_range

    seg_len = np.sqrt(np.diff(xn) ** 2 + np.diff(swcn) ** 2)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    arc /= arc[-1]

    s_anc = np.array(
        [
            arc[np.argmin((xn - xa / x_range) ** 2 + (swcn - fa / swc_range) ** 2)]
            for xa, fa in zip(x_anchors, swc_anchors)
        ]
    )
    s_starts = np.array([arc[np.argmin((xn - xs / x_range) ** 2)] for xs in x_starts])
    s_ends = np.array([arc[np.argmin((xn - xe / x_range) ** 2)] for xe in x_ends])

    return s_anc, s_starts, s_ends


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

    def bootstrap_residuals(
        self,
        x_anchors,
        swc_anchors,
        x_starts,
        delta_x,
        delta_swc,
        n_boot=200,
        rng=None,
    ):
        """Resample anchor residuals to estimate fit sensitivity."""
        if rng is None:
            rng = np.random.default_rng()
        self.fit(x_anchors, swc_anchors, x_starts, delta_x, delta_swc)
        x_anchors, swc_anchors = np.asarray(x_anchors), np.asarray(swc_anchors)
        residuals = swc_anchors - self(x_anchors)

        boots = []
        for _ in range(n_boot):
            y_boot = self(x_anchors) + rng.choice(
                residuals, size=len(residuals), replace=True
            )
            b = MonotonicSpline(self._inner_knots, self._k, self._w_chord)
            try:
                b.fit(x_anchors, y_boot, x_starts, delta_x, delta_swc)
                boots.append(b)
            except RuntimeError:
                pass
        return boots

    def bootstrap_data(
        self,
        x_anchors,
        swc_anchors,
        x_starts,
        delta_x,
        delta_swc,
        n_boot=200,
        rng=None,
    ):
        """Resample chord observations to estimate epistemic uncertainty."""
        if rng is None:
            rng = np.random.default_rng()
        self.fit(x_anchors, swc_anchors, x_starts, delta_x, delta_swc)
        n_chords = len(x_starts)

        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n_chords, n_chords)
            b = MonotonicSpline(self._inner_knots, self._k, self._w_chord)
            try:
                b.fit(
                    x_anchors, swc_anchors, x_starts[idx], delta_x[idx], delta_swc[idx]
                )
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
        self.sswc = None

    def fit(self, x_anchors, swc_anchors, x_starts, delta_x, delta_swc):
        inner = np.linspace(0, 1, self.knots + 2)[1:-1]
        k = self._k
        t = np.concatenate(([0.0] * (k + 1), inner, [1.0] * (k + 1)))
        n_c = len(t) - k - 1

        x_anchors = np.asarray(x_anchors)
        swc_anchors = np.asarray(swc_anchors)
        x_ends = x_starts + delta_x
        sort_idx = np.argsort(x_anchors)
        x_sorted, swc_sorted = x_anchors[sort_idx], swc_anchors[sort_idx]
        swc_dir = np.sign(swc_sorted[-1] - swc_sorted[0])
        if swc_dir == 0:
            swc_dir = 1

        x_min = min(x_sorted.min(), x_starts.min(), x_ends.min())
        x_max = max(x_sorted.max(), x_starts.max(), x_ends.max())

        s_anc = (x_sorted - x_min) / (x_max - x_min)
        s_starts = (x_starts - x_min) / (x_max - x_min)
        s_ends = (x_ends - x_min) / (x_max - x_min)

        s_init = np.linspace(0, 1, n_c)
        c0 = np.concatenate(
            [
                np.interp(s_init, np.linspace(0, 1, len(x_sorted)), x_sorted),
                np.interp(s_init, np.linspace(0, 1, len(swc_sorted)), swc_sorted),
            ]
        )

        w = self._w_chord
        for _ in range(self._n_iter):

            def objective(coeffs, s_anc=s_anc, s_starts=s_starts, s_ends=s_ends):
                cx, cswc = coeffs[:n_c], coeffs[n_c:]
                sx = BSpline(t, cx, k)
                sswc = BSpline(t, cswc, k)
                err_anc = np.sum(
                    (sx(s_anc) - x_sorted) ** 2 + (sswc(s_anc) - swc_sorted) ** 2
                )
                err_chord = np.sum(
                    (sx(s_starts) - x_starts) ** 2
                    + (sx(s_ends) - x_ends) ** 2
                    + (sswc(s_ends) - sswc(s_starts) - delta_swc) ** 2
                )
                return err_anc + w * err_chord

            def monotonic_con(coeffs):
                return swc_dir * BSpline(t, coeffs[n_c:], k)(
                    np.linspace(0, 1, 50), nu=1
                )

            constraints = [{"type": "ineq", "fun": monotonic_con}]

            res = minimize(objective, c0, constraints=constraints)
            if not res.success:
                raise RuntimeError(f"Spline optimization failed: {res.message}")

            c0 = res.x
            s_anc, s_starts, s_ends = _arclen_reparameterise_chords(
                BSpline(t, res.x[:n_c], k),
                BSpline(t, res.x[n_c:], k),
                x_sorted,
                swc_sorted,
                x_starts,
                x_ends,
            )

        self.sx = BSpline(t, res.x[:n_c], k)
        self.sswc = BSpline(t, res.x[n_c:], k)
        return self

    def predict(self, x, n_fine=2000):
        s_fine = np.linspace(0, 1, n_fine)
        return np.interp(x, self.sx(s_fine), self.sswc(s_fine))

    def knot_positions(self):
        inner_s = np.linspace(0, 1, self.knots + 2)[1:-1]
        return self.sx(inner_s), self.sswc(inner_s)

    def bootstrap_residuals(
        self,
        x_anchors,
        swc_anchors,
        x_starts,
        delta_x,
        delta_swc,
        n_boot=200,
        rng=None,
    ):
        """Resample anchor residuals to estimate fit sensitivity."""
        if rng is None:
            rng = np.random.default_rng()
        self.fit(x_anchors, swc_anchors, x_starts, delta_x, delta_swc)
        x_anchors, swc_anchors = np.asarray(x_anchors), np.asarray(swc_anchors)
        residuals = swc_anchors - self.predict(x_anchors)

        boots = []
        for _ in range(n_boot):
            swc_boot = self.predict(x_anchors) + rng.choice(
                residuals, size=len(residuals), replace=True
            )
            b = ParametricMonotonicSpline(
                self.knots, self._k, self._w_chord, self._n_iter
            )
            try:
                b.fit(x_anchors, swc_boot, x_starts, delta_x, delta_swc)
                boots.append(b)
            except RuntimeError:
                pass
        return boots

    def bootstrap_data(
        self,
        x_anchors,
        swc_anchors,
        x_starts,
        delta_x,
        delta_swc,
        n_boot=200,
        rng=None,
    ):
        """Resample chord observations to estimate epistemic uncertainty."""
        if rng is None:
            rng = np.random.default_rng()
        self.fit(x_anchors, swc_anchors, x_starts, delta_x, delta_swc)
        n_chords = len(x_starts)

        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n_chords, n_chords)
            b = ParametricMonotonicSpline(
                self.knots, self._k, self._w_chord, self._n_iter
            )
            try:
                b.fit(
                    x_anchors, swc_anchors, x_starts[idx], delta_x[idx], delta_swc[idx]
                )
                boots.append(b)
            except RuntimeError:
                pass
        return boots
