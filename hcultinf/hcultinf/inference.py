from __future__ import annotations

import numpy as np

from scipy.interpolate import BSpline
from scipy.optimize import minimize


def _make_knots(inner_knots, x_min, x_max, k):
    return np.concatenate([[x_min] * (k + 1), inner_knots, [x_max] * (k + 1)])


def find_overlapping_groups(samples):
    if not samples:
        return []

    # Sort by start point
    sorted_samples = sorted(samples, key=lambda s: s["x"])
    groups = []
    current_group = [sorted_samples[0]]
    current_max_x = sorted_samples[0]["x"] + sorted_samples[0]["dx"]

    for s in sorted_samples[1:]:
        # If the next sample starts before or exactly at the current max reached
        if s["x"] <= current_max_x:
            current_group.append(s)
            current_max_x = max(current_max_x, s["x"] + s["dx"])
        else:
            groups.append(current_group)
            current_group = [s]
            current_max_x = s["x"] + s["dx"]

    groups.append(current_group)
    return groups


def estimate_total_dy(group):
    """
    Computes total dy by integrating local slopes.
    To handle random overlaps perfectly, it uses a weighted average
    of slopes based on the sample length (dx).
    """
    # 1. Define atomic segments from all unique endpoints
    pts = sorted(list(set([s["x"] for s in group] + [s["x"] + s["dx"] for s in group])))

    total_dy = 0.0
    for i in range(len(pts) - 1):
        x0, x1 = pts[i], pts[i + 1]
        width = x1 - x0
        if width <= 0:
            continue

        # 2. Find samples covering this segment
        covering_samples = [
            s
            for s in group
            if s["x"] <= x0 + 1e-13 and (s["x"] + s["dx"]) >= x1 - 1e-13
        ]

        if covering_samples:
            # 3. Weighted average of slopes (dy/dx)
            # We weight by s['dx'] because longer samples provide a more
            # stable 'global' estimate of the slope over this segment.
            weights = np.array([s["dx"] for s in covering_samples])
            slopes = np.array([s["dy"] / s["dx"] for s in covering_samples])

            avg_slope = np.average(slopes, weights=weights)
            total_dy += avg_slope * width

    return total_dy


def estimate_swc_max(x_starts, delta_x, delta_swc, prior):
    x_ends = x_starts + delta_x

    # merge overlapping chord intervals into disjoint coverage regions
    order = np.argsort(x_starts)
    regions = []
    x_lo, x_hi = x_starts[order[0]], x_ends[order[0]]
    for i in order[1:]:
        if x_starts[i] <= x_hi:
            x_hi = max(x_hi, x_ends[i])
        else:
            regions.append((x_lo, x_hi))
            x_lo, x_hi = x_starts[i], x_ends[i]
    regions.append((x_lo, x_hi))

    total_water = 0.0
    total_fraction = 0.0
    for x_lo, x_hi in regions:
        mask = (x_starts >= x_lo) & (x_ends <= x_hi)
        xs, dx, dswc = x_starts[mask], delta_x[mask], delta_swc[mask]
        x_nodes = np.unique(np.concatenate([xs, xs + dx]))
        pwl = MonotonicPWL(x_nodes=x_nodes).fit(
            np.array([x_lo]), np.array([0.0]), xs, dx, dswc
        )
        total_water += float(pwl(x_hi))
        total_fraction += float(prior(x_hi) - prior(x_lo))

    return total_water / total_fraction


def fit_linear(x, y):
    m, c = np.polyfit(x, y, 1)
    return lambda v: m * v + c


class MonotonicPWL:
    """Monotone piecewise-linear fit to anchor points and chord observations."""

    def __init__(self, n_nodes=50, x_nodes=None):
        self._n_nodes = n_nodes
        self._x_nodes_init = x_nodes
        self._x_nodes = None
        self._y_nodes = None

    def fit(self, x_anchors, swc_anchors, x_starts, delta_x, delta_swc):
        x_ends = x_starts + delta_x
        x_min = min(x_anchors.min(), x_starts.min())
        x_max = max(x_anchors.max(), x_ends.max())
        if self._x_nodes_init is not None:
            self._x_nodes = self._x_nodes_init
        else:
            self._x_nodes = np.linspace(x_min, x_max, self._n_nodes)

        def interp(y_nodes, x):
            return np.interp(x, self._x_nodes, y_nodes)

        def objective(y_nodes):
            return np.sum(
                (interp(y_nodes, x_ends) - interp(y_nodes, x_starts) - delta_swc) ** 2
            )

        y0 = np.interp(self._x_nodes, x_anchors, swc_anchors)
        res = minimize(
            objective,
            y0,
            constraints=[
                {"type": "ineq", "fun": np.diff},
                {"type": "eq", "fun": lambda y: interp(y, x_anchors) - swc_anchors},
            ],
        )
        self._y_nodes = res.x
        return self

    def __call__(self, x):
        return np.interp(x, self._x_nodes, self._y_nodes)


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


class GP:
    def __init__(self, length_scale=None):
        self._length_scale = length_scale
        self._mean = None
        self._std = None

    def fit(self, x_anchors, swc_anchors, x_starts, delta_x, delta_swc):
        x_ends = x_starts + delta_x
        self._mean, self._std = fit_gp_chords(
            x_anchors, swc_anchors, x_starts, x_ends, delta_swc, self._length_scale
        )
        return self

    def __call__(self, x):
        return self._mean(x)

    def predict(self, x):
        """Returns (mean, std) at x."""
        return self._mean(x), self._std(x)


def fit_gp_chords(x_anchor, y_anchor, x_starts, x_ends, delta_y, length_scale=None):
    """
    Fits a Gaussian Process using 1-point anchors and N-chord differences.
    Returns (predict_mean, predict_std) functions.
    """
    if length_scale is None:
        length_scale = np.median(np.abs(x_ends - x_starts)) * 2.0

    variance = np.var(delta_y) if len(delta_y) > 1 else 1.0
    noise = 1e-6

    def kernel(x1, x2):
        sq_dist = np.subtract.outer(x1, x2) ** 2
        return variance * np.exp(-0.5 * sq_dist / length_scale**2)

    K_aa = kernel(x_anchor, x_anchor)
    K_ac = kernel(x_anchor, x_ends) - kernel(x_anchor, x_starts)
    K_cc = (
        kernel(x_ends, x_ends)
        - kernel(x_ends, x_starts)
        - kernel(x_starts, x_ends)
        + kernel(x_starts, x_starts)
    )

    K = np.block([[K_aa, K_ac], [K_ac.T, K_cc + noise * np.eye(len(delta_y))]])

    observations = np.concatenate([y_anchor, delta_y])
    weights = np.linalg.solve(K, observations)

    def _k_joint(x_test):
        k_ta = kernel(x_test, x_anchor)
        k_tc = kernel(x_test, x_ends) - kernel(x_test, x_starts)
        return np.hstack([k_ta, k_tc])

    def predict_mean(x_test):
        return _k_joint(np.atleast_1d(x_test)) @ weights

    def predict_std(x_test):
        x_test = np.atleast_1d(x_test)
        kj = _k_joint(x_test)
        prior_var = np.diag(kernel(x_test, x_test))
        post_var = prior_var - np.sum(kj * np.linalg.solve(K, kj.T).T, axis=1)
        return np.sqrt(np.maximum(post_var, 0.0))

    return predict_mean, predict_std


# --- Usage Example ---
# x0, y0 = [0.0], [0.0]
# xs, xe = np.array([1, 5, 10]), np.array([2, 7, 15])
# dy = np.array([0.5, 1.2, 2.0])
# model = fit_gp_chords(x0, y0, xs, xe, dy)
# print(model(np.array([12.0])))
