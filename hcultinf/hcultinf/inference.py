from __future__ import annotations

import numpy as np

from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.optimize import lsq_linear


def _make_knots(inner_knots, x_min, x_max, k):
    return np.concatenate([[x_min] * (k + 1), inner_knots, [x_max] * (k + 1)])


def find_overlapping_groups(x, dx):
    """
    Groups indices of samples that share a continuous span of x-support.
    Args:
        x, dx, dy: np.arrays of sample starts, lengths, and changes.
    Returns:
        List of np.arrays containing indices for each group.
    """
    if len(x) == 0:
        return []

    # Sort indices by start point x
    idx = np.argsort(x)
    x_s, dx_s = x[idx], dx[idx]

    groups = []
    current_group_indices = [idx[0]]
    current_max_x = x_s[0] + dx_s[0]

    for i in range(1, len(x)):
        # If the sample starts before or at the current group's end
        if x_s[i] <= current_max_x:
            current_group_indices.append(idx[i])
            current_max_x = max(current_max_x, x_s[i] + dx_s[i])
        else:
            groups.append(np.array(current_group_indices))
            current_group_indices = [idx[i]]
            current_max_x = x_s[i] + dx_s[i]

    groups.append(np.array(current_group_indices))
    return groups


def fit_linear(x, y):
    m, c = np.polyfit(x, y, 1)
    return lambda v: m * v + c


def estimate_total_dy_full_coverage(x, dx, dy, n_nodes, endpoint=None):
    """Estimate the total change in y given a sample of x intervals and dy values, assuming no holes in the x coverage."""
    if endpoint is None:
        endpoint = max(x + dx)
    x_nodes = np.linspace(x.min(), endpoint, n_nodes)
    pwl = MonotonicPWL(x_nodes=x_nodes).fit(
        np.array([x.min()]), np.array([0.0]), x, dx, dy
    )
    return pwl(endpoint), pwl


def estimate_total_dy(x, dx, dy, priorx, priory, n_nodes, endpoint=None):
    """Estimate the total change in y given a sample of x intervals and dy values.

    This function does not assume a single covered interval. Instead, it extracts
    groups of elements, finding the sum within each group, summing the group sums,
    and using a prior shape where there is no coverage.
    """

    # The prior func must integrate to 1; integrate and assert the result equals 1
    # prior_integral = np.trapezoid(priory, priorx)
    # assert np.isclose(
    #     prior_integral, 1.0
    # ), f"Prior function must integrate to 1, but got {prior_integral}"

    assert priory.min() == 0
    assert (
        priory.max() == 1
    ), "Prior must be normalized to [0, 1] range, it is not required to integrate to 1"
    total = 0
    covered_proportion = 0
    groups = find_overlapping_groups(x, dx)
    for group in groups:
        xg, dxg, dyg = x[group], dx[group], dy[group]
        partial_sum, _ = estimate_total_dy_full_coverage(
            xg, dxg, dyg, n_nodes, endpoint=endpoint
        )
        total += partial_sum

        # mask = (priorx >= xg.min()) & (priorx <= (xg + dxg).max())
        # print(f"Interval min and max: {xg.min()} to {(xg + dxg).max()}")
        mask = (priorx >= xg.min()) & (priorx <= (xg + dxg).max())
        # print(f"Prior x range for group: {priorx[mask].min()} to {priorx[mask].max()}")

        covered_proportion += priory[mask].max() - priory[mask].min()
        # import matplotlib.pyplot as plt
        # plt.plot(priorx, priory)
        # plt.scatter(priorx, priory)
        # plt.axvline((xg + dxg).max(), color="red", label="xmax")
        # plt.axhline(covered_proportion, color="red", label="xmax")
        # plt.show()

        # print(f"Group min max: {xg.min(), (xg + dxg).max()} vs prior: {priory[mask].min()} {priory[mask].max()}")
        # print(f"Group {group}: partial_sum={partial_sum}, covered_proportion={covered_proportion}")
        # print(f"Percentage covered: {covered_proportion * 100:.2f}%")
        # covered_proportion = priory[mask].max() - priory[mask].min()

    # Finally extrapolate to the uncovered proportion specified by the prior
    # if 3.33... is 1/3, then the total is 3.33 / (1/3) = 10.0
    total = total / covered_proportion if covered_proportion > 0 else 0.0

    return total


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
                {"type": "eq", "fun": lambda y: interp(y, x_anchors) - swc_anchors},
            ],
            tol=1e-7,
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
