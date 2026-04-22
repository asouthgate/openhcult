from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from .calibrator import CordCalibrator


def _u(x, xmin, xmax):
    return np.clip((xmax - x) / (xmax - xmin), 0.0, 1.0)


def exponential_target(x, k, f_int, xmin, xmax):
    # Form: (1 - y_int) * exp(k * (u - 1)) + y_int
    # u=1 (at xmin) -> exp(0) = 1 -> g = 1
    # u=0 (at xmax) -> exp(-k) -> g = (1-y_int)*exp(-k) + y_int
    u = _u(x, xmin, xmax)
    return (1.0 - f_int) * np.exp(k * (u - 1.0)) + f_int


class ExponentialCordCalibrator(CordCalibrator):
    def __init__(self, xmin, xmax, prior_weight=1.0):
        super().__init__()
        self._xmin = xmin
        self._xmax = xmax
        self._prior_weight = prior_weight

    def target_func(self, x, scale, k, f_int):
        return scale * exponential_target(x, k, f_int, self._xmin, self._xmax)

    def fit(
        self, x_anchors, swc_anchors, x_starts, delta_x, delta_swc, prior_x, prior_y
    ):
        x_anchors = np.asarray(x_anchors)
        swc_anchors = np.asarray(swc_anchors)
        x_starts = np.asarray(x_starts)
        delta_swc = np.asarray(delta_swc)
        prior_x = np.asarray(prior_x)
        prior_y = np.asarray(prior_y)
        x_ends = x_starts + np.asarray(delta_x)

        xmin, xmax = self._xmin, self._xmax

        # Initial guess for k and y_int based on prior
        # k0, y_int0 = _fit_prior_shape_exp(prior_x, prior_y, xmin, xmax)
        # Make an initial guess of scale based on the prior and chords
        # print("k0, yint0",k0, y_int0)
        k0 = 20.0
        y_int0 = 0.1

        scale0 = 1.0
        w = self._prior_weight

        def residuals(params):
            s, k, yi = params
            return np.concatenate(
                [
                    swc_anchors - self.target_func(x_anchors, s, k, yi),
                    delta_swc
                    - (
                        self.target_func(x_ends, s, k, yi)
                        - self.target_func(x_starts, s, k, yi)
                    ),
                    w * (prior_y - self.target_func(prior_x, s, k, yi) / s),
                ]
            )

        # Bounds: k is constrained to be positive for decay behavior relative to x
        result = least_squares(
            residuals,
            x0=[scale0, k0, y_int0],
            bounds=([0.0, 0.0001, 0.0], [1e4, 10000.0, 0.2]),
            method="trf",
        )

        scale, k, y_int = result.x

        # print("scale, k, y_int",scale, k, y_int)
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        # Covariance estimation
        J = result.jac
        n_par = J.shape[1]
        n_data_obs = len(swc_anchors) + len(delta_swc)
        data_res = result.fun[:n_data_obs]
        sigma2 = np.sum(data_res**2) / max(n_data_obs - n_par, 1)
        J_data = J[:n_data_obs]
        JtJ = J_data.T @ J_data
        try:
            pcov = sigma2 * np.linalg.inv(JtJ)
        except np.linalg.LinAlgError:
            pcov = sigma2 * np.linalg.pinv(JtJ)

        self.scale = scale
        self.nlml = float(np.sum(result.fun**2))
        self.noise = None

        def _mean(x):
            return self.target_func(np.atleast_1d(x), scale, k, y_int)

        def _std(x):
            x = np.atleast_1d(x)
            u = _u(x, self._xmin, self._xmax)
            exp_term = np.exp(k * (u - 1.0))
            g = (1.0 - y_int) * exp_term + y_int

            # Partial derivatives for the Jacobian
            # dg/ds = g
            # dg/dk = s * (1 - y_int) * (u - 1) * exp(k(u-1))
            # dg/dyi = s * (1 - exp(k(u-1)))
            grad = np.stack(
                [
                    g,
                    scale * (1.0 - y_int) * (u - 1.0) * exp_term,
                    scale * (1.0 - exp_term),
                ],
                axis=1,
            )
            return np.sqrt(np.maximum(np.sum((grad @ pcov) * grad, axis=1), 0.0))

        self._mean = _mean
        self._std = _std
        self._residuals = residuals(result.x)
        return self


def _fit_prior_shape_exp(prior_x, prior_y, xmin, xmax):
    """Finds initial k and y_int for the exponential model."""
    y_int = float(np.interp(xmax, prior_x, prior_y))
    u = np.clip((xmax - prior_x) / (xmax - xmin), 0.0, 1.0)
    span = max(1.0 - y_int, 1e-6)

    def loss(k):
        g = span * np.exp(k * (u - 1.0)) + y_int
        return float(np.sum((prior_y - g) ** 2))

    # Search for optimal k in a reasonable range
    res = minimize_scalar(loss, bounds=(0.001, 1000.0), method="bounded")
    return float(res.x), y_int
