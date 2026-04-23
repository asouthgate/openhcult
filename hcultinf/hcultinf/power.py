from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from .calibrator import CordCalibrator, _u, _estimate_covariance


class PowerCordCalibrator(CordCalibrator):
    def __init__(self, xmin, xmax, prior_weight=1.0):
        super().__init__()
        self._xmin = xmin
        self._xmax = xmax
        self._prior_weight = prior_weight

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

        def _g(x, power, y_int):
            return (1.0 - y_int) * _u(x, xmin, xmax) ** power + y_int

        power0, y_int0 = _fit_prior_shape(prior_x, prior_y, xmin, xmax)
        scale0 = 1.0

        w = self._prior_weight

        def residuals(params):
            s, p, yi = params
            return np.concatenate(
                [
                    swc_anchors - s * _g(x_anchors, p, yi),
                    delta_swc - s * (_g(x_ends, p, yi) - _g(x_starts, p, yi)),
                    w * s * (prior_y - _g(prior_x, p, yi)),
                ]
            )

        result = least_squares(
            residuals,
            x0=[scale0, power0, y_int0],
            bounds=([1e-6, 0.1, 0.0], [1e6, 20.0, 0.1]),
            method="trf",
        )
        scale, power, y_int = result.x
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        n_data_obs = len(swc_anchors) + len(delta_swc)
        pcov = _estimate_covariance(result, n_data_obs)

        self.scale = scale
        self.nlml = float(np.sum(result.fun**2))
        self.noise = None

        def _mean(x):
            return scale * _g(np.atleast_1d(x), power, y_int)

        def _std_internal(x):
            x = np.atleast_1d(x)
            u = _u(x, xmin, xmax)
            g = _g(x, power, y_int)
            grad = np.stack(
                [
                    g,
                    scale * (1.0 - y_int) * u**power * np.log(u),
                    scale * (1.0 - u**power),
                ],
                axis=1,
            )
            return np.sqrt(np.maximum(np.sum((grad @ pcov) * grad, axis=1), 0.0))

        def _ci_low(x):
            return _mean(x) - 1.96 * _std_internal(x)

        def _ci_high(x):
            return _mean(x) + 1.96 * _std_internal(x)

        self._mean = _mean
        self._ci_low = _ci_low
        self._ci_high = _ci_high
        return self


def _fit_prior_shape(prior_x, prior_y, xmin, xmax):
    y_int = float(np.interp(xmax, prior_x, prior_y))
    u = np.clip((xmax - prior_x) / (xmax - xmin), 1e-10, 1.0)
    span = max(1.0 - y_int, 1e-6)

    def loss(log_power):
        g = span * u ** np.exp(log_power) + y_int
        return float(np.sum((prior_y - g) ** 2))

    return (
        float(np.exp(minimize_scalar(loss, bounds=(-2.0, 3.0), method="bounded").x)),
        y_int,
    )
