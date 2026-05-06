from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .calibrator import CordCalibrator, _u, _estimate_covariance


def exponential_target(x, k, f_int, xmin, xmax):
    u = _u(x, xmin, xmax)
    return (1.0 - f_int) * np.exp(k * (u - 1.0)) + f_int


class ExponentialCordCalibrator(CordCalibrator):
    def __init__(self, xmax, prior_weight=1.0):
        super().__init__()
        self._xmax = xmax
        self._prior_weight = prior_weight
        self._data_xmin = None

    def target_func(self, x, scale, k, f_int):
        return scale * exponential_target(x, k, f_int, self._data_xmin, self._xmax)

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

        candidates = [x_starts.min(), x_ends.min(), x_anchors.min()]
        if len(prior_x) > 0:
            candidates.append(prior_x.min())
        self._data_xmin = float(min(candidates))

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

        result = least_squares(
            residuals,
            x0=[scale0, k0, y_int0],
            bounds=([0.0, 0.0001, 0.0], [1e4, 10000.0, 0.2]),
            method="trf",
        )

        scale, k, f_int = result.x
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        n_data_obs = len(swc_anchors) + len(delta_swc)
        pcov = _estimate_covariance(result, n_data_obs)

        self.scale = scale
        self.k = k
        self.f_int = f_int
        self.nlml = float(np.sum(result.fun**2))
        self.noise = None

        def _mean(x):
            return self.target_func(np.atleast_1d(x), scale, k, f_int)

        def _std_internal(x):
            x = np.atleast_1d(x)
            u = _u(x, self._data_xmin, self._xmax)
            exp_term = np.exp(k * (u - 1.0))
            g = (1.0 - f_int) * exp_term + f_int

            grad = np.stack(
                [
                    g,
                    scale * (1.0 - f_int) * (u - 1.0) * exp_term,
                    scale * (1.0 - exp_term),
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
