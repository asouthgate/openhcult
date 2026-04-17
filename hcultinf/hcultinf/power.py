from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from .calibrator import CordCalibrator


class PowerCordCalibrator(CordCalibrator):
    def __init__(self, xmin, xmax, prior_weight=None):
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

        def _u(x):
            return np.clip((xmax - x) / (xmax - xmin), 1e-10, 1.0)

        def _g(x, power, y_int):
            return (1.0 - y_int) * _u(x) ** power + y_int

        power0, y_int0 = _fit_prior_shape(prior_x, prior_y, xmin, xmax)
        scale0 = _init_scale(
            x_anchors, swc_anchors, x_starts, x_ends, delta_swc, power0, y_int0, _g
        )

        n_data = len(swc_anchors) + len(delta_swc)
        w = (
            self._prior_weight
            if self._prior_weight is not None
            else np.sqrt(n_data / len(prior_x))
        )

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
            bounds=([1e-6, 0.1, 0.0], [np.inf, 20.0, 0.99]),
            method="trf",
        )
        scale, power, y_int = result.x

        J = result.jac
        n_res, n_par = J.shape
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
            return scale * _g(np.atleast_1d(x), power, y_int)

        def _std(x):
            x = np.atleast_1d(x)
            u = _u(x)
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

        self._mean = _mean
        self._std = _std
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


def _init_scale(x_anchors, swc_anchors, x_starts, x_ends, delta_swc, power, y_int, _g):
    g_anc = _g(x_anchors, power, y_int)
    ratios = [s / g for s, g in zip(swc_anchors, g_anc) if g > 1e-6 and s > 1e-6]
    if ratios:
        return float(np.median(ratios))
    g_diffs = _g(x_ends, power, y_int) - _g(x_starts, power, y_int)
    valid = np.abs(g_diffs) > 1e-6
    if np.any(valid):
        return float(np.median(delta_swc[valid] / g_diffs[valid]))
    return 1.0
