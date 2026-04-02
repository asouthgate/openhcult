from __future__ import annotations

import numpy as np


class GPWithPriorShape:
    def __init__(self, length_scale=None):
        self._length_scale = length_scale
        self._mean = None
        self._std = None
        self.scale = None

    def fit(
        self, x_anchors, swc_anchors, x_starts, delta_x, delta_swc, prior_x, prior_y
    ):
        x_ends = x_starts + delta_x
        self._mean, self._std, self.scale = self.fit_gp_chords(
            x_anchors,
            swc_anchors,
            x_starts,
            x_ends,
            delta_swc,
            prior_x,
            prior_y,
            self._length_scale,
        )
        return self

    def __call__(self, x):
        return self._mean(x)

    def predict(self, x):
        return self._mean(x), self._std(x)

    def fit_gp_chords(
        self,
        x_anchor,
        y_anchor,
        x_starts,
        x_ends,
        delta_y,
        prior_x,
        prior_y,
        length_scale=None,
    ):
        if length_scale is None:
            length_scale = np.median(np.abs(x_ends - x_starts)) * 1.0
        variance = 20.0
        noise = 0.0001

        def kernel(x1, x2):
            sq_dist = np.subtract.outer(x1, x2) ** 2
            return variance * np.exp(-0.5 * sq_dist / length_scale**2)

        # Prior shape evaluated at anchors and as chord differences
        h_anchor = np.interp(x_anchor, prior_x, prior_y)
        h_chords = np.interp(x_ends, prior_x, prior_y) - np.interp(
            x_starts, prior_x, prior_y
        )
        h = np.concatenate([h_anchor, h_chords])
        # print(h)
        # Kernel matrices
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

        # Estimate scale: beta = (h^T K^-1 y) / (h^T K^-1 h)
        Kinv_h = np.linalg.solve(K, h)
        Kinv_y = np.linalg.solve(K, observations)
        scale = float(h @ Kinv_y) / float(h @ Kinv_h)

        # GP on residuals after removing scaled mean
        residuals = observations - scale * h
        weights = np.linalg.solve(K, residuals)

        def _k_joint(x_test):
            k_ta = kernel(x_test, x_anchor)
            k_tc = kernel(x_test, x_ends) - kernel(x_test, x_starts)
            return np.hstack([k_ta, k_tc])

        def predict_mean(x_test):
            x_test = np.atleast_1d(x_test)
            h_test = np.interp(x_test, prior_x, prior_y)
            return scale * h_test + _k_joint(x_test) @ weights

        def predict_std(x_test):
            x_test = np.atleast_1d(x_test)
            kj = _k_joint(x_test)
            prior_var = np.diag(kernel(x_test, x_test))
            post_var = prior_var - np.sum(kj * np.linalg.solve(K, kj.T).T, axis=1)
            return np.sqrt(np.maximum(post_var, 0.0))

        return predict_mean, predict_std, scale
