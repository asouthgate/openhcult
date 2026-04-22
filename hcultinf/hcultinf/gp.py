from __future__ import annotations

import numpy as np

from .calibrator import CordCalibrator


class GPWithPriorShape(CordCalibrator):
    def __init__(
        self,
        length_scale=None,
        variance=20.0,
        scale_prior_mean=None,
        scale_prior_std=None,
    ):
        super().__init__()
        self._length_scale = length_scale
        self._variance = variance
        self._scale_prior_mean = scale_prior_mean
        self._scale_prior_std = scale_prior_std

    def fit(
        self, x_anchors, swc_anchors, x_starts, delta_x, delta_swc, prior_x, prior_y
    ):
        x_ends = x_starts + delta_x
        self._mean, self._std, self.scale, self.nlml, self.noise = self._fit_gp_chords(
            x_anchors,
            swc_anchors,
            x_starts,
            x_ends,
            delta_swc,
            prior_x,
            prior_y,
            self._length_scale,
            self._variance,
            self._scale_prior_mean,
            self._scale_prior_std,
        )
        return self

    def _fit_gp_chords(
        self,
        x_anchor,
        y_anchor,
        x_starts,
        x_ends,
        delta_y,
        prior_x,
        prior_y,
        length_scale=None,
        variance=20.0,
        scale_prior_mean=None,
        scale_prior_std=None,
    ):
        from scipy.optimize import minimize_scalar

        if length_scale is None:
            length_scale = np.median(np.abs(x_ends - x_starts)) * 1.0

        def kernel(x1, x2):
            sq_dist = np.subtract.outer(x1, x2) ** 2
            return variance * np.exp(-0.5 * sq_dist / length_scale**2)

        h_anchor = np.interp(x_anchor, prior_x, prior_y)
        h_chords = np.interp(x_ends, prior_x, prior_y) - np.interp(
            x_starts, prior_x, prior_y
        )
        h = np.concatenate([h_anchor, h_chords])

        K_aa = kernel(x_anchor, x_anchor)
        K_ac = kernel(x_anchor, x_ends) - kernel(x_anchor, x_starts)
        K_cc = (
            kernel(x_ends, x_ends)
            - kernel(x_ends, x_starts)
            - kernel(x_starts, x_ends)
            + kernel(x_starts, x_starts)
        )

        observations = np.concatenate([y_anchor, delta_y])
        n_a = len(y_anchor)
        n_c = len(delta_y)
        n = n_a + n_c

        tau = (1.0 / scale_prior_std**2) if scale_prior_std is not None else 0.0
        mu_0 = scale_prior_mean if scale_prior_mean is not None else 0.0

        def _build_K(log_noise):
            noise = np.exp(log_noise)
            noise_mat = np.zeros((n, n))
            noise_mat[n_a:, n_a:] = noise * np.eye(n_c)
            return np.block([[K_aa, K_ac], [K_ac.T, K_cc]]) + noise_mat

        def neg_log_marginal_likelihood(log_noise):
            K = _build_K(log_noise)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e10
            Kinv_h = np.linalg.solve(K, h)
            Kinv_y = np.linalg.solve(K, observations)
            hKh = float(h @ Kinv_h)
            hKy = float(h @ Kinv_y)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            if tau > 0:
                post_prec = hKh + tau
                yKy = float(observations @ Kinv_y)
                return (
                    0.5 * yKy
                    - 0.5 * (hKy + mu_0 * tau) ** 2 / post_prec
                    + 0.5 * log_det
                    + 0.5 * np.log(post_prec)
                )
            beta = hKy / hKh
            residuals = observations - beta * h
            Kinv_r = np.linalg.solve(K, residuals)
            return 0.5 * float(residuals @ Kinv_r) + 0.5 * log_det

        result = minimize_scalar(
            neg_log_marginal_likelihood,
            bounds=(np.log(1e-8), np.log(1e2)),
            method="bounded",
        )

        K = _build_K(result.x)

        Kinv_h = np.linalg.solve(K, h)
        Kinv_y = np.linalg.solve(K, observations)
        hKh = float(h @ Kinv_h)
        hKy = float(h @ Kinv_y)

        if tau > 0:
            post_prec = hKh + tau
            scale = (hKy + mu_0 * tau) / post_prec
            beta_post_var = 1.0 / post_prec
        else:
            scale = hKy / hKh
            beta_post_var = 1.0 / hKh

        residuals = observations - scale * h
        weights = np.linalg.solve(K, residuals)

        L = np.linalg.cholesky(K)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        nlml = 0.5 * float(residuals @ np.linalg.solve(K, residuals)) + 0.5 * log_det

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
            prior_var_diag = np.diag(kernel(x_test, x_test))
            post_var = prior_var_diag - np.sum(kj * np.linalg.solve(K, kj.T).T, axis=1)
            h_test = np.interp(x_test, prior_x, prior_y)
            h_tilde = h_test - kj @ Kinv_h
            post_var += beta_post_var * h_tilde**2
            return np.sqrt(np.maximum(post_var, 0.0))

        return predict_mean, predict_std, scale, nlml, np.exp(result.x)
