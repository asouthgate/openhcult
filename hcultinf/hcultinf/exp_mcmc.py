from __future__ import annotations

import numpy as np
import emcee
from scipy.optimize import least_squares

from .calibrator import CordCalibrator
from .exp import exponential_target


class ExponentialCordCalibratorMCMC(CordCalibrator):
    def __init__(
        self,
        xmin,
        xmax,
        prior_weight=1.0,
        xmin_std=0.0,
        n_walkers=32,
        n_burn=500,
        n_steps=1000,
    ):
        super().__init__()
        self._xmin = xmin
        self._xmax = xmax
        self._prior_weight = prior_weight
        self._xmin_std = xmin_std
        self._n_walkers = n_walkers
        self._n_burn = n_burn
        self._n_steps = n_steps

    def fit(
        self, x_anchors, swc_anchors, x_starts, delta_x, delta_swc, prior_x, prior_y
    ):
        x_anchors = np.asarray(x_anchors)
        swc_anchors = np.asarray(swc_anchors)
        x_starts = np.asarray(x_starts)
        delta_swc = np.asarray(delta_swc)
        assert np.all(
            delta_swc > 0
        ), "delta_swc must be strictly positive for lognormal likelihood"
        prior_x = np.asarray(prior_x)
        prior_y = np.asarray(prior_y)
        delta_x = np.asarray(delta_x)
        x_ends = x_starts + delta_x

        xmin_hat = self._xmin
        xmax = self._xmax
        xmin_varies = self._xmin_std > 0
        sigma_anchor = 1.0
        sigma_prior = 1.0 / max(np.sqrt(self._prior_weight), 0.01)

        def _g(x, k, f_int, xmin):
            return exponential_target(x, k, f_int, xmin, xmax)

        try:
            mle = least_squares(
                lambda p: np.concatenate(
                    [
                        swc_anchors - p[0] * _g(x_anchors, p[1], p[2], xmin_hat),
                        delta_swc
                        - p[0]
                        * (
                            _g(x_ends, p[1], p[2], xmin_hat)
                            - _g(x_starts, p[1], p[2], xmin_hat)
                        ),
                        self._prior_weight
                        * (prior_y - _g(prior_x, p[1], p[2], xmin_hat)),
                    ]
                ),
                x0=[1.0, 20.0, 0.1],
                bounds=([0.0, 0.0001, 0.0], [1e4, 10000.0, 0.2]),
                method="trf",
            )
            scale0, k0, f_int0 = mle.x
        except RuntimeError:
            scale0, k0, f_int0 = 1.0, 20.0, 0.1

        n_dim = 4 + (1 if xmin_varies else 0)

        def log_prior(theta):
            scale, k, f_int, log_sigma = theta[:4]
            xmin = theta[4] if xmin_varies else xmin_hat
            if scale <= 0 or k < 0.01 or f_int < 0 or f_int > 0.3:
                return -np.inf
            if log_sigma < -5 or log_sigma > 3:
                return -np.inf
            if xmin_varies and abs(xmin - xmin_hat) > 5 * self._xmin_std:
                return -np.inf
            lp = 0.0
            if xmin_varies:
                lp -= 0.5 * ((xmin - xmin_hat) / self._xmin_std) ** 2
            return lp

        def log_likelihood(theta):
            scale, k, f_int, log_sigma = theta[:4]
            xmin = theta[4] if xmin_varies else xmin_hat
            sigma = np.exp(log_sigma)

            pred_a = scale * _g(x_anchors, k, f_int, xmin)
            ll = -0.5 * np.sum(
                ((swc_anchors - pred_a) / sigma_anchor) ** 2
                + np.log(2 * np.pi * sigma_anchor**2)
            )

            mu_c = scale * (_g(x_ends, k, f_int, xmin) - _g(x_starts, k, f_int, xmin))
            if np.any(mu_c <= 0):
                return -np.inf
            ll += np.sum(
                -0.5 * ((np.log(delta_swc) - np.log(mu_c)) / sigma) ** 2
                - log_sigma
                - np.log(delta_swc)
                - 0.5 * np.log(2 * np.pi)
            )

            g_norm = _g(prior_x, k, f_int, xmin)
            ll -= 0.5 * np.sum(
                ((prior_y - g_norm) / sigma_prior) ** 2
                + np.log(2 * np.pi * sigma_prior**2)
            )
            return ll

        def log_posterior(theta):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            ll = log_likelihood(theta)
            return lp + ll if np.isfinite(ll) else -np.inf

        p0 = np.array([scale0, k0, f_int0, np.log(0.3)])
        if xmin_varies:
            p0 = np.append(p0, xmin_hat)
        spread = np.array([max(0.1 * scale0, 1.0), max(0.1 * k0, 0.1), 0.01, 0.3])
        if xmin_varies:
            spread = np.append(spread, max(self._xmin_std, 1.0))
        pos = p0 + spread * np.random.randn(self._n_walkers, n_dim)
        pos[:, 0] = np.clip(pos[:, 0], 0.01, None)
        pos[:, 1] = np.clip(pos[:, 1], 0.01, None)
        pos[:, 2] = np.clip(pos[:, 2], 0.0, 0.29)

        sampler = emcee.EnsembleSampler(self._n_walkers, n_dim, log_posterior)
        state = sampler.run_mcmc(pos, self._n_burn, progress=False)
        sampler.reset()
        sampler.run_mcmc(state, self._n_steps, progress=False)
        samples = sampler.get_chain(flat=True)

        self._fit_samples = samples
        self._xmin_varies = xmin_varies

        self.scale = float(np.median(samples[:, 0]))
        self.noise = float(np.exp(np.median(samples[:, 3])))

        log_posts = sampler.get_log_prob(flat=True)
        self.nlml = float(-np.max(log_posts))

        thin = max(1, len(samples) // 2000)
        idx = np.arange(0, len(samples), thin)
        scale_s = samples[idx, 0]
        k_s = samples[idx, 1]
        f_int_s = samples[idx, 2]
        xmin_arr = samples[idx, 4] if xmin_varies else np.full(len(idx), xmin_hat)

        def _samples_at(x):
            x = np.atleast_1d(x)
            u = np.clip((xmax - x[None, :]) / (xmax - xmin_arr[:, None]), 1e-10, 1.0)
            g = (1.0 - f_int_s[:, None]) * np.exp(k_s[:, None] * (u - 1.0)) + f_int_s[
                :, None
            ]
            return scale_s[:, None] * g

        def _mean(x):
            return np.mean(_samples_at(x), axis=0)

        def _ci_low(x):
            return np.percentile(_samples_at(x), 2.5, axis=0)

        def _ci_high(x):
            return np.percentile(_samples_at(x), 97.5, axis=0)

        self._mean = _mean
        self._ci_low = _ci_low
        self._ci_high = _ci_high
        return self
