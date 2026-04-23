from __future__ import annotations

import numpy as np
import emcee

from .calibrator import CordCalibrator
from .exp import ExponentialCordCalibrator, exponential_target


def mcmc_log_prior(theta, xmin_varies, xmin_hat, xmin_std):
    xmin = theta[4] if xmin_varies else xmin_hat
    lp = 0.0
    if xmin_varies:
        lp -= 0.5 * ((xmin - xmin_hat) / xmin_std) ** 2
    return lp


def mcmc_log_likelihood(
    theta,
    x_anchors,
    swc_anchors,
    x_starts,
    x_ends,
    delta_swc,
    prior_x,
    prior_y,
    xmin_hat,
    xmax,
    xmin_varies,
    sigma_anchor,
    sigma_prior,
):
    scale, k, f_int, log_sigma = theta[:4]
    xmin = theta[4] if xmin_varies else xmin_hat
    sigma = np.exp(log_sigma)

    pred_a = scale * exponential_target(x_anchors, k, f_int, xmin, xmax)
    ll = -0.5 * np.sum(
        ((swc_anchors - pred_a) / sigma_anchor) ** 2
        + np.log(2 * np.pi * sigma_anchor**2)
    )

    mu_c = scale * (
        exponential_target(x_ends, k, f_int, xmin, xmax)
        - exponential_target(x_starts, k, f_int, xmin, xmax)
    )
    if np.any(mu_c <= 0):
        return -np.inf
    ll += np.sum(
        -0.5 * ((np.log(delta_swc) - np.log(mu_c)) / sigma) ** 2
        - log_sigma
        - np.log(delta_swc)
        - 0.5 * np.log(2 * np.pi)
    )

    g_norm = exponential_target(prior_x, k, f_int, xmin, xmax)
    ll -= 0.5 * np.sum(
        ((prior_y - g_norm) / sigma_prior) ** 2 + np.log(2 * np.pi * sigma_prior**2)
    )
    return ll


def mcmc_log_posterior(
    theta,
    x_anchors,
    swc_anchors,
    x_starts,
    x_ends,
    delta_swc,
    prior_x,
    prior_y,
    xmin_hat,
    xmax,
    xmin_varies,
    xmin_std,
    sigma_anchor,
    sigma_prior,
):
    lp = mcmc_log_prior(theta, xmin_varies, xmin_hat, xmin_std)
    if not np.isfinite(lp):
        return -np.inf
    ll = mcmc_log_likelihood(
        theta,
        x_anchors,
        swc_anchors,
        x_starts,
        x_ends,
        delta_swc,
        prior_x,
        prior_y,
        xmin_hat,
        xmax,
        xmin_varies,
        sigma_anchor,
        sigma_prior,
    )
    return lp + ll if np.isfinite(ll) else -np.inf


def samples_at(x, scale_s, k_s, f_int_s, xmin_arr, xmax):
    x = np.atleast_1d(x)
    g = exponential_target(
        x[None, :], k_s[:, None], f_int_s[:, None], xmin_arr[:, None], xmax
    )
    return scale_s[:, None] * g


def samples_mean(x, scale_s, k_s, f_int_s, xmin_arr, xmax):
    return np.mean(samples_at(x, scale_s, k_s, f_int_s, xmin_arr, xmax), axis=0)


def samples_ci_low(x, scale_s, k_s, f_int_s, xmin_arr, xmax):
    return np.percentile(
        samples_at(x, scale_s, k_s, f_int_s, xmin_arr, xmax), 2.5, axis=0
    )


def samples_ci_high(x, scale_s, k_s, f_int_s, xmin_arr, xmax):
    return np.percentile(
        samples_at(x, scale_s, k_s, f_int_s, xmin_arr, xmax), 97.5, axis=0
    )


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
        sigma_anchor=0.1,
        sigma_prior_floor=0.001,
        sigma_init=0.3,
        init_spread=0.1,
        init_xmin_nsigma=3,
        n_thin_target=2000,
    ):
        super().__init__()
        self._xmin = xmin
        self._xmax = xmax
        self._prior_weight = prior_weight
        self._xmin_std = xmin_std
        self._n_walkers = n_walkers
        self._n_burn = n_burn
        self._n_steps = n_steps
        self._sigma_anchor = sigma_anchor
        self._sigma_prior_floor = sigma_prior_floor
        self._sigma_init = sigma_init
        self._init_spread = init_spread
        self._init_xmin_nsigma = init_xmin_nsigma
        self._n_thin_target = n_thin_target

    def _prepare_fit_data(
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
        sigma_anchor = self._sigma_anchor
        sigma_prior = 1.0 / max(np.sqrt(self._prior_weight), self._sigma_prior_floor)

        quick = ExponentialCordCalibrator(xmin_hat, xmax, self._prior_weight).fit(
            x_anchors, swc_anchors, x_starts, delta_x, delta_swc, prior_x, prior_y
        )
        scale0, k0, f_int0 = quick.scale, quick.k, quick.f_int

        return dict(
            x_anchors=x_anchors,
            swc_anchors=swc_anchors,
            x_starts=x_starts,
            x_ends=x_ends,
            delta_swc=delta_swc,
            prior_x=prior_x,
            prior_y=prior_y,
            scale0=scale0,
            k0=k0,
            f_int0=f_int0,
            xmin_hat=xmin_hat,
            xmax=xmax,
            xmin_varies=xmin_varies,
            sigma_anchor=sigma_anchor,
            sigma_prior=sigma_prior,
        )

    def _init_walker_pos(self, data):
        scale0 = data["scale0"]
        k0 = data["k0"]
        f_int0 = data["f_int0"]
        xmin_hat = data["xmin_hat"]
        xmin_varies = data["xmin_varies"]

        n_dim = 4 + (1 if xmin_varies else 0)
        p0 = np.array([scale0, k0, f_int0, np.log(self._sigma_init)])
        if xmin_varies:
            p0 = np.append(p0, xmin_hat)
        s = self._init_spread
        spread = np.array([s * scale0, s * k0, s * f_int0, s])
        if xmin_varies:
            spread = np.append(spread, self._init_xmin_nsigma * self._xmin_std)

        return p0 + spread * np.random.randn(self._n_walkers, n_dim)

    def _posterior_kwargs(self, data):
        return dict(
            x_anchors=data["x_anchors"],
            swc_anchors=data["swc_anchors"],
            x_starts=data["x_starts"],
            x_ends=data["x_ends"],
            delta_swc=data["delta_swc"],
            prior_x=data["prior_x"],
            prior_y=data["prior_y"],
            xmin_hat=data["xmin_hat"],
            xmax=data["xmax"],
            xmin_varies=data["xmin_varies"],
            xmin_std=self._xmin_std,
            sigma_anchor=data["sigma_anchor"],
            sigma_prior=data["sigma_prior"],
        )

    def _run_sampler(self, pos, n_dim, posterior_kwargs):
        sampler = emcee.EnsembleSampler(
            self._n_walkers,
            n_dim,
            mcmc_log_posterior,
            kwargs=posterior_kwargs,
        )
        state = sampler.run_mcmc(pos, self._n_burn, progress=False)
        sampler.reset()
        sampler.run_mcmc(state, self._n_steps, progress=False)
        return sampler

    def _postprocess(self, sampler, data):
        samples = sampler.get_chain(flat=True)
        xmin_hat = data["xmin_hat"]
        xmin_varies = data["xmin_varies"]
        xmax = data["xmax"]

        self._fit_samples = samples
        self._xmin_varies = xmin_varies
        self.scale = float(np.median(samples[:, 0]))
        self.noise = float(np.exp(np.median(samples[:, 3])))

        log_posts = sampler.get_log_prob(flat=True)
        self.nlml = float(-np.max(log_posts))

        self._scale_s, self._k_s, self._f_int_s, self._xmin_arr = self._thin_samples(
            samples, xmin_varies, xmin_hat
        )
        self._mean = lambda x: samples_mean(
            x, self._scale_s, self._k_s, self._f_int_s, self._xmin_arr, xmax
        )
        self._ci_low = lambda x: samples_ci_low(
            x, self._scale_s, self._k_s, self._f_int_s, self._xmin_arr, xmax
        )
        self._ci_high = lambda x: samples_ci_high(
            x, self._scale_s, self._k_s, self._f_int_s, self._xmin_arr, xmax
        )

    @staticmethod
    def _thin_samples(samples, xmin_varies, xmin_hat):
        thin = max(1, len(samples) // 2000)
        idx = np.arange(0, len(samples), thin)
        scale_s = samples[idx, 0]
        k_s = samples[idx, 1]
        f_int_s = samples[idx, 2]
        xmin_arr = samples[idx, 4] if xmin_varies else np.full(len(idx), xmin_hat)
        return scale_s, k_s, f_int_s, xmin_arr

    def fit(
        self, x_anchors, swc_anchors, x_starts, delta_x, delta_swc, prior_x, prior_y
    ):
        data = self._prepare_fit_data(
            x_anchors, swc_anchors, x_starts, delta_x, delta_swc, prior_x, prior_y
        )
        pos = self._init_walker_pos(data)
        n_dim = 4 + (1 if data["xmin_varies"] else 0)
        posterior_kwargs = self._posterior_kwargs(data)
        sampler = self._run_sampler(pos, n_dim, posterior_kwargs)
        self._postprocess(sampler, data)
        return self
