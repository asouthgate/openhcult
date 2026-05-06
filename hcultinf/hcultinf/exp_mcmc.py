from __future__ import annotations

import logging
import os

import numpy as np
import emcee
from scipy.special import ndtr, ndtri

from .calibrator import CordCalibrator
from .exp import ExponentialCordCalibrator, exponential_target
from .plot_style import apply_dark_theme, CLOUD_BLUE, ORANGE
from .prior import (
    MCMCPriors,
    _bounded_log_prior,
)

_logger = logging.getLogger(__name__)


def mcmc_log_anchor_prior_likelihood(
    scale,
    k_all,
    f_int_all,
    data_xmin,
    x_anchors,
    swc_anchors,
    prior_x,
    prior_y,
    xmax,
    n_sensors,
    sigma_anchor,
    sigma_prior,
):
    k_all = np.atleast_1d(k_all)
    f_int_all = np.atleast_1d(f_int_all)
    data_xmin = np.atleast_1d(data_xmin)

    g_anchor_parts = np.stack(
        [
            exponential_target(x_anchors, k_all[j], f_int_all[j], data_xmin[j], xmax)
            for j in range(n_sensors)
        ]
    )
    g_anchor = np.nanmean(g_anchor_parts, axis=0)
    pred_a = scale * g_anchor
    if not np.all(np.isfinite(pred_a)):
        return -np.inf
    ll = -0.5 * np.sum(
        ((swc_anchors - pred_a) / sigma_anchor) ** 2
        + np.log(2 * np.pi * sigma_anchor**2)
    )

    if len(prior_x) > 0:
        g_prior_parts = np.stack(
            [
                exponential_target(prior_x, k_all[j], f_int_all[j], data_xmin[j], xmax)
                for j in range(n_sensors)
            ]
        )
        g_prior = np.nanmean(g_prior_parts, axis=0)
        if not np.all(np.isfinite(g_prior)):
            return -np.inf
        ll -= 0.5 * np.sum(
            ((prior_y - g_prior) / sigma_prior) ** 2
            + np.log(2 * np.pi * sigma_prior**2)
        )

    return ll


def mcmc_log_chord_likelihood(
    scale,
    k,
    f_int,
    log_sigma,
    data_xmin,
    x_starts,
    x_ends,
    delta_swc,
    xmax,
):
    sigma = np.exp(log_sigma)
    mu_c = scale * (
        exponential_target(x_ends, k, f_int, data_xmin, xmax)
        - exponential_target(x_starts, k, f_int, data_xmin, xmax)
    )
    if np.any(mu_c <= 0) or not np.all(np.isfinite(mu_c)):
        return -np.inf
    log_mu_c = np.log(mu_c) - sigma**2 / 2
    ll = np.sum(
        -0.5 * ((np.log(delta_swc) - log_mu_c) / sigma) ** 2
        - log_sigma
        - np.log(delta_swc)
        - 0.5 * np.log(2 * np.pi)
    )
    return ll


def mcmc_log_joint(
    theta,
    x_anchors,
    swc_anchors,
    x_starts,
    x_ends,
    delta_swc,
    prior_x,
    prior_y,
    sensor_chord_labels,
    xmax,
    data_xmin,
    sigma_anchor,
    sigma_prior,
    n_sensors,
    priors,
):
    assert len(sensor_chord_labels) == len(
        x_starts
    ), f"sensor_chord_labels must have the same length as x_starts, got {len(sensor_chord_labels)} vs {len(x_starts)}"
    log_scale = theta[0]
    scale = np.exp(log_scale)
    if not np.isfinite(scale) or scale <= 0:
        return -np.inf
    n = n_sensors
    k_all = theta[1 : 1 + n]
    f_int_all = theta[1 + n : 1 + 2 * n]
    log_sigma = theta[1 + 2 * n]

    lp = priors.scale_log_prior(log_scale)
    if not np.isfinite(lp):
        return -np.inf
    for j in range(n):
        lp += priors.k_log_prior(k_all[j])
        if not np.isfinite(lp):
            return -np.inf
        lp += priors.f_int_log_prior(f_int_all[j])
        if not np.isfinite(lp):
            return -np.inf
    lp += priors.log_sigma_log_prior(log_sigma)
    if not np.isfinite(lp):
        return -np.inf

    ll = mcmc_log_anchor_prior_likelihood(
        scale,
        k_all,
        f_int_all,
        data_xmin,
        x_anchors,
        swc_anchors,
        prior_x,
        prior_y,
        xmax,
        n_sensors,
        sigma_anchor,
        sigma_prior,
    )

    for j in range(n):
        mask = sensor_chord_labels == j
        ll += mcmc_log_chord_likelihood(
            scale,
            k_all[j],
            f_int_all[j],
            log_sigma,
            data_xmin[j],
            x_starts[mask],
            x_ends[mask],
            delta_swc[mask],
            xmax,
        )

    return lp + ll


def samples_swc_at(x, scale_s, k_s, f_int_s, data_xmin, xmax):
    n_sensors = k_s.shape[1] if k_s.ndim == 2 else 1
    data_xmin = np.atleast_1d(data_xmin)
    x = np.asarray(x)
    if x.ndim == 1:
        assert n_sensors == 1, (
            f"x is 1D but n_sensors={n_sensors}; "
            f"provide x as (n_points, n_sensors) array"
        )
        x = x[:, None]

    n_points = x.shape[0]
    n_samples = len(scale_s)
    scale_s = np.asarray(scale_s).reshape(n_samples)

    g_parts = np.empty((n_samples, n_points, n_sensors))
    for j in range(n_sensors):
        x_j = x[:, j]
        g_j = exponential_target(
            x_j[None, :],
            k_s[:, j][:, None],
            f_int_s[:, j][:, None],
            data_xmin[j],
            xmax,
        )
        g_parts[:, :, j] = g_j

    g = np.nanmean(g_parts, axis=2)
    swc = scale_s[:, None] * g
    return swc


def samples_mean(x, scale_s, k_s, f_int_s, data_xmin, xmax):
    vals = samples_swc_at(x, scale_s, k_s, f_int_s, data_xmin, xmax)
    with np.errstate(all="ignore"):
        return np.nanmean(vals, axis=0)


def samples_ci_low(x, scale_s, k_s, f_int_s, data_xmin, xmax):
    vals = samples_swc_at(x, scale_s, k_s, f_int_s, data_xmin, xmax)
    with np.errstate(all="ignore"):
        return np.nanpercentile(vals, 2.5, axis=0)


def samples_ci_high(x, scale_s, k_s, f_int_s, data_xmin, xmax):
    vals = samples_swc_at(x, scale_s, k_s, f_int_s, data_xmin, xmax)
    with np.errstate(all="ignore"):
        return np.nanpercentile(vals, 97.5, axis=0)


class ExponentialCordCalibratorMCMC(CordCalibrator):
    def __init__(
        self,
        xmax,
        prior_weight=1.0,
        n_walkers=32,
        n_burn=500,
        n_steps=1000,
        sigma_anchor=0.1,
        sigma_prior_floor=0.001,
        sigma_init=0.3,
        init_spread=1.0,
        n_thin_target=2000,
        f_int_max=0.2,
        f_int_min=0.0,
        f_int_beta_a=1.0,
        f_int_beta_b=30.0,
        debug=True,
        n_sensors=1,
        priors=None,
        system_capacity_mean=None,
        system_capacity_std=None,
    ):
        super().__init__()
        self._debug = debug
        self._xmax = xmax
        self._prior_weight = prior_weight
        self._n_walkers = n_walkers
        self._n_burn = n_burn
        self._n_steps = n_steps
        self._sigma_anchor = sigma_anchor
        self._sigma_prior_floor = sigma_prior_floor
        self._sigma_init = sigma_init
        self._init_spread = init_spread
        self._n_thin_target = n_thin_target
        self._system_capacity_mean = system_capacity_mean
        self._system_capacity_std = system_capacity_std

        self._init_spread_scale = init_spread * 0.3
        self._init_spread_k = init_spread
        self._init_spread_f_int = init_spread * 1.0
        self._init_spread_sigma = init_spread

        self.f_int_min = f_int_min
        self.f_int_max = f_int_max
        self.f_int_beta_a = f_int_beta_a
        self.f_int_beta_b = f_int_beta_b
        self.n_sensors = n_sensors

        if priors is not None:
            self._priors = priors
        else:
            from .prior import _beta_log_prior

            self._priors = MCMCPriors(
                f_int_log_prior=_beta_log_prior(self.f_int_beta_a, self.f_int_beta_b),
            )

        self._fit_samples = None
        self._scale_s = None
        self._k_s = None
        self._f_int_s = None
        self._data_xmin = None

    def _compute_mean(self, x):
        return samples_mean(
            x, self._scale_s, self._k_s, self._f_int_s, self._data_xmin, self._xmax
        )

    def _compute_ci_low(self, x):
        return samples_ci_low(
            x, self._scale_s, self._k_s, self._f_int_s, self._data_xmin, self._xmax
        )

    def _compute_ci_high(self, x):
        return samples_ci_high(
            x, self._scale_s, self._k_s, self._f_int_s, self._data_xmin, self._xmax
        )

    def _prepare_fit_data(
        self,
        x_anchors,
        swc_anchors,
        x_starts,
        delta_x,
        delta_swc,
        prior_x,
        prior_y,
        sensor_chord_labels,
    ):
        assert np.all(
            delta_swc > 0
        ), "delta_swc must be strictly positive for lognormal likelihood"
        xmax = self._xmax

        assert min(x_starts) <= xmax, (
            f"chord x_starts (min={min(x_starts):.1f}) exceed model xmax ({xmax:.1f}); "
            f"sensor readings are outside the calibration domain"
        )

        x_ends = x_starts + delta_x

        data_xmin_arr = np.full(self.n_sensors, np.inf, dtype=float)
        for sj in range(self.n_sensors):
            mask = sensor_chord_labels == sj
            xs_j = x_starts[mask]
            xe_j = x_ends[mask]
            candidates = [xs_j.min(), xe_j.min(), x_anchors.min()]
            if len(prior_x) > 0:
                candidates.append(prior_x.min())
            data_xmin_arr[sj] = min(candidates)

        xmax = self._xmax
        sigma_anchor = self._sigma_anchor
        sigma_prior = 1.0 / max(np.sqrt(self._prior_weight), self._sigma_prior_floor)

        k0s = []
        f_int0s = []
        scale0s = []
        for sj in range(self.n_sensors):
            mask = sensor_chord_labels == sj
            quick = ExponentialCordCalibrator(xmax, self._prior_weight)
            quick.fit(
                x_anchors,
                swc_anchors,
                x_starts[mask],
                delta_x[mask],
                delta_swc[mask],
                prior_x,
                prior_y,
            )
            k0s.append(quick.k)
            f_int0s.append(quick.f_int)
            scale0s.append(quick.scale)
        scale0 = float(np.median(scale0s))

        return dict(
            scale0=scale0,
            k0=np.array(k0s),
            f_int0=np.array(f_int0s),
            data_xmin=data_xmin_arr,
            xmax=xmax,
            sigma_anchor=sigma_anchor,
            sigma_prior=sigma_prior,
        )

    def _init_walker_pos(self, initial_estimate):
        scale0 = initial_estimate["scale0"]
        k0 = initial_estimate["k0"]
        f_int0 = initial_estimate["f_int0"]

        log_scale_bounds = np.log([1.0, 1e10])
        walker_log_scale = np.clip(
            np.log(max(scale0, 1.0))
            + np.random.normal(0.0, self._init_spread_scale, self._n_walkers),
            log_scale_bounds[0],
            log_scale_bounds[1],
        )
        cols = [walker_log_scale]
        for sj in range(self.n_sensors):
            k0j = k0[sj]
            walker_k_j = np.clip(
                k0j
                + np.random.uniform(
                    -self._init_spread_k, self._init_spread_k, self._n_walkers
                ),
                0.0001,
                10000.0,
            )
            cols.append(walker_k_j)
        for sj in range(self.n_sensors):
            f0j = f_int0[sj]
            walker_f_j = np.clip(
                f0j + np.random.normal(0, 0.01, self._n_walkers),
                self.f_int_min,
                self.f_int_max,
            )
            cols.append(walker_f_j)
        walker_sigma = np.clip(
            np.log(self._sigma_init)
            + np.random.normal(0.0, self._init_spread_sigma, self._n_walkers),
            np.log(0.001),
            np.log(10.0),
        )
        cols.append(walker_sigma)

        init_pos = np.array(cols).T
        return init_pos

    def _run_sampler(self, pos, posterior_kwargs):
        sampler = emcee.EnsembleSampler(
            self._n_walkers,
            2 + 2 * self.n_sensors,
            mcmc_log_joint,
            kwargs=posterior_kwargs,
        )
        sampler.run_mcmc(pos, self._n_burn + self._n_steps, progress=False)
        self._chain = sampler.get_chain()
        self._log_prob = sampler.get_log_prob()
        return sampler

    def _postprocess(self, sampler, data):
        chain = self._chain
        log_probs = sampler.get_log_prob()

        self._log_probs = log_probs

        n = self.n_sensors
        burn = self._n_burn
        post = chain[burn:]
        post_flat = post.reshape(-1, post.shape[-1])

        self._fit_samples = post_flat
        self.scale = float(np.exp(np.median(post_flat[:, 0])))
        self.noise = float(np.exp(np.median(post_flat[:, 1 + 2 * n])))

        flat_log_probs = log_probs[burn:].ravel()
        self.nlml = float(-np.max(flat_log_probs))

        thin = max(1, len(post_flat) // self._n_thin_target)
        idx = np.arange(0, len(post_flat), thin)
        self._scale_s = np.exp(post_flat[idx, 0])
        self._k_s = post_flat[idx][:, [1 + j for j in range(n)]]
        self._f_int_s = post_flat[idx][:, [1 + n + j for j in range(n)]]
        self._log_sigma_s = post_flat[idx, 1 + 2 * n]

        self._data_xmin = data["data_xmin"]

        self._mean = self._compute_mean
        self._ci_low = self._compute_ci_low
        self._ci_high = self._compute_ci_high

    def fit(
        self,
        x_anchors,
        swc_anchors,
        x_starts,
        delta_x,
        delta_swc,
        prior_x=None,
        prior_y=None,
        sensor_chord_labels=None,
    ):

        if sensor_chord_labels is None:
            assert (
                self.n_sensors == 1
            ), "sensor_chord_labels must be provided if n_sensors > 1"
            sensor_chord_labels = np.array([0] * len(x_starts))

        if prior_x is None:
            prior_x = []
        if prior_y is None:
            prior_y = []

        x_starts = np.asarray(x_starts)
        delta_x = np.asarray(delta_x)
        delta_swc = np.asarray(delta_swc)
        prior_x = np.asarray(prior_x)
        prior_y = np.asarray(prior_y)
        x_anchors = np.asarray(x_anchors)
        swc_anchors = np.asarray(swc_anchors)

        initial_estimate = self._prepare_fit_data(
            x_anchors,
            swc_anchors,
            x_starts,
            delta_x,
            delta_swc,
            prior_x,
            prior_y,
            sensor_chord_labels,
        )
        walker_pos = self._init_walker_pos(initial_estimate)
        data = dict(
            x_anchors=x_anchors,
            swc_anchors=swc_anchors,
            x_starts=x_starts,
            x_ends=x_starts + delta_x,
            delta_swc=delta_swc,
            prior_x=prior_x,
            prior_y=prior_y,
            sensor_chord_labels=sensor_chord_labels,
        )
        posterior_kwargs = dict(
            xmax=initial_estimate["xmax"],
            data_xmin=initial_estimate["data_xmin"],
            sigma_anchor=initial_estimate["sigma_anchor"],
            sigma_prior=initial_estimate["sigma_prior"],
            n_sensors=self.n_sensors,
            priors=self._priors,
        )
        posterior_kwargs.update(data)
        try:
            sampler = self._run_sampler(walker_pos, posterior_kwargs)
        except ValueError as exc:
            _logger.error(self._diagnostic_dump(posterior_kwargs, walker_pos, exc))
            raise
        self._postprocess(sampler, posterior_kwargs)
        if self._debug:
            _logger.debug(
                self._diagnostic_dump(
                    posterior_kwargs, walker_pos, "post-fit diagnostic"
                )
            )
        return self

    def predict(self, x):
        x = np.atleast_1d(x)
        if x.ndim == 1:
            if self.n_sensors > 1:
                x = np.column_stack([x] * self.n_sensors)
            else:
                x = x[:, None]
        assert (
            x.ndim == 2 and x.shape[1] == self.n_sensors
        ), f"x must have shape (n_points, {self.n_sensors}), got {x.shape}"
        vals = self.posterior_samples_swc_at(x)
        with np.errstate(all="ignore"):
            mean = np.nanmean(vals, axis=0)
            ci_low = np.nanpercentile(vals, 2.5, axis=0)
            ci_high = np.nanpercentile(vals, 97.5, axis=0)
        return mean, ci_low, ci_high

    def __call__(self, x):
        x = np.atleast_1d(x)
        if x.ndim == 1:
            if self.n_sensors > 1:
                x = np.column_stack([x] * self.n_sensors)
            else:
                x = x[:, None]
        assert (
            x.ndim == 2 and x.shape[1] == self.n_sensors
        ), f"x must have shape (n_points, {self.n_sensors}), got {x.shape}"
        vals = self.posterior_samples_swc_at(x)
        with np.errstate(all="ignore"):
            return np.nanmean(vals, axis=0)

    def posterior_samples_swc_at(self, x, n=None):
        x = np.asarray(x)
        if x.ndim == 1:
            if self.n_sensors > 1:
                x = np.column_stack([x] * self.n_sensors)
            else:
                x = x[:, None]
        assert (
            x.ndim == 2 and x.shape[1] == self.n_sensors
        ), f"x must have shape (n_points, {self.n_sensors}), got {x.shape}"
        if n is not None and n < len(self._scale_s):
            idx = np.random.choice(len(self._scale_s), n, replace=False)
            return samples_swc_at(
                x,
                self._scale_s[idx],
                self._k_s[idx],
                self._f_int_s[idx],
                self._data_xmin,
                self._xmax,
            )
        return samples_swc_at(
            x,
            self._scale_s,
            self._k_s,
            self._f_int_s,
            self._data_xmin,
            self._xmax,
        )

    def fractional_water_content(self, x, n_samples=None, seed=None):
        """Return (mean, ci_low, ci_high) of SWC(x)/system_capacity.

        Requires system_capacity_mean and system_capacity_std at construction.
        Draws system_capacity independently from its log-normal prior and
        combines with SWC posterior samples via Monte Carlo.

        This is statistically identical to including system_capacity in the
        MCMC, since the likelihood does not depend on it (posterior = prior).
        """
        assert (
            self._system_capacity_mean is not None
            and self._system_capacity_std is not None
        ), "system_capacity_mean and system_capacity_std must be specified at construction"
        var = self._system_capacity_std**2
        mean_cap = self._system_capacity_mean
        mu_log = np.log(mean_cap**2 / np.sqrt(var + mean_cap**2))
        sigma_log = np.sqrt(np.log(1.0 + var / mean_cap**2))

        swc_samples = self.posterior_samples_swc_at(x)
        n_swc = swc_samples.shape[0]
        if n_samples is not None:
            idx = np.random.choice(n_swc, n_samples, replace=False)
            swc_samples = swc_samples[idx]
            n_swc = n_samples

        rng = np.random.default_rng(seed)
        cap_samples = rng.lognormal(mu_log, sigma_log, size=n_swc)

        frac_samples = swc_samples / cap_samples[:, None]
        with np.errstate(all="ignore"):
            mean = np.nanmean(frac_samples, axis=0)
            ci_low = np.nanpercentile(frac_samples, 2.5, axis=0)
            ci_high = np.nanpercentile(frac_samples, 97.5, axis=0)
        return mean, ci_low, ci_high

    def curve_credible_region(
        self,
        sensor_idx=0,
        alpha=0.95,
        n_bins=200,
        n_points=500,
        propagate_noise=True,
    ):
        """Compute the (alpha*100)% 2D credible region in (x, SWC) space
        for a single sensor.

        For each posterior sample, evaluates the calibration curve for that
        sensor from data_xmin to xmax, then pools all (x, swc) points
        and extracts the highest-density contour containing `alpha` fraction
        of the total density.

        Parameters:
            sensor_idx: which sensor to compute the contour for (default 0)
            alpha: coverage level (default 0.95)
            n_bins: number of histogram bins per dimension (default 200)
            n_points: number of x points per posterior sample (default 500)
            propagate_noise: if True, add lognormal observation noise to
                             each sample (default True)

        Returns:
            contour_paths: list of (N, 2) numpy arrays, each a boundary
                           segment in (x, SWC) coordinates
            level: float, the density threshold used
        """
        rng = np.random.default_rng()
        all_x = []
        all_swc = []

        data_xmin_j = self._data_xmin[sensor_idx]

        for i in range(len(self._scale_s)):
            scale_i = self._scale_s[i]
            k_i = self._k_s[i, sensor_idx]
            f_int_i = self._f_int_s[i, sensor_idx]
            sigma_i = np.exp(self._log_sigma_s[i])

            x_i = np.linspace(data_xmin_j, self._xmax, n_points)
            g_i = exponential_target(x_i, k_i, f_int_i, data_xmin_j, self._xmax)
            mu = scale_i * g_i

            if propagate_noise:
                z = rng.standard_normal(len(x_i))
                mu = mu * np.exp(-(sigma_i**2) / 2 + sigma_i * z)

            valid = np.isfinite(mu) & (mu > 0)
            if valid.sum() > 0:
                all_x.append(x_i[valid])
                all_swc.append(mu[valid])

        if len(all_x) == 0:
            _logger.warning("No valid (x, SWC) points for credible region")
            return [], 0.0

        points = np.column_stack([np.concatenate(all_x), np.concatenate(all_swc)])

        H, xedges, yedges = np.histogram2d(
            points[:, 0],
            points[:, 1],
            bins=n_bins,
        )
        H = H / H.sum()

        flat = np.sort(H.ravel())[::-1]
        cumsum = np.cumsum(flat)
        thresh_idx = np.searchsorted(cumsum, alpha * cumsum[-1])
        level = flat[min(thresh_idx, len(flat) - 1)]

        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(xcenters, ycenters, indexing="ij")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        cs = ax.contour(X, Y, H, levels=[level])
        paths = [seg for seg in cs.allsegs[0]]
        plt.close(fig)

        return paths, level

    def estimate_velocity_samples(self, v_window, t_window):
        """Estimate velocity in a time window"""

        swc_samples = self.posterior_samples_swc_at(v_window)

        t_ref = t_window[0]
        t_norm = t_window - t_ref

        slopes = []
        intercepts = []

        for sample in swc_samples:
            if np.any(np.isnan(sample)) or np.any(np.isinf(sample)):
                continue
            m, c = np.polyfit(t_norm, sample, 1)
            slopes.append(m)
            intercepts.append(c - m * t_ref)

        return np.array(slopes), np.array(intercepts), t_window

    def posterior_params(self, n=None):
        """Return dict of posterior parameter samples.

        If n is given, subsample to at most n samples.
        Keys: scale (n_samples,), k (n_samples, n_sensors),
              f_int (n_samples, n_sensors), sigma2 (n_samples,).
        """
        s = self._fit_samples
        if n is not None and n < len(s):
            idx = np.random.choice(len(s), n, replace=False)
            s = s[idx]
        n_s = self.n_sensors
        k_cols = [1 + j for j in range(n_s)]
        f_cols = [1 + n_s + j for j in range(n_s)]
        sigma_col = 1 + 2 * n_s
        return dict(
            scale=np.exp(s[:, 0]),
            k=s[:, k_cols],
            f_int=s[:, f_cols],
            sigma2=np.exp(2 * s[:, sigma_col]),
        )

    def _diagnostic_dump(self, data, pos, exc):
        xs = data["x_starts"]
        xe = data["x_ends"]
        dswc = data["delta_swc"]
        x_anc = data["x_anchors"]
        dmin = float(min(xs.min(), xe.min(), x_anc.min()))
        prior_min = (
            float(data["prior_x"].min()) if len(data["prior_x"]) > 0 else float("nan")
        )
        prior_max = (
            float(data["prior_x"].max()) if len(data["prior_x"]) > 0 else float("nan")
        )
        return (
            f"MCMC diagnostic: {exc}\n"
            f"  Input params: xmax={self._xmax} n_walkers={self._n_walkers}"
            f" n_burn={self._n_burn} n_steps={self._n_steps}"
            f" prior_weight={self._prior_weight}\n"
            f"  Data summary: N_chords={len(xs)} data_min_x={dmin}\n"
            f"    x_starts:  min={xs.min():.4f} max={xs.max():.4f}\n"
            f"    x_ends:    min={xe.min():.4f} max={xe.max():.4f}\n"
            f"    delta_swc: min={dswc.min():.4f} max={dswc.max():.4f}\n"
            f"    x_anchors: min={x_anc.min():.4f} max={x_anc.max():.4f}\n"
            f"    prior_x:   min={prior_min} max={prior_max}\n"
            f"  Walker init: cond={np.linalg.cond(pos):.2e}\n"
            f"    per-col min: {pos.min(axis=0).tolist()}\n"
            f"    per-col max: {pos.max(axis=0).tolist()}\n"
            f"    per-col std: {pos.std(axis=0).tolist()}"
        )


def _sensor_col(param, sensor_idx):
    """Extract a sensor's column from a (n_samples,) or (n_samples, n_sensors) array."""
    if param.ndim == 1:
        return param
    return param[:, sensor_idx]


def plot_corner(cal, out=None, title=None, sensor_idx=0):
    import matplotlib.pyplot as plt

    apply_dark_theme()
    params = cal.posterior_params()
    param_names = ["scale", "k", "f_int", "sigma2"]
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i][j]
            pi = _sensor_col(params[param_names[i]], sensor_idx)
            pj = _sensor_col(params[param_names[j]], sensor_idx)
            if i == j:
                ax.hist(pi, bins=50, density=True, color=CLOUD_BLUE, alpha=0.7)
                ax.axvline(np.median(pi), color=ORANGE, linewidth=1)
            elif i > j:
                step = max(1, len(pj) // 500)
                ax.scatter(pj[::step], pi[::step], s=1, alpha=0.3, color=CLOUD_BLUE)
            else:
                ax.set_visible(False)
            if j == 0:
                ax.set_ylabel(param_names[i])
            if i == n_params - 1:
                ax.set_xlabel(param_names[j])
    fig.tight_layout()
    if title is not None:
        fig.suptitle(title)
    if out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out)
    if os.environ.get("HCULT_TEST_DEBUG_PLOT", "0") == "1":
        plt.show()
    plt.close(fig)
    return fig
