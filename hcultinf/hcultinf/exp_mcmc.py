from __future__ import annotations

import logging
import os

import numpy as np
import emcee
from .calibrator import CordCalibrator
from .exp import ExponentialCordCalibrator, exponential_target_u
from .logging import timed
from .plot_style import apply_dark_theme, CLOUD_BLUE, ORANGE
from .prior import (
    MCMCPriors,
    _bounded_log_prior,
)

_logger = logging.getLogger(__name__)

_LOG_2PI = np.log(2 * np.pi)


class _ParamLayout:
    """Encapsulates parameter indexing for the MCMC theta vector.

    Layout: [log_scale, k_0..k_{n-1}, f_int_0..f_int_{n-1}, log_sigma]
    """

    __slots__ = ("n", "_k_start", "_f_start", "_log_sigma_idx", "_dim")

    def __init__(self, n_sensors: int):
        self.n = n_sensors
        self._k_start = 1
        self._f_start = 1 + n_sensors
        self._log_sigma_idx = 1 + 2 * n_sensors
        self._dim = 2 + 2 * n_sensors

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def log_scale_idx(self) -> int:
        return 0

    @property
    def k_slice(self) -> slice:
        return slice(self._k_start, self._f_start)

    @property
    def f_int_slice(self) -> slice:
        return slice(self._f_start, self._log_sigma_idx)

    @property
    def log_sigma_idx(self) -> int:
        return self._log_sigma_idx


def _compute_g(k, f_int, u):
    """Compute the exponential response curve g(u) = (1-f)*exp(k*(u-1)) + f.

    Supports broadcasting: k, f_int shape (..., n_sensors), u shape (..., n_sensors).
    Returns shape matching the broadcast result.
    """
    one_minus_f = 1.0 - f_int
    return one_minus_f * np.exp(k * (u - 1.0)) + f_int


def mcmc_log_joint(
    theta,
    u_anchors_m1,
    swc_anchors,
    u_starts_m1,
    u_ends_m1,
    log_delta_swc_all,
    sensor_idx_per_chord,
    u_prior_m1,
    prior_y,
    sigma_anchor,
    sigma_prior,
    n_sensors,
    priors,
    log_const_anchor,
    log_const_prior,
):
    squeeze = False
    if theta.ndim == 1:
        theta = theta[None, :]
        squeeze = True

    layout = _ParamLayout(n_sensors)
    log_scale = theta[:, layout.log_scale_idx]
    scale = np.exp(log_scale)
    k_all = theta[:, layout.k_slice]
    f_int_all = theta[:, layout.f_int_slice]
    log_sigma = theta[:, layout.log_sigma_idx]

    lp = (
        priors.scale_log_prior(log_scale)
        + priors.k_log_prior(k_all).sum(axis=1)
        + priors.f_int_log_prior(f_int_all).sum(axis=1)
        + priors.log_sigma_log_prior(log_sigma)
    )

    g_a = np.mean(
        (1.0 - f_int_all[:, :, None])
        * np.exp(k_all[:, :, None] * u_anchors_m1[None, :, :])
        + f_int_all[:, :, None],
        axis=1,
    )
    pred_a = scale[:, None] * g_a
    ll = -0.5 * np.sum(
        ((swc_anchors - pred_a) / sigma_anchor) ** 2 + log_const_anchor, axis=1
    )

    if u_prior_m1 is not None:
        g_p = np.mean(
            (1.0 - f_int_all[:, :, None])
            * np.exp(k_all[:, :, None] * u_prior_m1[None, :, :])
            + f_int_all[:, :, None],
            axis=1,
        )
        ll += -0.5 * np.sum(
            ((prior_y - g_p) / sigma_prior) ** 2 + log_const_prior, axis=1
        )

    k_exp = k_all[:, sensor_idx_per_chord]
    f_exp = f_int_all[:, sensor_idx_per_chord]
    mu_c = (
        scale[:, None]
        * (1.0 - f_exp)
        * (np.exp(k_exp * u_ends_m1) - np.exp(k_exp * u_starts_m1))
    )
    sigma = np.exp(log_sigma)
    with np.errstate(divide="ignore"):
        log_mu_c = np.log(np.maximum(mu_c, 0.0)) - sigma[:, None] ** 2 / 2
    ll += np.sum(
        -0.5 * ((log_delta_swc_all - log_mu_c) / sigma[:, None]) ** 2
        - log_sigma[:, None]
        - log_delta_swc_all
        - 0.5 * _LOG_2PI,
        axis=1,
    )

    result = lp + ll
    if squeeze:
        return float(result[0])
    return result


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

    n_samples = len(scale_s)
    scale_s = np.asarray(scale_s).reshape(n_samples)

    inv_denom = xmax - data_xmin
    u = (xmax - x) / inv_denom[None, :]

    k_exp = k_s[:, None, :]
    f_exp = f_int_s[:, None, :]
    u_b = u[None, :, :]

    g_mean = np.nanmean(_compute_g(k_exp, f_exp, u_b), axis=2)
    return scale_s[:, None] * g_mean


def _aggregate_samples(x, scale_s, k_s, f_int_s, data_xmin, xmax, func):
    vals = samples_swc_at(x, scale_s, k_s, f_int_s, data_xmin, xmax)
    with np.errstate(all="ignore"):
        return func(vals, axis=0)


def samples_mean(x, scale_s, k_s, f_int_s, data_xmin, xmax):
    return _aggregate_samples(x, scale_s, k_s, f_int_s, data_xmin, xmax, np.nanmean)


def samples_ci_low(x, scale_s, k_s, f_int_s, data_xmin, xmax):
    return _aggregate_samples(
        x,
        scale_s,
        k_s,
        f_int_s,
        data_xmin,
        xmax,
        lambda v: np.nanpercentile(v, 2.5, axis=0),
    )


def samples_ci_high(x, scale_s, k_s, f_int_s, data_xmin, xmax):
    return _aggregate_samples(
        x,
        scale_s,
        k_s,
        f_int_s,
        data_xmin,
        xmax,
        lambda v: np.nanpercentile(v, 97.5, axis=0),
    )


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

    def _reshape_x(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if x.ndim == 1:
            if self.n_sensors == 1:
                x = x[:, None]
            else:
                x = np.column_stack([x] * self.n_sensors)
        if x.shape[1] < self.n_sensors:
            pad = np.full((x.shape[0], self.n_sensors - x.shape[1]), np.nan)
            x = np.column_stack([x, pad])
        if x.shape[1] > self.n_sensors:
            raise ValueError(
                f"x has {x.shape[1]} columns but model expects {self.n_sensors}; "
                f"provide x as (n_points, {self.n_sensors}) array"
            )
        return x

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

    @timed
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

        sensor_chord_labels = np.asarray(sensor_chord_labels)
        sensor_idx_per_chord = sensor_chord_labels

        chord_x_min = np.minimum(x_starts, x_ends)
        data_xmin_arr = np.full(self.n_sensors, np.inf, dtype=float)
        np.minimum.at(data_xmin_arr, sensor_idx_per_chord, chord_x_min)
        global_anchor_min = x_anchors.min()
        if len(prior_x) > 0:
            global_anchor_min = min(global_anchor_min, prior_x.min())
        data_xmin_arr = np.minimum(data_xmin_arr, global_anchor_min)

        inv_denom = xmax - data_xmin_arr
        u_anchors = (xmax - x_anchors) / inv_denom[:, None]
        u_anchors_m1 = u_anchors - 1.0
        u_prior = None if len(prior_x) == 0 else (xmax - prior_x) / inv_denom[:, None]
        u_prior_m1 = None if u_prior is None else u_prior - 1.0

        inv_denom_per_chord = inv_denom[sensor_idx_per_chord]
        u_starts_m1 = (xmax - x_starts) / inv_denom_per_chord - 1.0
        u_ends_m1 = (xmax - x_ends) / inv_denom_per_chord - 1.0
        log_delta_swc_all = np.log(delta_swc)

        sigma_anchor = self._sigma_anchor
        sigma_prior = 1.0 / max(np.sqrt(self._prior_weight), self._sigma_prior_floor)
        log_const_anchor = np.log(2 * np.pi * sigma_anchor**2)
        log_const_prior = np.log(2 * np.pi * sigma_prior**2)

        quick_fits = []
        for sj in range(self.n_sensors):
            m = sensor_chord_labels == sj
            q = ExponentialCordCalibrator(xmax, self._prior_weight)
            q.fit(
                x_anchors,
                swc_anchors,
                x_starts[m],
                delta_x[m],
                delta_swc[m],
                prior_x,
                prior_y,
            )
            quick_fits.append(q)
        scale0 = float(np.median([q.scale for q in quick_fits]))

        return dict(
            scale0=scale0,
            k0=np.array([q.k for q in quick_fits]),
            f_int0=np.array([q.f_int for q in quick_fits]),
            data_xmin=data_xmin_arr,
            xmax=xmax,
            sigma_anchor=sigma_anchor,
            sigma_prior=sigma_prior,
            u_anchors_m1=u_anchors_m1,
            u_prior_m1=u_prior_m1,
            u_starts_m1=u_starts_m1,
            u_ends_m1=u_ends_m1,
            log_delta_swc_all=log_delta_swc_all,
            sensor_idx_per_chord=sensor_idx_per_chord,
            log_const_anchor=log_const_anchor,
            log_const_prior=log_const_prior,
        )

    def _init_walker_pos(self, initial_estimate):
        scale0 = initial_estimate["scale0"]
        k0 = initial_estimate["k0"]
        f_int0 = initial_estimate["f_int0"]
        n = self.n_sensors
        w = self._n_walkers
        s = self._init_spread

        walker_log_scale = np.clip(
            np.log(max(scale0, 1.0)) + np.random.normal(0.0, s * 0.3, w),
            np.log(1.0),
            np.log(1e10),
        )

        walker_k = np.clip(
            k0[None, :] + np.random.uniform(-s, s, (w, n)),
            0.0001,
            10000.0,
        )

        walker_f_int = np.clip(
            f_int0[None, :] + np.random.normal(0, 0.01, (w, n)),
            self.f_int_min,
            self.f_int_max,
        )

        walker_log_sigma = np.clip(
            np.log(self._sigma_init) + np.random.normal(0.0, s, w),
            np.log(0.001),
            np.log(10.0),
        )

        return np.column_stack(
            [
                walker_log_scale,
                walker_k,
                walker_f_int,
                walker_log_sigma,
            ]
        )

    @timed
    def _run_sampler(self, pos, posterior_kwargs):
        sampler = emcee.EnsembleSampler(
            self._n_walkers,
            _ParamLayout(self.n_sensors).dim,
            mcmc_log_joint,
            kwargs=posterior_kwargs,
            vectorize=True,
        )
        sampler.run_mcmc(pos, self._n_burn + self._n_steps, progress=False)
        self._chain = sampler.get_chain()
        self._log_prob = sampler.get_log_prob()
        return sampler

    @timed
    def _postprocess(self, sampler, data):
        chain = self._chain
        log_probs = sampler.get_log_prob()

        self._log_probs = log_probs

        layout = _ParamLayout(self.n_sensors)
        burn = self._n_burn
        post = chain[burn:]
        post_flat = post.reshape(-1, post.shape[-1])

        self._fit_samples = post_flat
        self.scale = float(np.exp(np.median(post_flat[:, layout.log_scale_idx])))
        self.noise = float(np.exp(np.median(post_flat[:, layout.log_sigma_idx])))

        flat_log_probs = log_probs[burn:].ravel()
        self.nlml = float(-np.max(flat_log_probs))

        thin = max(1, len(post_flat) // self._n_thin_target)
        idx = np.arange(0, len(post_flat), thin)
        thinned = post_flat[idx]
        self._scale_s = np.exp(thinned[:, layout.log_scale_idx])
        self._k_s = thinned[:, layout.k_slice]
        self._f_int_s = thinned[:, layout.f_int_slice]
        self._log_sigma_s = thinned[:, layout.log_sigma_idx]

        self._data_xmin = data["data_xmin"]

        self._mean = self._compute_mean
        self._ci_low = self._compute_ci_low
        self._ci_high = self._compute_ci_high

    @timed
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
        posterior_kwargs = dict(
            u_anchors_m1=initial_estimate["u_anchors_m1"],
            swc_anchors=swc_anchors,
            u_starts_m1=initial_estimate["u_starts_m1"],
            u_ends_m1=initial_estimate["u_ends_m1"],
            log_delta_swc_all=initial_estimate["log_delta_swc_all"],
            sensor_idx_per_chord=initial_estimate["sensor_idx_per_chord"],
            u_prior_m1=initial_estimate["u_prior_m1"],
            prior_y=prior_y,
            sigma_anchor=initial_estimate["sigma_anchor"],
            sigma_prior=initial_estimate["sigma_prior"],
            n_sensors=self.n_sensors,
            priors=self._priors,
            log_const_anchor=initial_estimate["log_const_anchor"],
            log_const_prior=initial_estimate["log_const_prior"],
        )
        try:
            sampler = self._run_sampler(walker_pos, posterior_kwargs)
        except ValueError as exc:
            _logger.error(self._diagnostic_dump(initial_estimate, walker_pos, exc))
            raise
        self._postprocess(sampler, initial_estimate)
        if self._debug:
            _logger.debug(
                self._diagnostic_dump(
                    initial_estimate, walker_pos, "post-fit diagnostic"
                )
            )
        return self

    @timed
    def predict(self, x):
        x = self._reshape_x(x)
        vals = self.posterior_samples_swc_at(x)
        with np.errstate(all="ignore"):
            return (
                np.nanmean(vals, axis=0),
                np.nanpercentile(vals, 2.5, axis=0),
                np.nanpercentile(vals, 97.5, axis=0),
            )

    def __call__(self, x):
        x = self._reshape_x(x)
        vals = self.posterior_samples_swc_at(x)
        with np.errstate(all="ignore"):
            return np.nanmean(vals, axis=0)

    @timed
    def posterior_samples_swc_at(self, x, n=None, exclude_burnin=True):
        x = self._reshape_x(x)
        scale_s, k_s, f_int_s = self._scale_s, self._k_s, self._f_int_s
        if exclude_burnin:
            scale_s = scale_s[self._n_burn :]
            k_s = k_s[self._n_burn :]
            f_int_s = f_int_s[self._n_burn :]
        if n is not None and n < len(scale_s):
            idx = np.random.choice(len(scale_s), n, replace=False)
            scale_s, k_s, f_int_s = scale_s[idx], k_s[idx], f_int_s[idx]
        samples = samples_swc_at(x, scale_s, k_s, f_int_s, self._data_xmin, self._xmax)
        return samples

    def _capacity_lognorm_params(self):
        assert (
            self._system_capacity_mean is not None
            and self._system_capacity_std is not None
        ), "system_capacity_mean and system_capacity_std must be specified at construction"
        var = self._system_capacity_std**2
        mean_cap = self._system_capacity_mean
        mu_log = np.log(mean_cap**2 / np.sqrt(var + mean_cap**2))
        sigma_log = np.sqrt(np.log(1.0 + var / mean_cap**2))
        return mu_log, sigma_log

    @timed
    def fractional_water_content(self, x, n_samples=None, seed=None):
        """Return (mean, ci_low, ci_high) of SWC(x)/system_capacity.

        Requires system_capacity_mean and system_capacity_std at construction.
        Draws system_capacity independently from its log-normal prior and
        combines with SWC posterior samples via Monte Carlo.

        This is statistically identical to including system_capacity in the
        MCMC, since the likelihood does not depend on it (posterior = prior).
        """
        frac_samples = self.fractional_water_content_samples(x, n_samples, seed)
        with np.errstate(all="ignore"):
            return (
                np.nanmean(frac_samples, axis=0),
                np.nanpercentile(frac_samples, 2.5, axis=0),
                np.nanpercentile(frac_samples, 97.5, axis=0),
            )

    @timed
    def fractional_water_content_samples(self, x, n_samples=None, seed=None):
        """Return raw (n_samples, n_times) array of fractional SWC samples."""
        mu_log, sigma_log = self._capacity_lognorm_params()
        swc_samples = self.posterior_samples_swc_at(x)
        n_swc = swc_samples.shape[0]
        if n_samples is not None:
            idx = np.random.choice(n_swc, n_samples, replace=False)
            swc_samples = swc_samples[idx]
            n_swc = n_samples
        return (
            swc_samples
            / np.random.default_rng(seed).lognormal(mu_log, sigma_log, size=n_swc)[
                :, None
            ]
        )

    @timed
    def estimate_velocity_samples(self, v_window, t_window):
        """Estimate velocity in a time window"""

        swc_samples = self.posterior_samples_swc_at(x=v_window)

        t_ref = t_window[0]
        t_norm = t_window - t_ref

        valid = ~np.any(np.isnan(swc_samples) | np.isinf(swc_samples), axis=1)
        valid_swc = swc_samples[valid]

        if len(valid_swc) == 0:
            return np.array([]), np.array([]), t_window

        A = np.column_stack([t_norm, np.ones(len(t_norm))])
        coeffs = np.linalg.lstsq(A, valid_swc.T, rcond=None)[0]
        slopes = coeffs[0]
        intercepts = coeffs[1] - coeffs[0] * t_ref

        return slopes, intercepts, t_window

    def posterior_params(self, n=None):
        """Return dict of posterior parameter samples.

        If n is given, subsample to at most n samples.
        Keys: scale (n_samples,), k (n_samples, n_sensors),
              f_int (n_samples, n_sensors), sigma2 (n_samples,).
        """
        s = self._fit_samples
        if n is not None and n < len(s):
            s = s[np.random.choice(len(s), n, replace=False)]
        layout = _ParamLayout(self.n_sensors)
        return dict(
            scale=np.exp(s[:, layout.log_scale_idx]),
            k=s[:, layout.k_slice],
            f_int=s[:, layout.f_int_slice],
            sigma2=np.exp(2 * s[:, layout.log_sigma_idx]),
        )

    def _diagnostic_dump(self, data, pos, exc):
        u_starts_m1 = data["u_starts_m1"]
        u_ends_m1 = data["u_ends_m1"]
        log_delta_swc_all = data["log_delta_swc_all"]
        u_anc_m1 = data["u_anchors_m1"]
        u_prior_m1 = data["u_prior_m1"]
        prior_minmax = (
            (float(u_prior_m1.min()), float(u_prior_m1.max()))
            if u_prior_m1 is not None
            else (float("nan"), float("nan"))
        )
        n_chords = len(u_starts_m1)
        return (
            f"MCMC diagnostic: {exc}\n"
            f"  Input params: xmax={self._xmax} n_walkers={self._n_walkers}"
            f" n_burn={self._n_burn} n_steps={self._n_steps}"
            f" prior_weight={self._prior_weight}\n"
            f"  Data summary: N_chords={n_chords} data_min_x={float(data['data_xmin'].min()):.1f}\n"
            f"    u_starts_m1:  min={u_starts_m1.min():.4f}"
            f" max={u_starts_m1.max():.4f}\n"
            f"    u_ends_m1:    min={u_ends_m1.min():.4f}"
            f" max={u_ends_m1.max():.4f}\n"
            f"    log_delta_swc: min={log_delta_swc_all.min():.4f}"
            f" max={log_delta_swc_all.max():.4f}\n"
            f"    u_anchors_m1: min={u_anc_m1.min():.4f} max={u_anc_m1.max():.4f}\n"
            f"    u_prior_m1:   min={prior_minmax[0]} max={prior_minmax[1]}\n"
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
