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
    _truncated_normal_logpdf,
)

_logger = logging.getLogger(__name__)


def mcmc_log_anchor_prior_likelihood(
    scale,
    k_all,
    f_int_all,
    xmin_all,
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
    xmin_all = np.atleast_1d(xmin_all)

    g_anchor_parts = np.stack(
        [
            exponential_target(x_anchors, k_all[j], f_int_all[j], xmin_all[j], xmax)
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
                exponential_target(prior_x, k_all[j], f_int_all[j], xmin_all[j], xmax)
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
    xmin,
    x_starts,
    x_ends,
    delta_swc,
    xmax,
):
    sigma = np.exp(log_sigma)
    mu_c = scale * (
        exponential_target(x_ends, k, f_int, xmin, xmax)
        - exponential_target(x_starts, k, f_int, xmin, xmax)
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
    xmin_mu,
    xmin_sigma,
    xmin_high,
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
    xmin_all = theta[1 + 2 * n + 1 : 1 + 3 * n + 1]

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
        if xmin_all[j] > xmin_high[j]:
            return -np.inf
        lp += _truncated_normal_logpdf(
            xmin_all[j], xmin_mu[j], xmin_sigma[j], xmin_high[j]
        )
        if not np.isfinite(lp):
            return -np.inf
    lp += priors.log_sigma_log_prior(log_sigma)
    if not np.isfinite(lp):
        return -np.inf

    ll = mcmc_log_anchor_prior_likelihood(
        scale,
        k_all,
        f_int_all,
        xmin_all,
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
            xmin_all[j],
            x_starts[mask],
            x_ends[mask],
            delta_swc[mask],
            xmax,
        )

    return lp + ll


def samples_swc_at(x, scale_s, k_s, f_int_s, xmin_arr, xmax):
    x = np.atleast_1d(x)
    scale_s = np.array(scale_s).reshape(-1, 1)
    k_s = np.atleast_2d(k_s)
    f_int_s = np.atleast_2d(f_int_s)
    xmin_arr = np.atleast_2d(xmin_arr)

    x_3d = x[None, :, None]  # (1, NX, 1)
    k_3d = k_s[:, None, :]  # (NSamples, 1, NSensors)
    f_3d = f_int_s[:, None, :]  # (NSamples, 1, NSensors)
    xmin_3d = xmin_arr[:, None, :]  # (NSamples, 1, NSensors)
    g_s = exponential_target(x_3d, k_3d, f_3d, xmin_3d, xmax)
    g_s[x_3d < xmin_3d] = np.nan

    g = np.nanmean(g_s, axis=2)
    swc = scale_s * g
    return swc


def samples_mean(x, scale_s, k_s, f_int_s, xmin_arr, xmax):
    vals = samples_swc_at(x, scale_s, k_s, f_int_s, xmin_arr, xmax)
    with np.errstate(all="ignore"):
        return np.nanmean(vals, axis=0)


def samples_ci_low(x, scale_s, k_s, f_int_s, xmin_arr, xmax):
    vals = samples_swc_at(x, scale_s, k_s, f_int_s, xmin_arr, xmax)
    with np.errstate(all="ignore"):
        return np.nanpercentile(vals, 2.5, axis=0)


def samples_ci_high(x, scale_s, k_s, f_int_s, xmin_arr, xmax):
    vals = samples_swc_at(x, scale_s, k_s, f_int_s, xmin_arr, xmax)
    with np.errstate(all="ignore"):
        return np.nanpercentile(vals, 97.5, axis=0)


class ExponentialCordCalibratorMCMC(CordCalibrator):
    def __init__(
        self,
        xmin_mu,
        xmin_sigma,
        xmin_high,
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
    ):
        super().__init__()
        self._debug = debug
        self._xmin_mu = xmin_mu
        self._xmin_sigma = xmin_sigma
        self._xmin_high = xmin_high
        assert self._xmin_sigma > 0, "xmin_sigma must be positive"
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
        self._xmin_arr = None

    def _compute_mean(self, x):
        return samples_mean(
            x, self._scale_s, self._k_s, self._f_int_s, self._xmin_arr, self._xmax
        )

    def _compute_ci_low(self, x):
        return samples_ci_low(
            x, self._scale_s, self._k_s, self._f_int_s, self._xmin_arr, self._xmax
        )

    def _compute_ci_high(self, x):
        return samples_ci_high(
            x, self._scale_s, self._k_s, self._f_int_s, self._xmin_arr, self._xmax
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

        xmin_mu_arr = np.full(self.n_sensors, self._xmin_mu, dtype=float)
        xmin_sigma_arr = np.full(self.n_sensors, self._xmin_sigma, dtype=float)
        xmin_high_arr = np.full(self.n_sensors, self._xmin_high, dtype=float)
        x_ends = x_starts + delta_x

        for sj in range(self.n_sensors):
            mask = sensor_chord_labels == sj
            xs_j = x_starts[mask]
            xe_j = x_ends[mask]
            data_min_x_j = min(min(xs_j), min(xe_j), min(x_anchors))
            if xmin_high_arr[sj] >= data_min_x_j:
                xmin_high_arr[sj] = data_min_x_j
                _logger.warning(
                    f"xmin_high {self._xmin_high} is greater than or equal to "
                    f"data minimum x {data_min_x_j} for sensor {sj}, adjusting xmin_high to {xmin_high_arr[sj]}"
                )

            assert xmin_high_arr[sj] <= min(
                xs_j
            ), f"xmin_high[{sj}] must be less than or equal to the smallest x_start for sensor {sj}"
            assert xmin_high_arr[sj] <= min(
                xe_j
            ), f"xmin_high[{sj}] must be less than or equal to the smallest x_end for sensor {sj}"
            assert xmin_high_arr[sj] <= min(
                x_anchors
            ), f"xmin_high[{sj}] must be less than or equal to the smallest x_anchor"
        xmin_hat_arr = np.clip(xmin_mu_arr, None, xmin_high_arr)
        xmax = self._xmax
        sigma_anchor = self._sigma_anchor
        sigma_prior = 1.0 / max(np.sqrt(self._prior_weight), self._sigma_prior_floor)

        k0s = []
        f_int0s = []
        scale0s = []
        for sj in range(self.n_sensors):
            mask = sensor_chord_labels == sj
            quick = ExponentialCordCalibrator(
                xmin_hat_arr[sj], xmax, self._prior_weight
            )
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
            xmin_hat=xmin_hat_arr,
            xmax=xmax,
            xmin_mu=xmin_mu_arr,
            xmin_sigma=xmin_sigma_arr,
            xmin_high=xmin_high_arr,
            sigma_anchor=sigma_anchor,
            sigma_prior=sigma_prior,
        )

    def _init_walker_pos(self, initial_estimate):
        scale0 = initial_estimate["scale0"]
        k0 = initial_estimate["k0"]
        f_int0 = initial_estimate["f_int0"]
        xmin_hat = initial_estimate["xmin_hat"]
        xmin_high = initial_estimate["xmin_high"]

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
        for sj in range(self.n_sensors):
            xhat_j = xmin_hat[sj]
            walker_xmin_j = np.clip(
                xhat_j + np.random.normal(0, 0.01, self._n_walkers),
                None,
                xmin_high[sj],
            )
            cols.append(walker_xmin_j)

        init_pos = np.array(cols).T
        return init_pos

    def _run_sampler(self, pos, posterior_kwargs):
        sampler = emcee.EnsembleSampler(
            self._n_walkers,
            2 + 3 * self.n_sensors,
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
        self._xmin_arr = post_flat[idx][:, [1 + 2 * n + 1 + j for j in range(n)]]

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
            xmin_mu=initial_estimate["xmin_mu"],
            xmin_sigma=initial_estimate["xmin_sigma"],
            xmin_high=initial_estimate["xmin_high"],
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

    def _warn_low_prob_xmin(self, x, prob_x, threshold=0.05):
        low_mask = prob_x < threshold
        if np.any(low_mask):
            low_x = x[low_mask]
            low_p = prob_x[low_mask]
            parts = [f"{xi:.4f} (P={pi:.1%})" for xi, pi in zip(low_x, low_p)]
            _logger.warning(
                f"x values below {threshold:.0%} posterior probability of being >= xmin: "
                f"{', '.join(parts)}"
            )

    def prob_xmin(self, x):
        """Return P(x >= xmin) for each x value.

        This is the fraction of posterior samples where the calibration
        curve is defined at the given x (i.e., the sampled xmin <= x).
        """
        vals = self.posterior_samples_swc_at(np.atleast_1d(x))
        return np.mean(~np.isnan(vals), axis=0)

    def predict(self, x, return_prob_x=False):
        x = np.atleast_1d(x)
        vals = self.posterior_samples_swc_at(x)
        prob_x = np.mean(~np.isnan(vals), axis=0)
        self._warn_low_prob_xmin(x, prob_x)
        with np.errstate(all="ignore"):
            mean = np.nanmean(vals, axis=0)
            ci_low = np.nanpercentile(vals, 2.5, axis=0)
            ci_high = np.nanpercentile(vals, 97.5, axis=0)
        if return_prob_x:
            return mean, ci_low, ci_high, prob_x
        return mean, ci_low, ci_high

    def __call__(self, x):
        x = np.atleast_1d(x)
        vals = self.posterior_samples_swc_at(x)
        prob_x = np.mean(~np.isnan(vals), axis=0)
        self._warn_low_prob_xmin(x, prob_x)
        with np.errstate(all="ignore"):
            return np.nanmean(vals, axis=0)

    def posterior_samples_swc_at(self, x, n=None):
        x = np.atleast_1d(x)
        if n is not None and n < len(self._scale_s):
            idx = np.random.choice(len(self._scale_s), n, replace=False)
            return samples_swc_at(
                x,
                self._scale_s[idx],
                self._k_s[idx],
                self._f_int_s[idx],
                self._xmin_arr[idx],
                self._xmax,
            )
        return samples_swc_at(
            x,
            self._scale_s,
            self._k_s,
            self._f_int_s,
            self._xmin_arr,
            self._xmax,
        )

    def curve_credible_region(self, x_grid, alpha=0.95, n_bins=200):
        """Compute the (alpha*100)% 2D credible region in (x, SWC) space.

        Uses 2D histogram density estimation over the posterior curve samples
        and extracts the highest-density contour containing `alpha` fraction of
        the posterior density mass.

        The credible region naturally handles xmin uncertainty: regions where
        few samples have x >= xmin will have lower density and fall outside
        high-probability contours.

        Parameters:
            x_grid: array of x values to evaluate curves on
            alpha: coverage level (default 0.95)
            n_bins: number of histogram bins per dimension (default 200)

        Returns:
            contour_paths: list of (N, 2) numpy arrays, each a boundary
                           segment in (x, SWC) coordinates
            level: float, the density threshold used
        """
        x_grid = np.atleast_1d(x_grid)
        vals = self.posterior_samples_swc_at(x_grid)

        n_samples = vals.shape[0]
        x_flat = np.tile(x_grid, n_samples)
        swc_flat = vals.ravel()
        valid = ~np.isnan(swc_flat)
        points = np.column_stack([x_flat[valid], swc_flat[valid]])

        if len(points) == 0:
            _logger.warning(
                "No valid (x, SWC) points for credible region; "
                "all sampled xmin > max(x_grid)"
            )
            return [], 0.0

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
              f_int (n_samples, n_sensors), sigma2 (n_samples,),
              xmin (n_samples, n_sensors).
        """
        s = self._fit_samples
        if n is not None and n < len(s):
            idx = np.random.choice(len(s), n, replace=False)
            s = s[idx]
        n_s = self.n_sensors
        k_cols = [1 + j for j in range(n_s)]
        f_cols = [1 + n_s + j for j in range(n_s)]
        sigma_col = 1 + 2 * n_s
        xmin_cols = [1 + 2 * n_s + 1 + j for j in range(n_s)]
        return dict(
            scale=np.exp(s[:, 0]),
            k=s[:, k_cols],
            f_int=s[:, f_cols],
            sigma2=np.exp(2 * s[:, sigma_col]),
            xmin=s[:, xmin_cols],
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
            f"  Input params: xmin_mu={self._xmin_mu} xmin_sigma={self._xmin_sigma} xmin_high_orig={self._xmin_high}"
            f" xmax={self._xmax} n_walkers={self._n_walkers}"
            f" n_burn={self._n_burn} n_steps={self._n_steps}"
            f" prior_weight={self._prior_weight}\n"
            f"  Adjusted bounds: xmin_mu={data['xmin_mu']} xmin_high={data['xmin_high']}\n"
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
    param_names = ["scale", "k", "f_int", "sigma2", "xmin"]
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
