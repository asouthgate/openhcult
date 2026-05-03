from __future__ import annotations

import logging
import os
import numpy as np
import emcee

from .calibrator import CordCalibrator
from .exp import ExponentialCordCalibrator, exponential_target
from .plot_style import apply_dark_theme, CLOUD_BLUE, ORANGE

_logger = logging.getLogger(__name__)


def mcmc_log_prior(f0, xmin, xmin_low, xmin_high, f_int_min, f_int_max):
    p = 0.0
    if not (xmin_low <= xmin <= xmin_high):
        p = -np.inf
    if f0 < f_int_min or f0 > f_int_max:
        p = -np.inf
    return p


def mcmc_log_likelihood(
    theta,
    x_anchors,
    swc_anchors,
    x_starts,
    x_ends,
    delta_swc,
    prior_x,
    prior_y,
    xmax,
    sigma_anchor,
    sigma_prior,
):
    scale, k, f_int, log_sigma, xmin = theta
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
    log_mu_c = np.log(mu_c) - sigma**2 / 2
    ll += np.sum(
        -0.5 * ((np.log(delta_swc) - log_mu_c) / sigma) ** 2
        - log_sigma
        - np.log(delta_swc)
        - 0.5 * np.log(2 * np.pi)
    )

    g_norm = exponential_target(prior_x, k, f_int, xmin, xmax)
    ll -= 0.5 * np.sum(
        ((prior_y - g_norm) / sigma_prior) ** 2 + np.log(2 * np.pi * sigma_prior**2)
    )
    return ll


def _scale_prior(s):
    return 1.0  # improper flat prior on scale; could be changed to something else if desired


def mcmc_log_joint(
    theta,
    x_anchors,
    swc_anchors,
    x_starts,
    x_ends,
    delta_swc,
    prior_x,
    prior_y,
    xmax,
    xmin_low,
    xmin_high,
    sigma_anchor,
    sigma_prior,
    f_int_min,
    f_int_max,
    n_sensors,
    sensor_chord_labels,
):
    assert len(sensor_chord_labels) == len(
        x_starts
    ), f"sensor_chord_labels must have the same length as x_starts, got {len(sensor_chord_labels)} vs {len(x_starts)}"
    s = theta[0]
    lps = _scale_prior(s)
    ll = 0.0
    dim = 4
    for sj in range(n_sensors):
        theta_j = theta[dim * sj + 1 : dim * sj + dim + 1]
        x_starts_j = x_starts[sensor_chord_labels == sj]
        x_ends_j = x_ends[sensor_chord_labels == sj]
        delta_swc_j = delta_swc[sensor_chord_labels == sj]
        f0 = theta_j[1]
        xmin = theta_j[3]
        _, f0, _, xmin = theta_j
        lpj = mcmc_log_prior(f0, xmin, xmin_low, xmin_high, f_int_min, f_int_max)
        if not np.isfinite(lpj):
            return -np.inf
        llj = mcmc_log_likelihood(
            [s] + list(theta_j),
            x_anchors,
            swc_anchors,
            x_starts_j,
            x_ends_j,
            delta_swc_j,
            prior_x,
            prior_y,
            xmax,
            sigma_anchor,
            sigma_prior,
        )
        ll += lpj + llj

    return lps + ll if np.isfinite(ll) else -np.inf


def samples_at(x, scale_s, k_s, f_int_s, xmin_arr, xmax):
    x = np.atleast_1d(x)
    g = exponential_target(
        x[None, :], k_s[:, None], f_int_s[:, None], xmin_arr[:, None], xmax
    )
    result = scale_s[:, None] * g
    out_of_domain = x[None, :] < xmin_arr[:, None]
    result[out_of_domain] = np.nan
    return result


def samples_sequences(x, scale_s, k_s, f_int_s, xmin_arr, xmax):
    """For each sample of parameters, compute the full sequence of f(x) = swc_est.

    This function returns, for N samples, N sequences f(x, theta_i), where theta_i are the sample parameters.
    """
    # For each sample of parameters, and sequence x
    # Predict f(x, theta_i), f(x, theta_i+1), to get full
    mapped_samples = []
    n_samples = len(scale_s)
    assert (
        len(k_s) == n_samples
        and len(f_int_s) == n_samples
        and len(xmin_arr) == n_samples
    )
    for i in range(n_samples):
        scale = scale_s[i]
        k = k_s[i]
        f_int = f_int_s[i]
        xmin = xmin_arr[i]
        g = exponential_target(x, k, f_int, xmin, xmax)
        f_x = scale * g
        mapped_samples.append(f_x)
    return mapped_samples


def samples_mean(x, scale_s, k_s, f_int_s, xmin_arr, xmax):
    vals = samples_at(x, scale_s, k_s, f_int_s, xmin_arr, xmax)
    with np.errstate(all="ignore"):
        return np.nanmean(vals, axis=0)


def samples_ci_low(x, scale_s, k_s, f_int_s, xmin_arr, xmax):
    vals = samples_at(x, scale_s, k_s, f_int_s, xmin_arr, xmax)
    with np.errstate(all="ignore"):
        return np.nanpercentile(vals, 2.5, axis=0)


def samples_ci_high(x, scale_s, k_s, f_int_s, xmin_arr, xmax):
    vals = samples_at(x, scale_s, k_s, f_int_s, xmin_arr, xmax)
    with np.errstate(all="ignore"):
        return np.nanpercentile(vals, 97.5, axis=0)


class ExponentialCordCalibratorMCMC(CordCalibrator):
    def __init__(
        self,
        xmin_low,
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
        f_int_max=0.3,
        f_int_min=0.0,
        debug=True,
        n_sensors=1,
    ):
        super().__init__()
        self._debug = debug
        self._xmin_low = xmin_low
        self._xmin_high = xmin_high
        assert (
            self._xmin_high > self._xmin_low
        ), "xmin_high must be greater than xmin_low"
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

        self._init_spread_scale = init_spread * 1.0
        self._init_spread_k = init_spread
        self._init_spread_f_int = init_spread * 1.0
        self._init_spread_sigma = init_spread

        self.f_int_min = f_int_min
        self.f_int_max = f_int_max

        self.n_sensors = n_sensors

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
        xmax = self._xmax

        assert min(x_starts) <= xmax, (
            f"chord x_starts (min={min(x_starts):.1f}) exceed model xmax ({xmax:.1f}); "
            f"sensor readings are outside the calibration domain"
        )
        assert max(x_starts) >= self._xmin_low, (
            f"chord x_starts (max={max(x_starts):.1f}) are below model xmin_low ({self._xmin_low:.1f}); "
            f"sensor readings are outside the calibration domain"
        )

        xmin_low = self._xmin_low
        xmin_high = self._xmin_high
        data_min_x = min(min(x_starts), min(x_starts + delta_x), min(x_anchors))
        if xmin_low >= data_min_x:
            xmin_low = data_min_x - 1
            _logger.warning(f"xmin_low {self._xmin_low} is greater than or equal to \
                data minimum x {data_min_x}, adjusting xmin_low to {xmin_low}")
        if xmin_high >= data_min_x:
            xmin_high = data_min_x
            _logger.warning(f"xmin_high {self._xmin_high} is greater than or equal to \
                data minimum x {data_min_x}, adjusting xmin_high to {xmin_high}")

        assert xmin_high <= min(
            x_starts
        ), "xmin_high must be less than or equal to the smallest x_start"
        assert xmin_high <= min(
            x_starts + delta_x
        ), "xmin_high must be less than or equal to the smallest x_start + dx"
        assert xmin_high <= min(
            x_anchors
        ), "xmin_high must be less than or equal to the smallest x_anchor"
        xmin_hat = (xmin_low + xmin_high) / 2.0
        xmax = self._xmax
        sigma_anchor = self._sigma_anchor
        sigma_prior = 1.0 / max(np.sqrt(self._prior_weight), self._sigma_prior_floor)

        quick = ExponentialCordCalibrator(xmin_hat, xmax, self._prior_weight).fit(
            x_anchors, swc_anchors, x_starts, delta_x, delta_swc, prior_x, prior_y
        )
        scale0, k0, f_int0 = quick.scale, quick.k, quick.f_int

        return dict(
            scale0=scale0,
            k0=k0,
            f_int0=f_int0,
            xmin_hat=xmin_hat,
            xmax=xmax,
            xmin_low=xmin_low,
            xmin_high=xmin_high,
            sigma_anchor=sigma_anchor,
            sigma_prior=sigma_prior,
        )

    def _init_walker_pos(self, initial_estimate):
        scale0 = initial_estimate["scale0"]
        k0 = initial_estimate["k0"]
        # f_int0 = data["f_int0"]
        # xmin_hat = data["xmin_hat"]
        xmin_low = initial_estimate["xmin_low"]
        xmin_high = initial_estimate["xmin_high"]

        walker_scale0 = np.clip(
            scale0
            + np.random.uniform(
                -self._init_spread_scale, self._init_spread_scale, self._n_walkers
            ),
            1.0,
            1e10,
        )
        walker_k0 = np.clip(
            k0
            + np.random.uniform(
                -self._init_spread_k, self._init_spread_k, self._n_walkers
            ),
            0.0001,
            10000.0,
        )
        walker_f0 = np.random.uniform(self.f_int_min, self.f_int_max, self._n_walkers)
        walker_sigma0 = np.clip(
            np.log(self._sigma_init)
            + np.random.normal(0.0, self._init_spread_sigma, self._n_walkers),
            np.log(0.001),
            np.log(10.0),
        )
        walker_xmin0 = np.random.uniform(xmin_low, xmin_high, self._n_walkers)
        init_pos = np.array(
            [
                walker_scale0,
                walker_k0,
                walker_f0,
                walker_sigma0,
                walker_xmin0,
            ]
        ).T

        return init_pos

    def _run_sampler(self, pos, posterior_kwargs):
        sampler = emcee.EnsembleSampler(
            self._n_walkers,
            1 + 4 * self.n_sensors,
            mcmc_log_joint,
            kwargs=posterior_kwargs,
        )
        state = sampler.run_mcmc(pos, self._n_burn, progress=False)
        sampler.reset()
        sampler.run_mcmc(state, self._n_steps, progress=False)
        return sampler

    def _postprocess(self, sampler, data):
        samples = sampler.get_chain(flat=True)
        xmax = data["xmax"]

        self._fit_samples = samples
        self.scale = float(np.median(samples[:, 0]))
        self.noise = float(np.exp(np.median(samples[:, 3])))

        log_posts = sampler.get_log_prob(flat=True)
        self.nlml = float(-np.max(log_posts))

        thin = max(1, len(samples) // self._n_thin_target)
        idx = np.arange(0, len(samples), thin)
        self._scale_s = samples[idx, 0]
        self._k_s = samples[idx, 1]
        self._f_int_s = samples[idx, 2]
        self._xmin_arr = samples[idx, 4]

        self._mean = lambda x: samples_mean(
            x, self._scale_s, self._k_s, self._f_int_s, self._xmin_arr, xmax
        )

        # print(np.mean(self._f_int_s), np.mean(self._scale_s), self._mean(self._xmax))
        self._ci_low = lambda x: samples_ci_low(
            x, self._scale_s, self._k_s, self._f_int_s, self._xmin_arr, xmax
        )
        self._ci_high = lambda x: samples_ci_high(
            x, self._scale_s, self._k_s, self._f_int_s, self._xmin_arr, xmax
        )

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

        initial_estimate = self._prepare_fit_data(
            x_anchors, swc_anchors, x_starts, delta_x, delta_swc, prior_x, prior_y
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
            xmin_low=initial_estimate["xmin_low"],
            xmin_high=initial_estimate["xmin_high"],
            sigma_anchor=initial_estimate["sigma_anchor"],
            sigma_prior=initial_estimate["sigma_prior"],
            f_int_min=self.f_int_min,
            f_int_max=self.f_int_max,
            n_sensors=self.n_sensors,
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

    def posterior_samples_at(self, x, n=None):
        x = np.atleast_1d(x)
        if n is not None and n < len(self._scale_s):
            idx = np.random.choice(len(self._scale_s), n, replace=False)
            return samples_at(
                x,
                self._scale_s[idx],
                self._k_s[idx],
                self._f_int_s[idx],
                self._xmin_arr[idx],
                self._xmax,
            )
        return samples_at(
            x,
            self._scale_s,
            self._k_s,
            self._f_int_s,
            self._xmin_arr,
            self._xmax,
        )

    def posterior_sequences(self, x, n=None):
        x = np.atleast_1d(x)
        params_d = self.posterior_params(n=n)
        return samples_sequences(
            x,
            params_d["scale"],
            params_d["k"],
            params_d["f_int"],
            params_d["xmin"],
            self._xmax,
        )

    def estimate_velocity_samples(self, v_window, t_window):
        """Estimate velocity in a time window"""

        swc_samples = self.posterior_sequences(v_window)

        t_ref = t_window[0]
        t_norm = t_window - t_ref

        slopes = []
        intercepts = []

        for sample in swc_samples:
            if np.any(np.isnan(sample)) or np.any(np.isinf(sample)):
                continue  # Skip this sample
            m, c = np.polyfit(t_norm, sample, 1)
            slopes.append(m)
            intercepts.append(c - m * t_ref)

        return np.array(slopes), np.array(intercepts), t_window

    def posterior_params(self, n=None):
        """Return dict of posterior parameter samples.

        If n is given, subsample to at most n samples.
        Keys: scale, k, f_int, sigma2, xmin.
        """
        s = self._fit_samples
        if n is not None and n < len(s):
            idx = np.random.choice(len(s), n, replace=False)
            s = s[idx]
        return dict(
            scale=s[:, 0],
            k=s[:, 1],
            f_int=s[:, 2],
            sigma2=np.exp(2 * s[:, 3]),
            xmin=s[:, 4],
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
            f"  Input params: xmin_low_orig={self._xmin_low} xmin_high_orig={self._xmin_high}"
            f" xmax={self._xmax} n_walkers={self._n_walkers}"
            f" n_burn={self._n_burn} n_steps={self._n_steps}"
            f" prior_weight={self._prior_weight}\n"
            f"  Adjusted bounds: xmin_low={data['xmin_low']} xmin_high={data['xmin_high']}\n"
            f"  Data summary: N_chords={len(xs)} data_min_x={dmin}\n"
            f"    x_starts:  min={xs.min():.4f} max={xs.max():.4f}\n"
            f"    x_ends:    min={xe.min():.4f} max={xe.max():.4f}\n"
            f"    delta_swc: min={dswc.min():.4f} max={dswc.max():.4f}\n"
            f"    x_anchors: min={x_anc.min():.4f} max={x_anc.max():.4f}\n"
            f"    prior_x:   min={prior_min} max={prior_max}\n"
            f"  Walker init: cond={np.linalg.cond(pos):.2f}\n"
            f"    per-col min: {pos.min(axis=0).tolist()}\n"
            f"    per-col max: {pos.max(axis=0).tolist()}\n"
            f"    per-col std: {pos.std(axis=0).tolist()}"
        )


def plot_corner(cal, out=None, title=None):
    import matplotlib.pyplot as plt

    apply_dark_theme()
    params = cal.posterior_params()
    param_names = ["scale", "k", "f_int", "sigma2", "xmin"]
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i][j]
            pi = params[param_names[i]]
            pj = params[param_names[j]]
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
