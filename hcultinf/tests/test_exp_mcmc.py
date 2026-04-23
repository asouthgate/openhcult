import numpy as np
import pytest

from hcultinf.exp import exponential_target
from hcultinf.exp_mcmc import (
    mcmc_log_prior,
    mcmc_log_likelihood,
    mcmc_log_posterior,
    samples_at,
    samples_mean,
    samples_ci_low,
    samples_ci_high,
    ExponentialCordCalibratorMCMC,
)

XMIN = 3.0
XMAX = 8.5


def _exp_func(x):
    return 10.0 * exponential_target(x, k=20.0, f_int=0.05, xmin=XMIN, xmax=XMAX)


def _simple_data():
    x_starts = np.array([5.0])
    delta_x = np.array([-0.5])
    x_ends = x_starts + delta_x
    delta_swc = _exp_func(x_ends) - _exp_func(x_starts)
    x_anchors = np.array([XMAX])
    swc_anchors = np.array([_exp_func(XMAX)])
    prior_x = np.array([XMIN, XMAX])
    prior_y = np.array([1.0, 0.0])
    return dict(
        x_anchors=x_anchors,
        swc_anchors=swc_anchors,
        x_starts=x_starts,
        x_ends=x_ends,
        delta_swc=delta_swc,
        prior_x=prior_x,
        prior_y=prior_y,
    )


class TestMcmcLogPrior:
    def test_uniform_in_bounds(self):
        theta = np.array([1.0, 20.0, 0.1, -1.0, 3.5])
        assert mcmc_log_prior(theta, xmin_low=1.0, xmin_high=5.0) == 0.0

    def test_uniform_out_of_bounds_low(self):
        theta = np.array([1.0, 20.0, 0.1, -1.0, 0.5])
        assert mcmc_log_prior(theta, xmin_low=1.0, xmin_high=5.0) == -np.inf

    def test_uniform_out_of_bounds_high(self):
        theta = np.array([1.0, 20.0, 0.1, -1.0, 5.5])
        assert mcmc_log_prior(theta, xmin_low=1.0, xmin_high=5.0) == -np.inf

    def test_at_bounds(self):
        theta_low = np.array([1.0, 20.0, 0.1, -1.0, 1.0])
        theta_high = np.array([1.0, 20.0, 0.1, -1.0, 5.0])
        assert mcmc_log_prior(theta_low, xmin_low=1.0, xmin_high=5.0) == 0.0
        assert mcmc_log_prior(theta_high, xmin_low=1.0, xmin_high=5.0) == 0.0


class TestMcmcLogLikelihood:
    def test_returns_finite_for_reasonable_theta(self):
        data = _simple_data()
        theta = np.array([10.0, 20.0, 0.05, np.log(0.3), XMIN])
        ll = mcmc_log_likelihood(
            theta,
            sigma_anchor=0.1,
            sigma_prior=1.0,
            xmax=XMAX,
            **data,
        )
        assert np.isfinite(ll)

    def test_returns_neg_inf_when_mu_c_nonpositive(self):
        data = _simple_data()
        reversed_data = {**data, "x_starts": data["x_ends"], "x_ends": data["x_starts"]}
        theta = np.array([10.0, 20.0, 0.05, np.log(0.3), XMIN])
        ll = mcmc_log_likelihood(
            theta,
            sigma_anchor=0.1,
            sigma_prior=1.0,
            xmax=XMAX,
            **reversed_data,
        )
        assert ll == -np.inf

    def test_likelihood_higher_at_true_params_than_random(self):
        data = _simple_data()
        theta_true = np.array([10.0, 20.0, 0.05, np.log(0.05), XMIN])
        theta_bad = np.array([0.1, 0.5, 0.25, np.log(5.0), XMIN])
        ll_true = mcmc_log_likelihood(
            theta_true,
            sigma_anchor=0.1,
            sigma_prior=1.0,
            xmax=XMAX,
            **data,
        )
        ll_bad = mcmc_log_likelihood(
            theta_bad,
            sigma_anchor=0.1,
            sigma_prior=1.0,
            xmax=XMAX,
            **data,
        )
        assert ll_true > ll_bad


class TestMcmcLogPosterior:
    def test_equals_prior_plus_likelihood(self):
        data = _simple_data()
        theta = np.array([10.0, 20.0, 0.05, np.log(0.3), XMIN])
        ll_kwargs = dict(
            sigma_anchor=0.1,
            sigma_prior=1.0,
            xmax=XMAX,
            **data,
        )
        lp = mcmc_log_prior(theta, xmin_low=1.0, xmin_high=5.0)
        ll = mcmc_log_likelihood(theta, **ll_kwargs)
        expected = lp + ll
        actual = mcmc_log_posterior(
            theta,
            sigma_anchor=0.1,
            sigma_prior=1.0,
            xmax=XMAX,
            xmin_low=1.0,
            xmin_high=5.0,
            **data,
        )
        np.testing.assert_allclose(actual, expected)

    def test_returns_neg_inf_if_likelihood_neg_inf(self):
        data = _simple_data()
        reversed_data = {**data, "x_starts": data["x_ends"], "x_ends": data["x_starts"]}
        theta = np.array([10.0, 20.0, 0.05, np.log(0.3), XMIN])
        lp = mcmc_log_posterior(
            theta,
            sigma_anchor=0.1,
            sigma_prior=1.0,
            xmax=XMAX,
            xmin_low=1.0,
            xmin_high=5.0,
            **reversed_data,
        )
        assert lp == -np.inf

    def test_returns_neg_inf_if_xmin_out_of_uniform_bounds(self):
        data = _simple_data()
        theta = np.array([10.0, 20.0, 0.05, np.log(0.3), 6.0])
        lp = mcmc_log_posterior(
            theta,
            sigma_anchor=0.1,
            sigma_prior=1.0,
            xmax=XMAX,
            xmin_low=1.0,
            xmin_high=5.0,
            **data,
        )
        assert lp == -np.inf


class TestSamplesAt:
    def test_matches_exponential_target(self):
        scale_s = np.array([2.0, 2.5])
        k_s = np.array([20.0, 15.0])
        f_int_s = np.array([0.05, 0.1])
        xmin_arr = np.array([XMIN, XMIN])
        x = np.array([5.0, 7.0])

        result = samples_at(x, scale_s, k_s, f_int_s, xmin_arr, XMAX)
        assert result.shape == (2, 2)

        for i in range(2):
            expected = scale_s[i] * exponential_target(
                x, k_s[i], f_int_s[i], XMIN, XMAX
            )
            np.testing.assert_allclose(result[i], expected, rtol=1e-10)

    def test_mean_matches_manual(self):
        scale_s = np.array([2.0, 2.0])
        k_s = np.array([20.0, 20.0])
        f_int_s = np.array([0.05, 0.05])
        xmin_arr = np.array([XMIN, XMIN])
        x = np.array([5.0])
        result = samples_mean(x, scale_s, k_s, f_int_s, xmin_arr, XMAX)
        expected = scale_s[0] * exponential_target(5.0, 20.0, 0.05, XMIN, XMAX)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_ci_bounds_enclose_mean(self):
        scale_s = np.array([2.0, 2.2, 1.8, 2.1, 1.9])
        k_s = np.array([20.0, 22.0, 18.0, 21.0, 19.0])
        f_int_s = np.array([0.05, 0.07, 0.03, 0.06, 0.04])
        xmin_arr = np.full(5, XMIN)
        x = np.array([4.0, 6.0])
        low = samples_ci_low(x, scale_s, k_s, f_int_s, xmin_arr, XMAX)
        high = samples_ci_high(x, scale_s, k_s, f_int_s, xmin_arr, XMAX)
        mean = samples_mean(x, scale_s, k_s, f_int_s, xmin_arr, XMAX)
        assert np.all(low <= mean + 1e-10)
        assert np.all(high >= mean - 1e-10)
        assert np.all(low < high)

    def test_single_x_scalar_input(self):
        scale_s = np.array([2.0])
        k_s = np.array([20.0])
        f_int_s = np.array([0.05])
        xmin_arr = np.array([XMIN])
        result = samples_at(5.0, scale_s, k_s, f_int_s, xmin_arr, XMAX)
        expected = 2.0 * exponential_target(np.array([5.0]), 20.0, 0.05, XMIN, XMAX)
        np.testing.assert_allclose(result.ravel(), expected, rtol=1e-10)


# class TestXminRecovery:
#     def test_recovers_xmin_with_varying_xmin(self):
#         rng = np.random.default_rng(42)
#         true_xmin = 3.0
#         true_k = 20.0
#         true_f_int = 0.05
#         true_scale = 10.0
#         data = _make_synthetic(
#             true_xmin,
#             XMAX,
#             true_k,
#             true_f_int,
#             true_scale,
#             n_anchors=10,
#             n_chords=30,
#             noise_level=0.05,
#             rng=rng,
#         )
#         cal = ExponentialCordCalibratorMCMC(
#             xmin_low=1.0,
#             xmin_high=4.5,
#             xmax=XMAX,
#             prior_weight=1.0,
#             n_burn=200,
#             n_steps=400,
#         )
#         cal.fit(**data)
#         samples = cal._fit_samples
#         xmin_samples = samples[:, 4]
#         median_xmin = float(np.median(xmin_samples))
#         assert (
#             abs(median_xmin - true_xmin) < 1.0
#         ), f"xmin not recovered: median={median_xmin:.2f}, true={true_xmin}"


# class TestFIntRecovery:
#     def test_recovers_f_int_with_clear_floor(self):
#         rng = np.random.default_rng(42)
#         true_k = 20.0
#         true_f_int = 0.15
#         true_scale = 10.0
#         true_xmin = 3.0
#         data = _make_synthetic(
#             true_xmin,
#             XMAX,
#             true_k,
#             true_f_int,
#             true_scale,
#             n_anchors=10,
#             n_chords=30,
#             noise_level=0.05,
#             rng=rng,
#         )
#         cal = ExponentialCordCalibratorMCMC(
#             xmin_low=1.0,
#             xmin_high=5.0,
#             xmax=XMAX,
#             prior_weight=1.0,
#             n_burn=200,
#             n_steps=400,
#         )
#         cal.fit(**data)
#         samples = cal._fit_samples
#         f_int_samples = samples[:, 2]
#         median_f_int = float(np.median(f_int_samples))
#         assert (
#             abs(median_f_int - true_f_int) < 0.1
#         ), f"f_int not recovered: median={median_f_int:.3f}, true={true_f_int}"


def _make_synthetic(xmin, xmax, k, f_int, scale, n_anchors, n_chords, noise_level, rng):
    y = lambda x: scale * exponential_target(x, k, f_int, xmin, xmax)
    g = lambda x: exponential_target(x, k, f_int, xmin, xmax)

    x_anchors = np.linspace(xmin + 0.5, xmax - 0.5, n_anchors)
    swc_anchors = y(x_anchors) + rng.normal(0, noise_level * scale, n_anchors)

    x_starts = rng.uniform(xmin + 0.3, xmax - 1.0, n_chords)
    delta_x = -rng.uniform(0.3, 0.8, n_chords)
    x_ends = np.clip(x_starts + delta_x, xmin, xmax)
    delta_x = x_ends - x_starts

    true_delta_swc = y(x_ends) - y(x_starts)
    delta_swc = true_delta_swc * rng.lognormal(0, noise_level, n_chords)

    prior_x = np.linspace(xmin, xmax, 100)
    prior_y = g(prior_x)

    return dict(
        x_anchors=x_anchors,
        swc_anchors=swc_anchors,
        x_starts=x_starts,
        delta_x=delta_x,
        delta_swc=delta_swc,
        prior_x=prior_x,
        prior_y=prior_y,
    )
