import numpy as np
import pytest

from hcultinf.exp_mcmc import ExponentialCordCalibratorMCMC
from hcultinf.combiner import combine_bayesian, combine_empirical_bayes, fuse_swc
from hcultinf.simulation import simulate_calibration_data_samples

TEST_XMIN = 3.0
TEST_XMAX = 8.5
TEST_DXMAX = 0.5
TEST_NOISE_LEVEL = 0.1


def _fit_mcmc(n=30, seed=None):
    from hcultinf.exp import exponential_target

    if seed is not None:
        np.random.seed(seed)
    test_fn = lambda x: 10.0 * exponential_target(
        x, k=20.0, f_int=0.0, xmin=TEST_XMIN, xmax=TEST_XMAX
    )
    x, dx, dy = simulate_calibration_data_samples(
        TEST_XMIN,
        TEST_XMAX,
        TEST_DXMAX,
        TEST_DXMAX,
        TEST_NOISE_LEVEL,
        n,
        test_fn,
        uniform=True,
    )
    cal = ExponentialCordCalibratorMCMC(
        xmin_low=2.5,
        xmin_high=3.0,
        xmax=TEST_XMAX,
        prior_weight=1.0,
        n_burn=30,
        n_steps=60,
    ).fit(
        np.array([TEST_XMAX]),
        np.array([0.0]),
        x,
        dx,
        dy,
        np.array([TEST_XMIN, TEST_XMAX]),
        np.array([1.0, 0.0]),
    )
    assert np.all(np.isfinite(cal(np.linspace(TEST_XMIN, TEST_XMAX, 10))))
    assert np.all(cal(np.linspace(TEST_XMIN, TEST_XMAX, 10)) >= 0)
    return cal


def _test_combined_result(cal1, cal2, x_grid, mean, lo, hi):
    mean1 = np.asarray(cal1(x_grid))
    mean2 = np.asarray(cal2(x_grid))
    assert np.all(mean >= np.minimum(mean1, mean2) - 1e-6)
    assert np.all(mean <= np.maximum(mean1, mean2) + 1e-6)
    assert np.all(np.isfinite(mean))
    ci_width_combined = np.asarray(hi) - np.asarray(lo)
    _, lo1, hi1 = cal1.predict(x_grid)
    _, lo2, hi2 = cal2.predict(x_grid)
    ci_width_1 = np.asarray(hi1) - np.asarray(lo1)
    ci_width_2 = np.asarray(hi2) - np.asarray(lo2)
    min_individual = np.minimum(ci_width_1, ci_width_2)
    assert np.all(ci_width_combined <= min_individual + 1e-6)


def test_combine_bayesian():
    cal1 = _fit_mcmc(seed=1)
    cal2 = _fit_mcmc(seed=2)
    x_grid = np.linspace(TEST_XMIN + 0.5, TEST_XMAX - 0.5, 50)
    mean, lo, hi = combine_bayesian([cal1, cal2], x_grid, sigma_bias=0.0)
    _test_combined_result(cal1, cal2, x_grid, mean, lo, hi)


def test_combine_empirical_bayes():
    cal1 = _fit_mcmc(seed=1)
    cal2 = _fit_mcmc(seed=2)
    x_grid = np.linspace(TEST_XMIN + 0.5, TEST_XMAX - 0.5, 50)
    mean, lo, hi, sigma_bias = combine_empirical_bayes([cal1, cal2], x_grid)
    assert sigma_bias >= 0
    _test_combined_result(cal1, cal2, x_grid, mean, lo, hi)


def _test_fuse_swc_result(cal1, cal2, v1, v2, result):
    # assert that the resulting estimate is between the two calibrators' predictions where both are available
    pred1 = np.asarray(cal1(v1))
    pred2 = np.asarray(cal2(v2))
    mean = np.asarray(result["mean"])
    valid = ~np.isnan(pred1) & ~np.isnan(pred2)
    assert np.all(mean[valid] >= np.minimum(pred1[valid], pred2[valid]) - 1e-6)
    assert np.all(mean[valid] <= np.maximum(pred1[valid], pred2[valid]) + 1e-6)


def test_fuse_swc_basic():
    cal1 = _fit_mcmc(seed=1)
    cal2 = _fit_mcmc(seed=2)
    n = 20
    rng = np.random.default_rng(42)
    v1 = rng.uniform(TEST_XMIN + 0.5, TEST_XMAX - 0.5, n)
    v2 = rng.uniform(TEST_XMIN + 0.5, TEST_XMAX - 0.5, n)
    result = fuse_swc([cal1, cal2], [v1, v2], sigma_bias=0.0)
    _test_fuse_swc_result(cal1, cal2, v1, v2, result)


def test_fuse_swc_with_missing():
    cal1 = _fit_mcmc(seed=1)
    cal2 = _fit_mcmc(seed=2)
    n = 20
    rng = np.random.default_rng(42)
    v1 = rng.uniform(TEST_XMIN + 0.5, TEST_XMAX - 0.5, n)
    v2 = rng.uniform(TEST_XMIN + 0.5, TEST_XMAX - 0.5, n)
    v2[3] = np.nan
    v2[7] = np.nan
    result = fuse_swc([cal1, cal2], [v1, v2], sigma_bias=0.5)
    assert result["mean"].shape == (n,)
    assert np.all(np.isfinite(result["mean"]))
    assert result["n_sensors"][3] == 1
    assert result["n_sensors"][7] == 1
    assert np.all(result["n_sensors"][(np.arange(n) != 3) & (np.arange(n) != 7)] == 2)
    assert np.all(np.isfinite(result["ci_low"]))
    assert np.all(np.isfinite(result["ci_high"]))
