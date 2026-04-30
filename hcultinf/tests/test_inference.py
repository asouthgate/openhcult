import json
import os
from pathlib import Path

import numpy as np
import pytest

from hcultinf.exp import ExponentialCordCalibrator, exponential_target
from hcultinf.exp_mcmc import ExponentialCordCalibratorMCMC, plot_corner
from hcultinf.combiner import combine_bayesian, combine_empirical_bayes
from hcultinf.drying import linear_drying_rate
from hcultinf.simulation import (
    simulate_calibration_data_samples,
    Y_TEST_FUNCTION,
    power_function,
)

TEST_XMIN = 3.0
TEST_XMAX = 8.5
TEST_DXMIN = 0.2
TEST_DXMAX = 0.5
TEST_NOISE_LEVEL = 0.2

TEST_POWER_FUNCTION = lambda x: 10.0 * power_function(x, 5.0, 0.0, TEST_XMIN, TEST_XMAX)


def _get_mixed_prior_decreasing(xmin, xmax, p):
    priorx = np.linspace(xmin, xmax, 1000)
    linear_prior_y = np.interp(priorx, [TEST_XMIN, TEST_XMAX], [1.0, 0.0])
    fn_vals = TEST_POWER_FUNCTION(priorx)
    normalized_fn = (fn_vals - fn_vals.min()) / (fn_vals.max() - fn_vals.min())
    priory = linear_prior_y * p + normalized_fn * (1 - p)
    return priorx, priory


TEST_POWER_FUNCTION = lambda x: 10.0 * power_function(x, 5.0, 0.0, TEST_XMIN, TEST_XMAX)
TEST_EXPONENTIAL_FUNCTION = lambda x: 10.0 * exponential_target(
    x, k=20.0, f_int=0.0, xmin=TEST_XMIN, xmax=TEST_XMAX
)


@pytest.mark.parametrize(
    "estimator_func_pair",
    [
        (
            ExponentialCordCalibratorMCMC(
                xmin_low=2.5,
                xmin_high=3.0,
                xmax=TEST_XMAX,
                prior_weight=1.0,
                n_burn=30,
                n_steps=60,
            ),
            TEST_EXPONENTIAL_FUNCTION,
        ),
        (
            ExponentialCordCalibrator(TEST_XMIN, TEST_XMAX, 1e-8),
            TEST_EXPONENTIAL_FUNCTION,
        ),
    ],
)
def test_convergence_in_n_bad_prior(estimator_func_pair):
    """Test that, as data increases, the curve estimate approaches the true curve."""
    preverrs = []
    last_pwl = last_x = last_dx = last_dy = None
    estimator, test_function = estimator_func_pair

    anchorx = np.array([TEST_XMAX])
    anchory = np.array([0.0])
    for n in [4, 64]:
        priorx_pts = np.array([TEST_XMIN, TEST_XMAX])
        priory_pts = np.array([1.0, 0.0])
        errs = []
        for _ in range(5):
            x, dx, dy = simulate_calibration_data_samples(
                TEST_XMIN,
                TEST_XMAX,
                TEST_DXMAX,
                TEST_DXMAX,
                TEST_NOISE_LEVEL / 2,
                n,
                test_function,
                uniform=True,
            )
            assert priory_pts.max() <= 1
            pwl = estimator.fit(
                anchorx,
                anchory,
                x,
                dx,
                dy,
                priorx_pts,
                priory_pts,
            )
            pwlx = np.linspace(TEST_XMIN, TEST_XMAX, 1000)
            _curve_error = np.mean(np.abs(test_function(pwlx) - pwl(pwlx)))
            errs.append(_curve_error)
        last_pwl, last_x, last_dx, last_dy = pwl, x, dx, dy
        curve_error = np.mean(errs)
        preverrs.append(curve_error)

    plot_x = np.linspace(TEST_XMIN * 0.75, TEST_XMAX, 500)
    plot_y = np.interp(plot_x, priorx_pts, priory_pts)
    last_pwl.plot(
        plot_x,
        plot_y,
        anchorx,
        anchory,
        last_x,
        last_dx,
        last_dy,
        true_y=test_function(plot_x),
        out=f"artifacts/convergence_n_bad_prior_{estimator.__class__.__name__}.png",
        title=f"Test convergence in n with bad prior ({estimator.__class__.__name__})",
        show_chords_pane=False,
    )

    assert all(
        np.diff(preverrs) < 0
    ), f"Curve error did not decrease with increasing n: {preverrs}"
    assert curve_error < 0.02 * Y_TEST_FUNCTION(pwlx).max()


def test_realistic():
    data_path = Path(__file__).parent / "cord_data.json"
    with open(data_path) as f:
        d = json.load(f)

    x = np.array(d["chords_x"])
    dx = np.array(d["chords_dx"])
    dy = np.array(d["chords_dy"])
    prior_x = np.array(d["prior_x"])
    prior_y = np.array(d["prior_y"])

    xmax = 2009.0
    anchor_x = np.array([xmax])
    anchor_swc = np.array([0.0])

    estimator = ExponentialCordCalibratorMCMC(
        xmin_low=750.0,
        xmin_high=1000.0,
        xmax=xmax,
        n_burn=250,
        n_steps=400,
    )
    cal = estimator.fit(anchor_x, anchor_swc, x, dx, dy, prior_x, prior_y)

    cal.plot(
        prior_x,
        prior_y,
        anchor_x,
        anchor_swc,
        x,
        dx,
        dy,
        out="artifacts/realistic_mcmc.png",
        title="Realistic MCMC calibration",
        show_chords_pane=False,
    )

    import matplotlib.pyplot as plt
    from hcultinf.plot_style import apply_dark_theme, CLOUD_BLUE, ORANGE

    apply_dark_theme()

    mean, ci_low, ci_high = cal.predict(prior_x)
    EST_SWC = 800.0
    assert np.abs(max(mean) - EST_SWC) <= 100
    assert all(np.abs(ci_low - mean) <= 500)
    assert all(np.abs(ci_high - mean) <= 500)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(ci_low))
    assert np.all(np.isfinite(ci_high))
    assert np.all(ci_low <= mean)
    assert np.all(mean <= ci_high)

    plot_corner(
        cal, out="artifacts/realistic_corner.png", title="Realistic MCMC posterior"
    )


@pytest.mark.parametrize(
    "estimator",
    [
        ExponentialCordCalibrator(TEST_XMIN, TEST_XMAX, prior_weight=0.01),
    ],
)
def test_unbiasedness(estimator):
    """Mean prediction across trials should match the true curve pointwise."""
    R = 50
    n = 30
    priorx, priory = _get_mixed_prior_decreasing(TEST_XMIN, TEST_XMAX, 0.5)
    eval_x = np.linspace(TEST_XMIN, TEST_XMAX, 200)
    true_y = TEST_POWER_FUNCTION(eval_x)

    estimates = np.zeros((R, len(eval_x)))
    for i in range(R):
        x, dx, dy = simulate_calibration_data_samples(
            TEST_XMIN,
            TEST_XMAX,
            TEST_DXMIN,
            TEST_DXMAX,
            TEST_NOISE_LEVEL,
            n,
            TEST_POWER_FUNCTION,
            uniform=True,
        )

        pwl = estimator.fit(
            np.array([TEST_XMAX]), np.array([0.0]), x, dx, dy, priorx, priory
        )
        mean = pwl(eval_x)
        estimates[i] = mean

    mean_pred = estimates.mean(axis=0)
    std_of_means = estimates.std(axis=0) / np.sqrt(R)

    import matplotlib.pyplot as plt

    from hcultinf.plot_style import apply_dark_theme, CLOUD_BLUE, CLOUD_WHITE

    apply_dark_theme()

    x_pad = (TEST_XMAX - TEST_XMIN) * 0.15
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eval_x, true_y, label="true", color=CLOUD_WHITE)
    ax.plot(
        eval_x, mean_pred, label="mean prediction", linestyle="--", color=CLOUD_BLUE
    )
    ax.fill_between(
        eval_x,
        mean_pred - 1.96 * std_of_means,
        mean_pred + 1.96 * std_of_means,
        alpha=0.3,
        color=CLOUD_BLUE,
        label="95% CI on mean",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(TEST_XMIN - x_pad, TEST_XMAX + x_pad)
    ax.legend()
    fig.tight_layout()
    fig.savefig("artifacts/unbiasedness.png")
    if os.environ.get("HCULT_TEST_DEBUG_PLOT", "0") == "1":
        plt.show()
    plt.close(fig)

    bias = np.abs(mean_pred - true_y).mean()
    assert (
        bias < 0.05 * true_y.max()
    ), f"Mean absolute bias {bias:.4f} exceeds threshold"


def _fit_mcmc(n=30, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x, dx, dy = simulate_calibration_data_samples(
        TEST_XMIN,
        TEST_XMAX,
        TEST_DXMAX,
        TEST_DXMAX,
        TEST_NOISE_LEVEL / 2,
        n,
        TEST_EXPONENTIAL_FUNCTION,
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
    # assert that the results non-trivial, non zero and non inf
    assert np.all(np.isfinite(cal(np.linspace(TEST_XMIN, TEST_XMAX, 10))))
    assert np.all(cal(np.linspace(TEST_XMIN, TEST_XMAX, 10)) >= 0)
    return cal


def test_std_consistent_with_ci():
    cal = _fit_mcmc(seed=42)
    x_grid = np.linspace(TEST_XMIN, TEST_XMAX, 50)
    mean, ci_low, ci_high = cal.predict(x_grid)
    std = cal.std(x_grid)
    ci_width = np.asarray(ci_high) - np.asarray(ci_low)
    expected_width = 2 * 1.96 * np.asarray(std)
    np.testing.assert_allclose(ci_width, expected_width, rtol=1e-6)


def test_posterior_samples_at():
    cal = _fit_mcmc(seed=42)
    x_grid = np.linspace(TEST_XMIN + 0.5, TEST_XMAX - 0.5, 20)
    samples = cal.posterior_samples_at(x_grid, n=50)
    assert samples.shape[0] == 50
    assert samples.shape[1] == len(x_grid)
    assert np.all(np.isfinite(samples[~np.isnan(samples)]))
    sample_mean = np.nanmean(samples, axis=0)[cal._n_burn :]
    pred_mean = np.asarray(cal(x_grid))
    valid = ~np.isnan(sample_mean)
    assert all(sample_mean[valid] == pred_mean[valid])


def _test_combined_result(cal1, cal2, x_grid, mean, lo, hi):
    mean1 = np.asarray(cal1(x_grid))
    mean2 = np.asarray(cal2(x_grid))
    # assert combined mean is always between the two individual means
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


def test_linear_drying_rate():
    cal = _fit_mcmc(seed=42)
    x_grid = np.linspace(TEST_XMIN, TEST_XMAX, 50)
    true_y = TEST_EXPONENTIAL_FUNCTION(x_grid)
    true_rate = true_y[-1] - true_y[0]
    n = 20
    t_days = np.linspace(0, 1, n)
    t_ms = t_days * 24 * 3600 * 1000
    voltages = np.linspace(TEST_XMIN + 0.5, TEST_XMAX - 0.5, n)
    result = linear_drying_rate(cal, t_ms, voltages)
    assert result is not None
    assert result["n_points"] == n
    assert np.isfinite(result["rate_ml_per_day"])
    assert result["rate_ci_low"] <= result["rate_ml_per_day"]
    assert result["rate_ml_per_day"] <= result["rate_ci_high"]
