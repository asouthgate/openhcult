import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from hcultinf.plot_style import (
    apply_dark_theme,
    CLOUD_BLUE,
    CLOUD_WHITE,
    ORANGE,
    YELLOW,
)

apply_dark_theme()
import numpy as np
import pytest

from hcultinf.exp import ExponentialCordCalibrator, exponential_target
from hcultinf.exp_mcmc import ExponentialCordCalibratorMCMC, plot_corner

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
                xmin_mu=2.75,
                xmin_sigma=0.125,
                xmin_high=3.5,
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
        for _ in range(10):
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
        xmin_mu=875.0,
        xmin_sigma=75.0,
        xmin_high=1100.0,
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

    mean, ci_low, ci_high = cal.predict(prior_x)
    EST_SWC = 800.0
    assert np.abs(max(mean) - EST_SWC) <= 150
    assert all(np.abs(ci_low - mean) <= 750)
    assert all(np.abs(ci_high - mean) <= 750)
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

    # import matplotlib.pyplot as plt

    # from hcultinf.plot_style import apply_dark_theme, CLOUD_BLUE, CLOUD_WHITE

    # apply_dark_theme()

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
        xmin_mu=2.75,
        xmin_sigma=0.125,
        xmin_high=3.5,
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


def test_std_consistent_with_ci():
    cal = _fit_mcmc(seed=42)
    x_grid = np.linspace(TEST_XMIN, TEST_XMAX, 50)
    mean, ci_low, ci_high = cal.predict(x_grid)
    std = cal.std(x_grid)
    ci_width = np.asarray(ci_high) - np.asarray(ci_low)
    expected_width = 2 * 1.96 * np.asarray(std)
    np.testing.assert_allclose(ci_width, expected_width, rtol=1e-6)


def test_posterior_samples_swc_at():
    cal = _fit_mcmc(seed=42)
    x_grid = np.linspace(TEST_XMIN + 0.5, TEST_XMAX - 0.5, 20)
    samples = cal.posterior_samples_swc_at(x_grid, n=50)
    assert samples.shape[0] == 50
    assert samples.shape[1] == len(x_grid)
    assert np.all(np.isfinite(samples[~np.isnan(samples)]))
    sample_mean = np.nanmean(samples, axis=0)[cal._n_burn :]
    pred_mean = np.asarray(cal(x_grid))
    valid = ~np.isnan(sample_mean)
    assert all(sample_mean[valid] == pred_mean[valid])


def test_multi_sensor_happy_path():
    SCALE = 270.0
    K0, K1 = 15.0, 15.0
    F_INT0, F_INT1 = 0.05, 0.05
    XMIN0, XMIN1 = 2.5, 2.8
    xmin_high = 3.0
    XMIN_MU = 2.65
    XMIN_SIGMA = 0.2
    samples = 500
    burnin = 250

    fn0 = lambda x: SCALE * exponential_target(x, K0, F_INT0, XMIN0, TEST_XMAX)
    fn1 = lambda x: SCALE * exponential_target(x, K1, F_INT1, XMIN1, TEST_XMAX)

    n_chords_per_sensor = 15

    x0, dx0, dy0 = simulate_calibration_data_samples(
        3.0,
        4.0,
        TEST_DXMIN,
        TEST_DXMAX,
        TEST_NOISE_LEVEL,
        n_chords_per_sensor,
        fn0,
        uniform=True,
    )
    x1, dx1, dy1 = simulate_calibration_data_samples(
        3.0,
        4.0,
        TEST_DXMIN,
        TEST_DXMAX,
        TEST_NOISE_LEVEL,
        n_chords_per_sensor,
        fn1,
        uniform=True,
    )

    x_starts = np.concatenate([x0, x1])
    delta_x = np.concatenate([dx0, dx1])
    delta_swc = np.concatenate([dy0, dy1])
    sensor_chord_labels = np.array([0] * len(x0) + [1] * len(x1))
    print(TEST_XMAX)
    estimator_multi = ExponentialCordCalibratorMCMC(
        xmin_mu=XMIN_MU,
        xmin_sigma=XMIN_SIGMA,
        xmin_high=xmin_high,
        xmax=TEST_XMAX,
        prior_weight=1.0,
        n_burn=burnin,
        n_steps=samples,
        n_sensors=2,
    )

    cal_multi = estimator_multi.fit(
        np.array([TEST_XMAX]),
        np.array([0.0]),
        x_starts,
        delta_x,
        delta_swc,
        np.array([TEST_XMAX]),
        np.array([0.0]),
        sensor_chord_labels=sensor_chord_labels,
    )

    # chain = estimator_multi._chain
    # print(chain.shape)
    # for j in range(4):
    #     for wi in range(chain.shape[1]):
    #         plt.plot(chain[:, wi, j])
    #     plt.show()

    # plt.plot(cal_multi._log_prob)
    # plt.show()

    print("Joint xmin MAP:", np.mean(cal_multi.posterior_params()["xmin"], axis=0))
    print("Joint scale MAP:", np.mean(cal_multi.posterior_params()["scale"], axis=0))
    est_kw = dict(
        xmax=TEST_XMAX, prior_weight=1.0, n_burn=burnin, n_steps=samples, n_sensors=1
    )
    cal_s0 = ExponentialCordCalibratorMCMC(
        xmin_mu=XMIN_MU, xmin_sigma=XMIN_SIGMA, xmin_high=xmin_high, **est_kw
    ).fit(
        np.array([TEST_XMAX]),
        np.array([0.0]),
        x0,
        dx0,
        dy0,
        np.array([TEST_XMAX]),
        np.array([0.0]),
    )
    cal_s1 = ExponentialCordCalibratorMCMC(
        xmin_mu=XMIN_MU, xmin_sigma=XMIN_SIGMA, xmin_high=xmin_high, **est_kw
    ).fit(
        np.array([TEST_XMAX]),
        np.array([0.0]),
        x1,
        dx1,
        dy1,
        np.array([TEST_XMAX]),
        np.array([0.0]),
    )
    # print MAP estimates of scaleand xmin
    print("Sensor 0 MAP scale:", float(cal_s0.posterior_params()["scale"].mean()))
    print("Sensor 0 MAP xmin:", float(cal_s0.posterior_params()["xmin"].mean()))
    print("Sensor 1 MAP scale:", float(cal_s1.posterior_params()["scale"].mean()))
    print("Sensor 1 MAP xmin:", float(cal_s1.posterior_params()["xmin"].mean()))

    cal = cal_multi

    x_grid = np.linspace(TEST_XMIN, TEST_XMAX, 20)
    x_grid_2d = np.column_stack([x_grid, x_grid])
    mean, ci_low, ci_high = cal.predict(x_grid_2d)
    assert np.all(np.isfinite(mean))
    assert np.all(mean >= 0)
    assert np.all(np.isfinite(ci_low))
    assert np.all(np.isfinite(ci_high))
    assert np.all(ci_low <= mean)
    assert np.all(mean <= ci_high)

    params = cal.posterior_params()
    assert "scale" in params
    assert "k" in params
    assert "f_int" in params
    assert "sigma2" in params
    assert "xmin" in params
    assert params["scale"].ndim == 1
    assert params["k"].shape == (params["scale"].shape[0], 2)
    assert params["f_int"].shape == (params["scale"].shape[0], 2)
    assert params["sigma2"].ndim == 1
    assert params["xmin"].shape == (params["scale"].shape[0], 2)

    samples = cal.posterior_samples_swc_at(x_grid_2d, n=20)
    assert samples.shape == (20, len(x_grid))

    scale_est = cal.scale
    # assert 1.0 < scale_est < 2 * SCALE, f"Scale estimate {scale_est} out of range"

    cal.plot(
        np.array([TEST_XMIN, TEST_XMAX]),
        np.array([1.0, 0.0]),
        np.array([TEST_XMAX]),
        np.array([0.0]),
        x_starts,
        delta_x,
        delta_swc,
        out="artifacts/multi_sensor_mcmc.png",
        title="Multi-sensor MCMC calibration",
        show_chords_pane=False,
    )

    # import matplotlib.pyplot as plt
    # from hcultinf.plot_style import (
    #     apply_dark_theme,
    #     CLOUD_BLUE,
    #     CLOUD_WHITE,
    #     ORANGE,
    #     YELLOW,
    # )

    # apply_dark_theme()
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, c, cal_i, sidx in [
        ("sensor 0", CLOUD_BLUE, cal_s0, 0),
        ("sensor 1", ORANGE, cal_s1, 0),
        ("joint s0", CLOUD_WHITE, cal_multi, 0),
        ("joint s1", YELLOW, cal_multi, 1),
    ]:
        paths, level = cal_i.curve_credible_region(
            sensor_idx=sidx, alpha=0.99, n_bins=100
        )
        for j, path in enumerate(paths):
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=c,
                linewidth=1.5,
                label=label if j == 0 else None,
            )
        p = cal_i.posterior_params()
        mean_scale = float(np.mean(p["scale"]))
        xmin_val = float(
            np.mean(p["xmin"][:, sidx] if p["xmin"].ndim > 1 else p["xmin"])
        )
        ax.scatter(
            [xmin_val],
            [mean_scale],
            marker="x",
            s=80,
            zorder=5,
            label=f"{label} xmin",
        )

    ax.set_xlabel("sensor reading")
    ax.set_ylabel("SWC")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("artifacts/multi_sensor_comparison.png")
    if os.environ.get("HCULT_TEST_DEBUG_PLOT", "0") == "1":
        plt.show()
    plt.close(fig)

    plot_corner(
        cal,
        out="artifacts/multi_sensor_corner.png",
        title="Multi-sensor MCMC posterior",
        sensor_idx=0,
    )
    plot_corner(
        cal,
        out="artifacts/multi_sensor_corner_s1.png",
        title="Multi-sensor MCMC posterior (sensor 1)",
        sensor_idx=1,
    )


def test_predict_warns_below_xmin(caplog):
    import logging

    cal = _fit_mcmc(seed=42)

    x_below = 2.0
    caplog.set_level(logging.WARNING)
    mean, ci_low, ci_high = cal.predict(np.array([x_below]))
    assert "x values below 5% posterior probability" in caplog.text
    assert "2.0000" in caplog.text
    assert "(P=0.0%)" in caplog.text

    caplog.clear()
    x_ok = TEST_XMIN
    mean, ci_low, ci_high = cal.predict(np.array([x_ok]))
    assert "posterior probability of being >= xmin" not in caplog.text

    mean_ok, ci_low_ok, ci_high_ok = mean, ci_low, ci_high
    assert np.all(np.isfinite(mean_ok))
    assert np.all(mean_ok >= 0)


def test_prob_xmin():
    cal = _fit_mcmc(seed=42)

    p_at_xmax = cal.prob_xmin(np.array([TEST_XMAX]))
    assert p_at_xmax[0] == 1.0

    p_below = cal.prob_xmin(np.array([2.0]))
    assert p_below[0] == 0.0

    p_mid = cal.prob_xmin(np.array([2.75]))
    assert 0.0 < p_mid[0] < 1.0


def test_predict_return_prob_x():
    cal = _fit_mcmc(seed=42)
    x_grid = np.array([2.0, 3.0, TEST_XMAX])
    mean, ci_low, ci_high, prob_x = cal.predict(x_grid, return_prob_x=True)
    assert prob_x[0] == 0.0
    assert prob_x[1] > 0.0
    assert prob_x[2] == 1.0
    assert len(mean) == 3
    assert len(ci_low) == 3
    assert len(ci_high) == 3


def test_curve_credible_region():
    cal = _fit_mcmc(seed=42)
    paths, level = cal.curve_credible_region(alpha=0.95, n_bins=50, n_points=200)
    assert level > 0.0
    assert len(paths) >= 1
    for path in paths:
        assert path.ndim == 2
        assert path.shape[1] == 2
        assert np.all(path[:, 0] >= TEST_XMIN - 1.0)
        assert np.all(path[:, 0] <= TEST_XMAX)
        assert np.all(np.isfinite(path))
