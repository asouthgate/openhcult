import os

import numpy as np
import pytest

from hcultinf.gp import GPWithPriorShape
from hcultinf.power import PowerCordCalibrator
from hcultinf.exp import ExponentialCordCalibrator, exponential_target
from hcultinf.exp_mcmc import ExponentialCordCalibratorMCMC
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
                xmin_low=1.0,
                xmin_high=5.0,
                xmax=TEST_XMAX,
                prior_weight=1e-6,
                n_burn=30,
                n_steps=60,
            ),
            TEST_EXPONENTIAL_FUNCTION,
        ),
        (
            ExponentialCordCalibrator(TEST_XMIN, TEST_XMAX, 1e-8),
            TEST_EXPONENTIAL_FUNCTION,
        ),
        (
            PowerCordCalibrator(TEST_XMIN, TEST_XMAX, prior_weight=0.001),
            TEST_POWER_FUNCTION,
        ),
        (GPWithPriorShape(length_scale=1.0), TEST_POWER_FUNCTION),
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


@pytest.mark.parametrize(
    "estimator",
    [
        ExponentialCordCalibratorMCMC(
            xmin_low=2.0, xmin_high=4.0, xmax=TEST_XMAX, n_burn=1, n_steps=10
        ),
        PowerCordCalibrator(TEST_XMIN, TEST_XMAX, prior_weight=0.001),
        ExponentialCordCalibrator(TEST_XMIN, TEST_XMAX),
    ],
)
def test_performance_realistic_parameters(estimator):
    """Test that with realistic parameters the estimate stays close to the true curve."""
    n = 10
    anchorx = np.array([TEST_XMAX])
    anchory = np.array([0.0])
    DR1 = (TEST_XMAX - TEST_XMIN) * 0.1
    DR2 = (TEST_XMAX - TEST_XMIN) * 0.5
    x, dx, dy = simulate_calibration_data_samples(
        TEST_XMIN + DR1, TEST_XMIN + DR2, 1.0, 2.0, 0.15, n, TEST_POWER_FUNCTION
    )
    priorx, priory = _get_mixed_prior_decreasing(TEST_XMIN, TEST_XMAX, 1.0)

    pwl = estimator.fit(np.array(anchorx), np.array(anchory), x, dx, dy, priorx, priory)
    pwl.plot(
        priorx,
        priory,
        anchorx,
        anchory,
        x,
        dx,
        dy,
        true_y=TEST_POWER_FUNCTION(priorx) - TEST_POWER_FUNCTION(TEST_XMAX),
        out=f"artifacts/realistic_parameters_{estimator.__class__.__name__}.png",
        title=f"Test performance with realistic parameters ({estimator.__class__.__name__})",
        show_chords_pane=False,
    )

    curve_error = np.abs(
        (TEST_POWER_FUNCTION(priorx) - TEST_POWER_FUNCTION(TEST_XMAX))
        - (pwl(priorx) - pwl(TEST_XMAX))
    ).mean()
    assert curve_error < 0.35 * TEST_POWER_FUNCTION(priorx).max()


@pytest.mark.parametrize(
    "estimator",
    [
        PowerCordCalibrator(TEST_XMIN, TEST_XMAX, prior_weight=0.01),
        GPWithPriorShape(),
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


# def test_mcmc_std_does_not_collapse_with_n():
#     n = 50
#     anchorx = np.array([TEST_XMAX])
#     anchory = np.array([0.0])
#     priorx, priory = _get_mixed_prior_decreasing(TEST_XMIN, TEST_XMAX, 0.5)

#     stds_single, stds_double = [], []
#     for _ in range(3):
#         x, dx, dy = simulate_calibration_data_samples(
#             TEST_XMIN, TEST_XMAX, 1.0, 2.0, 0.15, n, TEST_POWER_FUNCTION
#         )
#         cal = ExponentialCordCalibratorMCMC(
#             TEST_XMIN, TEST_XMAX, n_burn=100, n_steps=200
#         ).fit(anchorx, anchory, x, dx, dy, priorx, priory)
#         _, ci_low, ci_high = cal.predict(priorx)
#         stds_single.append((ci_high - ci_low).mean())

#     for _ in range(3):
#         x1, dx1, dy1 = simulate_calibration_data_samples(
#             TEST_XMIN, TEST_XMAX, 1.0, 2.0, 0.15, n, TEST_POWER_FUNCTION
#         )
#         x2, dx2, dy2 = simulate_calibration_data_samples(
#             TEST_XMIN, TEST_XMAX, 1.0, 2.0, 0.15, n, TEST_POWER_FUNCTION
#         )
#         x = np.concatenate([x1, x2])
#         dx = np.concatenate([dx1, dx2])
#         dy = np.concatenate([dy1, dy2])
#         cal = ExponentialCordCalibratorMCMC(
#             TEST_XMIN, TEST_XMAX, n_burn=100, n_steps=200
#         ).fit(anchorx, anchory, x, dx, dy, priorx, priory)
#         _, ci_low, ci_high = cal.predict(priorx)
#         stds_double.append((ci_high - ci_low).mean())

#     mean_single = np.mean(stds_single)
#     mean_double = np.mean(stds_double)
#     assert mean_double > 0, "MCMC std collapsed to zero"
#     assert (
#         mean_double / mean_single > 0.3
#     ), f"Std collapsed too fast: 2n std is {mean_double / mean_single:.2f}x of 1n std"


# def test_mcmc_xmin_uncertainty():
#     n = 30
#     xmin_true = 0.5
#     anchorx = np.array([TEST_XMAX])
#     anchory = np.array([0.0])
#     priorx = np.linspace(xmin_true, TEST_XMAX, 500)
#     priory = np.interp(priorx, [xmin_true, TEST_XMAX], [1.0, 0.0])

#     x, dx, dy = simulate_calibration_data_samples(
#         xmin_true + 0.1, TEST_XMAX - 0.1, 0.5, 1.0, 0.15, n, TEST_POWER_FUNCTION
#     )


# def test_mcmc_curve_convergence():
#     xmin, xmax = 3.0, 8.5
#     true_k, true_f_int, true_scale = 20.0, 0.0, 10.0
#     true_func = lambda x: true_scale * exponential_target(
#         x, true_k, true_f_int, xmin, xmax
#     )
#     g_func = lambda x: exponential_target(x, true_k, true_f_int, xmin, xmax)

#     anchorx = np.array([xmax])
#     anchory = np.array([0.0])
#     prior_x = np.linspace(xmin, xmax, 100)
#     prior_y = g_func(prior_x)
#     eval_x = np.linspace(xmin, xmax, 500)
#     true_y = true_func(eval_x)

#     errors = []
#     last_cal = None
#     last_x = last_dx = last_dy = None
#     for n in [4, 16, 64, 128]:
#         x, dx, dy = simulate_calibration_data_samples(
#             xmin,
#             xmax,
#             0.5,
#             0.5,
#             0.1,
#             n,
#             true_func,
#             uniform=True,
#         )
#         cal = ExponentialCordCalibratorMCMC(
#             xmin_low=1.0,
#             xmin_high=5.0,
#             xmax=xmax,
#             prior_weight=1.0,
#             n_burn=30,
#             n_steps=50,
#         )
#         cal.fit(anchorx, anchory, x, dx, dy, prior_x, prior_y)
#         pred = cal(eval_x)
#         mae = np.nanmean(np.abs(pred - true_y))
#         errors.append(mae)
#         last_cal, last_x, last_dx, last_dy = cal, x, dx, dy

#     plot_x = np.linspace(xmin, xmax, 500)
#     plot_prior_y = np.interp(plot_x, prior_x, prior_y)
#     last_cal.plot(
#         plot_x,
#         plot_prior_y,
#         anchorx,
#         anchory,
#         last_x,
#         last_dx,
#         last_dy,
#         true_y=true_func(plot_x),
#         out="artifacts/mcmc_curve_convergence.png",
#         title=f"MCMC curve convergence (errors={[f'{e:.4f}' for e in errors]})",
#         show_chords_pane=False,
#     )
#     _, ci_low, ci_high = last_cal.predict(eval_x)
#     ci_width = ci_high - ci_low
#     kernel = np.ones(5) / 5
#     smooth_width = np.convolve(ci_width, kernel, mode='same')
#     max_spike = np.max(np.abs(ci_width - smooth_width))
#     assert max_spike < 0.5, (
#         f"CI has spike of {max_spike:.3f} — posterior predictive should be smooth"
#     )
#     assert errors[-1] < 0.05, f"Errors at each n: {errors}"
