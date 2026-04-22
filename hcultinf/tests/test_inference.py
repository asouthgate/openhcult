import copy
import os

import numpy as np
import pytest

from hcultinf.gp import GPWithPriorShape
from hcultinf.power import PowerCordCalibrator
from hcultinf.exp import ExponentialCordCalibrator, exponential_target
from hcultinf.simulation import (
    simulate_calibration_data_samples,
    Y_TEST_FUNCTION_NONORM,
    Y_TEST_FUNCTION,
    Y_TEST_FUNCTION_DECREASING,
    power_function,
)

TEST_XMIN = 0.0
TEST_XMAX = 5.5
TEST_DXMIN = 0.2
TEST_DXMAX = 0.5
TEST_NOISE_LEVEL = 0.5

_POWER_TEST_FN = lambda x: 10.0 * power_function(x, 5.0, 0.0, TEST_XMIN, TEST_XMAX)


def _get_mixed_prior_decreasing(xmin, xmax, p):
    priorx = np.linspace(xmin, xmax, 1000)
    linear_prior_y = np.interp(priorx, [TEST_XMIN, TEST_XMAX], [1.0, 0.0])
    fn_vals = _POWER_TEST_FN(priorx)
    normalized_fn = (fn_vals - fn_vals.min()) / (fn_vals.max() - fn_vals.min())
    priory = linear_prior_y * p + normalized_fn * (1 - p)
    return priorx, priory


def _get_pointwise_random_mixed_prior_decreasing(xmin, xmax, p):
    priorx = np.linspace(xmin, xmax, 1000)
    linear_prior_y = np.interp(priorx, [TEST_XMIN, TEST_XMAX], [1.0, 0.0])
    fn_vals = _POWER_TEST_FN(priorx)
    normalized_fn = (fn_vals - fn_vals.min()) / (fn_vals.max() - fn_vals.min())
    randomps = np.random.uniform(0, p, size=len(priorx))
    priory = linear_prior_y * randomps + normalized_fn * (1 - randomps)
    return priorx, priory


TEST_POWER_FUNCTION = lambda x: 10.0 * power_function(x, 5.0, 0.0, TEST_XMIN, TEST_XMAX)


@pytest.mark.parametrize(
    "estimator_func_pair",
    [
        (
            ExponentialCordCalibrator(
                TEST_XMIN, TEST_XMAX, 1e-8
            ),  # must be low prior weight or we wont converge
            lambda x: 10.0
            * exponential_target(x, k=20.0, f_int=0.0, xmin=TEST_XMIN, xmax=TEST_XMAX),
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
    for n in [4, 16, 128]:
        priorx_pts = np.array([TEST_XMIN, TEST_XMAX])
        priory_pts = np.array([1.0, 0.0])
        errs = []
        for _ in range(5):
            x, dx, dy = simulate_calibration_data_samples(
                TEST_XMIN,
                TEST_XMAX,
                TEST_DXMAX,
                TEST_DXMAX,
                TEST_NOISE_LEVEL
                / 5,  # must have some noise to work well for convergence in n
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
        # print(errs)
        curve_error = np.mean(errs)
        preverrs.append(curve_error)

    plot_x = np.linspace(TEST_XMIN, TEST_XMAX, 500)
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
        PowerCordCalibrator(TEST_XMIN, TEST_XMAX, prior_weight=0.001),
        ExponentialCordCalibrator(TEST_XMIN, TEST_XMAX),
        # GPWithPriorShape(), doesn't really perform well
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
        TEST_XMIN + DR1, TEST_XMIN + DR2, 1.0, 2.0, 0.15, n, _POWER_TEST_FN
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
        true_y=_POWER_TEST_FN(priorx) - _POWER_TEST_FN(TEST_XMAX),
        out=f"artifacts/realistic_parameters_{estimator.__class__.__name__}.png",
        title=f"Test performance with realistic parameters ({estimator.__class__.__name__})",
        show_chords_pane=False,
    )

    curve_error = np.abs(
        (_POWER_TEST_FN(priorx) - _POWER_TEST_FN(TEST_XMAX))
        - (pwl(priorx) - pwl(TEST_XMAX))
    ).mean()
    assert curve_error < 0.2 * _POWER_TEST_FN(priorx).max()


@pytest.mark.parametrize(
    "estimator_func_pair",
    [
        (
            ExponentialCordCalibrator(
                TEST_XMIN, TEST_XMAX, 1e-8
            ),  # must be low prior weight or we wont converge
            lambda x: 10.0
            * exponential_target(x, k=20.0, f_int=0.0, xmin=TEST_XMIN, xmax=TEST_XMAX),
        ),
        (
            PowerCordCalibrator(TEST_XMIN, TEST_XMAX, prior_weight=0.001),
            _POWER_TEST_FN,
        ),
        (GPWithPriorShape(length_scale=1.0), _POWER_TEST_FN),
    ],
)
def test_total_estimate_improves_and_std_shrinks_with_coverage(estimator_func_pair):
    """Test that wider data coverage shrinks std and improves the curve estimate."""
    estimator, test_func = estimator_func_pair
    pwlprevs = []
    preverrs = []
    prevstds = []
    n = 50
    anchorx = np.array([TEST_XMAX])
    anchory = np.array([0.0])
    for ddx in [0.01, 0.15, 0.45]:
        x, dx, dy = simulate_calibration_data_samples(
            TEST_XMIN + ((TEST_XMAX - TEST_XMIN) / 2.0) - ddx * (TEST_XMAX - TEST_XMIN),
            TEST_XMIN + ((TEST_XMAX - TEST_XMIN) / 2.0) + ddx * (TEST_XMAX - TEST_XMIN),
            1.0,
            2.0,
            TEST_NOISE_LEVEL * 0.0,
            n,
            test_func,
            uniform=True,
        )
        priorx, priory = _get_mixed_prior_decreasing(TEST_XMIN, TEST_XMAX, 0.5)

        pwl = estimator.fit(anchorx, anchory, x, dx, dy, priorx, priory)
        prevstds.append(pwl.predict(priorx)[1].mean())
        pwlprevs.append(lambda x, fn=pwl._mean: fn(x))
        curve_error = (
            (
                (_POWER_TEST_FN(priorx) - _POWER_TEST_FN(TEST_XMAX))
                - (pwl(priorx) - pwl(TEST_XMAX))
            )
            ** 2
        ).mean()
        preverrs.append(curve_error)

    pwl.plot(
        priorx,
        priory,
        anchorx,
        anchory,
        x,
        dx,
        dy,
        pwlprevs=pwlprevs[:-1],
        true_y=_POWER_TEST_FN(priorx) - _POWER_TEST_FN(TEST_XMAX),
        out=f"artifacts/coverage_std_shrinks_{estimator.__class__.__name__}.png",
        title=f"Test estimate improves and std shrinks with coverage ({estimator.__class__.__name__})",
        show_chords_pane=False,
    )

    assert all(
        np.diff(prevstds) < 0
    ), f"Curve std did not decrease with increasing coverage: {prevstds}"
    assert all(
        np.diff(preverrs) < 0
    ), f"Curve error did not decrease with increasing coverage: {preverrs}"


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
    true_y = _POWER_TEST_FN(eval_x)

    estimates = np.zeros((R, len(eval_x)))
    for i in range(R):
        x, dx, dy = simulate_calibration_data_samples(
            TEST_XMIN,
            TEST_XMAX,
            TEST_DXMIN,
            TEST_DXMAX,
            TEST_NOISE_LEVEL,
            n,
            _POWER_TEST_FN,
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

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eval_x, true_y, label="true", color="black")
    ax.plot(eval_x, mean_pred, label="mean prediction", linestyle="--")
    ax.fill_between(
        eval_x,
        mean_pred - 1.96 * std_of_means,
        mean_pred + 1.96 * std_of_means,
        alpha=0.3,
        label="95% CI on mean",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
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
