import os

import numpy as np
import pytest

from hcultinf.gp import GPWithPriorShape
from hcultinf.power import PowerCordCalibrator
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


def _get_mixed_prior(xmin, xmax, p):
    priorx = np.linspace(xmin, xmax, 1000)
    linear_prior_y = np.interp(priorx, [TEST_XMIN, TEST_XMAX], [0.0, 1.0])
    assert linear_prior_y.min() == 0.0 and linear_prior_y.max() == 1.0
    normalized_true_prior_y = (Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN)) / (
        Y_TEST_FUNCTION(priorx).max() - Y_TEST_FUNCTION(priorx).min()
    )
    priory = linear_prior_y * p + normalized_true_prior_y * (1 - p)
    return priorx, priory


@pytest.mark.parametrize(
    "estimator",
    [
        PowerCordCalibrator(TEST_XMIN, TEST_XMAX, prior_weight=0.001),
        GPWithPriorShape(length_scale=1.0),
    ],
)
def test_convergence_in_n_bad_prior(estimator):
    """Test that, as data increases, the curve estimate approaches the true curve."""
    preverrs = []
    last_pwl = last_x = last_dx = last_dy = None
    test_function = lambda x: 10.0 * power_function(x, 5.0, 0.0, TEST_XMIN, TEST_XMAX)
    anchorx = np.array([TEST_XMAX])
    anchory = np.array([0.0])
    for n in [4, 32, 256]:
        priorx_pts = np.array([TEST_XMIN, TEST_XMAX])
        priory_pts = np.array([1.0, 0.0])
        errs = []
        for _ in range(5):
            x, dx, dy = simulate_calibration_data_samples(
                TEST_XMIN,
                TEST_XMAX,
                TEST_DXMIN,
                TEST_DXMAX,
                0.0,
                n,
                test_function,
            )
            pwl = estimator.fit(
                anchorx,
                anchory,
                x,
                dx,
                dy,
                priorx_pts,
                priory_pts,
            )
            pwlx = np.linspace(TEST_XMIN, TEST_XMAX, 2000)
            _curve_error = np.abs(
                (test_function(pwlx) - test_function(TEST_XMIN))
                - (pwl(pwlx) - pwl(TEST_XMIN))
            ).mean()
            errs.append(_curve_error)
        last_pwl, last_x, last_dx, last_dy = pwl, x, dx, dy
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
        true_y=test_function(plot_x) - test_function(plot_x).min(),
        out=f"artifacts/convergence_n_bad_prior_{estimator.__class__.__name__}.png",
        title=f"Test convergence in n with bad prior ({estimator.__class__.__name__})",
        show_chords_pane=False,
    )

    assert all(
        np.diff(preverrs) < 0
    ), f"Curve error did not decrease with increasing n: {preverrs}"
    assert curve_error < 0.02 * Y_TEST_FUNCTION(pwlx).max()


def test_convergence_in_prior_low_n():
    """Test that, as data increases, the curve estimate approaches the true curve."""
    n = 4
    x, dx, dy = simulate_calibration_data_samples(
        TEST_XMIN,
        TEST_XMAX,
        TEST_DXMIN,
        TEST_DXMAX,
        TEST_NOISE_LEVEL * 0.1,
        n,
        Y_TEST_FUNCTION,
        uniform=True,
    )
    pwlprevs = []
    preverrs = []

    for p in [1.0, 2 / 3, 1 / 3, 0.0]:
        priorx, priory = _get_mixed_prior(TEST_XMIN, TEST_XMAX, p)
        pwl = GPWithPriorShape().fit(
            np.array([TEST_XMIN]), np.array([0.0]), x, dx, dy, priorx, priory
        )
        curve_error = (
            (
                (Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN))
                - (pwl(priorx) - pwl(TEST_XMIN))
            )
            ** 2
        ).mean()
        pwlprevs.append(pwl)
        preverrs.append(curve_error)

    pwl.plot(
        priorx,
        priory,
        [TEST_XMIN],
        [0.0],
        x,
        dx,
        dy,
        pwlprevs=pwlprevs[:-1],
        true_y=Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN),
        out="artifacts/convergence_prior_low_n.png",
        title="Test convergence in prior with low n",
        show_chords_pane=False,
    )

    assert all(
        np.diff(preverrs) < 0
    ), f"Curve error did not decrease with increasing p: {preverrs}"
    assert curve_error < 0.02 * Y_TEST_FUNCTION(priorx).max()


def test_performance_realistic_parameters():
    """Test that, as data increases, the curve estimate approaches the true curve."""
    n = 10
    x, dx, dy = simulate_calibration_data_samples(
        1.5, 3.5, 0.1, 0.9, 1.0, n, Y_TEST_FUNCTION
    )
    anchors_x = [TEST_XMIN]
    anchors_y = [0.0]
    priorx, priory = _get_mixed_prior(TEST_XMIN, TEST_XMAX, 1.0)

    pwl = GPWithPriorShape().fit(
        np.array(anchors_x),
        np.array(anchors_y),
        x,
        dx,
        dy,
        priorx,
        priory,
    )
    pwl.plot(
        priorx,
        priory,
        anchors_x,
        anchors_y,
        x,
        dx,
        dy,
        true_y=Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN),
        out="artifacts/realistic_parameters.png",
        title="Test performance with realistic parameters",
        show_chords_pane=False,
    )

    curve_error = np.abs(
        (Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN))
        - (pwl(priorx) - pwl(TEST_XMIN))
    ).mean()
    assert curve_error < 0.2 * Y_TEST_FUNCTION(priorx).max()


def test_total_estimate_improves_and_std_shrinks_with_coverage():
    """Test that, as data increases, the curve estimate approaches the true curve."""
    pwlprevs = []
    preverrs = []
    prevstds = []
    n = 100
    half_cov = (TEST_XMAX - TEST_XMIN) * 0.5
    eps = (TEST_XMAX - TEST_XMIN) * 0.1
    for ddx in np.linspace(eps, half_cov - eps, 4):
        x, dx, dy = simulate_calibration_data_samples(
            TEST_XMIN + ((TEST_XMAX - TEST_XMIN) / 2.0) - ddx,
            TEST_XMIN + ((TEST_XMAX - TEST_XMIN) / 2.0) + ddx,
            0.5,
            0.5,
            0.0,
            n,
            Y_TEST_FUNCTION,
            uniform=True,
        )
        anchors_x = [TEST_XMIN]
        anchors_y = [0.0]
        priorx, priory = _get_mixed_prior(TEST_XMIN, TEST_XMAX, 0.5)

        pwl = GPWithPriorShape().fit(
            np.array(anchors_x),
            np.array(anchors_y),
            x,
            dx,
            dy,
            priorx,
            priory,
        )
        prevstds.append(pwl.predict(priorx)[1].mean())
        pwlprevs.append(pwl)
        curve_error = np.abs(
            (Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN))
            - (pwl(priorx) - pwl(TEST_XMIN))
        ).mean()
        preverrs.append(curve_error)

    pwl.plot(
        priorx,
        priory,
        anchors_x,
        anchors_y,
        x,
        dx,
        dy,
        pwlprevs=pwlprevs[:-1],
        true_y=Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN),
        out="artifacts/coverage_std_shrinks.png",
        title="Test estimate improves and std shrinks with coverage",
        show_chords_pane=False,
    )

    assert all(
        np.diff(prevstds) < 0
    ), f"Curve std did not decrease with increasing delta size: {prevstds}"
    assert all(
        np.diff(preverrs) < 0
    ), f"Curve error did not decrease with increasing delta size: {preverrs}"


def test_noise_recovery():
    """MLE-optimised noise variance should track the true noise variance."""
    n = 50
    priorx, priory = _get_mixed_prior(TEST_XMIN, TEST_XMAX, 0.5)
    noise_levels = np.linspace(0.1, TEST_NOISE_LEVEL * 8, 8)
    R = 50

    mean_fitted = []
    for noise in noise_levels:
        fitted = []
        for _ in range(R):
            x, dx, dy = simulate_calibration_data_samples(
                TEST_XMIN,
                TEST_XMAX,
                TEST_DXMIN,
                TEST_DXMAX,
                noise,
                n,
                Y_TEST_FUNCTION,
            )
            pwl = GPWithPriorShape().fit(
                np.array([TEST_XMIN]), np.array([0.0]), x, dx, dy, priorx, priory
            )
            fitted.append(pwl.noise)
        mean_fitted.append(np.mean(fitted))

    true_variances = noise_levels**2
    mean_fitted = np.array(mean_fitted)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(true_variances, mean_fitted, marker="o", label="fitted noise variance")
    ax.plot(
        true_variances,
        true_variances,
        linestyle="--",
        marker="o",
        color="black",
        label="ideal (fitted = true)",
    )
    ax.set_xlabel("true noise variance")
    ax.set_ylabel("fitted noise variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig("artifacts/noise_recovery.png")
    if os.environ.get("HCULT_TEST_DEBUG_PLOT", "0") == "1":
        plt.show()
    plt.close(fig)

    correlation = np.corrcoef(true_variances, mean_fitted)[0, 1]
    assert (
        correlation > 0.9
    ), f"Fitted noise poorly correlated with true noise (r={correlation:.3f})"


def test_error_degrades_gracefully_with_noise():
    """Mean absolute error should increase smoothly as observation noise grows."""
    n = 30
    priorx, priory = _get_mixed_prior(TEST_XMIN, TEST_XMAX, 0.5)
    eval_x = np.linspace(TEST_XMIN, TEST_XMAX, 200)
    true_y = Y_TEST_FUNCTION(eval_x) - Y_TEST_FUNCTION(eval_x[0])
    noise_levels = np.linspace(0.0, TEST_NOISE_LEVEL * 4, 8)
    R = 50

    mean_errors = []
    for noise in noise_levels:
        errs = []
        for _ in range(R):
            x, dx, dy = simulate_calibration_data_samples(
                TEST_XMIN,
                TEST_XMAX,
                TEST_DXMIN,
                TEST_DXMAX,
                noise,
                n,
                Y_TEST_FUNCTION,
            )
            pwl = GPWithPriorShape().fit(
                np.array([TEST_XMIN]), np.array([0.0]), x, dx, dy, priorx, priory
            )
            mean = pwl(eval_x)
            errs.append(np.abs((mean - mean[0]) - true_y).mean())
        mean_errors.append(np.mean(errs))

    import matplotlib.pyplot as plt

    slope, intercept = np.polyfit(noise_levels, mean_errors, 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(noise_levels, mean_errors, marker="o", label="mean error")
    ax.plot(
        noise_levels,
        slope * noise_levels + intercept,
        linestyle="--",
        label=f"trend (slope={slope:.2f})",
    )
    ax.set_xlabel("noise level")
    ax.set_ylabel("mean absolute error")
    ax.legend()
    fig.tight_layout()
    fig.savefig("artifacts/error_vs_noise.png")
    if os.environ.get("HCULT_TEST_DEBUG_PLOT", "0") == "1":
        plt.show()
    plt.close(fig)

    slope = np.polyfit(noise_levels, mean_errors, 1)[0]
    assert (
        slope > 0
    ), f"Error trend not increasing with noise (slope={slope:.4f}): {mean_errors}"
    assert (
        mean_errors[-1] < 0.4 * true_y.max()
    ), f"Error at max noise {mean_errors[-1]:.4f} too large"


def test_unbiasedness():
    """Mean prediction across trials should match the true curve pointwise."""
    R = 200
    n = 30
    priorx, priory = _get_mixed_prior(TEST_XMIN, TEST_XMAX, 0.5)
    eval_x = np.linspace(TEST_XMIN, TEST_XMAX, 200)
    true_y = Y_TEST_FUNCTION(eval_x) - Y_TEST_FUNCTION(TEST_XMIN)

    estimates = np.zeros((R, len(eval_x)))
    for i in range(R):
        x, dx, dy = simulate_calibration_data_samples(
            TEST_XMIN,
            TEST_XMAX,
            TEST_DXMIN,
            TEST_DXMAX,
            TEST_NOISE_LEVEL,
            n,
            Y_TEST_FUNCTION,
        )
        pwl = GPWithPriorShape().fit(
            np.array([TEST_XMIN]), np.array([0.0]), x, dx, dy, priorx, priory
        )
        mean = pwl(eval_x)
        estimates[i] = mean - mean[0]

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


def test_ci_calibration():
    """Empirical CI calibration: stated 95% CI should contain the true curve ~95% of trials."""
    R = 200
    n = 30
    priorx, priory = _get_mixed_prior(TEST_XMIN, TEST_XMAX, 0.5)
    eval_x = np.linspace(TEST_XMIN, TEST_XMAX, 200)
    true_y = Y_TEST_FUNCTION(eval_x) - Y_TEST_FUNCTION(TEST_XMIN)

    inside = np.zeros(len(eval_x))
    for _ in range(R):
        x, dx, dy = simulate_calibration_data_samples(
            TEST_XMIN,
            TEST_XMAX,
            TEST_DXMIN,
            TEST_DXMAX,
            TEST_NOISE_LEVEL,
            n,
            Y_TEST_FUNCTION,
        )
        pwl = GPWithPriorShape().fit(
            np.array([TEST_XMIN]), np.array([0.0]), x, dx, dy, priorx, priory
        )
        mean, std = pwl.predict(eval_x)
        mean_shifted = mean - mean[0]
        inside += (true_y >= mean_shifted - 1.96 * std) & (
            true_y <= mean_shifted + 1.96 * std
        )

    empirical_calibration = inside / R

    pwl.plot(
        priorx,
        priory,
        [TEST_XMIN],
        [0.0],
        x,
        dx,
        dy,
        true_y=Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN),
        out="artifacts/ci_calibration.png",
        title="CI calibration (last trial)",
        show_chords_pane=False,
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eval_x, empirical_calibration, label="empirical calibration")
    ax.axhline(0.95, color="gray", linestyle="--", label="nominal 95%")
    ax.set_xlabel("x")
    ax.set_ylabel("fraction of trials CI contains truth")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig("artifacts/ci_calibration_curve.png")
    plt.close(fig)

    mean_cal = empirical_calibration.mean()
    assert (
        0.85 <= mean_cal <= 0.99
    ), f"CI calibration {mean_cal:.3f} outside [0.85, 0.99]"


def test_good_data_robust_to_high_gp_variance():
    """Good data produces low error even at high GP variance (loose prior)."""
    n = 256
    x, dx, dy = simulate_calibration_data_samples(
        TEST_XMIN,
        TEST_XMAX,
        TEST_DXMIN,
        TEST_DXMAX,
        TEST_NOISE_LEVEL * 0.1,
        n,
        Y_TEST_FUNCTION,
    )
    priorx, priory = _get_mixed_prior(TEST_XMIN, TEST_XMAX, 1.0)
    anchors_x = np.array([TEST_XMIN])
    anchors_y = np.array([0.0])

    pwlprevs = []
    errs = []
    variances = [1.0, 10.0, 100.0, 1000.0]
    for var in variances:
        pwl = GPWithPriorShape(variance=var).fit(
            anchors_x, anchors_y, x, dx, dy, priorx, priory
        )
        curve_error = np.abs(
            (Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN))
            - (pwl(priorx) - pwl(TEST_XMIN))
        ).mean()
        errs.append(curve_error)
        pwlprevs.append(pwl)

    pwl.plot(
        priorx,
        priory,
        anchors_x,
        anchors_y,
        x,
        dx,
        dy,
        pwlprevs=pwlprevs[:-1],
        true_y=Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN),
        out="artifacts/high_variance_good_data.png",
        title="Test good data robust to high GP variance",
        show_chords_pane=False,
    )

    threshold = 0.1 * Y_TEST_FUNCTION(priorx).max()
    assert all(
        e < threshold for e in errs
    ), f"Error too high for some variances: {list(zip(variances, errs))}"


def test_total_estimate_improves_and_std_shrinks_with_delta_size():
    """Test that, as data increases, the curve estimate approaches the true curve."""
    pwlprevs = []
    preverrs = []
    prevstds = []
    n = 50
    for dx_max in np.linspace(0.01, 2.0, 4):
        x, dx, dy = simulate_calibration_data_samples(
            1.5,
            3.5,
            dx_max / 2.0,
            dx_max,
            0.0,
            n,
            Y_TEST_FUNCTION,
            uniform=True,
        )
        anchors_x = [TEST_XMIN]
        anchors_y = [0.0]
        priorx, priory = _get_mixed_prior(TEST_XMIN, TEST_XMAX, 0.5)

        pwl = GPWithPriorShape().fit(
            np.array(anchors_x),
            np.array(anchors_y),
            x,
            dx,
            dy,
            priorx,
            priory,
        )
        prevstds.append(pwl.predict(priorx)[1].mean())
        pwlprevs.append(pwl)
        curve_error = np.abs(
            (Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN))
            - (pwl(priorx) - pwl(TEST_XMIN))
        ).mean()
        preverrs.append(curve_error)

    pwl.plot(
        priorx,
        priory,
        anchors_x,
        anchors_y,
        x,
        dx,
        dy,
        pwlprevs=pwlprevs[:-1],
        true_y=Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN),
        out="artifacts/delta_size_std_shrinks.png",
        title="Test estimate improves and std shrinks with delta size",
        show_chords_pane=False,
    )

    assert all(
        np.diff(prevstds) < 0
    ), f"Curve std did not decrease with increasing delta size: {prevstds}"
    assert all(
        np.diff(preverrs) < 0
    ), f"Curve error did not decrease with increasing delta size: {preverrs}"
