import numpy as np
import pytest

from hcultinf.inference import GPWithPriorShape

Y_TEST_FUNCTION = (
    lambda x: x**2 + np.log(x + 1) + np.exp(0.5 * x) + np.sqrt(x) - np.sin(3 * x) + 3.3
)

TEST_XMIN = 0.0
TEST_XMAX = 5.5
TEST_DXMIN = 0.1
TEST_DXMAX = 0.5
TEST_NOISE_LEVEL = 0.05


def _sample_data(xmin, xmax, dxmin, dxmax, noise_level, n, y):
    x = np.random.uniform(xmin, xmax, n)
    x = np.clip(x, xmin, xmax)
    dx = np.random.uniform(dxmin, dxmax, n)
    ends = np.clip(x + dx, xmin, xmax)
    dx = ends - x
    dy = y(x + dx) - y(x)
    dy += np.random.normal(0, noise_level, n)
    return x, dx, dy


def test_convergence_in_n_bad_prior():
    """Test that, as data increases, the curve estimate approaches the true curve."""
    curve_error_prev = 1e10
    # pwlprev = None
    for n in [4, 16, 256]:  # Square the number of points, 5 chosen for convenience
        x, dx, dy = _sample_data(
            TEST_XMIN,
            TEST_XMAX,
            TEST_DXMIN,
            TEST_DXMAX,
            TEST_NOISE_LEVEL,
            n,
            Y_TEST_FUNCTION,
        )
        # Use linear prior (bad)
        priorx = np.array([TEST_XMIN, TEST_XMAX])
        priory = np.array([0.0, 1.0])
        pwl = GPWithPriorShape().fit(
            np.array([]), np.array([]), x, dx, dy, priorx, priory
        )
        # estimated_total_y = pwl(TEST_XMAX) - pwl(TEST_XMIN)
        # import matplotlib.pyplot as plt

        pwlx = np.linspace(TEST_XMIN, TEST_XMAX, 1000)
        # mean, std = pwl.predict(pwlx)
        # ci_lower = mean - 1.96 * std
        # ci_upper = mean + 1.96 * std

        # plt.plot(pwlx, Y_TEST_FUNCTION(pwlx) - Y_TEST_FUNCTION(pwlx.min()), label="true")
        # plt.scatter(pwlx, Y_TEST_FUNCTION(pwlx) - Y_TEST_FUNCTION(pwlx.min()), label="true")
        # plt.scatter(
        #     priorx,
        #     priory * (Y_TEST_FUNCTION(pwlx).max() - Y_TEST_FUNCTION(pwlx).min()),
        #     label="rescaled prior",
        # )
        # plt.plot(pwlx, pwl(pwlx) - pwl(pwlx.min()), label="true")
        # plt.scatter(pwlx, pwl(pwlx) - pwl(pwlx.min()), label="true")
        # plt.fill_between(
        #     pwlx,
        #     ci_lower - mean.min(),
        #     ci_upper - mean.min(),
        #     color="gray",
        #     alpha=0.3,
        #     label="95% CI",
        # )
        # plt.plot(pwlx, mean - mean.min(), label="GP mean")
        # if pwlprev is not None:
        #     plt.plot(pwlx, pwlprev(pwlx) - pwlprev(pwlx.min()), label="previous", alpha=0.5)

        # plt.legend()
        # plt.show()

        # pwlprev = pwl

        curve_error = np.abs(
            (Y_TEST_FUNCTION(pwlx) - Y_TEST_FUNCTION(TEST_XMIN))
            - (pwl(pwlx) - pwl(TEST_XMIN))
        ).mean()
        assert curve_error < curve_error_prev
        curve_error_prev = curve_error

    assert curve_error < 0.01 * Y_TEST_FUNCTION(pwlx).max()


def test_convergence_in_prior_low_n():
    """Test that, as data increases, the curve estimate approaches the true curve."""
    curve_error_prev = 1e10
    # pwlprev = None
    n = 2
    x, dx, dy = _sample_data(
        TEST_XMIN,
        TEST_XMAX,
        TEST_DXMIN,
        TEST_DXMAX,
        TEST_NOISE_LEVEL,
        n,
        Y_TEST_FUNCTION,
    )
    for p in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
        priorx = np.linspace(TEST_XMIN, TEST_XMAX, 1000)
        linear_prior_y = np.interp(priorx, [TEST_XMIN, TEST_XMAX], [0.0, 1.0])
        assert linear_prior_y.min() == 0.0 and linear_prior_y.max() == 1.0
        normalized_true_prior_y = (
            Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN)
        ) / (Y_TEST_FUNCTION(priorx).max() - Y_TEST_FUNCTION(priorx).min())
        priory = linear_prior_y * p + normalized_true_prior_y * (1 - p)

        pwl = GPWithPriorShape().fit(
            np.array([]), np.array([]), x, dx, dy, priorx, priory
        )
        # estimated_total_y = pwl(TEST_XMAX) - pwl(TEST_XMIN)
        # import matplotlib.pyplot as plt

        # mean, std = pwl.predict(priorx)
        # ci_lower = mean - 1.96 * std
        # ci_upper = mean + 1.96 * std

        # plt.plot(priorx, Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(priorx.min()), label="true")
        # plt.scatter(priorx, Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(priorx.min()), label="true")
        # plt.scatter(
        #     priorx,
        #     priory * (Y_TEST_FUNCTION(priorx).max() - Y_TEST_FUNCTION(priorx).min()),
        #     label="rescaled prior",
        # )
        # plt.plot(priorx, pwl(priorx) - pwl(priorx.min()), label="GP")
        # plt.scatter(priorx, pwl(priorx) - pwl(priorx.min()), label="GP")
        # plt.fill_between(
        #     priorx,
        #     ci_lower - mean.min(),
        #     ci_upper - mean.min(),
        #     color="gray",
        #     alpha=0.3,
        #     label="95% CI",
        # )
        # plt.plot(priorx, mean - mean.min(), label="GP mean")
        # if pwlprev is not None:
        #     plt.plot(priorx, pwlprev(priorx) - pwlprev(priorx.min()), label="previous", alpha=0.5)

        # plt.legend()
        # plt.show()

        # pwlprev = pwl

        curve_error = np.abs(
            (Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN))
            - (pwl(priorx) - pwl(TEST_XMIN))
        ).mean()
        assert curve_error < curve_error_prev
        curve_error_prev = curve_error

    assert curve_error < 0.02 * Y_TEST_FUNCTION(priorx).max()
