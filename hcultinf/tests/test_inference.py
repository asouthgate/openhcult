import numpy as np
import pytest

from hcultinf.inference import GPWithPriorShape

Y_TEST_FUNCTION_NONORM = (
    lambda x: x**2 + np.log(x + 1) + np.exp(0.5 * x) + np.sqrt(x) - np.sin(3 * x) + 3.3
)

Y_TEST_FUNCTION = lambda x: Y_TEST_FUNCTION_NONORM(x) - Y_TEST_FUNCTION_NONORM(0.0)

TEST_XMIN = 0.0
TEST_XMAX = 5.5
TEST_DXMIN = 0.2
TEST_DXMAX = 0.5
TEST_NOISE_LEVEL = 0.5


def _sample_data(xmin, xmax, dxmin, dxmax, noise_level, n, y, uniform=False):
    if uniform:
        x = np.linspace(xmin, xmax, n)
    else:
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
    for n in [4, 32, 256]:  # Square the number of points, 5 chosen for convenience
        # Use linear prior (bad)
        priorx = np.array([TEST_XMIN, TEST_XMAX])
        priory = np.array([0.0, 1.0])
        errs = []
        for _ in range(5):
            x, dx, dy = _sample_data(
                TEST_XMIN,
                TEST_XMAX,
                TEST_DXMIN,
                TEST_DXMAX,
                TEST_NOISE_LEVEL,
                n,
                Y_TEST_FUNCTION,
            )

            pwl = GPWithPriorShape(length_scale=1.0).fit(
                np.array([TEST_XMIN]), np.array([0.0]), x, dx, dy, priorx, priory
            )

            pwlx = np.linspace(TEST_XMIN, TEST_XMAX, 2000)

            _curve_error = np.abs(
                (Y_TEST_FUNCTION(pwlx) - Y_TEST_FUNCTION(TEST_XMIN))
                - (pwl(pwlx) - pwl(TEST_XMIN))
            ).mean()
            errs.append(_curve_error)
        curve_error = np.mean(errs)
        assert (
            curve_error < curve_error_prev
        ), f"n={n} had curve error {curve_error} which is not less than previous {curve_error_prev}"
        curve_error_prev = curve_error
    assert curve_error < 0.02 * Y_TEST_FUNCTION(pwlx).max()


def test_convergence_in_prior_low_n():
    """Test that, as data increases, the curve estimate approaches the true curve."""
    curve_error_prev = 1e10
    # pwlprev = None
    n = 4  # has to be high enough that it is much more likely to fit real prior
    pwlprev = None
    x, dx, dy = _sample_data(
        TEST_XMIN,
        TEST_XMAX,
        TEST_DXMIN,
        TEST_DXMAX,
        TEST_NOISE_LEVEL
        * 0.1,  # reduce noise to make it more likely to fit the true prior
        n,
        Y_TEST_FUNCTION,
        uniform=True,
    )

    for p in [1.0, 2 / 3, 1 / 3, 0.0]:
        priorx = np.linspace(TEST_XMIN, TEST_XMAX, 10000)
        linear_prior_y = np.interp(priorx, [TEST_XMIN, TEST_XMAX], [0.0, 1.0])
        assert linear_prior_y.min() == 0.0 and linear_prior_y.max() == 1.0
        normalized_true_prior_y = (
            Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN)
        ) / (Y_TEST_FUNCTION(priorx).max() - Y_TEST_FUNCTION(priorx).min())
        assert (
            normalized_true_prior_y.min() == 0.0
            and normalized_true_prior_y.max() == 1.0
        )
        priory = linear_prior_y * p + normalized_true_prior_y * (1 - p)

        # for _ in range(5):
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
        # print(curve_error)

        # import matplotlib.pyplot as plt

        # mean, std = pwl.predict(priorx)
        # ci_lower = mean - 1.96 * std
        # ci_upper = mean + 1.96 * std

        # plt.scatter(x, Y_TEST_FUNCTION(x), label="data points")
        # plt.plot(priorx, Y_TEST_FUNCTION(priorx), label="true")
        # plt.plot(
        #     priorx,
        #     priory * (Y_TEST_FUNCTION(priorx).max() - Y_TEST_FUNCTION(priorx).min()),
        #     label="rescaled prior",
        # )
        # plt.plot(priorx, pwl(priorx) - pwl(priorx.min()), label="GP")
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
        #     plt.plot(priorx, pwlprev(priorx) - pwlprev(priorx.min()), label="prev")

        # plt.legend()
        # plt.show()

        assert (
            curve_error < curve_error_prev
        ), f"p={p} had curve error {curve_error} which is not less than previous {curve_error_prev}"
        curve_error_prev = curve_error
        pwlprev = pwl

    assert curve_error < 0.02 * Y_TEST_FUNCTION(priorx).max()


# def test_performance_realistic_parameters():
#     """Test that, as data increases, the curve estimate approaches the true curve."""
#     # pwlprev = None
#     n = 200
#     x, dx, dy = _sample_data(
#         TEST_XMIN,
#         2.0,
#         TEST_DXMIN,
#         TEST_DXMAX,
#         0.1,
#         n,
#         Y_TEST_FUNCTION,
#     )
#     priorx = np.linspace(TEST_XMIN, TEST_XMAX, 1000)
#     linear_prior_y = np.interp(priorx, [TEST_XMIN, TEST_XMAX], [0.0, 1.0])
#     assert linear_prior_y.min() == 0.0 and linear_prior_y.max() == 1.0
#     normalized_true_prior_y = (Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN)) / (
#         Y_TEST_FUNCTION(priorx).max() - Y_TEST_FUNCTION(priorx).min()
#     )
#     priory = linear_prior_y * 0.5 + normalized_true_prior_y * (1 - 0.5)

#     pwl = GPWithPriorShape().fit(
#         np.array([TEST_XMIN]),
#         np.array([0.0]),
#         x,
#         dx,
#         dy,
#         priorx,
#         priory,
#         prior_variance=5.0,
#     )
#     estimated_total_y = pwl(TEST_XMAX) - pwl(TEST_XMIN)
#     import matplotlib.pyplot as plt

#     mean, std = pwl.predict(priorx)
#     ci_lower = mean - 1.96 * std
#     ci_upper = mean + 1.96 * std

#     plt.scatter(x, Y_TEST_FUNCTION(x) - Y_TEST_FUNCTION(x.min()), label="data points")
#     plt.plot(
#         priorx, Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(priorx.min()), label="true"
#     )
#     plt.plot(
#         priorx,
#         priory * (Y_TEST_FUNCTION(priorx).max() - Y_TEST_FUNCTION(priorx).min()),
#         label="rescaled prior",
#     )
#     plt.plot(priorx, pwl(priorx) - pwl(priorx.min()), label="GP")
#     plt.fill_between(
#         priorx,
#         ci_lower - mean.min(),
#         ci_upper - mean.min(),
#         color="gray",
#         alpha=0.3,
#         label="95% CI",
#     )
#     plt.plot(priorx, mean - mean.min(), label="GP mean")

#     plt.legend()
#     plt.show()

#     curve_error = np.abs(
#         (Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN))
#         - (pwl(priorx) - pwl(TEST_XMIN))
#     ).mean()

#     # assert curve_error < 0.02 * Y_TEST_FUNCTION(priorx).max()
