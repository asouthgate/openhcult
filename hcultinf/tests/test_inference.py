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


def test_convergence_true_curve_bad_prior():
    """Test that, as data increases, the curve estimate approaches the true curve."""
    curve_error_prev = 1e10
    pwlprev = None
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
        estimated_total_y = pwl(TEST_XMAX) - pwl(TEST_XMIN)
        import matplotlib.pyplot as plt

        pwlx = np.linspace(TEST_XMIN, TEST_XMAX, 1000)
        mean, std = pwl.predict(pwlx)
        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std

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

        pwlprev = pwl

        curve_error = np.abs(
            (Y_TEST_FUNCTION(pwlx) - Y_TEST_FUNCTION(TEST_XMIN))
            - (pwl(pwlx) - pwl(TEST_XMIN))
        ).mean()
        assert curve_error < curve_error_prev
        curve_error_prev = curve_error

    assert curve_error < 0.01 * Y_TEST_FUNCTION(pwlx).max()


def test_estimate_dy_partial_coverage_perfect_prior():
    mask_xlim_l = 1.0
    mask_xlim_u = 2.0
    x, dx, dy = _sample_data(
        TEST_XMIN,
        TEST_XMAX,
        TEST_DXMIN,
        TEST_DXMAX,
        TEST_NOISE_LEVEL,
        100,
        Y_TEST_FUNCTION,
    )

    # Mask the data to simulate partial coverage, but use a perfect prior that matches the true curve
    mask = np.logical_and(mask_xlim_l <= (x + dx), (x + dx) <= mask_xlim_u)
    x, dx, dy = x[mask], dx[mask], dy[mask]

    priorx = np.linspace(TEST_XMIN, TEST_XMAX, 100)  # Need a finely grained prior
    priory = Y_TEST_FUNCTION(priorx)

    true_total_y = Y_TEST_FUNCTION(TEST_XMAX) - Y_TEST_FUNCTION(TEST_XMIN)
    priory = (priory - priory.min()) / (priory.max() - priory.min())
    pwl = GPWithPriorShape().fit(np.array([]), np.array([]), x, dx, dy, priorx, priory)
    estimated_total_y_partial = pwl(TEST_XMAX) - pwl(TEST_XMIN)

    pwlx = np.linspace(TEST_XMIN, TEST_XMAX, 50)
    mean, std = pwl.predict(pwlx)
    ci_lower = mean - 1.96 * std
    ci_upper = mean + 1.96 * std

    import matplotlib.pyplot as plt

    plt.plot(pwlx, Y_TEST_FUNCTION(pwlx) - Y_TEST_FUNCTION(pwlx.min()), label="true")
    plt.scatter(pwlx, Y_TEST_FUNCTION(pwlx) - Y_TEST_FUNCTION(pwlx.min()), label="true")
    plt.scatter(
        priorx,
        priory * (Y_TEST_FUNCTION(pwlx).max() - Y_TEST_FUNCTION(pwlx).min()),
        label="rescaled prior",
    )
    plt.plot(pwlx, pwl(pwlx) - pwl(pwlx.min()), label="true")
    plt.scatter(pwlx, pwl(pwlx) - pwl(pwlx.min()), label="true")
    plt.fill_between(
        pwlx,
        ci_lower - mean.min(),
        ci_upper - mean.min(),
        color="gray",
        alpha=0.3,
        label="95% CI",
    )
    plt.plot(pwlx, mean - mean.min(), label="GP mean")
    plt.legend()
    plt.show()
    assert estimated_total_y_partial == pytest.approx(true_total_y, rel=0.02)
