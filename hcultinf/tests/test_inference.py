import numpy as np
import os

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

# See if debug flag is set
DEBUG_PLOT = os.getenv("TEST_DEBUG_PLOT", "0") == "1"


def _get_mixed_prior(xmin, xmax, p):
    priorx = np.linspace(xmin, xmax, 1000)
    linear_prior_y = np.interp(priorx, [TEST_XMIN, TEST_XMAX], [0.0, 1.0])
    assert linear_prior_y.min() == 0.0 and linear_prior_y.max() == 1.0
    normalized_true_prior_y = (Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN)) / (
        Y_TEST_FUNCTION(priorx).max() - Y_TEST_FUNCTION(priorx).min()
    )
    priory = linear_prior_y * p + normalized_true_prior_y * (1 - p)
    return priorx, priory


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


def _debug_plot(
    pwl,
    priorx,
    priory,
    anchors_x,
    anchors_y,
    x,
    dx,
    dy,
    pwlprevs=None,
    artifact_name=None,
):

    if not DEBUG_PLOT and artifact_name is None:
        return

    import matplotlib.pyplot as plt

    mean, std = pwl.predict(priorx)
    ci_lower = mean - 1.96 * std
    ci_upper = mean + 1.96 * std

    for i in range(len(dx)):
        plt.scatter(
            [x[i], x[i] + dx[i]],
            [pwl(x[i]) - pwl(TEST_XMIN), pwl(x[i]) - pwl(TEST_XMIN) + dy[i]],
        )
        plt.plot(
            [x[i], x[i] + dx[i]],
            [pwl(x[i]) - pwl(TEST_XMIN), pwl(x[i]) - pwl(TEST_XMIN) + dy[i]],
        )

    plt.scatter(
        anchors_x,
        anchors_y,
        label="anchors",
    )
    plt.plot(
        priorx, Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(priorx.min()), label="true"
    )
    plt.plot(
        priorx,
        priory * (pwl(priorx).max() - pwl(priorx).min()),
        label="rescaled prior",
    )
    plt.plot(priorx, pwl(priorx) - pwl(priorx.min()), label="GP")
    plt.fill_between(
        priorx,
        ci_lower - mean.min(),
        ci_upper - mean.min(),
        color="gray",
        alpha=0.3,
        label="95% CI",
    )
    plt.plot(priorx, mean - mean.min(), label="GP mean")
    if pwlprevs is not None:
        for i, pwlprev in enumerate(pwlprevs):
            plt.plot(
                priorx,
                pwlprev(priorx) - pwlprev(priorx.min()),
                label=f"prev{i}",
                alpha=0.5,
                color="brown",
                linestyle="--",
            )

    if artifact_name is not None:
        # create artifacts directory
        if not os.path.exists("artifacts"):
            os.makedirs("artifacts")
        plt.savefig(f"artifacts/debug_plot_{artifact_name}.png")

    if DEBUG_PLOT:
        plt.show()

    plt.legend()


def test_convergence_in_n_bad_prior():
    """Test that, as data increases, the curve estimate approaches the true curve."""
    preverrs = []
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
        preverrs.append(curve_error)

    assert all(
        np.diff(preverrs) < 0
    ), f"Curve error did not decrease with increasing n: {preverrs}"
    assert curve_error < 0.02 * Y_TEST_FUNCTION(pwlx).max()


def test_convergence_in_prior_low_n():
    """Test that, as data increases, the curve estimate approaches the true curve."""
    # pwlprev = None
    n = 4  # has to be high enough that it is much more likely to fit real prior
    x, dx, dy = _sample_data(
        TEST_XMIN,
        TEST_XMAX,
        TEST_DXMIN,
        TEST_DXMAX,
        TEST_NOISE_LEVEL
        * 0.1,  # reduce noise to make it more likely to fit the true prior
        n,
        Y_TEST_FUNCTION,
        uniform=True,  # use uniform sampling to make it more likely to fit the true prior
    )
    pwlprevs = []
    preverrs = []

    for p in [1.0, 2 / 3, 1 / 3, 0.0]:
        priorx, priory = _get_mixed_prior(TEST_XMIN, TEST_XMAX, p)

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

        pwlprevs.append(pwl)
        preverrs.append(curve_error)

    assert all(
        np.diff(preverrs) < 0
    ), f"Curve error did not decrease with increasing p: {preverrs}"
    assert curve_error < 0.02 * Y_TEST_FUNCTION(priorx).max()


def test_performance_realistic_parameters():
    """Test that, as data increases, the curve estimate approaches the true curve."""
    # pwlprev = None
    n = 10
    x, dx, dy = _sample_data(
        1.5,
        3.5,
        0.1,
        0.9,
        1.0,
        n,
        Y_TEST_FUNCTION,
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
    _debug_plot(
        pwl,
        priorx,
        priory,
        anchors_x,
        anchors_y,
        x,
        dx,
        dy,
        pwlprevs=None,
        artifact_name="realistic_parameters",
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
        x, dx, dy = _sample_data(
            TEST_XMIN + ((TEST_XMAX - TEST_XMIN) / 2.0) - ddx,
            TEST_XMIN + ((TEST_XMAX - TEST_XMIN) / 2.0) + ddx,
            0.5,
            0.5,
            0.0,
            n,
            Y_TEST_FUNCTION,
            uniform=True,  # use uniform sampling to make it more likely to fit the true prior
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
        # _debug_plot(
        #     pwl, priorx, priory, anchors_x, anchors_y, x, dx, dy, pwlprevs=pwlprevs
        # )
        prevstds.append(
            pwl.predict(priorx)[1].mean()
        )  # this is mean of stds, not mean trend
        pwlprevs.append(pwl)
        curve_error = np.abs(
            (Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN))
            - (pwl(priorx) - pwl(TEST_XMIN))
        ).mean()
        preverrs.append(curve_error)

    assert all(
        np.diff(prevstds) < 0
    ), f"Curve std did not decrease with increasing delta size: {prevstds}"
    assert all(
        np.diff(preverrs) < 0
    ), f"Curve error did not decrease with increasing delta size: {preverrs}"


def test_total_estimate_improves_and_std_shrinks_with_delta_size():
    """Test that, as data increases, the curve estimate approaches the true curve."""
    pwlprevs = []
    preverrs = []
    prevstds = []
    n = 50
    for dx_max in np.linspace(0.01, 2.0, 4):
        x, dx, dy = _sample_data(
            1.5,
            3.5,
            dx_max / 2.0,
            dx_max,
            0.0,
            n,
            Y_TEST_FUNCTION,
            uniform=True,  # use uniform sampling to make it more likely to fit the true prior
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
        _debug_plot(
            pwl, priorx, priory, anchors_x, anchors_y, x, dx, dy, pwlprevs=pwlprevs
        )
        prevstds.append(
            pwl.predict(priorx)[1].mean()
        )  # this is mean of stds, not mean trend
        pwlprevs.append(pwl)
        curve_error = np.abs(
            (Y_TEST_FUNCTION(priorx) - Y_TEST_FUNCTION(TEST_XMIN))
            - (pwl(priorx) - pwl(TEST_XMIN))
        ).mean()
        preverrs.append(curve_error)

    assert all(
        np.diff(prevstds) < 0
    ), f"Curve std did not decrease with increasing delta size: {prevstds}"
    assert all(
        np.diff(preverrs) < 0
    ), f"Curve error did not decrease with increasing delta size: {preverrs}"
