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


def test_convergence_in_n_bad_prior():
    """Test that, as data increases, the curve estimate approaches the true curve."""
    preverrs = []
    last_pwl = last_x = last_dx = last_dy = None
    for n in [4, 32, 256]:
        priorx_pts = np.array([TEST_XMIN, TEST_XMAX])
        priory_pts = np.array([0.0, 1.0])
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
                np.array([TEST_XMIN]),
                np.array([0.0]),
                x,
                dx,
                dy,
                priorx_pts,
                priory_pts,
            )
            pwlx = np.linspace(TEST_XMIN, TEST_XMAX, 2000)
            _curve_error = np.abs(
                (Y_TEST_FUNCTION(pwlx) - Y_TEST_FUNCTION(TEST_XMIN))
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
        [TEST_XMIN],
        [0.0],
        last_x,
        last_dx,
        last_dy,
        true_y=Y_TEST_FUNCTION(plot_x) - Y_TEST_FUNCTION(TEST_XMIN),
        out="artifacts/convergence_n_bad_prior.png",
        title="Test convergence in n with bad prior",
    )

    assert all(
        np.diff(preverrs) < 0
    ), f"Curve error did not decrease with increasing n: {preverrs}"
    assert curve_error < 0.02 * Y_TEST_FUNCTION(pwlx).max()


def test_convergence_in_prior_low_n():
    """Test that, as data increases, the curve estimate approaches the true curve."""
    n = 4
    x, dx, dy = _sample_data(
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
    )

    assert all(
        np.diff(preverrs) < 0
    ), f"Curve error did not decrease with increasing p: {preverrs}"
    assert curve_error < 0.02 * Y_TEST_FUNCTION(priorx).max()


def test_performance_realistic_parameters():
    """Test that, as data increases, the curve estimate approaches the true curve."""
    n = 10
    x, dx, dy = _sample_data(1.5, 3.5, 0.1, 0.9, 1.0, n, Y_TEST_FUNCTION)
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
    )

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
    )

    assert all(
        np.diff(prevstds) < 0
    ), f"Curve std did not decrease with increasing delta size: {prevstds}"
    assert all(
        np.diff(preverrs) < 0
    ), f"Curve error did not decrease with increasing delta size: {preverrs}"
