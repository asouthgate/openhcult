import numpy as np
import pytest

from hcultinf.inference import (
    MonotonicSpline,
    MonotonicPWL,
    GP,
    estimate_swc_max,
    find_overlapping_groups,
    estimate_total_dy,
)
from tests.sim import (
    simulate_calibration,
    plot_calibration,
    invlogistic_swc_of_x,
    logistic_x_of_swc,
)


def test_no_noise_heavy_overlaps():
    """
    Test 1-50 samples with heavy overlaps (dx > 0.5 interval length).
    Should produce perfect results for any linear function.
    """
    m_true = 4.2
    for n in range(1, 51):
        samples = []
        # Sample from a fixed interval [0, 1]
        for _ in range(n):
            # Start anywhere, but make dx large to ensure heavy overlapping
            x = np.random.uniform(0, 0.4)
            dx = np.random.uniform(0.5, 0.6)  # Always > 0.5
            samples.append({"x": x, "dx": dx, "dy": m_true * dx})

        groups = find_overlapping_groups(samples)

        # Since dx > 0.5, they will all form one group in [0, 1]
        assert len(groups) == 1

        g = groups[0]
        actual_span = max(s["x"] + s["dx"] for s in g) - min(s["x"] for s in g)
        expected_dy = m_true * actual_span

        result = estimate_total_dy(g)
        print(expected_dy, result)
        assert result == pytest.approx(expected_dy, rel=1e-12)


def test_noiseless_accuracy():
    """Estimation methods recover the true curve within tight tolerance when there is no noise."""
    swc_max = 447.219
    rng = np.random.default_rng(42)
    x_0, x_1, x_starts, swc_starts, delta_x, delta_swc = simulate_calibration(
        n_chords=1000, x0_noise=0.0, x1_noise=0.0, x_noise=0.0, swc_max=swc_max, rng=rng
    )
    x_curve = logistic_x_of_swc(np.linspace(0.01, 0.99, 500))

    gp = GP().fit(
        np.array([x_0, x_1]), np.array([0.0, swc_max]), x_starts, delta_x, delta_swc
    )
    ms = MonotonicSpline(50, k=5).fit(
        np.array([x_0]), np.array([0.0]), x_starts, delta_x, delta_swc
    )
    pwl = MonotonicPWL(n_nodes=100).fit(
        np.array([x_0, x_1]), np.array([0.0, swc_max]), x_starts, delta_x, delta_swc
    )

    for method, tag in [(gp, "gp"), (ms, "ms"), (pwl, "pwl")]:
        plot_calibration(
            x_0,
            x_1,
            x_starts,
            swc_starts,
            delta_x,
            delta_swc,
            swc_max=swc_max,
            title="noiseless accuracy",
            inferred=ms,
            save_path=f"tests/artifacts/noiseless_accuracy_{tag}.png",
        )

        x_eval = np.linspace(x_0, x_1, 200)
        swc_true = invlogistic_swc_of_x(x_eval) * swc_max

        assert abs(method(x_0) - 0.0) < 0.02
        assert np.mean(np.abs(method(x_eval) - swc_true)) < 0.01 * swc_max
        assert np.all(np.diff(method(x_curve)) >= 0)


def test_estimate_swc_max_good_prior_any_coverage():
    """Good prior + 50% x-coverage: estimate is accurate despite partial data."""
    swc_max = 447.219
    rng = np.random.default_rng(42)

    # slide the mask
    for lower in [0.1, 0.3, 0.45]:
        upper = lower + 0.2
        x_0, x_1, x_starts, swc_starts, delta_x, delta_swc = simulate_calibration(
            n_chords=100,
            x0=lower,
            xend=upper,
            x0_noise=0.0,
            x1_noise=0.0,
            x_noise=0.0,
            swc_max=swc_max,
            rng=rng,
        )
        swc_max_est = estimate_swc_max(
            x_starts, delta_x, delta_swc, invlogistic_swc_of_x
        )
        assert abs(swc_max_est - swc_max) < 0.05 * swc_max  # 5% error overall

        plot_calibration(
            x_0,
            x_1,
            x_starts,
            swc_starts,
            delta_x,
            delta_swc,
            swc_max=swc_max,
            title=f"good prior, partial coverage — est={swc_max_est:.1f} true={swc_max}",
            swc_max_est=swc_max_est,
            # save_path="tests/artifacts/swc_max_good_prior_partial.png",
        )


def test_estimate_swc_max_bad_prior_full_coverage():
    """Bad prior (linear) + 100% coverage: estimate is data-driven, prior shape irrelevant."""
    swc_max = 447.219
    rng = np.random.default_rng(42)
    x_0, x_1, x_starts, swc_starts, delta_x, delta_swc = simulate_calibration(
        n_chords=500, x0_noise=0.0, x1_noise=0.0, x_noise=0.0, swc_max=swc_max, rng=rng
    )
    linear_prior = lambda x: (x - x_0) / (x_1 - x_0)
    swc_max_est = estimate_swc_max(x_starts, delta_x, delta_swc, linear_prior)

    plot_calibration(
        x_0,
        x_1,
        x_starts,
        swc_starts,
        delta_x,
        delta_swc,
        swc_max=swc_max,
        title=f"bad prior, 100% coverage — est={swc_max_est:.1f} true={swc_max}",
        swc_max_est=swc_max_est,
        save_path="tests/artifacts/swc_max_bad_prior_full.png",
    )

    assert abs(swc_max_est - swc_max) < 20.0


def test_estimate_swc_max_bad_prior_partial_coverage():
    """Bad prior + partial coverage: estimate fails — both conditions must not hold simultaneously."""
    swc_max = 447.219
    rng = np.random.default_rng(42)
    x_0, x_1, x_starts, swc_starts, delta_x, delta_swc = simulate_calibration(
        n_chords=500, x0_noise=0.0, x1_noise=0.0, x_noise=0.0, swc_max=swc_max, rng=rng
    )
    mask = (x_starts + delta_x) <= 0.3
    x_starts, delta_x, delta_swc, swc_starts = (
        x_starts[mask],
        delta_x[mask],
        delta_swc[mask],
        swc_starts[mask],
    )
    linear_prior = lambda x: (x - x_0) / (x_1 - x_0)
    swc_max_est = estimate_swc_max(x_starts, delta_x, delta_swc, linear_prior)

    plot_calibration(
        x_0,
        x_1,
        x_starts,
        swc_starts,
        delta_x,
        delta_swc,
        swc_max=swc_max,
        title=f"bad prior, 30% coverage — est={swc_max_est:.1f} true={swc_max}",
        swc_max_est=swc_max_est,
        save_path="tests/artifacts/swc_max_bad_prior_partial.png",
    )

    assert abs(swc_max_est - swc_max) > 100.0


def test_swc_max_estimate_improves_with_coverage():
    """Error decreases monotonically as coverage expands from center toward full range."""
    swc_max = 447.219
    rng = np.random.default_rng(42)
    x_0, x_1, x_starts_all, swc_starts_all, delta_x_all, delta_swc_all = (
        simulate_calibration(
            n_chords=500,
            x0_noise=0.0,
            x1_noise=0.0,
            x_noise=0.0,
            swc_max=swc_max,
            rng=rng,
        )
    )
    linear_prior = lambda x: (x - x_0) / (x_1 - x_0)

    half_widths = [0.1, 0.2, 0.3, 0.4, 0.5]
    errors = []
    for hw in half_widths:
        mask = (swc_starts_all >= (0.5 - hw) * swc_max) & (
            swc_starts_all + delta_swc_all <= (0.5 + hw) * swc_max
        )
        x_s, dx, dswc, swc_s = (
            x_starts_all[mask],
            delta_x_all[mask],
            delta_swc_all[mask],
            swc_starts_all[mask],
        )
        assert len(x_s) > 0, f"no chords for half_width={hw}"
        swc_max_est = estimate_swc_max(x_s, dx, dswc, linear_prior)
        errors.append(abs(swc_max_est - swc_max))

    assert all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))


def test_swc_max_estimate_improves_with_prior():
    """Error decreases monotonically as prior interpolates from linear toward true logistic."""
    import matplotlib.pyplot as plt

    swc_max = 447.219
    rng = np.random.default_rng(42)
    x_0, x_1, x_starts, swc_starts, delta_x, delta_swc = simulate_calibration(
        n_chords=50, x0_noise=0.0, x1_noise=0.0, x_noise=0.0, swc_max=swc_max, rng=rng
    )
    linear_prior = lambda x: (x - x_0) / (x_1 - x_0)
    x_curve = logistic_x_of_swc(np.linspace(0.01, 0.99, 500))

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    errors = []
    ax = None
    for alpha in alphas:
        prior = lambda x, a=alpha: a * invlogistic_swc_of_x(x) + (1 - a) * linear_prior(
            x
        )
        swc_max_est = estimate_swc_max(x_starts, delta_x, delta_swc, prior)
        errors.append(abs(swc_max_est - swc_max))
        fig, ax = plot_calibration(
            x_0,
            x_1,
            x_starts,
            swc_starts,
            delta_x,
            delta_swc,
            swc_max=swc_max,
            title=f"bad prior, 30% coverage — est={swc_max_est:.1f} true={swc_max}",
            swc_max_est=swc_max_est,
            ax=ax,
            legend=False,
        )
        ax.plot(x_curve, prior(x_curve) * swc_max_est)

    ax.legend()
    plt.show()
    assert all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))
