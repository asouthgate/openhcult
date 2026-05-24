import os

import numpy as np
import pandas as pd
import pytest

from hcultinf.drying import compute_drying_rate
from hcultinf.simulation import simulate_plant_moisture


def _ms_to_datetime64(times_ms):
    return pd.to_datetime(times_ms, unit="ms").values


def _simulate_drying_data(n_days=60):
    readings, watering_times, ml_amounts, before_vals, after_vals = (
        simulate_plant_moisture(
            max_swc_ml=75.0,
            base=3000,
            wet=1000,
            dose_frac_range=(0.4, 0.8),
            target_content_range=(0.05, 0.2),
            drain_per_day=0.1,
            noise=25,
            n_days=n_days,
            step_ms=30 * 60 * 1000,
        )
    )
    times_ms = np.array([t for t, _, _ in readings])
    values = np.array([mv for _, _, mv in readings], dtype=float)
    times = _ms_to_datetime64(times_ms)
    return times, values, watering_times


def test_drying_rate_detects_drying():
    times, values, watering_times = _simulate_drying_data()

    result = compute_drying_rate(times, values)

    resampled_times = result["times"]
    rate = result["rate"]

    assert len(resampled_times) > 0
    assert len(rate) == len(resampled_times)

    mean_rate = np.mean(rate)
    assert mean_rate < 0, f"Expected negative drying rate, got {mean_rate}"


def test_drying_rate_too_few_points_raises():
    t = pd.date_range("2026-03-17", periods=1, freq="1min").values
    v = np.array([2000.0])
    with pytest.raises(ValueError, match="at least 2"):
        compute_drying_rate(t, v)


def test_drying_rate_custom_lambda():
    times, values, _ = _simulate_drying_data(n_days=10)

    result_small = compute_drying_rate(times, values, lambda_tv=0.1)
    result_large = compute_drying_rate(times, values, lambda_tv=1000.0)

    small_var = np.var(result_small["rate"])
    large_var = np.var(result_large["rate"])
    assert (
        small_var > large_var
    ), "Higher lambda should produce smoother (less variable) rates"


def test_drying_rate_plot():
    times, values, watering_times = _simulate_drying_data(n_days=30)

    result = compute_drying_rate(times, values)

    import matplotlib.pyplot as plt
    from hcultinf.plot_style import apply_dark_theme, ORANGE, MUTED

    apply_dark_theme()

    rt = result["times"]
    rate = result["rate"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.scatter(times, values, s=1, alpha=0.3, color=MUTED, label="raw")
    ax1.set_ylabel("sensor value")
    ax1.legend()

    ax2.plot(rt, rate, color=ORANGE, label="rate (trend filtered)")
    ax2.axhline(0, color=MUTED, linewidth=0.5)
    ax2.set_ylabel("rate")
    ax2.legend()

    fig.suptitle("Drying rate")
    fig.tight_layout()

    os.makedirs("artifacts", exist_ok=True)
    fig.savefig("artifacts/drying_rate.png")
    if os.environ.get("HCULT_TEST_DEBUG_PLOT", "0") == "1":
        plt.show()
    plt.close(fig)
