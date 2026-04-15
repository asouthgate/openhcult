import numpy as np
import random
import time

Y_TEST_FUNCTION_NONORM = (
    lambda x: x**2 + np.log(x + 1) + np.exp(0.5 * x) + np.sqrt(x) - np.sin(3 * x) + 3.3
)

Y_TEST_FUNCTION_NONORM_DECREASING = lambda x: -Y_TEST_FUNCTION_NONORM(x) + 50.0

Y_TEST_FUNCTION = lambda x: Y_TEST_FUNCTION_NONORM(x) - Y_TEST_FUNCTION_NONORM(0.0)
Y_TEST_FUNCTION_DECREASING = lambda x: Y_TEST_FUNCTION_NONORM_DECREASING(
    x
) - Y_TEST_FUNCTION_NONORM_DECREASING(0.0)


def simulate_calibration_data_samples(
    xmin, xmax, dxmin, dxmax, noise_level, n, y, uniform=False
):
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


def linear_response(fc_frac, x0, xmax):
    """Linear sensor response: dry (WC=0)→base, full FC (WC=1)→wet."""
    return int(x0 - fc_frac * (x0 - xmax))


def simulate_plant_moisture(
    max_swc_ml,
    base,
    wet,
    dose_frac_range=(0.3, 0.7),
    target_fc_range=(0.15, 0.5),
    drain_per_day=0.08,
    noise=25,
    response_fn=None,
    rng=None,
    n_days=300,
    step_ms=30 * 60 * 1000,
):
    """Simulate plant watering holding FC within target_fc_range.

    Drains linearly at drain_per_day (fraction of FC per day). Waters when FC
    drops to target_fc_range[0], adding a random dose drawn from dose_frac_range
    (as fraction of FC). Sensor readings are derived via response_fn throughout.

    Returns (readings, watering_times, ml_amounts, before_readings, after_readings).
    ml_amounts are absolute ml (dose_frac * max_swc_ml).
    before/after_readings are sensor values at the moment of each watering event.
    """
    if response_fn is None:
        response_fn = lambda fc: linear_response(fc, base, wet)
    if rng is None:
        rng = random.Random(42)

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - n_days * 24 * 3600 * 1000
    dt_days = step_ms / (24 * 3600 * 1000)
    n_steps = (n_days * 24 * 3600 * 1000) // step_ms

    fc = sum(target_fc_range) / 2
    readings, watering_times, ml_amounts, before_fcs, after_fcs = [], [], [], [], []
    next_trigger = rng.uniform(*target_fc_range)

    for i in range(n_steps):
        t = start_ms + i * step_ms
        fc = max(0.0, fc - drain_per_day * dt_days)
        if fc <= next_trigger:
            before_fcs.append(fc)
            watering_times.append(t)
            dose = rng.uniform(*dose_frac_range)
            fc = min(1.0, fc + dose)
            after_fcs.append(fc)
            ml_amounts.append(dose * max_swc_ml)
            next_trigger = rng.uniform(*target_fc_range)
        mv = max(wet, min(base, response_fn(fc) + rng.randint(-noise, noise)))
        readings.append((t, mv, mv))

    before_readings = [response_fn(fc) for fc in before_fcs]
    after_readings = [response_fn(fc) for fc in after_fcs]
    return readings, watering_times, ml_amounts, before_readings, after_readings
