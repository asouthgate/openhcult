import numpy as np
import random
import time

Y_TEST_FUNCTION_NONORM = (
    lambda x: x**2 + np.log(x + 1) + np.exp(0.5 * x) + np.sqrt(x) - np.sin(3 * x) + 3.3
)

Y_TEST_FUNCTION_POWER = (
    lambda x, xmin, xmax: x**2
    + np.log(x + 1)
    + np.exp(0.5 * x)
    + np.sqrt(x)
    - np.sin(3 * x)
    + 3.3
)

Y_TEST_FUNCTION_NONORM_DECREASING = lambda x: -Y_TEST_FUNCTION_NONORM(x) + 50.0

Y_TEST_FUNCTION = lambda x: Y_TEST_FUNCTION_NONORM(x) - Y_TEST_FUNCTION_NONORM(0.0)
Y_TEST_FUNCTION_DECREASING = lambda x: Y_TEST_FUNCTION_NONORM_DECREASING(
    x
) - Y_TEST_FUNCTION_NONORM_DECREASING(0.0)


def power_function(x, power, y_int=0.0, xmin=0.0, xmax=1.0):
    scale = np.clip((xmax - x) / (xmax - xmin), 1e-10, 1.0)
    return (1.0 - y_int) * scale**power + y_int


def simulate_calibration_data_samples(
    xmin, xmax, dxmin, dxmax, noise_level, n, y, uniform=False
):
    if uniform:
        x = np.linspace(xmin, xmax, n)
    else:
        x = np.random.uniform(xmin, xmax, n)
    dx = -np.random.uniform(dxmin, dxmax, n)

    assert all(dx < 0), "dx should be negative (decreasing function)"

    x2 = np.clip(x + dx, xmin, xmax)
    dx = x2 - x

    # exclude the zero dx case to avoid zero division in lognormal noise
    x = x[dx < 0]
    dx = dx[dx < 0]

    dy = y(x + dx) - y(x)
    dy *= np.random.lognormal(-(noise_level**2) / 2, noise_level, len(dx))
    assert all(dx <= 0), "dx should be negative (decreasing function)"
    assert all(dy >= 0), "dy should be positive (decreasing function)"
    return x, dx, dy


def linear_response(content_frac, x0, xmax):
    """Linear sensor response: dry (content=0)→base, full (content=1)→wet."""
    return int(x0 - content_frac * (x0 - xmax))


def simulate_plant_moisture(
    max_swc_ml,
    base,
    wet,
    dose_frac_range=(0.3, 0.7),
    target_content_range=(0.15, 0.5),
    drain_per_day=0.08,
    noise=25,
    response_fn=None,
    rng=None,
    n_days=300,
    step_ms=30 * 60 * 1000,
):
    """Simulate plant watering holding content within target_content_range.

    Drains linearly at drain_per_day (fraction of content per day). Waters when content
    drops to target_content_range[0], adding a random dose drawn from dose_frac_range
    (as fraction of content). Sensor readings are derived via response_fn throughout.

    Returns (readings, watering_times, ml_amounts, before_readings, after_readings).
    ml_amounts are absolute ml (dose_frac * max_swc_ml).
    before/after_readings are sensor values at the moment of each watering event.
    """
    if response_fn is None:
        response_fn = lambda content: linear_response(content, base, wet)
    if rng is None:
        rng = random.Random(42)

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - n_days * 24 * 3600 * 1000
    dt_days = step_ms / (24 * 3600 * 1000)
    n_steps = (n_days * 24 * 3600 * 1000) // step_ms

    content = sum(target_content_range) / 2
    readings, watering_times, ml_amounts, before_contents, after_contents = (
        [],
        [],
        [],
        [],
        [],
    )
    next_trigger = rng.uniform(*target_content_range)

    for i in range(n_steps):
        t = start_ms + i * step_ms
        content = max(0.0, content - drain_per_day * dt_days)
        if content <= next_trigger:
            before_contents.append(content)
            watering_times.append(t)
            dose = rng.uniform(*dose_frac_range)
            content = min(1.0, content + dose)
            after_contents.append(content)
            ml_amounts.append(dose * max_swc_ml)
            next_trigger = rng.uniform(*target_content_range)
        mv = max(wet, min(base, response_fn(content) + rng.randint(-noise, noise)))
        readings.append((t, mv, mv))

    before_readings = [response_fn(content) for content in before_contents]
    after_readings = [response_fn(content) for content in after_contents]
    return readings, watering_times, ml_amounts, before_readings, after_readings
