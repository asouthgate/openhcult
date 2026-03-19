import numpy as np
import pandas as pd

from hcultinf.training import compute_loss, match_events


def test_match_events_exact():
    t = np.datetime64("2026-03-17T10:00:00", "ms")
    confirmed = [(1, t, "WATER manual")]
    result = match_events(confirmed, [t])
    assert result == {"matched": 1, "missed": 0, "spurious": 0}


def test_match_events_within_epsilon():
    t = np.datetime64("2026-03-17T10:00:00", "ms")
    candidate = t + np.timedelta64(10 * 60 * 1000, "ms")  # +10 min
    confirmed = [(1, t, "WATER manual")]
    result = match_events(confirmed, [candidate], epsilon_ms=15 * 60 * 1000)
    assert result == {"matched": 1, "missed": 0, "spurious": 0}


def test_match_events_outside_epsilon():
    t = np.datetime64("2026-03-17T10:00:00", "ms")
    candidate = t + np.timedelta64(20 * 60 * 1000, "ms")  # +20 min
    confirmed = [(1, t, "WATER manual")]
    result = match_events(confirmed, [candidate], epsilon_ms=15 * 60 * 1000)
    assert result == {"matched": 0, "missed": 1, "spurious": 1}


def test_match_events_spurious():
    confirmed = []
    t = np.datetime64("2026-03-17T10:00:00", "ms")
    result = match_events(confirmed, [t, t + np.timedelta64(3600_000, "ms")])
    assert result == {"matched": 0, "missed": 0, "spurious": 2}


def test_compute_loss_perfect_detection():
    times = pd.date_range("2026-03-17 10:00:00", periods=120, freq="1min").values
    values = np.concatenate(
        [
            np.full(40, 3000.0),
            np.linspace(3000, 1000, 20),
            np.full(60, 1000.0),
        ]
    )
    timeseries = {"dev:s1": list(zip(times, values))}

    detector_params = {
        "emwa_tau_minutes": 5,
        "trigger_thresh": -5.0,
        "release_thresh": -1.0,
    }

    event_time = np.datetime64("2026-03-17T10:40:00", "ms")
    confirmed = [(1, event_time, "WATER manual")]

    result = compute_loss(confirmed, timeseries, detector_params)

    assert result["matched"] >= 1
    assert result["loss"] < 1.0


def test_compute_loss_no_confirmed_events():
    times = pd.date_range("2026-03-17 10:00:00", periods=120, freq="1min").values
    values = np.concatenate(
        [
            np.full(40, 3000.0),
            np.linspace(3000, 1000, 20),
            np.full(60, 1000.0),
        ]
    )
    timeseries = {"dev:s1": list(zip(times, values))}
    detector_params = {
        "emwa_tau_minutes": 5,
        "trigger_thresh": -5.0,
        "release_thresh": -1.0,
    }

    result = compute_loss([], timeseries, detector_params)

    assert result["matched"] == 0
    assert result["missed"] == 0


def test_compute_loss_no_detections():
    times = pd.date_range("2026-03-17 10:00:00", periods=120, freq="1min").values
    values = np.full(120, 2000.0)
    timeseries = {"dev:s1": list(zip(times, values))}
    detector_params = {
        "emwa_tau_minutes": 5,
        "trigger_thresh": -5.0,
        "release_thresh": -1.0,
    }

    event_time = np.datetime64("2026-03-17T10:30:00", "ms")
    confirmed = [(1, event_time, "WATER manual")]

    result = compute_loss(confirmed, timeseries, detector_params)

    assert result["matched"] == 0
    assert result["missed"] == 1
    assert result["loss"] == 1.0
