import numpy as np
import pandas as pd

from hcultinf.training import grid_search, match_events


def test_match_events_exact():
    t = np.datetime64("2026-03-17T10:00:00", "ms")
    confirmed = [(1, t, "WATER manual")]
    result = match_events(confirmed, [t])
    assert result == {"matched": 1, "missed": 0, "spurious": 0}


def test_match_events_within_epsilon():
    t = np.datetime64("2026-03-17T10:00:00", "ms")
    candidate = t + np.timedelta64(10 * 60 * 1000, "ms")
    confirmed = [(1, t, "WATER manual")]
    result = match_events(confirmed, [candidate], epsilon_ms=15 * 60 * 1000)
    assert result == {"matched": 1, "missed": 0, "spurious": 0}


def test_match_events_outside_epsilon():
    t = np.datetime64("2026-03-17T10:00:00", "ms")
    candidate = t + np.timedelta64(20 * 60 * 1000, "ms")
    confirmed = [(1, t, "WATER manual")]
    result = match_events(confirmed, [candidate], epsilon_ms=15 * 60 * 1000)
    assert result == {"matched": 0, "missed": 1, "spurious": 1}


def test_match_events_spurious():
    confirmed = []
    t = np.datetime64("2026-03-17T10:00:00", "ms")
    result = match_events(confirmed, [t, t + np.timedelta64(3600_000, "ms")])
    assert result == {"matched": 0, "missed": 0, "spurious": 2}


def _make_timeseries_with_event():
    times = pd.date_range("2026-03-17 10:00:00", periods=120, freq="1min").values
    values = np.concatenate(
        [
            np.full(40, 3000.0),
            np.linspace(3000, 1000, 20),
            np.full(60, 1000.0),
        ]
    )
    return {"dev:s1": list(zip(times, values))}


def test_grid_search_detects_event():
    timeseries = _make_timeseries_with_event()
    event_time = np.datetime64("2026-03-17T10:40:00", "ms")
    confirmed = [(1, event_time, "WATER manual")]
    param_grid = [
        {"emwa_tau_minutes": 5, "trigger_thresh": -5.0, "release_thresh": -1.0}
    ]

    results = grid_search(confirmed, timeseries, param_grid)

    assert len(results) == 1
    assert results[0]["tp"] >= 1
    assert results[0]["f1"] > 0


def test_grid_search_no_confirmed():
    timeseries = _make_timeseries_with_event()
    results = grid_search(
        [],
        timeseries,
        [{"emwa_tau_minutes": 5, "trigger_thresh": -5.0, "release_thresh": -1.0}],
    )
    assert results[0]["tp"] == 0
    assert results[0]["fn"] == 0


def test_grid_search_no_detections():
    times = pd.date_range("2026-03-17 10:00:00", periods=120, freq="1min").values
    values = np.full(120, 2000.0)
    timeseries = {"dev:s1": list(zip(times, values))}
    event_time = np.datetime64("2026-03-17T10:30:00", "ms")
    confirmed = [(1, event_time, "WATER manual")]

    results = grid_search(
        confirmed,
        timeseries,
        [{"emwa_tau_minutes": 5, "trigger_thresh": -5.0, "release_thresh": -1.0}],
    )

    assert results[0]["tp"] == 0
    assert results[0]["fn"] == 1
    assert results[0]["f1"] == 0.0
