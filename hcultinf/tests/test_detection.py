import pytest
import numpy as np
import pandas as pd
from hcultinf.detection import DynamicIntervalInfo, \
    merge_intervals, lerp_thresholds, \
    find_regions_with_hysteresis_adapative_thresh, compute_time_weighted_ewma, \
    get_complementary_intervals, SegmentDetector
from scipy.optimize import curve_fit
from scipy.integrate import quad
# Ensure these are imported in your source file as well

def test_lognormal_fit_and_active_interval():
    """
    Simulates a negative lognormal velocity signal and verifies 
    that the class fits it and identifies the active interval.
    """
    fs = 100
    duration_sec = 10
    t_num = np.linspace(0, duration_sec, fs * duration_sec)
    t_start = pd.Timestamp('2026-03-17 10:00:00')
    t_values = (t_start + pd.to_timedelta(t_num, unit='s')).values
    
    def true_model(t_val, h, x_p, w, shift):
        t_shifted = t_val - shift
        res = np.zeros_like(t_val)
        mask = t_shifted > 0.001 # Avoid log(0)
        exponent = -(np.log(t_shifted[mask] / x_p)**2) / (2 * w**2)
        res[mask] = h * np.exp(exponent)
        return res

    vel_sim = true_model(t_num, h=-10.0, x_p=1.5, w=0.4, shift=2.0)
    # x_values are required for init but not used for the lognormal fit
    x_dummy = np.zeros_like(vel_sim)
    
    interval = DynamicIntervalInfo(t_values, x_dummy, vel_sim, at_equilibrium=False)
    assert interval.pred_func is not None
    assert "h" in interval.pred_params_d
    assert interval.gof_d['rmse'] < 0.1 
    
    t0_active, tend_active = interval.get_model_active_interval(threshold_pct=0.05)
    
    start_offset = (t0_active - t_start) / np.timedelta64(1, 's')
    end_offset = (tend_active - t_start) / np.timedelta64(1, 's')
    
    assert 1.5 <= start_offset <= 2.5
    assert end_offset > 4.0
    assert end_offset > start_offset

def test_initialization_and_stats():
    t_values = pd.to_datetime(['2026-03-17 10:00:00', '2026-03-17 10:00:01', 
                               '2026-03-17 10:00:02', '2026-03-17 10:00:03', 
                               '2026-03-17 10:00:04']).values
    x_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    vel = np.diff(x_values, prepend=0)
    
    interval = DynamicIntervalInfo(t_values, x_values, vel, at_equilibrium=True)
    
    assert interval.n_samples == 5
    assert interval.mean == 30.0
    assert interval.max == 50.0
    assert interval.min == 10.0
    assert interval.duration == np.timedelta64(4, 's')
    assert pytest.approx(interval.m, rel=1e-3) == 0.01


def test_merge_intervals_complex_nesting():
    list_a = [(0, 10)]
    list_b = [(5, 15), (20, 25)]
    list_c = [(10, 12), (24, 30)]
    result = merge_intervals([list_a, list_b, list_c])
    assert result == [(0, 15), (20, 30)]

def test_lerp_thresholds_scaling():
    val = np.array([10, 55, 100]) # min, mid, max
    t_arr, r_arr = lerp_thresholds(val, max_val=100, min_val=10, 
                                   trigger_high=10, release_high=5, 
                                   trigger_low=2, release_low=1)
    assert np.allclose(t_arr, [2.0, 6.0, 10.0])
    assert np.allclose(r_arr, [1.0, 3.0, 5.0])


def test_hysteresis_negative_direction_and_trailing_active():
    times = np.arange(6)
    val = np.array([-10, -15, -8, -13, -11, -11])
    trigger = np.array([-12] * 6)
    release = np.array([-9] * 6)
    regions = find_regions_with_hysteresis_adapative_thresh(times, val, trigger, release, direction=-1)
    assert regions == [(1, 2), (3, 5)]


def test_ewma_time_gap_sensitivity():
    times = pd.to_datetime(['2026-01-01 10:00', '2026-01-01 10:10', '2026-01-01 11:50']).values
    values = np.array([10.0, 20.0, 20.0])
    tau = 30.0
    smoothed = compute_time_weighted_ewma(times, values, tau_minutes=tau)
    assert smoothed[1] < 15.0
    assert smoothed[2] > 19.0


def test_complementary_intervals_boundary_check():
    intervals = [(10, 20), (21, 30)]
    result = get_complementary_intervals(intervals, start_bound=0, end_bound=100)
    assert result == [[0, 10], [31, 100]]

def test_segment_detector_integration_flow():
    """
    Integration test: Verifies that a sharp drop in values triggers a 
    disequilibrium interval and that the resulting interval is 
    correctly classified as a 'watering event'.
    """
    times = pd.date_range("2026-03-17 10:00:00", periods=120, freq="1min").values
    
    values = np.concatenate([
        np.full(40, 3000.0),
        np.linspace(3000, 1000, 20),
        np.full(60, 1000.0)
    ])

    detector = SegmentDetector(
        time_arr=times,
        values_arr=values,
        emwa_tau_minutes=5,
        trigger_thresh=-5.0,
        release_thresh=-1.0
    )

    detector.debug_plot()

    diseq_intervals = detector.get_disequilibrium_intervals()
    eq_intervals = detector.get_equilibrium_intervals()

    assert len(diseq_intervals) > 0
    assert len(eq_intervals) >= 2

    watering_events = list(detector.get_watering_events(nmrse_max=0.5))
    
    assert len(watering_events) > 0

    event = watering_events[0]
    event_start_time = pd.Timestamp(event.start)
    assert event_start_time.hour == 10
    assert event_start_time.minute >= 35

    v_trigger, _ = detector.get_vel_thresholds()
    assert v_trigger[0] < v_trigger[-1]