import pytest
import numpy as np
import pandas as pd
from hcultinf.detection import DynamicIntervalInfo
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
    
    # Convert back to relative seconds for easy comparison
    start_offset = (t0_active - t_start) / np.timedelta64(1, 's')
    end_offset = (tend_active - t_start) / np.timedelta64(1, 's')
    
    assert 1.5 <= start_offset <= 2.5
    assert end_offset > 4.0
    assert end_offset > start_offset

def test_initialization_and_stats():
    # Setup: 5 samples, 1 second apart
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
    # Duration should be 4 seconds (4000ms)
    assert interval.duration == np.timedelta64(4, 's')
    # Since it's a perfect line, m should be roughly 10/1000 (unit per ms)
    assert pytest.approx(interval.m, rel=1e-3) == 0.01

def test_at_equilibrium_skips_fitting():
    t_values = pd.to_datetime(['2026-03-17 10:00:00', '2026-03-17 10:00:01']).values
    x_values = np.array([1.0, 1.1])
    vel = np.array([0.1, 0.1])
    
    # When at_equilibrium is True
    interval = DynamicIntervalInfo(t_values, x_values, vel, at_equilibrium=True)
    
    assert interval.pred_func is None
    assert interval.pred_params_d is None
    assert interval.gof_d is None

def test_initialization_assertion_error():
    t_values = pd.to_datetime(['2026-03-17 10:00:00', '2026-03-17 10:00:01']).values
    x_values = np.array([1.0]) # Only 1 value for 2 timestamps
    vel = np.array([0.0, 0.0])
    
    with pytest.raises(AssertionError):
        DynamicIntervalInfo(t_values, x_values, vel, at_equilibrium=True)


