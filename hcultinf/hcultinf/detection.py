
from __future__ import annotations

import logging
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Lowest level to capture everything

class DynamicIntervalInfo:
    def __init__(self, t_values, x_values, vel, at_equilibrium):
        assert len(t_values) > 0
        assert len(t_values) == len(x_values)
        self.n_samples = len(t_values)
        self.start = t_values[0]
        self.end = t_values[-1]
        self.at_equilibrium = at_equilibrium
        self.max = max(x_values)
        self.min = min(x_values)
        self.var = np.var(x_values)
        self.mean = np.mean(x_values)
        self.duration = self.end - self.start
        t_numeric = (t_values - self.start).astype('timedelta64[ms]').astype('int64')
        self.m, self.c = None, None
        try:
            self.m, self.c = np.polyfit(t_numeric, x_values, 1)
        except np.linalg.LinAlgError as e:
            print(e)
        self.pred_func, self.pred_params_d, self.gof_d = None, None, None
        if not at_equilibrium:
            try:
                self.pred_func, self.pred_params_d = self._estimate_negative_lognormal(t_values, vel)
                self.gof_d = self._cal_goodness_of_fit(t_values, vel)
            except RuntimeError as e:
                print(e)
            except ValueError as e:
                print(e)
    
    def _estimate_negative_lognormal(self, t, x):
        t_num = (t - t[0]).astype('timedelta64[ms]').astype(float) / 1000.0
    
        def model(t_val, h, x_p, w, shift):
            t_shifted = t_val - shift
            xp_shifted = x_p 
            res = np.zeros_like(t_val)
            mask = t_shifted > 0
            exponent = -(np.log(t_shifted[mask] / xp_shifted)**2) / (2 * w**2)
            res[mask] = h * np.exp(np.clip(exponent, -700, 0))
            return res

        min_val = np.min(x)
        if min_val >= 0:
            return lambda t_inp: np.zeros_like(t_inp).astype(float), {}

        max_t = t_num[-1]
        peak_t = t_num[np.argmin(x)]
        duration = t_num[-1] - t_num[0]

        lower_bounds = [min_val * 2.0, 0, 0.2, 0.0]
        upper_bounds = [0, max_t, 2.0, duration]

        try:
            popt, _ = curve_fit(
                model, t_num, x, 
                p0=[min_val, peak_t, 0.5, 0.0],
                bounds=(lower_bounds, upper_bounds)
            )
        except:
            return lambda t_inp: np.zeros_like(t_inp).astype(float), {}

        def fit_func(t_input):
            t_in = (t_input - t[0]).astype('timedelta64[ms]').astype(float) / 1000.0
            return model(t_in, *popt)
        
        h_fit, xp_fit, w_fit, shift_fit = popt
        return fit_func, {"h": h_fit, "xp:": xp_fit, "w": w_fit, "shift": shift_fit}
    
    def _cal_goodness_of_fit(self, t, x):
        """
        Calculates error metrics for a candidate fit.
        """
        if self.pred_func is None:
            return {"rmse": np.inf, "nrmse": np.inf}
        
        # t_num = (t - t[0]).astype('timedelta64[ms]').astype(float) / 1000.0

        y_pred = self.pred_func(t)
        residuals = x - y_pred

        
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Range-normalized RMSE
        data_range = np.max(x) - np.min(x)
        nrmse = rmse / data_range if data_range != 0 else np.inf
        
        return {
            "rmse": rmse,
            "nrmse": nrmse,
            "max_residual": np.max(np.abs(residuals))
        }
    
    def get_model_active_interval(self, threshold_pct=0.01):
        if self.pred_func is None:
            return self.start, self.end

        seg_dur_sec = self.duration / np.timedelta64(1, 's')
        search_start_num = -seg_dur_sec * 5 
        search_end_num = seg_dur_sec * 5
        
        num_points = 5000
        t_grid_num = np.linspace(search_start_num, search_end_num, num_points)
        t_grid_dt = self.start + (t_grid_num * 1e3).astype('timedelta64[ms]')
        
        y_values = self.pred_func(t_grid_dt)
        dy = np.abs(np.gradient(y_values, t_grid_num))
        
        max_slope = np.max(dy)
        if max_slope == 0:
            return self.start, self.end
            
        active_indices = np.where(dy > (max_slope * threshold_pct))[0]
        
        if len(active_indices) == 0:
            return self.start, self.end

        t_start_active = t_grid_num[active_indices[0]]
        t_end_active = t_grid_num[active_indices[-1]]
        
        t0_estimate = self.start + np.timedelta64(int(t_start_active * 1000), 'ms')
        tend_estimate = self.start + np.timedelta64(int(t_end_active * 1000), 'ms')
        
        return t0_estimate, tend_estimate
    
class SegmentDetector:
    """Takes sensor data and identifies regions in (dis)equilibrium."""
    def __init__(
            self,
            time_arr,
            values_arr,
            emwa_tau_minutes=30,
            trigger_thresh=-0.75,
            release_thresh = -0.70,
            trigger_thresh_acc = -0.015,
            release_thresh_acc = -0.0075,
        ):
        self._time_arr = time_arr
        self._values_arr = values_arr
        self._emwa_tau_minutes = emwa_tau_minutes
        self._trigger_thresh = trigger_thresh
        self._release_thresh = release_thresh

        self._trigger_thresh_acc = trigger_thresh_acc
        self._release_thresh_acc = release_thresh_acc

        self._time_diff_minutes = (time_arr[1:] - time_arr[:-1]) / np.timedelta64(1, 'm')
        self._values_diff = (values_arr[1:] - values_arr[:-1])

        self._values_emwa = compute_time_weighted_ewma(time_arr, values_arr, emwa_tau_minutes)
        tmp_emwa_series = pd.Series(self._values_emwa, index=pd.to_datetime(time_arr))
        self._resampled_emwa = tmp_emwa_series.resample('1min').mean().interpolate(method='linear')
        self._resampled_vel = self._resampled_emwa.diff().fillna(0)
        
        self._resampled_times = self._resampled_emwa.index.to_numpy()
        self._resampled_vel_smoothed = compute_time_weighted_ewma(self._resampled_times, self._resampled_vel.values, emwa_tau_minutes)
        self._resampled_acc = np.diff(self._resampled_vel_smoothed, prepend=0)
        self._resampled_acc_smoothed = compute_time_weighted_ewma(self._resampled_times, self._resampled_acc, emwa_tau_minutes / 4.0)


        self._acc_trigger_arr, self._acc_release_arr = lerp_thresholds(
            self._resampled_emwa,
            3000,
            1000,
            self._trigger_thresh_acc,
            self._release_thresh_acc,
            self._trigger_thresh_acc / 4.0,
            self._release_thresh_acc / 4.0
        )

        self._vel_trigger_arr, self._vel_release_arr = lerp_thresholds(
            self._resampled_emwa,
            3000,
            1000,
            self._trigger_thresh,
            self._release_thresh,
            self._trigger_thresh / 4.0,
            self._release_thresh / 4.0
        )
        
        # Indices for regions with velocity ON
        self._vel_inds = find_regions_with_hysteresis_adapative_thresh(
            self._resampled_times,
            self._resampled_vel_smoothed,
            self._vel_trigger_arr,
            self._vel_release_arr
        )

        # Indices for regions with acceleration ON
        self._neg_acc_inds = find_regions_with_hysteresis_adapative_thresh(
            self._resampled_times,
            self._resampled_acc_smoothed,
            self._acc_trigger_arr,
            self._acc_release_arr
        )

        self._pos_acc_inds = find_regions_with_hysteresis_adapative_thresh(
            self._resampled_times,
            self._resampled_acc_smoothed,
            - self._acc_trigger_arr,
            - self._acc_release_arr,
            1
        )

        # Indices for regions with velocity ON ^ acceleration ON
        self._neg_diseq_inds = merge_intervals([self._vel_inds, self._neg_acc_inds])
        self._diseq_inds = self._vel_inds
        self._acc_inds = merge_intervals([self._pos_acc_inds, self._neg_acc_inds])

        # Indices for regions with both OFF
        self._eq_inds = get_complementary_intervals(
            self._diseq_inds,
            0,
            len(self._resampled_times)
        )

        self._diseq_regions = self._ind_pairs_to_dynamic_interval(self._diseq_inds, False)
        self._eq_regions = self._ind_pairs_to_dynamic_interval(self._eq_inds, True)

    def _ind_pairs_to_dynamic_interval(self, ind_pairs, at_equilibrium):
        dis = []
        for start, end in ind_pairs:
            dis.append(
                DynamicIntervalInfo(
                    self._resampled_times[start:end],
                    self._resampled_emwa[start:end],
                    self._resampled_vel_smoothed[start:end],
                    at_equilibrium
                )
            )
        return dis

    def get_disequilibrium_intervals(self):
        return self._diseq_regions
    
    def get_equilibrium_intervals(self):
        return self._eq_regions
    
    def get_neg_acceleration_intervals(self):
        return self._ind_pairs_to_dynamic_interval(self._neg_acc_inds, False)
    
    def get_pos_acceleration_intervals(self):
        return self._ind_pairs_to_dynamic_interval(self._pos_acc_inds, False)

    def get_acc_thresholds(self):
        return self._acc_trigger_arr, self._acc_release_arr
    
    def get_vel_thresholds(self):
        return self._vel_trigger_arr, self._vel_release_arr
    
    def get_watering_events(self, nmrse_max=0.5) -> DynamicIntervalInfo:
        for deqr in self.get_disequilibrium_intervals():
            if deqr.gof_d:
                nmrse = deqr.gof_d['nrmse']
                if nmrse < nmrse_max:
                    yield deqr

    def debug_plot(self):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.scatter(self._time_arr, self._values_arr, color='tab:blue', label='Sensor values', alpha=0.5, marker='x')
        ax1.plot(self._resampled_times, self._resampled_emwa, color='tab:blue', label='Smoothed values (EWMA)', alpha=1.0, linewidth=1)
        ax1.set_ylabel('Sensor reading', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(
            self._resampled_times,
            self._resampled_vel_smoothed/max(self._resampled_vel_smoothed),
            color='#ad444f', label='Smoothed velocity (EWMA)', linewidth=1
        )
        ax2.plot(
            self._resampled_times-self._emwa_tau_minutes,
            self._resampled_acc_smoothed/max(self._resampled_acc_smoothed),
            color='purple',
            label='Smoothed acceleration (EWMA)',
            linewidth=1
        )

        deq_regions = self.get_disequilibrium_intervals()
        for deqr in deq_regions:
            ax2.axvspan(deqr.start, deqr.end, color='green', alpha=0.15)

        # eq_segments = [eqs for eqtriple in eq_triples for eqs in eqtriple]
        nmrse_max = 0

        for deqr in deq_regions:
            if deqr.gof_d and not np.isinf(deqr.gof_d['nrmse']):
                nmrse = deqr.gof_d['nrmse']
                nmrse_max = max(nmrse, nmrse_max)
        for deqr in deq_regions:
            if deqr.at_equilibrium:
                color = 'grey'
            else:
                color = 'orange'
            ax2.axvspan(deqr.start, deqr.end, color=color, alpha=0.05)

            if deqr.m is not None and deqr.at_equilibrium:
                x1 = deqr.c 
                duration_ms = (deqr.end - deqr.start).astype('timedelta64[ms]').astype('int64')
                x2 = (duration_ms * deqr.m) + deqr.c
                ax1.plot([deqr.start, deqr.end], [x1, x2], color='red', linewidth=2)
            if deqr.pred_func is not None:
                vpred = deqr.pred_func(self._resampled_times)
                ax2.plot(self._resampled_times, vpred/max(self._resampled_vel_smoothed), color='black', linestyle='dotted')
                model_start, model_end = deqr.get_model_active_interval()
                ax2.axvspan(model_start, model_end, color="blue", alpha=0.15)
                print(deqr.gof_d['nrmse'])
                ax2.axvline(x = model_start, ymax = deqr.gof_d['nrmse'] / nmrse_max, color='black')
        print()
        trigger, release = self.get_acc_thresholds()
        ax2.axhline(0, color='tab:blue', linestyle='--', alpha=0.3) # Zero baseline
        ax2.plot(self._resampled_times, trigger / max(self._resampled_acc_smoothed), color='purple', linestyle='--')
        ax2.plot(self._resampled_times, release / max(self._resampled_acc_smoothed), color='purple', linestyle='--')

        trigger, release = self.get_vel_thresholds()
        ax2.plot(self._resampled_times, trigger / max(self._resampled_vel_smoothed), color='red', linestyle='--')
        ax2.plot(self._resampled_times, release / max(self._resampled_vel_smoothed), color='red', linestyle='--')

        ax2.set_ylabel('Normalized values (derivatives)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()

        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        plt.title('Moisture Levels vs. Smoothed Velocity')
        plt.savefig("segmentation.png")
        fig.tight_layout()
        plt.show()


def merge_intervals(interval_lists):

    assert len(interval_lists) > 1, "Expected to merge more than one list"

    intervals = []
    for il in interval_lists:
        intervals += il

    if not intervals:
        return []

    # Sort intervals by the start value
    intervals.sort(key=lambda x: x[0])

    merged = [list(intervals[0])]

    for current_start, current_end in intervals[1:]:
        _, last_end = merged[-1]

        # Check if they overlap or touch at the boundary
        if current_start <= last_end:
            # Update the end of the last interval in the list
            merged[-1][1] = max(last_end, current_end)
        else:
            # No overlap, add the current interval as a new entry
            merged.append([current_start, current_end])

    # Convert back to tuples if preferred
    return [tuple(i) for i in merged]


def lerp_thresholds(val, max_val, min_val, trigger_high, release_high, trigger_low, release_low):

    # Linear interp thresholds
    maxx = max_val
    minxx = min_val
    rangex = maxx - minxx

    range_trigger = trigger_high - trigger_low
    range_release = release_high - release_low

    m_trigger = range_trigger / rangex
    m_release = range_release / rangex

    c_trigger = trigger_low
    c_release = release_low

    trigger_arr = np.zeros(len(val))
    release_arr = np.zeros(len(val))

    for i, v in enumerate(val):
        trigger = m_trigger * (v-minxx) + c_trigger
        release = m_release * (v-minxx) + c_release
        trigger_arr[i] = trigger
        release_arr[i] = release
            
    return trigger_arr, release_arr


def find_regions_with_hysteresis_adapative_thresh(times, val, trigger_arr, release_arr, direction=-1):

    if direction == -1:
        assert all(trigger_arr < 0)
        assert all(release_arr < 0)

    regions = []
    active = False
    start_ind = None

    for ti, tup in enumerate(zip(times, val)):
        t, v = tup
        if active and direction * v < direction * release_arr[ti]:
            active = False
            regions.append((start_ind, ti))
        if not active and direction * v > direction * trigger_arr[ti]:
            active = True
            start_ind = ti
            
    # Handle event still active at end of data
    if active:
        regions.append((start_ind, len(times) - 1 ))
        
    return regions


def compute_time_weighted_ewma(times, values, tau_minutes=30.0):
    """
    tau_minutes: The 'memory' of the filter. 
    Larger tau = smoother, but stays 'stuck' longer after gaps.
    """
    # Convert times to float minutes
    t_min = times.astype('datetime64[m]').astype(float)
    n = len(values)
    smoothed = np.zeros(n)
    smoothed[0] = values[0] # Initialize
    for i in range(1, n):
        delta_t = t_min[i] - t_min[i-1]
        # Calculate dynamic alpha based on time gap
        alpha = 1 - np.exp(- delta_t / tau_minutes)
        smoothed[i] = (1 - alpha) * smoothed[i-1] + alpha * values[i]
    return smoothed


def get_complementary_intervals(intervals, start_bound, end_bound):
    """
    Returns the gaps between disjoint intervals within a specific range.
    """
    # 1. Ensure intervals are sorted by their start index
    sorted_intervals = sorted(intervals)
    complementary = []
    current_pos = start_bound

    for start, end in sorted_intervals:
        # If there is space between the current position and the next interval
        if start > current_pos:
            complementary.append([current_pos, start])
        
        # Move the cursor to just after the current interval
        assert current_pos != end + 1
        current_pos = max(current_pos, end + 1)

    # 2. Check if there is a remaining gap after the last interval
    if current_pos < end_bound:
        assert current_pos != end_bound
        complementary.append([current_pos, end_bound])

    return complementary