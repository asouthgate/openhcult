
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 


class DynamicIntervalInfo:
    def __init__(self, t_values, x_values, at_equilibrium):
        assert len(t_values) > 0
        self.start = t_values[0]
        self.end = t_values[-1]
        self.at_equilibrium = at_equilibrium
        self.max = max(x_values)
        self.min = min(x_values)
        self.var = np.var(x_values)
        self.mean = np.mean(x_values)
        self.duration = self.end - self.start
        t_numeric = (t_values - self.start).astype('float64')
        # self.m, self.c = np.polyfit(t_numeric, x_values, 1)


class DisequilibriumIntervalDetector:
    """Takes sensor data and identifies regions in (dis)equilibrium."""
    def __init__(
            self,
            time_arr,
            values_arr,
            emwa_tau_minutes=30,
            trigger_thresh=-0.25,
            release_thresh = -0.15,
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
        self._resampled_vel_smoothed = self._resampled_vel.ewm(span=emwa_tau_minutes).mean()
        self._resampled_acc_smoothed = self._resampled_vel_smoothed.diff().fillna(0).ewm(span=emwa_tau_minutes).mean()
        self._resampled_times = self._resampled_emwa.index.to_numpy()
        self._trigger_arr, self._release_arr = lerp_thresholds(
            self._resampled_emwa,
            self._trigger_thresh_acc,
            self._release_thresh_acc,
            self._trigger_thresh_acc / 4.0,
            self._release_thresh_acc / 4.0
        )

    def _ind_pairs_to_dynamic_interval(self, ind_pairs):
        return [DynamicIntervalInfo(
            self._resampled_times[start:end],
            self._resampled_emwa[start:end],
            False
        ) for start, end in ind_pairs]


    def get_disequilibrium_intervals(self):
        diseq_inds = find_regions_with_hysteresis(
            self._resampled_times, self._resampled_vel_smoothed, self._trigger_thresh, self._release_thresh)
        return self._ind_pairs_to_dynamic_interval(diseq_inds)
    
    def get_equilibrium_intervals(self):
        diseq_inds = find_regions_with_hysteresis(
            self._resampled_times, self._resampled_vel_smoothed, self._trigger_thresh, self._release_thresh)
        eq_regions = get_complementary_intervals(diseq_inds, 0, len(self._time_arr))
        return self._ind_pairs_to_dynamic_interval(eq_regions)
    
    def get_acceleration_intervals(self):
        return self._ind_pairs_to_dynamic_interval(
            find_regions_with_hysteresis_adapative_thresh(
            self._resampled_times,
            self._resampled_acc_smoothed,
            self._trigger_arr,
            self._release_arr
            )
        )
    
    def get_thresholds(self):
        return self._trigger_arr, self._release_arr

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
            self._resampled_acc_smoothed.values/max(self._resampled_acc_smoothed),
            color='purple',
            label='Smoothed acceleration (EWMA)',
            linewidth=1
        )

        deq_regions = self.get_disequilibrium_intervals()
        for deqr in deq_regions:
            ax2.axvspan(deqr.start, deqr.end, color='green', alpha=0.15)

        eq_regions = self.get_equilibrium_intervals()
        for deqr in eq_regions:
            ax2.axvspan(deqr.start, deqr.end, color='grey', alpha=0.15)

        acc_regions = self.get_acceleration_intervals()
        for deqr in acc_regions:
            ax2.axvspan(deqr.start, deqr.end, color='orange', alpha=0.15)


        trigger, release = self.get_thresholds()
        ax2.axhline(0, color='tab:blue', linestyle='--', alpha=0.3) # Zero baseline
        ax2.plot(self._resampled_times, trigger, color='red', linestyle='--')
        ax2.plot(self._resampled_times, release, color='red', linestyle='--')

        ax2.set_ylabel('Normalized values (derivatives)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()

        # 2. Combine them and call legend on just one of the axes
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        plt.title('Moisture Levels vs. Smoothed Velocity')
        plt.savefig("segmentation.png")
        fig.tight_layout()
        plt.show()


def find_regions_with_hysteresis(times, vel, trigger=-0.015, release=-0.005):
    regions = []
    active = False
    start_ind = None
    
    for j, tv in enumerate(zip(times, vel)):
        t, v = tv
        if active and v > release:
            active = False
            regions.append((start_ind, j))
        if not active and v < trigger:
            active = True
            start_ind = j
            
    # Handle event still active at end of data
    if active:
        regions.append((start_ind, len(times) - 1))
        
    return regions


def lerp_thresholds(val, trigger_high, release_high, trigger_low, release_low):

    # Linear interp thresholds
    maxx = max(val)
    minxx = min(val)
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


def find_regions_with_hysteresis_adapative_thresh(times, val, trigger_arr, release_arr):
    regions = []
    active = False
    start_ind = None

    for ti, tup in enumerate(zip(times, val)):
        t, v = tup
        if active and v > release_arr[ti]:
            active = False
            regions.append((start_ind, ti))
        if not active and v < trigger_arr[ti]:
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
        alpha = 1 - np.exp(-delta_t / tau_minutes)
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
            complementary.append([current_pos, start - 1])
        
        # Move the cursor to just after the current interval
        current_pos = max(current_pos, end + 1)

    # 2. Check if there is a remaining gap after the last interval
    if current_pos <= end_bound:
        complementary.append([current_pos, end_bound])

    return complementary