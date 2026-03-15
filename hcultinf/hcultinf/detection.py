
from __future__ import annotations

import logging
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Lowest level to capture everything

class DynamicIntervalInfo:
    def __init__(self, t_values, x_values, at_equilibrium):
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


class SegmentDetector:
    """Takes sensor data and identifies regions in (dis)equilibrium."""
    def __init__(
            self,
            time_arr,
            values_arr,
            emwa_tau_minutes=30,
            trigger_thresh=-0.45,
            release_thresh = -0.35,
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
            self._trigger_thresh_acc,
            self._release_thresh_acc,
            self._trigger_thresh_acc / 4.0,
            self._release_thresh_acc / 4.0
        )

        self._vel_trigger_arr, self._vel_release_arr = lerp_thresholds(
            self._resampled_emwa,
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
    
    def get_segment_data_triples(self):
        """Yield (eq, diseq, eq) triples."""
        all_segments = sorted(self._diseq_regions + self._eq_regions, key=lambda x: x.start)
        # If the very first element not at equilibrium, we cut it off
        # A baseline before an event is required for comparison
        if not all_segments[0].at_equilibrium:
            all_segments = all_segments[1:]
        # Same at the end
        if not all_segments[-1].at_equilibrium:
            all_segments = all_segments[:-1]
        for si in range(1, len(all_segments) - 1, 2):
            assert all_segments[si - 1].at_equilibrium
            assert not all_segments[si].at_equilibrium
            assert all_segments[si + 1].at_equilibrium
            yield (all_segments[si - 1], all_segments[si], all_segments[si + 1])

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

        eq_triples = list(self.get_segment_data_triples())
        eq_segments = [eqs for eqtriple in eq_triples for eqs in eqtriple]
        for deqr in eq_segments:
            if deqr.at_equilibrium:
                color = 'grey'
            else:
                color = 'orange'
            ax2.axvspan(deqr.start, deqr.end, color=color, alpha=0.15)

            if deqr.m is not None and deqr.at_equilibrium:
                x1 = deqr.c 
                
                # Force [ms] here too to match the slope 'm'
                duration_ms = (deqr.end - deqr.start).astype('timedelta64[ms]').astype('int64')
                
                x2 = (duration_ms * deqr.m) + deqr.c
                ax1.plot([deqr.start, deqr.end], [x1, x2], color='red', linewidth=2)   

        acc_regions = self.get_neg_acceleration_intervals()
        for deqr in acc_regions:
            ax2.axvspan(deqr.start, deqr.end, color='orange', alpha=0.15)


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

        # 2. Combine them and call legend on just one of the axes
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        plt.title('Moisture Levels vs. Smoothed Velocity')
        plt.savefig("segmentation.png")
        fig.tight_layout()
        plt.show()


def find_regions_with_hysteresis(times, vel, trigger=-0.015, release=-0.005, direction=-1):
    """ Use a two-threshold trigger and release to determine ON and OFF states

    Parameters
    ----------
        times: array of time values
        vel: array of values (e.g. velocity)
        trigger: threshold for trigger
        release: threshold for release
        direction: -1 up toward zero, 1 down toward zero; -1 corresponds to negative velocity increase
    
    """

    if direction == -1:
        assert release < 0
        assert trigger < 0

    regions = []
    active = False
    start_ind = None
    
    for j, tv in enumerate(zip(times, vel)):
        t, v = tv
        if active and direction * v < direction * release:
            active = False
            regions.append((start_ind, j))
        if not active and direction * v > direction * trigger:
            active = True
            start_ind = j
            
    # Handle event still active at end of data
    if active:
        regions.append((start_ind, len(times) - 1))
        
    return regions

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