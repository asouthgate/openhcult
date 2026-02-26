#!/usr/bin/env python3
"""Plot sensor time series from the configured database."""

from __future__ import annotations

import argparse
import configparser
import json
import math
from brokenaxes import brokenaxes
import pickle
import pandas as pd
import os
import sqlite3
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Dict, List, Tuple
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mplcolors
import numpy as np

from hcultinf.inference import compute_ewma, compute_zscore, classify_events
from hcultutils.fetch_data import fetch_data, fetch_series_from_ctrl

if __name__ == "__main__":

    # Get argparse arguments: csv of time intervals. start_utc, end_utc, reseating, pot number, pot water volume
    ap = argparse.ArgumentParser(description="Plot sensor time series from the configured database.")
    ap.add_argument(
        "intervals_csv",
        type=str,
        help="Path to CSV file containing time intervals and metadata.",
    )
    ap.add_argument(
        "--ctrl-url",
        type=str,
        default=None,
        help="URL of the ctrl server to fetch data from. Overrides config file.",
    )
    args = ap.parse_args()

    agg_times = []
    agg_equil_time_deltas = []
    agg_values = []
    agg_sensors = []
    agg_trial = []
    agg_water_volumes = []
    agg_total_change = []

    # Loop over rows of the csv, getting data
    with open(args.intervals_csv, "r") as f:
        next(f)
        for line in f:
            line = line.strip()
            water_utc, equil_utc, trial, pot_water_volume = line.split(",")
            if water_utc == "NA": continue
            end_utc = equil_utc
            # start_utc = end_utc - 3 minutes
            end_utc_dt = pd.to_datetime(end_utc)
            water_utc_dt = pd.to_datetime(water_utc)
            # start_utc_dt = end_utc_dt - pd.Timedelta(minutes=10)
            # start_utc = start_utc_dt.strftime("%Y-%m-%dT%H:%M:%S")
            start_utc = water_utc
            print(f"Fetching data for interval {start_utc} to {end_utc}, trial {trial}, pot water volume {pot_water_volume}")
            series = fetch_series_from_ctrl(
                args.ctrl_url,
                start_utc=start_utc,
                end_utc=end_utc,
                device=None,
                sensor=None,
                plant_name=None,
                limit=100000
            )

            # dump the sensor data with a date name
            with open(f"sensor_data_{start_utc}_{end_utc}.pkl", "wb") as f:
                pickle.dump(series, f)

            sensor_names = sorted(series.keys())
            c2i = {name: i for i, name in enumerate(sensor_names)}
            c = 0
            for name, points in series.items():
                last_time = points[-1][0]
                last_val = points[-1][1]
                agg_times.append(last_time)
                agg_values.append(last_val)
                agg_sensors.append(name)
                agg_trial.append(trial)
                agg_water_volumes.append(float(pot_water_volume))
                agg_equil_time_deltas.append((last_time - water_utc_dt).total_seconds())
                agg_total_change.append(last_val - points[0][1])
                # agg_times += [t for t, _ in points]
                # agg_values += [v for _, v in points]
                # agg_sensors += [c] * len(points)
                # c += 1
                # agg_trial += [trial] * len(points)
                # agg_water_volumes += [float(pot_water_volume)] * len(points)

    SOIL_MASS = 40
    df = pd.DataFrame({
        "time": agg_times,
        "time_index": range(len(agg_times)),
        "value": agg_values,
        "sensor": agg_sensors,
        "trial": agg_trial,
        "pot_water_volume": agg_water_volumes,
        "SWC": np.array(agg_water_volumes) / SOIL_MASS,
        "equil_time_hours": np.array(agg_equil_time_deltas) / 3600.0,
        "total_change": agg_total_change,
    })

    colors = ["#A9E5BB", "#F7B32B", "#8D2D3B", "#2D1E2F", "#FEFAD8"]
    sensor_colors = {name: colors[i] for i, name in enumerate(sensor_names)}
    sensor_indexes = {name: i for i, name in enumerate(sensor_names)}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    ax = axes.flatten()

    for sensor, subdf in df.groupby("sensor"):
        ax[0].scatter(subdf["SWC"], subdf["value"], c=sensor_colors[sensor], label=f"sensor {sensor_indexes[sensor]}")
    # plot the average line for sensors
    means = []
    vol_at_mean = []
    upper_95_percentile = []
    lower_95_percentile = []
    for pot_water_volume, subdf in df.groupby("SWC"):
        means.append(np.mean(subdf["value"]))
        upper_95_percentile.append(np.percentile(subdf["value"], 97.5))
        lower_95_percentile.append(np.percentile(subdf["value"], 2.5))
        vol_at_mean.append(pot_water_volume)
    ax[0].plot(vol_at_mean, means, color="grey", label="Mean", linewidth=2)
    ax[0].fill_between(vol_at_mean, lower_95_percentile, upper_95_percentile, color="grey", alpha=0.3, label="95% Percentile")
    ax[0].set_xlabel("SWC")
    ax[0].set_ylabel("Sensor Reading")
    ax[0].legend()


    for sensor, subdf in df.groupby("sensor"):
        for _, subsubdf in subdf.groupby("trial"):
            ax[1].plot(subsubdf["SWC"], subsubdf["value"], c=sensor_colors[sensor], label=f"sensor {sensor_indexes[sensor]}")
    ax[1].set_xlabel("SWC")
    ax[1].set_ylabel("Sensor Reading")

    # # Now for ax[1] plot mean curves for each trial, with one line per trial, volume vs reading
    # for trial, subdf in df.groupby("trial"):
    #     means = []
    #     vol_at_mean = []
    #     for pot_water_volume, subsubdf in subdf.groupby("pot_water_volume"):
    #         means.append(np.mean(subsubdf["value"]))
    #         vol_at_mean.append(pot_water_volume)
    #     ax[2].plot(vol_at_mean, means, label=f"Trial {trial} mean across sensors")
    #     ax[2].scatter(vol_at_mean, means)
    # ax[2].legend()

    # Now plot equilibriation time as a function of water content
    ax[2].scatter(df["SWC"], df["equil_time_hours"] , c=df["sensor"].map(sensor_colors), label="Equilibration Time Delta")
    ax[2].set_xlabel("SWC")
    ax[2].set_ylabel("Equilibration Time Delta (hours)")
    # plt.show()

    # Now, for each trial, plot delta value/delta volume vs volume
    # FOR EACH SENSOR
    for trial, subdf in df.groupby("trial"):
        for sensor, subsubdf in subdf.groupby("sensor"):    
            sorted_subdf = subsubdf.sort_values("SWC")
            delta_value = sorted_subdf["value"].diff()
            delta_volume = sorted_subdf["SWC"].diff()
            # print(delta_value)
            ax[3].scatter(sorted_subdf["SWC"], delta_value / delta_volume, c=sensor_colors[sensor], label=f"Sensor {sensor}")
    ax[3].set_xlabel("SWC")
    ax[3].set_ylabel("$\Delta X / \Delta Z$")
    # ax[3].legend()

    for trial, subdf in df.groupby("trial"):
        # Now calculate mean delta value/delta volume across sensors for this trial
        # You do this by groupby then .mean()
        dxdv_arrays = []
        for sensor, subsubdf in subdf.groupby("sensor"):
            sorted_subdf = subsubdf.sort_values("SWC")
            delta_value = sorted_subdf["value"].diff()
            delta_volume = sorted_subdf["SWC"].diff()
            dxdv = delta_value / delta_volume
            dxdv_arrays.append(dxdv.values)
        mean_dxdv = np.nanmean(dxdv_arrays, axis=0)
        ax[4].scatter(sorted_subdf["SWC"], mean_dxdv, label=f"Trial {trial}")
        # ax[4].scatter(sorted_subdf["SWC"], delta_value / delta_volume, label=f"Sensor {sensor}")

    ax[4].set_xlabel("SWC")
    ax[4].set_ylabel("mean $\Delta X / \Delta Z$")
    ax[4].legend()

    # For the final plot, we are going to plot delta value/delta volume vs equilibriation time, colored by water volume
    # This should be aggregated in one plot
    # ax[5].scatter(df["equil_time_hours"], df["total_change"] / df["SWC"], c=df["SWC"],
    #     cmap="viridis",     norm=mplcolors.PowerNorm(gamma=0.4), label="Total Change / SWC")
    # # Add color bar
    # cbar = plt.colorbar(ax[5].collections[0], ax=ax[5])
    # cbar.set_label("SWC")
    # ax[5].set_xlabel("Equilibration Time Delta (hours)")
    # ax[5].set_ylabel("Total Change / SWC")

    # Final plot, between sensor variation vs SWC
    for trial, subdf in df.groupby("trial"):
        for swc, subsubdf in subdf.groupby("SWC"):
            sensor_values = subsubdf["value"].values
            if len(sensor_values) > 1:
                sensor_variation = np.std(sensor_values)
                ax[5].scatter(swc, sensor_variation, label=f"Trial {trial}")
    ax[5].set_xlabel("SWC")
    ax[5].set_ylabel("Between Sensor Variation (std dev)")
    plt.savefig("titration_plots.png")

    plt.show()