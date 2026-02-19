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
    agg_values = []
    agg_deltas = []
    agg_sensors = []
    agg_reseating = []
    agg_pot_number = []
    agg_water_volumes = []

    # Loop over rows of the csv, getting data
    with open(args.intervals_csv, "r") as f:
        next(f)
        for line in f:
            line = line.strip()
            start_utc, end_utc, reseating, pot_number, pot_water_volume = line.split(",")
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
            for name, points in series.items():

                agg_times += [t for t, _ in points]
                agg_values += [v for _, v in points]
                agg_sensors += [name] * len(points)
                agg_reseating += [reseating] * len(points)
                agg_pot_number += [pot_number] * len(points)
                agg_deltas += [0] + list(np.diff([v for _, v in points]))
                agg_water_volumes += [float(pot_water_volume)] * len(points)

    df = pd.DataFrame({
        "time": agg_times,
        "time_index": range(len(agg_times)),
        "value": agg_values,
        "sensor": agg_sensors,
        "reseating": agg_reseating,
        "pot_number": agg_pot_number,
        "deltas": agg_deltas,
        "pot_water_volume": agg_water_volumes,
    })

    water_vol_means = {
        pot_water_volume: subdf["value"].mean() for pot_water_volume, subdf in df.groupby("pot_water_volume")
    }

    # add normalized X values by subtracting the mean for each pot water volume
    df["value_normalized"] = df.apply(lambda row: row["value"] - water_vol_means[row["pot_water_volume"]], axis=1)


    # Create a date range regular grid with 1 minute time period
    date_values = pd.date_range(start=df["time"].min(), end=df["time"].max(), freq="1min") 
    average_values = []
    # create a locally smoothed average at each time point 
    for date in date_values:
        # not just values falling in the same minute, but smoothly weighted by a Gaussian kernel with 5 minute bandwidth
        weights = norm.pdf((df["time"] - date).dt.total_seconds() / 60, scale=5)
        average = np.sum(weights * df["value"]) / np.sum(weights)
        average_values.append(average)


    colors = ["#A9E5BB", "#F7B32B", "#8D2D3B", "#2D1E2F", "#FEFAD8"]
    sensor_colors = {name: colors[i] for i, name in enumerate(sensor_names)}
    sensor_indexes = {name: i for i, name in enumerate(sensor_names)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    ax = axes.flatten()

    tend =0
    for water_volume, subsubdf in df.groupby("pot_water_volume"):
        # ax[0].axvline(subsubdf["time"].min(), color="gray", linestyle="--", linewidth=1)
        tend_new = 0
        ax[0].axvline(tend, color="gray", linestyle="--", linewidth=1)
        for sensor, subdf in subsubdf.groupby("sensor"):
            t0 = min(time for time in subdf["time"])
            time_arr = tend + (subdf["time"] - t0).dt.total_seconds()
            ax[0].plot(time_arr, subdf["value"], label=sensor_indexes[name], color=sensor_colors.get(sensor, "black"))
            ax[0].scatter(time_arr, subdf["value"], color=sensor_colors.get(sensor, "black"), s=10)
            tend_new = max(tend_new, max(time_arr))
            # plot a vertical line
        tend = tend_new
        ax[0].axvline(tend, color="gray", linestyle="--", linewidth=1)


    # ax[0].plot(date_values, average_values, label="Average", color="grey", linewidth=2)

    handles = [matplotlib.lines.Line2D([0], [0], color=color, label=sensor_indexes[sensor]) for sensor, color in sensor_colors.items()]
    ax[0].legend(handles=handles, title="Sensor")
    ax[0].set_ylim(0, 3000)
    # rotate the x axis labels a small angle so the dates are visible
    ax[0].tick_params(axis='x', rotation=45)
    # ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax[0].set_xlabel("Time index")
    ax[0].set_ylabel("X")

    for sensor, subdf in df.groupby("sensor"):
        print(sensor_colors)
        # boxplots instead of hist
        ax[1].boxplot(subdf["deltas"], positions=[sensor_indexes[sensor]], 
            widths=0.6, patch_artist=True, boxprops=dict(facecolor=sensor_colors[sensor], 
            color="grey"), medianprops=dict(color="black"))
        ax[1].set_ylabel("$X_t - X_{t-1}$")


    for sensor, subdf in df.groupby("sensor"):
        # ax[2].hist(subdf["value"], bins=10, color=sensor_colors[sensor], histtype='step')
        # instead a hist, show boxplots
        ax[2].boxplot(subdf["value_normalized"], positions=[sensor_indexes[sensor]], 
            widths=0.6, patch_artist=True, boxprops=dict(facecolor=sensor_colors[sensor], 
            color="grey"), medianprops=dict(color="black"))
        ax[2].set_ylabel("$X - \overline{X}(Z=z)$")

    # Add box plots for aggregated value data across sensors
    water_vol_boxplots = []
    water_vol_positions = []
    for pot_water_volume, subdf in df.groupby("pot_water_volume"):
        # plot a a box plot for the values in the subdf
        water_vol_boxplots.append(subdf["value"])
        water_vol_positions.append(float(pot_water_volume))
    ax[3].boxplot(water_vol_boxplots, positions=water_vol_positions, 
        widths=5.0,patch_artist=True, boxprops=dict(facecolor=colors[-1], 
        color="grey"), medianprops=dict(color="black"))

    # ax[3].hist(df["value"], bins=10, color="grey", edgecolor="black", histtype='bar', alpha=0.8, rwidth=0.7)
    # ax[3].hist(df["value"], bins=10, color="#4287f5")
    ax[3].set_xlabel("Z")
    ax[3].set_ylabel("X")

    plt.savefig("equilibrium_noise_distribution.png", dpi=300)
    plt.show()

    # We are going to characterise 4 plots:
    # First, the within-sensor delta distribution, aggregate, expected to be low
    # Secondly, the value distribution for a given Z/pot aggregated across reseatings, colored by sensor
    # Thirdly, boxplots showing the variation as a function of Z
    # Lastly, a plot showing one raw time series of all sensors, for illustrative purposes, across reseatings, with time point vertical lines
