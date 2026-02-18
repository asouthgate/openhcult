#!/usr/bin/env python3
"""Plot sensor time series from the configured database."""

from __future__ import annotations

import argparse
import configparser
import json
import math
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
            sensor_names = sorted(series.keys())
            for name, points in series.items():

                agg_times += [t for t, _ in points]
                agg_values += [v for _, v in points]
                agg_sensors += [name] * len(points)
                agg_reseating += [reseating] * len(points)
                agg_pot_number += [pot_number] * len(points)
                agg_deltas += [0] + list(np.diff([v for _, v in points]))

    df = pd.DataFrame({
        "time": agg_times,
        "value": agg_values,
        "sensor": agg_sensors,
        "reseating": agg_reseating,
        "pot_number": agg_pot_number,
        "deltas": agg_deltas,
    })

    # Create a date range regular grid with 1 minute time period
    date_values = pd.date_range(start=df["time"].min(), end=df["time"].max(), freq="1min") 
    average_values = []
    # create a locally smoothed average at each time point 
    for date in date_values:
        # not just values falling in the same minute, but smoothly weighted by a Gaussian kernel with 5 minute bandwidth
        weights = norm.pdf((df["time"] - date).dt.total_seconds() / 60, scale=5)
        average = np.sum(weights * df["value"]) / np.sum(weights)
        average_values.append(average)


    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    sensor_colors = {name: colors[i] for i, name in enumerate(sensor_names)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    ax = axes.flatten()

    for reseating, subsubdf in df.groupby("reseating"):
        ax[0].axvline(subsubdf["time"].min(), color="gray", linestyle="--", linewidth=1)
        ax[0].axvline(subsubdf["time"].max(), color="gray", linestyle="--", linewidth=1)
        for sensor, subdf in subsubdf.groupby("sensor"):
            ax[0].plot(subdf["time"], subdf["value"], label=name, color=sensor_colors.get(sensor, "black"))
            ax[0].scatter(subdf["time"], subdf["value"], color=sensor_colors.get(sensor, "black"), s=10)
            # plot a vertical line


    ax[0].plot(date_values, average_values, label="Average", color="black", linewidth=2)

    handles = [matplotlib.lines.Line2D([0], [0], color=color, label=sensor) for sensor, color in sensor_colors.items()]
    ax[0].legend(handles=handles, title="Sensor")
    ax[0].set_ylim(0, 3000)

    for sensor, subdf in df.groupby("sensor"):
        ax[1].hist(subdf["deltas"], bins=50, color=sensor_colors[sensor], histtype='step')

    for sensor, subdf in df.groupby("sensor"):
        ax[2].hist(subdf["value"], bins=50, color=sensor_colors[sensor], histtype='step')

    plt.show()

    # We are going to characterise 4 plots:
    # First, the within-sensor delta distribution, aggregate, expected to be low
    # Secondly, the value distribution for a given Z/pot aggregated across reseatings, colored by sensor
    # Thirdly, boxplots showing the variation as a function of Z
    # Lastly, a plot showing one raw time series of all sensors, for illustrative purposes, across reseatings, with time point vertical lines
