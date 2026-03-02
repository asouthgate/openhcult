#!/usr/bin/env python3
"""Plot sensor time series from the configured database."""

from __future__ import annotations

import argparse
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from hcultutils.fetch_data import fetch_data, fetch_series_from_ctrl
from hcultinf.inference import fit_monotonic_spline, \
    fit_parametric_monotonic_spline, bootstrap_parametric_spline, \
    bootstrap_monotonic_spline, compute_lookup_table_from_bootstrap

def get_spline_derivative_midpoints(df):
    x_der_midpoint = []
    x_dswcdv_midpoint = []
    swc_der_midpoint = []

    # now compute derivative of water with respect to value
    for trial, subdf in df.groupby("trial"):
        # Now calculate mean delta value/delta volume across sensors fo
        # r this trial
        # You do this by groupby then .mean()
        for sensor, subsubdf in subdf.groupby("sensor"):
            sorted_subdf = subsubdf.sort_values("SWC")
            # assert that the swc value is unique
            assert len(sorted_subdf["SWC"].unique()) == len(sorted_subdf)
            x = sorted_subdf["value"].values
            swc = sorted_subdf["SWC"].values

            # 1. Forward Difference for the very first point (x[0])
            dx_start = x[1] - x[0]
            dswc_start = swc[1] - swc[0]
            d_start = dswc_start / dx_start if dswc_start != 0 else 0
            # 3. Backward Difference for the very last point (x[-1])
            dx_end = x[-1] - x[-2]
            dswc_end = swc[-1] - swc[-2]
            d_end = dswc_end / dx_end if dx_end != 0 else 0
            # Calculate deltas (length N-1)
            # 1. Forward Difference for the very first point (x[0])
            dx = np.diff(x)
            dswc = np.diff(swc)

            # Calculate midpoints (length N-1)
            x_mid = (x[:-1] + x[1:]) / 2.0
            swc_mid = (swc[:-1] + swc[1:]) / 2.0
            # x_mid = x[:-1]
            # swc_mid = swc[:-1]
            # Calculate derivatives (length N-1)
            # You wanted dswc/dv (which is 1 / (dv/ds))
            dswc_dx_mid = dswc / dx

            x_dswcdv_midpoint.append(d_start)
            x_der_midpoint.append(x[0])
            swc_der_midpoint.append(swc[0])

            for i in range(len(x_mid)):
                if np.isfinite(dswc_dx_mid[i]):
                    x_dswcdv_midpoint.append(dswc_dx_mid[i])
                    x_der_midpoint.append(x_mid[i])
                    swc_der_midpoint.append(swc_mid[i])   

            x_dswcdv_midpoint.append(d_end)
            x_der_midpoint.append(x[-1])
            swc_der_midpoint.append(swc[-1]) 

    x_der_midpoint = np.array(x_der_midpoint).flatten()
    x_dswcdv_midpoint = np.array(x_dswcdv_midpoint).flatten()
    return x_der_midpoint, x_dswcdv_midpoint, swc_der_midpoint


def _get_data(args):
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
            end_utc_dt = pd.to_datetime(end_utc)
            water_utc_dt = pd.to_datetime(water_utc)
            start_utc = water_utc
            print(f"Fetching data for interval {start_utc} to {end_utc}, trial {trial}, pot water volume {pot_water_volume}")
            try:
                with open(f"sensor_data_{start_utc}_{end_utc}.pkl", "rb") as f:
                    series = pickle.load(f)
            except FileNotFoundError:
                print("Could not load cached data, fetching from ctrl...")
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
    return df




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
        help="URL of the ctrl server to fetch data from.",
    )
    ap.add_argument(
        "--n-bootstraps",
        type=int,
        default=100,
        help="Number of bootstrap samples to generate for uncertainty estimation.",
    )
    ap.add_argument(
        "--n-bootstraps-parametric",
        type=int,
        default=10,
        help="Number of bootstrap samples to generate for uncertainty estimation.",
    )

    args = ap.parse_args()

    df = _get_data(args)
    sensor_names = sorted(df['sensor'].unique())

    x_der_midpoint, x_dswcdv, swc_der_midpoint = get_spline_derivative_midpoints(df)

    # let's take x_anchors to be only the values at dryest and the values at wettest
    min_swc = df['SWC'].min()
    max_swc = df['SWC'].max()
    inds_min = np.where(df['SWC'] == min_swc)[0]
    inds_max = np.where(df['SWC'] == max_swc)[0]
    # print(inds_min)
    # print(inds_max)
    inds_anchor = np.concatenate([np.random.choice(inds_min, 4), inds_max[:4]])
    swc_anchor = df['SWC'].values[inds_anchor]
    value_anchor = df['value'].values[inds_anchor]
    n_inner_knots = 8
    w_der = 10.0
    k_spline = 3
    spline_x, spline_z = fit_parametric_monotonic_spline(value_anchor, swc_anchor, x_der_midpoint, x_dswcdv, n_inner_knots, k=k_spline, w_der=w_der)
    
    n_boots_parametric = args.n_bootstraps_parametric
    boot_results = bootstrap_parametric_spline(
        value_anchor, swc_anchor, x_der_midpoint, x_dswcdv, knots=n_inner_knots, k=k_spline, w_der=w_der, n_boots=n_boots_parametric)
    
    x_rang = np.linspace(df['value'].min(), df['value'].max() , 100)

    s_fine = np.linspace(0, 1, 500)

    x_plot = spline_x(s_fine)
    z_plot = spline_z(s_fine)
    knots_s = np.unique(spline_x.t)
    knots_x = spline_x(knots_s)
    knots_z = spline_z(knots_s)

    colors = ["#A9E5BB", "#F7B32B", "#8D2D3B", "#2D1E2F", "#FEFAD8"]
    sensor_colors = {name: colors[i] for i, name in enumerate(sensor_names)}
    sensor_indexes = {name: i for i, name in enumerate(sensor_names)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
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

    spline_k = 2
    n_inner_knots = 8
    inner_knots = ( np.linspace(0, 1, n_inner_knots)**2 * (df['value'].max() - df['value'].min()) + df['value'].min() ) [1:]
    inner_knots[-1] = (inner_knots[-1] + inner_knots[-2]) / 2
    print("Inner knots:", inner_knots)
    print("X range:", df['value'].min(), df['value'].max())

    spline_model = fit_monotonic_spline(df['value'].values, df['SWC'].values, inner_knots=inner_knots, k=spline_k)
    boot_splines = bootstrap_monotonic_spline(
        df['value'].values,
        df['SWC'].values,
        inner_knots=inner_knots,
        k=spline_k, 
        n_boots=args.n_bootstraps
    )
    residual_spline = fit_monotonic_spline(
        df['value'].values,
        np.abs(df['SWC'].values - spline_model(df['value'].values)),
        inner_knots=inner_knots,
        k=spline_k, 
    )

    lookup_table = compute_lookup_table_from_bootstrap(boot_splines, df['value'].min(), df['value'].max(), n_points=1000)
    # write the lookup table to csv
    lookup_df = pd.DataFrame({
        "x": lookup_table[0],
        "SWC": lookup_table[1],
        "swc_upper_95%": lookup_table[2],
        "swc_lower_95%": lookup_table[3],
    })
    lookup_df.to_csv("spline_lookup_table.csv", index=False)
    spline_lookup_table_out_file = "spline_lookup_table.csv"
    print(f"Lookup table saved to {spline_lookup_table_out_file}")
    ax[0].plot(lookup_table[1], lookup_table[0], color='red', label='Monotonic Spline Fit')
    ax[0].plot(lookup_table[1] + residual_spline(lookup_table[0]), lookup_table[0] , color='blue', label='Spline Fit Upper')
    ax[0].plot(lookup_table[1] - residual_spline(lookup_table[0]), lookup_table[0], color='blue', label='Spline Fit Lower')
    ax[0].fill_betweenx(
        lookup_table[0],
        lookup_table[2] - residual_spline(lookup_table[0]),
        lookup_table[3] + residual_spline(lookup_table[0]),
        color='blue', alpha=0.3, label='Spline Fit Residuals'
    )
    ax[0].plot(z_plot, x_plot, color='green', label='Parametric Derivative Fit')
    ax[0].set_xlabel("SWC")
    ax[0].set_ylabel("Sensor Reading")
    ax[0].legend()


    ax[1].plot(z_plot, x_plot, color='green', label='Parametric Derivative Fit')

    # plot the bootstrap results
    i = 0
    for sx, sz in boot_results:
        x_boot = sx(s_fine)
        z_boot = sz(s_fine)
        ax[1].plot(z_boot, x_boot, color='green', alpha=5.0/n_boots_parametric, label=f"Bootstrap Samples" if i == 0 else None)
        i += 1

    # ax[7].scatter(df['SWC'], df['value'], color='grey', label='Data Points')
    ax[1].scatter(swc_anchor, value_anchor, color='red', marker='x', s=100, zorder=5, label='Anchor Points')  
    ax[1].scatter(knots_z, knots_x, 
                color='black', 
                marker='D', 
                s=50, 
                zorder=6, 
                label='Spline Knots')
    ax[1].legend()
    # cut the swc axis to 0


    # Now plot equilibriation time as a function of water content
    ax[2].scatter(df["SWC"], df["equil_time_hours"] , c=df["sensor"].map(sensor_colors), label="Equilibration Time Delta")
    ax[2].set_xlabel("SWC")
    ax[2].set_ylabel("Equilibration Time Delta (hours)")

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


    plt.savefig("titration_plots.png")
    plt.show()