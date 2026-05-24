from __future__ import annotations

import numpy as np

PAPER_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

PAPER_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#333333",
    "font.family": "serif",
    "font.size": 11,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
}


def apply_paper_theme():
    import matplotlib.pyplot as plt

    plt.rcParams.update(PAPER_RC)


def plot_corner(cal, n_sensors=None, out=None, title=None):
    import matplotlib.pyplot as plt

    apply_paper_theme()
    if n_sensors is None:
        n_sensors = getattr(cal, "n_sensors", 1)
    params = cal.posterior_params()
    sensor_color = PAPER_COLORS[:n_sensors]
    sigma_color = PAPER_COLORS[2] if len(PAPER_COLORS) > 2 else PAPER_COLORS[-1]
    scale_color = sigma_color

    if n_sensors == 1:
        param_arrays = [
            ("$S$", params["scale"]),
            ("$k$", params["k"].flatten()),
            ("$q_{\\rm dry}$", params["f_int"].flatten()),
            ("$\\sigma^2$", params["sigma2"]),
        ]
        param_colors = [scale_color, sensor_color[0], sensor_color[0], sigma_color]
    else:
        param_arrays = [("$S$", params["scale"])]
        param_colors = [scale_color]
        for i in range(n_sensors):
            param_arrays.append((f"$k_{i+1}$", params["k"][:, i]))
            param_colors.append(sensor_color[i])
        for i in range(n_sensors):
            param_arrays.append(
                ("$q_{{\\rm dry," + str(i + 1) + "}}$", params["f_int"][:, i])
            )
            param_colors.append(sensor_color[i])
        param_arrays.append(("$\\sigma^2$", params["sigma2"]))
        param_colors.append(sigma_color)

    n_params = len(param_arrays)
    names = [p[0] for p in param_arrays]
    arrays = [p[1] for p in param_arrays]
    step = max(1, len(arrays[0]) // 500)

    fig, axes = plt.subplots(
        n_params, n_params, figsize=(2.5 * n_params, 2.5 * n_params)
    )
    if n_params == 1:
        axes = np.array([[axes]])
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i][j]
            if i == j:
                ax.hist(
                    arrays[i][::step],
                    bins=40,
                    density=True,
                    color=param_colors[i],
                    alpha=0.7,
                )
                ax.axvline(np.median(arrays[i]), color="#333333", linewidth=1)
            elif i > j:
                ax.scatter(
                    arrays[j][::step],
                    arrays[i][::step],
                    s=8,
                    alpha=0.4,
                    color=param_colors[i],
                    edgecolors="none",
                )
            else:
                ax.set_visible(False)
            if j == 0:
                ax.set_ylabel(names[i])
            if i == n_params - 1:
                ax.set_xlabel(names[j])
    fig.tight_layout()
    if title is not None:
        fig.suptitle(title)
    if out is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, dpi=300)
    plt.close(fig)
    return fig


def plot_calibration_curves(single_cals, joint_cal, d, out=None):
    import matplotlib.pyplot as plt

    apply_paper_theme()
    n_sensors = d["n_sensors"]
    colors = PAPER_COLORS[:n_sensors] + [PAPER_COLORS[n_sensors % len(PAPER_COLORS)]]
    names = [f"Sensor {i+1}" for i in range(n_sensors)] + ["Joint"]

    t = np.linspace(0, 1, 500)
    singular_xmin = []
    for i in range(n_sensors):
        xm = single_cals[i].data_xmin
        singular_xmin.append(float(xm[0]) if xm.ndim > 0 else float(xm))

    fig, ax = plt.subplots(figsize=(7, 5))

    for i in range(n_sensors):
        xmin_i = singular_xmin[i]
        x_i = xmin_i + t * (d["xmax"] - xmin_i)
        mean_i, ci_lo_i, ci_hi_i = single_cals[i].predict(x_i)
        ax.plot(t, mean_i, color=colors[i], label=names[i], linewidth=1.0)
        ax.fill_between(t, ci_lo_i, ci_hi_i, color=colors[i], alpha=0.15)

        mask = d["labels"] == i
        c_x = d["chords_x"][mask]
        c_dx = d["chords_dx"][mask]
        c_dy = d["chords_dy"][mask]
        for j in range(len(c_x)):
            t_start = (c_x[j] - xmin_i) / (d["xmax"] - xmin_i)
            t_end = (c_x[j] + c_dx[j] - xmin_i) / (d["xmax"] - xmin_i)
            if t_start < 0 or t_end > 1:
                continue
            swc_at_start = float(np.interp(c_x[j], x_i, mean_i))
            ax.plot(
                [t_start, t_end],
                [swc_at_start, swc_at_start + c_dy[j]],
                color=colors[i],
                linewidth=1.0,
                alpha=0.5,
            )

    x_joint = np.column_stack(
        [
            singular_xmin[i] + t * (d["xmax"] - singular_xmin[i])
            for i in range(n_sensors)
        ]
    )
    mean_joint, ci_lo_joint, ci_hi_joint = joint_cal.predict(x_joint)
    ax.plot(t, mean_joint, color=colors[-1], label=names[-1], linewidth=1.0)
    ax.fill_between(t, ci_lo_joint, ci_hi_joint, color=colors[-1], alpha=0.15)

    ax.set_xlabel("$t$ (normalised sensor range)")
    ax.set_ylabel("SWC")
    ax.legend()
    fig.tight_layout()
    if out is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, dpi=300)
    plt.close(fig)
    return fig


def plot_timeseries(single_cals, joint_cal, d, out=None):
    import matplotlib.pyplot as plt

    apply_paper_theme()
    n_sensors = d["n_sensors"]
    colors = PAPER_COLORS[:n_sensors] + [PAPER_COLORS[n_sensors % len(PAPER_COLORS)]]
    names = [f"Sensor {i+1}" for i in range(n_sensors)] + ["Joint"]

    ts = d["common_ts"]
    volts = d["volt_matrix"]
    ts_hours = (ts - ts[0]) / np.timedelta64(1, "h")

    single_ts_means = [single_cals[i].predict(volts[:, i])[0] for i in range(n_sensors)]
    mean_ts_joint, ci_lo_ts_joint, ci_hi_ts_joint = joint_cal.predict(volts)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n_sensors):
        _, ci_lo, ci_hi = single_cals[i].predict(volts[:, i])
        ax.plot(
            ts_hours, single_ts_means[i], color=colors[i], label=names[i], linewidth=0.8
        )
        ax.fill_between(ts_hours, ci_lo, ci_hi, color=colors[i], alpha=0.15)
    ax.plot(ts_hours, mean_ts_joint, color=colors[-1], label=names[-1], linewidth=0.8)
    ax.fill_between(
        ts_hours, ci_lo_ts_joint, ci_hi_ts_joint, color=colors[-1], alpha=0.15
    )
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("SWC")
    ax.legend()
    fig.tight_layout()
    if out is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, dpi=300)
    plt.close(fig)
    return fig


def plot_smoothed_voltage(d, out=None):
    import matplotlib.pyplot as plt

    from hcultinf.detection import smooth_and_downsample

    apply_paper_theme()
    n_sensors = d["n_sensors"]
    colors = PAPER_COLORS[:n_sensors] + [PAPER_COLORS[n_sensors % len(PAPER_COLORS)]]

    ts = d["common_ts"]
    volts = d["volt_matrix"]
    ts_hours = (ts - ts[0]) / np.timedelta64(1, "h")

    fig, axes = plt.subplots(n_sensors, 1, figsize=(10, 3 * n_sensors), sharex=True)
    if n_sensors == 1:
        axes = [axes]
    for i in range(n_sensors):
        ax = axes[i]
        ds_ts, ds_vals = smooth_and_downsample(ts, volts[:, i])
        ds_hours = (ds_ts - ds_ts[0]) / np.timedelta64(1, "h")
        ax.scatter(ts_hours, volts[:, i], color=colors[i], alpha=0.15, s=3, marker="x")
        ax.plot(
            ds_hours,
            ds_vals,
            color=colors[i],
            linewidth=1.0,
            label=f"Sensor {i+1} (smoothed)",
        )
        ax.set_ylabel("mV")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (hours)")
    fig.tight_layout()
    if out is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, dpi=300)
    plt.close(fig)
    return fig
