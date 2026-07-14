from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from hcultinf.detection import smooth_and_downsample
from hcultinf.exp_mcmc import _ParamLayout


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
    plt.rcParams.update(PAPER_RC)


def plot_corner(cal, n_sensors=None, out=None, title=None, true_values=None):
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
        _true_vals_builder = lambda tv: [
            tv.get("scale"),
            tv.get("k"),
            tv.get("f_int"),
            tv.get("sigma2"),
        ]
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

        def _true_vals_builder(tv):
            k_vals = tv.get("k", [])
            f_vals = tv.get("f_int", [])
            result = [tv.get("scale")]
            result.extend(k_vals)
            result.extend(f_vals)
            result.append(tv.get("sigma2"))
            return result

    if true_values is not None:
        true_vals = _true_vals_builder(true_values)
    else:
        true_vals = None

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
                if true_vals is not None and true_vals[i] is not None:
                    ax.axvline(
                        true_vals[i],
                        color="#333333",
                        linewidth=1.5,
                        linestyle="--",
                    )
            elif i > j:
                ax.scatter(
                    arrays[j][::step],
                    arrays[i][::step],
                    s=8,
                    alpha=0.4,
                    color=param_colors[i],
                    edgecolors="none",
                )
                if (
                    true_vals is not None
                    and true_vals[j] is not None
                    and true_vals[i] is not None
                ):
                    ax.axvline(true_vals[j], color="#333333", linewidth=0.8, linestyle="--", alpha=0.6)
                    ax.axhline(true_vals[i], color="#333333", linewidth=0.8, linestyle="--", alpha=0.6)
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

    ax.set_xlabel("$u$ (normalised sensor range)")
    ax.set_ylabel("SWC")
    ax.legend()
    fig.tight_layout()
    if out is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, dpi=300)
    plt.close(fig)
    return fig


def plot_calibration_curve_simulated(
    single_cals,
    joint_cal,
    true_fns,
    xmins,
    xmax,
    chords_x,
    chords_dx,
    chords_dy,
    sensor_labels,
    out=None,
    title=None,
):

    apply_paper_theme()
    n_sensors = len(single_cals)
    colors = PAPER_COLORS[:n_sensors] + [PAPER_COLORS[n_sensors % len(PAPER_COLORS)]]

    fig, axes = plt.subplots(
        n_sensors, 1, figsize=(8, 4 * n_sensors), sharex=False
    )
    if n_sensors == 1:
        axes = [axes]

    t_grid = np.linspace(0, 1, 500)

    for i in range(n_sensors):
        ax = axes[i]
        xmin_i = float(xmins[i])
        x_range = xmin_i + t_grid * (xmax - xmin_i)

        x_2d = np.full((len(x_range), n_sensors), np.nan)
        x_2d[:, i] = x_range

        true_y = np.asarray(true_fns[i](x_range))

        mean_single, ci_lo_s, ci_hi_s = single_cals[i].predict(x_range)
        mean_joint, ci_lo_j, ci_hi_j = joint_cal.predict(x_2d)

        ax.plot(
            x_range,
            true_y,
            color="#333333",
            linestyle="--",
            linewidth=1.5,
            label="True curve",
        )
        ax.plot(
            x_range,
            mean_single,
            color=colors[i],
            linewidth=1.0,
            label=f"Sensor {i+1} alone",
        )
        ax.fill_between(x_range, ci_lo_s, ci_hi_s, color=colors[i], alpha=0.15)
        ax.plot(
            x_range,
            mean_joint,
            color=colors[-1],
            linewidth=1.0,
            label="Joint",
        )
        ax.fill_between(x_range, ci_lo_j, ci_hi_j, color=colors[-1], alpha=0.15)

        mask = np.asarray(sensor_labels) == i
        c_x = np.asarray(chords_x)[mask]
        c_dx = np.asarray(chords_dx)[mask]
        c_dy = np.asarray(chords_dy)[mask]
        for j in range(len(c_x)):
            swc_start = float(np.interp(c_x[j], x_range, mean_joint))
            ax.plot(
                [c_x[j], c_x[j] + c_dx[j]],
                [swc_start, swc_start + c_dy[j]],
                color=colors[i],
                linewidth=1.0,
                alpha=0.4,
                label="chord" if j == 0 and i == 0 else None,
            )

        ax.set_xlabel(f"Sensor {i+1} reading")
        ax.set_ylabel("SWC")
        ax.legend(fontsize=9)

    fig.tight_layout()
    if title is not None:
        fig.suptitle(title)
    if out is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, dpi=300)
    plt.close(fig)
    return fig


def plot_convergence_error(n_values, error_dict, out=None, title=None):
    apply_paper_theme()
    colors = [PAPER_COLORS[0], PAPER_COLORS[1], PAPER_COLORS[2]]

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (label, (means, stds)) in enumerate(error_dict.items()):
        n_vals = np.asarray(n_values)
        m = np.asarray(means)
        s = np.asarray(stds)
        color = colors[idx % len(colors)]
        ax.errorbar(
            n_vals,
            m,
            yerr=s,
            color=color,
            marker="o",
            capsize=3,
            label=label,
            linewidth=1.2,
        )

    ax.set_xlabel("N (chords per sensor)")
    ax.set_ylabel("Curve error (MAE)")
    # ax.set_xscale("log", base=2)
    ax.legend()
    fig.tight_layout()
    if title is not None:
        fig.suptitle(title)
    if out is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, dpi=300)
    plt.close(fig)
    return fig


def plot_timeseries(single_cals, joint_cal, d, out=None):
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


def plot_traces(cal, n_sensors=None, n_burn=None, out=None, title=None):

    apply_paper_theme()
    if n_sensors is None:
        n_sensors = getattr(cal, "n_sensors", 1)

    layout = _ParamLayout(n_sensors)
    chain = cal._chain
    log_prob = cal._log_prob

    param_names = (
        ["log_scale"]
        + [f"k_{i}" for i in range(n_sensors)]
        + [f"f_int_{i}" for i in range(n_sensors)]
        + ["log_sigma"]
    )

    n_params = layout.dim
    n_rows = n_params + 1
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(12, 2.0 * n_rows), sharex=True
    )
    if n_rows == 1:
        axes = [axes]

    for p_idx in range(n_params):
        ax = axes[p_idx]
        for w in range(chain.shape[1]):
            ax.plot(chain[:, w, p_idx], linewidth=0.3, alpha=0.5, color="#333333")
        if n_burn is not None:
            ax.axvline(n_burn, color="#999999", linewidth=0.8, linestyle="--")
        ax.set_ylabel(param_names[p_idx], fontsize=8)

    ax = axes[-1]
    for w in range(log_prob.shape[1]):
        ax.plot(log_prob[:, w], linewidth=0.3, alpha=0.5, color="#333333")
    if n_burn is not None:
        ax.axvline(n_burn, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_ylabel("log_prob", fontsize=8)
    ax.set_xlabel("step")

    fig.tight_layout()
    if title is not None:
        fig.suptitle(title)
    if out is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, dpi=300)
    plt.close(fig)
    return fig


def plot_chord_noise(
    chords_x,
    chords_dx,
    chords_dy,
    sensor_labels,
    true_fns,
    noise_level,
    out=None,
):
    apply_paper_theme()
    n_sensors = len(true_fns)
    colors = PAPER_COLORS[:n_sensors]

    chords_x = np.asarray(chords_x, dtype=float)
    chords_dx = np.asarray(chords_dx, dtype=float)
    chords_dy = np.asarray(chords_dy, dtype=float)
    labels = np.asarray(sensor_labels)

    all_true_dy = []
    all_obs_dy = []
    all_lbls = []

    for s in range(n_sensors):
        mask = labels == s
        true_dy = true_fns[s](chords_x[mask] + chords_dx[mask]) - true_fns[s](chords_x[mask])
        all_true_dy.append(true_dy)
        all_obs_dy.append(chords_dy[mask])
        all_lbls.extend([s] * np.sum(mask))

    true_dy_all = np.concatenate(all_true_dy)
    obs_dy_all = np.concatenate(all_obs_dy)
    lbl_all = np.array(all_lbls)

    fig, ax = plt.subplots(figsize=(6, 5))

    # ax.set_xscale("log")
    # ax.set_yscale("log")

    ax_min = 1.0
    ax_max = max(obs_dy_all.max(), true_dy_all.max()) * 1.2
    ax.plot(
        [ax_min, ax_max],
        [ax_min, ax_max],
        color="#333333",
        linewidth=0.8,
        linestyle="--",
        label="y = mx",
    )

    for s in range(n_sensors):
        mask = lbl_all == s
        ax.scatter(
            true_dy_all[mask],
            obs_dy_all[mask],
            s=24,
            alpha=0.7,
            color=colors[s],
            edgecolors="none",
            label=f"Sensor {s + 1}",
        )

    ax.set_xlabel("True $\\Delta$ SWC")
    ax.set_ylabel("Observed $\\Delta$ SWC")
    # ax.set_title(
    #     f"Chord noise (lognormal, $\\sigma$={noise_level})"
    # )
    ax.legend()
    fig.tight_layout()

    if out is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, dpi=300)
    plt.close(fig)
    return fig
