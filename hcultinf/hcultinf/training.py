from __future__ import annotations

import numpy as np

from hcultinf.detection import SegmentDetector


def filter_confirmed_watering_events(observations):
    return [
        (obs_id, ts, note)
        for obs_id, ts, note in observations
        if "WATER" in note and "AUTO" not in note
    ]


def _match_indices(confirmed_times, candidate_times, epsilon):
    matched_confirmed = set()
    matched_candidates = set()
    pairs = []
    for i, ct in enumerate(confirmed_times):
        for j, dt in enumerate(candidate_times):
            if j not in matched_candidates and abs(ct - dt) <= epsilon:
                matched_confirmed.add(i)
                matched_candidates.add(j)
                pairs.append((i, j))
                break
    return matched_confirmed, matched_candidates, pairs


def match_events(confirmed_events, candidate_times, epsilon_ms=900_000):
    """Match candidate timestamps to confirmed events within an epsilon window.

    Returns dict with matched, missed, spurious counts.
    """
    epsilon = np.timedelta64(epsilon_ms, "ms")
    confirmed_times = [ts for _, ts, _ in confirmed_events]
    matched_confirmed, matched_candidates, _ = _match_indices(
        confirmed_times, candidate_times, epsilon
    )
    return {
        "matched": len(matched_confirmed),
        "missed": len(confirmed_times) - len(matched_confirmed),
        "spurious": len(candidate_times) - len(matched_candidates),
    }


def _run_detector(timeseries, params):
    """Run SegmentDetector over all sensors, return list of (sensor_key, event, detector, t0)."""
    entries = []
    for sensor, points in timeseries.items():
        times = np.array([t for t, _ in points])
        values = np.array([v for _, v in points], dtype=float)
        if len(times) < 2:
            continue
        detector = SegmentDetector(times, values, **params)
        for event in detector.get_watering_events():
            t0, _ = event.get_model_active_interval()
            entries.append((sensor, event, detector, t0))
    return entries


def grid_search(confirmed_events, timeseries, param_grid, match_window_ms=900_000):
    """Run detector for each param combination, return list of result dicts."""
    epsilon = np.timedelta64(match_window_ms, "ms")
    confirmed_times = [ts for _, ts, _ in confirmed_events]
    results = []
    for params in param_grid:
        entries = _run_detector(timeseries, params)
        detected_times = [t0 for *_, t0 in entries]
        matched_confirmed, matched_candidates, _ = _match_indices(
            confirmed_times, detected_times, epsilon
        )
        tp = len(matched_confirmed)
        fn = len(confirmed_times) - tp
        fp = len(detected_times) - len(matched_candidates)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        results.append(
            {
                "params": params,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tpr": tpr,
                "fdr": fdr,
                "f1": f1,
            }
        )
    return results


def plot_roc(results):
    """Scatter of TPR vs FDR (1-precision) for all grid points, with Pareto frontier."""
    import matplotlib.pyplot as plt

    tprs = [r["tpr"] for r in results]
    fdrs = [r["fdr"] for r in results]
    f1s = [r["f1"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sc = ax1.scatter(fdrs, tprs, c=f1s, cmap="RdYlGn", vmin=0, vmax=1, alpha=0.6, s=20)
    plt.colorbar(sc, ax=ax1, label="F1")

    sorted_by_fdr = sorted(results, key=lambda r: r["fdr"])
    pareto, max_tpr = [], -1.0
    for r in sorted_by_fdr:
        if r["tpr"] > max_tpr:
            max_tpr = r["tpr"]
            pareto.append(r)
    ax1.plot(
        [r["fdr"] for r in pareto],
        [r["tpr"] for r in pareto],
        "k-",
        linewidth=1.5,
        label="Pareto front",
    )

    best = max(results, key=lambda r: r["f1"])
    ax1.scatter(
        [best["fdr"]],
        [best["tpr"]],
        marker="*",
        color="red",
        s=200,
        zorder=5,
        label=f"Best F1={best['f1']:.2f}",
    )
    ax1.set_xlabel("FDR (false discovery rate = 1 - precision)")
    ax1.set_ylabel("TPR (recall)")
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()

    top = sorted(results, key=lambda r: r["f1"], reverse=True)[:10]
    for i, r in enumerate(top):
        p = r["params"]
        label = (
            f"tau={p['emwa_tau_minutes']:.1f} "
            f"trig={p['trigger_thresh']:.2f} "
            f"rel={p['release_thresh']:.2f} "
            f"F1={r['f1']:.2f}"
        )
        ax2.barh(i, r["f1"])
        ax2.text(r["f1"] + 0.01, i, label, va="center", fontsize=8)
    ax2.set_xlabel("F1")
    ax2.set_xlim(0, 1.3)
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels([f"#{i + 1}" for i in range(len(top))])
    ax2.set_title("Top 10 by F1")

    fig.suptitle("Grid search results")
    fig.tight_layout()
    plt.show()
    return best


def plot_debug_events(timeseries, params, confirmed_events, epsilon_ms=900_000):
    """Cycle through detected events showing the lognormal fit. Keys: ←/→ to navigate, q to quit."""
    import matplotlib.pyplot as plt

    epsilon = np.timedelta64(epsilon_ms, "ms")
    confirmed_times = [ts for _, ts, _ in confirmed_events]

    entries = _run_detector(timeseries, params)
    if not entries:
        print("No events detected with these params.")
        return

    detected_times = [t0 for *_, t0 in entries]
    _, matched_candidates, pairs = _match_indices(
        confirmed_times, detected_times, epsilon
    )
    candidate_to_confirmed = {di: ci for ci, di in pairs}

    state = {"idx": 0}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.canvas.manager.set_window_title("Debug: detected events")

    def draw(idx):
        ax1.cla()
        ax2.cla()
        sensor, event, detector, t0 = entries[idx]
        is_tp = idx in matched_candidates

        dur_ms = int((event.end - event.start).astype("timedelta64[ms]").astype(int))
        pad = np.timedelta64(max(dur_ms, 60_000) * 2, "ms")
        w_start = event.start - pad
        w_end = event.end + pad

        rt = detector._resampled_times
        mask = (rt >= w_start) & (rt <= w_end)
        t_win = rt[mask]
        emwa_win = np.array(detector._resampled_emwa)[mask]
        vel_win = detector._resampled_vel_smoothed[mask]

        raw_mask = (detector._time_arr >= w_start) & (detector._time_arr <= w_end)
        ax1.scatter(
            detector._time_arr[raw_mask],
            detector._values_arr[raw_mask],
            alpha=0.4,
            s=10,
            color="tab:blue",
            label="raw",
        )
        ax1.plot(t_win, emwa_win, color="tab:blue", label="EWMA")
        ax1.axvspan(event.start, event.end, alpha=0.2, color="orange", label="segment")
        ax1.axvline(
            t0, color="green", linestyle="--", linewidth=1.5, label="detected t0"
        )

        if is_tp:
            ct = confirmed_times[candidate_to_confirmed[idx]]
            ax1.axvline(
                ct, color="blue", linestyle=":", linewidth=1.5, label="confirmed"
            )

        ax1.legend(fontsize=8)
        ax1.set_ylabel("sensor value")

        ax2.plot(t_win, vel_win, color="tab:red", label="velocity (EWMA)")
        if event.pred_func is not None:
            vpred = event.pred_func(rt[mask])
            ax2.plot(
                t_win,
                vpred,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="lognormal fit",
            )
        ax2.axhline(0, color="grey", linewidth=0.5)
        ax2.legend(fontsize=8)
        ax2.set_ylabel("velocity")

        gof = event.gof_d or {}
        status = "TP" if is_tp else "FP"
        nrmse = f"nrmse={gof.get('nrmse', float('nan')):.3f}" if gof else ""
        fig.suptitle(
            f"[{idx + 1}/{len(entries)}] {status} — {sensor} — {np.datetime_as_string(t0, unit='m')}  {nrmse}\n"
            f"← → to navigate, q to quit"
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "n"):
            state["idx"] = (state["idx"] + 1) % len(entries)
            draw(state["idx"])
        elif event.key in ("left", "p"):
            state["idx"] = (state["idx"] - 1) % len(entries)
            draw(state["idx"])
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw(0)
    plt.tight_layout()
    plt.show()
