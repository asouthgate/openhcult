#!/usr/bin/env python3
"""Simulate calibration with watering events, piecewise X(Q), and sensor noise."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class SimParams:
    t_end: int
    dt: float
    n_pots: int
    n_sensors: int
    z_min: float
    z_max: float
    x_min: float
    x_max: float
    q0: float
    q_ranges: List[Tuple[float, float]]
    water_events: int
    water_volume: float
    soil_volume: float
    eta: float
    noise_std: float
    sensor_offset_std: float
    k_bins: int
    x_edges: List[float] | None
    bias_decay: float
    bias_scale_power: float
    seed: int


def build_piecewise_xq(
    k_bins: int, x_min: float, x_max: float, seed: int, x_edges: List[float] | None
) -> Tuple[np.ndarray, np.ndarray]:
    q_edges = np.linspace(0.0, 1.0, k_bins + 1)
    if x_edges is not None:
        if len(x_edges) != k_bins + 1:
            raise ValueError("x_edges must have length k_bins + 1")
        return q_edges, np.array(x_edges, dtype=float)
    rng = random.Random(seed)
    weights = [rng.random() + 0.1 for _ in range(k_bins)]
    total = sum(weights)
    slopes = [(x_max - x_min) * w / total for w in weights]
    built = [x_max]
    acc = x_max
    for s in slopes:
        acc -= s
        built.append(acc)
    return q_edges, np.array(built)


def x_from_q(q: np.ndarray, q_edges: np.ndarray, x_edges: np.ndarray) -> np.ndarray:
    q = np.clip(q, 0.0, 1.0)
    return np.interp(q, q_edges, x_edges)


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _simulate_pot(params: SimParams, pot_idx: int, q_low: float, q_high: float, q_edges, x_edges):
    steps = int(params.t_end / params.dt) + 1
    times = np.arange(steps) * params.dt

    rng_events = random.Random(params.seed + pot_idx * 17)
    event_steps = sorted(rng_events.sample(range(1, steps - 1), params.water_events))
    dQ = np.zeros(steps)
    for step in event_steps:
        dQ[step] += (params.water_volume / params.soil_volume) / (params.z_max - params.z_min)

    Q = np.zeros(steps)
    q0 = params.q0
    if not math.isfinite(q0):
        q0 = 0.5 * (q_low + q_high)
    Q[0] = q0
    for t in range(1, steps):
        Q[t] = Q[t - 1] + dQ[t] - params.eta
        Q[t] = max(q_low, min(q_high, Q[t]))

    X = x_from_q(Q, q_edges, x_edges)

    rng = random.Random(params.seed + pot_idx * 31)
    offsets = [rng.gauss(0.0, params.sensor_offset_std) for _ in range(params.n_sensors)]
    offset_series = np.zeros((params.n_sensors, steps))
    for i in range(params.n_sensors):
        offset_series[i, 0] = offsets[i]
    for t in range(1, steps):
        for i in range(params.n_sensors):
            offset_series[i, t] = offset_series[i, t - 1]
            if t in event_steps:
                decay = max(0.0, min(1.0, params.bias_decay))
                offset_series[i, t] = offset_series[i, t] * (1.0 - decay)

    X_i = np.zeros((params.n_sensors, steps))
    for i in range(params.n_sensors):
        noise = np.random.default_rng(params.seed + pot_idx * 11 + i).normal(
            0.0, params.noise_std, size=steps
        )
        scale = (X - params.x_min) / max(1e-6, params.x_max - params.x_min)
        scale = np.clip(scale, 0.0, 1.0) ** params.bias_scale_power
        X_i[i] = X + offset_series[i] * scale + noise

    X_hat = X_i.mean(axis=0)
    return {
        "times": times,
        "Q": Q,
        "dQ": dQ,
        "X": X,
        "X_i": X_i,
        "X_hat": X_hat,
        "events": event_steps,
    }


def simulate(params: SimParams):
    q_edges, x_edges = build_piecewise_xq(
        params.k_bins, params.x_min, params.x_max, params.seed, params.x_edges
    )
    pots = []
    for idx, (q_low, q_high) in enumerate(params.q_ranges):
        pots.append(_simulate_pot(params, idx, q_low, q_high, q_edges, x_edges))
    return q_edges, x_edges, pots


def plot_all(
    q_edges: np.ndarray,
    x_edges: np.ndarray,
    pots: List[dict],
    smooth_window: int,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 13), constrained_layout=True)

    for idx, pot in enumerate(pots):
        axes[0].plot(pot["times"], pot["Q"], label=f"Q(t) pot {idx + 1}")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("Q")
    axes[0].legend()

    axes[1].step(x_edges, q_edges, where="post", label="true Q(X)")
    pooled_Q = np.concatenate([pot["Q"] for pot in pots])
    pooled_X = np.concatenate([pot["X_hat"] for pot in pots])
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    q_hat = iso.fit_transform(pooled_X, pooled_Q)
    order = np.argsort(pooled_X)
    x_sorted = pooled_X[order]
    q_sorted = q_hat[order]
    x_grid = np.linspace(min(x_sorted), max(x_sorted), 200)
    q_grid = np.interp(x_grid, x_sorted, q_sorted)
    q_grid = smooth_series(q_grid, smooth_window)
    q_grid = iso.fit_transform(x_grid, q_grid)
    axes[1].plot(x_grid, q_grid, color="black", linewidth=2, label="estimated Q(X)")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Q")
    axes[1].legend()

    for idx, pot in enumerate(pots):
        times = pot["times"]
        X = pot["X"]
        Q = pot["Q"]
        xs = []
        slopes = []
        for event_idx in pot["events"]:
            if event_idx <= 0 or event_idx >= len(times):
                continue
            dQ = Q[event_idx] - Q[event_idx - 1]
            dX = X[event_idx] - X[event_idx - 1]
            if abs(dX) < 1e-6:
                continue
            xs.append(X[event_idx - 1])
            slopes.append(dQ / dX)
        axes[2].scatter(
            xs,
            slopes,
            alpha=0.7,
            s=18,
            label=f"pot {idx + 1}",
        )
    axes[2].axhline(0.0, color="gray", linewidth=0.5)
    axes[2].set_xlabel("X (at event)")
    axes[2].set_ylabel("dQ/dX")
    axes[2].legend(ncol=2)

    for idx, pot in enumerate(pots):
        times = pot["times"]
        X = pot["X"]
        X_hat = pot["X_hat"]
        X_i = pot["X_i"]
        (true_line,) = axes[3].plot(times, X, linewidth=2.0, label=f"X(t) true pot {idx + 1}")
        axes[3].plot(
            times,
            X_hat,
            linestyle="--",
            linewidth=1.6,
            color=true_line.get_color(),
            label=f"X(t) mean pot {idx + 1}",
        )
        for s_idx in range(X_i.shape[0]):
            axes[3].plot(
                times,
                X_i[s_idx],
                alpha=0.35,
                linewidth=0.8,
                color=true_line.get_color(),
            )
    axes[3].set_xlabel("time")
    axes[3].set_ylabel("X")
    axes[3].legend(ncol=2)

    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-end", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--n-pots", type=int, default=3)
    parser.add_argument("--n-sensors", type=int, default=2)
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=0.5)
    parser.add_argument("--x-min", type=float, default=200.0)
    parser.add_argument("--x-max", type=float, default=900.0)
    parser.add_argument("--q0", type=float, default=float("nan"))
    parser.add_argument(
        "--q-ranges",
        type=str,
        default="0.05-0.35,0.25-0.6,0.5-0.9",
        help="Comma-separated q_low-q_high ranges per pot.",
    )
    parser.add_argument("--water-events", type=int, default=8)
    parser.add_argument("--water-volume", type=float, default=0.04)
    parser.add_argument("--soil-volume", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=20.0)
    parser.add_argument("--sensor-offset-std", type=float, default=300.0)
    parser.add_argument("--k-bins", type=int, default=5)
    parser.add_argument(
        "--x-edges",
        type=str,
        default=None,
        help="Comma-separated list of x-edges for piecewise X(Q). Length must be k_bins+1.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--bias-decay", type=float, default=0.0)
    parser.add_argument("--bias-scale-power", type=float, default=0.5)
    args = parser.parse_args()

    x_edges = None
    if args.x_edges:
        x_edges = [float(val.strip()) for val in args.x_edges.split(",") if val.strip()]

    q_ranges = []
    for chunk in args.q_ranges.split(","):
        bounds = chunk.split("-")
        if len(bounds) != 2:
            raise ValueError("q-ranges must be in q_low-q_high format")
        q_ranges.append((float(bounds[0]), float(bounds[1])))
    if len(q_ranges) != args.n_pots:
        raise ValueError("q-ranges must provide one range per pot")

    params = SimParams(
        t_end=int(args.t_end),
        dt=args.dt,
        n_pots=args.n_pots,
        n_sensors=args.n_sensors,
        z_min=args.z_min,
        z_max=args.z_max,
        x_min=args.x_min,
        x_max=args.x_max,
        q0=args.q0,
        q_ranges=q_ranges,
        water_events=args.water_events,
        water_volume=args.water_volume,
        soil_volume=args.soil_volume,
        eta=args.eta,
        noise_std=args.noise_std,
        sensor_offset_std=args.sensor_offset_std,
        k_bins=args.k_bins,
        x_edges=x_edges,
        bias_decay=args.bias_decay,
        bias_scale_power=args.bias_scale_power,
        seed=args.seed,
    )

    q_edges, x_edges, pots = simulate(params)
    plot_all(q_edges, x_edges, pots, args.smooth_window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
