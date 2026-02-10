#!/usr/bin/env python3
"""Simulate calibration with watering events, piecewise X(Q), and sensor noise."""

from __future__ import annotations

import argparse
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
    n_sensors: int
    z_min: float
    z_max: float
    x_min: float
    x_max: float
    q0: float
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


def simulate(params: SimParams):
    rng = random.Random(params.seed)
    steps = int(params.t_end / params.dt) + 1
    times = np.arange(steps) * params.dt

    event_steps = sorted(rng.sample(range(1, steps - 1), params.water_events))
    dQ = np.zeros(steps)
    for step in event_steps:
        dQ[step] += (params.water_volume / params.soil_volume) / (params.z_max - params.z_min)

    Q = np.zeros(steps)
    Q[0] = params.q0
    for t in range(1, steps):
        Q[t] = Q[t - 1] + dQ[t] - params.eta
        Q[t] = max(0.0, min(1.0, Q[t]))

    q_edges, x_edges = build_piecewise_xq(
        params.k_bins, params.x_min, params.x_max, params.seed, params.x_edges
    )
    X = x_from_q(Q, q_edges, x_edges)

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
        noise = np.random.default_rng(params.seed + i).normal(0.0, params.noise_std, size=steps)
        scale = (X - params.x_min) / max(1e-6, params.x_max - params.x_min)
        scale = np.clip(scale, 0.0, 1.0) ** params.bias_scale_power
        X_i[i] = X + offset_series[i] * scale + noise

    return times, Q, dQ, q_edges, x_edges, X, X_i, event_steps


def plot_all(
    times: np.ndarray,
    Q: np.ndarray,
    q_edges: np.ndarray,
    x_edges: np.ndarray,
    X: np.ndarray,
    X_i: np.ndarray,
    x_hat: np.ndarray,
    event_steps: List[int],
    smooth_window: int,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    axes[0].plot(times, Q, label="Q(t)")
    axes[0].scatter(times[event_steps], Q[event_steps], color="red", s=12, label="Water events")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("Q")
    axes[0].legend()

    axes[1].step(q_edges, x_edges, where="post", label="true X(Q)")
    order = np.argsort(Q)
    q_sorted = Q[order]
    x_sorted = x_hat[order]
    q_grid = np.linspace(0.0, 1.0, 200)
    x_grid = np.interp(q_grid, q_sorted, x_sorted)
    x_grid = smooth_series(x_grid, smooth_window)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    x_grid = iso.fit_transform(q_grid, x_grid)
    axes[1].plot(q_grid, x_grid, color="black", linewidth=2, label="estimated X(Q)")
    axes[1].set_xlabel("Q")
    axes[1].set_ylabel("X")
    axes[1].legend()

    axes[2].plot(times, X, label="X(t) mean", color="black")
    for i in range(X_i.shape[0]):
        axes[2].plot(times, X_i[i], alpha=0.6, label=f"X_i (sensor {i+1})")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("X")
    axes[2].legend(ncol=2)

    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-end", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--n-sensors", type=int, default=2)
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=0.5)
    parser.add_argument("--x-min", type=float, default=200.0)
    parser.add_argument("--x-max", type=float, default=900.0)
    parser.add_argument("--q0", type=float, default=0.0)
    parser.add_argument("--water-events", type=int, default=8)
    parser.add_argument("--water-volume", type=float, default=0.04)
    parser.add_argument("--soil-volume", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=2.0)
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

    params = SimParams(
        t_end=int(args.t_end),
        dt=args.dt,
        n_sensors=args.n_sensors,
        z_min=args.z_min,
        z_max=args.z_max,
        x_min=args.x_min,
        x_max=args.x_max,
        q0=args.q0,
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

    times, Q, dQ, q_edges, x_edges, X, X_i, event_steps = simulate(params)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    x_hat = iso.fit_transform(Q, X)
    plot_all(times, Q, q_edges, x_edges, X, X_i, x_hat, event_steps, args.smooth_window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
