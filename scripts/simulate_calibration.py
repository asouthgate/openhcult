from __future__ import annotations
from typing import List, Tuple, Callable, Dict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar

def sigmoid(x):
  return 1 / (1 - np.exp(-x))

def sim_pot_watering_sequence(W, n_watering_events, Z0, sigma2, response_func):
    """Samples tuples (Z, X).
    Params:
        W: water quantity, fixed dZ
        n_watering_events: number of watering events to draw
        Z0: starting water content
        sigma2: noise in response
        response func: function mapping Z to X
    """
    dZ = np.ones(n_watering_events) * W
    Z = np.cumsum(dZ) + Z0
    Z = np.insert(Z, 0, Z0)
    X = np.array([response_func(z) for z in Z])
    X += np.random.normal(0, sigma2, len(Z))
    return Z, X

def Z2dZ(Z):
    return np.diff(Z)  # we want the first val to be zero

def dZ2S(dZ):
    # This is Z up to a constant. Z = c + sum dZ. We miss c.
    S = np.cumsum(dZ)
    S = np.insert(S, 0, 0)
    return S

def Z2X(Z, response_func, sigma2):
    return response_func(Z) + np.random.normal(0, sigma2, len(Z))

def infer_response_func(
    samples: List[Tuple[np.ndarray, np.ndarray]],
    anchor_points: List[Tuple[float, float]],
    *,
    anchor_weight: float = 2.0,
    shift_ridge: float = 0.0,
    shift_prior: float = 0.5,
    c_init: np.ndarray | None = None,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> Dict[str, object]:
    """
    Jointly estimate:
        - monotone decreasing function h
        - per-trajectory shifts c_s

    Model:
        Z_{s,i} = c_s + S_{s,i}
        X_{s,i} = h(Z_{s,i}) + eps
        Anchor points: (Z_a, X_a) noisy observations of same h

    Parameters
    ----------
    samples : list of (S, X) arrays
    anchor_points : list of (Z, X) anchor tuples
    anchor_weight : weight applied to anchor points
    shift_ridge : L2 penalty on shifts c_s (stabilizes weak overlap)
    shift_prior : prior mean for c_s (used with shift_ridge)
    c_init : optional initial shifts, length must match samples
    max_iter : max coordinate descent iterations
    tol : convergence tolerance on shifts

    Returns
    -------
    tuple with:
        c: np.ndarray of shifts
        h: callable h(u)
        errors: list of weighted SSE values per iteration
    """

    # ---- prepare data ----
    n_traj = len(samples)

    S_list = []
    X_list = []
    for S, X in samples:
        S = np.asarray(S).ravel()
        X = np.asarray(X).ravel()
        if S.shape != X.shape:
            raise ValueError(f"Each (S, X) must have same shape, got {S.shape} {X.shape}")
        S_list.append(S)
        X_list.append(X)

    if len(anchor_points) > 0:
        Z_anchor = np.array([z for (z, _) in anchor_points], dtype=float)
        X_anchor = np.array([x for (_, x) in anchor_points], dtype=float)
    else:
        Z_anchor = np.empty((0,), dtype=float)
        X_anchor = np.empty((0,), dtype=float)

    # initialize shifts
    if c_init is None:
        c = np.zeros(n_traj)
    else:
        c = np.asarray(c_init, dtype=float).ravel().copy()
        if c.shape[0] != n_traj:
            raise ValueError(f"c_init length {c.shape[0]} does not match samples {n_traj}")

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")

    def fit_h() -> Callable[[np.ndarray], np.ndarray]:
        """
        Fit monotone decreasing h given current shifts c_s.
        """
        U_all = []
        X_all = []
        W_all = []

        for s in range(n_traj):
            U_all.append(S_list[s] + c[s])
            X_all.append(X_list[s])
            W_all.append(np.ones_like(X_list[s]))

        if len(Z_anchor) > 0:
            U_all.append(Z_anchor)
            X_all.append(X_anchor)
            W_all.append(np.full_like(X_anchor, anchor_weight))

        U_all = np.concatenate(U_all)
        X_all = np.concatenate(X_all)
        W_all = np.concatenate(W_all)

        order = np.argsort(U_all)
        U_sorted = U_all[order]
        Y_sorted = -X_all[order]
        W_sorted = W_all[order]

        iso.fit(U_sorted, Y_sorted, sample_weight=W_sorted)

        def h(u: np.ndarray) -> np.ndarray:
            u = np.asarray(u, dtype=float)
            return -iso.predict(u)

        return h

    def update_shift(s: int, h: Callable[[np.ndarray], np.ndarray]) -> float:
        """
        Update shift c_s via 1D minimization.
        """

        S = S_list[s]
        X = X_list[s]

        span = np.ptp(S)
        if span == 0.0:
            span = 1.0

        def objective(cs: float) -> float:
            resid = X - h(S + cs)
            val = np.sum(resid ** 2)
            if shift_ridge > 0.0:
                val += shift_ridge * (cs - shift_prior) ** 2
            return val

        min_shift = -float(np.min(S))
        max_shift = 1.0 - float(np.max(S))
        result = minimize_scalar(
            objective,
            bounds=(max(c[s] - span, min_shift), min(c[s] + span, max_shift)),
            method="bounded",
        )
        return float(result.x)

    def compute_error(h: Callable[[np.ndarray], np.ndarray]) -> float:
        sse = 0.0
        for s in range(n_traj):
            resid = X_list[s] - h(S_list[s] + c[s])
            sse += float(np.sum(resid ** 2))
        if len(Z_anchor) > 0:
            resid = X_anchor - h(Z_anchor)
            sse += float(np.sum(anchor_weight * (resid ** 2)))
        return sse

    errors = []

    # ---- coordinate descent ----
    for _ in range(max_iter):

        h = fit_h()

        c_old = c.copy()

        for s in range(n_traj):
            c[s] = update_shift(s, h)

        h = fit_h()
        errors.append(compute_error(h))

        if np.max(np.abs(c - c_old)) < tol:
            break

    return c, h, errors

if __name__ == "__main__":
    W = 0.1 # normalized, between 0 and 1
    n_watering_events = 4
    sigma2 = 0.1
    response_func = sigmoid
    n_runs = 20
    # Simulate sequences of dQs for each pot, they may not span the whole range Qmin, Qmax (plants have narrow viability ranges)
    nS = 50
    anchor_sigma2 = 5.0

    hests = []
    errors_list = []

    example_samps = None
    example_Z_real = None
    example_c0 = None
    example_cest = None
    example_anchors = None

    for run in range(n_runs):
        samps = []
        Z_real = []

        for s in range(nS):
            Z0 = np.random.uniform(0.1, 1.0 - W * n_watering_events)
            Z, X = sim_pot_watering_sequence(W, n_watering_events, Z0, sigma2, response_func)
            dZs = Z2dZ(Z)
            S = dZ2S(dZs)
            samps.append((S, X))
            Z_real.append(Z)

        anchors = [
            (x, max(0.0, response_func(x) + np.random.normal(0.0, anchor_sigma2)))
            for x in np.linspace(0.1, 0.9, num=10)
        ]

        c0 = np.random.uniform(0.0, 1.0, size=len(samps))
        cest, hest, errors = infer_response_func(samps, anchors, c_init=c0, max_iter=10)
        hests.append(hest)
        errors_list.append(errors)

        if run == 0:
            example_samps = samps
            example_Z_real = Z_real
            example_c0 = c0
            example_cest = cest
            example_anchors = anchors

    samps = example_samps
    Z_real = example_Z_real
    c0 = example_c0
    cest = example_cest
    anchors = example_anchors

    fig, axes = plt.subplots(2, 3, sharex=False, figsize=(14, 8))
    ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()

    ax0.scatter([z for z, _ in anchors], [x for _, x in anchors], c="black", s=30)
    ax0.set_title(f"Anchor points ($\sigma^2 = {anchor_sigma2}$)")
    ax0.set_ylabel("X")
    ax0.set_xlabel("Z")

    for s in range(nS):
        S, X = samps[s]
        Zrs = Z_real[s]

        Zest0 = S + c0[s]
        ax2.plot(Zest0, X, c="#6b6b6b")
        ax2.plot(Zrs, X, c="#4136a3")

        Zest = S + cest[s]
        ax3.plot(Zest, X, c="#6b6b6b")
        ax3.plot(Zrs, X, c="#4136a3")

    ax1.hist(c0, bins=20, color="#4136a3", alpha=0.8)
    ax1.set_title("$c_0$ histogram (uniform distribution)")
    ax1.set_xlabel("c0")
    ax1.set_ylabel("Count")
    ax2.set_title("Data alignment with randomly initialized $c_0$")
    ax2.set_ylabel("X")
    ax2.set_xlabel("Z")
    ax3.set_title("Data alignment with estimated $\hat{c}$")
    ax3.set_ylabel("X")
    ax3.set_xlabel("Z")
    z_grid = np.linspace(0.1, 0.9, num=200)
    ax4.plot(z_grid, response_func(z_grid), c="#4136a3", label="True $h$")
    for i, hest in enumerate(hests):
        label = "Estimated $\hat{h}$" if i == 0 else None
        ax4.plot(z_grid, hest(z_grid), c="#e6a532", alpha=0.35, label=label)
    ax4.set_title("Response curve")
    ax4.set_xlabel("Z")
    ax4.set_ylabel("X")
    ax4.legend(frameon=False)

    for errors in errors_list:
        ax5.plot(range(1, len(errors) + 1), errors, c="#4136a3", alpha=0.35)
    ax5.set_title("Inference error (weighted SSE)")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Error")

    fig.suptitle(f"Example result for {n_runs} simulations ($W={W},\sigma^2={sigma2},h=1/(1 - exp(-x))$)", fontsize=16)
    plt.tight_layout()
    plt.savefig("simulation_example.png")
    plt.show()
