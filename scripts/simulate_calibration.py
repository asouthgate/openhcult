from __future__ import annotations
from typing import List, Tuple, Callable, Dict
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from numpy.polynomial import Chebyshev

def decreasing_logistic(x: np.ndarray, *, mid: float, L: float, k) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return L / (1.0 + np.exp(k * (x - mid)))

def invert_monotone(h, target, *, z_min: float, z_max: float, n: int = 512) -> float:
    """Invert a monotone function by grid search and linear interpolation."""
    z_grid = np.linspace(z_min, z_max, num=n)
    h_grid = h(z_grid)
    if h_grid[0] > h_grid[-1]:
        h_grid = h_grid[::-1]
        z_grid = z_grid[::-1]
    return float(np.interp(target, h_grid, z_grid))

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
    Zmax_true: float,
    *,
    anchor_weight: float = 1.0,
    c_init: np.ndarray | None = None,
    h_model: str = "poly",
    poly_degree: int = 4,
    max_iter: int = 20,
    tol: float = 1e-6,
    return_diag: bool = False,
) -> Dict[str, object]:
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

    Z_anchor = Zmax_true * np.array([q for (q, _) in anchor_points], dtype=float)
    X_anchor = np.array([x for (_, x) in anchor_points], dtype=float)

    # initialize shifts
    if c_init is None:
        c = np.zeros(n_traj)
    else:
        c = np.asarray(c_init, dtype=float).ravel().copy()
        if c.shape[0] != n_traj:
            raise ValueError(f"c_init length {c.shape[0]} does not match samples {n_traj}")

    def fit_h_model(
        U_sorted: np.ndarray,
        X_sorted: np.ndarray,
        W_sorted: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        if h_model == "poly":
            # coeffs = np.polyfit(U_sorted, X_sorted, deg=poly_degree, w=W_sorted)

            # def h(u: np.ndarray) -> np.ndarray:
            #     u = np.asarray(u, dtype=float)
            #     return np.polyval(coeffs, u)
            ch = Chebyshev.fit(U_sorted, X_sorted, deg=poly_degree, w=W_sorted)
            return lambda u: ch(u)
            return h

        raise ValueError(f"Unknown h_model '{h_model}'")

    def fit_h(anchor_only=False) -> Callable[[np.ndarray], np.ndarray]:
        """
        Fit monotone decreasing h given current shifts c_s.
        """
        U_all = []
        X_all = []
        W_all = []

        if not anchor_only:
            for s in range(n_traj):
                U_all.append(S_list[s] + c[s])
                X_all.append(X_list[s])
                W_all.append(np.ones_like(X_list[s]))

        U_all.append(Z_anchor)
        X_all.append(X_anchor)
        W_all.append(np.full_like(X_anchor, anchor_weight))

        U_all = np.concatenate(U_all)
        X_all = np.concatenate(X_all)
        W_all = np.concatenate(W_all)

        order = np.argsort(U_all)
        U_sorted = U_all[order]
        W_sorted = W_all[order]
        X_sorted = X_all[order]
        return fit_h_model(U_sorted, X_sorted, W_sorted)

    def update_shift(s: int, h: Callable[[np.ndarray], np.ndarray], Z_max) -> tuple[float, bool]:
        """
        Update shift c_s via 1D minimization.
        """

        S = S_list[s]
        X = X_list[s]

        def objective(cs: float) -> float:
            resid = X - h(S + cs)
            val = np.sum(resid ** 2)
            return val

        result = minimize_scalar(
            objective,
            method="bounded",
            bounds=(-min(S), Z_max - max(S))
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
    bound_total = 0

    h = fit_h(True)
    errors.append(compute_error(h))

    # ---- coordinate descent ----
    for _ in range(max_iter):

        h = fit_h()

        c_old = c.copy()

        e_bsh = compute_error(h)
        for s in range(n_traj):
            e_bsh_i = compute_error(h)
            c_s = update_shift(s, h, Zmax_true)
            # c_s = max(-samples[s][0][0], c_s)
            # c_s = min(c_s, Zmax_true * 2)
            c_s_prev = c[s]
            c[s] = c_s
            e_ssh = compute_error(h)
            # debug = True
            # if debug and e_ssh > e_bsh_i + 0.0001 * abs(e_bsh_i):
            #     print(f"\t{s} moving to error: {e_bsh_i}->{e_ssh}")   
            #     plt.scatter(Z_anchor, X_anchor, color='grey')
            #     z_h = np.linspace(0, Zmax_true, 100)
            #     plt.plot(z_h, h(z_h), color='black')

            #     print(f"Something very bad has happened, shift optimisation failed for {s}")
            #     Ssi, Xsi = samples[s]
            #     plt.plot(Ssi + c_s, Xsi, linestyle="--", color='red')   
            #     plt.plot(Ssi + c_s_prev, Xsi, linestyle="--", color='blue') 

            #     plt.show()
            c[s] = c_s

        errors.append(compute_error(h))
        if np.max(np.abs(c - c_old)) < tol:
            break
    print(len(errors))
    print(f"final error: {errors[-1]}")
    return c, h, errors

def bootstrap_inference(n_boot, nS, samps, anchors, anchor_weight, max_iter, start_zmax):
    hests = []
    errors_list =[]
    for _ in range(n_boot):
        idx = np.random.randint(0, nS, size=nS)
        boot_samps = [samps[i] for i in idx]
        # boot_anchors = [anchors[bi] for bi in np.random.randint(0, len(anchors), size=len(anchors))]
        boot_anchors = anchors
        c0_boot = np.random.uniform(0.0, start_zmax, size=len(boot_samps))
        _, hest_boot, errors_boot = infer_response_func(
            boot_samps,
            boot_anchors,
            Zmax_true,
            c_init=c0_boot,
            anchor_weight=anchor_weight,
            max_iter=max_iter,
        )
        hests.append(hest_boot)
        errors_list.append(errors_boot)
    return hests, errors_list


if __name__ == "__main__":
    W = 10.0
    n_watering_events = 4
    sigma2 = 0.1
    n_boot = 50
    # Simulate sequences of dQs for each pot, they may not span the whole range Qmin, Qmax (plants have narrow viability ranges)
    nS = 30
    anchor_weight = 0.2
    anchor_sigma2 = 90.0
    n_anchors = 8
    Zmax_true = 100.0
    X_at_Zmax = 50
    X_at_Zmin = 200
    max_iter = 30
    response_func = lambda z: X_at_Zmax + decreasing_logistic(z, mid= 0.5 * Zmax_true, L=X_at_Zmin, k=0.1)

#    debug_z = np.linspace(0, Zmax_true)
#    plt.scatter(debug_z, [response_func(z) for z in debug_z])
#    plt.show()

    hests = []
    errors_list = []

    samps = []
    Z_real = []

    start_zmax = Zmax_true - W * n_watering_events
    for s in range(nS):
        Z0 = np.random.uniform(0.0, start_zmax)
        Z, X = sim_pot_watering_sequence(W, n_watering_events, Z0, sigma2, response_func)
        dZs = Z2dZ(Z)
        S = dZ2S(dZs)
        samps.append((S, X))
        Z_real.append(Z)

    anchor_q = np.linspace(0.0, 1.0, num=n_anchors)
    anchor_x = [
        max(0.0, response_func(Zmax_true * q) + np.random.normal(0.0, anchor_sigma2))
        for q in anchor_q
    ]
    plt.scatter(anchor_q, anchor_x)
    plt.show()

    c0 = np.random.uniform(0.0, start_zmax, size=len(samps))

    anchors = list(zip(anchor_q, anchor_x))
    cest, hest, errors = infer_response_func(
        samps,
        anchors,
        Zmax_true,
        c_init=c0,
        anchor_weight=anchor_weight,
        max_iter=100,
    )

    cest_anchor_only, hest_anchor_only, errors_anchor_only = infer_response_func(
        samps,
        anchors,
        Zmax_true,
        c_init=c0,
        anchor_weight=anchor_weight,
        max_iter=0,
    )

    hests, errors_list = bootstrap_inference(n_boot, nS, samps, anchors, anchor_weight, max_iter, start_zmax)

    fig, axes = plt.subplots(2, 3, sharex=False, figsize=(14, 8))
    ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()

    ax0.scatter(anchor_q, anchor_x, c="black", s=30)
    ax0.set_title(f"Anchor points ($\sigma^2 = {anchor_sigma2}$)")
    ax0.set_ylabel("X")
    ax0.set_xlabel("Q")

    for s in range(nS):
        S, X = samps[s]
        Zrs = Z_real[s]

        Zest0 = S + c0[s]
        ax2.plot(
            Zest0,
            X,
            c="#6b6b6b",
            linestyle="--",
            label="Unaligned sequences" if s == 0 else None,
        )
        ax2.plot(
            Zrs,
            X,
            c="#424161",
            label="Samples" if s == 0 else None,
        )

        Zest = S + cest[s]
        ax3.plot(
            Zest,
            X,
            c="#6b6b6b",
            linestyle="--",
            label="Aligned sequences" if s == 0 else None,
        )
        ax3.plot(
            Zrs,
            X,
            c="#313045",
            label="Samples" if s == 0 else None,
        )

    ax1.hist(c0, bins=20, color="#424161", alpha=1.0)
    ax1.set_title("$c_0$ histogram (uniform distribution)")
    ax1.set_xlabel("c0")
    ax1.set_ylabel("Count")
    ax2.set_title("Data alignment with randomly initialized $c_0$")
    ax2.set_ylabel("X")
    ax2.set_xlabel("Z")
    ax2.legend(frameon=False)

    ax3.set_title("Data alignment with estimated $\hat{c}$")
    ax3.set_ylabel("X")
    ax3.set_xlabel("Z")
    ax3.legend(frameon=False)

    z_grid = np.linspace(0.0, Zmax_true, num=200)
    ax4.plot(z_grid, response_func(z_grid), c="#4136a3", label="True $h$")
    ax4.plot(z_grid, hest(z_grid), c="#e6a532", label="Estimated $\hat{h}$")
    ax4.plot(z_grid, hest_anchor_only(z_grid), c="red", label="Estimated $\hat{h}$ (anchors only)")
    if len(hests) > 0:
        boot_preds = np.vstack([h(z_grid) for h in hests])
        lo = np.percentile(boot_preds, 2.5, axis=0)
        hi = np.percentile(boot_preds, 97.5, axis=0)
        ax4.fill_between(z_grid, lo, hi, color="#e6a532", alpha=0.2, label="Bootstrap 95% CI")
    ax4.set_title("Response curve")
    ax4.set_xlabel("Z")
    ax4.set_ylabel("X")
    ax4.legend(frameon=False)

    ax5.plot(range(1, len(errors) + 1), np.log(errors), c="#313045", alpha=0.8)
    ax5.set_title("Inference error (weighted SSE)")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Log Error")

    fig.suptitle(f"Example result with {n_boot} bootstraps ($W={W},\sigma^2={sigma2},h=1/(1 - exp(-x))$)", fontsize=16)
    plt.tight_layout()
    plt.savefig("simulation_example.png")
    plt.show()
