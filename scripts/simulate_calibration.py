from __future__ import annotations
from typing import List, Tuple, Callable, Dict
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from numpy.polynomial import Chebyshev

def decreasing_logistic(x: np.ndarray, *, mid: float, L: float, k) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return L / (1.0 + np.exp(k * (x - mid)))

def sim_pot_watering_sequence(W, n_watering_events, Z0, sigma2, response_func, Z_max):
    """Samples tuples (Z, X).
    Params:
        W: water quantity, fixed dZ
        n_watering_events: number of watering events to draw
        Z0: starting water content
        sigma2: noise in response
        response func: function mapping Z to X
        Zmax: maximum Z
    """
    dZ = np.ones(n_watering_events) * W
    Z = np.cumsum(dZ) + Z0
    Z = np.insert(Z, 0, Z0)
    X = np.array([response_func(z) for z in Z])
    X += np.random.normal(0, sigma2, len(Z))

    # bounds = (-min(S), (Z_max - max(S)) )

    hit_zmax = False

    for zi, z in enumerate(Z):
        if z >= Z_max:
            Z[zi:] = Z_max
            X[zi:] = response_func(Z_max) + np.random.normal(0, sigma2, len(Z) - zi)
            # bounds = ( (Z_max - W * zi) , (Z_max - W * (zi + 1)) ) # The end must be fixed at Zmax now 
            hit_zmax = True
            return Z[:zi + 1], X[:zi + 1], hit_zmax

    return Z, X, hit_zmax

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
    Zmax_0: float,
    Zmin,
    Zmax,
    X_at_Zmax,
    *,
    anchor_weight: float = 1.0,
    c_init: np.ndarray | None = None,
    h_model: str = "poly",
    poly_degree: int = 3,
    max_iter: int = 20,
    tol: float = 1e-6,
    return_diag: bool = False,
) -> Dict[str, object]:
    # ---- prepare data ----
    n_traj = len(samples)

    S_list = []
    X_list = []
    hit_zmax_list = []
    for S, X, hit_zmax in samples:
        S = np.asarray(S).ravel()
        X = np.asarray(X).ravel()
        if S.shape != X.shape:
            raise ValueError(f"Each (S, X) must have same shape, got {S.shape} {X.shape}")
        S_list.append(S)
        X_list.append(X)
        hit_zmax_list.append(hit_zmax)

    X_anchor = np.array([x for (_, x) in anchor_points], dtype=float)
    Zmax_est = Zmax_0
    Q_anchor = np.array([q for (q, _) in anchor_points], dtype=float)
    
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
                U_all.append((S_list[s] + c[s])/Zmax_est)
                X_all.append(X_list[s])
                W_all.append(np.ones_like(X_list[s]))

        U_all.append(Q_anchor)
        X_all.append(X_anchor)

        if anchor_only:
            W_all.append(np.ones_like(X_anchor))
        else:
            W_all.append(np.full_like(X_anchor, anchor_weight))

        X_all.append([X_at_Zmax])
        U_all.append([1.0])
        W_all.append([1.0])

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
        hit_zmax = hit_zmax_list[s]

        def objective(cs: float) -> float:
            resid = X - h((S + cs) / Z_max)
            val = np.sum(resid ** 2)
            return val

        bounds = (-min(S), Z_max - max(S))
        if hit_zmax:
            bounds = ( (Z_max - W * (len(S) + 1)) , (Z_max - W * (len(S))) ) # The end must be fixed at Zmax now 
        
        result = minimize_scalar(
            objective,
            method="bounded",
            bounds=bounds
        )
        return float(result.x)

    def update_Zmax(h, c, Zmin, Zmax):
        def objective(Z):
            if Z <= 0:
                return np.inf
            sse = 0.0
            for s in range(n_traj):
                U = (S_list[s] + c[s]) / Z
                resid = X_list[s] - h(U)
                sse += np.sum(resid**2)
            return float(sse)

        result = minimize_scalar(
            objective,
            method="bounded",
            bounds=(Zmin, Zmax),
        )
        return float(result.x)


    def compute_error(h: Callable[[np.ndarray], np.ndarray]) -> float:
        sse = 0.0
        for s in range(n_traj):
            resid = X_list[s] - h((S_list[s] + c[s])/Zmax_est)
            sse += float(np.sum(resid ** 2))
        if len(Q_anchor) > 0:
            resid = X_anchor - h(Q_anchor)
            sse += float(np.sum(anchor_weight * (resid ** 2)))
        return sse


    h = fit_h(True)
    errors = []
    errors.append(compute_error(h))

    for s in range(n_traj):
        hit_z = hit_zmax_list[s]
        if hit_z == True:
            e_bsh_i = compute_error(h)
            c_s = update_shift(s, h, Zmax_est)
            # c_s = max(-samples[s][0][0], c_s)
            # c_s = min(c_s, Zmax_true * 2)
            c_s_prev = c[s]
            c[s] = c_s


    bound_total = 0

    # h = fit_h(True)
    errors.append(compute_error(h))

    # ---- coordinate descent ----
    for _ in range(max_iter):

        h = fit_h()

        c_old = c.copy()

        e_bsh = compute_error(h)
        for s in range(n_traj):
            e_bsh_i = compute_error(h)
            c_s = update_shift(s, h, Zmax_est)
            # c_s = max(-samples[s][0][0], c_s)
            # c_s = min(c_s, Zmax_true * 2)
            c_s_prev = c[s]
            c[s] = c_s
            e_ssh = compute_error(h)
            debug = False
            # if debug and e_ssh > e_bsh_i + 0.0001 * abs(e_bsh_i):
            if debug:
                print(f"\t{s} moving to error: {e_bsh_i}->{e_ssh}")   
                plt.scatter(Q_anchor, X_anchor, color='grey')
                q_h = np.linspace(0, 1.0, 100)
                plt.plot(q_h, h(q_h), color='black')

                print(f"Something very bad has happened, shift optimisation failed for {s}")
                Ssi, Xsi, hit_zmax = samples[s]
                plt.plot((Ssi + c_s) /Zmax_est, Xsi, linestyle="--", color='red')   
                plt.plot((Ssi + c_s_prev) /Zmax_est, Xsi, linestyle="--", color='blue') 

            c[s] = c_s
        if debug: plt.show()
        print(Zmax_est)
        # Zmax_est = update_Zmax(h, c, Zmin, Zmax)
        errors.append(compute_error(h))
        if np.max(np.abs(c - c_old)) < tol:
            break
    print(len(errors))
    print(f"final error: {errors[-1]}")
    return c, h, errors

def bootstrap_inference(n_boot, nS, samps, anchors, anchor_weight, max_iter, start_zmax, X_at_Zmax):
    hests = []
    errors_list =[]
    for _ in range(n_boot):
        idx = np.random.randint(0, nS, size=nS)
        boot_samps = [samps[i] for i in idx]
        # boot_anchors = [anchors[bi] for bi in np.random.randint(0, len(anchors), size=len(anchors))]
        boot_anchors = anchors
        c0_boot = np.random.uniform(0.0, 1.0, size=len(boot_samps))
        _, hest_boot, errors_boot = infer_response_func(
            boot_samps,
            boot_anchors,
            Zmax_true,
            Zmax_true / 2,
            Zmax_true * 2,
            X_at_Zmax,
            c_init=c0_boot,
            anchor_weight=anchor_weight,
            max_iter=max_iter
        )
        hests.append(hest_boot)
        errors_list.append(errors_boot)
    return hests, errors_list


if __name__ == "__main__":
    import sys
    W = 10.0
    n_watering_events = 4
    sigma2 = 0.1
    n_boot = int(sys.argv[1])
    # Simulate sequences of dQs for each pot, they may not span the whole range Qmin, Qmax (plants have narrow viability ranges)
    nS = 30
    anchor_weight = 0.05
    anchor_sigma2 = 90.0
    n_anchors = 8
    Zmax_true = 100.0
    X_at_Zmax = 50
    X_at_Zmin = 200
    max_iter = 20

    response_func_z = lambda z: X_at_Zmax + decreasing_logistic(z, mid= 0.5 * Zmax_true, L=X_at_Zmin, k=0.1)
    response_func_q = lambda q: X_at_Zmax + decreasing_logistic(q, mid= 0.5, L=X_at_Zmin, k=0.1 * Zmax_true)


    # debug_z = np.linspace(0, 1.0)
    # plt.scatter(debug_z, [response_func_q(z) for z in debug_z])
    # plt.show()

    # debug_z = np.linspace(0, Zmax_true)
    # plt.scatter(debug_z, [response_func_z(z) for z in debug_z])
    # plt.show()


    hests = []
    errors_list = []

    samps = []
    Z_real = []

    start_zmax = Zmax_true
    for s in range(nS):
        Z0 = np.random.uniform(0.0, Zmax_true)
        Z, X, hit_zmax = sim_pot_watering_sequence(W, n_watering_events, Z0, sigma2, response_func_z, Zmax_true)
        dZs = Z2dZ(Z)
        S = dZ2S(dZs)
        samps.append((S, X, hit_zmax))
        Z_real.append(Z)

    anchor_q = np.linspace(0.0, 1.0, num=n_anchors)
    anchor_x = [
        max(0.0, response_func_q(q) + np.random.normal(0.0, anchor_sigma2))
        for q in anchor_q
    ]
    plt.scatter(anchor_q, anchor_x)
    plt.show()

    c0 = np.random.uniform(0.0, 1.0, size=len(samps))

    anchors = list(zip(anchor_q, anchor_x))
    cest, hest, errors = infer_response_func(
        samps,
        anchors,
        Zmax_true,
        Zmax_true / 2,
        Zmax_true * 2,
        X_at_Zmax,
        c_init=c0,
        anchor_weight=anchor_weight,
        max_iter=max_iter
    )

    cest_anchor_only, hest_anchor_only, errors_anchor_only = infer_response_func(
        samps,
        anchors,
        Zmax_true,
        Zmax_true / 2,
        Zmax_true * 2,
        X_at_Zmax,
        c_init=c0,
        anchor_weight=anchor_weight,
        max_iter=0
    )

    hests, errors_list = bootstrap_inference(n_boot, nS, samps, anchors, anchor_weight, max_iter, start_zmax, X_at_Zmax)

    fig, axes = plt.subplots(2, 3, sharex=False, figsize=(14, 8))
    ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()

    ax0.scatter(anchor_q, anchor_x, c="black", s=30)
    ax0.set_title(f"Anchor points ($\sigma^2 = {anchor_sigma2}$)")
    ax0.set_ylabel("X")
    ax0.set_xlabel("Q")

    for s in range(nS):
        S, X, hit_zmax = samps[s]
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

    q_grid = np.linspace(0.0, 1.0, num=200)
    ax4.plot(q_grid, response_func_q(q_grid), c="#4136a3", label="True $h$")
    ax4.plot(q_grid, hest(q_grid), c="#e6a532", label="Estimated $\hat{h}$")
    ax4.plot(q_grid, hest_anchor_only(q_grid), c="red", label="Estimated $\hat{h}$ (anchors only)")
    if len(hests) > 0:
        boot_preds = np.vstack([h(q_grid) for h in hests])
        lo = np.percentile(boot_preds, 2.5, axis=0)
        hi = np.percentile(boot_preds, 97.5, axis=0)
        ax4.fill_between(q_grid, lo, hi, color="#e6a532", alpha=0.2, label="Bootstrap 95% CI")
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
