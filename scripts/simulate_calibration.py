from __future__ import annotations
from typing import List, Tuple, Callable, Dict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar
from scipy.interpolate import UnivariateSpline

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
    *,
    anchor_weight: float = 2.0,
    shift_ridge: float = 0.0,
    shift_prior: float = 0.5,
    c_init: np.ndarray | None = None,
    h_model: str = "isotonic",
    spline_s: float | None = None,
    max_iter: int = 20,
    tol: float = 1e-6,
    return_diag: bool = False,
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
    h_model : "isotonic" or "spline" for looser fits
    spline_s : smoothing factor for spline fits
    max_iter : max coordinate descent iterations
    tol : convergence tolerance on shifts
    return_diag : when True, return diagnostics

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

    def fit_h_model(
        U_sorted: np.ndarray,
        X_sorted: np.ndarray,
        W_sorted: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        if h_model == "isotonic":
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(U_sorted, -X_sorted, sample_weight=W_sorted)

            def h(u: np.ndarray) -> np.ndarray:
                u = np.asarray(u, dtype=float)
                return -iso.predict(u)

            return h

        if h_model == "spline":
            uniq_u, inv = np.unique(U_sorted, return_inverse=True)
            if uniq_u.size != U_sorted.size:
                sum_w = np.zeros(uniq_u.size)
                sum_xw = np.zeros(uniq_u.size)
                for i, idx in enumerate(inv):
                    sum_w[idx] += W_sorted[i]
                    sum_xw[idx] += W_sorted[i] * X_sorted[i]
                X_use = sum_xw / sum_w
                W_use = sum_w
                U_use = uniq_u
            else:
                U_use = U_sorted
                X_use = X_sorted
                W_use = W_sorted

            spline = UnivariateSpline(U_use, X_use, w=W_use, s=spline_s)

            def h(u: np.ndarray) -> np.ndarray:
                u = np.asarray(u, dtype=float)
                return spline(u)

            return h

        raise ValueError(f"Unknown h_model '{h_model}'")

    def fit_h_from_anchors() -> Callable[[np.ndarray], np.ndarray] | None:
        """
        Fit a monotone decreasing h using anchor points only.
        """
        if len(Z_anchor) == 0:
            return None

        order = np.argsort(Z_anchor)
        Z_sorted = Z_anchor[order]
        X_sorted = X_anchor[order]
        W_sorted = np.full_like(X_sorted, anchor_weight)

        return fit_h_model(Z_sorted, X_sorted, W_sorted)

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
        W_sorted = W_all[order]
        X_sorted = X_all[order]
        return fit_h_model(U_sorted, X_sorted, W_sorted)

    def update_shift(s: int, h: Callable[[np.ndarray], np.ndarray]) -> tuple[float, bool]:
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

        lower = c[s] - span
        upper = c[s] + span
        result = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
        )
        bound_eps = 1e-6 * max(1.0, span)
        hit_bound = abs(result.x - lower) <= bound_eps or abs(result.x - upper) <= bound_eps
        return float(result.x), hit_bound

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
    bound_hits = 0
    bound_total = 0

    # ---- initialize shifts from anchor-only h, if available ----
    if c_init is None:
        h_anchor = fit_h_from_anchors()
        if h_anchor is not None:
            for s in range(n_traj):
                c_s, hit_bound = update_shift(s, h_anchor)
                c[s] = c_s
                bound_hits += int(hit_bound)
                bound_total += 1

    # ---- coordinate descent ----
    for _ in range(max_iter):

        h = fit_h()

        c_old = c.copy()

        for s in range(n_traj):
            c_s, hit_bound = update_shift(s, h)
            c[s] = c_s
            bound_hits += int(hit_bound)
            bound_total += 1

        h = fit_h()
        errors.append(compute_error(h))

        if np.max(np.abs(c - c_old)) < tol:
            break

    if return_diag:
        diag = {"bound_hits": bound_hits, "bound_total": bound_total}
        return c, h, errors, diag
    return c, h, errors

if __name__ == "__main__":
    W = 10.0
    n_watering_events = 4
    sigma2 = 0.1
    n_boot = 1
    # Simulate sequences of dQs for each pot, they may not span the whole range Qmin, Qmax (plants have narrow viability ranges)
    nS = 30
    anchor_weight = 0.2
    anchor_sigma2 = 0.0
    Zmax_true = 100.0
    XofZmax = 50

    response_func = lambda z: decreasing_logistic(z, mid= 0.5 * Zmax_true, L=XofZmax, k=0.1)

#    debug_z = np.linspace(0, Zmax_true)
#    plt.scatter(debug_z, [response_func(z) for z in debug_z])
#    plt.show()

    hests = []
    errors_list = []

    samps = []
    Z_real = []

    for s in range(nS):
        Z0 = np.random.uniform(0.0, Zmax_true - W * n_watering_events)
        Z, X = sim_pot_watering_sequence(W, n_watering_events, Z0, sigma2, response_func)
        dZs = Z2dZ(Z)
        S = dZ2S(dZs)
        samps.append((S, X))
        Z_real.append(Z)

    x_at_zmax = response_func(Zmax_true)
    anchor_q = np.linspace(0.1, 0.9, num=10)
    anchor_x = [
        max(0.0, response_func(Zmax_true * q) + np.random.normal(0.0, anchor_sigma2))
        for q in anchor_q
    ]
    plt.scatter(anchor_q, anchor_x)
    plt.show()

    c0 = np.random.uniform(0.0, 1.0, size=len(samps))
    zmax_est = 0.8 * Zmax_true
    for _ in range(5):
        anchors = [(zmax_est * q, x) for q, x in zip(anchor_q, anchor_x)]
        cest, hest, errors, diag = infer_response_func(
            samps,
            anchors,
            c_init=c0,
            anchor_weight=anchor_weight,
            max_iter=8,
            return_diag=True,
        )
        zmax_est = invert_monotone(hest, x_at_zmax, z_min=0.1 * Zmax_true, z_max=2.0 * Zmax_true)
    if diag["bound_total"] > 0:
        hit_rate = 100.0 * diag["bound_hits"] / diag["bound_total"]
        print(f"Shift bounds hit: {diag['bound_hits']} / {diag['bound_total']} ({hit_rate:.1f}%)")


    for _ in range(n_boot):
        idx = np.random.randint(0, nS, size=nS)
        boot_samps = [samps[i] for i in idx]
        c0_boot = np.random.uniform(0.0, 1.0, size=len(boot_samps))
        _, hest_boot, errors_boot = infer_response_func(
            boot_samps,
            anchors,
            c_init=c0_boot,
            anchor_weight=anchor_weight,
            max_iter=10,
        )
        hests.append(hest_boot)
        errors_list.append(errors_boot)

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
            label="Aligned sequences" if s == 0 else None,
        )

        Zest = S + cest[s]
        ax3.plot(
            Zest,
            X,
            c="#6b6b6b",
            linestyle="--",
            label="Unaligned sequences" if s == 0 else None,
        )
        ax3.plot(
            Zrs,
            X,
            c="#313045",
            label="Aligned sequences" if s == 0 else None,
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
    if len(hests) > 0:
        boot_preds = np.vstack([h(z_grid) for h in hests])
        lo = np.percentile(boot_preds, 2.5, axis=0)
        hi = np.percentile(boot_preds, 97.5, axis=0)
        ax4.fill_between(z_grid, lo, hi, color="#e6a532", alpha=0.2, label="Bootstrap 95% CI")
    ax4.set_title("Response curve")
    ax4.set_xlabel("Z")
    ax4.set_ylabel("X")
    ax4.legend(frameon=False)

    ax5.plot(range(1, len(errors) + 1), errors, c="#313045", alpha=0.8)
    ax5.set_title("Inference error (weighted SSE)")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Error")

    fig.suptitle(f"Example result with {n_boot} bootstraps ($W={W},\sigma^2={sigma2},h=1/(1 - exp(-x))$)", fontsize=16)
    plt.tight_layout()
    plt.savefig("simulation_example.png")
    plt.show()
