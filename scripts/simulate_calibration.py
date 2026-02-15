from __future__ import annotations
from typing import List, Tuple, Callable, Dict
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from numpy.polynomial import Chebyshev
from scipy.interpolate import BSpline
from scipy.linalg import solve
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator, UnivariateSpline

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

def cal_hit_bounds(S, Z_max):
    w = S[-1] - S[-2]
    return ( max(0, (Z_max - w * len(S))), max(0.0, min(Z_max, (Z_max - w * (len(S) - 1)))) )

def cal_hit_bounds_01(S, Z_max):
    w = S[-1] - S[-2]
    lower = 1.0 - len(S) * (w / Z_max)
    upper = 1.0 - (len(S) - 1) * (w / Z_max)
    return ( max(0, lower), min(1.0, upper) )

def infer_response_func(
    samples: List[Tuple[np.ndarray, np.ndarray]],
    anchor_points: List[Tuple[float, float]],
    Zmax_0: float,
    Zmin,
    Zmax,
    X_at_Zmax,
    X_at_Zmin,
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
    
    Smax = max(np.max(S) for (S, _, _) in samples)
    if Zmin < Smax:
        print(f"Warning: Zmin {Zmin} is less than max S {Smax}, which is a bound on the min, setting Zmin to {Smax}")
        Zmin = Smax

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

    def fit_h_model(U_sorted, X_sorted, W_sorted):
        U = np.asarray(U_sorted)
        X = np.asarray(X_sorted)
        W = np.asarray(W_sorted)

        # knots
        n_knots = 8
        knots = np.linspace(0, 1, n_knots)
        degree = 3

        # augmented knot vector
        t = np.concatenate((
            np.repeat(knots[0], degree),
            knots,
            np.repeat(knots[-1], degree)
        ))

        # build spline basis matrix
        n_basis = len(t) - degree - 1
        B = np.zeros((len(U), n_basis))
        for i in range(n_basis):
            coeff = np.zeros(n_basis)
            coeff[i] = 1
            spline = BSpline(t, coeff, degree)
            B[:, i] = spline(U)

        # weighted ridge regression
        W_sqrt = np.sqrt(W)
        B_w = B * W_sqrt[:, None]
        X_w = X * W_sqrt

        lam = 1e-4
        beta = solve(B_w.T @ B_w + lam*np.eye(n_basis),
                    B_w.T @ X_w)

        def h(u):
            u = np.asarray(u)
            B_u = np.zeros((len(u), n_basis))
            for i in range(n_basis):
                coeff = np.zeros(n_basis)
                coeff[i] = 1
                spline = BSpline(t, coeff, degree)
                B_u[:, i] = spline(u)
            return B_u @ beta

    #     plt.scatter(U_sorted, X_sorted, color='grey')
    #     plt.plot(np.linspace(0, 1.0, 100), ch(np.linspace(0, 1.0, 100)), color='black')
    #     plt.show()


        return h

    # def fit_h_model(
    #     U_sorted: np.ndarray,
    #     X_sorted: np.ndarray,
    #     W_sorted: np.ndarray,
    # ) -> Callable[[np.ndarray], np.ndarray]:

        # if h_model == "poly":

        #     ch = Chebyshev.fit(U_sorted, X_sorted, deg=poly_degree, w=W_sorted, domain=[0.0, 1.0])

        #     plt.scatter(U_sorted, X_sorted, color='grey')
        #     plt.plot(np.linspace(0, 1.0, 100), ch(np.linspace(0, 1.0, 100)), color='black')
        #     plt.show()

        #     return lambda u: ch(u)

        # raise ValueError(f"Unknown h_model '{h_model}'")

    def fit_h(anchor_only=False) -> Callable[[np.ndarray], np.ndarray]:
        """
        Fit monotone decreasing h given current shifts c_s.
        """
        U_all = []
        X_all = []
        W_all = []

        if not anchor_only:
            for s in range(n_traj):
                U_all.append((S_list[s]/Zmax_est + c[s]))
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
        X_all.append([X_at_Zmin])
        U_all.append([0.0])
        W_all.append([1.0])

        U_all = np.concatenate(U_all)
        # U_all = U_all / np.max(U_all)
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
            resid = X - h((S/Z_max + cs))
            val = np.sum(resid ** 2)
            return val

        bounds = (0.0, 1.0 - max(S)/Z_max)
        if hit_zmax:
            bounds = cal_hit_bounds_01(S, Z_max) # The end must be fixed at Zmax now 
        # print(bounds)
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
                U = (S_list[s]/Z + c[s])
                resid = X_list[s] - h(U)
                sse += np.sum(resid**2)
            return float(sse)

        result = minimize_scalar(
            objective,
            method="bounded",
            bounds=(Zmin, Zmax),
        )
        return float(result.x)


    def compute_error(h: Callable[[np.ndarray], np.ndarray], z) -> float:
        sse = 0.0
        for s in range(n_traj):
            resid = X_list[s] - h((S_list[s]/z + c[s]))
            sse += float(np.sum(resid ** 2))
        if len(Q_anchor) > 0:
            resid = X_anchor - h(Q_anchor)
            sse += float(np.sum(anchor_weight * (resid ** 2)))
        return sse


    h = fit_h(True)
    errors = []
    errors.append(compute_error(h, Zmax_est))

    for s in range(n_traj):
        hit_z = hit_zmax_list[s]
        if hit_z == True:
            e_bsh_i = compute_error(h, Zmax_est)
            c_s = update_shift(s, h, Zmax_est)
            # c_s = max(-samples[s][0][0], c_s)
            # c_s = min(c_s, Zmax_true * 2)
            c_s_prev = c[s]
            c[s] = c_s


    bound_total = 0

    # h = fit_h(True)
    errors.append(compute_error(h, Zmax_est))

    # ---- coordinate descent ----
    for j in range(max_iter):

        h_prev = h
        e_pre_h = compute_error(h, Zmax_est)
        h = fit_h()
        e_post_h = compute_error(h, Zmax_est)
        # if (e_post_h > e_pre_h):
            # h = h_prev
        if (e_post_h > e_pre_h + e_pre_h * 1e-12):
            print(f"Something very bad has happened at iteration {j}, h optimisation failed")


        c_old = c.copy()

        e_bsh = compute_error(h, Zmax_est)
        for s in range(n_traj):
            e_bsh_i = compute_error(h, Zmax_est)
            c_s = update_shift(s, h, Zmax_est)
            c_s_prev = c[s]
            c[s] = c_s
            e_ssh = compute_error(h, Zmax_est)
            debug = True
            # if debug and e_ssh > e_bsh_i + 0.0001 * abs(e_bsh_i):
            shift_opt_failed = e_ssh > e_bsh_i + 1e-8 * abs(e_bsh_i)
            if shift_opt_failed:
                print(f"THIS SHOULD ONLY EVER HAPPEN IF c_s_prev IS OUT OF BOUNDS, WE TAKE ERROR IMMEDIATELY BEFORE: shift optimisation failed for {s}, hit_zmax={hit_z}, {e_ssh} > {e_bsh_i}")
                print(f"\t c_s_prev: {c_s_prev}, c_s: {c_s}")
                if hit_z:
                    print(f"\tBounds for c: {cal_hit_bounds_01(S_list[s], Zmax_est)}")
            if debug and shift_opt_failed:
                # print(f"\t{s} moving to error: {e_bsh_i}->{e_ssh}")   
                plt.scatter(Q_anchor, X_anchor, color='grey')
                q_h = np.linspace(0, 1.0, 100)
                plt.plot(q_h, h(q_h), color='black')

                Ssi, Xsi, hit_zmax = samples[s]
                plt.plot((Ssi /Zmax_est + c_s), Xsi, linestyle="--", color='red')   
                plt.plot((Ssi /Zmax_est + c_s_prev), Xsi, linestyle="--", color='blue') 
                plt.show()
            # if e_ssh > e_bsh:  # can be numerical reasons for tiny tiny diff, ifts small enough no problem
            #     c[s] = c_s_prev
            # else:
            c[s] = c_s
        if debug: plt.show()
        print(Zmax_est)
        e_pre_zmove = compute_error(h, Zmax_est)
        # Zmax_est = update_Zmax(h, c, Zmin, Zmax)
        e_post_zmove = compute_error(h, Zmax_est)
        # if debug:
        if (e_post_zmove > e_pre_zmove):
            print(f"Something very bad has happened at iteration {j}, Zmax optimisation failed")

        errors.append(compute_error(h, Zmax_est))
        if (errors[-1] > errors[-2]):
            print(f"Something very bad has happened at iteration {j}, overall optimisation failed")

        
        if np.max(np.abs(c - c_old)) < tol:
            break
    print(len(errors))
    print(f"final error: {errors[-1]}")
    return c, h, Zmax_est, errors

def bootstrap_inference(n_boot, nS, samps, anchors, anchor_weight, max_iter, start_zmax, X_at_Zmax):
    hests = []
    errors_list =[]
    for _ in range(n_boot):
        idx = np.random.randint(0, nS, size=nS)
        boot_samps = [samps[i] for i in idx]
        # boot_anchors = [anchors[bi] for bi in np.random.randint(0, len(anchors), size=len(anchors))]
        boot_anchors = anchors
        c0_boot = np.random.uniform(0.0, 1.0, size=len(boot_samps))
        _, hest_boot, _, errors_boot = infer_response_func(
            boot_samps,
            boot_anchors,
            Zmax_true,
            Zmax_true / 2,
            Zmax_true * 2,
            X_at_Zmax,
            X_at_Zmin,
            c_init=c0_boot,
            anchor_weight=anchor_weight,
            max_iter=max_iter
        )
        hests.append(hest_boot)
        errors_list.append(errors_boot)
    return hests, errors_list

def plot_reconstructed_curve(samps):
    xs = []
    derivatives = []
    for s in range(len(samps)):
        S, X, dZs, hit_zmax = samps[s]
        for j in range(1, len(X) - 1):
            dXj = (X[j] - X[j-1])
            derivative = dZs[j] / dXj
            xmid = (X[j] + X[j-1]) / 2
            xs.append(xmid)
            derivatives.append(derivative)

    plt.scatter(xs, derivatives)
    plt.ylabel("dZ/dX")
    plt.xlabel("X")
    plt.show()
    x = np.asarray(xs, dtype=float)
    g = np.asarray(derivatives, dtype=float)

    # 1) Clean
    m = np.isfinite(x) & np.isfinite(g)
    x, g = x[m], g[m]

    # Optional: drop insane spikes (robust winsorization via MAD)
    med = np.median(g)
    mad = np.median(np.abs(g - med)) + 1e-12
    z = 0.6745 * (g - med) / mad
    g = np.clip(g, med - 6*mad, med + 6*mad)

    # 2) Sort
    idx = np.argsort(x)
    x, g = x[idx], g[idx]

    # 3) Collapse near-duplicates in x (robustly) by binning x very finely, taking median g per bin
    #    (this avoids oscillations / overweighting dense regions)
    nb = 1000  # raise/lower depending on how fine you want the "unique x" support
    edges = np.linspace(x.min(), x.max(), nb + 1)
    bin_id = np.digitize(x, edges) - 1
    good = (bin_id >= 0) & (bin_id < nb)
    x, g, bin_id = x[good], g[good], bin_id[good]

    x_u = np.empty(nb)
    g_u = np.empty(nb)
    x_u[:] = np.nan
    g_u[:] = np.nan

    for b in range(nb):
        sel = (bin_id == b)
        if np.any(sel):
            x_u[b] = np.nanmedian(x[sel])
            g_u[b] = np.nanmedian(g[sel])

    keep = np.isfinite(x_u) & np.isfinite(g_u)
    x_u, g_u = x_u[keep], g_u[keep]

    # 4) Interpolate g(x) across gaps
    #    PCHIP is shape-preserving and avoids overshoot; good default for derivatives.
    interp = PchipInterpolator(x_u, g_u, extrapolate=False)

    # Regular grid to integrate on
    xgrid = np.linspace(x_u.min(), x_u.max(), 2000)
    ggrid = interp(xgrid)

    # If there are big gaps, interp gives NaN there. We'll integrate only where we have values.
    valid = np.isfinite(ggrid)
    xv = xgrid[valid]
    gv = ggrid[valid]

    # Optional smoothing before integrating (helps if derivatives are noisy)
    # spline = UnivariateSpline(xv, gv, s=len(xv)*0.5)
    # gv = spline(xv)

    # 5) Integrate
    Z = cumulative_trapezoid(gv, xv, initial=0.0)  # Z(x) up to an additive constant

    Z += abs(min(Z))
    return xv, Z

def derivative_gp_simulation(x, dy_noisy):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel


    # -----------------------------
    # 3. GP regression on derivatives only
    # -----------------------------
    X_train = X.reshape(-1, 1)
    y_train = dy_noisy

    kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    gp.fit(X_train, y_train)

    # Dense grid for prediction
    X_test = np.linspace(0, max(x), 400).reshape(-1, 1)

    # Sample derivative functions from GP posterior
    n_samples = 100
    dy_samples = gp.sample_y(X_test, n_samples=n_samples)

    # -----------------------------
    # 4. Integrate samples
    # -----------------------------
    dx_test = X_test[1] - X_test[0]
    f_samples = np.cumsum(dy_samples, axis=0) * dx_test

    # Anchor each sample at first true value
    f_samples += y_true[0] - f_samples[0, :]

    # Compute mean and std of reconstructed function
    f_mean = np.mean(f_samples, axis=1)
    f_std = np.std(f_samples, axis=1)

    # -----------------------------
    # 5. Plot (single plot only)
    # -----------------------------
    plt.figure()
    plt.plot(X, y_true)
    plt.plot(X_test.flatten(), f_mean)
    plt.fill_between(
        X_test.flatten(),
        f_mean - 2 * f_std,
        f_mean + 2 * f_std,
        alpha=0.3
    )
    plt.xlabel("X")
    plt.ylabel("f(X)")
    plt.title("Function Recovery from Derivative-Only GP (with Uncertainty)")
    plt.show()



if __name__ == "__main__":
    import sys
    W = 10.0
    n_watering_events = 8
    sigma2 = 0.0
    n_boot = int(sys.argv[1])
    # Simulate sequences of dQs for each pot, they may not span the whole range Qmin, Qmax (plants have narrow viability ranges)
    nS = 30
    anchor_weight = 0.0
    anchor_sigma2 = 10.0
    n_anchors = 8
    Zmax_true = 150.0
    X_at_Zmax = 50
    X_at_Zmin = 200
    max_iter = 10

    response_func_z = lambda z: X_at_Zmax + decreasing_logistic(z, mid= 0.5 * Zmax_true, L=X_at_Zmin, k=0.05)
    response_func_q = lambda q: decreasing_logistic(q, mid= 0.5, L=1.0, k=0.1 * Zmax_true)


    # -----------------------------
    # 1. Simulate true function
    # -----------------------------
    np.random.seed(0)
    n = 100
    X = np.linspace(0, 1, n)

    # def f(x):
    #     return np.sin(3.0 * x) + 0.3 * x

    y_true = response_func_q(X)

    plt.plot(X, y_true, label="True function")
    plt.show()

    # -----------------------------
    # 2. Finite difference derivative
    # -----------------------------
    dx = X[1] - X[0]
    dy = np.gradient(y_true, dx)

    # Add noise to derivative observations
    noise_std = 0.5
    dy_noisy = dy + np.random.normal(0, noise_std, size=n)


    derivative_gp_simulation(X, dy_noisy)

    debug_q = np.linspace(0, 1.0)
    plt.scatter(debug_q, [response_func_q(q) for q in debug_q])
    plt.show()



    hests = []
    errors_list = []

    samps = []
    Z_real = []


    start_zmax = Zmax_true
    for s in range(nS):
        Z0 = np.random.uniform(0.0, Zmax_true)
        Z, X, hit_zmax = sim_pot_watering_sequence(W, n_watering_events, Z0, sigma2, response_func_z, Zmax_true)
        Q = Z / Zmax_true
        dQs = Z2dZ(Q)
        dZs = Z2dZ(Z)
        S = dZ2S(dZs)
        samps.append((S, X, dZs, dQs, hit_zmax))
        Z_real.append(Z)


    xs = []
    derivatives = []
    for s in range(len(samps)):
        S, X, dZs, dQs, hit_zmax = samps[s]
        for j in range(1, len(X) - 1):
            dXj = (X[j] - X[j-1])
            derivative = dQs[j] / dXj
            xmid = (X[j] + X[j-1]) / 2
            xs.append(xmid)
            derivatives.append(derivative)
    
    xs = np.array(xs)
    derivatives = np.array(derivatives)

    xmin, xmax = xs.min(), xs.max()
    xs = (xs - xmin) / (xmax - xmin)
    # Anchor x0 also scaled
    xs_01 = (X_at_Zmax - xmin) / (xmax - xmin)

    print(len(xs), len(derivatives))
    plt.scatter(xs, derivatives)
    plt.show()

    print(f"max derivative: {max(derivatives)}, min derivative: {min(derivatives)}")

    gp, mean, std, x_test = gp_fit_Z_from_derivatives(xs, derivatives)

    # xv, Z = plot_reconstructed_curve(samps)

    # debug_z = np.linspace(0, Zmax_true)
    # plt.scatter(debug_z, [response_func_z(z) for z in debug_z])
    # plt.plot(Z, xv)
    # plt.show()



    anchor_q = np.linspace(0.0, 1.0, num=n_anchors)
    anchor_x = [
        max(0.0, response_func_q(q) + np.random.normal(0.0, anchor_sigma2))
        for q in anchor_q
    ]
    plt.scatter(anchor_q, anchor_x)
    plt.show()

    c0 = np.random.uniform(0.0, 1.0, size=len(samps))

    anchors = list(zip(anchor_q, anchor_x))
    cest, hest, zest, errors = infer_response_func(
        samps,
        anchors,
        Zmax_true,
        Zmax_true / 2,
        Zmax_true * 2,
        X_at_Zmax,
        X_at_Zmin,
        c_init=c0,
        anchor_weight=anchor_weight,
        max_iter=max_iter
    )

    _, hest_anchor_only, _, errors_anchor_only = infer_response_func(
        samps,
        anchors,
        Zmax_true,
        Zmax_true / 2,
        Zmax_true * 2,
        X_at_Zmax,
        X_at_Zmin,
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

        Zest0 = S + c0[s] * Zmax_true
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

        Zest = S + cest[s] * zest
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
