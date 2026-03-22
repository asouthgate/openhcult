from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.interpolate import LSQUnivariateSpline, BSpline, interp1d
from scipy.optimize import minimize


def fit_monotonic_spline(x, y, inner_knots, k=3):
    """Regress y against x with specified inner_knots and spline order k."""
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    x_min, x_max = xs.min(), xs.max()

    tmp_spline = LSQUnivariateSpline(xs, ys, inner_knots, k=k)
    t = tmp_spline.get_knots()
    c0 = tmp_spline.get_coeffs()

    def objective(coeffs):
        spl = BSpline(t, coeffs, k)
        return np.sum((spl(x) - y) ** 2)

    x_check = np.linspace(x_min, x_max, 50)

    def monotonic_constraint(coeffs):
        spl = BSpline(t, coeffs, k)
        return -spl(x_check, nu=1)

    res = minimize(
        objective, c0, constraints={"type": "ineq", "fun": monotonic_constraint}
    )
    if not res.success:
        raise RuntimeError(f"Spline optimization failed: {res.message}")

    return BSpline(t, res.x, k)


def _arclen_reparameterise(sx, sz, x_anchors, z_anchors, x_der, n_fine=500):
    s_fine = np.linspace(0, 1, n_fine)
    x_fine = sx(s_fine)
    z_fine = sz(s_fine)

    # Normalise axes so neither dominates arc-length (x and z are on different scales)
    x_range = np.ptp(x_fine) or 1.0
    z_range = np.ptp(z_fine) or 1.0
    xn = x_fine / x_range
    zn = z_fine / z_range

    seg_len = np.sqrt(np.diff(xn) ** 2 + np.diff(zn) ** 2)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    arc /= arc[-1]

    s_anc = np.array(
        [
            arc[np.argmin((xn - xa / x_range) ** 2 + (zn - za / z_range) ** 2)]
            for xa, za in zip(x_anchors, z_anchors)
        ]
    )

    s_der = np.array([arc[np.argmin((xn - xd / x_range) ** 2)] for xd in x_der])

    return s_anc, s_der


def fit_parametric_monotonic_spline(
    x_anchors, z_anchors, x_der, dz_dx, knots=10, k=3, w_der=1.0, n_iter=4
):
    """Compute a parametric monotonic spline given function and derivative data.

    Params:
        x_anchors: x values for anchors
        z_anchors: z(x) values for anchors
        x_der: x values at which derivative points are computed
        dz_dx: derivative of z wrt x at x_der (must be d(SWC)/d(Value))
        knots: number of spline knots
        k: spline order
        w_der: weighting for derivative error term
        n_iter: arc-length reparameterisation iterations
    """
    inner = np.linspace(0, 1, knots + 2)[1:-1]
    t = np.concatenate(([0.0] * (k + 1), inner, [1.0] * (k + 1)))
    n_c = len(t) - k - 1

    x_min, x_max = min(x_anchors.min(), x_der.min()), max(x_anchors.max(), x_der.max())
    s_anc = (x_anchors - x_min) / (x_max - x_min)
    s_der = (x_der - x_min) / (x_max - x_min)

    # Bootstrap initial curve directly from anchor data so the optimizer
    # starts close to a feasible solution and doesn't collapse to a trivial flat curve.
    sort_idx = np.argsort(x_anchors)
    x_sorted, z_sorted = x_anchors[sort_idx], z_anchors[sort_idx]
    s_init = np.linspace(0, 1, n_c)
    n_anc = len(x_sorted)
    c0 = np.concatenate(
        [
            np.interp(s_init, np.linspace(0, 1, n_anc), x_sorted),
            np.interp(s_init, np.linspace(0, 1, n_anc), z_sorted),
        ]
    )

    # Infer monotone direction from anchors sorted by x
    z_dir = np.sign(z_sorted[-1] - z_sorted[0])  # +1 increasing, -1 decreasing

    for _ in range(n_iter):

        def objective(coeffs, s_anc=s_anc, s_der=s_der):
            cx, cz = coeffs[:n_c], coeffs[n_c:]
            sx, sz = BSpline(t, cx, k), BSpline(t, cz, k)
            # Chain Rule: dz/ds = dz/dx * dx/ds
            err_anc = np.sum((sx(s_anc) - x_anchors) ** 2) + np.sum(
                (sz(s_anc) - z_anchors) ** 2
            )
            norm = np.sqrt(1.0 + dz_dx**2)
            err_der = np.sum(((sz(s_der, nu=1) - dz_dx * sx(s_der, nu=1)) / norm) ** 2)
            return err_anc + w_der * err_der

        def monotonic_con(coeffs):
            cz = coeffs[n_c:]
            return z_dir * BSpline(t, cz, k)(np.linspace(0, 1, 50), nu=1)

        res = minimize(
            objective, c0, constraints={"type": "ineq", "fun": monotonic_con}
        )
        if not res.success:
            raise RuntimeError(f"Spline optimization failed: {res.message}")

        c0 = res.x
        s_anc, s_der = _arclen_reparameterise(
            BSpline(t, res.x[:n_c], k),
            BSpline(t, res.x[n_c:], k),
            x_anchors,
            z_anchors,
            x_der,
        )

    return BSpline(t, res.x[:n_c], k), BSpline(t, res.x[n_c:], k)


def bootstrap_parametric_spline(
    x_anchors, z_anchors, x_der, dz_dx, knots=10, k=2, w_der=0.5, n_boots=50
):
    """
    Performs bootstrapping on the derivative data to produce a distribution of splines.
    """
    boot_results = []
    n_der = len(x_der)

    print(f"Starting bootstrap ({n_boots} iterations)...")

    for i in range(n_boots):
        print(f"Bootstrap iteration {i+1}/{n_boots}")
        indices = np.random.choice(n_der, size=n_der, replace=True)

        x_resampled = x_der[indices]
        dz_dx_resampled = dz_dx[indices]

        try:
            sx, sz = fit_parametric_monotonic_spline(
                x_anchors,
                z_anchors,
                x_resampled,
                dz_dx_resampled,
                knots=knots,
                k=k,
                w_der=w_der,
            )
            boot_results.append((sx, sz))
        except Exception as e:
            print(f"Iteration {i} failed: {e}")
            continue

    return boot_results


def bootstrap_monotonic_spline(x, y, inner_knots, k=3, n_boots=50):
    boot_splines = []
    n = len(x)

    print(f"Starting bootstrap ({n_boots} iterations)...")

    for i in range(n_boots):
        print(f"Bootstrap iteration {i+1}/{n_boots}")
        indices = np.random.choice(n, size=n, replace=True)
        x_resampled = x[indices]
        y_resampled = y[indices]

        try:
            inner_knots_r = np.linspace(
                x_resampled.min(), x_resampled.max(), len(inner_knots) + 2
            )[1:-1]
            spline = fit_monotonic_spline(x_resampled, y_resampled, inner_knots_r, k=k)
            boot_splines.append(spline)
        except Exception as e:
            print(f"Iteration {i} failed: {e}")
            continue

    return boot_splines


def compute_lookup_table_from_bootstrap(boot_splines, x_min, x_max, n_points=100):
    x_grid = np.linspace(x_min, x_max, n_points)
    z_grid = np.array([spline(x_grid) for spline in boot_splines])

    z_mean = np.mean(z_grid, axis=0)
    z_lower = np.percentile(z_grid, 2.5, axis=0)
    z_upper = np.percentile(z_grid, 97.5, axis=0)
    z_std = np.std(z_grid, axis=0)

    lookup_df = pd.DataFrame(
        {
            "x": x_grid,
            "swc": z_mean,
            "swc_std": z_std,
            "swc_upper_95%": z_upper,
            "swc_lower_95%": z_lower,
        }
    )

    return lookup_df


def compute_lookup_table_parametric_forward(
    boot_mod_splines, s_min=0, s_max=1, n_points=1000
):
    s_grid = np.linspace(s_min, s_max, n_points)

    x_samples = []
    z_samples = []

    for sx, sz in boot_mod_splines:
        x_samples.append(sx(s_grid))
        z_samples.append(sz(s_grid))

    x_samples = np.array(x_samples)
    z_samples = np.array(z_samples)

    x_mean = np.mean(x_samples, axis=0)
    z_mean = np.mean(z_samples, axis=0)
    z_std = np.std(z_samples, axis=0)

    lookup_df = pd.DataFrame(
        {"s": s_grid, "sensor_val": x_mean, "swc": z_mean, "swc_std": z_std}
    )

    return lookup_df


def compute_mrt(times, vals):
    """
    Computes the Mean Residence Time (MRT) of the sensor transition.
    Handles datetime64/timedelta64 and is direction-agnostic.

    Returns:
        float: MRT in seconds (the 'delta T' of the transition).
    """
    if np.issubdtype(times.dtype, np.datetime64) or np.issubdtype(
        times.dtype, np.timedelta64
    ):
        t_numeric = (times - times[0]) / np.timedelta64(1, "s")
    else:
        t_numeric = times - times[0]

    # Extract boundaries with median filtering for noise robustness
    v_start = np.median(vals[:5])
    v_end = np.median(vals[-5:])

    # If the sensor didn't move, MRT is undefined/zero
    if np.isclose(v_start, v_end, atol=1e-7):
        return vals, 0.0

    # Normalise to [0,1], flipping so both wetting and drying curves rise from 0 to 1
    v_norm = np.clip((vals - v_start) / (v_end - v_start), 0, 1)

    # MRT = Integral from 0 to T of (1 - v_norm) dt
    auc = np.trapezoid(v_norm, t_numeric)

    total_duration = t_numeric[-1]
    mrt = total_duration - auc

    return vals, mrt


def fit_parametric_spline_with_residuals(
    x, y, xmid, dydx, n_inner_knots, k_spline, w_der, n_boots
):
    """Compute a parametric spline using both x,y data as well as xmid (x_i+1/2) and derivative at midpoint data.

    Params:
        x, y, xmid, ydx: arrays
        n_inner_knots: int spline knots
        k_spline: int order of spline
        w_der: weight of derivative samples
        n_boots: number of bootstrap samples
    """

    spline_mod_x, spline_mod_y = fit_parametric_monotonic_spline(
        x, y, xmid, dydx, n_inner_knots, k_spline, w_der
    )

    boot_mod_splines = bootstrap_parametric_spline(
        x, y, xmid, dydx, n_inner_knots, k_spline, w_der, n_boots=n_boots
    )

    # After fitting the parametric splines, we now have to invert from x -> s so that we can calculate y(x)
    s_fine = np.linspace(0, 1, 1000)
    x_fine = spline_mod_x(s_fine)
    x_to_s_map = interp1d(x_fine, s_fine, bounds_error=False, fill_value="extrapolate")
    s_data = x_to_s_map(x)
    y_pred = spline_mod_y(s_data)
    abs_residuals = np.abs(y - y_pred)

    inner_knots = np.linspace(x.min(), x.max(), n_inner_knots + 2)[1:-1]

    residual_spline_mod = fit_monotonic_spline(
        x,
        abs_residuals,
        inner_knots=inner_knots,
        k=k_spline,
    )
    return spline_mod_x, spline_mod_y, boot_mod_splines, residual_spline_mod
