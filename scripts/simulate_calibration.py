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

def derivative_gp_simulation(x, dy_noisy, y_xmax):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel


    # -----------------------------
    # 3. GP regression on derivatives only
    # -----------------------------
    X_train = x.reshape(-1, 1)
    y_train = dy_noisy

    print(X_train.shape, len(y_train))

    kernel = C(1.0) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 1.0))
    kernel  += WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 2.0))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    gp.fit(X_train, y_train)

    # Dense grid for prediction
    X_test = np.linspace(min(x), max(x), 400).reshape(-1, 1)

    # Sample derivative functions from GP posterior
    n_samples = 100
    dy_samples = gp.sample_y(X_test, n_samples=n_samples)

    # -----------------------------
    # 4. Integrate samples
    # -----------------------------
    dx_test = X_test[1] - X_test[0]
    f_samples = np.cumsum(dy_samples, axis=0) * dx_test

    # # Anchor each sample at first true value
    print(y_xmax)
    f_samples += y_xmax - f_samples[0, :]

    # Compute mean and std of reconstructed function
    f_mean = np.mean(f_samples, axis=1)
    f_std = np.std(f_samples, axis=1)

    return X_test, dy_samples, f_mean, f_std



if __name__ == "__main__":
    import sys
    W = 10.0
    Zmax = 150.0
    Q_at_Xmin = 1.0
    sigma2 = 0.1
    sigma2_z = 0
    X_at_Zmin = 2000
    Z_at_Xmax = 0.0
    # X_at_Zmax = 300
    # X_unscaled = np.random.uniform(X_at_Zmax, X_at_Zmin, size=100)
    # response_func_z = lambda z: X_at_Zmax + decreasing_logistic(z, mid= 0.5 * Zmax, L=X_at_Zmin, k=0.05)

    response_func_z = lambda z: X_at_Zmin  + (1.2/Zmax) * z - (5.2/Zmax) * z**2
    print(response_func_z(0), response_func_z(Zmax))
    # # now known dZ, choose dX to get derivatives; take dZ = 1, we compute response X
    zs_ = np.random.uniform(0, Zmax-W, 100)
    plt.hist(zs_)
    plt.show()
    dzdx_ = []
    xs_ = []
    zmids_ = []
    for z in zs_:
        rfz = response_func_z(z) + np.random.normal(0, np.sqrt(sigma2_z))
        rfz_w = response_func_z(z + W) + np.random.normal(0, np.sqrt(sigma2_z))
        zmids_.append(z + 0.5 * W)
        dx = rfz_w - rfz
        xs_.append(rfz + 0.5 * dx)
        dzdx_.append(W/dx)


    sorted_zs_inds = np.argsort(zs_)
    sorted_zmids = np.array(zmids_)[sorted_zs_inds]
    sorted_dzdx_ = np.array(dzdx_)[sorted_zs_inds]
    sorted_xs_ = np.array(xs_)[sorted_zs_inds]


    # np.random.seed(50)

    # response_func_q = lambda q: 1.0 + 0.2 * q - 1.2 * q**2

    # n = 100
    # X = np.random.uniform(0, 1.0, size=n)
    # noise_std = 0.1
    # y_true = response_func_q(X) + np.random.normal(0, noise_std, size=n)

    # Xmid = []
    # dqdx = []
    # for i in range(1, n-1):
    #     x_mid = 0.5 * (X[i-1] + X[i])
    #     dy_dx = (y_true[i] - y_true[i-1]) / (X[i] - X[i-1])
    #     Xmid.append(x_mid)
    #     dqdx.append(dy_dx)

    # Xmid = np.array(Xmid)
    # dqdx = np.array(dqdx)

    X_test, f_mean, f_std = derivative_gp_simulation(sorted_xs_, sorted_dzdx_, Zmax)


    # -----------------------------
    # 5. Plot (single plot only)
    # -----------------------------
    
    fig, axes = plt.subplots(nrows=2, ncols=3)
    ax = axes.flatten()



    # Xsorted_inds = np.argsort(X)
    # Xsorted = X[Xsorted_inds]
    # Qsorted = y_true[Xsorted_inds]

    ax[0].plot(sorted_zmids, sorted_xs_)
    ax[0].scatter(sorted_zmids, sorted_xs_)
    ax[0].set_xlabel("Z")
    ax[0].set_ylabel("X")

    ax[1].plot(sorted_zmids, sorted_dzdx_)
    ax[1].scatter(sorted_zmids, sorted_dzdx_)
    ax[1].set_xlabel("Z")
    ax[1].set_ylabel("dZ/dX")

    ax[2].plot(sorted_xs_, 1.0/sorted_dzdx_)
    ax[2].scatter(sorted_xs_, 1.0/sorted_dzdx_)
    ax[2].set_xlabel("X")
    ax[2].set_ylabel("dX/dZ")

    ax[3].scatter(sorted_xs_, sorted_zmids, c="blue", label="Noisy data")
    ax[3].plot(sorted_xs_, sorted_zmids)
    ax[3].plot(X_test.flatten(), f_mean)
    ax[3].fill_between(
        X_test.flatten(),
        f_mean - 2 * f_std,
        f_mean + 2 * f_std,
        alpha=0.3
    )
    ax[4].set_xlabel("X")
    ax[4].set_ylabel("f(X)")
    ax[4].set_title("Function Recovery from Derivative-Only GP (with Uncertainty)")
    plt.show()


    # ax0.scatter(anchor_q, anchor_x, c="black", s=30)
    # ax0.set_title(f"Anchor points ($\sigma^2 = {anchor_sigma2}$)")
    # ax0.set_ylabel("X")
    # ax0.set_xlabel("Q")

    # for s in range(nS):
    #     S, X, hit_zmax = samps[s]
    #     Zrs = Z_real[s]

    #     Zest0 = S + c0[s] * Zmax_true
    #     ax2.plot(
    #         Zest0,
    #         X,
    #         c="#6b6b6b",
    #         linestyle="--",
    #         label="Unaligned sequences" if s == 0 else None,
    #     )
    #     ax2.plot(
    #         Zrs,
    #         X,
    #         c="#424161",
    #         label="Samples" if s == 0 else None,
    #     )

    #     Zest = S + cest[s] * zest
    #     ax3.plot(
    #         Zest,
    #         X,
    #         c="#6b6b6b",
    #         linestyle="--",
    #         label="Aligned sequences" if s == 0 else None,
    #     )
    #     ax3.plot(
    #         Zrs,
    #         X,
    #         c="#313045",
    #         label="Samples" if s == 0 else None,
    #     )

    # ax1.hist(c0, bins=20, color="#424161", alpha=1.0)
    # ax1.set_title("$c_0$ histogram (uniform distribution)")
    # ax1.set_xlabel("c0")
    # ax1.set_ylabel("Count")
    # ax2.set_title("Data alignment with randomly initialized $c_0$")
    # ax2.set_ylabel("X")
    # ax2.set_xlabel("Z")
    # ax2.legend(frameon=False)

    # ax3.set_title("Data alignment with estimated $\hat{c}$")
    # ax3.set_ylabel("X")
    # ax3.set_xlabel("Z")
    # ax3.legend(frameon=False)

    # q_grid = np.linspace(0.0, 1.0, num=200)
    # ax4.plot(q_grid, response_func_q(q_grid), c="#4136a3", label="True $h$")
    # ax4.plot(q_grid, hest(q_grid), c="#e6a532", label="Estimated $\hat{h}$")
    # ax4.plot(q_grid, hest_anchor_only(q_grid), c="red", label="Estimated $\hat{h}$ (anchors only)")
    # if len(hests) > 0:
    #     boot_preds = np.vstack([h(q_grid) for h in hests])
    #     lo = np.percentile(boot_preds, 2.5, axis=0)
    #     hi = np.percentile(boot_preds, 97.5, axis=0)
    #     ax4.fill_between(q_grid, lo, hi, color="#e6a532", alpha=0.2, label="Bootstrap 95% CI")
    # ax4.set_title("Response curve")
    # ax4.set_xlabel("Z")
    # ax4.set_ylabel("X")
    # ax4.legend(frameon=False)

    # ax5.plot(range(1, len(errors) + 1), np.log(errors), c="#313045", alpha=0.8)
    # ax5.set_title("Inference error (weighted SSE)")
    # ax5.set_xlabel("Iteration")
    # ax5.set_ylabel("Log Error")

    # fig.suptitle(f"Example result with {n_boot} bootstraps ($W={W},\sigma^2={sigma2},h=1/(1 - exp(-x))$)", fontsize=16)
    # plt.tight_layout()
    # plt.savefig("simulation_example.png")
    # plt.show()
