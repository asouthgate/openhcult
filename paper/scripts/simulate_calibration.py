from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

from hcultinf.inference import fit_gp

def decreasing_logistic(x: np.ndarray, *, mid: float, L: float, k) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return L / (1.0 + np.exp(k * (x - mid)))

prop = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",
            shrinkA=0,shrinkB=0)

def _plot_arrows(zmids, xss, dzdxs, ax, step, reverse_arrow):
    for zi, zmid in enumerate(zmids): 
        dzdx = dzdxs[zi]
        xmid = xss[zi]
        # compute little line segment forward and back using xmid, zmid, and dzdx
        # step = W * 10
        x1 = xmid + 0.5 * step
        x2 = xmid - 0.5 * step
        z1 = zmid + 0.5 * step * dzdx
        z2 = zmid - 0.5 * step * dzdx
        # add an arrowhead
        if reverse_arrow:
            ax.annotate("", xy=(x2, z2), xytext=(x1, z1), arrowprops=prop, alpha=0.5)
        else:
            ax.annotate("", xy=(x1, z1), xytext=(x2, z2), arrowprops=prop, alpha=0.5)
        ax.scatter(xmid, zmid, color='black', s=10)

def get_gamma_shape_scale(z, Zmax, varmin, varmax):

    peak_location = 0.2 * Zmax
    # use 2 term linear interpolation to get maximum variance varmax at peak location, vmin at 0 and zmax
    if z < peak_location:
        fz = z / peak_location
    else:
        fz = (Zmax - z) / (Zmax - peak_location)
    
    
    varZ = varmin + (varmax - varmin) * fz ** 2

    shape = 1.0 / varZ
    scale = varZ
    return shape, scale

def sample_gamma_func(z, Zmax, varmin, varmax):
    shape, scale = get_gamma_shape_scale(z, Zmax, varmin, varmax)
    return np.random.gamma(shape, scale)

def get_gamma_pdf(x, z, Zmax, varZ0, varZmax):
    shape, scale = get_gamma_shape_scale(z, Zmax, varZ0, varZmax)
    return gamma.pdf(x=x, a=shape, scale=scale)

def get_gamma_mean(z, Zmax, varZ0, varZmax):
    shape, scale = get_gamma_shape_scale(z, Zmax, varZ0, varZmax)
    return shape * scale

def get_gamma_95_percent_ci(z, Zmax, varZ0, varZmax):
    shape, scale = get_gamma_shape_scale(z, Zmax, varZ0, varZmax)
    return gamma.ppf(0.95, a=shape, scale=scale)

def get_gamma_05_percent_ci(z, Zmax, varZ0, varZmax):
    shape, scale = get_gamma_shape_scale(z, Zmax, varZ0, varZmax)
    return gamma.ppf(0.05, a=shape, scale=scale)

if __name__ == "__main__":
    import sys
    W = 0.2
    Zmax = 2.0
    Q_at_Xmin = 1.0
    sigma2 = 0.0
    sigma2_w = 0.0
    sigma2_x = 0.0
    X_at_Zmin = 2832
    Z_at_Xmax = 959
    n_samps_per_sensor = 50
    n_sensors = 3

    gamma_var_min = 0.01
    gamma_var_max = 0.3

    pcoeffs = [-272.75690642, 1624.74832501, -3105.44572851, 2897.46597558]
    response_func_z = lambda z: np.polyval(pcoeffs, z)
    # set response func to an exponential intersecting at X_at_Zmin
    # response_func_z = lambda z: X_at_Zmin * np.exp(-0.02 * z) + 50 * np.exp(-0.01 * z)
    print(response_func_z(0), response_func_z(Zmax))

    # # now known dZ, choose dX to get derivatives; take dZ = 1, we compute response X
    zs_ = np.random.uniform(0, Zmax-W-W/2, n_samps_per_sensor)  # /2 because of the midpointing later
    zs_ = np.concatenate([zs_, np.random.uniform(0, Zmax / 2, n_samps_per_sensor)])
    zs_ = sorted(zs_)

    dzdx_ = []
    dxs_all = []
    dzs_ = []
    dxs_ = []
    xs_ = []
    zmids_ = []
    sensors = []


    for z in zs_:
        dxs = []
        # dzdxs = []
        zmids_.append(z + 0.5 * W)
        for sensor in range(n_sensors):
            # if alpha_W:
            M = sample_gamma_func(z, Zmax, gamma_var_min, gamma_var_max)
            # else:
            #     M = 1.0
            assert M > 0
            W_scaled = W * M
            assert W_scaled > 0
            rfz = response_func_z(z) + np.random.normal(0, np.sqrt(sigma2_x))
            rfz_w = response_func_z(z + W_scaled) + np.random.normal(0, np.sqrt(sigma2_x))
            dx_ = (rfz_w - rfz) 
            # assert wdx_ < 0.0
            dxs.append(dx_)
            # dzdxs.append(wdx_)
        dxs_all.append(dxs)
        # dx = sum(dxs) / len(dxs)
        dx = np.median(dxs)
        dzdx = W / dx
        xs_.append(rfz + 0.5 * dx)
        dzdx_.append(dzdx)
        dxs_.append(dx)


    sensors = np.array(sensors)
    xs_ = np.array(xs_)
    zmids_ = np.array(zmids_)
    colors = ["brown", "green", "purple", "red"]
    sensor_colors = {i:colors[i] for i in sensors}

    sorted_zs_inds = np.argsort(zs_)
    sorted_zmids = np.array(zmids_)[sorted_zs_inds]
    sorted_dzdx_ = np.array(dzdx_)[sorted_zs_inds]
    sorted_xs_ = np.array(xs_)[sorted_zs_inds]
    sorted_dxdz_ = 1.0 / np.array(sorted_dzdx_)
    
    X_test, dy_samples, f_mean, f_std = fit_gp(sorted_xs_, sorted_dzdx_, Zmax, None)
    
    # fig, axes = plt.subplots(nrows=2, ncols=3)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    ax = axes.flatten()

    # plot gamma func noise distribtion over z; need to take many samples from the distribution at each value of x, then plot between
    gamma_mean_z = [get_gamma_mean(z, Zmax, gamma_var_min, gamma_var_max) for z in zs_]
    gamma_ci_z_upper = [get_gamma_95_percent_ci(z, Zmax, gamma_var_min, gamma_var_max) for z in zs_]
    gamma_ci_z_lower = [get_gamma_05_percent_ci(z, Zmax, gamma_var_min, gamma_var_max) for z in zs_]
    ax[0].plot(zs_, gamma_mean_z, label="Gamma noise mean")
    ax[0].fill_between(zs_, gamma_ci_z_lower, gamma_ci_z_upper, alpha=0.3, label="Gamma noise 90% CI")
    ax[0].set_xlabel("Z")
    ax[0].set_ylabel("Gamma noise")

    for sdi, sensor_dxs in enumerate(dxs_all):
        z = zs_[sdi] 
        for si, sdx in enumerate(sensor_dxs):
            # plot vertical line of length sdx at position z, centered on response(z)
            response = response_func_z(z)
            ax[1].plot([z, z], [response, response+sdx], color=colors[si], alpha=1.0, label=f"Sensor {si}" if sdi == 0 else None)
            ax[1].plot([z, z+W], [response+sdx, response+sdx], color=colors[si], alpha=1.0)
    ax[1].set_xlabel("Z")
    ax[1].set_ylabel("X")
    ax[1].legend()

    for sdi, sensor_dxs in enumerate(dxs_all):
        z = zs_[sdi] 
        for si, sdx in enumerate(sensor_dxs):
            # plot vertical line of length sdx at position z, centered on response(z)
            response = response_func_z(z)
            ax[2].plot([response+sdx, response+sdx], [z, z+W], color=colors[si], alpha=1.0, label=f"Sensor {si}" if sdi == 0 else None)
            ax[2].plot([response, response+sdx], [z, z], color=colors[si], alpha=1.0)
    ax[2].set_xlabel("X")
    ax[2].set_ylabel("Z")
    ax[2].legend()

    ax[3].plot([response_func_z(z) for z in zs_], zs_, color='blue', alpha=0.5)
    _plot_arrows(zmids_, xs_, dzdx_, ax[3], 1/W, True)
    
    ax[3].set_xlabel("$X$ ")
    ax[3].set_ylabel("Z")

    ax[4].plot(zs_, [response_func_z(z) for z in zs_])
    ax[4].set_xlabel("Z")
    ax[4].set_ylabel("$X$")
    _plot_arrows(xs_, zmids_, 1.0/np.array(dzdx_), ax[4], W, False)

    ax[5].scatter(sorted_zmids, sorted_dzdx_)
    ax[5].set_xlabel("Z")
    ax[5].set_ylabel("$dZ/dX$")

    ax[6].scatter(sorted_xs_, 1.0/sorted_dzdx_)
    ax[6].set_xlabel("X")
    ax[6].set_ylabel("$dX/dZ$")
    dmean2 = np.mean(dy_samples_2, axis=1)
    dstd2 = np.std(dy_samples_2, axis=1)
    ax[6].plot(X_test_2.flatten(), dmean2, color='orange', label="GP mean")
    ax[6].fill_between(
        X_test_2.flatten(),
        dmean2 - 2 * dstd2,
        dmean2 + 2 * dstd2,
        alpha=0.3, color='orange', label="GP samples at 95% CI",
    )

    ax[7].scatter(sorted_xs_, sorted_dzdx_, label="Sampled derivative data")
    dmean = np.mean(dy_samples, axis=1)
    dstd = np.std(dy_samples, axis=1)
    ax[7].plot(X_test.flatten(), dmean, label="GP mean")
    ax[7].fill_between(
        X_test.flatten(),
        dmean - 2 * dstd,
        dmean + 2 * dstd,
        alpha=0.3,
        label="GP 95% CI",
    )
    ax[7].set_xlabel("$X$")
    ax[7].set_ylabel("$f'(X)$")
    ax[7].legend()

    ax[8].plot([response_func_z(z) for z in zs_], zs_, label="True response curve")
    ax[8].plot(X_test.flatten(), f_mean, color = 'orange', label="Integrated GP mean")
    ax[8].fill_between(
        X_test.flatten(),
        f_mean - 2 * f_std,
        f_mean + 2 * f_std,
        alpha=0.3, color = 'orange', label="Integrated GP samples at 95% CI",
    )
    ax[8].set_xlabel("$X$")
    ax[8].set_ylabel("$f(X)$")
    ax[8].legend()
    plt.suptitle("GP regression on derivatives with integration to reconstruct response curve")
    plt.savefig("simulation_example.png", dpi=300)
    plt.show()