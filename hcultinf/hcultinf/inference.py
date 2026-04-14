from __future__ import annotations

import os

import numpy as np


class GPWithPriorShape:
    def __init__(
        self,
        length_scale=None,
        variance=20.0,
        scale_prior_mean=None,
        scale_prior_std=None,
    ):
        self._length_scale = length_scale
        self._variance = variance
        self._scale_prior_mean = scale_prior_mean
        self._scale_prior_std = scale_prior_std
        self._mean = None
        self._std = None
        self.scale = None
        self.nlml = None
        self.noise = None

    def fit(
        self, x_anchors, swc_anchors, x_starts, delta_x, delta_swc, prior_x, prior_y
    ):
        x_ends = x_starts + delta_x
        self._mean, self._std, self.scale, self.nlml, self.noise = self.fit_gp_chords(
            x_anchors,
            swc_anchors,
            x_starts,
            x_ends,
            delta_swc,
            prior_x,
            prior_y,
            self._length_scale,
            self._variance,
            self._scale_prior_mean,
            self._scale_prior_std,
        )
        return self

    def __call__(self, x):
        return self._mean(x)

    def predict(self, x):
        return self._mean(x), self._std(x)

    def fit_gp_chords(
        self,
        x_anchor,
        y_anchor,
        x_starts,
        x_ends,
        delta_y,
        prior_x,
        prior_y,
        length_scale=None,
        variance=20.0,
        scale_prior_mean=None,
        scale_prior_std=None,
    ):
        from scipy.optimize import minimize_scalar

        if length_scale is None:
            length_scale = np.median(np.abs(x_ends - x_starts)) * 1.0

        def kernel(x1, x2):
            sq_dist = np.subtract.outer(x1, x2) ** 2
            return variance * np.exp(-0.5 * sq_dist / length_scale**2)

        # Prior shape evaluated at anchors and as chord differences
        h_anchor = np.interp(x_anchor, prior_x, prior_y)
        h_chords = np.interp(x_ends, prior_x, prior_y) - np.interp(
            x_starts, prior_x, prior_y
        )
        h = np.concatenate([h_anchor, h_chords])

        # Kernel matrices (noise-free parts)
        K_aa = kernel(x_anchor, x_anchor)
        K_ac = kernel(x_anchor, x_ends) - kernel(x_anchor, x_starts)
        K_cc = (
            kernel(x_ends, x_ends)
            - kernel(x_ends, x_starts)
            - kernel(x_starts, x_ends)
            + kernel(x_starts, x_starts)
        )

        observations = np.concatenate([y_anchor, delta_y])
        n_a = len(y_anchor)
        n_c = len(delta_y)
        n = n_a + n_c

        tau = (1.0 / scale_prior_std**2) if scale_prior_std is not None else 0.0
        mu_0 = scale_prior_mean if scale_prior_mean is not None else 0.0

        def _build_K(log_noise):
            noise = np.exp(log_noise)
            noise_mat = np.zeros((n, n))
            noise_mat[n_a:, n_a:] = noise * np.eye(n_c)
            return np.block([[K_aa, K_ac], [K_ac.T, K_cc]]) + noise_mat

        def neg_log_marginal_likelihood(log_noise):
            K = _build_K(log_noise)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e10
            Kinv_h = np.linalg.solve(K, h)
            Kinv_y = np.linalg.solve(K, observations)
            hKh = float(h @ Kinv_h)
            hKy = float(h @ Kinv_y)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            if tau > 0:
                post_prec = hKh + tau
                yKy = float(observations @ Kinv_y)
                return (
                    0.5 * yKy
                    - 0.5 * (hKy + mu_0 * tau) ** 2 / post_prec
                    + 0.5 * log_det
                    + 0.5 * np.log(post_prec)
                )
            beta = hKy / hKh
            residuals = observations - beta * h
            Kinv_r = np.linalg.solve(K, residuals)
            return 0.5 * float(residuals @ Kinv_r) + 0.5 * log_det

        # Optimize noise in log space
        result = minimize_scalar(
            neg_log_marginal_likelihood,
            bounds=(np.log(1e-8), np.log(1e2)),
            method="bounded",
        )

        # Build final K with optimized noise
        K = _build_K(result.x)

        Kinv_h = np.linalg.solve(K, h)
        Kinv_y = np.linalg.solve(K, observations)
        hKh = float(h @ Kinv_h)
        hKy = float(h @ Kinv_y)

        if tau > 0:
            post_prec = hKh + tau
            scale = (hKy + mu_0 * tau) / post_prec
            beta_post_var = 1.0 / post_prec
        else:
            scale = hKy / hKh
            beta_post_var = 1.0 / hKh

        # GP on residuals after removing scaled mean
        residuals = observations - scale * h
        weights = np.linalg.solve(K, residuals)

        # Negative log marginal likelihood at optimized params (lower = better fit)
        L = np.linalg.cholesky(K)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        nlml = 0.5 * float(residuals @ np.linalg.solve(K, residuals)) + 0.5 * log_det

        def _k_joint(x_test):
            k_ta = kernel(x_test, x_anchor)
            k_tc = kernel(x_test, x_ends) - kernel(x_test, x_starts)
            return np.hstack([k_ta, k_tc])

        def predict_mean(x_test):
            x_test = np.atleast_1d(x_test)
            h_test = np.interp(x_test, prior_x, prior_y)
            return scale * h_test + _k_joint(x_test) @ weights

        def predict_std(x_test):
            x_test = np.atleast_1d(x_test)
            kj = _k_joint(x_test)
            prior_var_diag = np.diag(kernel(x_test, x_test))
            post_var = prior_var_diag - np.sum(kj * np.linalg.solve(K, kj.T).T, axis=1)
            # Extra term from uncertainty in beta
            h_test = np.interp(x_test, prior_x, prior_y)
            h_tilde = h_test - kj @ Kinv_h
            post_var += beta_post_var * h_tilde**2
            return np.sqrt(np.maximum(post_var, 0.0))

        return predict_mean, predict_std, scale, nlml, np.exp(result.x)

    def plot(
        self,
        priorx,
        priory,
        anchors_x,
        anchors_y,
        x,
        dx,
        dy,
        pwlprevs=None,
        true_y=None,
        out=None,
        title=None,
        show_chords_pane=True,
    ):
        import matplotlib.pyplot as plt

        priorx = np.asarray(priorx)
        priory = np.asarray(priory)
        plot_x = np.linspace(priorx.min(), priorx.max(), 500)
        plot_prior_y = np.interp(plot_x, priorx, priory)
        plot_true_y = (
            np.interp(plot_x, priorx, np.asarray(true_y))
            if true_y is not None
            else None
        )
        mean_at_x = self(x)
        mean, std = self.predict(plot_x)
        fig = plot_response_curve(
            plot_x,
            plot_prior_y,
            mean,
            std,
            anchors_x,
            anchors_y,
            x,
            dx,
            dy,
            mean_at_x,
            true_y=plot_true_y,
            show_chords_pane=show_chords_pane,
        )
        if pwlprevs is not None:
            ax = fig.axes[0]
            ref = mean.min()
            for i, pwlprev in enumerate(pwlprevs):
                prev_vals = pwlprev(plot_x)
                ax.plot(
                    plot_x,
                    prev_vals - prev_vals.min(),
                    label=f"prev{i}",
                    alpha=0.5,
                    color="brown",
                    linestyle="--",
                )
            ax.legend()
        if out is not None:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            fig.savefig(out)
        if title is not None:
            fig.suptitle(title)
        if os.environ.get("HCULT_TEST_DEBUG_PLOT", "0") == "1":
            plt.show()

        plt.close(fig)
        return fig


def plot_response_curve(
    prior_x,
    prior_y,
    mean,
    std,
    anchors_x,
    anchors_y,
    x,
    dx,
    dy,
    mean_at_x,
    true_y=None,
    xlabel="sensor reading",
    ylabel="SWC",
    pct_fc=False,
    scale=1.0,
    show_chords_pane=True,
):
    import matplotlib.pyplot as plt

    prior_x = np.asarray(prior_x)
    prior_y = np.asarray(prior_y)
    mean = np.asarray(mean)
    std = np.asarray(std)
    x = np.asarray(x)
    dx = np.asarray(dx)
    dy = np.asarray(dy)
    mean_at_x = np.asarray(mean_at_x)

    if pct_fc:
        mean = mean / scale * 100
        std = std / scale * 100
        dy = dy / scale * 100
        mean_at_x = mean_at_x / scale * 100
        if true_y is not None:
            true_y = np.asarray(true_y) / scale * 100

    ci_lower = mean - 1.96 * std
    ci_upper = mean + 1.96 * std

    if show_chords_pane:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for i in range(len(dx)):
        start_y = mean_at_x[i]
        label = "chords" if i == 0 else None
        ax.scatter(
            [x[i], x[i] + dx[i]],
            [start_y, start_y + dy[i]],
            color="steelblue",
            alpha=0.5,
        )
        ax.plot(
            [x[i], x[i] + dx[i]],
            [start_y, start_y + dy[i]],
            color="steelblue",
            alpha=0.5,
            label=label,
        )

    ax.scatter(anchors_x, anchors_y, label="anchors")
    gp_range = mean.max() - mean.min()
    ax.plot(prior_x, prior_y * gp_range, label="rescaled prior", linestyle="--")
    ax.fill_between(
        prior_x, ci_lower, ci_upper, color="gray", alpha=0.3, label="95% CI"
    )
    ax.plot(prior_x, mean, label="GP mean", linestyle="dotted")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if true_y is not None:
        ax.plot(prior_x, np.asarray(true_y), label="true")

    ax.legend()

    if show_chords_pane:
        ax2.scatter(dx, dy, alpha=0.7)
        for i, (dxi, dyi) in enumerate(zip(dx, dy)):
            ax2.annotate(str(i), (dxi, dyi), fontsize=8, alpha=0.6)
        ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax2.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax2.set_xlabel(f"Δ{xlabel}")
        ax2.set_ylabel(f"Δ{ylabel}")
        ax2.set_title("chord Δx vs Δy")

    fig.tight_layout()
    return fig
