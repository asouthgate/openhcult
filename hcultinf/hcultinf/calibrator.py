from __future__ import annotations

import os
from abc import ABC

import numpy as np

from hcultinf.plot_style import (
    apply_dark_theme,
    CLOUD_BLUE,
    YELLOW,
    ORANGE,
    CLOUD_WHITE,
)


def _u(x, xmin, xmax):
    return np.clip((xmax - x) / (xmax - xmin), 1e-10, 1.0)


def _estimate_covariance(result, n_data_obs):
    J = result.jac
    n_par = J.shape[1]
    data_res = result.fun[:n_data_obs]
    sigma2 = np.sum(data_res**2) / max(n_data_obs - n_par, 1)
    J_data = J[:n_data_obs]
    JtJ = J_data.T @ J_data
    try:
        return sigma2 * np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        return sigma2 * np.linalg.pinv(JtJ)


class CordCalibrator(ABC):
    def __init__(self):
        self._mean = None
        self._ci_low = None
        self._ci_high = None
        self.scale = None
        self.nlml = None
        self.noise = None

    def __call__(self, x):
        return self._mean(x)

    def predict(self, x):
        return self._mean(x), self._ci_low(x), self._ci_high(x)

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

        apply_dark_theme()
        priorx = np.asarray(priorx)
        plot_x = np.linspace(priorx.min(), priorx.max(), 500)
        plot_prior_y = np.interp(plot_x, priorx, priory)
        plot_true_y = (
            np.interp(plot_x, priorx, np.asarray(true_y))
            if true_y is not None
            else None
        )
        mean_at_x = self(x)
        mean, ci_low, ci_high = self.predict(plot_x)
        fig = plot_response_curve(
            plot_x,
            plot_prior_y,
            mean,
            ci_low,
            ci_high,
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
            for i, pwlprev in enumerate(pwlprevs):
                prev_vals = pwlprev(plot_x)
                ax.plot(
                    plot_x,
                    prev_vals - prev_vals.min(),
                    label=f"prev{i}",
                    alpha=(i + 1) / (1 + len(pwlprevs)),
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
    ci_low,
    ci_high,
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

    apply_dark_theme()

    prior_x = np.asarray(prior_x)
    prior_y = np.asarray(prior_y)
    mean = np.asarray(mean)
    ci_low = np.asarray(ci_low)
    ci_high = np.asarray(ci_high)
    x = np.asarray(x)
    dx = np.asarray(dx)
    dy = np.asarray(dy)
    mean_at_x = np.asarray(mean_at_x)

    if pct_fc:
        mean = mean / scale * 100
        ci_low = ci_low / scale * 100
        ci_high = ci_high / scale * 100
        dy = dy / scale * 100
        mean_at_x = mean_at_x / scale * 100
        if true_y is not None:
            true_y = np.asarray(true_y) / scale * 100

    if show_chords_pane:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x_margin = (prior_x.max() - prior_x.min()) * 0.08
    ax.set_xlim(prior_x.min() - x_margin, prior_x.max() + x_margin)

    for i in range(len(dx)):
        start_y = mean_at_x[i]
        label = "chords" if i == 0 else None
        ax.scatter(
            [x[i], x[i] + dx[i]],
            [start_y, start_y + dy[i]],
            color=YELLOW,
            alpha=0.7,
        )
        ax.plot(
            [x[i], x[i] + dx[i]],
            [start_y, start_y + dy[i]],
            color=YELLOW,
            alpha=0.5,
            label=label,
        )

    ax.scatter(anchors_x, anchors_y, label="anchors", color=ORANGE)
    gp_range = mean.max() - mean.min()
    ax.plot(
        prior_x,
        prior_y * gp_range,
        label="rescaled prior",
        linestyle="--",
        color=ORANGE,
    )
    ax.fill_between(
        prior_x, ci_low, ci_high, color=CLOUD_BLUE, alpha=0.2, label="95% CI"
    )
    ax.plot(prior_x, mean, label="Estimated mean", color=CLOUD_BLUE)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if true_y is not None:
        ax.plot(prior_x, np.asarray(true_y), label="true")

    ax.legend()

    if show_chords_pane:
        ax2.scatter(dx, dy, alpha=0.7, color=YELLOW)
        for i, (dxi, dyi) in enumerate(zip(dx, dy)):
            ax2.annotate(str(i), (dxi, dyi), fontsize=8, alpha=0.6, color=CLOUD_WHITE)
        ax2.axhline(0, color=CLOUD_BLUE, linewidth=0.5, linestyle="--", alpha=0.4)
        ax2.axvline(0, color=CLOUD_BLUE, linewidth=0.5, linestyle="--", alpha=0.4)
        ax2.set_xlabel(f"Δ{xlabel}")
        ax2.set_ylabel(f"Δ{ylabel}")
        ax2.set_title("chord Δx vs Δy")

    fig.tight_layout()
    return fig
