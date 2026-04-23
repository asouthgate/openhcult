DARK_BLUE = "#0041a1"
YELLOW = "#ffe23f"
ORANGE = "#ffc61c"
CLOUD_WHITE = "#fffff2"
CLOUD_BLUE = "#d0fffc"
BG_DARK = "#191e2b"
BG_RAISED = "#222838"
MUTED = "#8892a4"

PALETTE = [
    CLOUD_BLUE,
    YELLOW,
    ORANGE,
    DARK_BLUE,
    "#ff8a65",
    "#a5d6a7",
    "#ce93d8",
    "#90caf9",
]

MPL_RC = {
    "figure.facecolor": BG_DARK,
    "axes.facecolor": BG_DARK,
    "axes.edgecolor": "#3a4258",
    "axes.labelcolor": MUTED,
    "axes.grid": True,
    "grid.color": "#3a4258",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": CLOUD_WHITE,
    "font.family": "monospace",
    "font.size": 10,
    "lines.color": CLOUD_BLUE,
    "legend.facecolor": BG_RAISED,
    "legend.edgecolor": "#3a4258",
    "scatter.edgecolors": "none",
    "savefig.facecolor": BG_DARK,
    "savefig.edgecolor": BG_DARK,
}


def apply_dark_theme():
    import matplotlib.pyplot as plt

    plt.rcParams.update(MPL_RC)
