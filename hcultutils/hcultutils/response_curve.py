import urllib.parse

import numpy as np

from hcultinf.calibrator import plot_response_curve
from hcultinf.plot_style import apply_dark_theme, YELLOW, ORANGE, CLOUD_BLUE
from hcultutils.query import request_ctrl

from hcultinf.exp_mcmc import ExponentialCordCalibratorMCMC


def response_curve_estimate_main(ctrl_url: str, args) -> int:
    import matplotlib

    if args.out:
        matplotlib.use("Agg")

    params = {
        "plant": args.plant_name,
        "gp_std_ml": args.gp_std_ml,
        "offset_ms": args.offset_min * 60 * 1000,
        "width_ms": args.width_min * 60 * 1000,
        "estimator": "exp_mcmc",
        "prior_weight": 1.0,
    }
    if args.scale_prior_mean is not None:
        params["scale_prior_mean"] = args.scale_prior_mean
    if args.scale_prior_std is not None:
        params["scale_prior_std"] = args.scale_prior_std
    url = f"{ctrl_url.rstrip('/')}/water_calibration?{urllib.parse.urlencode(params)}"
    data = request_ctrl("GET", url)

    import matplotlib.pyplot as plt

    apply_dark_theme()
    pct_fc = getattr
    nlml = data.get("nlml")
    fig = plot_response_curve(
        prior_x=np.array(data["prior_x"]),
        prior_y=np.array(data["prior_y"]),
        mean=np.array(data["mean"]),
        std=np.array(data["std"]),
        anchors_x=data["anchors_x"],
        anchors_y=data["anchors_y"],
        x=np.array(data["chords_x"]),
        dx=np.array(data["chords_dx"]),
        dy=np.array(data["chords_dy"]),
        mean_at_x=np.array(data["mean_at_chord_starts"]),
        xlabel="sensor reading",
        ylabel="%SC" if pct_fc else "SWC (ml)",
        pct_fc=pct_fc,
        scale=data["scale"],
    )

    for xi, x in enumerate(data["chords_x"]):
        print(
            f"Chord {xi}: start={x:.1f}, dx={data['chords_dx'][xi]:.1f}, dy={data['chords_dy'][xi]:.1f})"
        )

    cal = ExponentialCordCalibratorMCMC(
        xmin_mu=950,
        xmin_sigma=75,
        xmin_high=1100,
        xmax=2000,
        prior_weight=0.1,
    ).fit(
        x_anchors=data["anchors_x"],
        swc_anchors=data["anchors_y"],
        x_starts=data["chords_x"],
        delta_x=data["chords_dx"],
        delta_swc=data["chords_dy"],
        prior_x=data["prior_x"],
        prior_y=data["prior_y"],
    )

    ax1 = fig.axes[0]

    ax1.plot(
        data["prior_x"],
        cal(data["prior_x"]),
        color=ORANGE,
        linewidth=2,
        label=f"Power fit (scale={cal.scale:.2f})",
    )

    if nlml is not None:
        fig.suptitle(f"NLML: {nlml:.2f}", fontsize=10)

    if args.out:
        fig.savefig(args.out)
        print(f"Wrote {args.out}")
    else:
        plt.show()

    plt.close(fig)
    return 0
