import json
import urllib.parse
import urllib.request

import numpy as np

from hcultinf.inference import plot_response_curve


def response_curve_estimate_main(ctrl_url: str, args) -> int:
    import matplotlib

    if args.out:
        matplotlib.use("Agg")

    url = (
        f"{ctrl_url.rstrip('/')}/water_calibration"
        f"?{urllib.parse.urlencode({'plant': args.plant_name})}"
    )
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    import matplotlib.pyplot as plt

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
        ylabel="SWC (ml)",
    )

    if args.out:
        fig.savefig(args.out)
        print(f"Wrote {args.out}")
    else:
        plt.show()

    plt.close(fig)
    return 0
