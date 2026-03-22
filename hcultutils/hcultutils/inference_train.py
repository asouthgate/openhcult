import itertools

import numpy as np

from hcultutils.fetch_data import fetch_data
from hcultinf.training import (
    filter_confirmed_watering_events,
    grid_search,
    plot_roc,
    plot_segmentation,
    plot_debug_events,
)


def inference_train_main(ctrl_url: str, args) -> int:
    plant_name = getattr(args, "plant_name", None)
    series, observations = fetch_data(
        ctrl_url, args.start_utc, args.end_utc, plant_name=plant_name, limit=args.limit
    )
    if not series or not observations:
        print("No data found.")
        return 0

    confirmed = filter_confirmed_watering_events(observations)
    print(f"Found {len(confirmed)} confirmed watering events.")
    if not confirmed:
        print("No confirmed events to train against.")
        return 0

    n = args.grid_n
    tau_range = np.linspace(5, 60, n)
    trigger_range = np.linspace(-15.0, -0.5, n)
    release_range = np.linspace(-10.0, -0.1, n)
    param_grid = [
        {
            "emwa_tau_minutes": float(tau),
            "trigger_thresh": float(trig),
            "release_thresh": float(rel),
        }
        for tau, trig, rel in itertools.product(tau_range, trigger_range, release_range)
    ]
    print(f"Running grid search over {len(param_grid)} parameter combinations...")

    results = grid_search(confirmed, series, param_grid)

    top = sorted(results, key=lambda r: r["f1"], reverse=True)[:5]
    print("\nTop 5 by F1:")
    for i, r in enumerate(top):
        p = r["params"]
        print(
            f"  #{i+1} F1={r['f1']:.3f} TP={r['tp']} FP={r['fp']} FN={r['fn']} "
            f"tau={p['emwa_tau_minutes']:.1f} trig={p['trigger_thresh']:.2f} rel={p['release_thresh']:.2f}"
        )

    best = plot_roc(results)

    if args.debug:
        print(f"\nDebug mode: best params F1={best['f1']:.3f}")
        plot_segmentation(series, best["params"])
        plot_debug_events(series, best["params"], confirmed)

    return 0
