#!/usr/bin/env python3
"""Generate paper figures from inference test data.

Produces publication-quality plots (white background, 300 dpi) for
inclusion in the LaTeX paper:

  images/real_data_timeseries.png     - SWC over time per sensor
  images/real_data_calibration.png    - parametric calibration curves with chords
  images/real_data_corner.png         - MCMC posterior corner plot
  images/real_data_smoothed.png       - per-sensor raw voltage + smoothed
"""

from __future__ import annotations

import sys
from pathlib import Path

from hcultinf.data import load_calibration_data, fit_calibrators
from hcultinf.plot import (
    plot_calibration_curves,
    plot_timeseries,
    plot_smoothed_voltage,
    plot_corner,
)


def generate_figures(output_dir: Path, data_dir: Path | None = None):
    d = load_calibration_data(data_dir)
    single_cals, joint_cal = fit_calibrators(d)

    plot_calibration_curves(single_cals, joint_cal, d,
                            out=output_dir / "real_data_calibration.png")
    plot_timeseries(single_cals, joint_cal, d,
                     out=output_dir / "real_data_timeseries.png")
    plot_corner(joint_cal, n_sensors=d["n_sensors"],
                out=output_dir / "real_data_corner.png")
    plot_smoothed_voltage(d, out=output_dir / "real_data_smoothed.png")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "images"
    data = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    generate_figures(out, data)