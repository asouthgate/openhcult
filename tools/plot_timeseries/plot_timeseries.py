#!/usr/bin/env python3
"""Backwards-compatible wrapper for hcultutils.plot_timeseries."""

from hcultutils.plot_timeseries import main


if __name__ == "__main__":
    raise SystemExit(main())
