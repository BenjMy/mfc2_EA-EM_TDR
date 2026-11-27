#!/usr/bin/env python3
"""
replot_maps.py -- redraw the XY water-content maps of a finished ``--stage map`` run
from its saved ``output_model/maps.nc``, WITHOUT re-running any inversion.

Used by ``hyp*.py --plot-map`` (see hyp_common.py); also usable directly:

    python replot_maps.py <map run folder> [--window-m 3.0] [--theta-limits zero_p98]

Figures go to ``<run folder>/replot/`` (the originals are never overwritten):
    water_content_xy_<scenario>.png               one square per point (NaN cells not drawn)
    water_content_xy_<scenario>_imshow_binned.png per-pixel mean, empty pixels blank
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

import maps_export as mx
from plot_water_content import _theta_color_limits, plot_water_content_xy_timelapse


def latest_map_run(output_dir: str, scenario: str) -> str | None:
    """Newest ``<NNNN>_<scenario>_map`` folder under ``output_dir`` (None when there is none)."""
    cands = glob.glob(os.path.join(output_dir, f"*_{scenario}_map"))
    cands = [c for c in cands if os.path.isfile(os.path.join(c, "output_model", "maps.nc"))]

    def key(p: str) -> tuple[int, str]:
        m = re.match(r"(\d+)_", os.path.basename(p))
        return (int(m.group(1)) if m else -1, p)

    return max(cands, key=key) if cands else None


def replot_map_run(run_dir: str, scenario: str, window_m: float | None = None,
                   theta_limits: str = "zero_p98") -> list[str]:
    """Redraw the XY time-lapse maps of one map run; returns the written PNG paths."""
    ds = mx.load_maps(os.path.join(run_dir, "output_model", "maps.nc"))
    if ds is None:
        raise FileNotFoundError(f"no readable maps.nc under {run_dir}/output_model "
                                "(needs xarray + netcdf4)")
    theta = np.asarray(ds["theta"].values, dtype=float) * 100.0       # m3/m3 -> % vol
    x, y = np.asarray(ds["x"].values), np.asarray(ds["y"].values)
    if np.unique(y).size < 2:
        raise ValueError("this is a 1-D run (all y equal): there is no XY plan view to draw")
    vmin, vmax = _theta_color_limits([theta], theta_limits)
    out_dir = os.path.join(run_dir, "replot")
    common = dict(theta_calc=theta, utm_x=x, utm_y=y, theta_vmin=vmin, theta_vmax=vmax,
                  output_dir=out_dir, scenario_name=scenario, show_probes=False)
    plot_water_content_xy_timelapse(**common)
    written = [os.path.join(out_dir, f"water_content_xy_{scenario}.png")]
    if window_m:
        plot_water_content_xy_timelapse(**common, render_mode="imshow", interp_method="mean",
                                        window_m=window_m, min_grid_res=4,
                                        filename_suffix="_imshow_binned")
        written.append(os.path.join(out_dir, f"water_content_xy_{scenario}_imshow_binned.png"))
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", help="a '<NNNN>_<scenario>_map' run folder")
    ap.add_argument("--scenario", default=None, help="name used in titles/files (default: from the folder name)")
    ap.add_argument("--window-m", type=float, default=3.0, help="pixel = 2 x window_m for the binned map (0 = skip it)")
    ap.add_argument("--theta-limits", default="zero_p98", choices=["zero_p98", "percentile"])
    a = ap.parse_args(argv)
    scen = a.scenario or re.sub(r"^\d+_|_map$", "", os.path.basename(os.path.normpath(a.run_dir)))
    for p in replot_map_run(a.run_dir, scen, a.window_m or None, a.theta_limits):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
