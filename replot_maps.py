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


def epoch_labels(em_dir: str | None, n_time: int) -> list[str] | None:
    """Survey dates (YYYY-MM-DD) of the EM files, ordered as the pipeline orders its epochs
    (data_real.prepare_real_dataset: ``*.csv`` with a DDMMYY token, sorted by date).
    None when the folder is unknown or its dated-file count differs from ``n_time``."""
    import datetime as dt
    if not em_dir or not os.path.isdir(em_dir):
        return None
    dates = []
    for f in glob.glob(os.path.join(em_dir, "*.csv")):
        m = re.search(r"(?:^|\D)(\d{6})(?:\D|$)", os.path.basename(f))
        if m:
            try:
                dates.append(dt.datetime.strptime(m.group(1), "%d%m%y"))
            except ValueError:
                pass
    if len(dates) != n_time:
        return None
    return [d.strftime("%Y-%m-%d") for d in sorted(dates)]


def load_probe_xy(coord_file: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Station UTM x/y from the sensor coordinate CSV (real ``SOILNET_MFC2_coordinates_sensors.csv``
    or the synthetic mirror's ``standin_coordinates.csv``): ';'-separated, header lines starting
    with '[' ignored, columns ``x``/``y`` or ``X_UTM``/``Y_UTM`` (decimal commas accepted)."""
    import pandas as pd
    if not coord_file or not os.path.isfile(coord_file):
        return None
    df = pd.read_csv(coord_file, sep=";", comment="[", dtype=str)
    df.columns = df.columns.str.strip()
    cx, cy = (("X_UTM", "Y_UTM") if {"X_UTM", "Y_UTM"} <= set(df.columns) else ("x", "y"))
    if cx not in df.columns or cy not in df.columns:
        return None
    conv = lambda c: pd.to_numeric(df[c].str.strip().str.replace(",", ".", regex=False), errors="coerce")
    x, y = conv(cx).to_numpy(float), conv(cy).to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    return (x[ok], y[ok]) if ok.any() else None


def replot_map_run(run_dir: str, scenario: str, window_m: float | None = None,
                   theta_limits: str = "zero_p98", em_dir: str | None = None,
                   coord_file: str | None = None) -> list[str]:
    """Redraw the XY time-lapse maps of one map run; returns the written PNG paths.

    ``em_dir`` -> epoch panels titled with the survey dates; ``coord_file`` -> probe markers
    (both optional: without them the panels are 't = i' and no probes are drawn)."""
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
                  output_dir=out_dir, scenario_name=scenario, show_probes=False,
                  survey_labels=epoch_labels(em_dir, theta.shape[0]))
    probes = load_probe_xy(coord_file)
    if probes is not None:
        px, py = probes
        pad = 100.0      # guard against a CRS / frame mismatch: probes must sit near the survey
        near = (px > np.nanmin(x) - pad) & (px < np.nanmax(x) + pad) & \
               (py > np.nanmin(y) - pad) & (py < np.nanmax(y) + pad)
        if near.any():
            common.update(probe_utm_x=px[near], probe_utm_y=py[near], show_probes=True)
        else:
            print(f"[plot-map] WARNING: no probe of {coord_file} lies within {pad:.0f} m of the "
                  "survey points (different coordinate frame?): probe markers skipped")
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
    ap.add_argument("--em-dir", default=None, help="EM survey folder (epoch dates in the panel titles)")
    ap.add_argument("--coord-file", default=None, help="sensor coordinates CSV (probe markers)")
    a = ap.parse_args(argv)
    scen = a.scenario or re.sub(r"^\d+_|_map$", "", os.path.basename(os.path.normpath(a.run_dir)))
    for p in replot_map_run(a.run_dir, scen, a.window_m or None, a.theta_limits, a.em_dir, a.coord_file):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
