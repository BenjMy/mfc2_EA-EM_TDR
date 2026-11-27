#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_synthetic_truth.py -- stand-alone TRUTH maps of the synthetic mirror (the same figures the
pipeline writes during a run when the scenario sets ``ground_truth_dir``).

Writes ``truth_xy.png`` (theta, two surveys, and sigma_s), ``truth_xz.png`` (vertical section)
and ``truth_ws_params.png`` (all of a, m, sigma_w, sigma_s). Details: ``plots/synthetic_truth.py``.

Usage
-----
    python plot_synthetic_truth.py                          # mfc2_synthetic/, driest + wettest survey
    python plot_synthetic_truth.py --epochs 1 4 6           # chosen survey indices (1-7)
    python plot_synthetic_truth.py --y 4468350 --zmax 0.8   # xz section at northing y (m)
    python plot_synthetic_truth.py --gt-dir other/ground_truth --out-dir other/truth_plots

Needs emagpy (imported by the generator) and matplotlib.
"""
from __future__ import annotations

import argparse
import os
import sys

from plots.synthetic_truth import plot_truth_maps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", default="mfc2_synthetic/ground_truth", help="ground_truth folder of the mirror")
    ap.add_argument("--coord-file", default="mfc2_synthetic/standin_coordinates.csv",
                    help="TDR station coordinates (Pos;x;y); skipped if missing")
    ap.add_argument("--out-dir", default=None, help="default: <gt-dir>/../truth_plots")
    ap.add_argument("--epochs", nargs="+", type=int, default=None, metavar="N",
                    help="survey indices (1-7) to draw; default: the driest and the wettest")
    ap.add_argument("--y", type=float, default=None, help="northing (m) of the xz section; default: field centre")
    ap.add_argument("--band", type=float, default=15.0, help="stations within +-band m of the section are marked")
    ap.add_argument("--zmax", type=float, default=1.0, help="maximum depth of the xz section (m)")
    ap.add_argument("--nx", type=int, default=150, help="grid points along x")
    ap.add_argument("--ny", type=int, default=150, help="grid points along y (xy maps)")
    args = ap.parse_args(argv)
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.gt_dir)), "truth_plots")
    for p in plot_truth_maps(args.gt_dir, out_dir, coord_file=args.coord_file, epochs=args.epochs,
                             y_sec=args.y, band=args.band, zmax=args.zmax, nx=args.nx, ny=args.ny):
        print(f"[truth] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
