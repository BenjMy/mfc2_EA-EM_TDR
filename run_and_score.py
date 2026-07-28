#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_and_score.py
=================
Runs the pipeline's ``run_pipeline.main()`` on a synthetic-real-mirror
dataset (see ``generate_synthetic_real_mirror.py``) and captures everything
needed to score the output against known ground truth -- WITHOUT modifying
any project file.

Why a monkey-patch instead of a project edit
---------------------------------------------
``run_pipeline.py``'s Step 4/5 result (``SelectionOutput`` -- holds
``theta_calc``, ``cv_map``, ``K``, ``rmse``, ``coverage``, ``best_fit``) is
only ever plotted, never written to disk. Rather than editing the project's
``run_pipeline.py`` (and having that edit drift out of sync with future
project changes), this script imports ``4_run_select_assess`` FIRST and
replaces its ``run_select_assess`` function with a thin wrapper that saves
the return value before handing it back unchanged. Because Python caches
imported modules by name (``sys.modules``), ``run_pipeline.py``'s own
``importlib.import_module("4_run_select_assess")`` call picks up the SAME,
already-patched module object -- so ``run_pipeline.main()`` runs completely
unmodified and unaware it's being observed.

The spatial reference-grid coordinates (``ref_x``/``ref_y``, needed to know
WHERE each ``theta_calc`` point actually is) are read back from the Step 1
ensemble cache pickle that ``run_pipeline.py`` already writes
(``cfg.cache_path``), which stores them under the ``"reference_xy"`` key for
any real-data time-lapse run.

Usage
-----
    python3 run_and_score.py \\
        --project-dir /path/to/pipeline/project \\
        --em-dir ./synthetic_real_mirror/EMI_MFC2_synthetic \\
        --tdr-dir ./synthetic_real_mirror/daily_MFC_synthetic \\
        --coord-file SOILNET_MFC2_coordinates_sensors.csv \\
        --scenario mfc2_real_synthetic_mirror \\
        --config ./synthetic_real_mirror/mfc2_real_synthetic_mirror.json \\
        --out ./pipeline_capture.npz

Output
------
An ``.npz`` file with keys: ``theta_calc``, ``cv_map``, ``coverage``,
``is_calibrated`` (per-horizon bool), ``K``, ``rmse``, ``ref_x``, ``ref_y``.
Feed this straight into ``score_against_truth.py``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import sys

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", required=True,
                     help="Directory containing run_pipeline.py, scenario_loader.py, etc.")
    ap.add_argument("--em-dir", required=True)
    ap.add_argument("--tdr-dir", required=True)
    ap.add_argument("--coord-file", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--config", required=True,
                     help="Path to the scenario JSON (e.g. mfc2_real_synthetic_mirror.json).")
    ap.add_argument("--mode", default="2d", choices=["1d", "2d"])
    ap.add_argument("--out", default="./pipeline_capture.npz")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.project_dir))

    # --- Patch 4_run_select_assess.run_select_assess BEFORE run_pipeline
    #     imports it, so the cached sys.modules entry run_pipeline picks up
    #     already carries the patched function. ---------------------------
    step4_5 = importlib.import_module("4_run_select_assess")
    captured: dict = {}
    _orig_run_select_assess = step4_5.run_select_assess

    def _capturing_run_select_assess(*a, **kw):
        out = _orig_run_select_assess(*a, **kw)
        captured["K"] = np.array(out.K)
        captured["rmse"] = np.array(out.rmse)
        captured["theta_calc"] = np.asarray(out.theta_calc)
        captured["cv_map"] = np.asarray(out.cv_map)
        if out.coverage is not None:
            captured["coverage"] = np.asarray(out.coverage)
        captured["is_calibrated"] = np.array(
            [h.is_calibrated for h in out.best_fit.horizon_fits]
        )
        print(f"[CAPTURE] Step 4/5 result intercepted: K={out.K}, "
              f"theta_calc shape={np.asarray(out.theta_calc).shape}, "
              f"calibrated horizons={captured['is_calibrated'].tolist()}")
        return out

    step4_5.run_select_assess = _capturing_run_select_assess

    scenario_loader = importlib.import_module("scenario_loader")
    run_pipeline = importlib.import_module("run_pipeline")

    cfg = scenario_loader.load_scenario(args.scenario, config_path=args.config)

    run_pipeline.main(
        cfg, mode=args.mode, data_source="real",
        em_dir=args.em_dir, tdr_dir=args.tdr_dir, coord_file=args.coord_file,
    )

    if "theta_calc" not in captured:
        raise RuntimeError(
            "run_pipeline.main() completed but Step 4/5 was never captured -- "
            "the monkey-patch didn't take effect. Check that --project-dir "
            "points at the directory containing 4_run_select_assess.py."
        )

    # --- Reference-grid coordinates from the Step 1 ensemble cache --------
    with open(cfg.cache_path, "rb") as f:
        cache = pickle.load(f)
    ref_xy = cache.get("reference_xy")
    if ref_xy is None:
        raise RuntimeError(
            "No 'reference_xy' in the ensemble cache -- this run wasn't "
            "real-data time-lapse (>1 EM epoch), so there is no shared "
            "spatial reference grid to score theta_calc against."
        )
    captured["ref_x"], captured["ref_y"] = ref_xy

    np.savez(args.out, **captured)
    print(f"\n[CAPTURE] Wrote {args.out}")
    print(f"[CAPTURE] Keys: {list(captured.keys())}")
    print(f"[CAPTURE] theta_calc shape: {captured['theta_calc'].shape}  "
          f"(n_time, n_points, n_layers)")
    print(f"[CAPTURE] reference grid size: {len(captured['ref_x'])} points")


if __name__ == "__main__":
    main()
