#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_synthetic_theta.py
===========================
Runs ``run_pipeline.main()`` on a **synthetic** scenario and saves the
Step 4/5 result (``theta_calc``, ``cv_map``, ``K``, ``rmse``, ``coverage``)
straight into that run's ``output_model/`` folder, next to the
``synthetic_*`` truth arrays that ``0_generate_synthetic_data.py`` already
puts there.

Why this exists
----------------
``run_pipeline.py`` never writes the Step 4/5 ``SelectionOutput`` to disk on
its own -- ``theta_calc`` is only ever plotted, then discarded (confirmed by
grepping the project for ``np.save(``: the only array ``run_pipeline.py``
itself persists is ``mcmc_conductivity_pstd.npy``). ``run_and_score.py``
solves this the same way (monkey-patching ``3_run_average``/
``4_run_select_assess`` before ``run_pipeline`` imports them), but it's
built specifically around the real-data-ingestion mirror scenario: it
requires a ``reference_xy`` entry in the Step-1 ensemble cache, which is
only populated on the real-data time-lapse code path. A plain synthetic
run has no such cache entry, so ``run_and_score.py`` raises
``RuntimeError`` on it -- which is almost certainly why ``pipeline_capture.npz``
stopped showing up once you moved to plain synthetic scenarios.

For synthetic scenarios the point positions are already known *before*
the run even starts (``synthetic_x_coords.npy`` / ``synthetic_y_coords.npy``,
written by ``0_generate_synthetic_data.py``), so no reference-grid capture
is needed at all -- this script just saves theta_calc etc. next to them.

Output (written into ``<output_dir>/output_model/``)
------------------------------------------------------
    theta_calc.npy   (n_time, n_points, n_layers)  -- final inverted model
    cv_map.npy
    K.npy            -- scalar, optimal stack depth
    rmse.npy
    coverage.npy     (only if the pipeline computed it)
    is_calibrated.npy (per-horizon bool)

These sit alongside the ``synthetic_time_lapse_theta.npy`` /
``synthetic_x_coords.npy`` / ``synthetic_y_coords.npy`` /
``synthetic_layer_centers.npy`` files already written by
``0_generate_synthetic_data.py`` before the pipeline ran, on the same
point grid -- ``plot_truth_comparison.py`` reads both straight out of
``output_model/``.

Usage
-----
    python3 capture_synthetic_theta.py \\
        --project-dir . \\
        --scenario moisture_2d_free \\
        --config ensemble_scenarios.json \\
        --mode 2d

Any extra ``run_pipeline.main()`` keyword can be passed through with
repeated ``--set key=value`` flags (values are parsed with
``ast.literal_eval`` when possible, else kept as a string), e.g.:

    --set homogeneous=True --set gamma=0.1
"""

from __future__ import annotations

import argparse
import ast
import importlib
import os
import sys

import numpy as np


def _parse_set(items: list[str] | None) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        try:
            out[key.strip()] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key.strip()] = raw
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", required=True,
                     help="Directory containing run_pipeline.py, scenario_loader.py, etc.")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--config", required=True,
                     help="Path to the scenario JSON (e.g. ensemble_scenarios.json).")
    ap.add_argument("--mode", default="1d", choices=["1d", "2d"])
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                     help="Extra run_pipeline.main() kwarg as key=value; repeatable.")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.project_dir))

    # --- Patch Step 3 / Step 4-5 BEFORE run_pipeline imports either, same
    #     trick as run_and_score.py, so run_pipeline's own
    #     importlib.import_module(...) calls pick up these patched
    #     module objects via sys.modules. -----------------------------
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

    kwargs = _parse_set(args.overrides)
    kwargs.setdefault("data_source", "synthetic")
    run_pipeline.main(cfg, mode=args.mode, **kwargs)

    if "theta_calc" not in captured:
        raise RuntimeError(
            "run_pipeline.main() completed but Step 4/5 was never captured -- "
            "the monkey-patch didn't take effect. Check that --project-dir "
            "points at the directory containing 4_run_select_assess.py."
        )

    output_model_dir = os.path.join(cfg.output_dir, "output_model")
    os.makedirs(output_model_dir, exist_ok=True)
    for key, arr in captured.items():
        path = os.path.join(output_model_dir, f"{key}.npy")
        np.save(path, arr)
        print(f"[CAPTURE] Wrote {path}  shape={np.shape(arr)}")

    print(f"\n[CAPTURE] Done -- theta_calc.npy etc. now live in {output_model_dir}")
    print("[CAPTURE] Run plot_truth_comparison.py against this output_dir to "
          "compare against synthetic_time_lapse_theta.npy.")


if __name__ == "__main__":
    main()
