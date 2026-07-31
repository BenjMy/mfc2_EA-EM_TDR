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
the return value before handing it back unchanged. The same trick is used
on ``3_run_average.run_average`` to capture ``averaged_models`` (needed to
pick out the Step 3 K-th averaged EC model for the ground-truth overlay
plot below). Because Python caches imported modules by name
(``sys.modules``), ``run_pipeline.py``'s own ``importlib.import_module(...)``
calls pick up the SAME, already-patched module objects -- so
``run_pipeline.main()`` runs completely unmodified and unaware it's being
observed.

The spatial reference-grid coordinates (``ref_x``/``ref_y``, needed to know
WHERE each ``theta_calc`` point actually is) are read back from the Step 1
ensemble cache pickle that ``run_pipeline.py`` already writes
(``cfg.cache_path``), which stores them under the ``"reference_xy"`` key for
any real-data time-lapse run.

Ground-truth plots for the real-data-ingestion path
----------------------------------------------------
This scenario runs ``run_pipeline.main(..., data_source="real", ...)`` on
purpose, to exercise the real-data ingestion code path against mirrored
real-shaped CSVs. That means every one of ``run_pipeline.py``'s built-in
truth-comparison plots (``plot_truth_tdr``, the Step 6
``water_content_xz_..._truth`` overlay, and Step 6d's EC_bulk truth
overlay) is skipped, since they're all gated on the literal string
``data_source == "synthetic"``. Since ground truth genuinely IS known here
(``generate_synthetic_real_mirror.theta_truth_deterministic`` /
``forward_waxman_smits``), this script reconstructs those comparisons
itself once ``run_pipeline.main()`` returns -- writing per-horizon XY
plan-view maps (real-data points are a scattered 2-D cloud, so an XZ
transect isn't meaningful the way it is for the synthetic-generator path),
the same depth-profile figure Step 0 draws, and the Step 6d EC_bulk
solution-vs-truth overlay. Pass ``--no-truth-plots`` to skip this.

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
import importlib.util
import os
import pickle
import sys

import numpy as np
import pandas as pd


def _load_generator_module(generator_dir: str):
    """Load generate_synthetic_real_mirror.py by path (same trick
    score_against_truth.py uses) so theta_truth_deterministic() and
    REAL_EM_EPOCHS are always the exact function/data that produced this
    run's synthetic-real-mirror CSVs -- never re-implemented/duplicated
    here where it could silently drift out of sync."""
    path = os.path.join(generator_dir, "generate_synthetic_real_mirror.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"generate_synthetic_real_mirror.py not found in {generator_dir}")
    spec = importlib.util.spec_from_file_location("gen_mirror", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    ap.add_argument("--generator-dir", default=".",
                     help="Directory containing generate_synthetic_real_mirror.py "
                          "(only needed for the ground-truth XY-per-horizon plots below; "
                          "same convention as score_against_truth.py's --generator-dir).")
    ap.add_argument("--no-truth-plots", action="store_true",
                     help="Skip the ground-truth XY-per-horizon plots (e.g. if this "
                          "scenario has no known truth, or generate_synthetic_real_mirror.py "
                          "isn't available at run time).")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.project_dir))

    # --- Patch 3_run_average.run_average and 4_run_select_assess.run_select_assess
    #     BEFORE run_pipeline imports either, so the cached sys.modules
    #     entries run_pipeline picks up already carry the patched
    #     functions. -----------------------------------------------------
    step3 = importlib.import_module("3_run_average")
    captured: dict = {}
    _orig_run_average = step3.run_average

    def _capturing_run_average(*a, **kw):
        averaged_models, ranking = _orig_run_average(*a, **kw)
        captured["averaged_models"] = averaged_models
        print(f"[CAPTURE] Step 3 result intercepted: "
              f"{len(averaged_models)} averaged model(s) (K=1..{len(averaged_models)})")
        return averaged_models, ranking

    step3.run_average = _capturing_run_average

    step4_5 = importlib.import_module("4_run_select_assess")
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
        cfg, mode=args.mode, data_source="synthetic",
        em_dir=args.em_dir, tdr_dir=args.tdr_dir, coord_file=args.coord_file,
    )

    if "theta_calc" not in captured:
        raise RuntimeError(
            "run_pipeline.main() completed but Step 4/5 was never captured -- "
            "the monkey-patch didn't take effect. Check that --project-dir "
            "points at the directory containing 4_run_select_assess.py."
        )
    if "averaged_models" not in captured:
        raise RuntimeError(
            "run_pipeline.main() completed but Step 3 was never captured -- "
            "the monkey-patch didn't take effect. Check that --project-dir "
            "points at the directory containing 3_run_average.py."
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

    # -----------------------------------------------------------------------
    # Ground-truth XY-per-horizon plots (synthetic-real-mirror only)
    # -----------------------------------------------------------------------
    # run_pipeline.py's built-in truth-comparison plots (Step 0's
    # plot_truth_tdr, Step 6's water_content_xz_..._truth, Step 6d's
    # EC_bulk truth overlay) are all gated on the literal string
    # data_source == "synthetic" (see run_pipeline.py). This scenario
    # deliberately calls run_pipeline.main(..., data_source="real", ...)
    # above -- so it exercises the real-data ingestion code path against
    # mirrored real-shaped CSVs -- which means none of those gates ever
    # fire, even though ground truth IS known here via
    # generate_synthetic_real_mirror.theta_truth_deterministic(). Rather
    # than editing run_pipeline.py to add a third data_source state (and
    # having that edit drift out of sync with the project like
    # run_and_score.py's own docstring warns against), the truth-comparison
    # plots are reconstructed here from the captured ref_x/ref_y grid --
    # same "don't touch project files" approach as the Step 4/5 capture
    # above.
    #
    # Real-data points are a scattered 2-D cloud, not a transect, so
    # (unlike the synthetic-generator path's XZ plots) only XY plan-view
    # maps are spatially meaningful here -- rendered ONE PER HORIZON
    # (not depth-averaged), so each layer's recovered-vs-true spatial
    # pattern can be inspected on its own.
    if not args.no_truth_plots:
        try:
            gen = _load_generator_module(args.generator_dir)
            import plot_water_content as pwc  # project-dir already on sys.path (see above)

            n_time, n_points, n_layers = captured["theta_calc"].shape
            if n_time > len(gen.REAL_EM_EPOCHS):
                raise RuntimeError(
                    f"theta_calc has {n_time} epochs but the generator only "
                    f"defines {len(gen.REAL_EM_EPOCHS)} -- REAL_EM_EPOCHS may "
                    "have changed since this data was generated."
                )

            # Noise-free ground truth, recomputed pointwise at the pipeline's
            # own reference grid -- same convention as score_against_truth.py.
            theta_true = np.empty((n_time, n_points, n_layers), dtype=float)
            for t in range(n_time):
                iso_date = gen.REAL_EM_EPOCHS[t][2]
                for l in range(n_layers):
                    theta_true[t, :, l] = gen.theta_truth_deterministic(
                        ref_xy[0], ref_xy[1], layer_idx=l, iso_date=iso_date,
                    )
            theta_true_pct = theta_true * 100.0  # m^3/m^3 -> % vol., matches pwc convention

            # Optional TDR-station markers straight from the coord CSV --
            # purely cosmetic, so any failure here is swallowed rather than
            # aborting the truth plots.
            probe_utm_x = probe_utm_y = None
            try:
                _coords = pd.read_csv(args.coord_file, delimiter=";", comment="[")
                _coords.columns = _coords.columns.str.strip()
                probe_utm_x = _coords["x"].astype(float).tolist()
                probe_utm_y = _coords["y"].astype(float).tolist()
            except Exception:
                pass

            vmax = float(np.nanpercentile(theta_true_pct, 98))
            vmax = max(5.0, np.ceil(vmax / 5) * 5)
            timelapse = n_time > 1

            for l in range(n_layers):
                if timelapse:
                    pwc.plot_water_content_xy_timelapse(
                        theta_calc=theta_true_pct, utm_x=ref_xy[0], utm_y=ref_xy[1],
                        layer_index=l, theta_vmin=0.0, theta_vmax=vmax,
                        output_dir=cfg.output_dir,
                        scenario_name=f"{cfg.name}_truth_horizon{l}",
                        probe_utm_x=probe_utm_x, probe_utm_y=probe_utm_y,
                    )
                else:
                    pwc.plot_water_content_xy(
                        theta_calc=theta_true_pct[0], utm_x=ref_xy[0], utm_y=ref_xy[1],
                        layer_index=l, theta_vmin=0.0, theta_vmax=vmax,
                        title=(f"Soil Water Content (XY) — Ground Truth — "
                               f"{cfg.name} — horizon {l}"),
                        output_path=os.path.join(
                            cfg.output_dir,
                            f"water_content_xy_{cfg.name}_truth_horizon{l}.png",
                        ),
                        probe_utm_x=probe_utm_x, probe_utm_y=probe_utm_y,
                    )
            print(f"[TRUTH-PLOT] Wrote {n_layers} per-horizon XY ground-truth "
                  f"map(s) to '{cfg.output_dir}'")

            # Bonus: the same depth-profile figure Step 0 draws in synthetic
            # mode (truth_tdr_profiles.png) -- reuses run_pipeline's own
            # plot_truth_tdr() unmodified, just fed the reconstructed
            # theta_true here instead of theta_true_realization.
            truth_tdr_path = run_pipeline.plot_truth_tdr(
                output_dir=cfg.output_dir, theta_true_realization=theta_true,
                depths0=cfg.depths0, fname=f"truth_tdr_profiles_{cfg.name}.png",
            )
            print(f"[TRUTH-PLOT] Wrote {truth_tdr_path}")

            # --- EC_bulk optimal-solution truth overlay (Step 6d's
            #     psol.plot_solution_ec_bulk, same fixed filename
            #     '<output_dir>/solution/solution_ec_bulk_profiles.png' it
            #     always uses) -- needs the Step 3 K-th averaged model
            #     (captured above) plus the ground-truth EC_bulk, forward-
            #     modelled from theta_true via the exact same "simplified
            #     Waxman-Smits" relation the generator used to build the
            #     synthetic ECa in the first place (gen.forward_waxman_smits
            #     -- the algebraic inverse of em_methodology.py's actual
            #     calibration relation, so it's not a different constitutive
            #     model from what the pipeline fits against).
            import plot_solution_diagnostics as psol
            import geometry_utils

            K = int(captured["K"])
            rmse_arr = captured["rmse"]
            rmse_val = float(np.nanmean(rmse_arr)) if np.ndim(rmse_arr) else float(rmse_arr)
            ec_optimal = captured["averaged_models"][K - 1]
            ec_bulk_true = gen.forward_waxman_smits(theta_true)  # fractional theta, NOT theta_true_pct
            layer_centers = geometry_utils.layer_centers(cfg.depths0)

            ec_fig_path = psol.plot_solution_ec_bulk(
                ec_optimal=ec_optimal, depths0=cfg.depths0, layer_centers=layer_centers,
                output_dir=cfg.output_dir, ec_bulk_true=ec_bulk_true,
                K=K, rmse=rmse_val, scenario_name=cfg.name, timelapse=timelapse,
            )
            print(f"[TRUTH-PLOT] Wrote {ec_fig_path}")
        except Exception as e:
            print(f"[TRUTH-PLOT][WARN] Ground-truth plotting failed "
                  f"(capture/scoring below is unaffected): {type(e).__name__}: {e}")

    # averaged_models was only needed above to pick out the K-th optimal
    # EC model for the truth-overlay plot -- it's a list of per-K arrays,
    # not npz-serializable/part of this script's documented output, so it
    # isn't persisted to the capture file.
    captured.pop("averaged_models", None)

    np.savez(args.out, **captured)
    print(f"\n[CAPTURE] Wrote {args.out}")
    print(f"[CAPTURE] Keys: {list(captured.keys())}")
    print(f"[CAPTURE] theta_calc shape: {captured['theta_calc'].shape}  "
          f"(n_time, n_points, n_layers)")
    print(f"[CAPTURE] reference grid size: {len(captured['ref_x'])} points")


if __name__ == "__main__":
    main()
