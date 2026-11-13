"""Pipeline Step 6 plot blocks, one function per step (Phase 6 of refactoring.md).

Each function takes what it reads as explicit arguments and returns what the
rest of the pipeline may need; ``run_pipeline.main`` only calls it. No import
of ``run_pipeline`` here (it would be circular).
"""
from __future__ import annotations

import os

import numpy as np

import plot_solution_diagnostics as psol
from steps._common import resolve_npy_path as _resolve_npy_path


def run_step6d_solution_plots(reporter, *, data_source, theta_true_realization,
                              output_model_dir, output_dir, realization_index,
                              selected_output, averaged_models, depths0,
                              layer_centers, cfg, timelapse):
    """STEP 6d: SOLUTION SUMMARY PLOTS (SWC + EC_bulk, synthetic truth overlaid).

    Renders the final solution (RMSE-weighted top-K averaged model selected in
    Step 5) as vertical soil-water-content and bulk-EC profiles. In synthetic
    mode the known ground truth (``theta_true_realization`` /
    ``synthetic_time_lapse_ec.npy``) is overlaid as a dashed reference; real
    runs get the same plots without it.

    Returns ``(swc_fig_path, ec_fig_path)``.
    """
    reporter.step("Step 6d", "Solution SWC / EC_bulk summary plots")
    reporter.log("\n" + "=" * 80)
    reporter.log(" [PIPELINE] STEP 6d: SOLUTION SUMMARY PLOTS (SWC + EC_bulk)")
    reporter.log("=" * 80)

    theta_true_for_plot   = None
    ec_bulk_true_for_plot = None
    if data_source == "synthetic":
        theta_true_for_plot = theta_true_realization
        _ec_true_path = _resolve_npy_path("synthetic_time_lapse_ec.npy", output_model_dir, output_dir)
        if os.path.exists(_ec_true_path):
            ec_bulk_true_for_plot = np.load(_ec_true_path)[realization_index]
        else:
            reporter.log(
                "[STEP 6d][WARN] 'synthetic_time_lapse_ec.npy' not found in "
                f"'{output_dir}' — EC_bulk ground-truth overlay skipped."
            )

    swc_fig_path = psol.plot_solution_swc(
        theta_calc      = selected_output.theta_calc,
        depths0         = depths0,
        layer_centers   = layer_centers,
        output_dir      = output_dir,
        theta_true      = theta_true_for_plot,
        K               = selected_output.K,
        rmse            = selected_output.rmse,
        scenario_name   = cfg.name if cfg else "default",
        timelapse       = timelapse,
    )
    reporter.log(f"[STEP 6d] Solution SWC profiles      → {swc_fig_path}")

    ec_fig_path = psol.plot_solution_ec_bulk(
        ec_optimal      = averaged_models[selected_output.K - 1],
        depths0         = depths0,
        layer_centers   = layer_centers,
        output_dir      = output_dir,
        ec_bulk_true    = ec_bulk_true_for_plot,
        K               = selected_output.K,
        rmse            = selected_output.rmse,
        scenario_name   = cfg.name if cfg else "default",
        timelapse       = timelapse,
    )
    reporter.log(f"[STEP 6d] Solution EC_bulk profiles  → {ec_fig_path}")

    reporter.end_step("Step 6d")
    return swc_fig_path, ec_fig_path
