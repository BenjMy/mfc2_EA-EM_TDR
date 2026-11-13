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


def run_step6e_forward_eca(reporter, *, em_fnames, averaged_models, selected_output,
                           depths0, output_dir, x_transect, forward_model, cfg,
                           coils_list, matched_indices):
    """STEP 6e: FORWARD-SIMULATE ECa FROM THE OPTIMAL MODEL.

    The weighted-average EC model is forward-simulated (emagpy
    ``Problem.forward()``, mirroring data_synthetic.py's ground-truth
    generation) to obtain predicted ECa, compared with the raw survey ECa that
    was fed to the inversion: a direct forward-model check of the pipeline's
    final output. Records one ``Forward ECa RMSE (optimal, t=...)`` metric per
    survey time.

    ``coils_list``: ``step0.COILS_LIST`` for synthetic data, else None.
    ``matched_indices``: ``REAL_DATA_MATCHED_INDICES`` for real time-lapse runs,
    else None (synthetic and real single-survey runs, where the raw CSV row
    order already IS the model's order). INDEXING_AUDIT_REPORT.md finding #5:
    for real time-lapse data ``averaged_models[K-1]``'s spatial axis is the
    reference grid (``geometry_utils.match_epochs_to_reference_grid``), so the
    observed ECa must be gathered through the per-epoch row -> reference-grid
    map in the SAME order.

    Returns the list of ``(time_index, rmse_mS_per_m)`` pairs.
    """
    reporter.step("Step 6e", "Forward-simulate ECa from optimal model")
    reporter.log("\n" + "=" * 80)
    reporter.log(" [PIPELINE] STEP 6e: FORWARD-SIMULATE ECa FROM OPTIMAL MODEL")
    reporter.log("=" * 80)

    # INDEXING_AUDIT_REPORT.md finding #5: for real-data, time-lapse
    # runs, averaged_models[K-1]'s spatial axis is the reference grid
    # (geometry_utils.match_epochs_to_reference_grid), not any one
    # epoch's own raw-CSV row order/count. forward_simulate_eca_optimal
    # forward-simulates ECa from that reference-grid-ordered model, so
    # the observed ECa it's compared against must be gathered in the
    # SAME reference-grid order -- REAL_DATA_MATCHED_INDICES (built
    # alongside REAL_DATA_REFERENCE_XY, see Step 1 above) is exactly
    # that per-epoch row->reference-grid-index map, and
    # plot_solution_diagnostics.forward_simulate_eca_optimal now
    # accepts it and gathers eca_obs/xpos through it accordingly.
    #
    # None for synthetic data and for real single-survey (non-time-
    # lapse) runs, where raw CSV order already IS the model's own
    # order and no remap ever happened -- forward_simulate_eca_optimal
    # keeps reading raw CSV rows directly in that case.
    eca_forward_results = psol.forward_simulate_eca_optimal(
        csv_paths       = em_fnames,
        averaged_models = averaged_models,
        selected_output = selected_output,
        depths0         = depths0,
        output_dir      = output_dir,
        coils_list      = coils_list,
        x_transect      = x_transect,
        forward_model   = forward_model,
        scenario_name   = cfg.name if cfg else "default",
        matched_indices = matched_indices,
    )
    for _t_idx, _rmse_eca in eca_forward_results:
        reporter.metric(f"Forward ECa RMSE (optimal, t={_t_idx})", f"{_rmse_eca:.4f} mS/m")

    reporter.end_step("Step 6e")
    return eca_forward_results
