"""Pipeline Step 6 plot blocks, one function per step (Phase 6 of refactoring.md).

Each function takes what it reads as explicit arguments and returns what the
rest of the pipeline may need; ``run_pipeline.main`` only calls it. No import
of ``run_pipeline`` here (it would be circular).
"""
from __future__ import annotations

import os

import numpy as np

import plot_emagpy_diagnostics as pediag
import plot_solution_diagnostics as psol
import plot_water_content as pwc
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


def run_step6c_swc_stage_plots(reporter, *, models_clean, averaged_models, selected_output,
                               fits_step4, x_transect, depths0, output_dir, cfg, timelapse,
                               utm_x, utm_y, probe_x_coords, probe_y_coords, probe_depths,
                               theta_obs, sample_positions, sample_times,
                               probe_layer_indices):
    """STEP 6c: SWC plots for all three pipeline stages (+ 6c-bis 1:1 scatter).

    Stage A: sample of raw realisations (``models_clean``); stage B: averaged
    model at K=1, K_opt, K_max (``averaged_models``); stage C: optimal model
    (``selected_output``). 6c-bis passes the full ``sample_positions`` /
    ``sample_times`` / ``probe_layer_indices`` vectors explicitly (they are
    not stored inside ``selected_output``) instead of relying on the fallback
    reconstruction inside ``plot_swc_ensemble_stages``.
    """
    reporter.step("Step 6c", "SWC plots for all pipeline stages")
    reporter.log(" [PIPELINE] STEP 6c: SWC STAGE PLOTS (A/B/C)")

    pwc.plot_swc_ensemble_stages(
        models_clean     = models_clean,
        averaged_models  = averaged_models,
        selected_output  = selected_output,
        best_fit         = fits_step4[selected_output.K - 1],
        x_transect       = x_transect,
        depths0          = depths0,
        output_dir       = output_dir,
        scenario_name    = cfg.name if cfg else "default",
        timelapse        = timelapse,
        survey_labels    = None,
        utm_x            = utm_x,
        utm_y            = utm_y,
        probe_x_coords   = probe_x_coords,
        probe_y_coords   = probe_y_coords,
        probe_depths     = probe_depths,
        max_realisations = 4,
    )

    # -----------------------------------------------------------------------
    # STEP 6c-bis — 1:1 validation scatter (Stages A / B / C vs. TDR truth)
    # Called explicitly here so that the full sample_positions / sample_times /
    # probe_layer_indices vectors — which are available in this scope but are
    # not stored inside selected_output — are passed directly instead of
    # relying on the fallback reconstruction inside plot_swc_ensemble_stages.
    # -----------------------------------------------------------------------
    pwc.plot_one_to_one_stages(
        models_clean      = models_clean,
        averaged_models   = averaged_models,
        selected_output   = selected_output,
        best_fit          = fits_step4[selected_output.K - 1],
        theta_obs         = theta_obs,
        sample_positions  = sample_positions,
        sample_times      = sample_times,
        output_dir        = output_dir,
        scenario_name     = cfg.name if cfg else "default",
        probe_layer_indices = probe_layer_indices,
    )

    reporter.end_step("Step 6c")


def run_step6b_emagpy_plots(reporter, *, em_fnames, mode, selected_output, models_clean,
                            averaged_models, tasks, depths_clean, depths0, output_dir,
                            fixed_depths, invert_kwargs, em_unit):
    """STEP 6b: emagpy native diagnostic plots.

    (a) per-realisation: raw inversion for a sample of ensemble members;
    (b) optimal model: weighted average of the top-K realisations (Step 4/5).
    ``tasks`` are the Step 1 inversion tasks; ``invert_kwargs`` the base
    inversion options (alpha, beta, gamma, method, forwardModel, bnds).
    """
    reporter.step("Step 6b", "emagpy native diagnostic plots")
    reporter.log(" [PIPELINE] STEP 6b: emagpy DIAGNOSTIC PLOTS")
    reporter.log(f"            surveys        : {len(em_fnames)}")
    reporter.log(f"            mode           : {mode.upper()}")
    reporter.log(f"            optimal K      : {selected_output.K}")
    reporter.log(f"            ensemble size  : {models_clean.shape[0]}")

    # (a) Per-realisation plots — a uniform sample of individual inversions
    pediag.run_emagpy_plots_per_realisation(
        csv_paths        = em_fnames,
        tasks            = tasks,            # built in Step 1
        models_clean     = models_clean,
        depths_clean     = depths_clean,
        depths0          = depths0,
        output_dir       = output_dir,
        fixed_depths     = fixed_depths,
        mode             = mode,
        max_realisations = 1,                # cap: 1 sample realisation rendered (was 4 —
                                               # each realisation re-inverts the full survey
                                               # from scratch per time step; with real-data
                                               # point counts of 20k-47k this was the main
                                               # source of multi-hour Step 6b runtimes)
        em_unit          = em_unit,
    )

    # (b) Optimal model plots — weighted average of top-K realisations
    pediag.run_emagpy_plots_optimal(
        csv_paths        = em_fnames,
        selected_output  = selected_output,  # SelectionOutput from Step 4/5
        averaged_models  = averaged_models,  # list from Step 3
        depths0          = depths0,
        output_dir       = output_dir,
        fixed_depths     = fixed_depths,
        conds0           = None,
        invert_kwargs    = invert_kwargs,
        mode             = mode,
        em_unit          = em_unit,
    )

    reporter.end_step("Step 6b")
