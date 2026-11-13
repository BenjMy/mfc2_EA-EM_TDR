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


def run_step6_water_content(reporter, *, mode, data_source, selected_output, x_transect,
                            depths0, probe_x_coords, probe_y_coords, probe_depths,
                            probe_x_coords_utm, probe_y_coords_utm, probe_labels, utm_x, utm_y,
                            output_dir, output_model_dir, scenario_name, timelapse,
                            xy_render_mode, window_m, theta_true_realization,
                            realization_index, cond_ranges, coils_list, forward_model):
    """STEP 6: water-content XZ (always) / XY (2-D) maps; synthetic runs add the truth maps,
    calculated-vs-truth MAE/RMSE maps (metrics logged to the report) and the initial-model-vs-truth
    EC_bulk / ECa diagnostics. Opens "Step 6"; the caller closes it after Steps 6b-6e.
    """
    reporter.step("Step 6", "Rendering water content visualizations")
    reporter.log("\n" + "=" * 80)
    reporter.log(" [PIPELINE] STEP 6: RENDERING WATER CONTENT VISUALIZATIONS")
    reporter.log(f"            XZ cross-section : always")
    reporter.log(f"            XY plan view      : {'YES (2-D mode)' if mode == '2d' else 'NO (1-D mode)'}")
    reporter.log("=" * 80)

    pwc.plot_from_pipeline(
        selected_output=selected_output,
        x_transect=x_transect,
        depths0=depths0,
        probe_x_coords=probe_x_coords,
        probe_y_coords=probe_y_coords,   # local frame -- XZ probe markers only
        probe_depths=probe_depths,        # Z axis (unchanged)
        output_dir=output_dir,
        scenario_name=scenario_name,
        timelapse=timelapse,
        survey_labels=None,
        utm_x=utm_x,
        utm_y=utm_y,
        # FIX: without these, plot_from_pipeline's own fallback treated
        # the LOCAL probe_x_coords/probe_y_coords above as if they were
        # already absolute UTM, mislocating every probe marker on
        # water_content_xy_{scenario}.png relative to the real UTM
        # survey grid. probe_x_coords_utm/probe_y_coords_utm are the
        # correct absolute-UTM counterparts computed earlier.
        probe_utm_x=probe_x_coords_utm,
        probe_utm_y=probe_y_coords_utm,
        # Real TDR station names in real-data mode (overridden above),
        # synthetic "P0","P1",... labels otherwise.
        probe_ids=probe_labels,
        xy_render_mode=xy_render_mode,
        # Auto-sizes the extra water_content_xy_{scenario}_imshow.png
        # raster's pixel size to 2 * window_m -- see plot_from_pipeline.
        window_m=window_m,
    )

    # Ground-truth counterpart of the plot above (water_content_xz_<scenario>_truth.png
    # / water_content_xy_<scenario>_truth.png) -- only available in synthetic mode,
    # where theta_true_realization is the known ground truth used to generate the
    # synthetic surveys. No truth exists for real field data, so this is skipped there.
    if data_source == "synthetic" and theta_true_realization is not None:
        try:
            pwc.plot_truth_from_pipeline(
                theta_true=theta_true_realization,
                x_transect=x_transect,
                depths0=depths0,
                probe_x_coords=probe_x_coords,
                probe_y_coords=probe_y_coords,
                probe_depths=probe_depths,
                output_dir=output_dir,
                scenario_name=scenario_name,
                timelapse=timelapse,
                survey_labels=None,
                utm_x=utm_x,
                utm_y=utm_y,
                # FIX: same probe_utm_x/y fix as plot_from_pipeline above --
                # otherwise water_content_xy_{scenario}_truth.png mislocates
                # every probe marker the same way.
                probe_utm_x=probe_x_coords_utm,
                probe_utm_y=probe_y_coords_utm,
                probe_ids=probe_labels,
                xy_render_mode=xy_render_mode,
                window_m=window_m,
            )
            reporter.log(
                f"[STEP 6] Ground-truth XZ (and XY, if 2-D) water-content "
                f"plots -> {output_dir}/water_content_xz_"
                f"{scenario_name}_truth.png"
            )
        except Exception as e:
            reporter.log(f"[STEP 6][WARN] plot_truth_from_pipeline failed: {e}")

        # Calculated-vs-truth error maps: MAE and RMSE (%VWC), XZ always
        # and XY too in 2-D mode -- these supersede the old signed
        # %-difference plot (water_content_xy_{scenario}_pctdiff.png),
        # which is undefined/blows up wherever truth ~ 0. Also logs the
        # pooled overall MAE/RMSE into the report's Metrics Summary so
        # compare_scenarios.py can pull them into scenario comparisons.
        try:
            error_metrics = pwc.plot_error_metrics_from_pipeline(
                selected_output=selected_output,
                theta_true=theta_true_realization,
                x_transect=x_transect,
                depths0=depths0,
                probe_x_coords=probe_x_coords,
                probe_y_coords=probe_y_coords,
                probe_depths=probe_depths,
                output_dir=output_dir,
                scenario_name=scenario_name,
                timelapse=timelapse,
                survey_labels=None,
                utm_x=utm_x,
                utm_y=utm_y,
                probe_utm_x=probe_x_coords_utm,
                probe_utm_y=probe_y_coords_utm,
                xy_render_mode="imshow",
                window_m=window_m,
            )
            reporter.log(f"[STEP 6] Calculated-vs-truth XZ RMSE map -> {error_metrics['xz_rmse_path']}")
            reporter.log(f"[STEP 6] Calculated-vs-truth XZ MAE map  -> {error_metrics['xz_mae_path']}")
            reporter.metric("Water Content MAE vs Truth (XZ)", f"{error_metrics['xz_mae_overall']:.3f} %VWC")
            reporter.metric("Water Content RMSE vs Truth (XZ)", f"{error_metrics['xz_rmse_overall']:.3f} %VWC")
            if "xy_rmse_path" in error_metrics:
                reporter.log(f"[STEP 6] Calculated-vs-truth XY RMSE map -> {error_metrics['xy_rmse_path']}")
                reporter.log(f"[STEP 6] Calculated-vs-truth XY MAE map  -> {error_metrics['xy_mae_path']}")
                reporter.metric("Water Content MAE vs Truth (XY)", f"{error_metrics['xy_mae_overall']:.3f} %VWC")
                reporter.metric("Water Content RMSE vs Truth (XY)", f"{error_metrics['xy_rmse_overall']:.3f} %VWC")
        except Exception as e:
            reporter.log(f"[STEP 6][WARN] plot_error_metrics_from_pipeline failed: {e}")

        # Initial model (starting guess, conds0's layer MIN/MAX bounds) vs.
        # ground truth -- a QC check for whether the inversion's starting
        # point plausibly brackets reality, BEFORE any inversion runs.
        # Two flavours: depth-resolved EC_bulk (XZ heatmap, same style as
        # water_content_xz_*.png) and per-coil apparent conductivity (ECa),
        # which isn't depth-resolved so gets its own per-coil panel style.
        try:
            _ec_true_path = _resolve_npy_path("synthetic_time_lapse_ec.npy", output_model_dir, output_dir)
            _eca_true_path = _resolve_npy_path("synthetic_time_lapse_eca.npy", output_model_dir, output_dir)

            if os.path.exists(_ec_true_path):
                _ec_bulk_true = np.load(_ec_true_path)[realization_index]
                ini_ec_path = pwc.plot_initial_model_vs_true_ec(
                    cond_ranges=cond_ranges,
                    depths0=depths0,
                    x_coords=x_transect,
                    ec_bulk_true=_ec_bulk_true,
                    output_dir=output_dir,
                    scenario_name=scenario_name,
                    probe_x=probe_x_coords,
                    probe_depths=probe_depths,
                )
                reporter.log(f"[STEP 6] Initial model vs. true EC_bulk -> {ini_ec_path}")
            else:
                reporter.log(
                    "[STEP 6][WARN] 'synthetic_time_lapse_ec.npy' not found; "
                    "skipping initial-model-vs-true-EC_bulk plot."
                )

            if os.path.exists(_eca_true_path):
                _eca_true = np.load(_eca_true_path)[realization_index]
                ini_eca_path = pwc.plot_initial_eca_vs_true_eca(
                    cond_ranges=cond_ranges,
                    depths0=depths0,
                    x_coords=x_transect,
                    eca_true=_eca_true,
                    coils=coils_list,
                    output_dir=output_dir,
                    scenario_name=scenario_name,
                    forward_model=forward_model,
                )
                if ini_eca_path is not None:
                    reporter.log(f"[STEP 6] Initial model vs. true ECa -> {ini_eca_path}")
            else:
                reporter.log(
                    "[STEP 6][WARN] 'synthetic_time_lapse_eca.npy' not found; "
                    "skipping initial-model-vs-true-ECa plot."
                )
        except Exception as e:
            reporter.log(f"[STEP 6][WARN] Initial-model-vs-truth diagnostics failed: {e}")
