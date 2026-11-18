"""Pipeline Step 5: convergence selection & reliability mapping. Phase 6 of refactoring.md.

Four blocks moved verbatim out of ``run_pipeline.main()``:

* ``run_step5_select``          -- opens "Step 5" and runs ``step4_select.run_select_assess``
* ``report_step5_selection``    -- log K, RMSE, CV-map and theta_calc statistics
* ``persist_step5_result``      -- save theta_calc / cv_map / K / rmse / ... to ``output_model/``
* ``run_step5_quality_gate``    -- post-Step-5 quality gate (never aborts the run)

The ``--stage`` hand-over and Step 5c stay in ``main()`` between the first and
second block (they need many run-level variables). ``main()`` closes "Step 5"
itself. No import of ``run_pipeline`` (it would be circular).
"""
from __future__ import annotations

import os

import numpy as np

import em_methodology as EMmethod
import step4_select as step4_5


def run_step5_select(reporter, *, models_clean, averaged_models, ranking, sample_positions,
                     sample_times, theta_obs, fits_step4, depths0, output_dir,
                     probe_layer_indices):
    """STEP 5: select the optimal stack depth K and map reliability (CV).

    Returns the ``SelectionOutput`` of ``step4_select.run_select_assess``.
    ``fits_step4`` are the Step 4 fits (passed as ``fits_step2=`` downstream).
    """
    reporter.step("Step 5", "Convergence selection & reliability mapping")
    reporter.log("\n" + "=" * 80)
    reporter.log(" [PIPELINE] STEP 5: CONVERGENCE SELECTION & RELIABILITY MAPPING")
    reporter.log("=" * 80)

    return step4_5.run_select_assess(
        models=models_clean, averaged_models=averaged_models, ranking=ranking,
        sample_positions=sample_positions, sample_times=sample_times,
        theta_obs=theta_obs, fits_step2=fits_step4,
        depths0=depths0, output_dir=output_dir, generate_plots=True,
        probe_layer_indices=probe_layer_indices,
    )


def report_step5_selection(reporter, *, selected_output, n_candidates):
    """Report Step 5 selection and uncertainty details (log only)."""
    _cv = selected_output.cv_map
    _tc = selected_output.theta_calc
    reporter.log(
        f"[STEP 5] Optimal stack depth K = {selected_output.K}  "
        f"(out of {n_candidates} candidates)"
    )
    reporter.log(f"[STEP 5] Minimum RMSE at K={selected_output.K}: {selected_output.rmse:.6f} m³/m³")

    # CV map
    if _cv.ndim == 3:
        reporter.log(
            f"[STEP 5] CV map shape : {_cv.shape}  →  "
            f"(n_time={_cv.shape[0]}, n_measured_points={_cv.shape[1]}, n_layers={_cv.shape[2]})  "
            f"— Coefficient of Variation = weighted_std / weighted_mean per cell; "
            f">1.0 indicates >100 % relative uncertainty"
        )
    elif _cv.ndim == 2:
        reporter.log(
            f"[STEP 5] CV map shape : {_cv.shape}  →  "
            f"(n_measured_points={_cv.shape[0]}, n_layers={_cv.shape[1]})  "
            f"— Coefficient of Variation = weighted_std / weighted_mean per cell"
        )
    else:
        reporter.log(f"[STEP 5] CV map shape : {_cv.shape}")

    reporter.log(
        f"[STEP 5] CV map stats — mean: {np.nanmean(_cv):.4f}  "
        f"max: {np.nanmax(_cv):.4f}  "
        f"fraction of cells with CV>0.5: "
        f"{100*np.nanmean(_cv > 0.5):.1f} %  "
        f"(NaN cells, e.g. unmatched real-data time-lapse reference-grid "
        f"points, excluded from these statistics: "
        f"{100*np.mean(np.isnan(_cv)):.1f} % of cells are NaN)"
    )

    # theta_calc
    if _tc is not None:
        if _tc.ndim == 3:
            reporter.log(
                f"[STEP 5] theta_calc shape : {_tc.shape}  →  "
                f"(n_time={_tc.shape[0]}, n_measured_points={_tc.shape[1]}, n_layers={_tc.shape[2]})  "
                f"— predicted VWC [m³/m³] grid at optimal K, derived from EC→θ "
                f"via per-horizon Waxman-Smits inversion"
            )
        else:
            reporter.log(f"[STEP 5] theta_calc shape : {_tc.shape}")
        reporter.log(
            f"[STEP 5] theta_calc stats — mean: {np.nanmean(_tc):.4f}  "
            f"min: {np.nanmin(_tc):.4f}  "
            f"max: {np.nanmax(_tc):.4f}  [m³/m³]"
        )


def persist_step5_result(reporter, *, selected_output, averaged_models, theta_obs,
                         sample_positions, sample_times, probe_layer_indices,
                         output_model_dir):
    """Persist the Step 4/5 result (theta_calc, cv_map, K, rmse, coverage, is_calibrated, ...).

    Previously this array was only ever plotted, then discarded --
    compare_scenarios.py / plot_truth_comparison.py need it on disk as
    'theta_calc.npy' next to the synthetic_*.npy truth arrays to do
    calc-vs-truth comparisons. Saving it here makes every run self-sufficient
    for that comparison, synthetic or real, with no extra step.

    Returns the list of keys written (one ``<key>.npy`` each).
    """
    _tc = selected_output.theta_calc
    _step5_capture = {
        "K": np.array(selected_output.K),
        "rmse": np.array(selected_output.rmse),
        "cv_map": np.asarray(selected_output.cv_map),
    }
    if _tc is not None:
        _step5_capture["theta_calc"] = np.asarray(_tc)
    # Calibration inputs/outputs for pass 2 of the TDR-pinned experiment
    # (pin_from_tdr.py / run_pinned_tdr.py): train TDR obs + calibrated WS.
    try:
        _step5_capture["ws_params"] = np.asarray(selected_output.best_fit.params)
        _step5_capture["tdr_train_theta"] = np.asarray(theta_obs, dtype=float)
        _step5_capture["tdr_train_positions"] = np.asarray(sample_positions, dtype=int)
        _step5_capture["tdr_train_times"] = np.asarray(sample_times, dtype=int)
        if probe_layer_indices is not None:
            _step5_capture["tdr_train_layers"] = np.asarray(probe_layer_indices, dtype=int)
    except Exception as _e:
        reporter.log(f"[STEP 5][WARN] could not capture WS params / TDR obs: {_e}")
    if getattr(selected_output, "coverage", None) is not None:
        _step5_capture["coverage"] = np.asarray(selected_output.coverage)
    if getattr(selected_output, "best_fit", None) is not None:
        _step5_capture["is_calibrated"] = np.array(
            [h.is_calibrated for h in selected_output.best_fit.horizon_fits]
        )
    # Optimal RMSE-weighted bulk-EC model (n_time, n_pts, n_layers) -- needed
    # by plot_validation_1to1.py for the per-horizon EC truth-vs-optimal 1:1.
    try:
        _step5_capture["ec_optimal"] = np.asarray(averaged_models[selected_output.K - 1])
    except Exception as _e:
        reporter.log(f"[STEP 5][WARN] could not capture ec_optimal: {_e}")
    for _key, _arr in _step5_capture.items():
        _path = os.path.join(output_model_dir, f"{_key}.npy")
        np.save(_path, _arr)
    reporter.log(
        f"[STEP 5] Saved Step 4/5 result to '{output_model_dir}': "
        f"{', '.join(f'{k}.npy' for k in _step5_capture)}"
    )
    return list(_step5_capture)


def run_step5_quality_gate(reporter, *, selected_output, cfg):
    """Post-Step-5 quality gate (report item 4).

    Turns the CV map and per-horizon calibration status -- already computed
    but previously only ever printed/plotted -- into an explicit pass/warn
    verdict. Never aborts the run; only downgrades the final reported status
    (see ``reporter.finish()``) so a degraded-but-completed run is visibly
    distinguishable from a clean one. ``cfg`` may be ``None`` (defaults).

    Returns the ``EMmethod.evaluate_quality_gates`` result.
    """
    quality_gate_result = EMmethod.evaluate_quality_gates(
        cv_map=selected_output.cv_map,
        best_fit=selected_output.best_fit,
        cv_mean_warn_threshold=(
            getattr(cfg, "quality_gate_cv_mean_warn", 0.3) if cfg is not None else 0.3
        ),
        cv_high_threshold=(
            getattr(cfg, "quality_gate_cv_high_threshold", 1.0) if cfg is not None else 1.0
        ),
        cv_high_fraction_warn_threshold=(
            getattr(cfg, "quality_gate_cv_high_fraction_warn", 0.3) if cfg is not None else 0.3
        ),
        require_all_horizons_calibrated=(
            getattr(cfg, "quality_gate_require_all_horizons_calibrated", True)
            if cfg is not None else True
        ),
    )
    reporter.log(quality_gate_result.message)
    reporter.metric("Quality gate level", quality_gate_result.level)
    reporter.metric("Quality gate — mean CV", f"{quality_gate_result.cv_mean:.4f}")
    reporter.metric(
        "Quality gate — CV>threshold fraction",
        f"{quality_gate_result.cv_high_fraction*100:.1f}%",
    )
    reporter.metric(
        "Quality gate — calibrated horizons",
        f"{quality_gate_result.n_calibrated_horizons}/{quality_gate_result.n_total_horizons}",
    )
    if quality_gate_result.level == "WARN":
        reporter.log(
            "[PIPELINE][WARN] Final solution failed one or more quality gates -- "
            "see 'Quality gate reasons' in the report summary. The run still "
            "completed and all outputs were written; treat the solution with "
            "extra caution before relying on it."
        )
    return quality_gate_result
