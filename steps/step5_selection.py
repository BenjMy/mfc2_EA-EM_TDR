"""Pipeline Step 5: convergence selection & reliability mapping. Phase 6 of refactoring.md.

Blocks moved out of ``run_pipeline.main()``:

* ``run_step5_select``          -- opens "Step 5" and runs ``step4_select.run_select_assess``
* ``run_step5_localized_mapping`` -- (new, default) converts the optimal EC model to theta with
  kernel-localized Waxman-Smits parameters instead of one global set per horizon
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
import localized_mapping as lmap
import step4_select as step4_5
import ws_spatial as wss


def run_step5_select(reporter, *, models_clean, averaged_models, ranking, sample_positions,
                     sample_times, theta_obs, fits_step4, depths0, output_dir,
                     probe_layer_indices, k_selection=None, bandwidth=None, point_x=None,
                     point_y=None, lab_p0=None, fixed_params=None, theta_obs_std=None,
                     ws_bounds=None):
    """STEP 5: select the optimal stack depth K and map reliability (CV).

    Returns the ``SelectionOutput`` of ``step4_select.run_select_assess``.
    ``fits_step4`` are the Step 4 fits (passed as ``fits_step2=`` downstream).

    ``k_selection`` None / ``"global"`` (default): K minimises the probe RMSE of the global
    Step 4 fit of each stack. ``"localized-loo"``: K minimises the leave-one-probe-out RMSE
    of the kernel-localized fit (``localized_mapping.localized_loo_rmse_by_k``, bandwidth
    ``bandwidth``, ``point_x`` / ``point_y`` = EM point coordinates); the global curve is
    still computed, logged and plotted. Downgraded to global, with the reason logged, when
    there is no bandwidth, no point coordinates, or no probe can be scored. The ensemble
    ranking (Steps 2-3) is global in both modes.
    """
    reporter.step("Step 5", "Convergence selection & reliability mapping")
    reporter.log("\n" + "=" * 80)
    reporter.log(" [PIPELINE] STEP 5: CONVERGENCE SELECTION & RELIABILITY MAPPING")
    reporter.log("=" * 80)

    mode, why = lmap.resolve_k_selection(k_selection, bandwidth)
    if mode == "localized-loo" and point_x is None:
        mode, why = "global", "no EM point coordinates"
    loo_curve = None
    if mode == "localized-loo":
        reporter.log(f"[STEP 5] K selection: localized leave-one-probe-out, bandwidth c={bandwidth:g} "
                     f"({len(averaged_models)} stacks x {len(set(sample_positions))} probe locations "
                     "refits)")
        try:
            loo_curve, n_scored = lmap.localized_loo_rmse_by_k(
                averaged_models, bandwidth=bandwidth, point_x=point_x, point_y=point_y,
                sample_positions=sample_positions, sample_times=sample_times, theta_obs=theta_obs,
                probe_layer_indices=probe_layer_indices, lab_p0=lab_p0, fixed_params=fixed_params,
                theta_obs_std=theta_obs_std, ws_bounds=ws_bounds)
        except Exception as exc:   # noqa: BLE001 -- keep the run on the global selection
            mode, why = "global", f"localized LOO failed ({type(exc).__name__}: {exc})"
        else:
            if np.all(np.isnan(loo_curve)):
                loo_curve, mode = None, "global"
                why = ("no probe has another probe of its horizon within 2c, nothing to score: "
                       "increase the bandwidth")
            else:
                n_total = len(sample_positions)
                if int(np.max(n_scored)) < n_total:
                    reporter.log(f"[STEP 5] K selection: {int(np.max(n_scored))} of {n_total} probe "
                                 "observations are scorable (the rest have no other probe within 2c "
                                 "and are skipped for every K).")
    if mode == "global":
        reporter.log("[STEP 5] K selection: global per-horizon fit"
                     + (f" (localized-loo requested but unavailable: {why})" if why else ""))

    out = step4_5.run_select_assess(
        models=models_clean, averaged_models=averaged_models, ranking=ranking,
        sample_positions=sample_positions, sample_times=sample_times,
        theta_obs=theta_obs, fits_step2=fits_step4,
        depths0=depths0, output_dir=output_dir, generate_plots=True,
        probe_layer_indices=probe_layer_indices, selection_rmse_by_k=loo_curve,
    )
    reporter.metric("Step 5 — K selection", getattr(out, "k_selection", "global"))
    if getattr(out, "rmse_by_k_localized_loo", None) is not None:
        k_loc, k_glob = out.K, out.K_global
        loo_at = out.rmse_by_k_localized_loo[k_loc - 1]
        reporter.log(f"[STEP 5] K = {k_loc} (localized LOO, RMSE {loo_at:.5f}); the global curve "
                     f"would pick K = {k_glob}"
                     + ("" if k_loc != k_glob else " (the two agree)")
                     + f". Global RMSE at K = {k_loc}: {out.rmse:.5f} m³/m³.")
        reporter.metric("Step 5 — K (global curve minimum)", str(k_glob))
        reporter.metric("Step 5 — localized LOO RMSE at K", f"{loo_at:.5f} m³/m³")
    return out


def run_step5_localized_mapping(reporter, *, selected_output, averaged_models, requested_mode,
                                bandwidth, bandwidth_source, point_x, point_y, sample_positions,
                                sample_times, theta_obs, probe_layer_indices, lab_p0,
                                fixed_params, theta_obs_std, ws_bounds,
                                ws_spatial=None, sensor_ids=None, output_dir=None,
                                spatial_free=("sigma_s",), spatial_power=2.0,
                                spatial_shrinkage=1.0):
    """Switch ``selected_output.theta_calc`` to the localized-WS map (see localized_mapping.py).

    K and its RMSE come from the global Step 4/5 selection and are not touched. Only the
    EC -> theta conversion of the maps changes: each EM point uses the parameters fitted
    for it by the Gaspari-Cohn kernel, and points outside the kernel support fall back to
    the global per-horizon fit. The global map is kept in ``selected_output.theta_calc_global``.
    ``requested_mode`` None = default (localized); it is downgraded to global, with the
    reason logged, when no bandwidth is available or the localized fit fails.

    ``ws_spatial`` (``global`` | ``localized`` | ``voronoi`` | ``idw``; None = keep ``requested_mode``;
    the old names ``pooled`` / ``kernel`` are still accepted) chooses the calibration unit: ``global`` =
    one set per horizon, ``localized`` = Gaspari-Cohn per EM point, and ``voronoi`` /
    ``idw`` fit ``sigma_s`` per station and spread it over the field
    (:func:`ws_spatial.build_station_mapping`; same container, so Steps 5b / 6 need no change).

    Returns the ``LocalizedMapping`` or None (global maps).
    """
    if ws_spatial is not None:
        ws_spatial = wss.normalize_spatial(ws_spatial)
        if ws_spatial not in wss.SPATIAL_CHOICES:
            raise ValueError(f"ws_spatial must be one of {wss.SPATIAL_CHOICES}, got {ws_spatial!r}")
        if ws_spatial in wss.STATION_METHODS:
            return _run_station_mapping(
                reporter, selected_output=selected_output, averaged_models=averaged_models,
                method=ws_spatial, point_x=point_x, point_y=point_y, sample_positions=sample_positions,
                sample_times=sample_times, theta_obs=theta_obs, probe_layer_indices=probe_layer_indices,
                sensor_ids=sensor_ids, fixed_params=fixed_params, theta_obs_std=theta_obs_std,
                ws_bounds=ws_bounds, output_dir=output_dir, free=spatial_free,
                power=spatial_power, shrinkage=spatial_shrinkage)
        requested_mode = "global" if ws_spatial == "global" else "localized"
    mode, why = lmap.resolve_map_calibration(requested_mode, bandwidth)
    selected_output.ws_map_mode = "global"
    if mode == "global":
        reporter.log("[STEP 5] maps: global per-horizon Waxman-Smits fit"
                     + (f" (localized requested but unavailable: {why})" if why else ""))
        reporter.metric("Step 5 — WS calibration behind the maps", "global")
        return None

    avg = averaged_models[selected_output.K - 1]
    reporter.log(f"[STEP 5] maps: kernel-localized Waxman-Smits, bandwidth c={bandwidth:g} "
                 f"({bandwidth_source}); global per-horizon fit outside the kernel support "
                 f"({'2-D Euclidean (x, y)' if point_y is not None else '1-D along-transect (x)'} distance).")
    try:
        mapping = lmap.build_localized_mapping(
            avg, selected_output.best_fit, bandwidth=bandwidth, point_x=point_x, point_y=point_y,
            sample_positions=sample_positions, sample_times=sample_times, theta_obs=theta_obs,
            probe_layer_indices=probe_layer_indices, lab_p0=lab_p0, fixed_params=fixed_params,
            theta_obs_std=theta_obs_std, ws_bounds=ws_bounds)
    except Exception as exc:   # noqa: BLE001 -- keep the run, but say loudly that maps are global
        reporter.log(f"[STEP 5][WARN] localized mapping failed ({type(exc).__name__}: {exc}); "
                     f"the maps use the global per-horizon fit instead.")
        reporter.metric("Step 5 — WS calibration behind the maps", "global (localized failed)")
        return None

    theta = mapping.theta[np.newaxis, ...] if mapping.theta.ndim == 2 else mapping.theta
    selected_output.theta_calc_global = selected_output.theta_calc
    selected_output.theta_calc = theta
    selected_output.localized_mapping = mapping
    selected_output.ws_map_mode = "localized"

    frac = mapping.fallback_fraction()
    reporter.log("[STEP 5] maps: share of EM points on the GLOBAL fit (no probe within 2c) per horizon: "
                 + ", ".join(f"Hz {i}: {100 * f:.0f}%" for i, f in enumerate(frac)))
    if any(f > 0.5 for f in frac):
        reporter.log("[STEP 5][WARN] more than half of the points of a horizon are outside the kernel "
                     "support: the localized map is mostly the global one there. Increase the "
                     "bandwidth (support is 2c) or add probes.")
    diff = np.nanmean(np.abs(theta - selected_output.theta_calc_global))
    reporter.metric("Step 5 — WS calibration behind the maps", f"localized (c={bandwidth:g})")
    reporter.metric("Step 5 — localized vs global map, mean |dθ|", f"{diff:.5f} m³/m³")
    reporter.metric("Step 5 — points on global fallback (per horizon)",
                    ", ".join(f"{100 * f:.0f}%" for f in frac))
    return mapping


def _run_station_mapping(reporter, *, selected_output, averaged_models, method, point_x, point_y,
                         sample_positions, sample_times, theta_obs, probe_layer_indices, sensor_ids,
                         fixed_params, theta_obs_std, ws_bounds, output_dir, free, power, shrinkage):
    """Per-station WS fit spread over the field (``voronoi`` | ``idw``); None on failure (global)."""
    selected_output.ws_map_mode = "global"
    n_st = wss.count_stations(sensor_ids, fallback=len(set(np.asarray(sample_positions).tolist())))
    reporter.log(f"[STEP 5] maps: per-station Waxman-Smits ({method}), free = {', '.join(free)}, "
                 f"{n_st} stations, prior shrinkage x{shrinkage:g}"
                 + (f", IDW power {power:g}" if method == "idw" else "") + ".")
    try:
        mapping = wss.build_station_mapping(
            averaged_models[selected_output.K - 1], selected_output.best_fit, method=method,
            point_x=point_x, point_y=point_y, sample_positions=sample_positions,
            sample_times=sample_times, theta_obs=theta_obs, probe_layer_indices=probe_layer_indices,
            sensor_ids=sensor_ids, fixed_params=fixed_params, theta_obs_std=theta_obs_std,
            ws_bounds=ws_bounds, free=free, power=power, shrinkage=shrinkage)
    except Exception as exc:   # noqa: BLE001 -- keep the run, but say loudly that maps are global
        reporter.log(f"[STEP 5][WARN] station mapping failed ({type(exc).__name__}: {exc}); "
                     f"the maps use the global per-horizon fit instead.")
        reporter.metric("Step 5 — WS calibration behind the maps", "global (station mapping failed)")
        return None

    theta = mapping.theta[np.newaxis, ...] if mapping.theta.ndim == 2 else mapping.theta
    selected_output.theta_calc_global = selected_output.theta_calc
    selected_output.theta_calc = theta
    selected_output.localized_mapping = mapping       # same attribute: Steps 5b / 6 read it
    selected_output.ws_map_mode = method
    sf = mapping.station_fits
    for note in sf.notes:
        reporter.log(f"[STEP 5][WARN] {note}")
    n_bound = int(sf.at_bound.sum())
    n_fit = int(sf.valid.sum())
    reporter.log(f"[STEP 5] per-station fits: {n_fit} of {sf.valid.size} (horizon, station) pairs fitted; "
                 f"{n_bound} with a free parameter on a bound"
                 + (" -- those values are not identified, do not interpret them" if n_bound else "") + ".")
    diff = np.nanmean(np.abs(theta - selected_output.theta_calc_global))
    reporter.metric("Step 5 — WS calibration behind the maps", f"{method} (per-station {', '.join(free)})")
    reporter.metric("Step 5 — stations used", str(n_st))
    reporter.metric("Step 5 — per-station fits on a bound", f"{n_bound} of {n_fit}")
    reporter.metric(f"Step 5 — {method} vs global map, mean |dθ|", f"{diff:.5f} m³/m³")
    if output_dir:
        wss.write_ws_by_station_csv(sf, os.path.join(output_dir, "ws_by_station.csv"))
        try:
            from plots import ws_spatial_plots
            ws_spatial_plots.plot_station_params(
                mapping, point_x, point_y, os.path.join(output_dir, "ws_spatial_sigma_s_map.png"))
        except Exception as exc:   # noqa: BLE001 -- a plot must never stop a run
            reporter.log(f"[STEP 5][WARN] ws_spatial plot failed ({type(exc).__name__}: {exc}).")
    return mapping


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
    if getattr(selected_output, "rmse_by_k_global", None) is not None:
        _step5_capture["rmse_by_k_global"] = np.asarray(selected_output.rmse_by_k_global)
    if getattr(selected_output, "rmse_by_k_localized_loo", None) is not None:
        _step5_capture["rmse_by_k_localized_loo"] = np.asarray(selected_output.rmse_by_k_localized_loo)
        _step5_capture["K_global"] = np.array(selected_output.K_global)
    if getattr(selected_output, "ws_map_mode", "global") == "localized":
        # theta_calc.npy is the LOCALIZED map; keep the global one and the per-point parameters.
        _lm = selected_output.localized_mapping
        _step5_capture["theta_calc_global"] = np.asarray(selected_output.theta_calc_global)
        _step5_capture["ws_localized_params"] = np.asarray(_lm.params)       # (4, n_layers, n_points): a, m, sigma_w, sigma_s
        _step5_capture["ws_localized_fallback"] = np.asarray(_lm.fallback)   # (n_layers, n_points) True = global fit used
        _step5_capture["ws_localized_bandwidth"] = np.array(_lm.bandwidth)
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
