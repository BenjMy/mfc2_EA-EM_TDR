"""Pipeline Step 2 LOCALIZED SWEEP: bandwidth (c) selection by leave-one-probe-out CV. Phase 6 of refactoring.md.

Moved byte-for-byte (apart from variable names) out of ``run_pipeline.main``. It is a tuning
aid only: any failure is logged and the fixed ``ws_localized_bandwidth`` is kept.
No import of ``run_pipeline`` (it would be circular).
"""
from __future__ import annotations

import numpy as np

import bandwidth_sweep_check as bwc
import step2_petro as step2


def run_step2_localized_sweep(reporter, *, bandwidths, bandwidth, fits_step2, models_clean,
                              sample_positions, sample_times, theta_obs, x_transect, localized_y,
                              probe_layer_indices, lab_p0, fixed_params, theta_obs_std, ws_bounds,
                              output_dir):
    """Evaluate the candidate bandwidths once, against the rank-1 Step 2 member.

    Evaluated once against ``fits_step2[0]`` (a one-off tuning decision, not per ensemble
    member). ``bandwidth`` is the fixed scenario value (or None), returned unchanged when the
    sweep fails.

    Returns ``(bandwidth, from_sweep, sweep_results)``: the bandwidth to use afterwards, whether
    it came from the sweep, and the sweep rows (None if the sweep failed).
    """
    from_sweep = False
    sweep_results = None
    reporter.step(
        "Step 2 LOCALIZED SWEEP",
        "Bandwidth (c) selection via leave-one-probe-out CV",
    )
    _sweep_ref_idx = fits_step2[0].model_index
    reporter.log(
        f"[STEP 2 LOCALIZED SWEEP] ws_localized_bandwidth_sweep="
        f"{list(bandwidths)} set -- evaluating candidate "
        f"bandwidth(s) via leave-one-probe-out CV against the rank-1 "
        f"ensemble member (model_index={_sweep_ref_idx}, "
        f"RMSE={fits_step2[0].rmse:.5f} m³/m³) "
        f"({'2-D Euclidean (x, y)' if localized_y is not None else '1-D along-transect (x)'} distance)."
    )
    try:
        # kernel support is 2c: warn when no candidate can reach a neighbouring probe
        _sp = np.asarray(sample_positions, dtype=int)
        _bw_warn = bwc.sweep_spacing_warning(
            bandwidths,
            np.asarray(x_transect, dtype=float)[_sp],
            None if localized_y is None else np.asarray(localized_y, dtype=float)[_sp],
        )
        if _bw_warn:
            reporter.log(f"[STEP 2 LOCALIZED SWEEP][WARN] {_bw_warn}")
    except Exception as _e:  # diagnostic only: never stop the run
        reporter.log(f"[STEP 2 LOCALIZED SWEEP] spacing check skipped: {_e}")
    try:
        _sigma_b_reference = models_clean[_sweep_ref_idx]
        sweep_results, _recommended_bandwidth = step2.run_bandwidth_sweep(
            sigma_b_reference=_sigma_b_reference,
            sample_positions=sample_positions,
            sample_times=sample_times,
            theta_obs=theta_obs,
            point_x_coords=x_transect,
            point_y_coords=localized_y,
            bandwidths=bandwidths,
            output_dir=output_dir,
            probe_layer_indices=probe_layer_indices,
            lab_p0=lab_p0,
            fixed_params=fixed_params,
            theta_obs_std=theta_obs_std,
            ws_bounds=ws_bounds,
        )
        _sweep_rows = [
            [
                f"{r.bandwidth:g}",
                f"{r.loo_rmse:.5f}" if not np.isnan(r.loo_rmse) else "N/A",
                r.n_folds,
                r.n_fallback,
            ]
            for r in sweep_results
        ]
        reporter.table(
            "Step 2 LOCALIZED — bandwidth sweep (leave-one-probe-out CV)",
            ["Bandwidth c", "Pooled LOO-CV RMSE [m³/m³]", "Valid folds", "Folds fell back to prior"],
            _sweep_rows,
            note=(
                f"Recommended bandwidth: c={_recommended_bandwidth:g} (minimum pooled "
                f"LOO-CV RMSE). Evaluated once against the rank-1 ensemble member "
                f"(model_index={_sweep_ref_idx}) -- a one-off tuning decision, not "
                f"repeated per ensemble member."
            ),
        )
        reporter.log(
            f"[STEP 2 LOCALIZED SWEEP] Recommended bandwidth: c={_recommended_bandwidth:g} "
            f"-> {output_dir}/bandwidth_sweep.png"
        )
        reporter.metric("Step 2 LOCALIZED — bandwidth selection method", "LOO-CV sweep")
        reporter.metric("Step 2 LOCALIZED — recommended bandwidth (c)", f"{_recommended_bandwidth:g}")
        reporter.metric("Step 2 LOCALIZED — sweep candidates evaluated", len(sweep_results))
        bandwidth = _recommended_bandwidth
        from_sweep = True
    except Exception as e:
        # Sweep is a tuning aid only -- never let it fail a run that
        # would otherwise succeed; fall back to the fixed
        # ws_localized_bandwidth (if any) below, unchanged.
        reporter.log(
            f"[STEP 2 LOCALIZED SWEEP][WARN] Bandwidth sweep failed: {e}. "
            f"Falling back to fixed ws_localized_bandwidth="
            f"{bandwidth!r} (unswept)."
        )
    reporter.end_step("Step 2 LOCALIZED SWEEP")
    return bandwidth, from_sweep, sweep_results
