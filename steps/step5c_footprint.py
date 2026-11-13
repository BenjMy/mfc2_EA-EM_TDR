"""Pipeline Step 5c: EM footprint discrepancy (McNeill depth weighting). Phase 6 of refactoring.md.

Diagnostic only: neither RMSE feeds back into calibration or K-selection.
No import of ``run_pipeline`` (it would be circular).
"""
from __future__ import annotations

import numpy as np

import em_sensitivity as emsens
import plot_em_footprint as pef


def run_step5c_footprint(reporter, *, selected_output, em_layer_weights, sample_positions,
                         probe_layer_indices, theta_obs, sample_times, timelapse,
                         data_source, coil_orientation, survey_df, calib_position_results,
                         effective_depths0, output_dir):
    """STEP 5c: compare hard-routed vs McNeill-weighted theta against the TDR probes.

    Returns ``(discrepancies, hr_rmse, w_rmse)``.
    """
    reporter.step("Step 5c", "EM footprint discrepancy (McNeill depth weighting)")

    # theta_calc is ALWAYS (n_time, num_measured_points, num_layers) --
    # SelectionOutput guarantees this (step4_select.py), so t_idx = 0 covers
    # the single-time case directly.
    theta_calc = selected_output.theta_calc
    eff_theta = emsens.effective_theta(theta_calc, em_layer_weights)
    # eff_theta: (n_time, num_measured_points) -- one depth-weighted value per
    # position per time step ("what the coil actually saw").

    discrepancies = []
    for i, (pos, li, obs) in enumerate(zip(sample_positions, probe_layer_indices, theta_obs)):
        t_idx = sample_times[i] if timelapse else 0
        hard_routed = theta_calc[t_idx, pos, li]
        weighted = eff_theta[t_idx, pos]
        discrepancies.append({
            "probe": i, "pos": int(pos), "obs": obs,
            # probe's own routed layer index, carried through so the XZ plot
            # can place this record on the depth axis.
            "horizon": int(li),
            "hard_routed": hard_routed, "weighted": weighted,
            "hard_routed_err": hard_routed - obs,
            "weighted_err": weighted - obs,
        })

    hr_rmse = np.sqrt(np.nanmean([d["hard_routed_err"] ** 2 for d in discrepancies]))
    w_rmse = np.sqrt(np.nanmean([d["weighted_err"] ** 2 for d in discrepancies]))
    reporter.log(
        f"[STEP 5c] RMSE vs TDR — hard-routed (nearest layer): {hr_rmse:.5f} m³/m³  "
        f"|  depth-weighted (McNeill {coil_orientation if data_source == 'real' else 'multi-coil avg'}): "
        f"{w_rmse:.5f} m³/m³"
    )
    reporter.metric("Footprint RMSE — hard-routed layer", f"{hr_rmse:.5f} m³/m³")
    reporter.metric("Footprint RMSE — McNeill-weighted", f"{w_rmse:.5f} m³/m³")
    reporter.table(
        "Step 5c — EM footprint discrepancy (per probe)",
        ["Probe #", "TDR obs [m³/m³]", "Hard-routed θ", "Weighted θ",
         "Hard-routed err", "Weighted err"],
        [
            [d["probe"], f"{d['obs']:.4f}", f"{d['hard_routed']:.4f}", f"{d['weighted']:.4f}",
             f"{d['hard_routed_err']:+.4f}", f"{d['weighted_err']:+.4f}"]
            for d in discrepancies
        ],
        note=(
            "'Hard-routed' = theta_calc at the probe's single assigned layer "
            "(current Step 2/4 calibration routing, unchanged). 'Weighted' = "
            "McNeill depth-weighted blend across all layers (EM_LAYER_WEIGHTS). "
            "Diagnostic only -- neither RMSE feeds back into calibration or "
            "K-selection."
        ),
    )
    if w_rmse < hr_rmse:
        reporter.log(
            "[STEP 5c] Depth-weighted comparison fits TDR better than the "
            "hard-routed layer -- the current nearest-layer probe routing "
            "may be discarding real cross-layer signal contribution. "
            "See the explainer's Step D for what it would take to fold "
            "this into calibration itself, and note the caveats there "
            "before doing so."
        )
    else:
        reporter.log(
            "[STEP 5c] Hard-routed layer already matches TDR at least as "
            "well as the McNeill-weighted blend -- either your layers are "
            "thick relative to the coil spacing used (weight concentrates in "
            "one layer anyway) or CS/high-frequency LIN departure is "
            "degrading the kernel here. Check EM_LAYER_WEIGHTS -- if one "
            "layer already carries >90% of the weight, this result is "
            "expected and the current routing is fine."
        )

    try:
        probe_x = np.array([r.probe_x for r in calib_position_results])
        pef.plot_footprint_probe_comparison(
            discrepancies=discrepancies, output_dir=output_dir,
            hr_rmse=hr_rmse, w_rmse=w_rmse)
        pef.plot_footprint_spatial_discrepancy(
            discrepancies=discrepancies, survey_df=survey_df, output_dir=output_dir,
            probe_x=probe_x, probe_y=np.array([r.probe_y for r in calib_position_results]))
        # XZ (depth) counterpart; effective_depths0 (not the static DEPTHS0) =
        # this run's actual, possibly fitted, interfaces.
        pef.plot_footprint_spatial_discrepancy_xz(
            discrepancies=discrepancies, survey_df=survey_df, depths0=effective_depths0,
            output_dir=output_dir, probe_x=probe_x,
            probe_depths=np.array([r.probe_depth for r in calib_position_results]))
    except Exception as exc:
        reporter.log(
            f"[STEP 5c][WARN] Footprint diagnostic plots failed "
            f"(pipeline continues): {type(exc).__name__}: {exc}"
        )

    reporter.end_step("Step 5c")
    return discrepancies, hr_rmse, w_rmse
