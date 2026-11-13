"""Pipeline Step 5b: held-out validation (Loiseau et al., 2026 Sec. 2.6). Phase 6 of refactoring.md.

Forward-only evaluation of the TRAIN-fitted Waxman-Smits parameters on the
VALIDATION subset; no re-fitting. No import of ``run_pipeline`` (circular).
"""
from __future__ import annotations

import os

import numpy as np

import plot_train_val_diagnostics as ptv
import step2_petro as step2

_ALL_FIXED = {"a": True, "m": True, "sigma_w": True, "sigma_s": True}


def _val_fit_at_k(k, params, *, averaged_models, sample_positions_val, sample_times_val,
                  theta_obs_val, probe_layer_val, depths0, output_dir, depths_are_fixed,
                  probe_depths_reflect_fitted_interfaces, theta_obs_std):
    """Forward pass with every WS parameter pinned to the TRAIN-fitted ``params`` at stack depth index k."""
    return step2.run_petro(
        # averaged_models[k] already has the ensemble axis collapsed (Step 3);
        # wrap it back into a length-1 ensemble since run_petro calls `.ndim`.
        models=np.stack([averaged_models[k]]),
        sample_positions=sample_positions_val, sample_times=sample_times_val,
        theta_obs=theta_obs_val, depths0=depths0, output_dir=output_dir,
        generate_plots=False,   # diagnostic-only pass
        probe_layer_indices=probe_layer_val,
        depths_are_fixed=depths_are_fixed,
        fixed_params=dict(_ALL_FIXED),
        probe_depths_reflect_fitted_interfaces=probe_depths_reflect_fitted_interfaces,
        lab_p0=params,   # pinned to TRAIN-fitted values, not LAB_P0
        theta_obs_std=theta_obs_std,
    )[0]


def run_step5b_validation(reporter, *, split, selected_output, fits_step4, train_rmse_by_k,
                          averaged_models, sample_positions_val, sample_times_val,
                          theta_obs_val, probe_layer_val, depths0, output_dir, fixed_depths,
                          probe_depths_reflect_fitted_interfaces, theta_obs_std):
    """STEP 5b. Returns ``(train_rmse, val_rmse, val_rmse_by_k)``."""
    reporter.step("Step 5b", "Held-out validation (Sec. 2.6)")
    reporter.log("\n" + "=" * 80)
    reporter.log(" [PIPELINE] STEP 5b: HELD-OUT VALIDATION")
    reporter.log("=" * 80)

    common = dict(averaged_models=averaged_models, sample_positions_val=sample_positions_val,
                  sample_times_val=sample_times_val, theta_obs_val=theta_obs_val,
                  probe_layer_val=probe_layer_val, depths0=depths0, output_dir=output_dir,
                  depths_are_fixed=fixed_depths if isinstance(fixed_depths, bool) else all(fixed_depths),
                  probe_depths_reflect_fitted_interfaces=probe_depths_reflect_fitted_interfaces,
                  theta_obs_std=theta_obs_std)

    # Validation RMSE at EVERY candidate K (train-vs-val RMSE-vs-K curve). Cheap:
    # fixed_params makes each call a forward pass, not an optimisation.
    val_rmse_by_k = []
    val_fit = None
    for k in range(len(fits_step4)):
        fit_k = _val_fit_at_k(k, fits_step4[k].params, **common)   # params: (num_layers, 4)
        val_rmse_by_k.append(fit_k.rmse)
        if k == selected_output.K - 1:
            val_fit = fit_k
    val_rmse_by_k = np.array(val_rmse_by_k)

    if val_fit is None:
        # Defensive: selected_output.K should always be in 1..len(fits_step4).
        reporter.log("[STEP 5b][WARN] selected_output.K was not covered by "
                     "the per-K validation loop -- re-evaluating it directly.")
        val_fit = _val_fit_at_k(selected_output.K - 1, fits_step4[selected_output.K - 1].params,
                                **common)

    train_rmse_by_k = np.array(train_rmse_by_k)   # same fits_step4 list, index for index
    train_rmse = selected_output.rmse
    val_rmse = val_fit.rmse
    reporter.log(f"[STEP 5b] Train RMSE (K={selected_output.K}, n={len(split.train)}): "
                 f"{train_rmse:.5f} m³/m³")
    reporter.log(f"[STEP 5b] Val   RMSE (K={selected_output.K}, n={len(split.val)}): "
                 f"{val_rmse:.5f} m³/m³")
    gap = val_rmse - train_rmse
    reporter.metric("Train RMSE", f"{train_rmse:.5f} m³/m³")
    reporter.metric("Validation RMSE", f"{val_rmse:.5f} m³/m³")
    reporter.metric("Train↔Val RMSE gap", f"{gap:+.5f} m³/m³")
    if not split.independent:
        reporter.log(
            f"[STEP 5b][NOTE] split method='{split.method}' is not "
            f"fully independent (see [SPLIT][WARN] above) -- a "
            f"small train/val gap here is expected and does not by "
            f"itself rule out overfitting the way it would for an "
            f"independent split."
        )
    if abs(gap) > 0.3 * train_rmse:
        reporter.log(
            "[STEP 5b][WARN] Val RMSE diverges >30% from train RMSE "
            "-- possible overfitting in K selection, or val subset "
            "too small to be representative. Treat the selected K "
            "with caution."
        )

    train_val_dir = os.path.join(output_dir, "train_val")
    try:
        os.makedirs(train_val_dir, exist_ok=True)
        ptv.plot_rmse_vs_k(train_rmse_by_k=train_rmse_by_k, val_rmse_by_k=val_rmse_by_k,
                           selected_K=selected_output.K, output_dir=train_val_dir)
        best = fits_step4[selected_output.K - 1]
        ptv.plot_parity_train_val(
            train_obs=best.observed_theta, train_pred=best.predicted_theta,
            val_obs=val_fit.observed_theta, val_pred=val_fit.predicted_theta,
            output_dir=train_val_dir)
        reporter.log(f"[STEP 5b] Train/val diagnostic plots -> {train_val_dir}")
    except Exception as exc:
        reporter.log(f"[STEP 5b][WARN] Train/val diagnostic plots failed "
                     f"(pipeline continues): {type(exc).__name__}: {exc}")

    reporter.end_step("Step 5b")
    return train_rmse, val_rmse, val_rmse_by_k
