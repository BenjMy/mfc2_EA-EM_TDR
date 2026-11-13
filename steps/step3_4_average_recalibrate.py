"""Pipeline Steps 3 and 4: RMSE-weighted averaging and petrophysical recalibration (Phase 6).

No import of ``run_pipeline`` (circular). Each function opens its own reporter
step; the caller closes it (``end_step``) so step bookkeeping stays in ``main``.
"""
from __future__ import annotations

import numpy as np

import em_methodology as EMmethod
import step3_average as step3


def run_step3_average(reporter, *, models_clean, fits_step2, depths0, output_dir, timelapse):
    """STEP 3: rank ensemble members by RMSE, build cumulative averaged stacks.

    Returns ``(averaged_models, ranking)``.
    """
    reporter.step("Step 3", "RMSE-weighted ranking-averaging engine")
    reporter.log("\n" + "=" * 80)
    reporter.log(" [PIPELINE] STEP 3: RMSE-WEIGHTED RANKING-AVERAGING ENGINE")
    reporter.log("=" * 80)

    averaged_models, ranking = step3.run_average(
        models=models_clean, fits_step2=fits_step2, depths0=depths0, output_dir=output_dir)

    ranked_rmse = [fits_step2[i].rmse for i in ranking]
    reporter.log(f"[STEP 3] Ranked {len(ranking)} ensemble members by RMSE (ascending).")
    reporter.log(f"[STEP 3] Best  member  : index {ranking[0]},  RMSE = {ranked_rmse[0]:.5f} m³/m³")
    reporter.log(f"[STEP 3] Worst member  : index {ranking[-1]}, RMSE = {ranked_rmse[-1]:.5f} m³/m³")
    reporter.log(
        f"[STEP 3] Cumulative averaged models produced: {len(averaged_models)} "
        f"(one averaged stack per K from 1 to {len(averaged_models)})"
    )
    shape = averaged_models[0].shape
    if timelapse:
        reporter.log(
            f"[STEP 3] Each averaged_model shape: {shape}  →  "
            f"(n_time={shape[0]}, n_measured_points={shape[1]}, n_layers={shape[2]})"
        )
    else:
        reporter.log(
            f"[STEP 3] Each averaged_model shape: {shape}  →  "
            f"(n_measured_points={shape[0]}, n_layers={shape[1]})"
        )
    return averaged_models, ranking


def run_step4_recalibrate(reporter, *, averaged_models, sample_positions, sample_times,
                          theta_obs, depths0, probe_layer_indices, fixed_depths,
                          ws_fixed_params, probe_depths_reflect_fitted_interfaces, lab_p0,
                          theta_obs_std, ws_bounds_override):
    """STEP 4: recalibrate Waxman-Smits on every cumulative averaged model.

    Returns ``(fits_step4, rmse_by_k)``.
    """
    reporter.step("Step 4", "Recalibrate petrophysical relationships")
    reporter.log("\n" + "=" * 80)
    reporter.log(" [PIPELINE] STEP 4: RECALIBRATE PETROPHYSICAL RELATIONSHIPS")
    reporter.log("=" * 80)

    fits = EMmethod.recalibrate_petrophysical_relationship(
        averaged_models=averaged_models, sample_positions=sample_positions,
        sample_times=sample_times, theta_obs=theta_obs, depths0=depths0,
        probe_layer_indices=probe_layer_indices,
        depths_are_fixed=fixed_depths if isinstance(fixed_depths, bool) else all(fixed_depths),
        fixed_params=ws_fixed_params,
        probe_depths_reflect_fitted_interfaces=probe_depths_reflect_fitted_interfaces,
        lab_p0=lab_p0, theta_obs_std=theta_obs_std, ws_bounds=ws_bounds_override,
    )

    n = len(fits)
    rmse_by_k = [f.rmse for f in fits]
    reporter.log(f"[STEP 4] Recalibrated {n} Waxman-Smits fits "
                 f"(one per cumulative averaged model K=1…{n}).")
    reporter.log(
        f"[STEP 4] RMSE across K — mean: {np.nanmean(rmse_by_k):.5f}  "
        f"std: {np.nanstd(rmse_by_k):.5f}  "
        f"min: {np.nanmin(rmse_by_k):.5f} (best K)  "
        f"max: {np.nanmax(rmse_by_k):.5f}  [m³/m³]"
    )
    best_idx = int(np.nanargmin(rmse_by_k))
    rows: dict[str, str] = {}
    for hz in fits[best_idx].horizon_fits:
        key = f"Hz {hz.horizon} — a / m / σ_w / σ_s / RMSE"
        if hz.is_calibrated:
            rows[key] = (f"{hz.a:.4f} / {hz.m:.4f} / {hz.sigma_w:.2f} mS/m / "
                         f"{hz.sigma_s:.2f} mS/m / {hz.rmse:.5f} m³/m³")
        else:
            rows[key] = (f"UNCALIBRATED — no TDR probe in this horizon; showing prior "
                         f"p0 only: {hz.a:.4f} / {hz.m:.4f} / "
                         f"{hz.sigma_w:.2f} mS/m / {hz.sigma_s:.2f} mS/m / RMSE=N/A")
    reporter.args_table(
        f"Step 4 — Best recalibrated fit (K={best_idx + 1}, "
        f"RMSE={np.nanmin(rmse_by_k):.5f} m³/m³)", rows)
    return fits, rmse_by_k
