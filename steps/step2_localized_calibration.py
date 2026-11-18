"""Pipeline Step 2 LOCALIZED: kernel-localized Waxman-Smits calibration (diagnostic sibling of Step 2).
Phase 6 of refactoring.md.

Moved byte-for-byte (apart from variable names) out of ``run_pipeline.main``. It never
changes Step 2/4 calibration or Step 3/5 K-selection: any failure is logged as a warning and
the run continues. The bandwidth comes from ``steps.step2_localized_sweep`` or the scenario.
No import of ``run_pipeline`` (it would be circular).
"""
from __future__ import annotations

import numpy as np

import plot_calibration_density as pcd
import step2_petro as step2


def run_step2_localized_diagnostic(reporter, *, mode, bandwidth, bandwidth_from_sweep, localized_y,
                                   models_clean, sample_positions, sample_times, theta_obs,
                                   x_transect, depths0, effective_depths0, output_dir,
                                   probe_layer_indices, fixed_params, lab_p0, theta_obs_std,
                                   fits_step2, sample_sensor_ids, ws_bounds, survey_df,
                                   calib_position_results):
    """Run the kernel-localized calibration and its spatial diagnostics.

    ``bandwidth`` is the fixed scenario value or the sweep's recommendation (None -> skipped).
    ``fits_step2[0]`` is the rank-1 global fit passed as ``global_fit``. ``depths0`` is the
    static depth list for the fit; ``effective_depths0`` is what probe routing really used
    (for the plots). Returns the ``run_petro_localized`` result (a list of ``LocalizedPetroFit``,
    one per ensemble member), or None when skipped or when the diagnostic failed.
    """
    fits_step2_localized = None
    if bandwidth is not None:
        reporter.log(
            f"[STEP 2 LOCALIZED] ws_localized_bandwidth="
            f"{bandwidth:g} "
            f"({'recommended by LOO-CV sweep above' if bandwidth_from_sweep else 'fixed scenario value'}) "
            f"-- running kernel-localized calibration as a diagnostic sibling to Step 2 "
            f"({'2-D Euclidean (x, y)' if localized_y is not None else '1-D along-transect (x)'} distance)."
        )
        try:
            fits_step2_localized = step2.run_petro_localized(
                models=models_clean,
                sample_positions=sample_positions,
                sample_times=sample_times,
                theta_obs=theta_obs,
                point_x_coords=x_transect,
                point_y_coords=localized_y,
                bandwidth=bandwidth,
                depths0=depths0,
                output_dir=output_dir,
                probe_layer_indices=probe_layer_indices,
                fixed_params=fixed_params,
                lab_p0=lab_p0,
                theta_obs_std=theta_obs_std,
                global_fit=fits_step2[0],
                sensor_ids=sample_sensor_ids,
                ws_bounds=ws_bounds,
            )
            reporter.log(
                f"[STEP 2 LOCALIZED] Diagnostic complete -> "
                f"{output_dir}/petro_calibration_localized.png"
            )

            # Spatial diagnostics for this same localized fit: how many
            # (effective, kernel-weighted) TDR probes back each EM point,
            # and what the fit actually produced spatially, per horizon.
            pcd.plot_calibration_density_kernel(
                survey_df=survey_df,
                position_results=calib_position_results,
                bandwidth=bandwidth,
                output_dir=output_dir,
            )
            _calib_probe_x = np.array([r.probe_x for r in calib_position_results])
            # run_petro_localized returns a list of LocalizedPetroFit, one
            # per ensemble member (symmetric with fits_step2) -- [0] is the
            # rank-1 member, matching global_fit=fits_step2[0] passed in
            # above. NOT a single LocalizedPetroFit with its own
            # .horizon_fits directly.
            #
            # One combined XZ-style figure (horizon = depth axis) rather
            # than one flat XY scatter per horizon -- directly comparable
            # to water_content_xz_*.png. depths0=effective_depths0, not
            # the static depths0, for the same reason EM_LAYER_WEIGHTS
            # uses it above: it's what this fit's probe routing actually
            # used.
            pcd.plot_ws_params_xz(
                horizon_fits=fits_step2_localized[0].horizon_fits,
                depths0=effective_depths0,
                output_dir=output_dir,
                probe_x=_calib_probe_x,
                # .probe_depth, not a re-derivation via probe_layer_indices:
                # sample_positions/probe_layer_indices can be reordered or
                # subset relative to calib_position_results (train/val
                # split, time-lapse expansion), so indexing one against
                # the other isn't safe. SamplePositionResult carries its
                # own installation depth directly.
                probe_depths=np.array([r.probe_depth for r in calib_position_results]),
            )
            # XY (plan-view) counterpart to the XZ figure above -- same
            # kernel-localized WS parameters, horizons collapsed into
            # one averaged value per target instead of laid out on the
            # depth axis. Directly comparable to
            # plot_footprint_spatial_discrepancy's XY map, the same way
            # plot_ws_params_xz is comparable to water_content_xz_*.png.
            # Only meaningful with real (x, y) spread, hence 2-D mode
            # gating below; in 1-D mode target_y is all zeros and this
            # would just collapse to a line.
            if mode == "2d":
                pcd.plot_ws_params_xy(
                    horizon_fits=fits_step2_localized[0].horizon_fits,
                    output_dir=output_dir,
                    probe_x=_calib_probe_x,
                    probe_y=np.array([r.probe_y for r in calib_position_results]),
                )
        except Exception as e:
            # Diagnostic sibling only -- never let it fail a run that
            # would otherwise succeed.
            reporter.log(f"[STEP 2 LOCALIZED][WARN] Localized diagnostic failed: {e}")
    else:
        reporter.log(
            "[STEP 2 LOCALIZED] ws_localized_bandwidth not set -- skipping "
            "kernel-localized diagnostic (unchanged default behaviour)."
        )

    return fits_step2_localized
