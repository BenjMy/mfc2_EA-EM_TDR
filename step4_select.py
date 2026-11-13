#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 & 5: Convergence Selection & Reliability Mapping
Calculates the RMSE metrics for each cumulative ensemble stack depth (K)
and generates the Fig. 3 optimization curve along with uncertainty maps.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import em_methodology as EMmethod
import geometry_utils

class SelectionOutput:
    def __init__(self, K: int, rmse: float, cv_map: np.ndarray, best_fit: list,
                 theta_calc: np.ndarray | None = None,
                 coverage: np.ndarray | None = None):
        self.K = K              # Optimal stack depth (1-based index)
        self.rmse = rmse        # Minimum RMSE achieved
        self.cv_map = cv_map    # Coefficient of Variation map at optimal depth
        self.best_fit = best_fit # Calibrated petrophysical parameters used
        self.theta_calc = theta_calc  # Predicted VWC grid at optimal stack depth (n_time, n_points, n_layers)
        self.coverage = coverage      # Normalised coverage map in [-1, 1] (n_time, n_points, n_layers)

def run_select_assess(
    models: np.ndarray,
    averaged_models: list[np.ndarray],
    ranking: np.ndarray,
    sample_positions: list[int],
    sample_times: list[int],
    theta_obs: np.ndarray,
    fits_step2: list,
    depths0: list[float],
    output_dir: str,
    generate_plots: bool = True,
    probe_layer_indices: list[int] | None = None,
) -> SelectionOutput:
    """
    Evaluates every stack depth K to find the optimal ensemble model,
    calculates reliability/uncertainty, and replicates Figure 3.

    Parameters
    ----------
    fits_step2 : list of PetroFit
        Despite the legacy name, this should be the Step 4 re-calibrated
        fits (one per cumulative stack depth K, aligned by index with
        ``averaged_models``), not the raw Step 2 per-model fits — see
        ``recalibrate_petrophysical_relationship``. ``run_pipeline.py``
        already passes ``fits_step4`` in through this argument.
    probe_layer_indices : list of int, optional
        0-based horizon index for each TDR probe, same order as
        ``sample_positions``/``theta_obs``. When supplied, must be the SAME
        array passed to ``EMmethod.recalibrate_petrophysical_relationship``
        upstream (``run_pipeline.py`` already does this), so the
        optimal-model calibration cross-plot below routes each point to the
        same horizon that ``best_fit_opt`` was actually fit against. When
        ``None``, falls back to the strided/modulo heuristic used
        elsewhere in the pipeline when probe depths aren't available.
    """
    num_models = len(models)
    rmse_by_k = []

    print("[STEP 4/5] Evaluating performance across all ensemble stack depths...")

    # 1. Loop through every stack depth K to evaluate the RMSE trajectory
    for k in range(num_models):
        avg_model_k = averaged_models[k]

        # FIX 3.10: pull the fit for *this* stack depth, not always index 0.
        # Despite the parameter's name, run_pipeline.py now passes the Step-4
        # recalibration fits here (fits_step2=fits_step4 at the call site),
        # and recalibrate_petrophysical_relationship() builds that list by
        # stacking averaged_models in order, so fits_step2[k] is the fit
        # calibrated on averaged_models[k] itself (K = k + 1). Using index 0
        # for every k previously collapsed the whole RMSE trajectory onto a
        # single fit and conflated Step 2 with Step 4.
        current_fit = fits_step2[k]

        # Extract forward-modeled Electrical Conductivity (EC) at probe locations.
        # NOTE: avg_model_k is an AVERAGED model (ensemble axis already
        # collapsed), so its dims are one fewer than the raw `models` array:
        #   single-survey -> 2D (num_points, num_layers)
        #   time-lapse    -> 3D (n_time, num_points, num_layers)
        #
        # FIX (layer-routing bug): the previous version pulled the EC
        # profile across ALL layers at each probe (`avg_model_k[t_idx, pos, :]`)
        # and averaged it into one scalar with `np.mean(ec_profile)`, then
        # called `ec_to_water_content(ec_predicted, current_fit)` with no
        # `probe_layer_indices`. Per that function's own docstring, omitting
        # `probe_layer_indices` makes it fall back to horizon 0's WS params
        # for every sample. So every probe -- regardless of installation
        # depth -- was scored using a layer-averaged EC pushed through
        # horizon 0's calibration only. This decorrelated the RMSE-vs-K
        # trajectory (and hence the optimal-K selection) from the actual
        # per-horizon fit quality, which is why it could disagree with
        # Step 4's own (correctly horizon-routed) best-K result.
        #
        # This version pulls each probe's EC from its OWN horizon only, and
        # passes probe_layer_indices through so ec_to_water_content() applies
        # that horizon's own (a, m, sigma_w, sigma_s). Falls back to the same
        # strided/modulo heuristic used elsewhere in the pipeline
        # (calibrate_petrophysical_relationship / _extract_ec_at_probes) when
        # probe_layer_indices isn't available, so behaviour is unchanged for
        # any caller that doesn't supply it.
        n_samples = len(sample_positions)
        num_layers_k = avg_model_k.shape[-1]

        if probe_layer_indices is not None:
            resolved_layer_indices = list(probe_layer_indices)
        elif n_samples >= num_layers_k:
            step = n_samples // num_layers_k
            resolved_layer_indices = [
                min(i // step, num_layers_k - 1) for i in range(n_samples)
            ]
        else:
            resolved_layer_indices = [i % num_layers_k for i in range(n_samples)]

        ec_predicted = []
        for idx, pos in enumerate(sample_positions):
            t_idx = sample_times[idx]
            layer_idx = min(max(int(resolved_layer_indices[idx]), 0), num_layers_k - 1)

            # FIX: averaged_models[k] already had the ensemble axis collapsed
            # by average_inverted_models() (np.sum(..., axis=0)), so it has
            # ONE FEWER dimension than the raw `models` ensemble array:
            #   single-survey: (points, layers)        -> ndim 2
            #   time-lapse:    (n_time, points, layers) -> ndim 3
            # Previously checked `ndim == 4`, which an averaged time-lapse
            # model can never satisfy, so real time-lapse data silently fell
            # into the single-file branch and indexed the time axis with a
            # spatial position index (IndexError on real datasets).
            if avg_model_k.ndim == 3:  # Time-lapse mode: (n_time, num_points, num_layers)
                # Extract EC for the specific time step, spatial index, AND
                # this probe's own horizon -- not averaged across horizons.
                ec_val = avg_model_k[t_idx, pos, layer_idx]
            else:                      # Single-file mode: (num_points, num_layers)
                ec_val = avg_model_k[pos, layer_idx]

            ec_predicted.append(ec_val)

        ec_predicted = np.array(ec_predicted)

        # Convert predicted Bulk EC to Volumetric Water Content (VWC / Theta) using Waxman-Smits,
        # routed per-probe through its own horizon's calibrated parameters.
        #
        # NOTE (real-data time-lapse spatial reference grid): a probe/epoch
        # pair whose reference-grid point wasn't covered by that particular
        # survey epoch (see geometry_utils.match_epochs_to_reference_grid and
        # run_pipeline.py Step 1) yields ec_val = NaN above, which propagates
        # through ec_to_water_content() into a NaN theta_pred entry for that
        # probe (documented on ec_to_water_content's own docstring). That is
        # expected and handled explicitly below via nanmean rather than mean,
        # so a handful of epoch-uncovered probes no longer poison the RMSE
        # for the ENTIRE stack depth K.
        theta_pred = EMmethod.ec_to_water_content(
            ec_predicted, current_fit, probe_layer_indices=resolved_layer_indices
        )

        # Compute RMSE for the current stack of size K = k + 1.
        # FIX (NaN propagation): plain np.mean turns a single NaN entry
        # (unmatched reference-grid point at this probe's survey epoch,
        # see note above) into a NaN for the WHOLE rmse_by_k[k] value, which
        # then silently breaks np.argmin(rmse_by_k) below (NaN comparisons
        # are never True, so argmin's behaviour with NaNs present is
        # undefined/unstable). np.nanmean correctly scores stack depth K
        # against only the probes that actually have data at each of their
        # respective survey epochs.
        _sq_err = (theta_obs - theta_pred) ** 2
        if np.all(np.isnan(_sq_err)):
            rmse_k = float("nan")
        else:
            rmse_k = float(np.sqrt(np.nanmean(_sq_err)))
        rmse_by_k.append(rmse_k)

    rmse_by_k = np.array(rmse_by_k)

    # 2. Identify the optimal stack depth (1-based index)
    if np.all(np.isnan(rmse_by_k)):
        raise ValueError(
            "[CRITICAL] Every candidate stack depth K produced an all-NaN "
            "RMSE (theta_obs vs. theta_pred never overlapped on a real "
            "value). This means none of the TDR probes had ANY valid EC at "
            "their own survey epoch on the time-lapse spatial reference "
            "grid -- check spatial_match_tolerance_m and the TDR<->EM "
            "window_m matching diagnostics (probe_em_matching*.png) before "
            "re-running."
        )
    optimal_idx = int(np.nanargmin(rmse_by_k))
    optimal_K = optimal_idx + 1
    min_rmse = rmse_by_k[optimal_idx]

    # 3. Compute reliability/uncertainty map (Coefficient of Variation) at optimal depth
    # Paper Eq. 8: CV_c = sd_c / rho_c  where rho_c and sd_c are WEIGHTED mean and
    # weighted standard deviation using the same RMSE-normalised weights (Eq. 6-7).
    top_k_indices = ranking[:optimal_K]
    selected_subset = models[top_k_indices].astype(float)   # (K, *spatial)

    # Recompute RMSE-normalised importance weights for the top-K subset (Eq. 6-7)
    top_k_rmse = np.array([fits_step2[i].rmse for i in top_k_indices], dtype=float)
    rmse_min_k, rmse_max_k = top_k_rmse.min(), top_k_rmse.max()
    spread_k = rmse_max_k - rmse_min_k
    if spread_k < 1e-15 or optimal_K == 1:
        weights_k = np.full(optimal_K, 1.0 / optimal_K, dtype=float)
    else:
        rmse_norm_k = (top_k_rmse - rmse_max_k) / (rmse_min_k - rmse_max_k)  # best→1
        total_k = rmse_norm_k.sum()
        weights_k = rmse_norm_k / total_k if total_k >= 1e-15 else np.full(optimal_K, 1.0 / optimal_K)

    # Broadcast weights over spatial/temporal trailing dimensions
    n_extra = selected_subset.ndim - 1
    w_view = weights_k.reshape((optimal_K,) + (1,) * n_extra)   # (K, 1, 1, …)

    # Weighted mean rho_c
    # NOTE (real-data time-lapse spatial reference grid): the NaN pattern in
    # `selected_subset` is identical across the ensemble (K) axis for any
    # fixed (t, point) -- it is a property of the survey epoch/reference-grid
    # point, not of which inversion parameters produced a given member (see
    # geometry_utils.match_epochs_to_reference_grid and
    # em_methodology.average_inverted_models). So this plain weighted sum
    # already yields NaN if and only if that cell genuinely has no data at
    # any of the K members, without needing a NaN-aware reduction here.
    subset_mean = np.sum(w_view * selected_subset, axis=0)

    # Weighted std sd_c with Bessel-like correction: sqrt(K/(K-1) * sum(w_i*(rho_i-rho_c)^2))
    if optimal_K == 1:
        subset_std = np.zeros_like(subset_mean)
    else:
        sq_dev = w_view * (selected_subset - subset_mean) ** 2
        subset_std = np.sqrt((optimal_K / (optimal_K - 1)) * np.sum(sq_dev, axis=0))

    # CV expressed as a fraction (> 1.0 means > 100%, i.e. high uncertainty).
    # FIX (NaN handling): a NaN `subset_mean` (all K members NaN at this
    # cell -- e.g. a real-data time-lapse reference-grid point uncovered at
    # every epoch that survived into the top-K subset) previously fell
    # through to the `else` branch of np.where (since comparisons with NaN
    # are always False) and was silently reported as CV = 0.0, i.e. "zero
    # uncertainty" for a cell that actually has NO data whatsoever. It is
    # now preserved as NaN, matching the convention used everywhere else in
    # this pipeline (see em_methodology.compute_coverage_map, which already
    # treats non-finite EC as "invalid" and will independently flag the same
    # cell via a low/negative `coverage` value used for plot hatching).
    cv_map = np.where(
        np.isfinite(subset_mean) & (np.abs(subset_mean) > 1e-5),
        subset_std / subset_mean,
        np.where(np.isfinite(subset_mean), 0.0, np.nan),
    )

    # 4. Generate and save Figure 3 Optimization Curve
    if generate_plots:
        _generate_figure_3(rmse_by_k, optimal_K, min_rmse, output_dir)
        _generate_uncertainty_plots(cv_map, depths0, output_dir)

    # -----------------------------------------------------------------------
    # Compute full-grid theta_calc at the optimal stack depth.
    # avg_model_opt shape: (n_time, n_points, n_layers) or (n_points, n_layers)
    # theta_calc will match that shape with EC→θ applied layer-by-layer.
    # -----------------------------------------------------------------------
    avg_model_opt = averaged_models[optimal_idx]
    best_fit_opt  = fits_step2[optimal_idx]

    # -----------------------------------------------------------------------
    # Petrophysical calibration cross-plot for the OPTIMAL ensemble model.
    # Mirrors step2_petro.plot_petrophysical_calibration, but plots EC from
    # the single RMSE-weighted K-averaged model (avg_model_opt) against
    # theta_obs, using the final Step 4 recalibrated parameters
    # (best_fit_opt) rather than a raw per-ensemble-member fit.
    # -----------------------------------------------------------------------
    if generate_plots:
        try:
            num_layers_opt = avg_model_opt.shape[-1]
            # _extract_ec_at_probes now returns (ec_extracted, kept_indices):
            # ec_at_probes_opt is horizon-routed AND NaN-dropped, so it lines
            # up 1:1 with best_fit_opt.observed_theta -- but it can be
            # SHORTER than len(sample_positions)/len(probe_layer_indices)
            # whenever the NaN-drop removed anything (real-data time-lapse
            # reference-grid gaps). probe_layer_indices below is still in
            # plain per-observation order, so it must be gathered through
            # kept_indices_opt before being used to mask ec_at_probes_opt /
            # theta_obs inside plot_optimal_petrophysical_calibration --
            # otherwise the mask and the values it's applied to disagree on
            # both order and length (finding #2's bug, reproduced here).
            ec_at_probes_opt, kept_indices_opt = _extract_ec_at_probes(
                avg_model=avg_model_opt,
                sample_positions=sample_positions,
                sample_times=sample_times,
                probe_layer_indices=probe_layer_indices,
                num_layers=num_layers_opt,
            )
            plot_optimal_petrophysical_calibration(
                ec_values=ec_at_probes_opt,
                theta_obs=best_fit_opt.observed_theta,
                petro_fit=best_fit_opt,
                K=optimal_K,
                rmse=min_rmse,
                output_path=os.path.join(output_dir, "petro_calibration_optimal.png"),
                probe_layer_indices=(
                    np.asarray(probe_layer_indices)[kept_indices_opt]
                    if probe_layer_indices is not None else None
                ),
            )
        except Exception as e:
            print(f"[WARN] Optimal-model petrophysical calibration plot failed: {e}")

    def _ec_grid_to_theta(ec_grid: np.ndarray, fit) -> np.ndarray:
        """Convert a (..., n_layers) EC grid to θ using per-horizon WS params.
        NaN EC cells (real-data time-lapse reference-grid points uncovered
        at this epoch) propagate to NaN θ cells -- expected and handled by
        downstream plotting, which already uses nan-aware reductions
        (e.g. plot_solution_diagnostics.plot_solution_swc uses np.nanmean)."""
        original_shape = ec_grid.shape
        n_layers = original_shape[-1]
        ec_flat = ec_grid.reshape(-1, n_layers)
        theta_flat = np.empty_like(ec_flat)
        for l_idx in range(n_layers):
            if l_idx < len(fit.horizon_fits):
                hf = fit.horizon_fits[l_idx]
                ec_col = np.clip(ec_flat[:, l_idx], 1e-6, None)
                ratio = hf.a * ec_col / (hf.sigma_w + hf.sigma_s)
                theta_flat[:, l_idx] = np.power(np.clip(ratio, 0.0, None), 1.0 / hf.m)
            else:
                theta_flat[:, l_idx] = 0.0
        return theta_flat.reshape(original_shape)

    theta_calc = _ec_grid_to_theta(avg_model_opt, best_fit_opt)

    # -----------------------------------------------------------------------
    # Compute coverage map: fraction of top-K members with valid EC per cell,
    # normalised to [-1, +1].  Delegates to EMmethod.compute_coverage_map so
    # the definition is single-sourced in em_methodology.py.
    # coverage > 0  → well constrained  (more than half K members valid)
    # coverage ≤ 0  → poorly constrained (hatched in XZ / XY plots)
    # -----------------------------------------------------------------------
    coverage = EMmethod.compute_coverage_map(models, top_k_indices)

    # Ensure coverage is always 3-D: (n_time, n_points, n_layers)
    if coverage.ndim == 2:
        coverage = coverage[np.newaxis, ...]

    # Ensure theta_calc is always 3-D: (n_time, n_points, n_layers)
    if theta_calc.ndim == 2:
        theta_calc = theta_calc[np.newaxis, ...]

    # Align cv_map to the same 3-D shape
    cv_map_3d = cv_map[np.newaxis, ...] if cv_map.ndim == 2 else cv_map

    return SelectionOutput(
        K=optimal_K,
        rmse=min_rmse,
        cv_map=cv_map_3d,
        best_fit=fits_step2[optimal_idx],
        theta_calc=theta_calc,
        coverage=coverage,
    )


def _generate_figure_3(rmse_by_k: np.ndarray, optimal_K: int, min_rmse: float, output_dir: str):
    """
    Renders the exact visual style of Fig. 3 from the article text.
    """
    num_models = len(rmse_by_k)
    x_axis = np.arange(1, num_models + 1)

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)

    # Plot the sequential performance trajectory of the ensemble stacks
    ax.plot(
        x_axis,
        rmse_by_k,
        color="black",
        linestyle="-",
        linewidth=1.2,
        marker="o",
        markersize=3,
        label="Ensemble Stack Depth (K)"
    )

    # Left-hand baseline: The green horizontal line matching the single best-ranked profile (K=1)
    ax.axhline(
        y=rmse_by_k[0],
        color="green",
        linestyle="--",
        linewidth=1.2,
        label="Highest-ranked single model"
    )

    # Green circle marking the optimal ensemble model balance point minimizing error
    ax.plot(
        optimal_K,
        min_rmse,
        color="green",
        marker="o",
        markersize=11,
        fillstyle="none",
        markeredgewidth=1.8,
        label="Lowest RMSE ensemble variant"
    )

    # Annotation text anchors matching Article layout labels
    ax.text(1, rmse_by_k[0], " Single Best Model ", verticalalignment="bottom", color="green", fontsize=9)
    ax.text(num_models, rmse_by_k[-1], " All Models ", verticalalignment="bottom", horizontalalignment="right", color="black", fontsize=9)

    # Labels and Plot Architecture
    ax.set_title("Ensemble Model Optimization Curve (Fig. 3 Replica)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of inverted models per ensemble model", fontsize=10)
    ax.set_ylabel("RMSE ($m^3 / m^3$ Volumetric Water Content)", fontsize=10)

    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figure_3_rmse_curve.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"[PLOTTING] Fig. 3 optimization plot written safely to: {fig_path}")


def _generate_uncertainty_plots(cv_map: np.ndarray, depths0: list[float], output_dir: str):
    """Placeholder helper to render cross-sectional spatial maps of your CV uncertainties."""
    # Custom 2D contour plotting logic matching your data structural layout would live here.
    pass


# ==========================================================================
# OPTIMAL-MODEL PETROPHYSICAL CALIBRATION CROSS-PLOT
# ==========================================================================

def _extract_ec_at_probes(
    avg_model: np.ndarray,
    sample_positions: list[int],
    sample_times: list[int],
    probe_layer_indices: list[int] | None,
    num_layers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract sigma_b (bulk EC) at each TDR probe location from an
    already-averaged (ensemble axis collapsed) EC model, routed to horizons
    with the SAME logic ``EMmethod.calibrate_petrophysical_relationship``
    uses internally. This guarantees the points plotted here line up
    exactly with ``best_fit_opt.observed_theta`` / ``best_fit_opt.horizon_fits``
    (the fit ``recalibrate_petrophysical_relationship`` actually produced
    for ``avg_model``), rather than an independently-derived pairing.

    Parameters
    ----------
    avg_model : np.ndarray
        Shape ``([n_time,] num_measured_points, num_layers)`` — NO ensemble
        axis (already the RMSE-weighted average at one stack depth K).
    sample_positions, sample_times : list of int
        Same arrays passed into ``recalibrate_petrophysical_relationship``.
    probe_layer_indices : list of int or None
        Same array passed into ``recalibrate_petrophysical_relationship``.
        When ``None``, falls back to the strided/modulo heuristic (mirrors
        ``calibrate_petrophysical_relationship``'s own fallback).
    num_layers : int
        ``avg_model.shape[-1]``.

    Returns
    -------
    ec_extracted : np.ndarray
        Flat sigma_b values, concatenated per horizon in the same order as
        ``PetroFit.observed_theta`` / ``PetroFit.predicted_theta``, with
        the same NaN-observation drop
        ``em_methodology.calibrate_petrophysical_relationship`` applies
        (real-data time-lapse probe/epoch pairs whose spatial
        reference-grid point wasn't covered by that epoch). Because those
        entries are dropped rather than left as NaN, ``ec_extracted`` can be
        SHORTER than ``len(sample_positions)``.
    kept_indices : np.ndarray
        For each element of ``ec_extracted`` (same order), the index into
        the ORIGINAL length-``len(sample_positions)`` per-observation
        arrays it came from. Callers must gather any other per-observation
        array (notably ``probe_layer_indices`` itself) through this before
        pairing it element-wise with ``ec_extracted`` /
        ``best_fit_opt.observed_theta`` — see
        ``geometry_utils.route_ec_to_horizons`` and finding #2 in
        INDEXING_AUDIT_REPORT.md for the class of bug this prevents.
    """
    # Horizon routing now delegates to the single shared implementation in
    # geometry_utils.py (pipeline_improvement_report.md, section 6 / action
    # item 3) instead of an independently reimplemented copy of the same
    # strided/modulo/explicit-index logic used by
    # em_methodology.calibrate_petrophysical_relationship and by
    # step2_petro.py::_extract_ec_for_plots. Note: this function's own
    # docstring's ``timelapse = avg_model.ndim == 3`` convention (no
    # ensemble axis) matches what ``geometry_utils.route_ec_to_horizons``
    # expects directly.
    #
    # route_ec_to_horizons now returns a THIRD value, kept_indices (see
    # geometry_utils.py finding #3 / this file's regression noted in
    # INDEXING_AUDIT_REPORT.md): it replicates calibrate_petrophysical_
    # relationship's post-routing NaN-observation drop, so ec_extracted can
    # come back shorter than len(sample_positions). Unpacking only two
    # values here (as before) raises "too many values to unpack" on every
    # call -- silently swallowed by run_select_assess's try/except, which
    # kills petro_calibration_optimal.png without crashing the pipeline.
    ec_extracted, _horizon_slices, kept_indices = geometry_utils.route_ec_to_horizons(
        ec_array=avg_model,
        sample_positions=sample_positions,
        sample_times=sample_times,
        probe_layer_indices=probe_layer_indices,
        num_layers=num_layers,
    )
    return ec_extracted, kept_indices


def plot_optimal_petrophysical_calibration(
    ec_values: np.ndarray,
    theta_obs: np.ndarray,
    petro_fit,
    K: int,
    rmse: float,
    output_path: str,
    probe_layer_indices: np.ndarray | None = None,
) -> None:
    """
    Cross-plot of inverted bulk EC from the OPTIMAL ensemble model (the
    RMSE-weighted average at stack depth K, ``averaged_models[K-1]``)
    against the observed TDR water content theta, using the final Step 4
    recalibrated Waxman-Smits parameters (``best_fit_opt`` /
    ``petro_fit``). Points and fit curves are colored by soil horizon.

    Mirrors ``step2_petro.plot_petrophysical_calibration`` (the Step 2
    per-raw-ensemble-member version), but here there is exactly ONE EC
    value per probe (from the single averaged model) instead of one per
    raw ensemble member, and the overlaid Waxman-Smits curves are the
    final recalibrated parameters rather than a Step 2 single-member fit.

    Saved to ``output_path`` (typically
    ``<output_dir>/petro_calibration_optimal.png``).
    """
    num_horizons = len(petro_fit.horizon_fits)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, num_horizons))

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # ---- Scatter of observed (EC, theta) pairs, colored by horizon -------
    if probe_layer_indices is not None and len(probe_layer_indices) == len(ec_values):
        probe_layer_indices = np.asarray(probe_layer_indices)
        for hz_idx in range(num_horizons):
            mask = (probe_layer_indices == hz_idx)
            if np.any(mask):
                ax.scatter(
                    ec_values[mask], theta_obs[mask],
                    color=colors[hz_idx], edgecolor='black', s=80, zorder=4,
                    label=f'Observed Data (Hz {hz_idx})',
                )
    else:
        # Fallback: split ec_values/theta_obs using each horizon's own
        # observed_theta length, mirroring plot_petrophysical_calibration.
        start_idx = 0
        matched_any = False
        for hz in petro_fit.horizon_fits:
            n_points = len(hz.observed_theta)
            end_idx = start_idx + n_points
            if end_idx <= len(ec_values):
                ax.scatter(
                    ec_values[start_idx:end_idx], theta_obs[start_idx:end_idx],
                    color=colors[hz.horizon], edgecolor='black', s=80, zorder=4,
                    label=f'Observed Data (Hz {hz.horizon})',
                )
                start_idx = end_idx
                matched_any = True
            else:
                matched_any = False
                break
        if not matched_any:
            ax.scatter(ec_values, theta_obs, color='darkorange', edgecolor='black',
                       s=75, zorder=3, label='Optimal-model TDR Data Pairs')

    # ---- Theoretical Waxman-Smits curves, per horizon ---------------------
    if len(ec_values) > 0:
        _finite_ec = ec_values[np.isfinite(ec_values)]
        if _finite_ec.size > 0:
            ec_line = np.linspace(_finite_ec.min() * 0.8, _finite_ec.max() * 1.2, 100)
        else:
            ec_line = np.linspace(0.1, 100.0, 100)
    else:
        ec_line = np.linspace(0.1, 100.0, 100)

    for hz in petro_fit.horizon_fits:
        theta_line = EMmethod.waxman_smits_theta(ec_line, hz.a, hz.m, hz.sigma_w, hz.sigma_s)
        is_calibrated = getattr(hz, 'is_calibrated', True)
        if is_calibrated:
            ax.plot(
                ec_line, theta_line,
                color=colors[hz.horizon], linestyle='--', linewidth=2, zorder=2,
                label=(
                    f'WS Fit Hz {hz.horizon} '
                    f'(a={hz.a:.2f}, m={hz.m:.2f}, '
                    f'σ_w={hz.sigma_w:.1f}, σ_s={hz.sigma_s:.1f}, '
                    f'RMSE={hz.rmse:.4f})'
                ),
            )
        else:
            ax.plot(
                ec_line, theta_line,
                color=colors[hz.horizon], linestyle=':', linewidth=1.5,
                alpha=0.45, zorder=1,
                label=(
                    f'Hz {hz.horizon} — UNCALIBRATED (prior only, no TDR probe): '
                    f'a={hz.a:.2f}, m={hz.m:.2f}, '
                    f'σ_w={hz.sigma_w:.1f}, σ_s={hz.sigma_s:.1f}'
                ),
            )

    ax.set_title(
        f"Optimal-Model Petrophysical Calibration (Waxman-Smits) — "
        f"K={K}, RMSE={rmse:.4f} m³/m³",
        fontsize=12, pad=12,
    )
    ax.set_xlabel("Inverted Bulk EC — optimal K-averaged model [mS/m]", fontsize=10)
    ax.set_ylabel("Volumetric Water Content ($\\theta$) [m$^3$/m$^3$]", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', frameon=True, fontsize=7, bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Optimal-model petrophysical calibration graph exported to '{output_path}'")
