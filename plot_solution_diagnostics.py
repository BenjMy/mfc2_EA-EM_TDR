#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_solution_diagnostics.py
=============================
Final-solution diagnostics for the EA-EM pipeline.

Three entry points, all writing under ``<output_dir>/solution/``:

  plot_solution_swc(...)
      Vertical soil-water-content (theta) profiles of the RMSE-weighted
      top-K "optimal" solution selected in Step 5 (``selected_output.theta_calc``).
      When synthetic ground truth is available it is overlaid as a dashed
      reference so the recovered solution can be sanity-checked at a glance.

  plot_solution_ec_bulk(...)
      Same idea but for the optimal bulk-EC model, i.e.
      ``averaged_models[selected_output.K - 1]`` from Step 3.

  forward_simulate_eca_optimal(...)
      Forward-simulates the apparent EC (ECa) that the optimal EC_bulk
      solution *would* produce at the surface, using emagpy's forward
      model (the same call pattern used in ``data_synthetic.py``
      to build the synthetic ground truth, and the same "inject the
      weighted-average model" idea used in
      ``plot_emagpy_diagnostics.run_emagpy_plots_optimal``). The simulated
      ECa is compared against the raw survey ECa that was actually fed to
      the inversion, giving a direct forward-model check of the final
      pipeline output.

Typical call from run_pipeline.py (Step 6d/6e, right after Step 6b)
--------------------------------------------------------------------
    import plot_solution_diagnostics as psol

    psol.plot_solution_swc(
        theta_calc     = selected_output.theta_calc,
        depths0        = DEPTHS0,
        layer_centers  = synthetic_geometry["layer_centers"],
        output_dir     = OUTPUT_DIR,
        theta_true     = theta_true_realization,   # None in real-data mode
        K              = selected_output.K,
        rmse           = selected_output.rmse,
        scenario_name  = cfg.name if cfg else "default",
        timelapse      = timelapse,
    )

    psol.plot_solution_ec_bulk(
        ec_optimal     = averaged_models[selected_output.K - 1],
        depths0        = DEPTHS0,
        layer_centers  = synthetic_geometry["layer_centers"],
        output_dir     = OUTPUT_DIR,
        ec_bulk_true   = ec_bulk_true_realization,  # None in real-data mode
        K              = selected_output.K,
        rmse           = selected_output.rmse,
        scenario_name  = cfg.name if cfg else "default",
        timelapse      = timelapse,
    )

    psol.forward_simulate_eca_optimal(
        csv_paths        = EM_FNAMES,
        averaged_models  = averaged_models,
        selected_output  = selected_output,
        depths0          = DEPTHS0,
        output_dir       = OUTPUT_DIR,
        x_transect       = x_transect,
        forward_model    = _base_inv_kwargs.get("forwardModel", "CS"),
        scenario_name    = cfg.name if cfg else "default",
        matched_indices  = REAL_DATA_MATCHED_INDICES if (data_source == "real" and timelapse) else None,
    )
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig: plt.Figure, path: str, dpi: int = 180) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    [SOLUTION] saved → {os.path.relpath(path)}")


def _time_slice(arr: np.ndarray, t_idx: int, timelapse: bool) -> np.ndarray:
    """Return the (n_pts, n_layers) slice for time step `t_idx`."""
    arr = np.asarray(arr)
    return arr[t_idx] if timelapse else arr


def _n_time_of(arr: np.ndarray, timelapse: bool) -> int:
    arr = np.asarray(arr)
    return arr.shape[0] if timelapse else 1


# Columns in an emagpy-style survey CSV that are NOT coil/ECa readings.
_NON_COIL_COLUMNS = {"x", "y", "utm_x", "utm_y", "elev", "line", "id", "lat", "lon", "date"}


def _infer_coils_from_csv(csv_path: str) -> List[str]:
    header_df = pd.read_csv(csv_path, nrows=1)
    return [c for c in header_df.columns if c.lower() not in _NON_COIL_COLUMNS]


def _as_csv_list(csv_paths) -> List[str]:
    return list(csv_paths) if isinstance(csv_paths, (list, tuple)) else [csv_paths]


# ---------------------------------------------------------------------------
# Public entry point 1 — solution SWC (theta) profiles
# ---------------------------------------------------------------------------

def plot_solution_swc(
    theta_calc: np.ndarray,
    depths0: Sequence[float],
    layer_centers: Sequence[float],
    output_dir: str,
    theta_true: Optional[np.ndarray] = None,
    K: Optional[int] = None,
    rmse: Optional[float] = None,
    scenario_name: str = "default",
    timelapse: bool = False,
) -> str:
    """
    Vertical soil-water-content (SWC / theta) profiles of the pipeline's
    final solution — the RMSE-weighted top-K averaged model chosen in
    Step 5 (``selected_output.theta_calc``) — at every measurement point
    and time step.

    When `theta_true` (the synthetic ground truth used to build the
    dataset, e.g. `theta_true_realization` from Step 0) is supplied, its
    spatial-mean profile is overlaid as a dashed reference curve so the
    recovered solution can be checked against the known truth. Pass
    `theta_true=None` for real-data runs (no ground truth available).

    Saved to '<output_dir>/solution/solution_swc_profiles.png'.

    Returns
    -------
    str : path to the saved figure.
    """
    sol_dir = _ensure_dir(os.path.join(output_dir, "solution"))
    theta_calc = np.asarray(theta_calc)
    n_time = _n_time_of(theta_calc, timelapse)
    num_pts = theta_calc.shape[-2]

    fig, axes = plt.subplots(1, n_time, figsize=(4.5 * n_time, 5.5), sharey=True)
    axes = [axes] if n_time == 1 else list(axes)
    colors_pts = plt.cm.plasma(np.linspace(0, 0.85, num_pts))

    for t in range(n_time):
        ax = axes[t]
        theta_t = _time_slice(theta_calc, t, timelapse)  # (n_pts, n_layers)

        for p in range(num_pts):
            ax.plot(theta_t[p, :], layer_centers, marker="o", color=colors_pts[p],
                    alpha=0.55, linestyle="-", linewidth=1)

        mean_profile = np.nanmean(theta_t, axis=0)
        ax.plot(mean_profile, layer_centers, color="black", linestyle="-",
                linewidth=2.5, label="Solution spatial mean" if t == 0 else "")

        if theta_true is not None:
            theta_true_t = _time_slice(np.asarray(theta_true), t, timelapse)
            true_mean = np.nanmean(theta_true_t, axis=0)
            ax.plot(true_mean, layer_centers, color="crimson", linestyle="--",
                    linewidth=2.0, marker="x",
                    label="Ground truth spatial mean" if t == 0 else "")

        ax.set_title(f"Solution SWC — Time Step {t + 1}", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$\theta$ [m$^3$/m$^3$]")
        ax.grid(True, linestyle=":", alpha=0.6)
        if t == 0:
            ax.set_ylabel("Depth [m]", fontsize=11)
            ax.invert_yaxis()
            ax.legend(loc="lower left", fontsize=8)

    _kr = f"  —  K={K}, RMSE={rmse:.4f} m³/m³" if (K is not None and rmse is not None) else ""
    fig.suptitle(f"Final Solution — Soil Water Content Profiles — {scenario_name}{_kr}", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(sol_dir, "solution_swc_profiles.png")
    _save(fig, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Public entry point 2 — solution EC_bulk profiles
# ---------------------------------------------------------------------------

def plot_solution_ec_bulk(
    ec_optimal: np.ndarray,
    depths0: Sequence[float],
    layer_centers: Sequence[float],
    output_dir: str,
    ec_bulk_true: Optional[np.ndarray] = None,
    K: Optional[int] = None,
    rmse: Optional[float] = None,
    scenario_name: str = "default",
    timelapse: bool = False,
) -> str:
    """
    Vertical bulk-EC profiles of the pipeline's final optimal solution
    (``averaged_models[selected_output.K - 1]`` from Step 3), at every
    measurement point and time step.

    When `ec_bulk_true` (the synthetic ground-truth EC_bulk used to
    generate the ECa data, e.g. sliced from
    ``synthetic_time_lapse_ec.npy[REALIZATION_INDEX]``) is supplied, its
    spatial-mean profile is overlaid as a dashed reference. Pass
    `ec_bulk_true=None` for real-data runs.

    Saved to '<output_dir>/solution/solution_ec_bulk_profiles.png'.

    Returns
    -------
    str : path to the saved figure.
    """
    sol_dir = _ensure_dir(os.path.join(output_dir, "solution"))
    ec_optimal = np.asarray(ec_optimal)
    n_time = _n_time_of(ec_optimal, timelapse)
    num_pts = ec_optimal.shape[-2]

    fig, axes = plt.subplots(1, n_time, figsize=(4.5 * n_time, 5.5), sharey=True)
    axes = [axes] if n_time == 1 else list(axes)
    colors_pts = plt.cm.plasma(np.linspace(0, 0.85, num_pts))

    for t in range(n_time):
        ax = axes[t]
        ec_t = _time_slice(ec_optimal, t, timelapse)  # (n_pts, n_layers)

        for p in range(num_pts):
            ax.plot(ec_t[p, :], layer_centers, marker="o", color=colors_pts[p],
                    alpha=0.55, linestyle="-", linewidth=1)

        mean_profile = np.nanmean(ec_t, axis=0)
        ax.plot(mean_profile, layer_centers, color="black", linestyle="-",
                linewidth=2.5, label="Solution spatial mean" if t == 0 else "")

        if ec_bulk_true is not None:
            ec_true_t = _time_slice(np.asarray(ec_bulk_true), t, timelapse)
            true_mean = np.nanmean(ec_true_t, axis=0)
            ax.plot(true_mean, layer_centers, color="crimson", linestyle="--",
                    linewidth=2.0, marker="x",
                    label="Ground truth spatial mean" if t == 0 else "")

        ax.set_title(f"Solution EC$_{{bulk}}$ — Time Step {t + 1}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Bulk EC [mS/m]")
        ax.grid(True, linestyle=":", alpha=0.6)
        if t == 0:
            ax.set_ylabel("Depth [m]", fontsize=11)
            ax.invert_yaxis()
            ax.legend(loc="lower left", fontsize=8)

    _kr = f"  —  K={K}, RMSE={rmse:.4f} m³/m³" if (K is not None and rmse is not None) else ""
    fig.suptitle(f"Final Solution — Bulk EC Profiles (RMSE-weighted top-K) — {scenario_name}{_kr}", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(sol_dir, "solution_ec_bulk_profiles.png")
    _save(fig, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Public entry point 3 — forward-simulate ECa from the optimal model
# ---------------------------------------------------------------------------

def forward_simulate_eca_optimal(
    csv_paths,
    averaged_models: list,
    selected_output: Any,
    depths0: Sequence[float],
    output_dir: str,
    coils_list: Optional[List[str]] = None,
    x_transect: Optional[np.ndarray] = None,
    forward_model: str = "CS",
    scenario_name: str = "default",
    save_csv: bool = True,
    matched_indices: Optional[Sequence[np.ndarray]] = None,
) -> List[str]:
    """
    Forward-simulate the apparent EC (ECa) that the pipeline's OPTIMAL
    bulk-EC solution — ``averaged_models[selected_output.K - 1]`` from
    Step 3 — would produce at the surface, using emagpy's forward model
    (same ``Problem.forward()`` call pattern as
    ``data_synthetic.forward_waxman_smits`` → ``Problem.forward``
    and the model-injection idea from
    ``plot_emagpy_diagnostics.run_emagpy_plots_optimal``).

    For each time step this renders two diagnostic figures and (optionally)
    a CSV, comparing the forward-simulated ECa against the RAW survey ECa
    that was actually fed to the inversion:

      * ``eca_forward_vs_observed_t{t}.png`` — per-coil profile along the
        transect (observed vs. forward-simulated from the optimal model).
      * ``eca_forward_1to1_t{t}.png`` — 1:1 scatter across all coils/points,
        annotated with the RMSE between observed and simulated ECa.
      * ``eca_forward_optimal_t{t}.csv`` — the underlying numbers.

    All outputs are written to ``<output_dir>/solution/eca_forward_optimal/``.

    Parameters
    ----------
    csv_paths : str or list of str
        Survey CSV path(s) (one per time step) — the same files used to
        drive the ensemble inversion (e.g. ``EM_FNAMES``).
    averaged_models : list
        Step 3 output; ``averaged_models[K-1]`` is the RMSE-weighted mean
        EC model, shape (n_time, n_pts, n_layers) or (n_pts, n_layers).
    selected_output : Any
        Step 4/5 ``SelectionOutput``; must expose ``.K`` and ``.rmse``.
    depths0 : list of float
        Layer bottom depths [m] (length = n_layers - 1).
    coils_list : list of str, optional
        Coil configuration names matching the model's forward response
        (e.g. ``['VCP0.5','HCP0.5', ...]``). If omitted, inferred from the
        CSV header (every column not in {x, y, utm_x, utm_y, ...}).
    x_transect : ndarray, optional
        Along-transect x-coordinates for the profile plot. Falls back to
        the survey CSV's own 'x' column if omitted.
    forward_model : str
        emagpy forward-model code (e.g. "CS", "FSlin"). Should match the
        model used elsewhere in the pipeline for consistency.
    matched_indices : sequence of np.ndarray, optional
        Real-data, time-lapse runs ONLY (``INDEXING_AUDIT_REPORT.md``
        finding #5). ``run_pipeline.py`` builds a union spatial reference
        grid across survey epochs
        (``geometry_utils.match_epochs_to_reference_grid``) and remaps
        ``averaged_models`` onto it (``_remap_combo_to_reference``), so
        ``avg_model_opt``'s spatial axis — and therefore ``n_pts`` here —
        is that REFERENCE grid, not any one epoch's own raw-CSV row
        order/count. ``matched_indices[t_idx]`` is that epoch's
        row → reference-grid-index map (same array
        ``geometry_utils.match_epochs_to_reference_grid`` returned, and
        the same one ``_remap_combo_to_reference`` used to build
        ``averaged_models`` itself): ``matched_indices[t_idx][j]`` is the
        reference-grid index that epoch's raw row ``j`` was matched to, or
        ``-1`` if unmatched. When supplied, ``eca_obs`` (and ``xpos``, if
        not overridden by ``x_transect``) are gathered through this map —
        reference-grid point ``i`` is filled from raw row
        ``argwhere(matched_indices[t_idx] == i)`` when one exists, else
        left ``NaN`` — so they stay row-aligned with ``eca_pred``, which
        is computed directly from the (already reference-grid-ordered)
        ``ec_slice``. When ``None`` (synthetic data, or a real-data
        single-survey run with no remap), behaviour is unchanged: raw CSV
        rows are read directly in their own order, which IS the model's
        own order in that case.

    Returns
    -------
    list of (t_idx, rmse_eca) tuples.
    """
    from emagpy import Problem

    sol_dir = _ensure_dir(os.path.join(output_dir, "solution", "eca_forward_optimal"))
    csv_list = _as_csv_list(csv_paths)

    K_opt = selected_output.K
    avg_model_opt = np.asarray(averaged_models[K_opt - 1])
    timelapse = avg_model_opt.ndim == 3
    n_time = avg_model_opt.shape[0] if timelapse else 1

    if coils_list is None:
        coils_list = _infer_coils_from_csv(csv_list[0])

    print(f"\n{'='*72}")
    print(f" [SOLUTION] Forward-simulating ECa from optimal model  "
          f"(K={K_opt}, RMSE={selected_output.rmse:.5f})")
    print(f"   forwardModel : {forward_model}   coils : {coils_list}")
    print(f"   time steps   : {n_time}")
    if matched_indices is not None:
        print(f"   observed ECa : gathered via matched_indices "
              f"(reference-grid order, real-data time-lapse)")
    print(f"{'='*72}\n")

    results = []
    for t_idx, csv_path in enumerate(csv_list):
        ec_slice = avg_model_opt[t_idx] if timelapse else avg_model_opt  # (n_pts, n_layers)
        n_pts = ec_slice.shape[0]
        depths_arr = np.tile(np.array(depths0, dtype=float), (n_pts, 1))

        models_input = [ec_slice.astype(float)]
        depths_input = [depths_arr]

        k = Problem()
        k.models = models_input
        k.depths = depths_input
        try:
            res_forward = k.forward(
                forwardModel=forward_model,
                coils=coils_list,
                models=models_input,
                depths=depths_input,
            )
        except Exception as e:
            print(f"    [WARN] Forward simulation failed for t={t_idx}: {e}")
            continue

        eca_pred = np.asarray(res_forward[0].values)  # (n_pts, n_coils)

        obs_df = pd.read_csv(csv_path)

        if matched_indices is not None and t_idx < len(matched_indices):
            # Real-data time-lapse run: ec_slice/eca_pred/n_pts live on the
            # shared spatial reference grid, NOT this epoch's own raw-CSV
            # row order/count -- see the matched_indices docstring above.
            # Gather eca_obs (and xpos, when not overridden by
            # x_transect) into that same reference-grid order, NaN-filling
            # any reference point this epoch didn't cover -- mirrors
            # run_pipeline.py's _remap_combo_to_reference exactly, so
            # eca_obs/eca_pred stay row-aligned even though this epoch may
            # not have covered every reference-grid point.
            idx_map = np.asarray(matched_indices[t_idx])
            raw_eca = obs_df[coils_list].to_numpy(dtype=float)
            raw_x = obs_df["x"].to_numpy(dtype=float)

            eca_obs = np.full((n_pts, len(coils_list)), np.nan, dtype=float)
            xpos_from_csv = np.full(n_pts, np.nan, dtype=float)
            valid = idx_map >= 0
            if np.any(valid):
                eca_obs[valid] = raw_eca[idx_map[valid]]
                xpos_from_csv[valid] = raw_x[idx_map[valid]]

            xpos = (np.asarray(x_transect[:n_pts], dtype=float) if x_transect is not None
                    else xpos_from_csv)
        else:
            eca_obs = obs_df[coils_list].to_numpy(dtype=float)[:n_pts]
            xpos = (np.asarray(x_transect[:n_pts], dtype=float) if x_transect is not None
                    else obs_df["x"].to_numpy(dtype=float)[:n_pts])

        # --- per-coil along-transect comparison ---------------------------
        n_coils = len(coils_list)
        fig, axes = plt.subplots(n_coils, 1, figsize=(9, 2.1 * n_coils), sharex=True)
        axes = [axes] if n_coils == 1 else list(axes)
        order = np.argsort(xpos)
        for c_idx, coil in enumerate(coils_list):
            ax = axes[c_idx]
            ax.plot(xpos[order], eca_obs[order, c_idx], "o-", color="tab:gray",
                     alpha=0.75, ms=4, label="Observed (input survey)" if c_idx == 0 else "")
            ax.plot(xpos[order], eca_pred[order, c_idx], "s--", color="tab:red",
                     alpha=0.9, ms=4, label="Forward-simulated (optimal model)" if c_idx == 0 else "")
            ax.set_ylabel(f"{coil}\nECa [mS/m]", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.5)
            if c_idx == 0:
                ax.legend(loc="upper right", fontsize=7)
        axes[-1].set_xlabel("x [m]")
        fig.suptitle(
            f"Forward-simulated ECa from optimal solution "
            f"(K={K_opt}, RMSE={selected_output.rmse:.4f} m³/m³) — "
            f"{scenario_name} — t={t_idx}", fontsize=10,
        )
        plt.tight_layout()
        _save(fig, os.path.join(sol_dir, f"eca_forward_vs_observed_t{t_idx:02d}.png"))

        # --- 1:1 scatter across all coils/points ---------------------------
        # NOTE: eca_obs may now contain NaN entries (reference-grid points
        # this epoch didn't cover, when matched_indices was supplied) --
        # use nan-aware reductions throughout so a handful of gaps don't
        # poison the whole plot's axis limits or the RMSE.
        rmse_eca = float(np.sqrt(np.nanmean((eca_obs - eca_pred) ** 2)))
        fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
        ax2.scatter(eca_obs.flatten(), eca_pred.flatten(), s=16, alpha=0.6, color="tab:blue")
        lo = float(np.nanmin([np.nanmin(eca_obs), np.nanmin(eca_pred)]))
        hi = float(np.nanmax([np.nanmax(eca_obs), np.nanmax(eca_pred)]))
        ax2.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        ax2.set_xlabel("Observed ECa [mS/m]  (input survey)")
        ax2.set_ylabel("Forward-simulated ECa [mS/m]  (optimal model)")
        ax2.set_title(f"t={t_idx} — RMSE = {rmse_eca:.3f} mS/m")
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(fontsize=8)
        plt.tight_layout()
        _save(fig2, os.path.join(sol_dir, f"eca_forward_1to1_t{t_idx:02d}.png"))

        # --- CSV export ------------------------------------------------------
        if save_csv:
            out_df = pd.DataFrame({"x": xpos})
            for c_idx, coil in enumerate(coils_list):
                out_df[f"{coil}_observed"]       = eca_obs[:, c_idx]
                out_df[f"{coil}_forward_optimal"] = eca_pred[:, c_idx]
            out_csv_path = os.path.join(sol_dir, f"eca_forward_optimal_t{t_idx:02d}.csv")
            out_df.to_csv(out_csv_path, index=False)
            print(f"    [SOLUTION] saved → {os.path.relpath(out_csv_path)}")

        print(f"  ── t={t_idx}  RMSE(ECa) = {rmse_eca:.4f} mS/m")
        results.append((t_idx, rmse_eca))

    print(f"\n[SOLUTION] Forward-simulated ECa (optimal model) written to '{sol_dir}'")
    return results
