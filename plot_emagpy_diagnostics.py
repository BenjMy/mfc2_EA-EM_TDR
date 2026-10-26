#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_emagpy_diagnostics.py
==========================
Wrapper around emagpy's built-in visualisation methods for the EA-EM pipeline.

Two complementary entry points:

  run_emagpy_plots_per_realisation(...)
      Renders showPseudo / showResults / showDepths / showMap / showDist /
      showMisfit for EVERY individual inversion realisation (each PARAM_GRID
      combo × conds0 variant × time-step).  Each emagpy Problem is built from
      the raw CSV survey, so the built-in plots reflect a single realisation's
      own inversion — not the ensemble average.

  run_emagpy_plots_optimal(...)
      Renders the SAME six panels but for the OPTIMAL ensemble model, i.e. the
      RMSE-weighted average of the top-K realisations identified by Step 4/5.
      Because emagpy's plot methods require a Problem object with internal
      `models` / `surveys` state, we inject the weighted-average EC directly
      into a fresh Problem before plotting.

Both functions save PNGs under ``<output_dir>/emagpy_diag/``.

Typical call from run_pipeline.py (Step 6b)
-------------------------------------------
    import plot_emagpy_diagnostics as pediag

    # --- per-realisation ---
    pediag.run_emagpy_plots_per_realisation(
        csv_paths      = EM_FNAMES,
        tasks          = tasks,          # list built in Step 1
        models_clean   = models_clean,   # (ensemble, [n_time,] n_pts, n_layers)
        depths_clean   = depths_clean,
        depths0        = DEPTHS0,
        output_dir     = OUTPUT_DIR,
        mode           = mode,
    )

    # --- optimal (weighted-average) model ---
    pediag.run_emagpy_plots_optimal(
        csv_paths      = EM_FNAMES,
        selected_output= selected_output,   # SelectionOutput from Step 4/5
        averaged_models= averaged_models,    # list built by Step 3
        depths0        = DEPTHS0,
        output_dir     = OUTPUT_DIR,
        mode           = mode,
    )
"""

from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a hard dependency elsewhere
    def tqdm(iterable, *args, **kwargs):
        return iterable

import em_methodology as EMmethod


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig: plt.Figure, path: str, dpi: int = 180) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    [DIAG] saved → {os.path.relpath(path)}")


def _safe_call(ax, fn, label: str) -> None:
    """Call fn(ax) and show a red error note on failure instead of crashing."""
    try:
        fn(ax)
    except Exception as e:
        ax.text(0.5, 0.5, f"{label}\nfailed:\n{e}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=7, color="red")
        ax.set_title(f"{label} — error", fontsize=9)


# ---------------------------------------------------------------------------
# Panel renderers  (all accept an already-inverted Problem `k`)
# ---------------------------------------------------------------------------

def _panel_pseudo(k, ax):
    k.showPseudo(index=0, ax=ax, dist=True)
    ax.set_title("Raw ECa – pseudo-section (XZ)")


def _panel_results(k, ax):
    k.showResults(index=0, ax=ax, dist=True, cmap="viridis_r")
    ax.set_title("Inverted EC model (XZ)")


def _panel_map(k, ax):
    k.showMap(index=0, ax=ax, contour=True, pts=True, cmap="viridis_r")
    ax.set_title("ECa plan-view map (XY)")


def _panel_dist(k, ax):
    k.showDist(index=0, ax=ax, nbins=20)
    ax.set_title("ECa distribution")


def _panel_misfit(k, ax):
    k.showMisfit(index=0, ax=ax)
    ax.set_title("ECa misfit (obs vs pred)")


def _panel_depth(k, ax, idepth: int):
    k.showDepths(index=0, idepth=idepth, ax=ax, contour=True, cmap="viridis_r")
    ax.set_title(f"EC depth-slice layer {idepth} (XY)")


def _summary_grid(k, n_layers: int, title: str, out_path: str) -> None:
    """2×3 overview canvas (pseudo, results, dist, map, depth-layer0, misfit)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(title, fontsize=12)
    panels = [
        ("showPseudo",      lambda ax: _panel_pseudo(k, ax)),
        ("showResults (XZ)",lambda ax: _panel_results(k, ax)),
        ("showDist",        lambda ax: _panel_dist(k, ax)),
        ("showMap (XY)",    lambda ax: _panel_map(k, ax)),
        (f"showDepths L0 (XY)", lambda ax: _panel_depth(k, ax, 0)),
        ("showMisfit",      lambda ax: _panel_misfit(k, ax)),
    ]
    for ax, (lbl, fn) in zip(axes.flat, panels):
        _safe_call(ax, fn, lbl)
    plt.tight_layout()
    _save(fig, out_path, dpi=150)


def _individual_panels(k, n_layers: int, t_dir: str, prefix: str) -> None:
    """Save every panel as its own PNG."""
    for lbl, fn, suffix in [
        ("showPseudo",   lambda ax: _panel_pseudo(k, ax),   "pseudo"),
        ("showResults",  lambda ax: _panel_results(k, ax),  "results_xz"),
        ("showMap",      lambda ax: _panel_map(k, ax),      "map_xy"),
        ("showDist",     lambda ax: _panel_dist(k, ax),     "dist"),
        ("showMisfit",   lambda ax: _panel_misfit(k, ax),   "misfit"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        _safe_call(ax, fn, lbl)
        _save(fig, os.path.join(t_dir, f"{prefix}_{suffix}.png"))

    for i in range(n_layers):
        fig, ax = plt.subplots(figsize=(7, 5))
        _safe_call(ax, lambda ax, i=i: _panel_depth(k, ax, i), f"showDepths L{i}")
        _save(fig, os.path.join(t_dir, f"{prefix}_depth_layer{i}_xy.png"))


# ---------------------------------------------------------------------------
# Problem factory helpers
# ---------------------------------------------------------------------------

def _build_problem_from_csv(csv_path: str,
                             depths0: list,
                             conds0: list,
                             fixed_depths,
                             invert_kwargs: dict,
                             em_unit: str = "ppt"):
    """Standard emagpy inversion on a raw survey CSV."""
    from emagpy import Problem
    k = Problem()
    k.createSurvey(csv_path, targetProjection=None, unit=em_unit)
    # Mirror the same coil exclusion used during the actual ensemble
    # inversion (Step 1), so this diagnostic Problem reflects what was
    # really inverted rather than re-including a dropped/noisy channel.
    EMmethod.drop_coils_from_problem(k, invert_kwargs.get("drop_coils"))
    _fixed = ([fixed_depths] * len(depths0)
              if isinstance(fixed_depths, bool) else list(fixed_depths))
    k.setInit(depths0=list(depths0), conds0=list(conds0), fixedDepths=_fixed)
    clean = {key: v for key, v in invert_kwargs.items()
             if key not in ("targetProjection", "drop_coils", "em_unit")}
    k.invert(**clean)
    return k


def _inject_weighted_model(csv_path: str,
                            weighted_ec: np.ndarray,
                            depths0: list,
                            conds0: list,
                            fixed_depths,
                            invert_kwargs: dict,
                            em_unit: str = "ppt"):
    """
    Build a Problem, invert it (so surveys[0] / forwardModel are populated),
    then REPLACE k.models[0] and k.depths[0] with the weighted-average EC so
    that showResults / showDepths / showMisfit render the optimal ensemble model.

    emagpy stores:
      k.models[i]  — numpy array (n_pts, n_layers)   EC in mS/m
      k.depths[i]  — numpy array (n_pts, n_interfaces)  where n_interfaces = n_layers - 1
                     these are the INTERIOR layer interfaces (emagpy prepends 0
                     and appends maxDepth itself inside showResults)

    weighted_ec : (n_pts, n_layers)  — the K-averaged EC for ONE time step
    depths0     : list of n_interfaces bottom depths [m]  (same length as
                  the depths0 used in setInit, i.e. n_layers - 1 values)
    """
    from emagpy import Problem

    k = Problem()
    k.createSurvey(csv_path, targetProjection=None, unit=em_unit)
    # Mirror the same coil exclusion used during the actual ensemble
    # inversion (Step 1) — see _build_problem_from_csv above.
    EMmethod.drop_coils_from_problem(k, invert_kwargs.get("drop_coils"))
    _fixed = ([fixed_depths] * len(depths0)
              if isinstance(fixed_depths, bool) else list(fixed_depths))
    k.setInit(depths0=list(depths0), conds0=list(conds0), fixedDepths=_fixed)
    clean = {key: v for key, v in invert_kwargs.items()
             if key not in ("targetProjection", "drop_coils", "em_unit")}
    k.invert(**clean)

    n_pts = weighted_ec.shape[0]

    # models[i]: (n_pts, n_layers) — plain numpy float array
    k.models[0] = weighted_ec.astype(float)

    # depths[i]: (n_pts, n_interfaces) where n_interfaces = len(depths0)
    # Tile the fixed depths0 row across all measurement points
    depths_arr = np.tile(np.array(depths0, dtype=float), (n_pts, 1))
    k.depths[0] = depths_arr

    return k


# ---------------------------------------------------------------------------
# Parallel worker — one (ens_idx, t_idx) pair per task
# ---------------------------------------------------------------------------
#
# Must be a top-level, module-scope function: ProcessPoolExecutor with the
# `spawn` start method pickles the callable by import path ("module.func"),
# so a nested/local function or a lambda would fail to pickle. Each spawned
# worker re-imports this module fresh (matplotlib.use("Agg") at the top of
# the file runs again there too), which is the same "fresh interpreter, no
# inherited half-initialised state" fix already used for Step 1's
# _invert_single_worker in run_pipeline.py.
def _per_realisation_worker(task: tuple):
    (ens_idx, t_idx, csv_path, depths0, conds0_task, fixed_list,
     inv_kwargs, em_unit, r_dir, n_layers) = task

    t_dir = _ensure_dir(os.path.join(r_dir, f"t{t_idx:02d}"))
    prefix = f"ens{ens_idx:03d}_t{t_idx:02d}"

    try:
        k = _build_problem_from_csv(
            csv_path, depths0, list(conds0_task),
            fixed_list, inv_kwargs, em_unit,
        )
    except Exception as e:
        return (ens_idx, t_idx, False, f"Inversion failed: {type(e).__name__}: {e}")

    try:
        _individual_panels(k, n_layers, t_dir, prefix)
        _summary_grid(
            k, n_layers,
            title=f"Realisation {ens_idx} — t={t_idx}",
            out_path=os.path.join(t_dir, f"{prefix}_summary_grid.png"),
        )
    except Exception as e:
        return (ens_idx, t_idx, False, f"Plotting failed: {type(e).__name__}: {e}")

    return (ens_idx, t_idx, True, None)


# ---------------------------------------------------------------------------
# Public entry point 1 — per-realisation plots
# ---------------------------------------------------------------------------

def run_emagpy_plots_per_realisation(
    csv_paths: List[str],
    tasks: list,
    models_clean: np.ndarray,
    depths_clean: np.ndarray,
    depths0: list,
    output_dir: str,
    fixed_depths=True,
    mode: str = "1d",
    max_realisations: int = 4,
    em_unit: str = "ppt",
) -> None:
    """
    Render emagpy diagnostic panels for up to `max_realisations` individual
    inversions (sampled uniformly from the ensemble).

    Each realisation is a distinct (PARAM_GRID combo, conds0 variant) pair.
    The plots use the actual single-realisation inversion result, NOT the
    weighted-average model.

    Parameters
    ----------
    csv_paths         : list of survey CSV paths (one per time step)
    tasks             : task list built in Step 1 of run_pipeline.py
                        Each element is (task_idx, file_path, depths0, conds0,
                        fixed_depths_list, invert_kwargs).
    models_clean      : (ensemble_size, [n_time,] n_pts, n_layers) — kept EC
    depths_clean      : matching depths array
    depths0           : layer bottom depths [m]
    output_dir        : root output directory
    max_realisations  : cap on how many realisations to render (default 4)
    """
    diag_dir = _ensure_dir(os.path.join(output_dir, "emagpy_diag", "per_realisation"))
    n_layers = len(depths0)
    timelapse = len(csv_paths) > 1
    n_time = len(csv_paths)
    ensemble_size = models_clean.shape[0]

    # Sample uniformly across the ensemble
    step = max(1, ensemble_size // max_realisations)
    sample_ens = list(range(0, ensemble_size, step))[:max_realisations]

    # Group tasks by (combo_idx × t_idx) — tasks list has n_combos*n_time entries
    # Each combo runs across all time steps sequentially in the tasks list
    n_combos = ensemble_size        # one kept combo per ensemble member
    n_tasks_per_combo = n_time

    # --- Build the flat list of (ens_idx, t_idx) diagnostic jobs up front ---
    # Each job re-inverts one survey CSV from scratch (this is the expensive
    # part — the whole reason this step used to take as long as Step 1 while
    # running on a single core). Building the list here, in the main process,
    # keeps directory creation and task-metadata lookup single-threaded and
    # side-effect-free; only the actual invert+plot work is farmed out.
    diag_tasks = []
    skipped = []
    for ens_idx in sample_ens:
        r_dir = _ensure_dir(os.path.join(diag_dir, f"realisation_{ens_idx:03d}"))

        flat_task_idx = ens_idx * n_tasks_per_combo
        if flat_task_idx >= len(tasks):
            skipped.append(ens_idx)
            continue

        _, _, _depths0_task, conds0_task, fixed_list, inv_kwargs = tasks[flat_task_idx]

        for t_idx, csv_path in enumerate(csv_paths):
            diag_tasks.append((
                ens_idx, t_idx, csv_path, depths0, conds0_task, fixed_list,
                inv_kwargs, em_unit, r_dir, n_layers,
            ))

    print(f"\n{'='*72}")
    print(f" [EMAGPY DIAG] Per-realisation plots  ({len(sample_ens)} of {ensemble_size} realisations,"
          f" {len(diag_tasks)} inversion+plot jobs)")
    print(f"   mode : {mode.upper()}  |  time steps : {n_time}")
    print(f"{'='*72}\n")
    for ens_idx in skipped:
        print(f"  [WARN] task index {ens_idx * n_tasks_per_combo} out of range — skipping ens {ens_idx}")

    if not diag_tasks:
        print(f"\n[EMAGPY DIAG] Nothing to render — '{diag_dir}'")
        return

    # --- Dispatch across a process pool, same pattern as Step 1 ---
    # `spawn` (not the Linux-default `fork`) so each worker starts as a clean
    # interpreter and can't inherit half-initialised state (matplotlib
    # backend, BLAS/OpenMP thread pools, open file handles, ...) from this
    # parent process. Mirrors _invert_single_worker in run_pipeline.py.
    failed = []
    with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as executor:
        results = tqdm(
            executor.map(_per_realisation_worker, diag_tasks),
            total=len(diag_tasks), desc="Diagnostic plots", unit="job",
            ncols=90,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        for ens_idx, t_idx, ok, err in results:
            if ok:
                print(f"  ── ens={ens_idx}  t={t_idx}  done")
            else:
                print(f"  [WARN] ens={ens_idx}  t={t_idx}  {err}")
                failed.append((ens_idx, t_idx, err))

    if failed:
        print(f"\n[EMAGPY DIAG] {len(failed)}/{len(diag_tasks)} jobs failed (see [WARN] lines above)")
    print(f"\n[EMAGPY DIAG] Per-realisation plots written to '{diag_dir}'")


# ---------------------------------------------------------------------------
# Public entry point 2 — optimal (weighted-average) model plots
# ---------------------------------------------------------------------------

def run_emagpy_plots_optimal(
    csv_paths: List[str],
    selected_output: Any,          # SelectionOutput from 4_run_select_assess.py
    averaged_models: list,
    depths0: list,
    output_dir: str,
    fixed_depths=True,
    conds0: Optional[list] = None,
    invert_kwargs: Optional[dict] = None,
    mode: str = "1d",
    em_unit: str = "ppt",
) -> None:
    """
    Render emagpy diagnostic panels for the OPTIMAL ensemble model.

    The optimal model is ``averaged_models[selected_output.K - 1]``, i.e. the
    RMSE-weighted average of the top-K ranked realisations.  Its EC values are
    injected into a fresh emagpy Problem (replacing the raw single inversion)
    so that showResults / showDepths reflect the true pipeline output.

    showPseudo and showMisfit still use the underlying single inversion
    (with the first default conds0) because they compare against the raw
    survey ECa — that comparison is meaningful regardless of which model
    variant produced the EC section.

    Parameters
    ----------
    csv_paths        : list of survey CSV paths (one per time step)
    selected_output  : SelectionOutput namedtuple / object from Step 4/5
                       Must have attributes: K, rmse, cv_map
    averaged_models  : list of averaged EC arrays from Step 3
                       averaged_models[K-1] is the optimal weighted-mean model
    depths0          : layer bottom depths [m]
    output_dir       : root output directory
    fixed_depths     : passed to setInit
    conds0           : initial conductivities (default [10]*(n_layers+1))
    invert_kwargs    : forwarded to k.invert() for the background inversion
    mode             : '1d' or '2d'
    """
    if invert_kwargs is None:
        invert_kwargs = {}
    if conds0 is None:
        conds0 = [10.0] * (len(depths0) + 1)

    n_layers = len(depths0)
    diag_dir = _ensure_dir(os.path.join(output_dir, "emagpy_diag", "optimal_model"))
    n_time = len(csv_paths)
    timelapse = n_time > 1

    # The optimal averaged model
    K_opt = selected_output.K
    avg_model_opt = averaged_models[K_opt - 1]
    # shape: (n_time, n_pts, n_layers) if timelapse, else (n_pts, n_layers)

    print(f"\n{'='*72}")
    print(f" [EMAGPY DIAG] Optimal model plots  (K={K_opt}, RMSE={selected_output.rmse:.5f})")
    print(f"   averaged_models shape : {np.asarray(avg_model_opt).shape}")
    print(f"   mode : {mode.upper()}  |  time steps : {n_time}")
    print(f"{'='*72}\n")

    for t_idx, csv_path in enumerate(csv_paths):
        t_dir = _ensure_dir(os.path.join(diag_dir, f"t{t_idx:02d}"))
        prefix = f"optimal_K{K_opt}_t{t_idx:02d}"
        print(f"  ── t={t_idx}  {os.path.basename(csv_path)}")

        # Extract the (n_pts, n_layers) EC slice for this time step
        if timelapse:
            ec_slice = np.asarray(avg_model_opt)[t_idx]   # (n_pts, n_layers)
        else:
            ec_slice = np.asarray(avg_model_opt)           # (n_pts, n_layers)

        try:
            k = _inject_weighted_model(
                csv_path, ec_slice,
                depths0, conds0, fixed_depths, invert_kwargs, em_unit,
            )
        except Exception as e:
            print(f"    [WARN] Could not build optimal Problem for t={t_idx}: {e}")
            continue

        _individual_panels(k, n_layers, t_dir, prefix)
        _summary_grid(
            k, n_layers,
            title=(f"Optimal ensemble model  K={K_opt}  RMSE={selected_output.rmse:.4f}"
                   f"  —  t={t_idx}"),
            out_path=os.path.join(t_dir, f"{prefix}_summary_grid.png"),
        )

    print(f"\n[EMAGPY DIAG] Optimal model plots written to '{diag_dir}'")
