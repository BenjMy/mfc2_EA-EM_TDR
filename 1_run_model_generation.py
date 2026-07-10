#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 1 Driver — Ensemble inversion parallelised with ProcessPoolExecutor
and tqdm progress tracking. Handles multi-parameter grid combinations.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm

import geometry_utils

# --------------------------------------------------------------------------
# Bounds builder — single shared implementation in geometry_utils.py
# (pipeline_improvement_report.md, section 6 / action item 3). This file
# and run_pipeline.py previously carried independent copies of the same
# logic; both now delegate here so a future fix can't land in one copy and
# not the other.
# --------------------------------------------------------------------------
_build_inversion_bnds = geometry_utils.build_inversion_bnds


# --------------------------------------------------------------------------
# Parallel Worker for Step 1 Inversions
# --------------------------------------------------------------------------
def _invert_single_worker(args) -> np.ndarray:
    """
    Worker function executed in an isolated process space.
    Prevents EMagPy cross-process state/file collisions.
    """
    fname, depths0, conds0, fixed_depths_list, invert_kwargs = args

    # Local import keeps process state completely isolated
    from emagpy import Problem

    try:
        k = Problem()
        k.createSurvey(fname, targetProjection=invert_kwargs.get("targetProjection"),
                        unit=invert_kwargs.get("em_unit", "ppt"))
        k.setInit(depths0=list(depths0), conds0=list(conds0), fixedDepths=fixed_depths_list)

        # Remove targetProjection and em_unit from kwargs before inverting
        clean_kwargs = {key: v for key, v in invert_kwargs.items()
                        if key not in ("targetProjection", "em_unit")}
        k.invert(**clean_kwargs)

        # Return standalone 2D numpy array: (num_measured_points, num_layers)
        return np.asarray(k.models[0])
    except Exception as e:
        # Prevent a single failed inversion from crashing the whole pool
        print(f"\n[ERROR] Inversion failed with parameters {invert_kwargs}: {e}")
        return None


# --------------------------------------------------------------------------
# Core Parallel Engine
# --------------------------------------------------------------------------
def run_parallel_ensemble(
    input_file: str,
    param_grid: dict,
    cond_ranges: list,
    depths0: list,
    fixed_depths: bool | list,
    max_workers: int | None = None,
    em_unit: str = "ppt",
) -> np.ndarray:
    """
    Constructs the inversion parameter grid and parallelises execution
    across multiple CPU cores via ProcessPoolExecutor with a dynamic progress bar.
    """
    # 1. Unpack parameter grid combinations manually without itertools
    alphas = param_grid.get("alpha", [0.03])
    methods = param_grid.get("method", ["L-BFGS-B"])
    forward_models = param_grid.get("forwardModel", ["FSlin"])

    combinations = []
    for alpha in alphas:
        for method in methods:
            for f_model in forward_models:
                combinations.append((alpha, method, f_model))

    # 2. Build initial conductivity variants from ranges
    conds0_variants = []
    for r in cond_ranges:
        if len(r) == 2:
            conds0_variants.append([r[0], r[1]])
        else:
            conds0_variants.append([r[0]])

    # Parse fixed vs floating depths structure
    # NOTE: emagpy's Problem.setInit requires len(fixedDepths) == len(depths0)
    # (NOT len(depths0) + 1 -- that length is for conds0/fixedConds instead).
    if isinstance(fixed_depths, bool):
        fixed_depths_list = [fixed_depths] * len(depths0)
    else:
        fixed_depths_list = list(fixed_depths)
        if len(fixed_depths_list) != len(depths0):
            raise ValueError(
                f"fixed_depths must have length {len(depths0)} (== len(depths0)), "
                f"got {len(fixed_depths_list)}."
            )

    # Per-layer (min, max) box constraints for the L-BFGS-B solver, derived
    # from the same cond_ranges used to build conds0_variants above. Without
    # this, EMagPy's k.invert() falls back to its own internal default bnds
    # (unrelated to cond_ranges), silently capping every inverted EC value
    # regardless of how wide cond_ranges is configured.
    #
    # FIX: build via _build_inversion_bnds() instead of a bare
    # `[tuple(r) for r in cond_ranges]`. The old version only had
    # len(cond_ranges) entries, which is correct ONLY when every depth is
    # fixed. If fixed_depths ever includes a free interface, EMagPy needs
    # len(bnds) == n_free_depths + n_free_conds, so a cond-only list would
    # raise (or, worse, get dropped to dodge the error -- silently
    # reintroducing the hardcoded 2-100 mS/m default bound bug).
    bnds_list = _build_inversion_bnds(depths0, fixed_depths_list, cond_ranges)

    # 3. Assemble task arguments for the process pool
    tasks = []
    for alpha, method, f_model in combinations:
        for cond_idx in range(len(conds0_variants[0])):
            conds0 = [v[cond_idx if len(v) > cond_idx else 0] for v in conds0_variants]

            invert_kwargs = {
                "alpha": alpha,
                "method": method,
                "forwardModel": f_model,
                "targetProjection": None,
                "em_unit": em_unit,
                "bnds": bnds_list,
            }
            tasks.append((input_file, depths0, conds0, fixed_depths_list, invert_kwargs))

    total_tasks = len(tasks)
    print(f"[INFO] Launching parallel sweep over {total_tasks} inversion tasks...")

    # 4. Execute workers in parallel and track progress using tqdm
    inverted_models_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = tqdm(
            executor.map(_invert_single_worker, tasks),
            total=total_tasks,
            desc="Running Inversions",
            unit="task",
            ncols=90,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        for model_arr in results:
            if model_arr is not None:
                inverted_models_list.append(model_arr)

    if not inverted_models_list:
        raise ValueError("[CRITICAL] All parallel inversion tasks failed.")

    # 5. Restructure the list into a single 3D ensemble matrix
    models_ensemble = np.stack(inverted_models_list, axis=0)
    print(f"[SUCCESS] Parallel collection complete. Ensemble array shape: {models_ensemble.shape}")

    return models_ensemble


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Fallback to current folder if relative path structure isn't present
    EM_FNAME = "data/synthetic/data_Model 1.csv"
    if not os.path.exists(EM_FNAME):
        EM_FNAME = "data_Model 1.csv"

    # Expanded hyperparameter grid setup
    PARAM_GRID = {
        "alpha": [0.005, 0.05, 0.5, 2.5],
        "method": ["L-BFGS-B", "CG"],
        "forwardModel": ["FSlin", "CS"],
    }

    COND_RANGES = [
        [5.0, 250.0],   # Layer 1 bounds
        [5.0, 250.0],   # Layer 2 bounds
        [10.0, 300.0],  # Layer 3 bounds
        [10.0, 300.0],  # Layer 4 bounds
    ]

    DEPTHS0 = [0.3, 0.6, 1.0]
    FIXED_DEPTHS = True
    OUTPUT_DIR = "mfc2_outputs"

    # Set max_workers=None to leverage all available CPU threads automatically
    models = run_parallel_ensemble(
        input_file=EM_FNAME,
        param_grid=PARAM_GRID,
        cond_ranges=COND_RANGES,
        depths0=DEPTHS0,
        fixed_depths=FIXED_DEPTHS,
        max_workers=None
    )

    # Save cached models matrix
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "step1_inverted_models.npy")
    np.save(output_path, models)
    print(f"[INFO] Successfully saved models to: {output_path}")
