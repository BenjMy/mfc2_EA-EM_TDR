#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometry_utils.py
==================
Single source of truth for a handful of small but easy-to-drift-apart
calculations that were previously implemented more than once across the
pipeline (see pipeline_improvement_report.md, section 6 "Code duplication &
consistency risk" and section 10, action item #3).

Consolidated here:

  layer_centers(depths0)
      Layer-centre depths for all ``num_layers = len(depths0) + 1`` soil
      horizons, INCLUDING the terminal half-space layer below the last
      fixed interface. Previously duplicated (with a real bug in one copy
      -- the "layer_centers KeyError" / off-by-one-layer incident) across
      ``0_generate_synthetic_data.py`` (inline), ``run_pipeline.py::_layer_centers``,
      and ``run_pipeline_short.py::_layer_centers``.

  build_inversion_bnds(depths0, fixed_depths_list, cond_ranges)
      Builds the explicit `bnds` list passed to EMagPy's ``Problem.invert()``,
      reproducing EMagPy's own depth-bound heuristic for any free interface
      but substituting the scenario's ``cond_ranges`` for the conductivity
      bounds instead of EMagPy's hardcoded (2, 100) mS/m default. Previously
      duplicated across ``run_pipeline.py::_build_inversion_bnds`` and
      ``1_run_model_generation.py::_build_inversion_bnds``.

  route_ec_to_horizons(ec_array, sample_positions, sample_times,
                        probe_layer_indices, num_layers)
      Given an EC array shaped ``([n_time,] num_measured_points, num_layers)``
      (i.e. with any ensemble axis already collapsed/selected by the
      caller), extracts one EC value per (sample_position, sample_time) and
      routes it to its soil horizon using EXACTLY the same rule
      ``em_methodology.calibrate_petrophysical_relationship`` uses
      internally: explicit ``probe_layer_indices`` when given, else a
      strided split, else wrap-around modulo assignment. Returns the values
      concatenated per-horizon (matching the order of
      ``PetroFit.observed_theta`` / ``PetroFit.predicted_theta``), plus the
      per-horizon start/end slice bounds for callers that need to know
      which returned elements belong to which horizon.

      Previously duplicated (independently, with parallel but subtly
      different code) across ``2_run_petro.py::_extract_ec_for_plots`` and
      ``4_run_select_assess.py::_extract_ec_at_probes``.

All three are deliberately dependency-light (only ``numpy``) so any module
in this pipeline can import them without pulling in ``emagpy`` or
``matplotlib``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 1. Layer-centre depths
# ---------------------------------------------------------------------------

def layer_centers(depths0: Sequence[float]) -> np.ndarray:
    """
    Layer-centre depths for all ``num_layers = len(depths0) + 1`` soil
    horizons, INCLUDING the terminal half-space layer below the last fixed
    interface.

    Mirrors the convention used throughout the pipeline:
    ``depth_boundaries = [0.0] + list(depths0) + [depths0[-1] * 1.5]``,
    then the centre of each consecutive pair of boundaries.

    Parameters
    ----------
    depths0 : sequence of float
        Fixed/starting layer interface depths [m], length = n_layers - 1.

    Returns
    -------
    np.ndarray, shape (len(depths0) + 1,)
        Centre depth [m] of every soil horizon, in the same order as the
        ``num_layers`` axis of any EC/theta array in this pipeline.
    """
    bottoms = np.asarray(depths0, dtype=float)
    depth_boundaries = np.concatenate(([0.0], bottoms, [bottoms[-1] * 1.5]))
    return 0.5 * (depth_boundaries[:-1] + depth_boundaries[1:])


# ---------------------------------------------------------------------------
# 2. Inversion solver bounds
# ---------------------------------------------------------------------------

def build_inversion_bnds(
    depths0: Sequence[float],
    fixed_depths_list: Sequence[bool],
    cond_ranges: Sequence[Sequence[float]],
) -> list[tuple[float, float]]:
    """
    Build the explicit `bnds` list to pass into `Problem.invert()`.

    Without an explicit `bnds`, EMagPy's own invert() falls back to a
    hardcoded internal default whenever any depth is variable:

        bot = np.r_[np.r_[0.2, mdepths], np.ones(nc)*2]
        top = np.r_[np.r_[mdepths, depths0[-1] + 0.2], np.ones(nc)*100]

    i.e. every conductivity layer gets silently clamped to [2, 100] mS/m
    regardless of `cond_ranges` in the scenario config. This function
    reproduces EMagPy's own depth-bound heuristic (so free-depth behaviour
    is unchanged) but substitutes the caller's `cond_ranges` for the
    conductivity bounds instead of the hardcoded (2, 100) default.

    Ordering matches what `Problem.invert(bnds=...)` expects: variable
    depths first, then variable conductivities (``fixedConds`` is never set
    anywhere in this pipeline's `k.setInit()` calls, so EMagPy defaults it
    to "all conductivities variable" -- every layer in `cond_ranges` is
    therefore included).

    Parameters
    ----------
    depths0 : sequence of float
        Starting depth interfaces (length = n_layers - 1).
    fixed_depths_list : sequence of bool
        Per-interface fixed/free flag, same length as depths0.
    cond_ranges : sequence of [min, max] (or [value]) per layer
        Length = n_layers = len(depths0) + 1.

    Returns
    -------
    list of (min, max) tuples, ordered [free depths..., all conductivities...]
    """
    depths0_arr = np.asarray(depths0, dtype=float)
    vd = ~np.asarray(fixed_depths_list, dtype=bool)

    if len(depths0_arr) > 1:
        mdepths = depths0_arr[:-1] + np.diff(depths0_arr) / 2
    else:
        mdepths = np.array([], dtype=float)
    depth_bot = np.r_[0.2, mdepths]
    depth_top = np.r_[mdepths, depths0_arr[-1] + 0.2]
    depth_bnds = list(zip(depth_bot[vd].tolist(), depth_top[vd].tolist()))

    cond_bnds = [tuple(r) if len(r) == 2 else (r[0], r[0]) for r in cond_ranges]

    return depth_bnds + cond_bnds


# ---------------------------------------------------------------------------
# 3. EC-at-probe extraction / horizon routing
# ---------------------------------------------------------------------------

def route_ec_to_horizons(
    ec_array: np.ndarray,
    sample_positions: Sequence[int],
    sample_times: Sequence[int],
    probe_layer_indices: Sequence[int] | None,
    num_layers: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Extract sigma_b (bulk EC) at each TDR probe location from an EC array
    with any ensemble axis already resolved by the caller, and route each
    observation to its soil horizon using EXACTLY the routing rule
    ``em_methodology.calibrate_petrophysical_relationship`` uses internally
    -- so results here always line up 1:1 with
    ``PetroFit.observed_theta`` / ``PetroFit.predicted_theta``.

    Routing rule (checked in this order):

    1. ``probe_layer_indices`` given -> each probe routed to its own known
       horizon; a horizon with zero probes mapped to it contributes an
       empty slice (never backfilled with unrelated observations).
    2. ``probe_layer_indices`` is None and ``n_samples >= num_layers`` ->
       strided slice: layer i gets a contiguous, non-overlapping subset.
    3. ``probe_layer_indices`` is None and ``n_samples < num_layers`` ->
       wrap-around: probe i assigned to layer ``i % num_layers``.

    Parameters
    ----------
    ec_array : np.ndarray
        Shape ``([n_time,] num_measured_points, num_layers)`` -- NO
        ensemble axis. Callers with a raw ``(ensemble, ..., n_layers)``
        array should select one member (e.g. ``models[0]``) before calling;
        callers with an already-averaged model pass it directly.
    sample_positions, sample_times : sequence of int
        Same arrays passed to
        ``em_methodology.calibrate_petrophysical_relationship``.
    probe_layer_indices : sequence of int or None
        Same array passed to
        ``em_methodology.calibrate_petrophysical_relationship``.
    num_layers : int
        ``ec_array.shape[-1]``.

    Returns
    -------
    ec_extracted : np.ndarray, shape (n_routed,)
        Flat EC values, concatenated per horizon in ascending horizon
        order -- matches ``PetroFit.observed_theta`` /
        ``PetroFit.predicted_theta`` ordering exactly. ``n_routed`` equals
        ``len(sample_positions)`` unless ``probe_layer_indices`` leaves one
        or more horizons uncovered (those probes are simply omitted, same
        as the calibration engine).
    horizon_slices : list of (start, end)
        For each horizon (index = list position), the ``[start, end)``
        slice into ``ec_extracted`` holding that horizon's values. Useful
        for callers (e.g. plotting) that need to know which points belong
        to which horizon without re-deriving the routing themselves.
    """
    timelapse = ec_array.ndim == 3
    sample_positions = list(sample_positions)
    sample_times = list(sample_times)
    n_samples = len(sample_positions)

    ec_parts: list[np.ndarray] = []
    horizon_slices: list[tuple[int, int]] = []
    cursor = 0

    for layer in range(num_layers):
        if timelapse:
            sigma_b_full = np.array(
                [ec_array[sample_times[i], pos, layer]
                 for i, pos in enumerate(sample_positions)],
                dtype=float,
            )
        else:
            sigma_b_full = np.array(
                [ec_array[pos, layer] for pos in sample_positions],
                dtype=float,
            )

        if probe_layer_indices is not None:
            indices = [i for i, li in enumerate(probe_layer_indices) if li == layer]
            layer_vals = sigma_b_full[indices] if indices else np.array([], dtype=float)
        elif n_samples >= num_layers:
            step = n_samples // num_layers
            start = layer * step
            end = start + step if layer < num_layers - 1 else n_samples
            layer_vals = sigma_b_full[start:end]
        else:
            indices = [i for i in range(n_samples) if i % num_layers == layer]
            if not indices:
                indices = list(range(n_samples))
            layer_vals = sigma_b_full[indices]

        ec_parts.append(layer_vals)
        horizon_slices.append((cursor, cursor + len(layer_vals)))
        cursor += len(layer_vals)

    ec_extracted = np.concatenate(ec_parts) if ec_parts else np.array([], dtype=float)
    return ec_extracted, horizon_slices
