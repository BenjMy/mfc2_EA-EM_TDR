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
      strided split, else wrap-around modulo assignment -- THEN drops any
      observation whose extracted EC is non-finite, again exactly
      mirroring that function's own NaN-observation filter (real-data
      time-lapse reference-grid gaps). Returns the values concatenated
      per-horizon (matching the order AND length of
      ``PetroFit.observed_theta`` / ``PetroFit.predicted_theta`` exactly),
      the per-horizon start/end slice bounds, and ``kept_indices`` --
      which other per-observation arrays should be gathered with, via
      ``arr[kept_indices]``, to stay aligned.

      Previously duplicated (independently, with parallel but subtly
      different code) across ``2_run_petro.py::_extract_ec_for_plots`` and
      ``4_run_select_assess.py::_extract_ec_at_probes`` -- the latter
      wasn't available to check against when this consolidation happened;
      worth confirming it replicates the NaN-drop too, or routes through
      this function instead.

  route_array_to_horizons(values, probe_layer_indices, num_layers)
      Routing only (no NaN-drop) for arrays that don't need to match
      ``PetroFit.observed_theta`` exactly. See its docstring for when NOT
      to use it -- prefer ``route_ec_to_horizons``'s ``kept_indices``
      whenever alignment with the fitted values matters.

  match_epochs_to_reference_grid(epoch_xy, tolerance_m)
      Builds a UNION spatial reference grid across multiple real-data
      survey epochs (which can have different point counts and different
      walk orders/paths) and returns, per epoch, the row index into that
      epoch's OWN point array matching each reference-grid point (or -1
      when that epoch has no point within tolerance). Replaces the old
      row-count trim (``arr[:min_pts]``) that silently paired unrelated
      physical locations under the same array index whenever epochs
      weren't walked in the same order. See ``run_pipeline.py`` Step 1
      (real + time-lapse branch) for the caller.

All are deliberately dependency-light (only ``numpy``) so any module
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

def _horizon_index_groups(
    n_samples: int,
    probe_layer_indices: Sequence[int] | None,
    num_layers: int,
) -> list[list[int]]:
    """
    Shared index-selection rule underlying both ``route_ec_to_horizons``
    and ``route_array_to_horizons``. Returns, per horizon (list position =
    horizon index), the list of observation indices (into a
    length-``n_samples`` per-observation array) that horizon routes to.

    Factored out so the two public routing functions can never drift out
    of sync with each other -- both call this, so there is exactly one
    place the strided/modulo/explicit-index rule is written down. See
    ``route_ec_to_horizons`` for the routing rule itself.
    """
    groups: list[list[int]] = []
    for layer in range(num_layers):
        if probe_layer_indices is not None:
            indices = [i for i, li in enumerate(probe_layer_indices) if li == layer]
        elif n_samples >= num_layers:
            step = n_samples // num_layers
            start = layer * step
            end = start + step if layer < num_layers - 1 else n_samples
            indices = list(range(start, end))
        else:
            indices = [i for i in range(n_samples) if i % num_layers == layer]
            if not indices:
                indices = list(range(n_samples))
        groups.append(indices)
    return groups


def route_array_to_horizons(
    values: Sequence,
    probe_layer_indices: Sequence[int] | None,
    num_layers: int,
) -> np.ndarray:
    """
    Reorder/subset an arbitrary per-observation array using the horizon
    ROUTING rule shared with ``route_ec_to_horizons`` -- but see the
    WARNING below before using this for anything that must line up with
    ``PetroFit.observed_theta``/``predicted_theta``.

    Typical use: routing ``sample_positions`` itself (rather than an EC
    array) so per-observation metadata -- e.g. which sensor an observation
    came from -- can be matched up with the fitted values for plotting,
    without duplicating the routing logic at the call site.

    WARNING -- this does NOT reproduce the NaN-observation drop
    -----------------------------------------------------------
    ``em_methodology.calibrate_petrophysical_relationship`` performs one
    extra step per horizon, AFTER routing: any observation whose bulk EC
    (``sigma_b``) is non-finite -- which happens for real-data time-lapse
    runs when a TDR observation lands on a (reference-grid point, epoch)
    pair that particular epoch didn't cover, see
    ``geometry_utils.match_epochs_to_reference_grid`` -- is dropped from
    that horizon's ``theta_layer``/``sigma_b_layer`` before fitting. This
    means ``PetroFit.observed_theta``/``predicted_theta`` can be SHORTER
    than what pure routing (this function, or ``route_ec_to_horizons``'s
    ``ec_extracted`` on its own) produces, whenever any such NaN exists.

    This function has no way to know about that drop -- it never sees the
    EC array, only ``probe_layer_indices``. If you need an array that
    lines up with ``PetroFit.observed_theta`` element-for-element, do NOT
    use this function; instead call ``route_ec_to_horizons`` (which DOES
    replicate the drop) and index your array with the ``kept_indices`` it
    returns, e.g.::

        ec_extracted, slices, kept_indices = route_ec_to_horizons(...)
        routed_positions = np.asarray(sample_positions)[kept_indices]

    Use this function only when you know the array in question is not
    affected -- e.g. synthetic-data runs (no reference grid, so no NaNs
    ever enter the EC array), or single-survey real-data runs.

    Parameters
    ----------
    values : sequence, length n_samples
        Any per-observation array, in the SAME order as the
        ``sample_positions``/``sample_times`` originally passed to
        ``em_methodology.calibrate_petrophysical_relationship``.
    probe_layer_indices, num_layers
        Same as ``route_ec_to_horizons``.

    Returns
    -------
    np.ndarray
        ``values`` reordered/subset per horizon, concatenated in
        ascending horizon order, WITHOUT the NaN-observation drop (see
        warning above).
    """
    values = np.asarray(values)
    n_samples = len(values)
    groups = _horizon_index_groups(n_samples, probe_layer_indices, num_layers)
    parts = [values[idx] for idx in groups if idx]
    return np.concatenate(parts) if parts else values[:0]


def route_ec_to_horizons(
    ec_array: np.ndarray,
    sample_positions: Sequence[int],
    sample_times: Sequence[int],
    probe_layer_indices: Sequence[int] | None,
    num_layers: int,
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
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

    AFTER routing, any observation whose extracted ``sigma_b`` is
    non-finite is dropped from its horizon -- replicating
    ``em_methodology.calibrate_petrophysical_relationship``'s own
    NaN-observation filter exactly (its docstring there: real-data
    time-lapse observations landing on a (reference-grid point, epoch)
    pair that epoch didn't cover come through as NaN and are dropped
    rather than fed to the optimizer or silently paired with an unrelated
    EC value). This is what makes ``ec_extracted`` here guaranteed to
    match ``PetroFit.observed_theta``/``predicted_theta`` length-for-length
    and order-for-order, including in that case.

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
        order, with non-finite entries dropped -- matches
        ``PetroFit.observed_theta`` / ``PetroFit.predicted_theta``
        ordering AND length exactly.
    horizon_slices : list of (start, end)
        For each horizon (index = list position), the ``[start, end)``
        slice into ``ec_extracted`` holding that horizon's (post-NaN-drop)
        values. Useful for callers (e.g. plotting) that need to know which
        returned elements belong to which horizon without re-deriving the
        routing themselves.
    kept_indices : np.ndarray, shape (n_routed,)
        For each element of ``ec_extracted`` (same order), the index into
        the ORIGINAL length-``len(sample_positions)`` per-observation
        arrays it came from. Index any other per-observation array (e.g.
        ``sample_positions`` itself, ``probe_layer_indices``,
        per-probe coordinates) with this -- ``arr = np.asarray(other)[kept_indices]``
        -- to get an array that lines up with ``ec_extracted`` /
        ``PetroFit.observed_theta`` exactly, NaN-drop included. Prefer
        this over ``route_array_to_horizons`` whenever the result needs to
        match ``PetroFit.observed_theta`` (see that function's warning).
    """
    timelapse = ec_array.ndim == 3
    sample_positions = list(sample_positions)
    sample_times = list(sample_times)
    n_samples = len(sample_positions)

    ec_parts: list[np.ndarray] = []
    kept_parts: list[np.ndarray] = []
    horizon_slices: list[tuple[int, int]] = []
    cursor = 0
    horizon_groups = _horizon_index_groups(n_samples, probe_layer_indices, num_layers)

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

        indices = horizon_groups[layer]
        layer_vals = sigma_b_full[indices] if indices else np.array([], dtype=float)
        layer_indices = np.asarray(indices, dtype=int)

        # NaN-observation drop -- mirrors em_methodology.
        # calibrate_petrophysical_relationship's own filter exactly (see
        # docstring above), so ec_extracted/kept_indices stay aligned with
        # PetroFit.observed_theta even when the reference-grid remap left
        # some (position, epoch) pairs uncovered for this horizon's probes.
        if layer_vals.size > 0:
            finite = np.isfinite(layer_vals)
            if not np.all(finite):
                layer_vals = layer_vals[finite]
                layer_indices = layer_indices[finite]

        ec_parts.append(layer_vals)
        kept_parts.append(layer_indices)
        horizon_slices.append((cursor, cursor + len(layer_vals)))
        cursor += len(layer_vals)

    ec_extracted = np.concatenate(ec_parts) if ec_parts else np.array([], dtype=float)
    kept_indices = (np.concatenate(kept_parts) if kept_parts
                     else np.array([], dtype=int))
    return ec_extracted, horizon_slices, kept_indices


# ---------------------------------------------------------------------------
# 4. Spatial reference-grid matching across time-lapse survey epochs
# ---------------------------------------------------------------------------

def match_epochs_to_reference_grid(
    epoch_xy: Sequence[tuple[np.ndarray, np.ndarray]],
    tolerance_m: float,
    verbose: bool = False,
    epoch_labels: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Build a UNION spatial reference grid across multiple survey epochs and
    return, for each epoch, the row index into that epoch's OWN point array
    matching each reference-grid point (or -1 if that epoch has no point
    within ``tolerance_m`` of the reference point).

    This replaces a plain row-count trim (``arr[:min_pts]``): real GEM-2
    survey epochs are inverted independently and can differ both in point
    COUNT and in walk ORDER (the surveyor need not retrace an identical
    path each campaign), so pairing "row i of epoch A" with "row i of
    epoch B" is only correct by coincidence. Matching is instead done
    purely by coordinate: no epoch's own point set is ever reordered,
    subsetted, or discarded -- every point from every epoch contributes to
    (or is folded into) the reference grid, and any epoch/reference-point
    combination without a spatial match is left for the caller to fill
    with NaN rather than being silently paired with an unrelated location.

    Algorithm
    ---------
    1. Reference grid = union of every epoch's own (x, y) points,
       deduplicated greedily: a point is added to the reference grid only
       if it is farther than ``tolerance_m`` from every point already in
       the grid; otherwise it is considered the same physical location as
       the nearest existing reference point.
    2. For each epoch, every reference-grid point is nearest-neighbour
       matched back against that epoch's own points; a match is kept only
       if it falls within ``tolerance_m``.

    Because the reference grid is seeded by iterating epochs in the given
    order, epoch 0's own points are always included in the reference grid
    unchanged (nothing can be closer than tolerance to a point that IS
    already the reference point for itself), so epoch 0 always matches
    every one of its own points with distance 0.

    Parameters
    ----------
    epoch_xy : sequence of (x, y) np.ndarray pairs
        One ``(x_epoch, y_epoch)`` pair per survey epoch, in the SAME row
        order as that epoch's own inverted EC model array. ``y`` may be
        all-zeros for a pure 1-D transect.
    tolerance_m : float
        Points within this distance [m] are considered the same physical
        location. Should be set to roughly the GPS/GNSS precision of the
        survey instrument (a few metres is typical), not the (usually
        tighter) TDR<->EM ``window_m`` used elsewhere in this pipeline --
        the two tolerances answer different questions ("is this the same
        survey point across dates" vs "is this EM point close enough to
        this probe to represent it").
    verbose : bool, optional
        When True, prints a structured console report covering: the
        tolerance used, the final union grid size, each epoch's
        contribution during grid CONSTRUCTION (how many of its own points
        seeded a brand-new reference cell vs. were folded into an
        already-existing one within tolerance), and each epoch's final
        COVERAGE of the union grid (how many reference cells it matches,
        vs. how many will come out NaN once a caller remaps its EC/depths
        array onto the shared grid -- e.g. ``run_pipeline.py``'s
        ``_remap_combo_to_reference``). Default ``False`` (silent, as
        before -- this flag never changes the returned values).
    epoch_labels : sequence of str, optional
        Human-readable label per epoch (e.g. survey filenames) used only
        in the verbose report. Defaults to ``"epoch {i}"`` when omitted or
        shorter than ``epoch_xy``.

    Returns
    -------
    ref_x, ref_y : np.ndarray
        Reference-grid coordinates (union across all epochs).
    matched_indices : list of np.ndarray, length = len(epoch_xy)
        ``matched_indices[e][r]`` = row index into epoch ``e``'s own
        ``(x, y)`` (and therefore its own inverted EC model row) that
        matches reference point ``r``, or ``-1`` if epoch ``e`` has no
        point within ``tolerance_m`` of reference point ``r``.
    """
    tol2 = float(tolerance_m) ** 2
    n_epochs = len(epoch_xy)

    ref_x_list: list[float] = []
    ref_y_list: list[float] = []

    # Per-epoch bookkeeping for the verbose grid-CONSTRUCTION report: how
    # many of this epoch's own points became a brand-new reference cell
    # ("new") vs. were close enough to an already-existing one to be
    # folded into it instead ("merged"). Tracked unconditionally (cheap --
    # just two counters per epoch) so verbose=True never needs a second pass.
    n_own_points: list[int] = []
    n_new_ref_points: list[int] = []
    n_merged_into_existing: list[int] = []

    for x_e, y_e in epoch_xy:
        x_e = np.asarray(x_e, dtype=float)
        y_e = np.asarray(y_e, dtype=float)
        n_new = 0
        n_merged = 0
        for xi, yi in zip(x_e, y_e):
            if ref_x_list:
                d2 = (np.asarray(ref_x_list) - xi) ** 2 + (np.asarray(ref_y_list) - yi) ** 2
                if d2.min() <= tol2:
                    n_merged += 1
                    continue
            ref_x_list.append(float(xi))
            ref_y_list.append(float(yi))
            n_new += 1
        n_own_points.append(len(x_e))
        n_new_ref_points.append(n_new)
        n_merged_into_existing.append(n_merged)

    ref_x = np.asarray(ref_x_list, dtype=float)
    ref_y = np.asarray(ref_y_list, dtype=float)
    n_ref = len(ref_x)

    matched_indices: list[np.ndarray] = []
    for x_e, y_e in epoch_xy:
        x_e = np.asarray(x_e, dtype=float)
        y_e = np.asarray(y_e, dtype=float)
        idx_for_epoch = np.full(n_ref, -1, dtype=int)
        if n_ref > 0 and len(x_e) > 0:
            d2 = (ref_x[:, None] - x_e[None, :]) ** 2 + (ref_y[:, None] - y_e[None, :]) ** 2
            nearest = d2.argmin(axis=1)
            nearest_dist2 = d2[np.arange(n_ref), nearest]
            within = nearest_dist2 <= tol2
            idx_for_epoch[within] = nearest[within]
        matched_indices.append(idx_for_epoch)

    if verbose:
        _print_reference_grid_report(
            tolerance_m=tolerance_m,
            n_epochs=n_epochs,
            n_ref=n_ref,
            n_own_points=n_own_points,
            n_new_ref_points=n_new_ref_points,
            n_merged_into_existing=n_merged_into_existing,
            matched_indices=matched_indices,
            epoch_labels=epoch_labels,
        )

    return ref_x, ref_y, matched_indices


def _print_reference_grid_report(
    tolerance_m: float,
    n_epochs: int,
    n_ref: int,
    n_own_points: list[int],
    n_new_ref_points: list[int],
    n_merged_into_existing: list[int],
    matched_indices: list[np.ndarray],
    epoch_labels: Sequence[str] | None,
) -> None:
    """
    Console report for ``match_epochs_to_reference_grid(verbose=True)``.

    Two tables are printed:

    1. GRID CONSTRUCTION -- per epoch, how many of its own raw points
       seeded a brand-new reference-grid cell vs. were folded into an
       already-existing one (within ``tolerance_m``) contributed by an
       earlier epoch. Every one of an epoch's own points ends up in
       exactly one of these two buckets -- none are ever dropped here.

    2. GRID COVERAGE -- per epoch, how many of the FINAL union grid's
       ``n_ref`` cells that epoch actually matches (``matched_indices[e] >= 0``)
       vs. how many will come out NaN once a caller remaps that epoch's
       EC/depths array onto the shared grid (e.g.
       ``run_pipeline.py``'s ``_remap_combo_to_reference``). This is the
       number that actually determines how much NaN downstream steps
       (Step 3 averaging, Step 4/5 RMSE, CV maps) will have to tolerate.
    """
    labels = list(epoch_labels) if epoch_labels else []
    while len(labels) < n_epochs:
        labels.append(f"epoch {len(labels)}")

    label_w = max(6, min(28, max((len(l) for l in labels), default=6)))

    print(f"\n{'='*78}")
    print(" [GEOMETRY] match_epochs_to_reference_grid -- spatial reference-grid report")
    print(f"{'='*78}")
    print(f"  tolerance_m           : {tolerance_m} m")
    print(f"  epochs                : {n_epochs}")
    print(f"  union reference grid  : {n_ref} point(s)")

    print(f"\n  -- Grid CONSTRUCTION (each epoch's own points, in order) --")
    print(f"  {'epoch':<7}{'label':<{label_w}}  {'n_own_pts':>10}  {'new_ref_pts':>12}  {'merged_existing':>16}")
    total_own = 0
    total_new = 0
    total_merged = 0
    for i in range(n_epochs):
        n_own = n_own_points[i]
        n_new = n_new_ref_points[i]
        n_merged = n_merged_into_existing[i]
        total_own += n_own
        total_new += n_new
        total_merged += n_merged
        print(f"  {i:<7}{labels[i][:label_w]:<{label_w}}  {n_own:>10}  {n_new:>12}  {n_merged:>16}")
    print(f"  {'':<7}{'TOTAL':<{label_w}}  {total_own:>10}  {total_new:>12}  {total_merged:>16}"
          f"   (new + merged == own_pts for every epoch; sanity check: "
          f"{'OK' if total_new == n_ref else f'MISMATCH -- new pts summed to {total_new}, expected {n_ref}'})")

    print(f"\n  -- Grid COVERAGE (each epoch matched back against the final "
          f"{n_ref}-point union grid) --")
    print(f"  {'epoch':<7}{'label':<{label_w}}  {'n_own_pts':>10}  {'matched':>8}  "
          f"{'NaN gap':>8}  {'coverage%':>10}")
    total_matched = 0
    worst_epoch = None
    worst_frac = None
    for i in range(n_epochs):
        idx_map = matched_indices[i]
        n_matched = int(np.sum(idx_map >= 0))
        n_gap = n_ref - n_matched
        frac = (n_matched / n_ref * 100.0) if n_ref else 0.0
        total_matched += n_matched
        if worst_frac is None or frac < worst_frac:
            worst_frac = frac
            worst_epoch = i
        print(f"  {i:<7}{labels[i][:label_w]:<{label_w}}  {n_own_points[i]:>10}  "
              f"{n_matched:>8}  {n_gap:>8}  {frac:>9.1f}%")

    total_cells = n_ref * n_epochs
    total_nan = total_cells - total_matched
    overall_frac = (total_matched / total_cells * 100.0) if total_cells else 0.0
    print(f"  {'':<7}{'TOTAL':<{label_w}}  {'':>10}  {total_matched:>8}  {total_nan:>8}  "
          f"{overall_frac:>9.1f}%")
    print(f"\n  NaN cells this remap will introduce into models_clean : "
          f"{total_nan} / {total_cells} EC cells ({100 - overall_frac:.1f}%), "
          f"before layers are even factored in.")
    if worst_epoch is not None and worst_frac is not None and worst_frac < 50.0:
        print(f"  [WARN] Epoch {worst_epoch} ('{labels[worst_epoch]}') covers only "
              f"{worst_frac:.1f}% of the union reference grid -- if this looks lower "
              f"than expected, check that epoch's own point count/coordinates and "
              f"consider raising tolerance_m (currently {tolerance_m} m).")
    print(f"{'='*78}\n")


# ---------------------------------------------------------------------------
# 5. Coordinate-system (CRS) classification, reprojection, and consistency
#    checking
#
# Fixes: "TDR <-> EM spatial '1 m window' -- verify it is actually matching
# in the same CRS". _prepare_real_em_csv's _COLUMN_VARIANTS mapping accepts
# EITHER a metric alias (X/Easting, Y/Northing) OR a geographic alias
# (Longitude/lon/LON/LONGITUDE, Latitude/lat/LAT/LATITUDE) for the same
# canonical 'x'/'y' columns, but every Problem.createSurvey(...) call on the
# real-data path passed targetProjection=None regardless of which alias
# actually matched. A GEM-2 export using the geographic alias therefore had
# survey_df['x']/['y'] left in decimal DEGREES, while probe_table's
# utm_x/utm_y (from SOILNET_MFC2_coordinates_sensors.csv) are in METRES --
# subtracting the two in em_methodology._probe_distances produces a
# "distance" off by ~5 orders of magnitude, so with window_m=1.0 literally
# every TDR probe falls back to nearest-point matching, and that fallback
# match is itself meaningless (nearest in mixed degrees/metres units, not
# nearest geographically).
#
# These helpers are consolidated here (rather than duplicated per-caller,
# following this module's own stated purpose) so every consumer that reads
# a raw real-data EM CSV -- run_pipeline.py's _prepare_real_em_csv,
# plot_pre_inversion_diagnostics.py's local copy of the same function, and
# plot_em_diagnostics_real.py's _parse_raw_csv -- classifies and reprojects
# coordinates identically, instead of three independently-drifting copies
# of the same alias table making three independently-drifting assumptions
# about units.
# ---------------------------------------------------------------------------

_METRIC_COORD_ALIASES: dict[str, set[str]] = {
    "x": {"X", "Easting"},
    "y": {"Y", "Northing"},
}
_GEOGRAPHIC_COORD_ALIASES: dict[str, set[str]] = {
    "x": {"Longitude", "lon", "LON", "LONGITUDE"},
    "y": {"Latitude", "lat", "LAT", "LATITUDE"},
}


def classify_coordinate_alias(raw_column: str, axis: str) -> str:
    """
    Classify a raw CSV column name as ``"metric"`` (already in the survey's
    local projected CRS, e.g. UTM easting/northing in metres) or
    ``"geographic"`` (decimal-degree longitude/latitude, WGS84) based on
    which alias in ``run_pipeline.py::_prepare_real_em_csv``'s
    ``_COLUMN_VARIANTS`` table it matched, or ``"unknown"`` if it matches
    neither known alias set for that axis.

    This is a table lookup on the COLUMN NAME, deliberately not a heuristic
    on the column's numeric values -- see the "Non-goals" note on the
    em_unit fix this pipeline already applies: guessing units/CRS from
    magnitude is explicitly avoided in favour of an unambiguous, explicit
    signal (here, which literal alias was present in the raw file).

    Parameters
    ----------
    raw_column : str
        The exact raw column name found in the CSV (e.g. ``'Longitude'``,
        ``'X'``, ``'Easting'``) -- NOT the canonical ``'x'``/``'y'`` name it
        gets renamed to.
    axis : {'x', 'y'}

    Returns
    -------
    str
        ``"metric"``, ``"geographic"``, or ``"unknown"``.
    """
    if raw_column in _METRIC_COORD_ALIASES.get(axis, ()):
        return "metric"
    if raw_column in _GEOGRAPHIC_COORD_ALIASES.get(axis, ()):
        return "geographic"
    return "unknown"


def auto_utm_epsg_from_lonlat(lon_mean: float, lat_mean: float) -> str:
    """
    Return the standard WGS84 UTM zone EPSG code (``'EPSG:326XX'`` for the
    northern hemisphere, ``'EPSG:327XX'`` for the southern) covering the
    given mean longitude/latitude, using the ordinary 6-degree UTM zoning
    formula. This is a real geodetic computation from the survey's own
    coordinates (not a guess), used so a real-data scenario doesn't need a
    manually-configured target EPSG for the common case.

    Parameters
    ----------
    lon_mean, lat_mean : float
        Mean longitude/latitude [decimal degrees, WGS84] of the point set
        being reprojected.

    Returns
    -------
    str
        EPSG code string, e.g. ``'EPSG:32633'`` for UTM Zone 33N.
    """
    zone = int((lon_mean + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    epsg_num = (32600 if lat_mean >= 0 else 32700) + zone
    return f"EPSG:{epsg_num}"


def reproject_lonlat_to_utm(
    lon: np.ndarray,
    lat: np.ndarray,
    epsg: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Reproject WGS84 decimal-degree longitude/latitude arrays to metric UTM
    coordinates via ``pyproj``.

    Parameters
    ----------
    lon, lat : np.ndarray
        Decimal-degree WGS84 coordinates.
    epsg : str, optional
        Target EPSG code (e.g. ``'EPSG:32633'``). When ``None`` (default),
        the UTM zone is auto-derived from the MEAN of ``lon``/``lat`` via
        ``auto_utm_epsg_from_lonlat`` -- appropriate for a single survey
        whose points don't straddle a UTM zone boundary (true for any
        realistic single-site field survey).

    Returns
    -------
    x_m, y_m : np.ndarray
        Projected easting/northing [m], same shape as ``lon``/``lat``.
    epsg_used : str
        The EPSG code actually used (useful for logging when auto-derived).

    Raises
    ------
    ImportError
        If ``pyproj`` is not installed -- raised with an actionable message
        rather than silently leaving coordinates in degrees.
    """
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError(
            "reproject_lonlat_to_utm() requires the 'pyproj' package to "
            "reproject geographic (Longitude/Latitude) EM survey coordinates "
            "to metric UTM -- install it with `pip install pyproj`. Refusing "
            "to silently leave coordinates in decimal degrees, since mixing "
            "degrees and metres elsewhere in this pipeline (e.g. TDR probe "
            "matching) produces a ~5-order-of-magnitude distance error."
        ) from exc

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    if epsg is None:
        epsg = auto_utm_epsg_from_lonlat(float(np.mean(lon)), float(np.mean(lat)))

    transformer = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    x_m, y_m = transformer.transform(lon, lat)
    return np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float), epsg


def check_crs_consistency(
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
    ratio_threshold: float = 10.0,
) -> str | None:
    """
    Defense-in-depth sanity check, independent of the alias-based
    classification above: compare two point clouds that are supposed to
    describe the same physical survey area (e.g. EM measurement points vs.
    TDR probe locations) and flag it if their spatial extents/centroids look
    inconsistent by roughly an order of magnitude or more -- the signature
    of one side being in decimal degrees and the other in metres (or any
    other CRS mismatch), independent of whether the alias-classification
    fix above already ran or was somehow bypassed.

    This intentionally does NOT try to identify or fix the mismatch -- only
    detect and report it, mirroring the em_unit sanity check's philosophy
    (warn loudly, let a human decide) rather than silently guessing a
    correction.

    Parameters
    ----------
    x_a, y_a : np.ndarray
        Coordinates of point cloud A [m or degrees].
    x_b, y_b : np.ndarray
        Coordinates of point cloud B [m or degrees].
    label_a, label_b : str, optional
        Human-readable labels for the two point clouds, used in the
        returned message.
    ratio_threshold : float, optional
        If the larger of the two point clouds' spans is more than this many
        times the smaller (or the centroid-to-centroid distance exceeds
        this many times the larger span), a warning message is returned.
        Default 10.0.

    Returns
    -------
    str or None
        A human-readable warning message if the two point clouds look
        CRS-inconsistent, else ``None``.
    """
    x_a = np.asarray(x_a, dtype=float)
    y_a = np.asarray(y_a, dtype=float)
    x_b = np.asarray(x_b, dtype=float)
    y_b = np.asarray(y_b, dtype=float)

    if x_a.size == 0 or x_b.size == 0:
        return None

    span_a = max(float(np.ptp(x_a)), float(np.ptp(y_a)), 1e-9)
    span_b = max(float(np.ptp(x_b)), float(np.ptp(y_b)), 1e-9)
    centroid_a = np.array([np.mean(x_a), np.mean(y_a)])
    centroid_b = np.array([np.mean(x_b), np.mean(y_b)])
    centroid_dist = float(np.linalg.norm(centroid_a - centroid_b))

    span_ratio = max(span_a, span_b) / min(span_a, span_b)
    centroid_ratio = centroid_dist / max(span_a, span_b)

    if span_ratio > ratio_threshold or centroid_ratio > ratio_threshold:
        return (
            f"'{label_a}' (span~{span_a:.2f}, centroid=({centroid_a[0]:.2f}, "
            f"{centroid_a[1]:.2f})) vs '{label_b}' (span~{span_b:.2f}, "
            f"centroid=({centroid_b[0]:.2f}, {centroid_b[1]:.2f})) -- "
            f"span ratio {span_ratio:.1f}x, centroid offset {centroid_dist:.2f} "
            f"({centroid_ratio:.1f}x the larger span). This is the signature "
            f"of a coordinate-system mismatch (e.g. one side in decimal "
            f"degrees, the other in metres) rather than a genuinely distant "
            f"survey point."
        )
    return None
