#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_pre_inversion_diagnostics.py
==================================
Pre-inversion sanity-check diagnostics for the real-data branch of the
EA-EM pipeline (run_pipeline.py).

Runs everything Step 0 / Step 1 does UP TO (but not including) the
Step 1 ensemble inversion itself:

  1. IMPORT           -- data_real.prepare_real_dataset()
                          (EM file discovery + TDR<->EM date matching).
  2. PRE-PROCESSING    -- delimiter-detection + column-rename of each raw
                          GEM-2 CSV (mirrors run_pipeline.py::_prepare_real_em_csv).
  3. FILTERING         -- duplicate/repeated-point removal per epoch
                          (data_real.filter_repeated_em_points,
                          i.e. emagpy's own Problem.filterRepeated).
  4. SPATIAL MATCHING  -- cross-epoch spatial reference-grid construction
                          (geometry_utils.match_epochs_to_reference_grid),
                          the step that replaced the old `arr[:min_pts]`
                          row-count trim -- see run_pipeline.py Step 1 /
                          pipeline_review.md, "spatial reference-grid
                          matching, not row trimming".

None of this requires the (potentially 30+ minute) Step 1 ensemble
inversion, so it is meant to be run FIRST -- right after Step 0 -- to
catch a wrong em_unit, a badly-set spatial_match_tolerance_m, a
collapsing TDR<->EM date match, or a filterRepeated tolerance that's
dropping too many/too few points, before committing to a full run.

Figures produced (under ``<output_dir>/diagnostics/pre_inversion/``)
----------------------------------------------------------------------
import_filter_summary.png
    Per-epoch bars: raw point count -> after filterRepeated -> (if
    time-lapse) matched onto the spatial reference grid, plus the
    TDR<->EM date-matching yield per epoch (flags epochs with zero
    matched TDR observations).

reference_grid_coverage.png
    Every reference-grid point (union across all epochs), coloured by
    how many of the N epochs actually cover it. TDR probes overlaid as
    black crosses. Direct visual check of match_epochs_to_reference_grid
    + spatial_match_tolerance_m.

tolerance_sensitivity.png   (only produced if tolerance_sweep is given)
    Reference-grid size and mean per-epoch coverage vs. a sweep of
    candidate spatial_match_tolerance_m values, to help pick a good
    value before committing to the scenario config.

Public API
----------
    run_pre_inversion_diagnostics(em_dir, tdr_dir, coord_file, output_dir, ...)

Standalone CLI
--------------
    python plot_pre_inversion_diagnostics.py --scenario 2d_real
    python plot_pre_inversion_diagnostics.py --em-dir ./data/EMI_MFC2/ \\
        --tdr-dir ./data/daily_MFC/ --spatial-match-tolerance-m 2.0 \\
        --tolerance-sweep 0.5 1 2 3 5 10
"""

from __future__ import annotations

import os
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import geometry_utils
from plots._common import ensure_dir as _ensure_dir


# ---------------------------------------------------------------------------
# Column aliases -- kept in sync with run_pipeline.py::_prepare_real_em_csv
# and plot_em_diagnostics_real.py's own copy. Duplicated locally (rather
# than imported from run_pipeline.py) so this diagnostic doesn't have to
# pull in run_pipeline's full downstream dependency chain
# (data_synthetic / step2_petro / step3_average / plot_water_content
# / etc.) just to reuse one small CSV-renaming helper -- this mirrors the
# same duplication pattern already used across the pipeline's other
# diagnostic modules (see plot_em_diagnostics_real.py's own _COLUMN_VARIANTS).
# ---------------------------------------------------------------------------
_COLUMN_VARIANTS: dict[str, list[str]] = {
    "x":                    ["X", "Longitude", "lon", "LON", "LONGITUDE", "Easting"],
    "y":                    ["Y", "Latitude",  "lat", "LAT", "LATITUDE",  "Northing"],
    "elevation":            ["GPSalt", "Altitude", "Alt", "ALT", "altitude", "ALTITUDE", "Height"],
    "HCP1.66f425h1_quad":   ["Q_425Hz",   "Q425Hz",   "Q_425",   "Q425"],
    "HCP1.66f1525h1_quad":  ["Q_1525Hz",  "Q1525Hz",  "Q_1525",  "Q1525"],
    "HCP1.66f5325h1_quad":  ["Q_5325Hz",  "Q5325Hz",  "Q_5325",  "Q5325"],
    "HCP1.66f18325h1_quad": ["Q_18325Hz", "Q18325Hz", "Q_18325", "Q18325"],
    "HCP1.66f63025h1_quad": ["Q_63025Hz", "Q63025Hz", "Q_63025", "Q63025"],
    "HCP1.66f92775h1_quad": ["Q_92775Hz", "Q92775Hz", "Q_92775", "Q92775"],
}


def _import_real_data_loader():
    """Import the real-data loader module (data_real.py, next to this script)."""
    import data_real
    return data_real


def _prepare_real_em_csv(raw_path: str, processed_dir: str, survey_date=None
                         ) -> tuple[str, str, "str | None"]:
    """Self-contained copy of run_pipeline.py::_prepare_real_em_csv (delimiter
    auto-detection + column-alias rename + geographic->UTM reprojection) --
    see module docstring for why this is duplicated locally rather than
    imported. Kept in sync with run_pipeline.py's version, including the
    CRS-mismatch fix: if the raw file's x/y columns resolved via the
    geographic alias (Longitude/Latitude), they are reprojected to metric
    UTM here too (via geometry_utils.reproject_lonlat_to_utm), so this
    diagnostic's own TDR<->EM matching figure (Figure 4) is checking the
    SAME coordinates Step 1 will actually use, not raw decimal degrees.

    Returns (out_path, coord_kind, epsg_used) -- see run_pipeline.py's
    version for the full contract.
    """
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "processed_" + os.path.basename(raw_path))

    with open(raw_path, "r", errors="replace") as fh:
        first_line = ""
        for line in fh:
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break
    delimiter = ";" if first_line.count(";") >= first_line.count(",") else ","

    df = pd.read_csv(raw_path, delimiter=delimiter)
    df.columns = df.columns.str.strip()
    available = set(df.columns)

    rename_map: dict[str, str] = {}
    unresolved: list[str] = []
    for emagpy_col, aliases in _COLUMN_VARIANTS.items():
        matched = next((a for a in aliases if a in available), None)
        if matched:
            rename_map[matched] = emagpy_col
        else:
            unresolved.append(emagpy_col)

    if unresolved:
        raise ValueError(
            f"[PRE-INVERSION DIAG] Cannot preprocess '{os.path.basename(raw_path)}': "
            f"no recognised alias for emagpy columns {unresolved} "
            f"(delimiter='{delimiter}', found columns: {sorted(available)})."
        )

    x_raw = next(raw for raw, canon in rename_map.items() if canon == "x")
    y_raw = next(raw for raw, canon in rename_map.items() if canon == "y")
    x_kind = geometry_utils.classify_coordinate_alias(x_raw, "x")
    y_kind = geometry_utils.classify_coordinate_alias(y_raw, "y")
    if x_kind != y_kind or x_kind == "unknown":
        raise ValueError(
            f"[PRE-INVERSION DIAG] Cannot preprocess '{os.path.basename(raw_path)}': "
            f"x resolved via '{x_raw}' ('{x_kind}') but y resolved via '{y_raw}' "
            f"('{y_kind}') -- inconsistent/unrecognised coordinate columns."
        )
    coord_kind = x_kind

    out_df = df[list(rename_map.keys())].rename(columns=rename_map)

    epsg_used: "str | None" = None
    if coord_kind == "geographic":
        x_m, y_m, epsg_used = geometry_utils.reproject_lonlat_to_utm(
            out_df["x"].to_numpy(dtype=float), out_df["y"].to_numpy(dtype=float),
        )
        print(
            f"[PRE-INVERSION DIAG] '{os.path.basename(raw_path)}': x/y resolved via "
            f"geographic alias ('{x_raw}'/'{y_raw}') -- reprojected to metric UTM "
            f"({epsg_used}, auto-detected)."
        )
        out_df["x"] = x_m
        out_df["y"] = y_m

    if survey_date is not None:
        out_df["date"] = pd.Timestamp(survey_date).strftime("%Y-%m-%d")
    out_df.to_csv(out_path, index=False)
    return out_path, coord_kind, epsg_used


# ===========================================================================
# PUBLIC ENTRY POINT
# ===========================================================================

def run_pre_inversion_diagnostics(
    em_dir: str,
    tdr_dir: str,
    coord_file: str,
    output_dir: str,
    time_tolerance_days: float = 1.0,
    spatial_match_tolerance_m: float = 2.0,
    em_unit: str = "ppt",
    filter_repeated_tolerance: float = 1e-10,
    tolerance_sweep: "list[float] | None" = None,
    window_m: float = 1.0,
    matching_warn_fraction: float = 0.5,
) -> dict:
    """
    Run the import -> pre-process -> filter -> spatial-match chain and
    render diagnostic figures, WITHOUT running any EM inversion.

    Parameters mirror the corresponding scenario/CLI fields in
    run_pipeline.py / scenario_loader.py so this can be pointed at the
    exact same real-data scenario before launching the full pipeline.

    ``window_m`` / ``matching_warn_fraction`` drive the TDR-probe <-> EM-point
    matching figure (which EM measurement point each TDR station is actually
    matched against) -- see FIGURE 4 below. This uses the same spatial
    window-matching logic as ``em_methodology.extract_sample_positions_from_survey``
    (the real function Step 1 calls), but needs no inverted EC values, so it
    is safe to run here purely from station/EM coordinates.

    Returns
    -------
    dict with keys: 'figures' (list of PNG paths), 'n_epochs',
    'n_ref_points' (None for single-survey runs), 'epoch_records',
    'zero_obs_epochs', 'matching_summary' (None if it could not be computed).
    """
    diag_dir = _ensure_dir(os.path.join(output_dir, "diagnostics", "pre_inversion"))
    print(f"\n{'='*78}")
    print(" [PRE-INVERSION DIAG] Import / filter / spatial-matching check")
    print(f"{'='*78}")
    print(f"  EM directory              : {em_dir}")
    print(f"  TDR directory             : {tdr_dir}")
    print(f"  Coordinates               : {coord_file}")
    print(f"  em_unit                   : {em_unit}")
    print(f"  time_tolerance_days       : {time_tolerance_days}")
    print(f"  spatial_match_tolerance_m : {spatial_match_tolerance_m}")
    print(f"  filter_repeated_tolerance : {filter_repeated_tolerance}")

    loader = _import_real_data_loader()

    # ------------------------------------------------------------------
    # 1. IMPORT -- real EM file discovery + TDR<->EM date matching
    # ------------------------------------------------------------------
    raw_em_files, em_dates, probe_table, observations = loader.prepare_real_dataset(
        em_dir=em_dir, tdr_dir=tdr_dir, coord_file=coord_file,
        time_tolerance_days=time_tolerance_days,
    )
    n_epochs = len(raw_em_files)
    timelapse = n_epochs > 1
    print(f"[IMPORT] {n_epochs} EM survey epoch(s), "
          f"{len(probe_table)} TDR station(s), "
          f"{len(observations)} matched (station, epoch, depth) observation(s).")

    obs_per_epoch = {i: 0 for i in range(n_epochs)}
    for obs in observations:
        obs_per_epoch[obs["time_index"]] = obs_per_epoch.get(obs["time_index"], 0) + 1
    zero_obs_epochs = [i for i, c in obs_per_epoch.items() if c == 0]
    if zero_obs_epochs:
        print(f"[IMPORT][WARN] {len(zero_obs_epochs)} epoch(s) matched ZERO TDR "
              f"observations within +/-{time_tolerance_days} day(s): epoch index(es) "
              f"{zero_obs_epochs}. Consider raising time_tolerance_days.")

    # ------------------------------------------------------------------
    # 2 & 3. PRE-PROCESS + FILTER each raw EM survey CSV
    # ------------------------------------------------------------------
    processed_dir = os.path.join(em_dir, "_emagpy_ready")
    epoch_records: list[dict] = []
    for t_idx, (raw_path, survey_date) in enumerate(zip(raw_em_files, em_dates)):
        record = {"t_idx": t_idx, "raw_path": raw_path, "date": survey_date}
        try:
            proc_path, coord_kind, epsg_used = _prepare_real_em_csv(
                raw_path, processed_dir, survey_date=survey_date,
            )

            n_before = len(pd.read_csv(proc_path))
            n_after = n_before
            try:
                from emagpy import Problem
                k = Problem()
                k.createSurvey(proc_path, unit=em_unit)
                n_before = len(k.surveys[0].df)
                k.filterRepeated(tolerance=filter_repeated_tolerance)
                n_after = len(k.surveys[0].df)
                if n_after < n_before:
                    k.surveys[0].df.to_csv(proc_path, index=False)
            except Exception as e:
                print(f"  [WARN] filterRepeated check failed for epoch {t_idx}: {e}")

            df_proc = pd.read_csv(proc_path)
            x = df_proc["x"].to_numpy(dtype=float)
            y = (df_proc["y"].to_numpy(dtype=float) if "y" in df_proc.columns
                 else np.zeros_like(x))

            record.update({
                "processed_path": proc_path,
                "n_raw": n_before,
                "n_filtered": n_after,
                "x": x, "y": y,
                "coord_kind": coord_kind,
                "epsg_used": epsg_used,
                "error": None,
            })
            epsg_note = f", reprojected via {epsg_used}" if epsg_used else ""
            print(f"  [OK] epoch {t_idx} ({os.path.basename(raw_path)}): "
                  f"{n_before} -> {n_after} pts after filterRepeated "
                  f"(tolerance={filter_repeated_tolerance}; coords: {coord_kind}{epsg_note})")
        except Exception as e:
            record.update({"processed_path": None, "n_raw": None, "n_filtered": None,
                           "x": np.array([]), "y": np.array([]),
                           "coord_kind": None, "epsg_used": None, "error": str(e)})
            print(f"  [ERROR] epoch {t_idx} ({os.path.basename(raw_path)}) failed: {e}")
        epoch_records.append(record)

    n_ok = sum(1 for r in epoch_records if r["error"] is None)
    if n_ok == 0:
        raise RuntimeError(
            "[PRE-INVERSION DIAG] Every EM epoch failed import/pre-processing -- "
            "see [ERROR] lines above. Cannot proceed to spatial matching."
        )

    # ------------------------------------------------------------------
    # CRS CONSISTENCY SANITY CHECK -- same defense-in-depth check
    # run_pipeline.py's _load_real_data now performs: compare the
    # (now-metric-by-construction) EM survey coordinates against
    # probe_table's own utm_x/utm_y, independent of whether the alias
    # classification above already caught a mismatch.
    # ------------------------------------------------------------------
    try:
        _first_ok = next(r for r in epoch_records if r["error"] is None)
        _px = probe_table["utm_x"].to_numpy(dtype=float) if "utm_x" in probe_table.columns else np.array([])
        _py = probe_table["utm_y"].to_numpy(dtype=float) if "utm_y" in probe_table.columns else np.array([])
        _valid_probes = (_px != 0.0) | (_py != 0.0)
        if _px.size and _valid_probes.any() and _first_ok["x"].size:
            _crs_msg = geometry_utils.check_crs_consistency(
                _first_ok["x"], _first_ok["y"], _px[_valid_probes], _py[_valid_probes],
                label_a=f"EM survey ({os.path.basename(_first_ok['raw_path'])})",
                label_b="TDR probes (probe_table utm_x/utm_y)",
            )
            if _crs_msg:
                print(f"\n[PRE-INVERSION DIAG][WARN] Possible CRS mismatch between EM "
                      f"survey and TDR probe coordinates: {_crs_msg}")
            else:
                print(f"\n[PRE-INVERSION DIAG] CRS consistency check: EM survey and TDR "
                      f"probe coordinates occupy consistent spatial extents. OK.")
    except Exception as _crs_exc:
        print(f"[PRE-INVERSION DIAG][WARN] CRS consistency check failed to run "
              f"({type(_crs_exc).__name__}: {_crs_exc}) -- skipping.")

    # ------------------------------------------------------------------
    # 4. SPATIAL REFERENCE-GRID MATCHING (time-lapse only)
    # ------------------------------------------------------------------
    ref_x = ref_y = None
    matched_indices = None
    if timelapse:
        epoch_xy = [(r["x"], r["y"]) for r in epoch_records if r["error"] is None]
        epoch_labels_ok = [os.path.basename(r["raw_path"]) for r in epoch_records
                           if r["error"] is None]
        ref_x, ref_y, matched_indices = geometry_utils.match_epochs_to_reference_grid(
            epoch_xy, tolerance_m=spatial_match_tolerance_m,
            verbose=True, epoch_labels=epoch_labels_ok,
        )
        n_ref = len(ref_x)
    else:
        print("\n[REFERENCE GRID] Single-survey (non-time-lapse) real-data run -- "
              "spatial reference-grid matching does not apply.")

    # ------------------------------------------------------------------
    # FIGURE 1 -- import & filtering summary
    # ------------------------------------------------------------------
    fig1_path = _plot_import_filter_summary(
        epoch_records, obs_per_epoch, time_tolerance_days,
        n_ref=(len(ref_x) if ref_x is not None else None),
        matched_indices=matched_indices, diag_dir=diag_dir,
    )

    # ------------------------------------------------------------------
    # FIGURE 2 -- reference-grid coverage map
    # ------------------------------------------------------------------
    fig2_path = ""
    if timelapse and ref_x is not None:
        fig2_path = _plot_reference_grid_coverage(
            ref_x, ref_y, matched_indices, probe_table,
            spatial_match_tolerance_m, diag_dir,
        )

    # ------------------------------------------------------------------
    # FIGURE 3 -- optional tolerance sensitivity sweep
    # ------------------------------------------------------------------
    fig3_path = ""
    if timelapse and tolerance_sweep:
        fig3_path = _plot_tolerance_sensitivity(
            [(r["x"], r["y"]) for r in epoch_records if r["error"] is None],
            tolerance_sweep, spatial_match_tolerance_m, diag_dir,
        )

    # ------------------------------------------------------------------
    # FIGURE 4 -- TDR probe <-> EM point matching ("which EM point does
    # each TDR sensor actually use?"). Uses the SAME coordinate set that
    # models_clean's spatial axis will represent once Step 1 runs: the
    # union reference grid for time-lapse real data, or the single epoch's
    # own positions otherwise -- exactly what run_pipeline.py's Step 1
    # matches TDR stations against (see EMmethod.extract_sample_positions_
    # from_survey / station_to_spatial_idx there). No inverted EC values
    # are needed for this -- only coordinates + window_m -- so it is safe
    # to compute here, before Step 1.
    # ------------------------------------------------------------------
    fig4_path = ""
    matching_summary = None
    try:
        if timelapse and ref_x is not None:
            survey_df = pd.DataFrame({"x": ref_x, "y": ref_y})
            coord_note = (
                f"{len(ref_x)}-point union reference grid "
                f"(tolerance={spatial_match_tolerance_m} m)"
            )
        else:
            first_ok = next(r for r in epoch_records if r["error"] is None)
            survey_df = pd.DataFrame({"x": first_ok["x"], "y": first_ok["y"]})
            coord_note = f"single-survey epoch {first_ok['t_idx']}'s own {len(survey_df)} points"

        fig4_path, matching_summary = _plot_tdr_em_matching(
            survey_df=survey_df,
            probe_table=probe_table,
            window_m=window_m,
            warn_fraction_threshold=matching_warn_fraction,
            coord_note=coord_note,
            output_dir=output_dir,
        )
    except Exception as _match_exc:
        print(f"[PRE-INVERSION DIAG][WARN] TDR<->EM matching figure failed: {_match_exc}")

    print(f"\n[PRE-INVERSION DIAG] Done -- figures written to '{diag_dir}'"
          + (f" and '{os.path.join(output_dir, 'diagnostics')}'" if fig4_path else "") + "\n")

    return {
        "figures": [p for p in (fig1_path, fig2_path, fig3_path, fig4_path) if p],
        "n_epochs": n_epochs,
        "n_ref_points": (len(ref_x) if ref_x is not None else None),
        "epoch_records": epoch_records,
        "zero_obs_epochs": zero_obs_epochs,
        "matching_summary": matching_summary,
    }


def _sniff_delimiter(raw_path: str) -> str:
    """Same first-non-empty-line delimiter heuristic as _prepare_real_em_csv,
    factored out so callers that only need a row COUNT (not the renamed/
    reprojected DataFrame) don't have to duplicate it."""
    with open(raw_path, "r", errors="replace") as fh:
        first_line = ""
        for line in fh:
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break
    return ";" if first_line.count(";") >= first_line.count(",") else ","


def run_pre_inversion_diagnostics_from_pipeline(
    em_files: "list[str]",
    raw_em_files: "list[str]",
    em_dates: list,
    probe_table,
    observations: "list[dict]",
    output_dir: str,
    time_tolerance_days: float = 1.0,
    spatial_match_tolerance_m: float = 2.0,
    tolerance_sweep: "list[float] | None" = None,
    window_m: float = 1.0,
    matching_warn_fraction: float = 0.5,
) -> dict:
    """
    Same diagnostic figures as ``run_pre_inversion_diagnostics``, but built
    from data the pipeline (run_pipeline.py's ``_load_real_data``, called at
    Step 0) has ALREADY imported, pre-processed (delimiter detection +
    column-alias rename + geographic->UTM reprojection), and filtered
    (``Problem.filterRepeated``) -- rather than re-running that whole
    IMPORT -> PRE-PROCESS -> FILTER chain a second time here, which is what
    ``run_pre_inversion_diagnostics`` does for standalone/CLI use.

    Only the SPATIAL MATCHING step (building the cross-epoch reference
    grid) and the figure rendering are actually (re)computed here -- and
    that step only needs x/y positions, which cost nothing to re-read from
    the already-processed CSVs; it is NOT redundant with anything the
    pipeline computes before its own (post-inversion) reference-grid
    remapping in Step 1, since that one needs the already-inverted models
    to exist first. Running it here means the coverage/collapse checks are
    available before committing to the potentially 30+ minute Step 1
    ensemble inversion, using the pipeline's own already-loaded objects
    instead of a second, independent load.

    Parameters
    ----------
    em_files : list[str]
        PROCESSED, already ``filterRepeated``-filtered EM survey CSV paths
        -- i.e. ``EM_FNAMES`` / the first element of ``_load_real_data``'s
        return tuple in run_pipeline.py. One per epoch, same order as
        ``raw_em_files`` and ``em_dates``.
    raw_em_files : list[str]
        The ORIGINAL, untouched GEM-2 CSV paths (``RAW_EM_FNAMES`` in
        run_pipeline.py) -- read only to report the pre-filter "raw" point
        count for Figure 1; never re-parsed for column aliases here.
    em_dates : list
        One date/timestamp per epoch (``EM_DATES`` in run_pipeline.py).
    probe_table : pd.DataFrame
        Already-loaded TDR probe/station table (``utm_x``/``utm_y``
        columns), as returned by ``_load_real_data``.
    observations : list[dict]
        Already date-matched (station, epoch, depth) observations
        (``observations_list`` in run_pipeline.py) -- used only to count
        TDR<->EM date-matching yield per epoch for Figure 1's lower panel.
    output_dir, time_tolerance_days, spatial_match_tolerance_m,
    tolerance_sweep, window_m, matching_warn_fraction :
        Same meaning as in ``run_pre_inversion_diagnostics``.

    Returns
    -------
    Same dict shape as ``run_pre_inversion_diagnostics``: 'figures',
    'n_epochs', 'n_ref_points' (None for single-survey runs),
    'epoch_records', 'zero_obs_epochs', 'matching_summary'.
    """
    diag_dir = _ensure_dir(os.path.join(output_dir, "diagnostics", "pre_inversion"))
    n_epochs = len(em_files)
    timelapse = n_epochs > 1
    print(f"\n{'='*78}")
    print(" [PRE-INVERSION DIAG] Reusing pipeline-loaded import/filter results")
    print(f"{'='*78}")
    print(f"  EM epoch(s) (already loaded) : {n_epochs}")
    print(f"  TDR station(s) (probe_table) : {len(probe_table)}")
    print(f"  spatial_match_tolerance_m     : {spatial_match_tolerance_m}")

    obs_per_epoch = {i: 0 for i in range(n_epochs)}
    for obs in observations:
        obs_per_epoch[obs["time_index"]] = obs_per_epoch.get(obs["time_index"], 0) + 1
    zero_obs_epochs = [i for i, c in obs_per_epoch.items() if c == 0]
    if zero_obs_epochs:
        print(f"[IMPORT][WARN] {len(zero_obs_epochs)} epoch(s) matched ZERO TDR "
              f"observations within +/-{time_tolerance_days} day(s): epoch index(es) "
              f"{zero_obs_epochs}. Consider raising time_tolerance_days.")

    # ------------------------------------------------------------------
    # epoch_records -- built from the ALREADY processed/filtered files.
    # Only row COUNTS and x/y positions are read here; no delimiter
    # sniffing + column-alias rename + reprojection (already done by
    # _load_real_data) and no re-running filterRepeated (ditto).
    # ------------------------------------------------------------------
    epoch_records: list[dict] = []
    for t_idx, (raw_path, proc_path, survey_date) in enumerate(
        zip(raw_em_files, em_files, em_dates)
    ):
        try:
            n_raw = len(pd.read_csv(raw_path, delimiter=_sniff_delimiter(raw_path)))
            df_proc = pd.read_csv(proc_path)
            n_filtered = len(df_proc)
            x = df_proc["x"].to_numpy(dtype=float)
            y = (df_proc["y"].to_numpy(dtype=float) if "y" in df_proc.columns
                 else np.zeros_like(x))
            epoch_records.append({
                "t_idx": t_idx, "raw_path": raw_path, "date": survey_date,
                "processed_path": proc_path, "n_raw": n_raw, "n_filtered": n_filtered,
                "x": x, "y": y, "error": None,
            })
            print(f"  [OK] epoch {t_idx} ({os.path.basename(raw_path)}): "
                  f"{n_raw} raw -> {n_filtered} after filterRepeated "
                  f"(already applied by the pipeline's own Step 0 load).")
        except Exception as e:
            epoch_records.append({
                "t_idx": t_idx, "raw_path": raw_path, "date": survey_date,
                "processed_path": None, "n_raw": None, "n_filtered": None,
                "x": np.array([]), "y": np.array([]), "error": str(e),
            })
            print(f"  [ERROR] epoch {t_idx} ({os.path.basename(raw_path)}) failed: {e}")

    n_ok = sum(1 for r in epoch_records if r["error"] is None)
    if n_ok == 0:
        raise RuntimeError(
            "[PRE-INVERSION DIAG] Every EM epoch's already-processed CSV failed to "
            "re-read -- see [ERROR] lines above. Cannot proceed to spatial matching."
        )

    # ------------------------------------------------------------------
    # SPATIAL REFERENCE-GRID MATCHING (time-lapse only) -- the one thing
    # that genuinely needs (re)computing here; see docstring above.
    # ------------------------------------------------------------------
    ref_x = ref_y = None
    matched_indices = None
    if timelapse:
        epoch_xy = [(r["x"], r["y"]) for r in epoch_records if r["error"] is None]
        epoch_labels_ok = [os.path.basename(r["raw_path"]) for r in epoch_records
                           if r["error"] is None]
        ref_x, ref_y, matched_indices = geometry_utils.match_epochs_to_reference_grid(
            epoch_xy, tolerance_m=spatial_match_tolerance_m,
            verbose=True, epoch_labels=epoch_labels_ok,
        )
        n_ref = len(ref_x)
    else:
        print("\n[REFERENCE GRID] Single-survey (non-time-lapse) real-data run -- "
              "spatial reference-grid matching does not apply.")

    # ------------------------------------------------------------------
    # FIGURE 1 -- import & filtering summary
    # ------------------------------------------------------------------
    fig1_path = _plot_import_filter_summary(
        epoch_records, obs_per_epoch, time_tolerance_days,
        n_ref=(len(ref_x) if ref_x is not None else None),
        matched_indices=matched_indices, diag_dir=diag_dir,
    )

    # ------------------------------------------------------------------
    # FIGURE 2 -- reference-grid coverage map
    # ------------------------------------------------------------------
    fig2_path = ""
    if timelapse and ref_x is not None:
        fig2_path = _plot_reference_grid_coverage(
            ref_x, ref_y, matched_indices, probe_table,
            spatial_match_tolerance_m, diag_dir,
        )

    # ------------------------------------------------------------------
    # FIGURE 3 -- optional tolerance sensitivity sweep
    # ------------------------------------------------------------------
    fig3_path = ""
    if timelapse and tolerance_sweep:
        fig3_path = _plot_tolerance_sensitivity(
            [(r["x"], r["y"]) for r in epoch_records if r["error"] is None],
            tolerance_sweep, spatial_match_tolerance_m, diag_dir,
        )

    # ------------------------------------------------------------------
    # FIGURE 4 -- TDR probe <-> EM point matching
    # ------------------------------------------------------------------
    fig4_path = ""
    matching_summary = None
    try:
        if timelapse and ref_x is not None:
            survey_df = pd.DataFrame({"x": ref_x, "y": ref_y})
            coord_note = (
                f"{len(ref_x)}-point union reference grid "
                f"(tolerance={spatial_match_tolerance_m} m)"
            )
        else:
            first_ok = next(r for r in epoch_records if r["error"] is None)
            survey_df = pd.DataFrame({"x": first_ok["x"], "y": first_ok["y"]})
            coord_note = f"single-survey epoch {first_ok['t_idx']}'s own {len(survey_df)} points"

        fig4_path, matching_summary = _plot_tdr_em_matching(
            survey_df=survey_df,
            probe_table=probe_table,
            window_m=window_m,
            warn_fraction_threshold=matching_warn_fraction,
            coord_note=coord_note,
            output_dir=output_dir,
        )
    except Exception as _match_exc:
        print(f"[PRE-INVERSION DIAG][WARN] TDR<->EM matching figure failed: {_match_exc}")

    print(f"\n[PRE-INVERSION DIAG] Done -- figures written to '{diag_dir}'"
          + (f" and '{os.path.join(output_dir, 'diagnostics')}'" if fig4_path else "") + "\n")

    return {
        "figures": [p for p in (fig1_path, fig2_path, fig3_path, fig4_path) if p],
        "n_epochs": n_epochs,
        "n_ref_points": (len(ref_x) if ref_x is not None else None),
        "epoch_records": epoch_records,
        "zero_obs_epochs": zero_obs_epochs,
        "matching_summary": matching_summary,
    }


# ===========================================================================
# PLOTTING HELPERS
# ===========================================================================

def _plot_import_filter_summary(epoch_records, obs_per_epoch, time_tolerance_days,
                                 n_ref, matched_indices, diag_dir) -> str:
    n = len(epoch_records)
    labels = [f"t{r['t_idx']}\n{os.path.basename(r['raw_path'])[:18]}" for r in epoch_records]
    n_raw = [r["n_raw"] if r["n_raw"] is not None else 0 for r in epoch_records]
    n_filt = [r["n_filtered"] if r["n_filtered"] is not None else 0 for r in epoch_records]
    n_obs = [obs_per_epoch.get(r["t_idx"], 0) for r in epoch_records]

    n_matched = None
    if n_ref is not None and matched_indices is not None:
        n_matched = []
        _ok_idx = 0
        for r in epoch_records:
            if r["error"] is not None:
                n_matched.append(0)
                continue
            n_matched.append(int(np.sum(matched_indices[_ok_idx] >= 0)))
            _ok_idx += 1

    x = np.arange(n)
    width = 0.25 if n_matched is not None else 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, n * 1.1), 8))

    ax1.bar(x - width, n_raw, width, label="Raw points", color="lightgrey", edgecolor="k")
    ax1.bar(x, n_filt, width, label="After filterRepeated", color="tab:blue", edgecolor="k")
    if n_matched is not None:
        ax1.bar(x + width, n_matched, width,
                label=f"Matched onto ref. grid (N={n_ref})", color="tab:green", edgecolor="k")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Point count")
    ax1.set_title("Import & duplicate-point filtering per EM survey epoch", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

    colors = ["tab:red" if c == 0 else "tab:purple" for c in n_obs]
    ax2.bar(x, n_obs, color=colors, edgecolor="k")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Matched TDR obs.")
    ax2.set_title(
        f"TDR<->EM date matching per epoch  (tolerance = +/-{time_tolerance_days} day(s); "
        f"red bars = zero matched observations)", fontsize=10,
    )
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    out_path = os.path.join(diag_dir, "import_filter_summary.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[PRE-INVERSION DIAG] Saved -> {out_path}")
    return out_path


def _plot_reference_grid_coverage(ref_x, ref_y, matched_indices, probe_table,
                                   tolerance_m, diag_dir) -> str:
    """
    Every union reference-grid point, coloured by how many of the N survey
    epochs actually cover it (0 = point exists in the union but no epoch
    landed there within tolerance -- shouldn't normally happen since every
    epoch's own points seed the grid; N = point covered by every epoch).

    NOTE: real GEM-2 surveys are very often walked as a single 1-D transect
    (y left at 0 / near-constant). ``ax.set_aspect('equal')`` on data like
    that squashes the whole plot into a hair-thin horizontal line -- easy
    to mistake for "no reference grid shown at all". This is detected below
    and an auto (non-equal) aspect is used instead, with the transect drawn
    across the full figure width and a small fixed y-margin so points are
    actually visible.
    """
    ref_x = np.asarray(ref_x, dtype=float)
    ref_y = np.asarray(ref_y, dtype=float)
    n_ref = len(ref_x)
    n_epochs = len(matched_indices)
    coverage_count = np.zeros(n_ref, dtype=int)
    for idx_map in matched_indices:
        coverage_count += (idx_map >= 0).astype(int)

    if n_ref == 0:
        print("[PRE-INVERSION DIAG][WARN] Reference grid is EMPTY (0 points) -- "
              "nothing to plot. Check that the EM survey CSVs actually contain "
              "'x'/'y' columns with real coordinates.")

    x_range = float(np.ptp(ref_x)) if n_ref else 0.0
    y_range = float(np.ptp(ref_y)) if n_ref else 0.0
    is_1d_transect = y_range < max(1e-6, 0.02 * max(x_range, 1.0))

    fig, ax = plt.subplots(figsize=(9.5, 5.5) if is_1d_transect else (8.5, 7.5))

    marker_size = 90 if is_1d_transect else 55
    sc = ax.scatter(ref_x, ref_y, c=coverage_count, cmap="RdYlGn",
                     vmin=0, vmax=max(1, n_epochs), s=marker_size,
                     edgecolor="k", linewidth=0.6, zorder=3)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                         ticks=list(range(0, n_epochs + 1)))
    cbar.set_label(f"# of {n_epochs} epochs covering this point", fontsize=9)

    if (probe_table is not None and not probe_table.empty and
            "utm_x" in probe_table.columns and "utm_y" in probe_table.columns):
        px = probe_table["utm_x"].to_numpy(float)
        py = probe_table["utm_y"].to_numpy(float)
        valid = (px != 0.0) | (py != 0.0)
        if valid.any():
            ax.scatter(px[valid], py[valid], marker="+", s=110, linewidths=1.8,
                       color="black", zorder=5, label="TDR probes")
            ax.legend(fontsize=9, loc="upper right")
            if "station_name" in probe_table.columns:
                ids = probe_table["station_name"].to_numpy()[valid]
                for x_i, y_i, id_i in zip(px[valid], py[valid], ids):
                    ax.annotate(
                        str(id_i), (x_i, y_i), xytext=(4, 4),
                        textcoords="offset points", fontsize=6.5,
                        color="black", zorder=6,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
                    )

    n_full = int(np.sum(coverage_count == n_epochs))
    n_partial = int(np.sum((coverage_count > 0) & (coverage_count < n_epochs)))
    n_zero = int(np.sum(coverage_count == 0))

    if is_1d_transect:
        ax.set_aspect("auto")
        pad = max(x_range * 0.03, 1.0)
        x_lo = float(ref_x.min()) - pad if n_ref else -1.0
        x_hi = float(ref_x.max()) + pad if n_ref else 1.0
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel("x / distance along transect [m]")
        ax.set_ylabel("(1-D transect: y ≈ 0 for all points)")
        ax.text(
            0.5, 1.10,
            "1-D transect detected — using a stretched (non-equal-aspect) view "
            "so points aren't squashed into an invisible line",
            transform=ax.transAxes, ha="center", fontsize=8, color="dimgrey",
        )
    else:
        ax.set_xlabel("x / Easting [m]")
        ax.set_ylabel("y / Northing [m]")
        ax.set_aspect("equal", adjustable="datalim")

    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title(
        f"Spatial reference grid — {n_ref} union point(s), tolerance={tolerance_m} m\n"
        f"green={n_full} pt(s) covered by ALL {n_epochs} epoch(s)  |  "
        f"yellow/orange={n_partial} pt(s) partial  |  "
        f"red={n_zero} pt(s) covered by 0 epochs (shouldn't normally occur)",
        fontsize=10,
    )

    fig.tight_layout()
    out_path = os.path.join(diag_dir, "reference_grid_coverage.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[PRE-INVERSION DIAG] Saved -> {out_path}")
    return out_path


def _plot_tdr_em_matching(survey_df, probe_table, window_m: float,
                           warn_fraction_threshold: float, coord_note: str,
                           output_dir: str):
    """
    FIGURE 4 -- "for each TDR sensor, which EM point(s) does it use?"

    This is the direct answer to that question: it reuses the exact same
    spatial window-matching logic Step 1 uses
    (``em_methodology.extract_sample_positions_from_survey``) against real
    TDR station coordinates (``probe_table['utm_x'/'utm_y']``), then renders
    it with ``plot_probe_matching.plot_probe_matching`` -- every EM point
    coloured by how many TDR stations were matched to it, with a thin line
    from each station to its matched EM point (fallback/no-window matches
    drawn in a distinct colour).

    Only spatial coordinates + ``window_m`` are needed -- NOT any inverted
    EC value -- so this is safe to compute before Step 1's ensemble
    inversion runs. A dummy all-ones EC array is passed in purely to satisfy
    ``extract_sample_positions_from_survey``'s array-shape requirement; its
    (unused here) ec_mean/ec_std diagnostic fields are meaningless and are
    not plotted.

    Returns
    -------
    (out_path, matching_summary) -- out_path is "" if plotting failed.
    """
    import em_methodology as EMmethod
    import plot_probe_matching as ppm

    if probe_table is None or probe_table.empty:
        print("[PRE-INVERSION DIAG][WARN] No probe_table -- skipping TDR<->EM matching figure.")
        return "", None
    if "utm_x" not in probe_table.columns or "utm_y" not in probe_table.columns:
        print("[PRE-INVERSION DIAG][WARN] probe_table has no utm_x/utm_y -- "
              "skipping TDR<->EM matching figure.")
        return "", None

    n_points = len(survey_df)
    dummy_models = np.ones((1, n_points, 1), dtype=float)  # shape only matters here

    position_results = EMmethod.extract_sample_positions_from_survey(
        survey_df=survey_df,
        models=dummy_models,
        probe_x_coords=probe_table["utm_x"].tolist(),
        probe_y_coords=probe_table["utm_y"].tolist(),
        probe_depths=None, depths0=None, window_m=window_m,
    )
    matching_summary = EMmethod.summarize_probe_matching(
        position_results, warn_fraction_threshold=warn_fraction_threshold,
    )
    print(f"\n[TDR<->EM MATCHING] (coords: {coord_note}, window_m={window_m})")
    print(f"  {matching_summary.message}")

    out_path = ppm.plot_probe_matching(
        survey_df=survey_df,
        position_results=position_results,
        output_dir=output_dir,   # writes to <output_dir>/diagnostics/, matching every other diagnostic module
        summary=matching_summary,
        title="TDR station -> EM point matching (pre-inversion check)",
        filename="probe_em_matching_pre_inversion.png",
    )
    return out_path, matching_summary


# ===========================================================================
# SYNTHETIC-DATA PRE-INVERSION DIAGNOSTICS
# ===========================================================================

def run_pre_inversion_diagnostics_synthetic_from_pipeline(
    em_files: "list[str]",
    output_dir: str,
    depths0: "list[float]",
    layer_centers: "list[float]",
    num_swc_positions: int = 4,
    swc_depths: "list[float] | None" = None,
    window_m: float = 1.0,
    matching_warn_fraction: float = 0.5,
    mode: str = "1d",
    utm_x: "np.ndarray | None" = None,
    utm_y: "np.ndarray | None" = None,
    probe_xyz_file: "str | None" = None,
) -> dict:
    """
    Synthetic-data counterpart of ``run_pre_inversion_diagnostics_from_pipeline``.

    Synthetic epochs share one fixed grid by construction (same generator,
    same point count/order at every time step), so none of the real-data
    machinery this module otherwise runs (import/filter summary, spatial
    reference-grid union+matching, tolerance sweep -- Figures 1-3, which
    exist specifically to handle real GEM-2 epochs that can differ in point
    count/walk order and need TDR-date matching) applies here. The one thing
    that IS meaningful to check before the (potentially long) Step 1
    ensemble inversion is the same thing Figure 4 checks on the real-data
    path: which EM measurement point does each SWC/TDR probe actually get
    matched to, spatially -- computed purely from coordinates + window_m
    (see ``em_methodology.extract_sample_positions_from_survey``), so it
    costs nothing and needs no inverted EC values.

    This reconstructs the SAME probe-placement logic ``run_pipeline.py``
    itself uses post-inversion (2-D mode -> probes spread across the UTM
    grid via ``utm_x``/``utm_y``; 1-D mode -> probes spread along the
    transect's own x range), applied directly against the raw synthetic
    survey CSV's own point count (via ``EMmethod.load_survey_positions``)
    instead of waiting for ``models_clean`` to exist. A dummy all-ones EC
    array satisfies ``extract_sample_positions_from_survey``'s shape
    requirement; only depths0-based horizon routing and window_m-based
    spatial matching are exercised.

    Also runs the same CRS-consistency sanity check used on the real-data
    path (``geometry_utils.check_crs_consistency``) between the survey's own
    x/y (as loaded from the CSV -- local grid coordinates) and the 2-D probe
    coordinates. As of the fix below, probe coordinates are always built
    from the survey's own local x_survey/y_survey (never from ``utm_x``/
    ``utm_y``, the ABSOLUTE-UTM grid offset by the Alento survey origin,
    ~515000 m E / ~4468000 m N), so both sides of this check are in the
    SAME frame by construction and it is expected to pass cleanly in
    synthetic mode. (Previously this compared local survey coordinates
    against the absolute-UTM grid and reliably flagged a false-positive
    mismatch -- see git history / pipeline_review.md for the old
    behaviour.) ``utm_x``/``utm_y`` are still accepted as parameters for
    backward compatibility but are no longer used for probe placement.

    Parameters
    ----------
    em_files : list of str
        Synthetic survey CSV path(s), one per time step (``EM_FNAMES`` in
        run_pipeline.py, coerced to a list even for a single-survey run).
    output_dir : str
        Root output directory of the current run (``cfg.output_dir`` /
        ``OUTPUT_DIR``). The figure is written to
        ``<output_dir>/diagnostics/``.
    depths0 : list of float
        Fixed layer interface depths [m] (same array passed to
        ``k.setInit()`` elsewhere in the pipeline). Used only for
        probe-depth -> horizon routing in the matching figure's legend/
        colouring; no inversion is run.
    layer_centers : list of float
        Layer-centre depths [m] (``synthetic_geometry["layer_centers"]``),
        used as the fallback probe depths when ``swc_depths`` is ``None``
        (mirrors ``run_pipeline.main()``'s own ``TDR_DEPTH`` resolution).
    num_swc_positions : int, optional
        Number of horizontal SWC/TDR probe positions (default 4, matching
        ``ScenarioConfig``'s own default).
    swc_depths : list of float, optional
        Explicit probe installation depths [m]. ``None`` (default) falls
        back to ``layer_centers``.
    window_m : float, optional
        Half-width [m] of the spatial averaging window used to match each
        probe to nearby EM measurement indices (default 1.0).
    matching_warn_fraction : float, optional
        TDR<->EM spatial-matching quality-gate warn threshold, forwarded to
        ``EMmethod.summarize_probe_matching`` (default 0.5).
    mode : {'1d', '2d'}, optional
        Spatial mode. In '2d' mode with ``utm_x``/``utm_y`` supplied, probes
        are spread across the UTM grid (mirrors ``run_pipeline.main()``'s
        own 2-D probe placement); otherwise probes are spread along the
        survey's own local x-range with y=0 (1-D transect).
    utm_x, utm_y : np.ndarray, optional
        2-D-mode UTM grid coordinates
        (``synthetic_utm_x.npy`` / ``synthetic_utm_y.npy``, written by Step
        0). Required for 2-D-mode probe placement; ignored in 1-D mode.
    probe_xyz_file : str, optional
        Path to an explicit probe-location CSV (``run_pipeline.py``'s
        ``--probe-xyz``). When the actual pipeline run is going to use an
        explicit probe file (rather than auto-placed ``num_swc_positions``
        x ``swc_depths``), this diagnostic MUST be told about it too --
        otherwise it fabricates a different, auto-placed probe layout
        (wrong count AND wrong locations) that has nothing to do with the
        probes the real inversion will use. When given, this loads the
        same file via ``run_pipeline.load_probe_xyz_file`` and
        re-localizes it into the survey's local frame using the same
        offset method ``run_pipeline.main()`` itself uses for synthetic
        runs (translate so the probe grid's own min x/y lands on the
        survey's min x/y, preserving relative spacing), overriding
        ``num_swc_positions``/``swc_depths`` entirely -- mirroring
        ``run_pipeline.main()``'s own precedence rules exactly.

    Returns
    -------
    dict with keys: 'figures' (list of PNG paths, possibly empty if the
    figure could not be produced), 'n_epochs', 'matching_summary' (None if
    it could not be computed), 'crs_consistency' (str message, or None if
    the check did not run or found no inconsistency).
    """
    import em_methodology as EMmethod
    import plot_probe_matching as ppm

    em_files = list(em_files)
    n_epochs = len(em_files)

    print(f"\n{'='*78}")
    print(" [PRE-INVERSION DIAG] Synthetic probe<->EM spatial-matching check")
    print(f"{'='*78}")
    print(f"  EM survey epoch(s)        : {n_epochs}")
    print(f"  Mode                      : {mode.upper()}")
    print(f"  window_m                  : {window_m}")

    # ------------------------------------------------------------------
    # Survey positions -- read directly from the first epoch's own CSV
    # (same file/row order the real inversion would use for epoch 0), no
    # inversion needed to get x/y.
    # ------------------------------------------------------------------
    first_fname = em_files[0]
    survey_df = EMmethod.load_survey_positions(first_fname, targetProjection=None, em_unit="ppt")
    if len(survey_df) == 0:
        print(f"[PRE-INVERSION DIAG][WARN] '{first_fname}' produced an empty position "
              f"table -- synthetic pre-inversion matching diagnostic skipped.")
        return {"figures": [], "n_epochs": n_epochs, "matching_summary": None,
                "crs_consistency": None}

    x_survey = survey_df["x"].to_numpy(dtype=float)
    y_survey = (survey_df["y"].to_numpy(dtype=float) if "y" in survey_df.columns
                else np.zeros_like(x_survey))
    num_points = len(survey_df)

    # ------------------------------------------------------------------
    # Probe placement -- mirrors run_pipeline.main()'s own logic exactly
    # (2-D: spread across utm_x/utm_y; 1-D: spread along the transect's
    # own local x-range), just computed pre-inversion against the raw
    # survey's own point count instead of models_clean's.
    # ------------------------------------------------------------------
    if probe_xyz_file is not None:
        # Mirror run_pipeline.main()'s own precedence: an explicit
        # --probe-xyz file overrides num_swc_positions/swc_depths entirely,
        # with no cartesian (position x depth) expansion -- the file's own
        # rows ARE the probes. Load it the same way run_pipeline.py does,
        # via a deferred import (avoids a module-level circular import,
        # since run_pipeline.py imports this module at load time; by the
        # time this function actually runs, run_pipeline is fully loaded).
        import run_pipeline as _rp
        _raw_px, _raw_py, probe_depths, _probe_labels = _rp.load_probe_xyz_file(probe_xyz_file)
        # The file's x/y are real-world absolute coordinates (e.g. actual
        # UTM), but the synthetic survey's own x/y (x_survey/y_survey) are
        # always a LOCAL grid near (0, 0) -- same mismatch run_pipeline.py
        # itself fixes for synthetic runs. Re-localize by translating so
        # the probe grid's own min x/y lands on the survey's min x/y,
        # preserving the file's internal relative layout exactly.
        _px = np.asarray(_raw_px, dtype=float)
        _py = np.asarray(_raw_py, dtype=float)
        if _px.size and len(x_survey):
            _dx = float(x_survey.min()) - float(_px.min())
            _dy = float(y_survey.min()) - float(_py.min())
            probe_x_coords = (_px + _dx).tolist()
            probe_y_coords = (_py + _dy).tolist()
        else:
            probe_x_coords = _raw_px
            probe_y_coords = _raw_py
        print(f"[PRE-INVERSION DIAG] Using {len(probe_x_coords)} explicit probe(s) "
              f"from --probe-xyz '{probe_xyz_file}' (re-localized to the synthetic "
              f"survey's local frame) instead of auto-placed num_swc_positions="
              f"{num_swc_positions} x swc_depths.")
    else:
        num_positions = min(num_swc_positions, num_points)
        tdr_depths = swc_depths if swc_depths is not None else list(layer_centers)

        # NOTE (fix): probe placement for synthetic scenarios must stay in the
        # SAME local coordinate frame as the survey CSV's own x/y (x_survey/
        # y_survey) -- never the absolute-UTM synthetic_utm_x/y grid (utm_x/
        # utm_y below), which is offset by the real-world survey origin
        # (~515000 mE / ~4468000 mN) and lives in a different frame entirely.
        # Mixing the two is exactly the CRS-mismatch bug this diagnostic used
        # to (correctly) flag. utm_x/utm_y are accepted for backward
        # compatibility but are intentionally UNUSED for placement now.
        if mode == "2d" and len(x_survey) > 0 and np.ptp(y_survey) > 0:
            step = max(1, num_points // num_positions)
            sample_indices = list(range(0, num_points, step))[:num_positions]
            unique_xy_coords = [(float(x_survey[i]), float(y_survey[i])) for i in sample_indices]
        else:
            unique_x = list(np.linspace(x_survey.min(), x_survey.max(), num_positions))
            unique_xy_coords = [(x, 0.0) for x in unique_x]

        probe_x_coords = [xy[0] for xy in unique_xy_coords for _ in tdr_depths]
        probe_y_coords = [xy[1] for xy in unique_xy_coords for _ in tdr_depths]
        probe_depths   = [d     for _  in unique_xy_coords for d in tdr_depths]

    # ------------------------------------------------------------------
    # CRS consistency sanity check -- same defense-in-depth check used on
    # the real-data path. In 2-D mode this compares the survey's own local
    # x/y against the ABSOLUTE-UTM probe coordinates built from utm_x/
    # utm_y above; a large span/centroid mismatch here is the signature of
    # exactly the CRS bug described in this function's docstring.
    # ------------------------------------------------------------------
    crs_message = None
    try:
        _probe_x_arr = np.asarray(probe_x_coords, dtype=float)
        _probe_y_arr = np.asarray(probe_y_coords, dtype=float)
        if _probe_x_arr.size and x_survey.size:
            crs_message = geometry_utils.check_crs_consistency(
                x_survey, y_survey, _probe_x_arr, _probe_y_arr,
                label_a=f"Synthetic EM survey ({os.path.basename(first_fname)})",
                label_b="Synthetic SWC/TDR probes",
            )
            if crs_message:
                print(f"\n[PRE-INVERSION DIAG][WARN] Possible CRS mismatch between "
                      f"synthetic EM survey and probe coordinates: {crs_message}")
            else:
                print(f"\n[PRE-INVERSION DIAG] CRS consistency check: synthetic EM "
                      f"survey and probe coordinates occupy consistent spatial "
                      f"extents. OK.")
    except Exception as _crs_exc:
        print(f"[PRE-INVERSION DIAG][WARN] CRS consistency check failed to run "
              f"({type(_crs_exc).__name__}: {_crs_exc}) -- skipping.")

    # ------------------------------------------------------------------
    # Probe <-> EM point matching figure -- only coordinates + window_m
    # are needed (a dummy all-ones EC array satisfies the shape
    # requirement), so this is safe to compute before Step 1.
    # ------------------------------------------------------------------
    dummy_models = np.ones((1, num_points, 1), dtype=float)
    matching_summary = None
    fig_path = ""
    try:
        position_results = EMmethod.extract_sample_positions_from_survey(
            survey_df=survey_df,
            models=dummy_models,
            probe_x_coords=probe_x_coords,
            probe_y_coords=probe_y_coords,
            probe_depths=probe_depths, depths0=depths0, window_m=window_m,
        )
        matching_summary = EMmethod.summarize_probe_matching(
            position_results, warn_fraction_threshold=matching_warn_fraction,
        )
        print(f"\n[TDR<->EM MATCHING] (synthetic, {num_points}-point survey epoch 0, "
              f"window_m={window_m})")
        print(f"  {matching_summary.message}")

        fig_path = ppm.plot_probe_matching(
            survey_df=survey_df,
            position_results=position_results,
            output_dir=output_dir,
            summary=matching_summary,
            title="Synthetic SWC/TDR probe -> EM point matching (pre-inversion check)",
            filename="probe_em_matching_pre_inversion_synthetic.png",
        )
    except Exception as _match_exc:
        print(f"[PRE-INVERSION DIAG][WARN] Synthetic probe<->EM matching figure "
              f"failed: {_match_exc}")

    print(f"\n[PRE-INVERSION DIAG] Done -- synthetic diagnostics written to "
          f"'{os.path.join(output_dir, 'diagnostics')}'\n")

    return {
        "figures": [fig_path] if fig_path else [],
        "n_epochs": n_epochs,
        "matching_summary": matching_summary,
        "crs_consistency": crs_message,
    }


def _plot_tolerance_sensitivity(epoch_xy, tolerance_sweep, current_tolerance,
                                 diag_dir) -> str:
    tolerances = sorted(set(tolerance_sweep) | {current_tolerance})
    n_ref_list = []
    mean_cov_list = []
    for tol in tolerances:
        rx, ry, matched = geometry_utils.match_epochs_to_reference_grid(
            epoch_xy, tolerance_m=tol,
        )
        n_ref_list.append(len(rx))
        cov = float(np.mean([np.mean(m >= 0) for m in matched])) if matched else 0.0
        mean_cov_list.append(cov)

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.plot(tolerances, n_ref_list, "o-", color="tab:blue", label="Reference grid size")
    ax1.set_xlabel("spatial_match_tolerance_m [m]")
    ax1.set_ylabel("Reference grid size (# union points)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(tolerances, [c * 100 for c in mean_cov_list], "s--", color="tab:orange",
             label="Mean per-epoch coverage")
    ax2.set_ylabel("Mean per-epoch coverage [%]", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.axvline(current_tolerance, color="grey", linestyle=":", linewidth=1.5,
                label=f"Current setting ({current_tolerance} m)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
    ax1.set_title("spatial_match_tolerance_m sensitivity sweep", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    out_path = os.path.join(diag_dir, "tolerance_sensitivity.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[PRE-INVERSION DIAG] Saved -> {out_path}")
    return out_path


# ===========================================================================
# STANDALONE CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse
    from scenario_loader import load_scenario

    parser = argparse.ArgumentParser(
        description="Pre-inversion real-data import/filter/spatial-matching diagnostics "
                    "for the EA-EM pipeline (runs before the expensive Step 1 ensemble "
                    "inversion)."
    )
    parser.add_argument("--scenario", default=None, metavar="NAME",
                        help="Real-data scenario name from ensemble_scenarios.json -- "
                             "uses its em_data_dir/tdr_data_dir/coord_file/"
                             "time_tolerance_days/spatial_match_tolerance_m/em_unit/"
                             "output_dir as defaults.")
    parser.add_argument("--config", default=None, metavar="PATH")
    parser.add_argument("--em-dir", default=None)
    parser.add_argument("--tdr-dir", default=None)
    parser.add_argument("--coord-file", default=None)
    parser.add_argument("--time-tolerance-days", type=float, default=None)
    parser.add_argument("--spatial-match-tolerance-m", type=float, default=None)
    parser.add_argument("--em-unit", default=None, choices=["ppm", "ppt"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tolerance-sweep", nargs="+", type=float, default=None,
                        metavar="M",
                        help="Extra tolerance values [m] to sweep, e.g. "
                             "--tolerance-sweep 0.5 1 2 3 5 10")
    parser.add_argument("--window-m", type=float, default=None, metavar="M",
                        help="Half-width [m] of the spatial averaging window used to "
                             "match each TDR station to EM measurement points (Figure 4). "
                             "Overrides the scenario's window_m field.")
    parser.add_argument("--matching-warn-fraction", type=float, default=None, metavar="F",
                        help="TDR<->EM matching warn threshold (fraction, 0-1). "
                             "Overrides the scenario's matching_warn_fraction field.")
    args = parser.parse_args()

    cfg = None
    if args.scenario:
        cfg = load_scenario(args.scenario, config_path=args.config)
        if getattr(cfg, "data_source", "synthetic") != "real":
            raise SystemExit(
                f"Scenario '{args.scenario}' is not a real-data scenario "
                f"(data_source={getattr(cfg, 'data_source', None)!r})."
            )

    em_dir = args.em_dir or (cfg.em_data_dir if cfg else None)
    tdr_dir = args.tdr_dir or (cfg.tdr_data_dir if cfg else None)
    coord_file = (args.coord_file or (cfg.coord_file if cfg else None)
                  or "./data/SOILNET_MFC2_coordinates_sensors.csv")
    if not em_dir or not tdr_dir:
        raise SystemExit(
            "Must supply --em-dir/--tdr-dir directly, or --scenario pointing to a "
            "real-data scenario that defines em_data_dir/tdr_data_dir."
        )

    run_pre_inversion_diagnostics(
        em_dir=em_dir,
        tdr_dir=tdr_dir,
        coord_file=coord_file,
        output_dir=args.output_dir or (cfg.output_dir if cfg else "mfc2_outputs"),
        time_tolerance_days=(
            args.time_tolerance_days if args.time_tolerance_days is not None
            else (cfg.time_tolerance_days if cfg else 1.0)
        ),
        spatial_match_tolerance_m=(
            args.spatial_match_tolerance_m if args.spatial_match_tolerance_m is not None
            else (cfg.spatial_match_tolerance_m if cfg else 2.0)
        ),
        em_unit=(args.em_unit or (cfg.em_unit if cfg else "ppt")),
        tolerance_sweep=args.tolerance_sweep,
        window_m=(
            args.window_m if args.window_m is not None
            else (cfg.window_m if cfg else 1.0)
        ),
        matching_warn_fraction=(
            args.matching_warn_fraction if args.matching_warn_fraction is not None
            else (cfg.matching_warn_fraction if cfg else 0.5)
        ),
    )
