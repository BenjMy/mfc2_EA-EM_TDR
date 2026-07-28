#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Pipeline Orchestrator — EA-EM Method (adapted from Loiseau et al., 2026)
================================================================================
Implements the five-step ensemble workflow in sequence with Parallel Inversions.

Spatial modes
-------------
--mode 1d  (default)
    Classic single-transect workflow.  Synthetic measurement points are placed
    along a straight line; no geographic coordinates are attached.

--mode 2d
    2-D grid workflow.  Measurement points are placed on a rectangular grid
    whose UTM origin and extent are derived from the Alento field survey
    (1_ALENTO_040820.csv).  Each synthetic survey CSV includes ``utm_x`` and
    ``utm_y`` columns.  The water-content visualisation step additionally
    produces XZ cross-section plots and XY plan-view maps.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import importlib
import pickle
import platform
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import em_methodology as EMmethod
import geometry_utils
from scenario_loader import load_scenario, ScenarioConfig
import plot_water_content as pwc
import plot_emagpy_diagnostics as pediag
import plot_em_diagnostics_real as pediag_real
import plot_tdr_diagnostics as ptdrdiag
import plot_solution_diagnostics as psol
import plot_ini_model as pim
import plot_probe_matching as ppm
import plot_pre_inversion_diagnostics as preinvdiag

step0   = importlib.import_module("0_generate_synthetic_data")
step2   = importlib.import_module("2_run_petro")
step3   = importlib.import_module("3_run_average")
step4_5 = importlib.import_module("4_run_select_assess")


# ===========================================================================
# PIPELINE REPORTER
# ===========================================================================

import sys
import datetime
import platform


class PipelineReporter:
    """
    Writes a Markdown run-report alongside normal stdout output.

    Every ``log()`` call echoes to both stdout and the report file.
    Dedicated helpers (``step``, ``args_table``, ``metric``, ``finish``)
    emit structured Markdown sections so the report is human-readable
    without post-processing.

    The report file is written to ``<output_dir>/pipeline_report_<timestamp>.md``.
    """

    def __init__(self, output_dir: str, scenario_name: str, mode: str, cfg,
                 data_source: str = "synthetic", em_dir: str = None, tdr_dir: str = None) -> None:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = os.path.join(output_dir, f"pipeline_report_{ts}.md")
        self._f = open(self.report_path, "w", encoding="utf-8")
        self._step_times: dict[str, datetime.datetime] = {}
        self._step_durations: dict[str, float] = {}
        self._metrics: list[tuple[str, str]] = []
        self._start = datetime.datetime.now()

        # Header
        self._write(f"# EA-EM Pipeline Run Report")
        self._write(f"")
        self._write(f"| Field | Value |")
        self._write(f"|-------|-------|")
        self._write(f"| Generated | {self._start.strftime('%Y-%m-%d %H:%M:%S')} |")
        self._write(f"| Data source | `{data_source.upper()}` |")
        if data_source == "real":
            self._write(f"| EM directory | `{em_dir}` |")
            self._write(f"| TDR directory | `{tdr_dir}` |")
        self._write(f"| Scenario  | `{scenario_name}` |")
        self._write(f"| Mode      | `{mode.upper()}` |")
        self._write(f"| Output dir| `{output_dir}` |")
        self._write(f"| Python    | {sys.version.split()[0]} on {platform.system()} {platform.machine()} |")
        self._write(f"| Host      | {platform.node()} |")
        self._write(f"")

        # Scenario config table
        if cfg is not None:
            self._write(f"## Scenario Configuration")
            self._write(f"")
            self._write(f"| Parameter | Value |")
            self._write(f"|-----------|-------|")
            for attr in ("name", "output_dir", "realization_index",
                         "depths0", "fixed_depths", "gamma",
                         "eca_noise_level", "tdr_noise_std"):
                val = getattr(cfg, attr, "—")
                self._write(f"| `{attr}` | `{val}` |")
            if mode == "2d":
                for attr in ("grid_num_x", "grid_num_y"):
                    val = getattr(cfg, attr, "—")
                    self._write(f"| `{attr}` | `{val}` |")
            if hasattr(cfg, "param_grid"):
                self._write(f"| `param_grid` | `{cfg.param_grid}` |")
            if hasattr(cfg, "cond_ranges"):
                self._write(f"| `cond_ranges` | `{cfg.cond_ranges}` |")
            self._write(f"")

        self._write(f"## Step Log")
        self._write(f"")
        print(f"[REPORTER] Report initialised → {self.report_path}")

    # ------------------------------------------------------------------
    def _write(self, line: str) -> None:
        self._f.write(line + "\n")
        self._f.flush()

    def log(self, msg: str) -> None:
        """Echo a message to stdout and append it to the report."""
        print(msg)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._write(f"`{ts}` {msg}")

    def step(self, name: str, detail: str = "") -> None:
        """Mark the start of a named pipeline step."""
        now = datetime.datetime.now()
        self._step_times[name] = now
        ts = now.strftime("%H:%M:%S")
        header = f"### {name}"
        if detail:
            header += f"  _{detail}_"
        self._write(f"")
        self._write(header)
        self._write(f"")
        self._write(f"_Started: {ts}_")
        self._write(f"")

    def end_step(self, name: str) -> None:
        """Record duration since the matching ``step()`` call."""
        if name in self._step_times:
            elapsed = (datetime.datetime.now() - self._step_times[name]).total_seconds()
            self._step_durations[name] = elapsed
            self._write(f"")
            self._write(f"_Completed in **{elapsed:.1f} s**_")

    def args_table(self, title: str, data: dict) -> None:
        """Write a key-value table under *title*."""
        self._write(f"")
        self._write(f"**{title}**")
        self._write(f"")
        self._write(f"| Key | Value |")
        self._write(f"|-----|-------|")
        for k, v in data.items():
            self._write(f"| `{k}` | `{v}` |")
        self._write(f"")

    def table(self, title: str, columns: list[str], rows: list[list], note: str = "") -> None:
        """
        Write a generic multi-column markdown table under *title*.

        Unlike ``args_table`` (fixed Key/Value layout), this renders an
        arbitrary number of named columns -- used e.g. for the per-probe
        SWC/TDR x/y/z position table, where a two-column key-value layout
        can't represent one row per probe with several fields each.
        """
        self._write(f"")
        self._write(f"**{title}**")
        if note:
            self._write(f"")
            self._write(f"_{note}_")
        self._write(f"")
        self._write(f"| " + " | ".join(columns) + " |")
        self._write(f"|" + "|".join(["-----"] * len(columns)) + "|")
        for row in rows:
            self._write(f"| " + " | ".join(str(v) for v in row) + " |")
        self._write(f"")

    def metric(self, name: str, value) -> None:
        """Record a scalar metric for the summary table."""
        self._metrics.append((name, str(value)))


    def failure(self, exc: "BaseException") -> None:
        """
        Record an unhandled exception into the report and close it.

        Writes a dedicated ``## ❌ Pipeline Failure`` section containing:
        * exception type and message
        * full Python traceback (each frame on its own line)
        * elapsed time up to the crash
        * last active step (if any)
        * partial step timings and metrics collected before the crash
        """
        import traceback as _tb

        elapsed = (datetime.datetime.now() - self._start).total_seconds()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self._write(f"")
        self._write(f"## ❌ Pipeline Failure")
        self._write(f"")
        self._write(f"| Field | Value |")
        self._write(f"|-------|-------|")
        self._write(f"| Timestamp | `{now_str}` |")
        self._write(f"| Elapsed   | `{elapsed:.1f} s` |")
        self._write(f"| Exception | `{type(exc).__name__}` |")

        _active = [s for s in self._step_times if s not in self._step_durations]
        if _active:
            self._write(f"| Failed in step | `{_active[-1]}` |")

        self._write(f"")
        self._write(f"### Error message")
        self._write(f"")
        self._write(f"```")
        self._write(str(exc))
        self._write(f"```")
        self._write(f"")
        self._write(f"### Full traceback")
        self._write(f"")
        self._write(f"```python")
        for line in _tb.format_exception(type(exc), exc, exc.__traceback__):
            for sub in line.rstrip("\n").split("\n"):
                self._write(sub)
        self._write(f"```")
        self._write(f"")

        if self._step_durations:
            self._write(f"### Completed step timings (before crash)")
            self._write(f"")
            self._write(f"| Step | Duration (s) |")
            self._write(f"|------|-------------|")
            for step_name, dur in self._step_durations.items():
                self._write(f"| {step_name} | {dur:.1f} |")
            self._write(f"| **Total so far** | **{elapsed:.1f}** |")
            self._write(f"")

        if self._metrics:
            self._write(f"### Metrics collected before crash")
            self._write(f"")
            self._write(f"| Metric | Value |")
            self._write(f"|--------|-------|")
            for name, val in self._metrics:
                self._write(f"| {name} | {val} |")
            self._write(f"")

        self._f.close()
        print(f"[REPORTER] ❌ Failure report written → {self.report_path}")

    def finish(self, exit_ok: bool = True, quality_level: str = "OK",
               quality_reasons: list[str] | None = None) -> None:
        """
        Write timing summary and close the file.

        Parameters
        ----------
        exit_ok : bool
            False only when the pipeline raised an unhandled exception
            (handled separately by ``failure()``, but kept here for
            backward-compat callers).
        quality_level : str
            "OK" (default) or "WARN" — the verdict from
            ``EMmethod.evaluate_quality_gates()`` (report item 4:
            "Uncertainty & quality as gates, not just numbers"). A WARN
            level means the pipeline completed successfully but the final
            solution's CV/calibration coverage failed one or more quality
            thresholds — this is surfaced as a distinct status from plain
            SUCCESS so it isn't silently indistinguishable from a clean run.
        quality_reasons : list[str], optional
            Human-readable reasons the quality gate fired, printed in the
            report when quality_level == "WARN".
        """
        elapsed_total = (datetime.datetime.now() - self._start).total_seconds()
        self._write(f"")
        self._write(f"## Metrics Summary")
        self._write(f"")
        self._write(f"| Metric | Value |")
        self._write(f"|--------|-------|")
        for name, val in self._metrics:
            self._write(f"| {name} | {val} |")
        self._write(f"")
        self._write(f"## Step Timing")
        self._write(f"")
        self._write(f"| Step | Duration (s) |")
        self._write(f"|------|-------------|")
        for step_name, dur in self._step_durations.items():
            self._write(f"| {step_name} | {dur:.1f} |")
        self._write(f"| **Total** | **{elapsed_total:.1f}** |")
        self._write(f"")

        if not exit_ok:
            status = "❌ FAILED"
        elif quality_level == "WARN":
            status = "⚠️ SUCCESS WITH QUALITY-GATE WARNINGS"
        else:
            status = "✅ SUCCESS"

        self._write(f"## Result")
        self._write(f"")
        self._write(f"**{status}** — pipeline finished at "
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if exit_ok and quality_level == "WARN" and quality_reasons:
            self._write(f"")
            self._write(f"**Quality gate reasons:**")
            self._write(f"")
            for reason in quality_reasons:
                self._write(f"- {reason}")
        self._write(f"")
        self._f.close()
        print(f"[REPORTER] Report written → {self.report_path}  (status: {status})")


# ===========================================================================
# HELPERS & PARALLEL WORKERS
# ===========================================================================

def _prepare_real_em_csv(raw_path: str, processed_dir: str, survey_date=None
                         ) -> tuple[str, str, "str | None"]:
    """
    Convert a raw GEM-2 survey CSV into the comma-separated, emagpy-column-named
    format that ``createSurvey`` requires.

    Handles two known GEM-2 export variants:
      Variant A – semicolon-delimited, columns: X, Y, GPSalt, Q_<freq>Hz
      Variant B – comma-delimited, columns: Longitude, Latitude, Altitude,
                  Q<freq> or Q_<freq> (with or without 'Hz' suffix)

    The function auto-detects the delimiter and tries each known column-name
    pattern in turn, so new file vintages only require adding an entry to
    ``_COLUMN_VARIANTS`` rather than touching the call sites.

    FIX (TDR<->EM CRS mismatch): Variant A's ``X``/``Y`` (or ``Easting``/
    ``Northing``) are already metric (UTM-like) coordinates, but Variant B's
    ``Longitude``/``Latitude`` are decimal-degree WGS84 -- and until this fix,
    BOTH were renamed to the same generic ``x``/``y`` columns with no CRS
    metadata, then handed to ``createSurvey(..., targetProjection=None)``
    unchanged. Since TDR probe coordinates (``probe_table['utm_x'/'utm_y']``)
    are always metric UTM, a Variant-B file left every distance computed in
    ``em_methodology._probe_distances`` off by ~5 orders of magnitude
    (degrees vs metres), silently forcing 100% of TDR probes into
    nearest-point ``window_m`` fallback matching against a geographically
    meaningless "nearest" point. This function now classifies which alias
    actually matched (``geometry_utils.classify_coordinate_alias``) and, if
    it was the geographic one, reprojects lon/lat to metric UTM
    (``geometry_utils.reproject_lonlat_to_utm``, auto-picking the UTM zone
    from this file's own mean coordinates) BEFORE writing the processed CSV
    -- so every downstream consumer (filterRepeated, Step 1 inversion
    workers, TDR<->EM window matching, plotting) reads already-metric x/y
    and needs no further CRS awareness at all.

    Parameters
    ----------
    survey_date : datetime.datetime, optional
        The date of this EM survey (as already extracted from the raw
        filename upstream by ``0_real_data_loader.extract_date_from_filename``).
        If given, every row of the processed CSV gets a constant ``date``
        column set to this value. emagpy's ``Survey.computeStat()`` (called
        internally by ``filterRepeated``) unconditionally looks for a
        'time'/'date' column and otherwise raises
        ``UnboundLocalError: local variable 'times' referenced before
        assignment`` — every processed survey previously had no such
        column at all, so this crashed unconditionally. A constant
        per-survey date is sufficient: ``computeStat`` only uses it to
        compute elapsed time between consecutive points, which isn't used
        by ``filterRepeated``'s actual filtering logic (interdist only).

    Returns
    -------
    out_path : str
        Path to the processed file written under *processed_dir*. Its
        'x'/'y' columns are ALWAYS metric after this function returns,
        regardless of which alias the raw file used.
    coord_kind : {'metric', 'geographic'}
        Which alias category the raw file's x/y columns matched. Purely
        informational for the caller's logging/report -- the processed CSV
        itself is metric either way.
    epsg_used : str or None
        The EPSG code the geographic->UTM reprojection used, or ``None``
        if the raw file was already metric (no reprojection needed).
    """
    import pandas as pd
    import geometry_utils as _geo

    # Canonical emagpy target column names mapped to lists of raw-CSV aliases
    # tried in order.  Add new aliases here when new file vintages appear.
    _COLUMN_VARIANTS: dict[str, list[str]] = {
        'x':                    ['X', 'Longitude', 'lon', 'LON', 'LONGITUDE', 'Easting'],
        'y':                    ['Y', 'Latitude',  'lat', 'LAT', 'LATITUDE',  'Northing'],
        'elevation':            ['GPSalt', 'Altitude', 'Alt', 'ALT', 'altitude', 'ALTITUDE', 'Height'],
        'HCP1.66f425h1_quad':   ['Q_425Hz',   'Q425Hz',   'Q_425',   'Q425'],
        'HCP1.66f1525h1_quad':  ['Q_1525Hz',  'Q1525Hz',  'Q_1525',  'Q1525'],
        'HCP1.66f5325h1_quad':  ['Q_5325Hz',  'Q5325Hz',  'Q_5325',  'Q5325'],
        'HCP1.66f18325h1_quad': ['Q_18325Hz', 'Q18325Hz', 'Q_18325', 'Q18325'],
        'HCP1.66f63025h1_quad': ['Q_63025Hz', 'Q63025Hz', 'Q_63025', 'Q63025'],
        'HCP1.66f92775h1_quad': ['Q_92775Hz', 'Q92775Hz', 'Q_92775', 'Q92775'],
    }

    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "processed_" + os.path.basename(raw_path))

    # --- 1. Auto-detect delimiter by sniffing the first non-empty line --------
    with open(raw_path, 'r', errors='replace') as fh:
        first_line = ''
        for line in fh:
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break
    delimiter = ';' if first_line.count(';') >= first_line.count(',') else ','

    df = pd.read_csv(raw_path, delimiter=delimiter)
    df.columns = df.columns.str.strip()
    available = set(df.columns)

    # --- 2. Resolve each logical column to whichever alias is present ---------
    rename_map: dict[str, str] = {}   # raw_name -> emagpy_name
    unresolved: list[str] = []

    for emagpy_col, aliases in _COLUMN_VARIANTS.items():
        matched = next((a for a in aliases if a in available), None)
        if matched:
            rename_map[matched] = emagpy_col
        else:
            unresolved.append(emagpy_col)

    if unresolved:
        raise ValueError(
            f"[REAL DATA] Cannot preprocess '{os.path.basename(raw_path)}': "
            f"no recognised alias for emagpy columns {unresolved} "
            f"(delimiter='{delimiter}', found columns: {sorted(available)}).  "
            f"Add the raw column name(s) to _COLUMN_VARIANTS in _prepare_real_em_csv()."
        )

    # --- 3. Classify which alias matched x/y, and reproject if geographic -----
    x_raw = next(raw for raw, canon in rename_map.items() if canon == 'x')
    y_raw = next(raw for raw, canon in rename_map.items() if canon == 'y')
    x_kind = _geo.classify_coordinate_alias(x_raw, 'x')
    y_kind = _geo.classify_coordinate_alias(y_raw, 'y')

    if x_kind != y_kind or x_kind == "unknown":
        raise ValueError(
            f"[REAL DATA] Cannot preprocess '{os.path.basename(raw_path)}': "
            f"x resolved via column '{x_raw}' (classified '{x_kind}') but y "
            f"resolved via column '{y_raw}' (classified '{y_kind}') -- refusing "
            f"to guess a coordinate system from an inconsistent/unrecognised "
            f"pairing. Both must be either metric (X/Easting, Y/Northing) or "
            f"geographic (Longitude/lon, Latitude/lat)."
        )
    coord_kind = x_kind  # == y_kind, both "metric" or "geographic"

    out_df = df[list(rename_map.keys())].rename(columns=rename_map)

    epsg_used: "str | None" = None
    if coord_kind == "geographic":
        x_m, y_m, epsg_used = _geo.reproject_lonlat_to_utm(
            out_df['x'].to_numpy(dtype=float), out_df['y'].to_numpy(dtype=float),
        )
        print(
            f"[REAL DATA] '{os.path.basename(raw_path)}': x/y resolved via "
            f"geographic alias ('{x_raw}'/'{y_raw}', decimal degrees) -- "
            f"reprojected to metric UTM ({epsg_used}, auto-detected from this "
            f"file's own mean coordinates) before writing the processed CSV."
        )
        out_df['x'] = x_m
        out_df['y'] = y_m
    # else: already metric (X/Easting, Y/Northing) -- no reprojection needed.

    if survey_date is not None:
        out_df['date'] = pd.Timestamp(survey_date).strftime('%Y-%m-%d')
    out_df.to_csv(out_path, index=False)
    return out_path, coord_kind, epsg_used




def _check_em_unit_sanity(csv_path: str, em_unit: str,
                          expected_range_ms_m: tuple[float, float] = (1.0, 300.0)) -> dict:
    """
    Defense-in-depth sanity check for the em_unit configuration (see fix
    "em_unit defaults to 'ppt' but real GEM-2 data is 'ppm'" -- item 4,
    "Non-goals": this deliberately does NOT try to auto-detect the correct
    unit from column names or magnitude and silently correct it; it only
    warns, loudly, so a human decides).

    After ``Problem.createSurvey(..., unit=em_unit)`` converts the raw
    quadrature/in-phase columns to LIN ECa [mS/m], this logs the median
    |ECa| across all coils and flags it if it lands more than three orders
    of magnitude outside the range typical of real field soils surveyed
    with a GEM-2 (~1-300 mS/m). Getting ``em_unit`` backwards (``'ppt'``
    instead of ``'ppm'`` or vice versa) changes every ECa value by exactly
    1000x, which is precisely the kind of error this catches -- and, per
    the "Non-goals" note, this generalises to catching a *different*
    instrument/unit mismatch later too, not just this one historical bug.

    Never raises -- failures here are logged as [WARN] and the caller
    proceeds; this is a diagnostic, not a hard gate (the hard gate for
    *missing* em_unit on real-data scenarios is
    ``scenario_loader._validate``).

    Parameters
    ----------
    csv_path : str
        Path to an already emagpy-ready (comma-separated, canonical coil
        column names) EM survey CSV -- i.e. the output of
        ``_prepare_real_em_csv`` / ``filter_repeated_em_points``.
    em_unit : str
        The 'ppm' / 'ppt' value actually being used for this survey.
    expected_range_ms_m : (float, float), optional
        Expected (min, max) |ECa| [mS/m] for a plausible field survey.
        Default (1.0, 300.0), matching GEM-2 field-soil experience
        elsewhere in this codebase (see the '2d_real' scenario's
        cond_ranges, comparably scoped).

    Returns
    -------
    dict
        ``{"file": <basename>, "level": "OK"|"INFO"|"WARN"|"ERROR",
        "median_abs_eca": float|None, "message": str}`` -- structured so a
        caller with access to a ``PipelineReporter`` can fold this into the
        Markdown run report (``reporter.log(result["message"])``) rather
        than this check only ever being visible in stdout, which is what
        happened before this fix (see fix "Is this step in the log
        recorder?").
    """
    fname = os.path.basename(csv_path)
    try:
        from emagpy import Problem
        k = Problem()
        k.createSurvey(csv_path, unit=em_unit)
        df = k.surveys[0].df
        coils = list(getattr(k.surveys[0], "coils", []))
        eca_cols = [c for c in coils if c in df.columns]
        if not eca_cols:
            msg = f"[REAL DATA] '{fname}': no LIN ECa columns found -- em_unit sanity check skipped."
            print(msg)
            return {"file": fname, "level": "OK", "median_abs_eca": None, "message": msg}
        vals = df[eca_cols].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            msg = f"[REAL DATA] '{fname}': no finite LIN ECa values -- em_unit sanity check skipped."
            print(msg)
            return {"file": fname, "level": "OK", "median_abs_eca": None, "message": msg}

        median_abs = float(np.median(np.abs(vals)))
        lo, hi = expected_range_ms_m

        if median_abs < lo / 1000.0 or median_abs > hi * 1000.0:
            level = "WARN"
            msg = (
                f"[REAL DATA][WARN] '{fname}': median |LIN ECa| = {median_abs:.4g} mS/m "
                f"is >1000x outside the typical field-soil range ({lo:g}-{hi:g} mS/m) "
                f"for em_unit='{em_unit}'. This is the signature of a wrong em_unit "
                f"(GEM-2 quadrature exports are 'ppm'; GF Instruments CMD exports are "
                f"'ppt' -- mixing them up scales every ECa value by ~1000x). Double-check "
                f"the 'em_unit' field for this scenario before trusting this run's results."
            )
        elif median_abs < lo or median_abs > hi:
            level = "INFO"
            msg = (
                f"[REAL DATA][INFO] '{fname}': median |LIN ECa| = {median_abs:.4g} mS/m "
                f"is outside the typical {lo:g}-{hi:g} mS/m field-soil range for "
                f"em_unit='{em_unit}', but not by the ~1000x factor a unit mismatch would "
                f"produce -- likely just unusual field conditions rather than a config "
                f"error. No action needed unless the inverted EC also looks implausible."
            )
        else:
            level = "OK"
            msg = (f"[REAL DATA] '{fname}': median |LIN ECa| = {median_abs:.4g} mS/m "
                   f"(em_unit='{em_unit}') -- within the expected field-soil range.")

        print(msg)
        return {"file": fname, "level": level, "median_abs_eca": median_abs, "message": msg}
    except Exception as e:
        msg = (f"[REAL DATA][WARN] em_unit sanity check failed for "
               f"'{fname}' ({type(e).__name__}: {e}) -- skipping.")
        print(msg)
        return {"file": fname, "level": "WARN", "median_abs_eca": None, "message": msg}


def _extract_temperature_by_epoch(
    load_tdr_dataframe_fn, tdr_dir: str, coord_file: str,
    em_dates: list, time_tolerance_days: float,
) -> "list[float | None]":
    """
    Best-effort extraction of measured soil temperature per EM survey
    epoch, for the temperature correction fix ("temperature correction is
    currently a complete no-op" -- run_pipeline.py previously hardcoded
    TEMP_PROFILES = TEMP_REF, so the correction factor
    ``1 + alpha*(T - T_ref)`` was always exactly 1, regardless of scenario
    or data source).

    Mirrors the same ``time_tolerance_days`` date-matching window
    ``0_real_data_loader.prepare_real_dataset`` uses for SWC, but for
    whichever column(s) look like a temperature reading (column name
    containing ``'temp'``, case-insensitive -- the same simple substring
    convention that module already uses for SWC column detection,
    ``'swc' in c.lower()``, in ``load_tdr_dataframe``).

    Deliberately reuses ``load_tdr_dataframe`` (which already returns every
    raw column from the TDR Excel sheets, including any temperature column,
    since it's a plain inner-merge with coordinates -- not just the SWC
    columns ``prepare_real_dataset`` explicitly extracts) rather than
    modifying ``prepare_real_dataset``'s own return contract, which other
    callers rely on unchanged.

    Parameters
    ----------
    load_tdr_dataframe_fn : callable
        The ``load_tdr_dataframe`` function from ``0_real_data_loader``,
        already resolved by the caller (``_load_real_data``) via whichever
        import path succeeded there.
    tdr_dir, coord_file : str
        Same paths passed to ``prepare_real_dataset``.
    em_dates : list of datetime.datetime
        One per EM survey epoch, same order as ``em_files``.
    time_tolerance_days : float
        Same matching half-window used for SWC<->EM date matching.

    Returns
    -------
    list of float or None, same length/order as ``em_dates``. ``None`` for
    any epoch with no temperature column found at all, or no matched
    reading within tolerance for that specific epoch -- callers MUST treat
    ``None`` as "no correction possible for this epoch", never as "0
    degrees" (which would silently apply a large, wrong correction).
    """
    try:
        df_probes, _ = load_tdr_dataframe_fn(tdr_dir, coord_file)
    except Exception as e:
        print(f"[REAL DATA][WARN] Could not reload TDR data for temperature "
              f"extraction ({type(e).__name__}: {e}) -- temperature correction "
              f"will be skipped for this run.")
        return [None] * len(em_dates)

    temp_cols = [c for c in df_probes.columns if "temp" in c.lower()]
    if not temp_cols:
        print("[REAL DATA] No temperature column found in TDR data (looked "
              "for any column containing 'temp') -- temperature correction "
              "will be skipped for this run unless 'soil_temperature_c' is "
              "set explicitly in the scenario config.")
        return [None] * len(em_dates)

    print(f"[REAL DATA] Temperature column(s) detected for correction: {temp_cols}")
    tolerance = datetime.timedelta(days=time_tolerance_days)
    temps: "list[float | None]" = []
    for em_dt in em_dates:
        window = df_probes[
            (df_probes["Timestamp"] >= em_dt - tolerance) &
            (df_probes["Timestamp"] <= em_dt + tolerance)
        ]
        if window.empty:
            temps.append(None)
            continue
        vals = window[temp_cols].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        temps.append(float(np.mean(vals)) if vals.size else None)

    n_found = sum(1 for t in temps if t is not None)
    print(f"[REAL DATA] Measured temperature matched for {n_found}/{len(em_dates)} "
          f"epoch(s) (tolerance ±{time_tolerance_days} day(s)).")
    return temps


def _load_real_data(em_dir: str, tdr_dir: str, coord_file: str,
                    time_tolerance_days: float = 1.0,
                    em_unit: str = "ppt",
                    filter_repeated_tolerance: float = 1e-10):
    """
    Load real EM and TDR field data.

    Preprocessing step added: raw GEM-2 CSV files use semicolon delimiters and
    instrument-specific column names that emagpy cannot parse.  Each file is
    converted to a comma-separated, emagpy-compatible CSV before being returned
    as part of ``em_files``.  The processed files are written to a sub-directory
    ``<em_dir>/_emagpy_ready/`` so the originals are never modified.

    Each processed CSV also has consecutive, near-duplicate spatial points
    (e.g. logger left stationary, overlapping GPS fixes) removed in place via
    ``filter_repeated_em_points`` — using emagpy's own
    ``Problem.filterRepeated`` — so this happens exactly once per survey
    here at load time, rather than redundantly inside every parallel
    inversion worker.

    Returns:
        tuple: (em_files, observations, probe_table, em_dates, raw_em_files,
                real_data_diagnostics)
               where em_files contains the PROCESSED paths ready for emagpy
               (always metric x/y, regardless of which raw column alias the
               instrument export used -- see the CRS-mismatch fix on
               _prepare_real_em_csv), observations is a list of dicts with
               keys: probe_index, time_index, depth, swc, em_dates is the
               list of datetime objects (one per EM survey epoch, same
               order/length as em_files), raw_em_files contains the
               original, untouched GEM-2 CSV paths (pre-processing, pre-
               filterRepeated) in the same order — kept around solely so
               diagnostic plotting (Step 1b) can show the full per-epoch
               footprint before the row-count trim, instead of the already
               column-renamed and duplicate-filtered processed files —
               and real_data_diagnostics is a dict with keys
               'coord_kinds' (list of per-file {'file','coord_kind',
               'epsg_used'} dicts), 'em_unit_checks' (list of per-file
               _check_em_unit_sanity() result dicts), 'crs_consistency'
               (single {'level','message'} verdict vs. probe_table),
               'n_geographic_reprojected' (int), and 'temp_by_epoch'
               (list[float|None], one measured soil temperature per
               em_dates entry, or None where no 'temp'-named TDR column /
               no matched reading was found -- feeds the temperature
               correction fix, see main()'s TEMPERATURE CORRECTION
               section) -- returned (rather than
               only printed) specifically so the caller can fold these
               diagnostics into the PipelineReporter's Markdown report;
               see fix "Is this step in the log recorder?".
    """
    from pathlib import Path

    # Import the real data loader
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from l0_real_data_loader import prepare_real_dataset, filter_repeated_em_points, load_tdr_dataframe  # Adjust import as needed
    except ImportError:
        # Try alternate name
        try:
            from _0_real_data_loader import prepare_real_dataset, filter_repeated_em_points, load_tdr_dataframe
        except ImportError:
            # Fallback: try from current script directory
            import importlib.util
            loader_script = Path(__file__).parent / "0_real_data_loader.py"
            if not loader_script.exists():
                raise ImportError(
                    f"Real data loader not found. Expected: {loader_script}\n"
                    "Ensure 0_real_data_loader.py is in the pipeline directory."
                )
            spec = importlib.util.spec_from_file_location("real_data_loader", loader_script)
            real_data_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(real_data_module)
            prepare_real_dataset = real_data_module.prepare_real_dataset
            filter_repeated_em_points = real_data_module.filter_repeated_em_points
            load_tdr_dataframe = real_data_module.load_tdr_dataframe

    print(f"[REAL DATA] Loading from:")
    print(f"  EM directory  : {em_dir}")
    print(f"  TDR directory : {tdr_dir}")
    print(f"  Coordinates   : {coord_file}")

    raw_em_files, em_dates, probe_table, observations = prepare_real_dataset(
        em_dir=em_dir,
        tdr_dir=tdr_dir,
        coord_file=coord_file,
        time_tolerance_days=time_tolerance_days,
    )

    # Convert each raw semicolon-delimited GEM-2 CSV into an emagpy-ready
    # comma-separated file with correctly named coil columns. As of the
    # CRS-mismatch fix, _prepare_real_em_csv also reprojects geographic
    # (Longitude/Latitude) files to metric UTM in-place, so every returned
    # em_files path is guaranteed metric regardless of which raw column
    # alias the instrument export used.
    processed_dir = os.path.join(em_dir, "_emagpy_ready")
    print(f"[REAL DATA] Pre-processing {len(raw_em_files)} EM files → {processed_dir}")
    em_files = []
    coord_kind_records: list[dict] = []
    em_unit_check_records: list[dict] = []
    for raw_path, survey_date in zip(raw_em_files, em_dates):
        try:
            proc_path, coord_kind, epsg_used = _prepare_real_em_csv(
                raw_path, processed_dir, survey_date=survey_date,
            )
            filter_repeated_em_points(proc_path, tolerance=filter_repeated_tolerance,
                                       unit=em_unit)
            em_unit_check_records.append(_check_em_unit_sanity(proc_path, em_unit))
            em_files.append(proc_path)
            coord_kind_records.append({
                "file": os.path.basename(raw_path), "coord_kind": coord_kind,
                "epsg_used": epsg_used,
            })
            epsg_note = f", reprojected via {epsg_used}" if epsg_used else ""
            print(f"  [OK] {os.path.basename(raw_path)} → {os.path.basename(proc_path)} "
                  f"(coords: {coord_kind}{epsg_note})")
        except Exception as e:
            print(f"  [WARN] Skipping '{os.path.basename(raw_path)}': {e}")

    if not em_files:
        raise RuntimeError(
            "[REAL DATA] No EM files could be pre-processed. "
            "Check that the raw CSVs match the expected GEM-2 column layout."
        )

    n_geo = sum(1 for r in coord_kind_records if r["coord_kind"] == "geographic")
    if n_geo:
        print(f"[REAL DATA] {n_geo}/{len(coord_kind_records)} EM survey file(s) used "
              f"geographic (Longitude/Latitude) columns and were reprojected "
              f"to metric UTM -- see per-file '(coords: ...)' lines above.")

    # ------------------------------------------------------------------
    # CRS CONSISTENCY SANITY CHECK (item 3 of the fix, defense-in-depth)
    #   Independent of the alias-based classification above: compare the
    #   final (now-metric, by construction) EM survey coordinates against
    #   probe_table's own utm_x/utm_y. Catches ANY residual CRS mismatch
    #   -- e.g. a raw file whose alias matched but whose actual values are
    #   in a different projection than probe_table's coordinates, or a
    #   pyproj/UTM-zone auto-detection edge case -- rather than relying
    #   solely on the classification fix above being sufficient.
    # ------------------------------------------------------------------
    crs_consistency = {"level": "OK", "message": None}
    try:
        import pandas as _pd
        _first_ok_path = em_files[0]
        _first_df = _pd.read_csv(_first_ok_path)
        _em_x = _first_df["x"].to_numpy(dtype=float)
        _em_y = (_first_df["y"].to_numpy(dtype=float)
                if "y" in _first_df.columns else np.zeros_like(_em_x))
        _px = probe_table["utm_x"].to_numpy(dtype=float) if "utm_x" in probe_table.columns else np.array([])
        _py = probe_table["utm_y"].to_numpy(dtype=float) if "utm_y" in probe_table.columns else np.array([])
        _valid_probes = (_px != 0.0) | (_py != 0.0)
        if _px.size and _valid_probes.any():
            _crs_msg = geometry_utils.check_crs_consistency(
                _em_x, _em_y, _px[_valid_probes], _py[_valid_probes],
                label_a=f"EM survey ({os.path.basename(_first_ok_path)})",
                label_b="TDR probes (probe_table utm_x/utm_y)",
            )
            if _crs_msg:
                crs_consistency = {
                    "level": "WARN",
                    "message": (f"Possible CRS mismatch between EM survey and TDR probe "
                                f"coordinates: {_crs_msg}"),
                }
                print(f"[REAL DATA][WARN] {crs_consistency['message']}")
            else:
                crs_consistency = {
                    "level": "OK",
                    "message": ("EM survey and TDR probe coordinates occupy consistent "
                                "spatial extents."),
                }
                print(f"[REAL DATA] CRS consistency check: {crs_consistency['message']} OK.")
        else:
            crs_consistency = {"level": "SKIPPED",
                               "message": "No valid probe_table utm_x/utm_y to compare against."}
    except Exception as _crs_exc:
        crs_consistency = {
            "level": "WARN",
            "message": f"CRS consistency check failed to run: {type(_crs_exc).__name__}: {_crs_exc}",
        }
        print(f"[REAL DATA][WARN] {crs_consistency['message']} -- skipping.")

    print(f"[REAL DATA] ✓ Loaded {len(em_files)} EM surveys (processed)")
    print(f"[REAL DATA] ✓ Loaded {len(probe_table)} probe locations")
    print(f"[REAL DATA] ✓ Loaded {len(observations)} matched observations")

    # ------------------------------------------------------------------
    # TEMPERATURE EXTRACTION (fix "temperature correction is currently a
    # complete no-op"). Best-effort: never raises, returns all-None on any
    # failure so the caller (main()) simply skips the correction and logs
    # that explicitly, rather than the run failing over an optional signal.
    # ------------------------------------------------------------------
    temp_by_epoch = _extract_temperature_by_epoch(
        load_tdr_dataframe, tdr_dir, coord_file, em_dates, time_tolerance_days,
    )

    real_data_diagnostics = {
        "coord_kinds": coord_kind_records,      # one dict per successfully-loaded EM file
        "em_unit_checks": em_unit_check_records,  # one dict per successfully-loaded EM file
        "crs_consistency": crs_consistency,       # single verdict vs. probe_table
        "n_geographic_reprojected": n_geo,
        "temp_by_epoch": temp_by_epoch,           # list[float|None], one per em_dates entry
    }

    return em_files, observations, probe_table, em_dates, raw_em_files, real_data_diagnostics


def _config_fingerprint(depths0, fixed_depths, param_grid, cond_ranges, gamma,
                          drop_coils=None, spatial_match_tolerance_m=None) -> str:
    """
    Stable content hash of everything that determines the inversion outputs
    cached in Step 1 (ensemble_cache_*.pkl). Used to auto-invalidate a cache
    whenever depths0 / fixed_depths / param_grid / cond_ranges / gamma /
    drop_coils / spatial_match_tolerance_m drift from what produced it —
    even if a human forgot to bump cache_tag. Relying on cache_tag alone is
    fragile: nothing previously verified that the cached
    depths_clean/models_clean actually matched the live scenario config, so
    an edited depths0 with a stale cache_tag would silently reuse ensemble
    models built from the old depths0.

    ``spatial_match_tolerance_m`` only matters for real-data time-lapse runs
    (see ``geometry_utils.match_epochs_to_reference_grid`` /
    ``main()``'s Step 1); pass ``None`` for synthetic or single-survey real
    runs so the fingerprint is unaffected by a field that plays no role in
    how those caches were built.
    """
    payload = {
        "depths0": list(depths0),
        "fixed_depths": fixed_depths if isinstance(fixed_depths, bool) else list(fixed_depths),
        "param_grid": param_grid,
        "cond_ranges": [list(r) for r in cond_ranges],
        "gamma": gamma,
        "drop_coils": drop_coils,
        "spatial_match_tolerance_m": spatial_match_tolerance_m,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _is_timelapse(fnames) -> bool:
    return not isinstance(fnames, str)


# _build_inversion_bnds is now the single shared implementation in
# geometry_utils.py (pipeline_improvement_report.md, section 6 / action
# item 3) -- kept as a thin alias here so every existing call site below
# (`_build_inversion_bnds(...)`) keeps working unchanged.
_build_inversion_bnds = geometry_utils.build_inversion_bnds


def _invert_single_worker(args) -> tuple[int, np.ndarray | None, np.ndarray | None, np.ndarray | None, str | None]:
    """
    Worker function executed in an isolated process space.
    Returns (task_idx, models_array, depths_array, pstd_array, error_message).

    ``error_message`` is ``None`` on success, otherwise a one-line
    "ExceptionType: message" string sent back to the main process for
    aggregation/reporting. The full traceback is printed locally (worker
    stdout) for live debugging but not shipped across the process
    boundary, to keep IPC payloads small.

    ``invert_kwargs`` may contain a ``"drop_coils"`` entry (coil name,
    frequency, or substring spec — see
    ``EMmethod.drop_coils_from_problem``) used to exclude a noisy
    frequency channel from the survey before inversion. It is stripped
    out before being forwarded to ``k.invert()``.

    ``pstd_array`` (FIX "use the full MCMC, not just a single selected
    model"): emagpy's MCMC-family inversion methods (MCMC/ROPE/DREAM/SCEUA)
    run a real posterior sampler per measurement point, but ``k.invert()``
    only ever keeps the lowest-misfit sample (-> ``k.models[0]``/
    ``k.depths[0]``, the only thing this worker used to return) plus a
    per-point, per-free-parameter posterior standard deviation in
    ``k.pstds[0]`` -- computed from the best 10% of the sampled chain. That
    array was previously read by nobody in this pipeline. It's captured
    here and returned alongside models/depths so Step 1 can propagate it
    through the same caching/packing/spatial-remap machinery already used
    for ``depths_clean``. For non-MCMC-family methods emagpy itself
    zero-initialises ``k.pstds`` and never overwrites it, so this is a
    harmless all-zero array in that case -- no method-branching needed here.
    """
    task_idx, fname, depths0, conds0, fixed_depths_list, invert_kwargs = args
    from emagpy import Problem
    import traceback

    # ------------------------------------------------------------------
    # DEFENSIVE GUARD: without an explicit `bnds`, EMagPy's own invert()
    # silently falls back to a HARDCODED conductivity ceiling of 100 mS/m
    # whenever any depth is variable (see Problem.invert(), the
    # `if bounds is None` branch) -- completely ignoring cond_ranges from
    # the scenario config. `_build_inversion_bnds()` (defined above this
    # worker) must be used to populate invert_kwargs["bnds"] upstream,
    # before tasks are dispatched. This assertion catches the case where
    # that got missed/regressed, rather than silently producing a capped,
    # wrong ensemble.
    # ------------------------------------------------------------------
    if invert_kwargs.get("bnds") is None:
        raise ValueError(
            "[CRITICAL] invert_kwargs['bnds'] is missing/None. Without it, "
            "EMagPy defaults to a hardcoded 2-100 mS/m conductivity bound "
            "for any free-depth inversion, ignoring cond_ranges entirely. "
            "Build bnds with _build_inversion_bnds(depths0, fixed_depths_list, "
            "cond_ranges) and include it in invert_kwargs before dispatch."
        )


    try:
        k = Problem()
        k.createSurvey(fname,
                        targetProjection=invert_kwargs.get("targetProjection"),
                        unit=invert_kwargs.get("em_unit", "ppt"))
        # NOTE: repeated/duplicate-point filtering used to happen here
        # (`k.filterRepeated(index=0)`), once per parallel inversion task.
        # It has been moved upstream into the real-data loader
        # (`0_real_data_loader.filter_repeated_em_points`), which runs once
        # per EM CSV at load time in `_load_real_data`. That also fixes a
        # bug: emagpy's `Problem.filterRepeated(self, tolerance=0.2)` has
        # never accepted an `index` keyword, so every call here was raising
        # `TypeError: filterRepeated() got an unexpected keyword argument
        # 'index'` and silently failing every single inversion task.
        drop_coils = invert_kwargs.get("drop_coils")
        if drop_coils is not None:
            EMmethod.drop_coils_from_problem(k, drop_coils)
        k.setInit(depths0=list(depths0), conds0=list(conds0),
                  fixedDepths=fixed_depths_list)

        clean_kwargs = {key: v for key, v in invert_kwargs.items()
                        if key not in ("targetProjection", "em_unit", "drop_coils")}
        k.invert(**clean_kwargs)
        models_arr = np.asarray(k.models[0])
        depths_arr = np.asarray(k.depths[0])
        pstd_list = getattr(k, "pstds", None)
        if pstd_list and len(pstd_list) > 0:
            pstd_arr = np.asarray(pstd_list[0], dtype=float)
        else:
            # Defensive fallback only -- emagpy always appends a (possibly
            # all-zero) entry to k.pstds inside invert(); this covers only
            # an unexpected emagpy version mismatch, not normal operation.
            pstd_arr = np.zeros((models_arr.shape[0], 0), dtype=float)
        return task_idx, models_arr, depths_arr, pstd_arr, None
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        print(f"\n[ERROR] Inversion failed for task {task_idx}: {err_msg}")
        print(traceback.format_exc())
        return task_idx, None, None, None, err_msg


# _layer_centers is now the single shared implementation in
# geometry_utils.py (pipeline_improvement_report.md, section 6 / action
# item 3) -- this used to be independently reimplemented here, in
# 0_generate_synthetic_data.py (inline), and in run_pipeline_short.py, and
# a mismatch between those copies is exactly what caused the Step 6d
# "x and y must have same first dimension" crash this function's old
# docstring described. Kept as a thin alias so every existing call site
# (`_layer_centers(...)`) keeps working unchanged.
_layer_centers = geometry_utils.layer_centers


def plot_truth_tdr(output_dir: str, theta_true_realization, depths0,
                    fname: str = "truth_tdr_profiles.png"):
    """
    Plot the ground-truth TDR (volumetric water content, theta) depth
    profiles for the exported realization, BEFORE any pedophysical/EM
    forward transformation. This is the theta that gets pedophysically
    converted to EC_bulk and then forward-modeled into the exported ECa
    surveys, so it's the reference point for sanity-checking that chain.

    Accepts theta_true_realization shaped (n_time, n_pos, n_layer) or
    (n_pos, n_layer) for a single-survey (non-time-lapse) run.

    Ported from run_pipeline_short.py (the synthetic-only minimal runner)
    so the full pipeline gets the same truth-check plot.
    """
    import matplotlib.pyplot as plt

    theta = np.asarray(theta_true_realization, dtype=float)
    if theta.ndim == 2:
        theta = theta[np.newaxis, ...]  # -> (1, n_pos, n_layer)
    if theta.ndim != 3:
        raise ValueError(
            f"[plot_truth_tdr] Unexpected theta_true_realization shape "
            f"{theta.shape}; expected (n_time, n_pos, n_layer) or (n_pos, n_layer)."
        )
    n_time, n_pos, n_layer = theta.shape

    # 1. Get centers for the explicitly defined bounded layers
    z_centers = _layer_centers(depths0)

    # 2. If there is an infinite half-space (n_layer > len(depths0)), append a representative depth
    if n_layer > len(z_centers):
        last_bottom = depths0[-1] if len(depths0) > 0 else 0.0
        # Extrapolate the final layer's plotting depth (e.g., 0.4m below the last boundary)
        half_space_center = last_bottom + 0.4
        z = np.concatenate((z_centers, [half_space_center]))[:n_layer]
    else:
        z = z_centers[:n_layer]

    fig, axes = plt.subplots(1, n_time, figsize=(4.5 * n_time, 5),
                              squeeze=False, sharey=True)
    axes = axes[0]

    for t in range(n_time):
        ax = axes[t]
        theta_t = theta[t]  # (n_pos, n_layer)
        for p in range(n_pos):
            ax.plot(theta_t[p], z, "o-", lw=1, alpha=0.30, color="tab:blue")
        ax.plot(theta_t.mean(axis=0), z, "o-", lw=2.5, color="tab:red",
                 label="Mean across positions")
        ax.invert_yaxis()
        ax.set_xlabel(r"$\theta$ (TDR truth) [m$^3$/m$^3$]")
        ax.set_title(f"t = {t}" if n_time > 1 else "TDR ground truth")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Depth [m]")

    fig.suptitle("Synthetic TDR Ground Truth — Volumetric Water Content Profiles",
                  fontsize=12)
    fig.tight_layout()
    fig_path = os.path.join(output_dir, fname)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] TDR ground-truth profiles     -> {fig_path}")
    return fig_path


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main(cfg: "ScenarioConfig | None" = None, mode: str = "1d",
         data_source: str = "synthetic", em_dir: str | None = None,
         tdr_dir: str | None = None, coord_file: str | None = None,
         drop_coils=None, homogeneous: bool | None = None,
         num_swc_positions: int | None = None,
         swc_depths: list[float] | None = None,
         window_m: float | None = None,
         em_unit: str | None = None,
         beta: list[float] | None = None,
         gamma: float | None = None):
    """
    Run the full EA-EM pipeline.

    Parameters
    ----------
    cfg : ScenarioConfig or None
        Scenario configuration object loaded from a JSON file (via
        ``--config``).  When None, sensible defaults are used.
        Ignored in real-data mode.
    mode : str
        Spatial mode: ``'1d'`` (default) or ``'2d'``.  In 2-D mode the
        dataset is built on a rectangular UTM grid and the water-content
        visualisation step additionally produces XY plan-view maps.
    data_source : str
        Data source: ``'synthetic'`` (default) or ``'real'``.
        Synthetic: generates forward models + noise.
        Real: loads empirical EM and TDR observations from field data.
    em_dir : str or None
        EM survey CSV directory (required when data_source='real').
    tdr_dir : str or None
        TDR Excel data directory (required when data_source='real').
    coord_file : str or None
        Sensor coordinates CSV file (used in real mode).
    drop_coils : None, str, int, float, or sequence thereof, optional
        Coil(s)/frequency channel(s) to exclude from every inversion in
        this run (e.g. a noisy GEM-2 frequency such as 425 Hz). Accepts
        the exact emagpy coil name, a frequency in Hz, or a unique coil-name
        substring like ``'f425'`` — see
        ``EMmethod.drop_coils_from_problem`` for details. When given, this
        takes priority over a ``drop_coils`` field on ``cfg`` (if any);
        when omitted, falls back to ``getattr(cfg, "drop_coils", None)``,
        so existing scenario configs without this field are unaffected.
    homogeneous : bool or None, optional
        Synthetic generation only. When True, disables spatial (XY/XZ)
        gradients in the synthetic ground truth so theta is spatially uniform.
        Temporal trend and noise are still applied. When None (default), uses
        the scenario config's 'homogeneous' value (or False if not set);
        when explicitly True or False, overrides the scenario config.
    num_swc_positions : int or None, optional
        Overrides the scenario's ``num_swc_positions`` (number of
        horizontal SWC/TDR sensor positions; total probes =
        ``num_swc_positions × n_layers``). ``None`` (default) uses the
        scenario value (or 4 if unset / no scenario given).
    swc_depths : list of float or None, optional
        Overrides the scenario's ``swc_depths`` with explicit sensor
        installation depths [m]. ``None`` (default) uses the scenario
        value, or auto-derived layer-centre depths if the scenario also
        leaves it unset.
    window_m : float or None, optional
        Overrides the scenario's ``window_m`` (half-width of the spatial
        averaging window used to match TDR probes to EM measurement
        indices). ``None`` (default) uses the scenario value (or 1.0 if
        unset / no scenario given).
    em_unit : {'ppm', 'ppt'} or None, optional
        Overrides the scenario's ``em_unit`` (physical unit of the
        quadrature/in-phase columns in the EM survey CSV: ``'ppm'`` for
        GEM-2 / CMD-Explorer instruments -- what this pipeline's real-data
        branch actually ingests -- ``'ppt'`` for GF Instruments CMD series).
        ``None`` (default) uses the scenario value; for real-data
        scenarios ``scenario_loader`` now REQUIRES ``em_unit`` to be set
        explicitly in the JSON (see ``scenario_loader._validate``), so this
        parameter exists purely as an ad-hoc CLI override, not a way to
        avoid setting it in the scenario. Ignored (falls back to
        ``"ppt"``) when no scenario is given at all.
    """
    mode = mode.lower().strip()

    # -----------------------------------------------------------------------
    # 0. PIPELINE CONFIGURATION
    # -----------------------------------------------------------------------
    if cfg is not None:
        OUTPUT_DIR        = cfg.output_dir
        REALIZATION_INDEX = cfg.realization_index
        DEPTHS0           = cfg.depths0
        FIXED_DEPTHS      = cfg.fixed_depths
        PARAM_GRID        = cfg.param_grid
        COND_RANGES       = cfg.cond_ranges
        GAMMA             = cfg.gamma
        _cache_path_override = cfg.cache_path
        EM_UNIT           = em_unit if em_unit is not None else getattr(cfg, "em_unit", "ppt")
    else:
        OUTPUT_DIR        = "mfc2_outputs"
        REALIZATION_INDEX = 0
        DEPTHS0           = [0.3, 0.6, 1.0]
        FIXED_DEPTHS      = True
        PARAM_GRID = {
            "alpha":        [0.03],
            "method":       ["L-BFGS-B"],
            "forwardModel": ["FSlin"],
        }
        COND_RANGES = [[5.0, 250.0], [5.0, 250.0], [10.0], [10.0]]
        GAMMA             = 0.0
        _cache_path_override = None
        EM_UNIT           = em_unit if em_unit is not None else "ppt"

    # --beta / --gamma CLI overrides (mirror the --window-m / --swc-depths
    # pattern: CLI value wins over whatever the scenario config set).
    if beta is not None:
        PARAM_GRID = dict(PARAM_GRID)
        PARAM_GRID["beta"] = list(beta)
    if gamma is not None:
        GAMMA = gamma

    ECA_NOISE_LEVEL = getattr(cfg, "eca_noise_level", 0.0) if cfg is not None else 0.0
    TDR_NOISE_STD   = getattr(cfg, "tdr_noise_std",   0.01) if cfg is not None else 0.01
    DROP_COILS      = (
        drop_coils if drop_coils is not None
        else (getattr(cfg, "drop_coils", None) if cfg is not None else None)
    )
    # Waxman-Smits parameters to hold fixed at their prior value during
    # Steps 2 and 4 calibration (e.g. {"m": True, "sigma_s": True}) instead
    # of letting all four (a, m, sigma_w, sigma_s) float. None (default,
    # when the scenario doesn't set ws_fixed_params) leaves everything free.
    WS_FIXED_PARAMS = getattr(cfg, "ws_fixed_params", None) if cfg is not None else None

    # Starting-guess / pinned-value Waxman-Smits parameters [a, m, sigma_w,
    # sigma_s] (flat, applied to every horizon, or one 4-tuple per horizon).
    # Used as the optimizer's starting point for free parameters AND as the
    # pinned value for any parameter listed in WS_FIXED_PARAMS. None
    # (default, when the scenario doesn't set lab_p0) falls back to
    # EMmethod.DEFAULT_WS_P0 = [1.5, 1.8, 50.0, 5.0] for every horizon.
    LAB_P0 = getattr(cfg, "lab_p0", None) if cfg is not None else None

    # Homogeneous mode: disable spatial gradients in synthetic generation.
    # CLI flag (if given) overrides scenario config.
    HOMOGENEOUS = (
        homogeneous if homogeneous is not None
        else (getattr(cfg, "homogeneous", False) if cfg is not None else False)
    )

    # TDR<->EM spatial-matching quality gate thresholds (report item 3.1).
    # largest-group-fraction = (probes sharing the single most-shared EM
    # point) / (total probes). Above MATCHING_WARN_FRACTION the run is
    # flagged with a [WARN] in the report; if MATCHING_ERROR_FRACTION is
    # also set and exceeded, the pipeline aborts rather than silently
    # calibrating against a near-degenerate spatial match. Both are read
    # from the scenario config (getattr with defaults keeps this backward
    # compatible with scenarios that don't define these fields).
    MATCHING_WARN_FRACTION  = getattr(cfg, "matching_warn_fraction", 0.5) if cfg is not None else 0.5
    MATCHING_ERROR_FRACTION = getattr(cfg, "matching_error_fraction", None) if cfg is not None else None

    # Real-data time-lapse spatial reference-grid tolerance (report:
    # "spatial reference-grid matching, not row trimming"). Only used in
    # Step 1 when data_source == "real" and multiple survey epochs are
    # present; see geometry_utils.match_epochs_to_reference_grid. Computed
    # here (rather than inline in Step 1) so it can feed the Step 1 cache
    # fingerprint below regardless of whether Step 1 ends up being a cache
    # hit or a fresh run.
    SPATIAL_MATCH_TOLERANCE_M = (
        getattr(cfg, "spatial_match_tolerance_m", 2.0) if cfg is not None else 2.0
    )

    # Initialise run reporter
    _scenario_name = cfg.name if cfg is not None else "default"
    reporter = PipelineReporter(
        output_dir=OUTPUT_DIR,
        scenario_name=_scenario_name,
        mode=mode,
        cfg=cfg,
        data_source=data_source,
        em_dir=em_dir,
        tdr_dir=tdr_dir,
    )
    reporter.args_table("Runtime Arguments", {
        "mode":               mode.upper(),
        "scenario":           _scenario_name,
        "output_dir":         OUTPUT_DIR,
        "eca_noise_level":    f"{ECA_NOISE_LEVEL*100:.1f}%",
        "tdr_noise_std":      f"{TDR_NOISE_STD*100:.1f}% vol",
        "realization_index":  REALIZATION_INDEX,
        "fixed_depths":       FIXED_DEPTHS,
        "gamma":              GAMMA,
        "num_swc_positions":  getattr(cfg, "num_swc_positions", 4) if cfg is not None else 4,
        "swc_depths":         (
            f"{getattr(cfg, 'swc_depths', None)}  "
            "(None = not set in scenario -> auto-derived from layer centres; "
            "see 'SWC / TDR Probe Network' table further down for the actual "
            "resolved x/y/z of every probe)"
            if (cfg is None or getattr(cfg, "swc_depths", None) is None)
            else str(cfg.swc_depths)
        ),
        "window_m":           getattr(cfg, "window_m", 1.0) if cfg is not None else 1.0,
        "alpha_range_log":    str(getattr(cfg, "alpha_range_log", None) if cfg is not None else None),
        "ws_fixed_params":    str(WS_FIXED_PARAMS) if WS_FIXED_PARAMS else "(none — a/m/sigma_w/sigma_s all free)",
        "lab_p0":             str(LAB_P0) if LAB_P0 else "(none — using DEFAULT_WS_P0 = [1.5, 1.8, 50.0, 5.0])",
        "homogeneous":        HOMOGENEOUS,
        "spatial_match_tolerance_m (real time-lapse)": SPATIAL_MATCH_TOLERANCE_M,
    })

    try:
        # -----------------------------------------------------------------------
        # STEP 0 — DATA PREPARATION (SYNTHETIC OR REAL)
        # -----------------------------------------------------------------------
        REAL_DATA_DIAG = None  # populated below only when data_source == "real";
                               # referenced later (temperature correction) regardless of branch
        if data_source == "real":
            reporter.step("Step 0", "Loading real EM and TDR field data")
            reporter.log(f"[PIPELINE] STEP 0: Loading real field observations...")
            reporter.log(f"           EM directory    : {em_dir}")
            reporter.log(f"           TDR directory   : {tdr_dir}")
            reporter.log(f"           Coordinates     : {coord_file}")
            reporter.log(f"           Drop coils      : {DROP_COILS if DROP_COILS else '(none)'}")

            # Load real data
            EM_FNAMES, observations_list, probe_table, EM_DATES, RAW_EM_FNAMES, REAL_DATA_DIAG = _load_real_data(
                em_dir, tdr_dir, coord_file,
                time_tolerance_days=getattr(cfg, "time_tolerance_days", 1.0),
                em_unit=EM_UNIT,
            )

            # -----------------------------------------------------------------------
            # Fold the CRS classification/reprojection + em_unit sanity-check
            # diagnostics into the Markdown run report. Previously these were
            # print()-only (visible in the console, but absent from
            # pipeline_report_*.md) -- see fix "Is this step in the log
            # recorder?". _load_real_data now returns them structured
            # specifically so they can be logged here.
            # -----------------------------------------------------------------------
            _coord_rows = [
                [r["file"], r["coord_kind"], r["epsg_used"] or "—"]
                for r in REAL_DATA_DIAG["coord_kinds"]
            ]
            reporter.table(
                "Step 0 — EM survey coordinate system per file",
                ["File", "Coordinate kind", "EPSG used (if reprojected)"],
                _coord_rows,
                note=(
                    f"{REAL_DATA_DIAG['n_geographic_reprojected']}/{len(_coord_rows)} file(s) "
                    "used geographic (Longitude/Latitude) columns and were auto-reprojected "
                    "to metric UTM before any distance/matching computation."
                ),
            )
            reporter.metric(
                "Real EM files reprojected (geographic → UTM)",
                f"{REAL_DATA_DIAG['n_geographic_reprojected']}/{len(_coord_rows)}",
            )

            _em_unit_rows = [
                [r["file"], r["level"],
                 f"{r['median_abs_eca']:.4g}" if r["median_abs_eca"] is not None else "—"]
                for r in REAL_DATA_DIAG["em_unit_checks"]
            ]
            reporter.table(
                "Step 0 — em_unit sanity check (median |LIN ECa| per file)",
                ["File", "Level", "Median abs(ECa) [mS/m]"],
                _em_unit_rows,
                note=f"em_unit='{EM_UNIT}'; expected field-soil range ~1-300 mS/m.",
            )
            _em_unit_warn_files = [r["file"] for r in REAL_DATA_DIAG["em_unit_checks"] if r["level"] == "WARN"]
            if _em_unit_warn_files:
                reporter.log(
                    f"[STEP 0][WARN] em_unit sanity check flagged {len(_em_unit_warn_files)} "
                    f"file(s) as likely em_unit mismatches: {_em_unit_warn_files}. "
                    "See the table above."
                )

            _crs = REAL_DATA_DIAG["crs_consistency"]
            reporter.log(f"[STEP 0] CRS consistency check ({_crs['level']}): {_crs['message']}")
            reporter.metric("CRS consistency check (EM vs. TDR probes)", _crs["level"])
            if _crs["level"] == "WARN":
                reporter.log(
                    "[STEP 0][WARN] EM survey and TDR probe coordinates look "
                    "spatially inconsistent -- see message above. TDR<->EM window "
                    "matching (Step 1) may be unreliable until this is resolved."
                )

            # Extract observations into dict structure for processing
            # observations_list contains: probe_index, time_index, depth, swc
            theta_true_realization = None  # Not used in real data mode
            TDR_NOISE_STD = 0.0  # Real data has actual noise, not synthetic

            # Build synthetic_geometry placeholder (needed for layer_centers etc.
            # downstream). DEPTHS0 is already resolved above (from cfg.depths0,
            # or the no-scenario fallback) so just reuse it directly here.
            #
            # FIX: the synthetic branch below gets "layer_centers" for free
            # from step0.generate_time_lapse_dataset(), but this real-data
            # placeholder previously only set "layer_bottoms" -- every
            # downstream consumer that does synthetic_geometry["layer_centers"]
            # (SWC/TDR probe depth resolution just below, and Step 6d's
            # psol.plot_solution_swc / psol.plot_solution_ec_bulk calls) then
            # crashed with `KeyError: 'layer_centers'` as soon as a real-data
            # run reached them. Compute the same layer-centre depths here
            # using the module-level _layer_centers() helper (assumes the
            # top of layer 0 is the surface, z=0 -- identical convention to
            # the synthetic generator) so both branches expose the same keys.
            synthetic_geometry = {
                "layer_bottoms": DEPTHS0,
                "layer_centers": _layer_centers(DEPTHS0).tolist(),
            }

            SURVEY_TIMESTAMPS = list(range(len(EM_FNAMES)))
            timelapse = _is_timelapse(EM_FNAMES)
            n_time    = len(EM_FNAMES) if timelapse else 1

            reporter.log(f"           Time steps      : {n_time}")
            reporter.log(f"           Observations    : {len(observations_list)}")
            reporter.end_step("Step 0")

        else:
            # SYNTHETIC DATA PATH (original behavior)
            reporter.step("Step 0", "Generating synthetic time-lapse EM/TDR dataset")
            reporter.log(f"[PIPELINE] Spatial mode     : {mode.upper()}")
            reporter.log("[PIPELINE] STEP 0: Generating synthetic time-lapse EM/TDR dataset...")
            reporter.log(f"           ECa noise level  : {ECA_NOISE_LEVEL*100:.1f}%")
            reporter.log(f"           TDR noise std    : {TDR_NOISE_STD*100:.1f}% vol.")
            reporter.log(f"           Drop coils       : {DROP_COILS if DROP_COILS else '(none)'}")

            synthetic_geometry = step0.generate_time_lapse_dataset(
                output_dir=OUTPUT_DIR,
                mode=mode,
                # 2-D grid size — override via scenario config if needed
                num_x=getattr(cfg, "grid_num_x", 10) if cfg is not None else 10,
                num_y=getattr(cfg, "grid_num_y",  5) if cfg is not None else 5,
                # Pass scenario layer geometry so the synthetic ground truth
                # has the same number of layers as the inversion.
                depths0=DEPTHS0,
                cond_ranges=COND_RANGES,
                homogeneous=HOMOGENEOUS,
            )
            step0.plot_synthetic_results(output_dir=OUTPUT_DIR)

            EM_FNAMES, theta_true_realization, TDR_NOISE_STD = step0.export_eca_surveys_to_csv(
                output_dir=OUTPUT_DIR,
                realization_index=REALIZATION_INDEX,
                eca_noise_level=ECA_NOISE_LEVEL,
                tdr_noise_std=TDR_NOISE_STD,
            )

            try:
                truth_tdr_path = plot_truth_tdr(
                    output_dir=OUTPUT_DIR,
                    theta_true_realization=theta_true_realization,
                    depths0=synthetic_geometry.get("layer_bottoms", DEPTHS0),
                )
                reporter.log(f"[STEP 0] TDR ground-truth profiles -> {truth_tdr_path}")
            except Exception as e:
                reporter.log(f"[STEP 0][WARN] plot_truth_tdr failed: {e}")

            SURVEY_TIMESTAMPS = list(range(len(EM_FNAMES)))
            if cfg is None:
                DEPTHS0 = synthetic_geometry["layer_bottoms"]

            timelapse = _is_timelapse(EM_FNAMES)
            n_time    = len(EM_FNAMES) if timelapse else 1
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            observations_list = None  # Not used in synthetic mode

        # -----------------------------------------------------------------------
        # EARLY TDR/SWC HORIZON COVERAGE CHECK  (report item 4, quality gates)
        #   Runs BEFORE the potentially 30+ minute Step 1 ensemble inversion.
        #   Uses only DEPTHS0 and the known probe/sensor installation depths
        #   (already available right after Step 0, for both real and
        #   synthetic data) -- neither requires the EM survey to have been
        #   inverted. Catches a depths0/sensor-depth mismatch that would
        #   otherwise only surface as an "UNCALIBRATED" horizon after the
        #   full inversion has already run.
        # -----------------------------------------------------------------------
        reporter.step("Step 0b", "Early TDR/SWC horizon coverage check")
        if data_source == "real":
            _early_probe_depths = sorted({obs["depth"] for obs in observations_list})
        else:
            _early_swc_depths_cfg = (
                getattr(cfg, "swc_depths", None) if cfg is not None else None
            )
            _early_probe_depths = (
                _early_swc_depths_cfg if _early_swc_depths_cfg is not None
                else synthetic_geometry["layer_centers"]
            )
        _coverage_check = EMmethod.check_early_horizon_coverage(
            probe_depths=_early_probe_depths, depths0=DEPTHS0,
        )
        reporter.log(_coverage_check.message)
        reporter.metric("Early coverage — level", _coverage_check.level)
        reporter.metric("Early coverage — uncovered horizons", len(_coverage_check.uncovered_horizons))
        if _coverage_check.level == "WARN":
            reporter.log(
                "[PIPELINE][WARN] Proceeding despite incomplete horizon coverage -- "
                "the affected horizon(s) will fall back to prior/uncalibrated "
                "Waxman-Smits parameters in Steps 2/4. Consider fixing depths0 or "
                "the SWC/TDR sensor depths before committing to a full ensemble run "
                "(re-run with --list / scenario_loader.py <name> to check quickly)."
            )
        reporter.end_step("Step 0b")

        # -----------------------------------------------------------------------
        # STEP 1a — PRE-INVERSION IMPORT/FILTER/SPATIAL-MATCHING DIAGNOSTICS
        #   Renders sanity-check figures (import/filter summary, reference-
        #   grid coverage, optional tolerance sweep, TDR<->EM point matching).
        #   Placed HERE -- immediately after Step 0b and before the cache
        #   check / ProcessPoolExecutor dispatch below -- so it genuinely
        #   runs before any inversion work, not merely before the
        #   reporter.step("Step 1", ...) bookkeeping call (that call is just
        #   where the log entry is made; the actual expensive computation
        #   starts right after this block, at the "cache_hit" check).
        #   Consumes EM_FNAMES/RAW_EM_FNAMES/EM_DATES/probe_table/
        #   observations_list -- already imported, pre-processed, and
        #   filterRepeated'd by Step 0's _load_real_data() call above -- so
        #   this does NOT re-run that IMPORT/PRE-PROCESS/FILTER chain a
        #   second time; only the (cheap, position-only) spatial reference-
        #   grid matching and the figures themselves are computed here.
        #   Only runs in real-data mode; wrapped in try/except so a plot
        #   failure never aborts the pipeline.
        # -----------------------------------------------------------------------
        if data_source == "real":
            reporter.step("Step 1a", "Pre-inversion import/filter/spatial-matching diagnostics")
            reporter.log(" [PIPELINE] STEP 1a: PRE-INVERSION DIAGNOSTICS (reusing Step 0 data)")
            try:
                _pre_inv_result = preinvdiag.run_pre_inversion_diagnostics_from_pipeline(
                    em_files=EM_FNAMES,
                    raw_em_files=RAW_EM_FNAMES,
                    em_dates=EM_DATES,
                    probe_table=probe_table,
                    observations=observations_list,
                    output_dir=OUTPUT_DIR,
                    time_tolerance_days=getattr(cfg, "time_tolerance_days", 1.0),
                    spatial_match_tolerance_m=SPATIAL_MATCH_TOLERANCE_M,
                    window_m=(
                        window_m if window_m is not None
                        else (getattr(cfg, "window_m", 1.0) if cfg is not None else 1.0)
                    ),
                    matching_warn_fraction=MATCHING_WARN_FRACTION,
                )
                reporter.log(
                    f"[STEP 1a] {len(_pre_inv_result['figures'])} diagnostic figure(s) written "
                    f"to '{os.path.join(OUTPUT_DIR, 'diagnostics', 'pre_inversion')}'"
                )
                reporter.metric("Pre-inversion diagnostic figures", len(_pre_inv_result["figures"]))
                reporter.metric("Pre-inversion — reference grid size", _pre_inv_result["n_ref_points"])
                if _pre_inv_result["zero_obs_epochs"]:
                    reporter.log(
                        f"[STEP 1a][WARN] {len(_pre_inv_result['zero_obs_epochs'])} epoch(s) "
                        f"matched ZERO TDR observations within the configured "
                        f"time_tolerance_days: epoch index(es) {_pre_inv_result['zero_obs_epochs']}. "
                        "Consider raising time_tolerance_days."
                    )
                _match_summary = _pre_inv_result.get("matching_summary")
                if _match_summary is not None:
                    reporter.log(f"[STEP 1a] {_match_summary.message}")
            except Exception as _pre_inv_exc:
                reporter.log(
                    f"[STEP 1a][WARN] Pre-inversion diagnostics failed (pipeline continues): "
                    f"{type(_pre_inv_exc).__name__}: {_pre_inv_exc}"
                )
            reporter.end_step("Step 1a")
        elif data_source == "synthetic":
            # Synthetic epochs share one fixed grid by construction, so the
            # real-data-only import/filter/reference-grid machinery above
            # (Figures 1-3) doesn't apply here. The one thing worth
            # checking before the (potentially long) Step 1 inversion is
            # still the probe<->EM spatial-matching diagnostic (Figure 4 on
            # the real-data path) -- see
            # preinvdiag.run_pre_inversion_diagnostics_synthetic_from_pipeline
            # for details, including the CRS-consistency check this reuses
            # from the real-data path.
            reporter.step("Step 1a", "Pre-inversion synthetic spatial-matching diagnostics")
            reporter.log(" [PIPELINE] STEP 1a: PRE-INVERSION DIAGNOSTICS (synthetic)")
            try:
                _synth_utm_x_path = os.path.join(OUTPUT_DIR, "synthetic_utm_x.npy")
                _synth_utm_y_path = os.path.join(OUTPUT_DIR, "synthetic_utm_y.npy")
                _synth_utm_x = (np.load(_synth_utm_x_path)
                                if mode == "2d" and os.path.exists(_synth_utm_x_path) else None)
                _synth_utm_y = (np.load(_synth_utm_y_path)
                                if mode == "2d" and os.path.exists(_synth_utm_y_path) else None)

                _pre_inv_result_synth = preinvdiag.run_pre_inversion_diagnostics_synthetic_from_pipeline(
                    em_files=EM_FNAMES if timelapse else [EM_FNAMES],
                    output_dir=OUTPUT_DIR,
                    depths0=DEPTHS0,
                    layer_centers=synthetic_geometry["layer_centers"],
                    num_swc_positions=(
                        getattr(cfg, "num_swc_positions", 4) if cfg is not None else 4
                    ),
                    swc_depths=(
                        getattr(cfg, "swc_depths", None) if cfg is not None else None
                    ),
                    window_m=(
                        window_m if window_m is not None
                        else (getattr(cfg, "window_m", 1.0) if cfg is not None else 1.0)
                    ),
                    matching_warn_fraction=MATCHING_WARN_FRACTION,
                    mode=mode,
                    utm_x=_synth_utm_x,
                    utm_y=_synth_utm_y,
                )
                reporter.log(
                    f"[STEP 1a] {len(_pre_inv_result_synth['figures'])} diagnostic figure(s) written "
                    f"to '{os.path.join(OUTPUT_DIR, 'diagnostics')}'"
                )
                reporter.metric(
                    "Pre-inversion diagnostic figures (synthetic)",
                    len(_pre_inv_result_synth["figures"]),
                )
                _match_summary_synth = _pre_inv_result_synth.get("matching_summary")
                if _match_summary_synth is not None:
                    reporter.log(f"[STEP 1a] {_match_summary_synth.message}")
                _crs_synth = _pre_inv_result_synth.get("crs_consistency")
                if _crs_synth:
                    reporter.log(
                        f"[STEP 1a][WARN] Possible CRS mismatch between synthetic EM "
                        f"survey and probe coordinates: {_crs_synth}"
                    )
            except Exception as _pre_inv_exc:
                reporter.log(
                    f"[STEP 1a][WARN] Pre-inversion synthetic diagnostics failed "
                    f"(pipeline continues): {type(_pre_inv_exc).__name__}: {_pre_inv_exc}"
                )
            reporter.end_step("Step 1a")

        # -----------------------------------------------------------------------
        # STEP 1 — PARALLEL ENSEMBLE INVERSION
        # -----------------------------------------------------------------------
        print("\n" + "=" * 80)
        print(" [PIPELINE] STEP 1: PARALLEL MULTI-INVERSION ENSEMBLE GENERATION")
        print(f"            Mode: {'TIME-LAPSE (' + str(n_time) + ' surveys)' if timelapse else 'SINGLE FILE'}")
        print("=" * 80)

        if _cache_path_override is not None:
            cache_path = _cache_path_override
        else:
            cache_filename = "ensemble_cache_timelapse.pkl" if timelapse else "ensemble_cache_single.pkl"
            cache_path = os.path.join(OUTPUT_DIR, cache_filename)

        # Content fingerprint of everything that determines Step 1's output.
        # Guards against a stale cache_tag: if depths0/fixed_depths/param_grid/
        # cond_ranges/gamma/drop_coils/spatial_match_tolerance_m changed since
        # the cache was written but cache_tag wasn't bumped to match, we still
        # detect the drift here and force a fresh inversion instead of
        # silently reusing outdated models. spatial_match_tolerance_m only
        # matters for real-data time-lapse runs -- pass None otherwise so it
        # never spuriously invalidates a synthetic/single-survey cache.
        _fingerprint_tolerance = (
            SPATIAL_MATCH_TOLERANCE_M if (data_source == "real" and timelapse) else None
        )
        current_fingerprint = _config_fingerprint(
            DEPTHS0, FIXED_DEPTHS, PARAM_GRID, COND_RANGES, GAMMA, DROP_COILS,
            spatial_match_tolerance_m=_fingerprint_tolerance,
        )

        # Real-data time-lapse spatial reference grid coordinates (union
        # across all survey epochs -- see geometry_utils.match_epochs_to_
        # reference_grid). None for synthetic data and for real single-survey
        # (non-time-lapse) runs, where no cross-epoch spatial matching is
        # needed. Populated either from a cache hit below or freshly built
        # in the "no cache" branch further down.
        REAL_DATA_REFERENCE_XY = None

        cache_hit = False
        if os.path.exists(cache_path):
            print(f"[CACHE] Found existing cache: '{cache_path}'")
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            cached_fingerprint = cache_data.get("config_fingerprint")
            if cached_fingerprint is None:
                print("[CACHE][WARN] Cache predates config fingerprinting — "
                      "cannot verify it matches the current scenario config. "
                      "Re-running inversion to be safe.")
            elif cached_fingerprint != current_fingerprint:
                print("[CACHE][STALE] Cached ensemble was built from a different "
                      "config (depths0/fixed_depths/param_grid/cond_ranges/gamma/"
                      "drop_coils/spatial_match_tolerance_m changed) but cache_tag "
                      "was not updated to match. "
                      f"cached={cached_fingerprint} current={current_fingerprint}. "
                      "Ignoring stale cache and re-running inversion.")
            else:
                models_clean   = cache_data["models_clean"]
                param_metadata = cache_data["param_metadata"]
                depths_clean   = cache_data.get("depths_clean")
                pstds_clean    = cache_data.get("pstds_clean")
                REAL_DATA_REFERENCE_XY = cache_data.get("reference_xy")
                if depths_clean is None:
                    print("[CACHE][WARN] Cache predates depth capture — re-run to update.")
                if pstds_clean is None:
                    print(
                        "[CACHE][WARN] Cache predates the MCMC-posterior-std capture "
                        "('use the full MCMC, not just a single selected model') and "
                        "has no stored pstds_clean. MCMC uncertainty reporting will "
                        "be skipped this run -- delete this cache and re-run to "
                        "populate it."
                    )
                if data_source == "real" and timelapse and REAL_DATA_REFERENCE_XY is None:
                    print(
                        "[CACHE][WARN] Cache predates the spatial reference-grid "
                        "fix (report: 'spatial reference-grid matching, not row "
                        "trimming') and has no stored reference-grid coordinates. "
                        "Falling back to loading epoch-0 survey positions only "
                        "for probe placement -- delete this cache and re-run for "
                        "the full union-of-all-epochs reference grid."
                    )
                print(f"[CACHE] Loaded {models_clean.shape[0]} ensemble members "
                      f"(config fingerprint verified: {current_fingerprint}).")
                cache_hit = True

        if not cache_hit:
            print("[CACHE] No valid cache found. Running parallel inversion tasks...")

            if timelapse:
                for t_idx, fname in enumerate(EM_FNAMES):
                    EMmethod.plot_raw_apparent_conductivity(
                        fname, os.path.join(OUTPUT_DIR, f"t{t_idx:02d}"),
                        em_unit=EM_UNIT,
                    )
            else:
                EMmethod.plot_raw_apparent_conductivity(EM_FNAMES, OUTPUT_DIR,
                                                        em_unit=EM_UNIT)

            alphas         = PARAM_GRID.get("alpha",        [0.03])
            betas          = PARAM_GRID.get("beta",         [0.0])
            methods        = PARAM_GRID.get("method",       ["L-BFGS-B"])
            forward_models = PARAM_GRID.get("forwardModel", ["FSlin"])
            combinations   = [(a, b, m, f) for a in alphas for b in betas
                              for m in methods for f in forward_models]

            # Full Cartesian-product expansion of per-layer starting conductivities
            # (was previously a zip-style pairing that only ever produced the
            # "all-minima" and "all-maxima" vectors — see EMmethod.build_conductivity_ensemble
            # for the canonical implementation, now reused here so both code paths
            # can never drift out of sync again).
            cond_sets = EMmethod.build_conductivity_ensemble(COND_RANGES)
            print(f"[INFO] Starting-conductivity grid: {len(COND_RANGES)} layers -> "
                  f"{len(cond_sets)} combinations (was 2 before the Cartesian-product fix).")

            fixed_depths_list = (
                [FIXED_DEPTHS] * len(DEPTHS0)
                if isinstance(FIXED_DEPTHS, bool) else list(FIXED_DEPTHS)
            )

            # FIX: explicitly build solver bounds from COND_RANGES instead of
            # letting EMagPy silently fall back to its hardcoded 2-100 mS/m
            # default (see _build_inversion_bnds() docstring above).
            bnds_list = _build_inversion_bnds(DEPTHS0, fixed_depths_list, COND_RANGES)
            print(f"[INFO] Inversion bnds (depths+conds, {len(bnds_list)} params): {bnds_list}")

            combo_list = []
            for alpha, beta, method, f_model in combinations:
                for conds0 in cond_sets:
                    combo_list.append((alpha, beta, method, f_model, list(conds0)))

            files_to_process = EM_FNAMES if timelapse else [EM_FNAMES]
            tasks = []
            task_meta = {}          # task_idx -> dict describing the combo, for failure reporting
            global_task_idx = 0
            for c_idx, (alpha, beta, method, f_model, conds0) in enumerate(combo_list):
                for t_idx, file_path in enumerate(files_to_process):
                    invert_kwargs = {
                        "alpha": alpha, "beta": beta, "gamma": GAMMA,
                        "method": method,
                        "forwardModel": f_model, "targetProjection": None,
                        "em_unit": EM_UNIT, "drop_coils": DROP_COILS,
                        "bnds": bnds_list,
                    }
                    tasks.append((global_task_idx, file_path, DEPTHS0, conds0,
                                   fixed_depths_list, invert_kwargs))
                    task_meta[global_task_idx] = {
                        "combo_idx": c_idx, "t_idx": t_idx, "file": file_path,
                        "alpha": alpha, "beta": beta, "method": method,
                        "forwardModel": f_model,
                        "conds0": conds0,
                    }
                    global_task_idx += 1

            print(f"[INFO] Dispatched {len(tasks)} tasks to parallel pool...")

            unordered_results = []
            failed_tasks = []       # list of (task_idx, error_message)
            with ProcessPoolExecutor() as executor:
                results = tqdm(
                    executor.map(_invert_single_worker, tasks),
                    total=len(tasks), desc="Inverting Ensembles", unit="task",
                    ncols=90,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                )
                for res in results:
                    t_idx_res, m_res, d_res, p_res, err_res = res
                    unordered_results.append((t_idx_res, m_res, d_res, p_res))
                    if err_res is not None:
                        failed_tasks.append((t_idx_res, err_res))

            sorted_results = [(m, d, p) for _, m, d, p in sorted(unordered_results, key=lambda x: x[0])]

            packed_models = []
            packed_depths = []
            # FIX ("use the full MCMC, not just a single selected model"):
            # per-point, per-free-parameter MCMC/ROPE/DREAM/SCEUA posterior
            # std from emagpy's k.pstds (all-zero for non-MCMC-family
            # methods), packed/cached/remapped in lockstep with
            # packed_models/packed_depths below. See
            # EMmethod.extract_conductivity_pstd for how the per-layer
            # conductivity columns are later sliced out of this.
            packed_pstds = []
            valid_param_metadata = []
            discarded_combos = []   # combos dropped because >=1 of their timesteps failed

            for c_idx, (alpha, beta, method, f_model, conds0) in enumerate(combo_list):
                combo_models = []
                combo_depths = []
                combo_pstds = []
                combo_failed = False
                fail_t_idx = None
                for t_idx in range(n_time):
                    flat_idx = c_idx * n_time + t_idx
                    model_result, depths_result, pstd_result = sorted_results[flat_idx]
                    if model_result is None:
                        combo_failed = True
                        fail_t_idx = t_idx
                        break
                    combo_models.append(model_result)
                    combo_depths.append(depths_result)
                    combo_pstds.append(pstd_result)
                if not combo_failed:
                    packed_models.append(combo_models if timelapse else combo_models[0])
                    packed_depths.append(combo_depths if timelapse else combo_depths[0])
                    packed_pstds.append(combo_pstds if timelapse else combo_pstds[0])
                    valid_param_metadata.append({
                        "alpha": alpha, "beta": beta, "method": method,
                        "forwardModel": f_model, "conds0": conds0,
                    })
                else:
                    discarded_combos.append({
                        "combo_idx": c_idx, "alpha": alpha, "beta": beta,
                        "method": method,
                        "forwardModel": f_model, "conds0": conds0,
                        "failed_at_t_idx": fail_t_idx,
                    })

            # ---------------------------------------------------------------
            # Failure tracking — surfaced in the report + a dedicated log file
            # ---------------------------------------------------------------
            n_total_tasks    = len(tasks)
            n_failed_tasks   = len(failed_tasks)
            n_total_combos   = len(combo_list)
            n_kept_combos    = len(valid_param_metadata)
            n_dropped_combos = len(discarded_combos)

            reporter.log(
                f"[STEP1] Tasks: {n_total_tasks - n_failed_tasks}/{n_total_tasks} succeeded "
                f"({n_failed_tasks} failed). Combos: {n_kept_combos}/{n_total_combos} kept "
                f"({n_dropped_combos} dropped because >=1 of their {n_time} timestep(s) failed)."
            )
            reporter.metric("Inversion tasks dispatched", n_total_tasks)
            reporter.metric("Inversion tasks failed", n_failed_tasks)
            reporter.metric("Ensemble combos kept", n_kept_combos)
            reporter.metric("Ensemble combos dropped", n_dropped_combos)

            if failed_tasks:
                fail_log_path = os.path.join(OUTPUT_DIR, "step1_failed_tasks.log")
                with open(fail_log_path, "w", encoding="utf-8") as flog:
                    flog.write(f"# {n_failed_tasks}/{n_total_tasks} inversion tasks failed\n\n")
                    for t_idx_f, err_f in sorted(failed_tasks, key=lambda x: x[0]):
                        meta = task_meta.get(t_idx_f, {})
                        flog.write(
                            f"task_idx={t_idx_f} combo_idx={meta.get('combo_idx')} "
                            f"t_idx={meta.get('t_idx')} file={meta.get('file')} "
                            f"alpha={meta.get('alpha')} beta={meta.get('beta')} "
                            f"method={meta.get('method')} "
                            f"forwardModel={meta.get('forwardModel')} conds0={meta.get('conds0')} "
                            f"-- {err_f}\n"
                        )
                    if discarded_combos:
                        flog.write(
                            f"\n# {n_dropped_combos} ensemble combo(s) discarded entirely "
                            f"(>=1 of their {n_time} timestep(s) failed)\n\n"
                        )
                        for dc in discarded_combos:
                            flog.write(f"{dc}\n")
                reporter.log(f"[STEP1] Failure details written to '{fail_log_path}'")

            if not packed_models:
                raise ValueError("[CRITICAL] All inversion combinations failed.")

            # ------------------------------------------------------------------
            # SPATIAL ASSEMBLY OF THE ENSEMBLE ACROSS EPOCHS
            #
            # Real GEM-2 time-lapse surveys are inverted independently per
            # epoch (no trimSurveys / temporal coupling) and can differ both
            # in point COUNT and in walk ORDER across campaigns -- the
            # surveyor need not retrace an identical path every visit.
            #
            # FIX ("spatial reference-grid matching, not row trimming"):
            # previously every epoch was sliced to `arr[:min_pts]` -- the
            # minimum point count across ALL epochs -- which pairs "row i of
            # epoch A" with "row i of epoch B" purely by array position. For
            # anything other than the shortest epoch (and any epoch not
            # walked in identical order), that silently paired UNRELATED
            # physical locations under the same spatial index, and threw
            # away every point beyond the shortest survey's count even
            # though those points were successfully inverted.
            #
            # Real data (time-lapse): build a UNION spatial reference grid
            # across every epoch's own measurement points
            # (geometry_utils.match_epochs_to_reference_grid), then
            # nearest-neighbour remap each epoch's own inverted EC array
            # onto that shared grid, leaving NaN for any epoch that didn't
            # cover a given reference point. No epoch's own points are ever
            # reordered, subsetted, or discarded; NaN-safe downstream
            # handling (Steps 2-5, see em_methodology.py /
            # 4_run_select_assess.py) treats an unmatched cell as "no data"
            # rather than silently pairing it with an unrelated location.
            #
            # Synthetic data: point counts are consistent across epochs by
            # construction (same generator, same grid, every timestep), so
            # the defensive min_pts trim is kept unchanged as a backward-
            # compatible no-op safety net.
            # ------------------------------------------------------------------
            if timelapse:
                if data_source == "real":
                    reporter.log(
                        f"[STEP1] Building spatial reference grid across {n_time} "
                        f"real-data survey epochs (tolerance="
                        f"{SPATIAL_MATCH_TOLERANCE_M} m) instead of trimming to a "
                        f"common point COUNT -- see 'spatial reference-grid "
                        f"matching, not row trimming'."
                    )

                    epoch_xy: list[tuple[np.ndarray, np.ndarray]] = []
                    for t_idx, fname in enumerate(EM_FNAMES):
                        _pos_df = EMmethod.load_survey_positions(
                            fname, targetProjection=None, em_unit=EM_UNIT,
                        )
                        _x_epoch = _pos_df["x"].to_numpy(dtype=float)
                        _y_epoch = (
                            _pos_df["y"].to_numpy(dtype=float)
                            if "y" in _pos_df.columns else np.zeros_like(_x_epoch)
                        )
                        _n_model_pts = packed_models[0][t_idx].shape[0]
                        if len(_x_epoch) != _n_model_pts:
                            # Position table and inverted model disagree on
                            # row count for this epoch (shouldn't normally
                            # happen since both are read from the same
                            # already-deduplicated processed CSV, but guard
                            # defensively rather than mis-pairing points).
                            reporter.log(
                                f"[STEP1][WARN] Epoch {t_idx} "
                                f"('{os.path.basename(fname)}'): position table has "
                                f"{len(_x_epoch)} rows but its inverted model has "
                                f"{_n_model_pts} rows. Truncating the shorter of the "
                                f"two before spatial matching so coordinates and EC "
                                f"values stay row-aligned for this epoch."
                            )
                            _n_use = min(len(_x_epoch), _n_model_pts)
                            _x_epoch = _x_epoch[:_n_use]
                            _y_epoch = _y_epoch[:_n_use]
                        epoch_xy.append((_x_epoch, _y_epoch))

                    ref_x, ref_y, matched_indices = geometry_utils.match_epochs_to_reference_grid(
                        epoch_xy, tolerance_m=SPATIAL_MATCH_TOLERANCE_M,
                    )
                    n_ref = len(ref_x)

                    for t_idx in range(n_time):
                        n_matched = int(np.sum(matched_indices[t_idx] >= 0))
                        reporter.log(
                            f"[STEP1]   Epoch {t_idx} ('{os.path.basename(EM_FNAMES[t_idx])}'): "
                            f"{len(epoch_xy[t_idx][0])} own point(s) -> "
                            f"{n_matched}/{n_ref} reference-grid cell(s) matched "
                            f"({n_ref - n_matched} left as NaN for this epoch)."
                        )
                    reporter.metric("Spatial reference grid — union points across epochs", n_ref)
                    reporter.metric(
                        "Spatial reference grid — tolerance (m)", SPATIAL_MATCH_TOLERANCE_M
                    )

                    def _remap_combo_to_reference(arr_per_time: list[np.ndarray]) -> list[np.ndarray]:
                        """Remap one combo's list of per-epoch (n_pts_t, n_layers)
                        (or (n_pts_t, n_interfaces) for depths) arrays onto the
                        shared (n_ref, n_layers) reference grid, NaN-filling any
                        reference point that particular epoch didn't cover."""
                        remapped = []
                        for t_idx, arr in enumerate(arr_per_time):
                            out = np.full((n_ref, arr.shape[-1]), np.nan, dtype=float)
                            idx_map = matched_indices[t_idx]
                            valid = idx_map >= 0
                            if np.any(valid):
                                out[valid] = arr[idx_map[valid]]
                            remapped.append(out)
                        return remapped

                    packed_models = [_remap_combo_to_reference(combo) for combo in packed_models]
                    packed_depths = [_remap_combo_to_reference(combo) for combo in packed_depths]
                    # pstd arrays share the exact same per-epoch row order/
                    # count as models/depths (same k, same survey), so the
                    # same coordinate-based remap applies unchanged.
                    packed_pstds  = [_remap_combo_to_reference(combo) for combo in packed_pstds]

                    REAL_DATA_REFERENCE_XY = (ref_x, ref_y)
                else:
                    # Synthetic data: point counts already consistent across
                    # epochs by construction; keep the historical defensive
                    # min_pts trim as a backward-compatible no-op safety net.
                    min_pts = min(
                        arr.shape[0]
                        for combo in packed_models
                        for arr in combo
                    )
                    if min_pts < max(arr.shape[0] for combo in packed_models for arr in combo):
                        reporter.log(
                            f"[STEP1][FIX] Inhomogeneous point counts across surveys detected. "
                            f"Trimming all time-step models to {min_pts} points "
                            f"(the minimum across all {n_time} survey epochs and "
                            f"{len(packed_models)} combos)."
                        )
                    packed_models = [
                        [arr[:min_pts] for arr in combo]
                        for combo in packed_models
                    ]
                    packed_depths = [
                        [arr[:min_pts] for arr in combo]
                        for combo in packed_depths
                    ]
                    packed_pstds = [
                        [arr[:min_pts] for arr in combo]
                        for combo in packed_pstds
                    ]
                    REAL_DATA_REFERENCE_XY = None

            models_clean   = np.array(packed_models)
            depths_clean   = np.array(packed_depths)
            # FIX ("use the full MCMC, not just a single selected model"):
            # per-point/per-free-parameter MCMC posterior std, packed
            # through the exact same combo/time/spatial-remap machinery as
            # models_clean/depths_clean above. All-zero for non-MCMC-family
            # inversion methods (emagpy's own convention -- see
            # _invert_single_worker's docstring). Object dtype is used when
            # packing wouldn't otherwise produce a clean rectangular array
            # (e.g. a mix of column counts should never happen within one
            # scenario, since num_layers/free-depths is fixed for the whole
            # run, but np.array(..., dtype=object) is a safe fallback
            # rather than a hard crash if it ever does).
            try:
                pstds_clean = np.array(packed_pstds)
            except ValueError:
                print("[CACHE][WARN] packed_pstds entries have inconsistent shapes "
                      "(unexpected -- num_layers/free-depths should be constant "
                      "within one scenario run). Storing as an object array; "
                      "downstream MCMC-uncertainty reporting will be skipped.")
                pstds_clean = np.array(packed_pstds, dtype=object)
            param_metadata = valid_param_metadata

            print(f"[CACHE] Saving matrix {models_clean.shape} to '{cache_path}' "
                  f"(config fingerprint: {current_fingerprint})")
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "models_clean": models_clean,
                    "param_metadata": param_metadata,
                    "depths_clean": depths_clean,
                    "pstds_clean": pstds_clean,
                    "config_fingerprint": current_fingerprint,
                    "reference_xy": REAL_DATA_REFERENCE_XY,
                }, f)

        # -----------------------------------------------------------------------
        # RECONSTRUCT TASKS FROM param_metadata
        # (tasks is only built inside the no-cache branch; reconstruct it here so
        #  Step 6b diagnostic plots can recover per-realisation inversion settings
        #  regardless of whether the cache was used.)
        # -----------------------------------------------------------------------
        files_to_process_diag = EM_FNAMES if timelapse else [EM_FNAMES]
        fixed_depths_list_diag = (
            [FIXED_DEPTHS] * len(DEPTHS0)
            if isinstance(FIXED_DEPTHS, bool) else list(FIXED_DEPTHS)
        )
        # FIX: same missing-bnds bug as the main Step 1 dispatch above --
        # reconstructed diagnostic-plot inversions must use the same
        # explicit, cond_ranges-derived bounds, or Step 6b's "per
        # realisation" plots would silently re-invert against EMagPy's
        # hardcoded 2-100 mS/m default and disagree with models_clean.
        bnds_list_diag = _build_inversion_bnds(DEPTHS0, fixed_depths_list_diag, COND_RANGES)
        tasks = []
        global_task_idx_diag = 0
        for meta in param_metadata:
            for t_idx_diag, file_path_diag in enumerate(files_to_process_diag):
                invert_kwargs_diag = {
                    "alpha": meta["alpha"],
                    "beta": meta.get("beta", 0.0),
                    "gamma": GAMMA,
                    "method": meta["method"],
                    "forwardModel": meta["forwardModel"],
                    "targetProjection": None,
                    "em_unit": EM_UNIT,
                    "drop_coils": DROP_COILS,
                    "bnds": bnds_list_diag,
                }
                tasks.append((
                    global_task_idx_diag,
                    file_path_diag,
                    DEPTHS0,
                    meta["conds0"],
                    fixed_depths_list_diag,
                    invert_kwargs_diag,
                ))
                global_task_idx_diag += 1

        # -----------------------------------------------------------------------
        # DIMENSION UNPACKING
        # -----------------------------------------------------------------------
        if timelapse:
            ensemble_size, n_time, num_measured_points, num_layers = models_clean.shape
        else:
            ensemble_size, num_measured_points, num_layers = models_clean.shape
            n_time = 1

        # -----------------------------------------------------------------------
        # MCMC POSTERIOR UNCERTAINTY  (FIX: "use the full MCMC, not just a
        # single selected model")
        #   emagpy's MCMC-family methods (MCMC/ROPE/DREAM/SCEUA) sample a
        #   real posterior per measurement point but k.invert() only ever
        #   kept the lowest-misfit sample; the per-point/per-layer posterior
        #   std it also computes (k.pstds) is now captured all the way
        #   through Step 1's caching/packing/spatial-remap (see
        #   _invert_single_worker and pstds_clean above) and summarised
        #   here. This is independent of, and complementary to, the
        #   across-realisation RMSE-weighted CV computed later in Step 5:
        #   that CV quantifies spread ACROSS the param_grid ensemble, this
        #   quantifies spread WITHIN each individual MCMC inversion.
        # -----------------------------------------------------------------------
        _MCMC_FAMILY_METHODS = {"MCMC", "ROPE", "DREAM", "SCEUA"}
        _methods_used = set(PARAM_GRID.get("method", []))
        _mcmc_used = bool(_methods_used & _MCMC_FAMILY_METHODS)

        if _mcmc_used and pstds_clean is not None and pstds_clean.dtype != object:
            try:
                cond_pstd = EMmethod.extract_conductivity_pstd(pstds_clean, num_layers)
                _finite = cond_pstd[np.isfinite(cond_pstd)]
                _nonzero = _finite[_finite > 0]
                pstd_npy_path = os.path.join(OUTPUT_DIR, "mcmc_conductivity_pstd.npy")
                np.save(pstd_npy_path, cond_pstd)
                reporter.log(
                    f"[STEP1] MCMC-family method(s) detected in param_grid "
                    f"({sorted(_methods_used & _MCMC_FAMILY_METHODS)}). Captured "
                    f"per-point/per-layer posterior std (best-10%-of-chain std, "
                    f"emagpy's own k.pstds) for all {ensemble_size} ensemble "
                    f"member(s) -> saved to '{pstd_npy_path}', shape {cond_pstd.shape}."
                )
                if _nonzero.size > 0:
                    reporter.log(
                        f"[STEP1] MCMC posterior std (conductivity, mS/m, cells with "
                        f"std>0 only, n={_nonzero.size}): "
                        f"mean={np.mean(_nonzero):.3f}  median={np.median(_nonzero):.3f}  "
                        f"p90={np.percentile(_nonzero, 90):.3f}  max={np.max(_nonzero):.3f}"
                    )
                    reporter.metric("MCMC posterior std — mean [mS/m]", f"{np.mean(_nonzero):.4f}")
                    reporter.metric("MCMC posterior std — median [mS/m]", f"{np.median(_nonzero):.4f}")
                else:
                    reporter.log(
                        "[STEP1][WARN] MCMC-family method configured, but every "
                        "captured posterior std is zero/NaN. This usually means "
                        "nSample was left too small for the chain to explore "
                        "beyond a single point, or every member actually ran a "
                        "non-MCMC fallback. Check invert_kwargs['options'] "
                        "(nSample) if you were expecting real posterior spread."
                    )
            except Exception as _pstd_exc:
                reporter.log(
                    f"[STEP1][WARN] MCMC posterior-std summary failed "
                    f"(pipeline continues): {type(_pstd_exc).__name__}: {_pstd_exc}"
                )
        elif _mcmc_used:
            reporter.log(
                "[STEP1][INFO] MCMC-family method configured but no usable "
                "pstds_clean is available (likely an older cache — see the "
                "[CACHE][WARN] above). Delete the ensemble cache and re-run "
                "to capture MCMC posterior uncertainty."
            )

        # -----------------------------------------------------------------------
        # EFFECTIVE DEPTHS FOR TDR PROBE HORIZON ROUTING
        #   Fix for: "[WARN] Step 2: probe_layer_indices were derived from the
        #   starting depths0 values ... inspect individual member interface
        #   depths before assigning probes to horizons."
        #
        #   When every interface is fixed (FIXED_DEPTHS is True or all-True),
        #   the starting DEPTHS0 *are* the final interface depths, so probe
        #   routing against DEPTHS0 is already exact -- EFFECTIVE_DEPTHS0 is
        #   just DEPTHS0 and no message fires downstream.
        #
        #   When one or more interfaces were free during Step 1, the horizon
        #   boundaries actually moved (captured per ensemble member and per
        #   measurement point in `depths_clean` / EMagPy's `k.depths[0]`).
        #   Routing TDR probes against the unmoved starting DEPTHS0 is then
        #   only an approximation. Instead of leaving that approximation in
        #   place, take the ensemble- and point-median of the FITTED interface
        #   depths (median is robust to the occasional inversion that pins an
        #   interface at a box-constraint edge) and route probes against that
        #   instead -- directly implementing what the original warning
        #   suggested ("inspect individual member interface depths before
        #   assigning probes to horizons"). DEPTHS0 itself is left untouched
        #   everywhere else (inversion setup, bounds, plot axes, etc.); only
        #   probe-to-horizon routing uses EFFECTIVE_DEPTHS0.
        #
        #   NOTE: real-data time-lapse depths_clean may contain NaN rows for
        #   reference-grid points a given epoch didn't cover (see the
        #   spatial reference-grid matching above); nanmedian is used below
        #   so those cells are simply excluded rather than poisoning the
        #   whole median.
        # -----------------------------------------------------------------------
        _depths_fixed_bool = (
            FIXED_DEPTHS if isinstance(FIXED_DEPTHS, bool) else all(FIXED_DEPTHS)
        )
        PROBE_ROUTING_USES_FITTED_DEPTHS = False
        if _depths_fixed_bool or depths_clean is None:
            EFFECTIVE_DEPTHS0 = DEPTHS0
        else:
            _fitted = np.asarray(depths_clean, dtype=float)
            _avg_axes = tuple(range(_fitted.ndim - 1))  # everything but the interfaces axis
            _fitted_median = np.nanmedian(_fitted, axis=_avg_axes)
            _fitted_sorted = np.sort(_fitted_median)
            if not np.array_equal(_fitted_median, _fitted_sorted):
                reporter.log(
                    "[PIPELINE][WARN] Ensemble-median fitted interface depths came "
                    f"out non-monotonic ({_fitted_median.tolist()}) -- some ensemble "
                    "members likely crossed interfaces during optimisation. Sorting "
                    "before use; treat probe-horizon routing with extra caution for "
                    "this scenario."
                )
            EFFECTIVE_DEPTHS0 = _fitted_sorted.tolist()
            PROBE_ROUTING_USES_FITTED_DEPTHS = True
            reporter.log(
                f"[PIPELINE] Free/partially-free layer interfaces detected "
                f"(fixed_depths={FIXED_DEPTHS}). TDR probe horizon routing will use "
                f"ensemble-median FITTED interface depths {[round(d, 3) for d in EFFECTIVE_DEPTHS0]} "
                f"instead of the static starting depths0={DEPTHS0}."
            )

        # -----------------------------------------------------------------------
        # STEP 1b — REAL-DATA EM DIAGNOSTICS
        #   Common-point footprint + ECa plan-view maps.
        #   Only runs in real-data mode; wrapped in try/except so a plot
        #   failure never aborts the pipeline.
        # -----------------------------------------------------------------------
        if data_source == "real":
            reporter.step("Step 1b", "Real-data EM diagnostic plots")
            reporter.log(" [PIPELINE] STEP 1b: REAL-DATA EM DIAGNOSTICS")
            reporter.log(f"            Common points (min_pts) : {num_measured_points}")
            reporter.log(f"            Raw EM files            : {len(RAW_EM_FNAMES)}")
            try:
                _diag_saved = pediag_real.plot_real_em_diagnostics(
                    raw_em_files=RAW_EM_FNAMES,
                    min_pts=num_measured_points,
                    probe_table=probe_table,
                    output_dir=OUTPUT_DIR,
                    em_unit=EM_UNIT,
                )
                reporter.log(f"[STEP 1b] {len(_diag_saved)} diagnostic figure(s) written "
                             f"to '{os.path.join(OUTPUT_DIR, 'diagnostics')}'")
                reporter.metric("Real-data EM diagnostic figures", len(_diag_saved))
            except Exception as _diag_exc:
                reporter.log(f"[STEP 1b][WARN] Diagnostic plots failed (pipeline continues): "
                             f"{type(_diag_exc).__name__}: {_diag_exc}")
            reporter.end_step("Step 1b")

        # -----------------------------------------------------------------------
        # STEP 1c — REAL-DATA TDR IMPORT DIAGNOSTICS
        #   Raw, unaligned/unaveraged per-station TDR time series + QC flags
        #   (NaN gaps, un-normalized >1.0 values, implausible VWC, stuck/
        #   flat-lined sensor runs). Only runs in real-data mode; wrapped in
        #   try/except so a plot failure never aborts the pipeline.
        # -----------------------------------------------------------------------
        if data_source == "real":
            reporter.step("Step 1c", "Real-data TDR import diagnostic plots")
            reporter.log(" [PIPELINE] STEP 1c: REAL-DATA TDR IMPORT DIAGNOSTICS")
            try:
                _tdr_diag_saved = ptdrdiag.plot_tdr_diagnostics(
                    tdr_dir=tdr_dir,
                    coord_file=coord_file,
                    em_dates=EM_DATES,
                    time_tolerance_days=getattr(cfg, "time_tolerance_days", 1.0),
                    output_dir=OUTPUT_DIR,
                )
                reporter.log(f"[STEP 1c] {len(_tdr_diag_saved)} diagnostic file(s) written "
                             f"to '{os.path.join(OUTPUT_DIR, 'diagnostics')}'")
                reporter.metric("Real-data TDR diagnostic figures", len(_tdr_diag_saved))
            except Exception as _tdr_diag_exc:
                reporter.log(f"[STEP 1c][WARN] TDR diagnostic plots failed (pipeline continues): "
                             f"{type(_tdr_diag_exc).__name__}: {_tdr_diag_exc}")
            reporter.end_step("Step 1c")

        # -----------------------------------------------------------------------
        # STEP 1d — INITIAL MODEL DIAGNOSTICS (raw field inputs, pre-inversion)
        #   Quick ECa (plan-view) and SWC (plan-view) scatter maps built
        #   directly from the raw GEM-2 EM survey CSV and the raw TDR field
        #   data — i.e. the "initial model" the rest of the pipeline starts
        #   from, produced BEFORE any inversion or calibration step runs.
        #   Only runs in real-data mode; wrapped in try/except so a plot
        #   failure never aborts the pipeline.
        # -----------------------------------------------------------------------
        if data_source == "real":
            reporter.step("Step 1d", "Initial model diagnostics (raw field inputs)")
            reporter.log(" [PIPELINE] STEP 1d: INITIAL MODEL DIAGNOSTICS (raw field inputs)")
            try:
                _ini_model_paths = pim.plot_initial_model(
                    em_dir=em_dir,
                    tdr_dir=tdr_dir,
                    coord_file=coord_file,
                    output_dir=OUTPUT_DIR,
                    em_unit=EM_UNIT,
                    em_date_index=0,
                )
                reporter.log(
                    "[STEP 1d] Initial-model plots written -> "
                    f"ECa: {_ini_model_paths['eca_plot']}, "
                    f"SWC: {_ini_model_paths['swc_plot']}, "
                    f"combined: {_ini_model_paths['combined_plot']}"
                )
                reporter.metric("Initial model diagnostic figures", len(_ini_model_paths))
            except Exception as _ini_exc:
                reporter.log(f"[STEP 1d][WARN] Initial model plots failed (pipeline continues): "
                             f"{type(_ini_exc).__name__}: {_ini_exc}")
            reporter.end_step("Step 1d")

        reporter.end_step("Step 0")
        reporter.step("Step 1", "Parallel multi-inversion ensemble generation")

        # ------------------------------------------------------------------
        # Report ensemble matrix with axis-by-axis annotation
        # ------------------------------------------------------------------
        _shape = models_clean.shape
        if timelapse:
            _ens, _nt, _npts, _nlyr = _shape
            _shape_label = (
                f"{_shape}  →  "
                f"(ensemble_size={_ens},  n_time={_nt},  "
                f"n_measured_points={_npts},  n_layers={_nlyr})"
            )
        else:
            _ens, _npts, _nlyr = _shape
            _shape_label = (
                f"{_shape}  →  "
                f"(ensemble_size={_ens},  n_measured_points={_npts},  n_layers={_nlyr})"
            )
        reporter.log(f"[PIPELINE] Ensemble matrix shape : {_shape_label}")
        reporter.log(f"[PIPELINE] Mode                  : {'TIME-LAPSE (' + str(n_time) + ' survey epochs)' if timelapse else 'SINGLE-FILE'}")
        reporter.args_table("Step 1 — Ensemble dimensions", {
            "matrix shape": str(_shape),
            "axis 0 — ensemble_size": f"{_ens} independent inversion realisations",
            **({"axis 1 — n_time": f"{_nt} survey epochs (time-lapse)",
                "axis 2 — n_measured_points": f"{_npts} spatial measurement positions",
                "axis 3 — n_layers": f"{_nlyr} soil horizons per profile"} if timelapse else
               {"axis 1 — n_measured_points": f"{_npts} spatial measurement positions",
                "axis 2 — n_layers": f"{_nlyr} soil horizons per profile"}),
            "param_grid α values": str(PARAM_GRID.get("alpha", [])),
            "param_grid β values (lateral smoothing)": str(PARAM_GRID.get("beta", [0.0])),
            "gamma (temporal smoothing, TL only)": str(GAMMA),
            "param_grid methods": str(PARAM_GRID.get("method", [])),
            "param_grid forward models": str(PARAM_GRID.get("forwardModel", [])),
            "initial conductivity ranges": str(COND_RANGES),
            "fixed depths": str(FIXED_DEPTHS),
            "depth interfaces (DEPTHS0)": str(DEPTHS0),
        })

        # -----------------------------------------------------------------------
        # TEMPERATURE CORRECTION
        #   FIX ("temperature correction is currently a complete no-op"): this
        #   used to hardcode TEMP_PROFILES = TEMP_REF, so the correction
        #   factor 1 + alpha*(T - T_ref) was always exactly 1 -- the step
        #   ran, was logged, and corrected nothing, for every scenario and
        #   every data source. Resolution order:
        #     1. scenario 'soil_temperature_c' override (works for both
        #        synthetic and real data -- explicit, always wins).
        #     2. real data: per-epoch measured temperature auto-detected
        #        from a 'temp'-named TDR column (_extract_temperature_by_epoch,
        #        called inside _load_real_data -> REAL_DATA_DIAG['temp_by_epoch']).
        #        Missing epochs are filled with the mean of whichever epochs
        #        WERE matched (rather than leaving those cells uncorrected
        #        while their neighbours are, which would introduce a
        #        different, undocumented artifact).
        #     3. otherwise: SKIP the correction entirely (identity) and log
        #        that explicitly -- never silently pretend to correct.
        # -----------------------------------------------------------------------
        TEMP_ALPHA = 0.023
        TEMP_REF   = 20.0
        SOIL_TEMPERATURE_C = getattr(cfg, "soil_temperature_c", None) if cfg is not None else None

        TEMP_PROFILES = None
        _temp_source = "none"
        if SOIL_TEMPERATURE_C is not None:
            _temp_val = np.asarray(SOIL_TEMPERATURE_C, dtype=float)
            TEMP_PROFILES = _temp_val if _temp_val.ndim > 0 else float(_temp_val)
            _temp_source = f"scenario override (soil_temperature_c={SOIL_TEMPERATURE_C})"
        elif data_source == "real" and REAL_DATA_DIAG is not None and REAL_DATA_DIAG.get("temp_by_epoch"):
            _temps = REAL_DATA_DIAG["temp_by_epoch"]
            if any(t is not None for t in _temps):
                _known = [t for t in _temps if t is not None]
                _mean_fallback = float(np.mean(_known))
                _resolved = [t if t is not None else _mean_fallback for t in _temps]
                _n_filled = len(_temps) - len(_known)
                if timelapse:
                    TEMP_PROFILES = np.asarray(_resolved, dtype=float).reshape(-1, 1, 1)  # (n_time,1,1)
                else:
                    TEMP_PROFILES = float(_resolved[0])
                _temp_source = (
                    f"measured (TDR 'temp' column, per-epoch mean; "
                    f"{len(_known)}/{len(_temps)} epoch(s) had a direct match"
                    + (f", {_n_filled} filled with the {len(_known)}-epoch mean" if _n_filled else "")
                    + ")"
                )

        if TEMP_PROFILES is None:
            reporter.log(
                "[STEP 1][INFO] Temperature correction SKIPPED (identity -- EC values "
                "are NOT corrected for temperature) -- no measured soil temperature "
                "available and no scenario 'soil_temperature_c' override set. Supply "
                "'soil_temperature_c' in the scenario, or ensure the TDR data includes "
                "a column with 'temp' in its name, to enable this correction."
            )
        else:
            reporter.log(
                f"[STEP 1] Applying temperature correction "
                f"(source: {_temp_source}, alpha={TEMP_ALPHA}, T_ref={TEMP_REF}°C)."
            )
            models_clean = EMmethod.apply_temperature_correction(
                models_clean, TEMP_PROFILES, alpha=TEMP_ALPHA, T_ref=TEMP_REF
            )
        reporter.metric("Temperature correction", _temp_source)

        plot_models = models_clean[:, 0] if timelapse else models_clean
        plot_depths = (depths_clean[:, 0]
                       if (timelapse and depths_clean is not None)
                       else depths_clean)
        EMmethod.plot_step1_ensemble_single_file(
            plot_models, DEPTHS0, OUTPUT_DIR,
            depths_are_fixed=FIXED_DEPTHS if isinstance(FIXED_DEPTHS, bool) else all(FIXED_DEPTHS),
            depths_ensemble=plot_depths,
        )

        # -----------------------------------------------------------------------
        # UTM GRID LOADING  (needed for 2-D probe placement and XY plots)
        # -----------------------------------------------------------------------
        utm_x = utm_y = None
        if mode == "2d":
            utm_x_path = os.path.join(OUTPUT_DIR, "synthetic_utm_x.npy")
            utm_y_path = os.path.join(OUTPUT_DIR, "synthetic_utm_y.npy")
            if os.path.exists(utm_x_path) and os.path.exists(utm_y_path):
                utm_x = np.load(utm_x_path)
                utm_y = np.load(utm_y_path)
                print(f"[PIPELINE] Loaded UTM grid: {len(utm_x)} points  "
                      f"(X {utm_x.min():.1f}\u2013{utm_x.max():.1f}, "
                      f"Y {utm_y.min():.1f}\u2013{utm_y.max():.1f})")
            else:
                print("[PIPELINE][WARN] --mode 2d but utm_x/y .npy not found; "
                      "XY plots and 2-D probe placement skipped. Re-run Step 0.")

        # -----------------------------------------------------------------------
        # SAMPLE POSITION EXTRACTION & GROUND TRUTH SYNTHESIS
        # -----------------------------------------------------------------------
        # FIX (2026-06-27): previously this called
        #   EMmethod.load_survey_positions(EM_FNAMES, ...)
        # passing the FULL list of time-lapse files. Internally that calls
        # emagpy's createTimeLapseSurvey() + trimSurveys(), which aligns all
        # N survey files by COORDINATE MATCHING, not by row count. Real GEM-2
        # field walks essentially never land on bit-identical GPS coordinates
        # across separate survey days, so for real multi-epoch data the
        # coordinate intersection across all 7 surveys silently collapsed to
        # ZERO points -- even though models_clean (trimmed by ROW COUNT a few
        # lines above, the same approach used here) had 23930 valid points.
        # That zero-length position table is what crashed x_transect.min()
        # downstream with "zero-size array to reduction operation minimum".
        #
        # Fix (first pass): read positions from the FIRST survey file only
        # (same file/row order emagpy used when inverting that file in
        # Step 1), and slice to the authoritative num_measured_points
        # already established by the row-count trim above. This avoided
        # emagpy's coordinate-based trimSurveys() entirely for position
        # extraction, but for real time-lapse data still implicitly assumed
        # "spatial index i" meant the same physical location at every
        # epoch.
        #
        # FIX (spatial reference-grid matching, this revision): for real
        # time-lapse runs, Step 1 above now builds a UNION spatial
        # reference grid across ALL epochs (see
        # geometry_utils.match_epochs_to_reference_grid) instead of a
        # row-count trim, and models_clean's spatial axis IS that
        # reference grid. Position extraction here must use the SAME
        # reference-grid coordinates -- reusing only EM_FNAMES[0]'s raw
        # survey positions would silently go back to "index i means
        # epoch-0's own point i", which is no longer what models_clean's
        # spatial axis represents for t > 0. This also means TDR-probe <->
        # EM-point window matching (further below) now operates against a
        # single, epoch-independent coordinate set, instead of implicitly
        # reusing epoch 0's geometry for every epoch's observations.
        # -----------------------------------------------------------------------
        if data_source == "real" and timelapse and REAL_DATA_REFERENCE_XY is not None:
            import pandas as pd

            ref_x, ref_y = REAL_DATA_REFERENCE_XY
            if len(ref_x) != num_measured_points:
                raise ValueError(
                    f"[CRITICAL] Spatial reference-grid point count ({len(ref_x)}) "
                    f"does not match models_clean's spatial axis "
                    f"({num_measured_points}); the ensemble cache or reference-grid "
                    "construction is inconsistent. Delete the ensemble cache "
                    f"('{cache_path}') and re-run."
                )
            survey_df = pd.DataFrame({"x": ref_x, "y": ref_y})
            x_transect = ref_x
            reporter.log(
                f"[PIPELINE][INFO] Real-data time-lapse: using the "
                f"{len(ref_x)}-point spatial reference grid (union of all "
                f"{n_time} survey epochs, see 'spatial reference-grid "
                "matching, not row trimming') for probe placement/matching "
                "and XY plotting, instead of the first epoch's raw survey "
                "positions alone."
            )
        else:
            first_em_fname = EM_FNAMES[0] if timelapse else EM_FNAMES
            survey_df = EMmethod.load_survey_positions(first_em_fname, targetProjection=None,
                                                       em_unit=EM_UNIT)

            if len(survey_df) == 0:
                raise ValueError(
                    f"[CRITICAL] First EM survey file '{first_em_fname}' produced an "
                    "empty position table (no rows). Check the file is readable and "
                    "contains valid 'x'/'y' columns."
                )
            if len(survey_df) < num_measured_points:
                raise ValueError(
                    f"[CRITICAL] First EM survey file '{first_em_fname}' has only "
                    f"{len(survey_df)} positions, fewer than the {num_measured_points} "
                    "points retained in the inverted ensemble (models_clean). "
                    "Positions and models would be misaligned -- aborting rather "
                    "than silently producing spatially incorrect output."
                )
            if len(survey_df) > num_measured_points:
                reporter.log(
                    f"[PIPELINE][INFO] First EM survey file has {len(survey_df)} raw "
                    f"positions; trimming to the {num_measured_points} points kept "
                    "across all combos/time-steps (row-count trim applied earlier "
                    "in Step 1)."
                )

            survey_df  = survey_df.iloc[:num_measured_points].reset_index(drop=True)
            x_transect = survey_df["x"].to_numpy(dtype=float)

        # ------------------------------------------------------------------
        # SWC sensor count and depths — driven by scenario config, with
        # optional CLI-level overrides (num_swc_positions / swc_depths /
        # window_m passed into main(), forwarded from run_pipeline.py's
        # --num-swc-positions / --swc-depths / --window-m flags). Each CLI
        # override takes priority over the scenario field when given
        # (not None); otherwise falls back to the scenario value, or the
        # same defaults as before when there's no scenario either.
        # num_swc_positions : number of horizontal positions
        # swc_depths        : per-layer sensor depths (None → layer centres)
        # window_m          : spatial averaging half-width
        # ------------------------------------------------------------------
        _num_swc_positions_cfg = (
            num_swc_positions if num_swc_positions is not None
            else (getattr(cfg, "num_swc_positions", 4) if cfg is not None else 4)
        )
        NUM_POSITIONS = min(_num_swc_positions_cfg, num_measured_points)

        WINDOW_M = (
            window_m if window_m is not None
            else (getattr(cfg, "window_m", 1.0) if cfg is not None else 1.0)
        )

        _swc_depths_cfg = (
            swc_depths if swc_depths is not None
            else (getattr(cfg, "swc_depths", None) if cfg is not None else None)
        )
        TDR_DEPTH = (
            _swc_depths_cfg
            if _swc_depths_cfg is not None
            else synthetic_geometry["layer_centers"]
        )

        # ------------------------------------------------------------------
        # Probe placement: XY (horizontal) + Z (depth)
        #   - 1-D mode: probes along the X transect only (y=0 for all)
        #   - 2-D mode: probes spread across the UTM XY grid
        # ------------------------------------------------------------------
        if mode == "2d" and utm_x is not None and utm_y is not None:
            # Sample NUM_POSITIONS points spread across the 2-D UTM grid
            step = max(1, len(utm_x) // NUM_POSITIONS)
            sample_indices = list(range(0, len(utm_x), step))[:NUM_POSITIONS]
            UNIQUE_XY_COORDS = [(float(utm_x[i]), float(utm_y[i])) for i in sample_indices]
        else:
            # 1-D transect: distribute along X, fix Y=0
            unique_x = list(np.linspace(x_transect.min(), x_transect.max(), NUM_POSITIONS))
            UNIQUE_XY_COORDS = [(x, 0.0) for x in unique_x]

        # Expand (x, y) × depth to get one probe entry per (position, layer)
        PROBE_X_COORDS = [xy[0] for xy in UNIQUE_XY_COORDS for _ in TDR_DEPTH]
        PROBE_Y_COORDS = [xy[1] for xy in UNIQUE_XY_COORDS for _ in TDR_DEPTH]
        PROBE_DEPTHS   = [d     for _  in UNIQUE_XY_COORDS for d in TDR_DEPTH]

        position_results = EMmethod.extract_sample_positions_from_survey(
            survey_df=survey_df, models=models_clean,
            probe_x_coords=PROBE_X_COORDS,
            probe_y_coords=PROBE_Y_COORDS,   # non-zero in 2D mode → XY distance auto-enabled
            probe_depths=PROBE_DEPTHS,        # <-- Z axis (unchanged)
            # Horizon routing uses EFFECTIVE_DEPTHS0 (== DEPTHS0 when fixed;
            # the ensemble-median FITTED interface depths otherwise -- see
            # "EFFECTIVE DEPTHS FOR TDR PROBE HORIZON ROUTING" above).
            depths0=EFFECTIVE_DEPTHS0, window_m=WINDOW_M,
        )
        sample_positions    = EMmethod.sample_positions_from_results(position_results)
        probe_layer_indices = EMmethod.sample_layer_indices_from_results(position_results)

        if data_source != "real":
            # Report the resolved x/y/z of every synthetic SWC/TDR probe.
            # This is the answer to "why is swc_depths None in the report":
            # the scenario's swc_depths field is None (auto), which the
            # pipeline resolves to synthetic_geometry["layer_centers"]
            # (TDR_DEPTH above) -- this table shows those resolved per-probe
            # depths together with their x/y placement and the EM survey
            # index / soil horizon each one was matched to.
            _probe_rows = []
            for _i, _res in enumerate(position_results):
                _probe_rows.append([
                    _i,
                    f"{_res.probe_x:.3f}",
                    f"{_res.probe_y:.3f}",
                    f"{_res.probe_depth:.3f}",
                    _res.layer_index,
                    _res.representative_index,
                ])
            reporter.table(
                "SWC / TDR Probe Network — resolved x/y/z positions",
                ["Probe #", "x [m]", "y [m]", "z / depth [m]",
                 "Horizon (layer idx)", "Matched EM point idx"],
                _probe_rows,
                note=(
                    f"{len(UNIQUE_XY_COORDS)} horizontal position(s) × "
                    f"{len(TDR_DEPTH)} depth(s) = {len(position_results)} probes. "
                    f"Depths resolved from "
                    f"{'scenario swc_depths' if _swc_depths_cfg is not None else 'layer centres (swc_depths was None/auto)'}"
                    f": {list(TDR_DEPTH)}."
                ),
            )

            # ------------------------------------------------------------
            # SPATIAL MATCHING SUMMARY / GATE (report items 3.1 & 3.2)
            # ------------------------------------------------------------
            matching_summary = EMmethod.summarize_probe_matching(
                position_results,
                warn_fraction_threshold=MATCHING_WARN_FRACTION,
                error_fraction_threshold=MATCHING_ERROR_FRACTION,
            )
            reporter.log(matching_summary.message)
            reporter.args_table("Step 1 — TDR<->EM spatial matching summary", {
                "probes": matching_summary.n_probes,
                "distinct EM points used": matching_summary.n_distinct_em_points,
                "largest group size": matching_summary.largest_group_size,
                "largest group EM index": matching_summary.largest_group_em_index,
                "largest group fraction": f"{matching_summary.largest_group_fraction*100:.1f}%",
                "window-fallback (nearest-point) matches": matching_summary.n_fallback_matches,
                "warn threshold": f"{MATCHING_WARN_FRACTION*100:.1f}%",
                "error threshold": (
                    f"{MATCHING_ERROR_FRACTION*100:.1f}%" if MATCHING_ERROR_FRACTION is not None
                    else "(disabled)"
                ),
                "level": matching_summary.level,
            })
            reporter.metric("Matching — distinct EM points used", matching_summary.n_distinct_em_points)
            reporter.metric("Matching — largest group fraction", f"{matching_summary.largest_group_fraction*100:.1f}%")
            reporter.metric("Matching — window-fallback matches", matching_summary.n_fallback_matches)
            if matching_summary.level == "WARN":
                reporter.log(
                    "[PIPELINE][WARN] TDR<->EM spatial matching is degenerate for a "
                    "large fraction of probes — Step 2/4 calibration for the "
                    "affected horizon(s) may be effectively fit against very few "
                    "distinct EC readings. Consider increasing window_m or "
                    "num_swc_positions, or reviewing the EM survey coverage."
                )
            elif matching_summary.level == "ERROR":
                raise ValueError(
                    "[CRITICAL] TDR<->EM spatial matching failed the configured "
                    f"quality gate: {matching_summary.message} "
                    "Increase window_m, increase EM survey density, or relax "
                    "'matching_error_fraction' in the scenario config if this "
                    "degree of collapse is expected/acceptable for this survey."
                )

            try:
                ppm.plot_probe_matching(
                    survey_df=survey_df,
                    position_results=position_results,
                    output_dir=OUTPUT_DIR,
                    summary=matching_summary,
                    title="SWC/TDR probe -> EM point matching",
                )
            except Exception as _match_plot_exc:
                reporter.log(
                    f"[PIPELINE][WARN] Probe-matching diagnostic plot failed "
                    f"(pipeline continues): {type(_match_plot_exc).__name__}: {_match_plot_exc}"
                )

        if timelapse:
            # Pool ALL n_time synthetic survey epochs per probe, instead of
            # the old round-robin `i % n_time` assignment. Round-robin gave
            # each probe exactly one observation at a single, essentially
            # arbitrary epoch (probe 0 -> epoch 0, probe 1 -> epoch 1, ...,
            # wrapping) and silently discarded the rest of that probe's
            # synthetic time series -- Step 2/4 calibration only ever saw a
            # fraction of the available observations. Every base probe
            # (position x layer) is now expanded across all n_time epochs,
            # mirroring how real TDR data is aggregated across every
            # matched EM survey date in
            # 0_real_data_loader.prepare_real_dataset (which gathers ALL
            # historical readings per probe rather than one snapshot).
            n_probes_base       = len(sample_positions)
            sample_positions    = [pos for pos in sample_positions for _ in range(n_time)]
            probe_layer_indices = [li  for li  in probe_layer_indices for _ in range(n_time)]
            sample_times        = [t for _ in range(n_probes_base) for t in range(n_time)]
        else:
            sample_times = [0] * len(sample_positions)

        if data_source == "real":
            # For real data, observations are already matched to time/position/depth
            # Convert to theta_obs array for downstream processing
            reporter.log(f"[REAL DATA] Preparing observation array from {len(observations_list)} matched obs...")

            # Build lookup: (time_idx, probe_idx, depth) -> swc_value
            theta_obs_list = []
            for obs in observations_list:
                # obs dict has: probe_index, time_index, depth, swc
                theta_obs_list.append(obs['swc'])

            theta_obs = np.array(theta_obs_list)

            # ------------------------------------------------------------
            # BUGFIX: spatially match each real TDR station to the EM
            # survey's measurement-point axis using its actual utm_x/utm_y
            # (from `probe_table`), instead of reusing the synthetic-grid
            # `position_results` computed above from PROBE_X_COORDS (which
            # was built from a synthetic placement, not real station
            # coordinates), and instead of treating
            # `obs['probe_index']` — the station's plain enumeration index
            # (0..n_stations-1) assigned by 0_real_data_loader.py — as if
            # it were already a spatial index into the
            # `num_measured_points` axis of `models_clean`.
            #
            # Left unfixed, every observation for a given station (at any
            # time) reads sigma_b from EM measurement points
            # 0..n_stations-1 regardless of where that station actually
            # sits on the survey transect/grid — collapsing the Step 2
            # EC-vs-theta cross-plot onto whatever conductivity happens to
            # sit at those first few survey indices (this is the dense
            # vertical stack of identical EC values visible in
            # petro_calibration.png).
            #
            # `survey_df` here is the spatial reference grid built above
            # for real time-lapse runs (or the epoch-0/single-survey
            # positions otherwise), so station matching is against the
            # SAME coordinate set models_clean's spatial axis actually
            # represents.
            # ------------------------------------------------------------
            station_position_results = EMmethod.extract_sample_positions_from_survey(
                survey_df=survey_df, models=models_clean,
                probe_x_coords=probe_table['utm_x'].tolist(),
                probe_y_coords=probe_table['utm_y'].tolist(),
                probe_depths=None, depths0=None, window_m=WINDOW_M,
            )
            station_to_spatial_idx = EMmethod.sample_positions_from_results(station_position_results)
            reporter.log(
                f"[REAL DATA] Spatially matched {len(station_to_spatial_idx)} station(s) "
                f"to EM survey indices: {station_to_spatial_idx}"
            )

            # ------------------------------------------------------------
            # SPATIAL MATCHING SUMMARY / GATE (report items 3.1 & 3.2)
            # Real-data stations are the case that motivated this: many
            # stations can collapse onto very few EM indices (e.g. 11/20
            # onto one point), which previously was only visible by
            # counting duplicates in the log line above.
            # ------------------------------------------------------------
            matching_summary = EMmethod.summarize_probe_matching(
                station_position_results,
                warn_fraction_threshold=MATCHING_WARN_FRACTION,
                error_fraction_threshold=MATCHING_ERROR_FRACTION,
            )
            reporter.log(matching_summary.message)
            reporter.args_table("Step 1 — TDR<->EM spatial matching summary (real stations)", {
                "stations": matching_summary.n_probes,
                "distinct EM points used": matching_summary.n_distinct_em_points,
                "largest group size": matching_summary.largest_group_size,
                "largest group EM index": matching_summary.largest_group_em_index,
                "largest group fraction": f"{matching_summary.largest_group_fraction*100:.1f}%",
                "window-fallback (nearest-point) matches": matching_summary.n_fallback_matches,
                "warn threshold": f"{MATCHING_WARN_FRACTION*100:.1f}%",
                "error threshold": (
                    f"{MATCHING_ERROR_FRACTION*100:.1f}%" if MATCHING_ERROR_FRACTION is not None
                    else "(disabled)"
                ),
                "level": matching_summary.level,
            })
            reporter.metric("Matching — distinct EM points used (stations)", matching_summary.n_distinct_em_points)
            reporter.metric("Matching — largest group fraction (stations)", f"{matching_summary.largest_group_fraction*100:.1f}%")
            reporter.metric("Matching — window-fallback matches (stations)", matching_summary.n_fallback_matches)
            if matching_summary.level == "WARN":
                reporter.log(
                    "[PIPELINE][WARN] Real TDR station <-> EM spatial matching is "
                    "degenerate for a large fraction of stations — Step 2/4 "
                    "calibration for the affected horizon(s) may be effectively "
                    "fit against very few distinct EC readings. Consider "
                    "increasing window_m or reviewing whether the EM survey "
                    "actually covers all station locations."
                )
            elif matching_summary.level == "ERROR":
                raise ValueError(
                    "[CRITICAL] Real TDR station <-> EM spatial matching failed the "
                    f"configured quality gate: {matching_summary.message} "
                    "Increase window_m, improve EM survey coverage, or relax "
                    "'matching_error_fraction' in the scenario config if this "
                    "degree of collapse is expected/acceptable for this survey."
                )

            try:
                ppm.plot_probe_matching(
                    survey_df=survey_df,
                    position_results=station_position_results,
                    output_dir=OUTPUT_DIR,
                    summary=matching_summary,
                    title="Real TDR station -> EM point matching",
                    filename="probe_em_matching_real_stations.png",
                )
            except Exception as _match_plot_exc:
                reporter.log(
                    f"[PIPELINE][WARN] Probe-matching diagnostic plot failed "
                    f"(pipeline continues): {type(_match_plot_exc).__name__}: {_match_plot_exc}"
                )

            # For real data mode, extract sample metadata from observations,
            # routing each observation's station-ordinal probe_index through
            # the station -> EM-spatial-index map computed above.
            sample_positions = np.array([
                station_to_spatial_idx[obs['probe_index']] for obs in observations_list
            ])
            sample_times = np.array([obs['time_index'] for obs in observations_list])

            # Map each probe's installation depth to its 0-based soil-horizon
            # index using EFFECTIVE_DEPTHS0 (== DEPTHS0 when fixed; the
            # ensemble-median FITTED interface depths otherwise -- see
            # "EFFECTIVE DEPTHS FOR TDR PROBE HORIZON ROUTING" above),
            # mirroring the synthetic-data path.
            probe_layer_indices = np.array([
                EMmethod.layer_index_from_depth(obs['depth'], EFFECTIVE_DEPTHS0)
                for obs in observations_list
            ])

            # Report the resolved x/y position of every real TDR station,
            # together with the depths actually observed at each one and
            # the EM survey index each was spatially matched to. Real-data
            # probes don't have a single fixed z like the synthetic grid --
            # each station carries whatever sensor depths its logger
            # reports (e.g. 15cm/30cm) -- so z is listed per-station as the
            # set of observed depths rather than one row per (position, layer).
            _station_depths: dict[int, set] = {}
            for _obs in observations_list:
                _station_depths.setdefault(_obs['probe_index'], set()).add(_obs['depth'])
            _probe_rows = []
            for _, _row in probe_table.iterrows():
                _p_idx = int(_row['probe_index'])
                _depths_here = sorted(_station_depths.get(_p_idx, set()))
                _probe_rows.append([
                    _p_idx,
                    _row['station_name'],
                    f"{_row['utm_x']:.2f}",
                    f"{_row['utm_y']:.2f}",
                    ", ".join(f"{d:.3f}" for d in _depths_here) or "(no matched obs)",
                    station_to_spatial_idx[_p_idx] if _p_idx < len(station_to_spatial_idx) else "—",
                ])
            reporter.table(
                "SWC / TDR Probe Network — resolved x/y positions (real stations)",
                ["Probe #", "Station", "utm_x [m]", "utm_y [m]",
                 "Observed z / depth(s) [m]", "Matched EM point idx"],
                _probe_rows,
                note=(
                    f"{len(probe_table)} real TDR station(s); depths come from each "
                    f"station's own logger columns, not a fixed swc_depths list."
                ),
            )

            reporter.log(f"[REAL DATA] Built observation array: {theta_obs.shape}")
        else:
            # SYNTHETIC DATA: original construction with noise
            np.random.seed(42)
            theta_obs = np.array([
                theta_true_realization[t_idx, pos, layer] + np.random.normal(0, TDR_NOISE_STD)
                for t_idx, pos, layer in zip(sample_times, sample_positions, probe_layer_indices)
            ])

        # -----------------------------------------------------------------------
        # STEP 2 — WAXMAN-SMITS PETROPHYSICAL CALIBRATION
        # -----------------------------------------------------------------------
        reporter.end_step("Step 1")

        reporter.step("Step 2", "Waxman-Smits petrophysical calibration")
        reporter.log("\n" + "=" * 80)
        reporter.log(" [PIPELINE] STEP 2: WAXMAN-SMITS PETROPHYSICAL CALIBRATION")
        reporter.log("=" * 80)

        # FIX ("TDR measurement uncertainty is collected but never used to
        # weight the calibration"): only pass a genuine positive std through
        # -- TDR_NOISE_STD is deliberately forced to 0.0 for real-data runs
        # (this pipeline has no real per-probe noise estimate there yet),
        # and 1/0^2 weighting would be nonsensical/unstable. None disables
        # weighting entirely inside calibrate_petrophysical_relationship,
        # i.e. no behaviour change for real-data runs from before this fix.
        _theta_obs_std_step2 = TDR_NOISE_STD if TDR_NOISE_STD and TDR_NOISE_STD > 0 else None
        reporter.log(
            f"[STEP 2] TDR observation weighting: "
            + (f"ENABLED (theta_obs_std={_theta_obs_std_step2})"
               if _theta_obs_std_step2 is not None else
               "disabled (no positive tdr_noise_std available for this run)")
        )

        fits_step2 = step2.run_petro(
            models=models_clean, sample_positions=sample_positions,
            sample_times=sample_times, theta_obs=theta_obs,
            depths0=DEPTHS0, output_dir=OUTPUT_DIR,
            probe_layer_indices=probe_layer_indices,
            depths_are_fixed=FIXED_DEPTHS if isinstance(FIXED_DEPTHS, bool) else all(FIXED_DEPTHS),
            fixed_params=WS_FIXED_PARAMS,
            probe_depths_reflect_fitted_interfaces=PROBE_ROUTING_USES_FITTED_DEPTHS,
            lab_p0=LAB_P0,
            theta_obs_std=_theta_obs_std_step2,
        )

        # ------------------------------------------------------------------
        # Report Step 2 calibration details
        # ------------------------------------------------------------------
        _n_horizons = len(fits_step2[0].horizon_fits) if fits_step2 else 0
        _all_rmse   = [f.rmse for f in fits_step2]
        reporter.log(
            f"[STEP 2] Calibrated Waxman-Smits fits: {len(fits_step2)} "
            f"(one per ensemble member, {_n_horizons} horizons each)"
        )
        reporter.log(
            f"[STEP 2] Ensemble RMSE — mean: {np.nanmean(_all_rmse):.5f}  "
            f"std: {np.nanstd(_all_rmse):.5f}  "
            f"min: {np.nanmin(_all_rmse):.5f}  "
            f"max: {np.nanmax(_all_rmse):.5f}  [m³/m³]"
        )

        # Count SWC sensors per horizon
        _probe_layer_arr = np.array(probe_layer_indices) if probe_layer_indices is not None else None
        reporter.log(
            f"[STEP 2] SWC sensors used for calibration: {len(theta_obs)} total observations "
            f"across {NUM_POSITIONS} spatial positions × {n_time} time step(s)"
        )
        if _probe_layer_arr is not None:
            for _hz in range(_n_horizons):
                _count = int(np.sum(_probe_layer_arr == _hz))
                reporter.log(f"[STEP 2]   Horizon {_hz}: {_count} sensor-epoch pairs")

        # Per-horizon WS parameters for the best-ranked fit
        _best = fits_step2[0]
        _hz_rows = {}
        for _hz in _best.horizon_fits:
            if _hz.is_calibrated:
                _hz_rows[f"Hz {_hz.horizon} — a / m / σ_w / σ_s / RMSE"] = (
                    f"{_hz.a:.4f} / {_hz.m:.4f} / "
                    f"{_hz.sigma_w:.2f} mS/m / {_hz.sigma_s:.2f} mS/m / "
                    f"{_hz.rmse:.5f} m³/m³"
                )
            else:
                _hz_rows[f"Hz {_hz.horizon} — a / m / σ_w / σ_s / RMSE"] = (
                    f"UNCALIBRATED — no TDR probe in this horizon; showing prior "
                    f"p0 only: {_hz.a:.4f} / {_hz.m:.4f} / "
                    f"{_hz.sigma_w:.2f} mS/m / {_hz.sigma_s:.2f} mS/m / RMSE=N/A"
                )
        _hz_rows["n_obs supporting best-fit"] = len(_best.observed_theta)
        reporter.args_table("Step 2 — Best-ranked Waxman-Smits fit (rank-1 ensemble member)", _hz_rows)

        # Ensemble-wide per-horizon parameter statistics
        _ws_table: dict[str, str] = {}
        for _hz_idx in range(_n_horizons):
            _hzs        = [f.horizon_fits[_hz_idx] for f in fits_step2 if _hz_idx < len(f.horizon_fits)]
            _n_calib    = sum(1 for h in _hzs if h.is_calibrated)
            _a_vals     = [h.a       for h in _hzs]
            _m_vals     = [h.m       for h in _hzs]
            _sw_vals    = [h.sigma_w for h in _hzs]
            _ss_vals    = [h.sigma_s for h in _hzs]
            _rmse_vals  = [h.rmse    for h in _hzs]  # NaN for uncalibrated members
            _ws_table[f"Hz {_hz_idx} — calibrated members"] = f"{_n_calib} / {len(_hzs)}"
            _ws_table[f"Hz {_hz_idx} — a (mean±std)"]      = f"{np.mean(_a_vals):.4f} ± {np.std(_a_vals):.4f}"
            _ws_table[f"Hz {_hz_idx} — m (mean±std)"]      = f"{np.mean(_m_vals):.4f} ± {np.std(_m_vals):.4f}"
            _ws_table[f"Hz {_hz_idx} — σ_w (mean±std)"]   = f"{np.mean(_sw_vals):.2f} ± {np.std(_sw_vals):.2f} mS/m"
            _ws_table[f"Hz {_hz_idx} — σ_s (mean±std)"]   = f"{np.mean(_ss_vals):.2f} ± {np.std(_ss_vals):.2f} mS/m"
            if _n_calib == 0:
                _ws_table[f"Hz {_hz_idx} — RMSE (mean±std)"] = "N/A — horizon uncalibrated (no TDR probe coverage)"
            else:
                _ws_table[f"Hz {_hz_idx} — RMSE (mean±std)"] = (
                    f"{np.nanmean(_rmse_vals):.5f} ± {np.nanstd(_rmse_vals):.5f} m³/m³"
                )
        reporter.args_table("Step 2 — Ensemble Waxman-Smits parameter statistics (all members)", _ws_table)

        step2.plot_ws_parameter_histograms(
            fits=fits_step2, best_fit=fits_step2[0], depths0=DEPTHS0, lab_p0=LAB_P0,
            output_path=os.path.join(OUTPUT_DIR, "ws_parameter_histograms.png"),
        )

        # -----------------------------------------------------------------------
        # STEP 3 — RMSE-WEIGHTED AVERAGING
        # -----------------------------------------------------------------------
        reporter.end_step("Step 2")

        reporter.step("Step 3", "RMSE-weighted ranking-averaging engine")
        reporter.log("\n" + "=" * 80)
        reporter.log(" [PIPELINE] STEP 3: RMSE-WEIGHTED RANKING-AVERAGING ENGINE")
        reporter.log("=" * 80)

        averaged_models, ranking = step3.run_average(
            models=models_clean, fits_step2=fits_step2,
            depths0=DEPTHS0, output_dir=OUTPUT_DIR,
        )

        # ------------------------------------------------------------------
        # Report Step 3 averaging details
        # ------------------------------------------------------------------
        _ranked_rmse = [fits_step2[i].rmse for i in ranking]
        reporter.log(
            f"[STEP 3] Ranked {len(ranking)} ensemble members by RMSE (ascending)."
        )
        reporter.log(
            f"[STEP 3] Best  member  : index {ranking[0]},  RMSE = {_ranked_rmse[0]:.5f} m³/m³"
        )
        reporter.log(
            f"[STEP 3] Worst member  : index {ranking[-1]}, RMSE = {_ranked_rmse[-1]:.5f} m³/m³"
        )
        reporter.log(
            f"[STEP 3] Cumulative averaged models produced: {len(averaged_models)} "
            f"(one averaged stack per K from 1 to {len(averaged_models)})"
        )
        if timelapse:
            _avg_shape = averaged_models[0].shape
            reporter.log(
                f"[STEP 3] Each averaged_model shape: {_avg_shape}  →  "
                f"(n_time={_avg_shape[0]}, n_measured_points={_avg_shape[1]}, n_layers={_avg_shape[2]})"
            )
        else:
            _avg_shape = averaged_models[0].shape
            reporter.log(
                f"[STEP 3] Each averaged_model shape: {_avg_shape}  →  "
                f"(n_measured_points={_avg_shape[0]}, n_layers={_avg_shape[1]})"
            )

        # -----------------------------------------------------------------------
        # STEP 4 — RECALIBRATE PETROPHYSICAL RELATIONSHIPS
        # -----------------------------------------------------------------------
        reporter.end_step("Step 3")

        reporter.step("Step 4", "Recalibrate petrophysical relationships")
        reporter.log("\n" + "=" * 80)
        reporter.log(" [PIPELINE] STEP 4: RECALIBRATE PETROPHYSICAL RELATIONSHIPS")
        reporter.log("=" * 80)

        fits_step4 = EMmethod.recalibrate_petrophysical_relationship(
            averaged_models=averaged_models, sample_positions=sample_positions,
            sample_times=sample_times, theta_obs=theta_obs,
            depths0=DEPTHS0, probe_layer_indices=probe_layer_indices,
            depths_are_fixed=FIXED_DEPTHS if isinstance(FIXED_DEPTHS, bool) else all(FIXED_DEPTHS),
            fixed_params=WS_FIXED_PARAMS,
            probe_depths_reflect_fitted_interfaces=PROBE_ROUTING_USES_FITTED_DEPTHS,
            lab_p0=LAB_P0,
            theta_obs_std=_theta_obs_std_step2,  # same resolved value used in Step 2, see fix note there
        )

        # ------------------------------------------------------------------
        # Report Step 4 recalibration details
        # ------------------------------------------------------------------
        _n_fits4   = len(fits_step4)
        _rmse4_all = [f.rmse for f in fits_step4]
        reporter.log(
            f"[STEP 4] Recalibrated {_n_fits4} Waxman-Smits fits "
            f"(one per cumulative averaged model K=1…{_n_fits4})."
        )
        reporter.log(
            f"[STEP 4] RMSE across K — mean: {np.nanmean(_rmse4_all):.5f}  "
            f"std: {np.nanstd(_rmse4_all):.5f}  "
            f"min: {np.nanmin(_rmse4_all):.5f} (best K)  "
            f"max: {np.nanmax(_rmse4_all):.5f}  [m³/m³]"
        )
        _best4 = fits_step4[int(np.nanargmin(_rmse4_all))]
        _hz4_rows: dict[str, str] = {}
        for _hz in _best4.horizon_fits:
            if _hz.is_calibrated:
                _hz4_rows[f"Hz {_hz.horizon} — a / m / σ_w / σ_s / RMSE"] = (
                    f"{_hz.a:.4f} / {_hz.m:.4f} / "
                    f"{_hz.sigma_w:.2f} mS/m / {_hz.sigma_s:.2f} mS/m / "
                    f"{_hz.rmse:.5f} m³/m³"
                )
            else:
                _hz4_rows[f"Hz {_hz.horizon} — a / m / σ_w / σ_s / RMSE"] = (
                    f"UNCALIBRATED — no TDR probe in this horizon; showing prior "
                    f"p0 only: {_hz.a:.4f} / {_hz.m:.4f} / "
                    f"{_hz.sigma_w:.2f} mS/m / {_hz.sigma_s:.2f} mS/m / RMSE=N/A"
                )
        reporter.args_table(
            f"Step 4 — Best recalibrated fit (K={int(np.nanargmin(_rmse4_all))+1}, "
            f"RMSE={np.nanmin(_rmse4_all):.5f} m³/m³)",
            _hz4_rows,
        )

        # -----------------------------------------------------------------------
        # STEP 5 — CONVERGENCE SELECTION & RELIABILITY MAPPING
        # -----------------------------------------------------------------------
        reporter.end_step("Step 4")

        reporter.step("Step 5", "Convergence selection & reliability mapping")
        reporter.log("\n" + "=" * 80)
        reporter.log(" [PIPELINE] STEP 5: CONVERGENCE SELECTION & RELIABILITY MAPPING")
        reporter.log("=" * 80)

        selected_output = step4_5.run_select_assess(
            models=models_clean, averaged_models=averaged_models, ranking=ranking,
            sample_positions=sample_positions, sample_times=sample_times,
            theta_obs=theta_obs, fits_step2=fits_step4,
            depths0=DEPTHS0, output_dir=OUTPUT_DIR, generate_plots=True,
            probe_layer_indices=probe_layer_indices,
        )

        # ------------------------------------------------------------------
        # Report Step 5 selection and uncertainty details
        # ------------------------------------------------------------------
        _cv = selected_output.cv_map
        _tc = selected_output.theta_calc
        reporter.log(
            f"[STEP 5] Optimal stack depth K = {selected_output.K}  "
            f"(out of {len(averaged_models)} candidates)"
        )
        reporter.log(f"[STEP 5] Minimum RMSE at K={selected_output.K}: {selected_output.rmse:.6f} m³/m³")

        # CV map
        if _cv.ndim == 3:
            reporter.log(
                f"[STEP 5] CV map shape : {_cv.shape}  →  "
                f"(n_time={_cv.shape[0]}, n_measured_points={_cv.shape[1]}, n_layers={_cv.shape[2]})  "
                f"— Coefficient of Variation = weighted_std / weighted_mean per cell; "
                f">1.0 indicates >100 % relative uncertainty"
            )
        elif _cv.ndim == 2:
            reporter.log(
                f"[STEP 5] CV map shape : {_cv.shape}  →  "
                f"(n_measured_points={_cv.shape[0]}, n_layers={_cv.shape[1]})  "
                f"— Coefficient of Variation = weighted_std / weighted_mean per cell"
            )
        else:
            reporter.log(f"[STEP 5] CV map shape : {_cv.shape}")

        reporter.log(
            f"[STEP 5] CV map stats — mean: {np.nanmean(_cv):.4f}  "
            f"max: {np.nanmax(_cv):.4f}  "
            f"fraction of cells with CV>0.5: "
            f"{100*np.nanmean(_cv > 0.5):.1f} %  "
            f"(NaN cells, e.g. unmatched real-data time-lapse reference-grid "
            f"points, excluded from these statistics: "
            f"{100*np.mean(np.isnan(_cv)):.1f} % of cells are NaN)"
        )

        # theta_calc
        if _tc is not None:
            if _tc.ndim == 3:
                reporter.log(
                    f"[STEP 5] theta_calc shape : {_tc.shape}  →  "
                    f"(n_time={_tc.shape[0]}, n_measured_points={_tc.shape[1]}, n_layers={_tc.shape[2]})  "
                    f"— predicted VWC [m³/m³] grid at optimal K, derived from EC→θ "
                    f"via per-horizon Waxman-Smits inversion"
                )
            else:
                reporter.log(f"[STEP 5] theta_calc shape : {_tc.shape}")
            reporter.log(
                f"[STEP 5] theta_calc stats — mean: {np.nanmean(_tc):.4f}  "
                f"min: {np.nanmin(_tc):.4f}  "
                f"max: {np.nanmax(_tc):.4f}  [m³/m³]"
            )

        # -----------------------------------------------------------------------
        # POST-STEP-5 QUALITY GATE  (report item 4, quality gates)
        #   Turns the CV map and per-horizon calibration status -- already
        #   computed above but previously only ever printed/plotted -- into
        #   an explicit pass/warn verdict. Never aborts the run; only
        #   downgrades the final reported status (see reporter.finish()
        #   below) so a degraded-but-completed run is visibly distinguishable
        #   from a clean one instead of both reporting "✅ SUCCESS".
        # -----------------------------------------------------------------------
        quality_gate_result = EMmethod.evaluate_quality_gates(
            cv_map=selected_output.cv_map,
            best_fit=selected_output.best_fit,
            cv_mean_warn_threshold=(
                getattr(cfg, "quality_gate_cv_mean_warn", 0.3) if cfg is not None else 0.3
            ),
            cv_high_threshold=(
                getattr(cfg, "quality_gate_cv_high_threshold", 1.0) if cfg is not None else 1.0
            ),
            cv_high_fraction_warn_threshold=(
                getattr(cfg, "quality_gate_cv_high_fraction_warn", 0.3) if cfg is not None else 0.3
            ),
            require_all_horizons_calibrated=(
                getattr(cfg, "quality_gate_require_all_horizons_calibrated", True)
                if cfg is not None else True
            ),
        )
        reporter.log(quality_gate_result.message)
        reporter.metric("Quality gate level", quality_gate_result.level)
        reporter.metric("Quality gate — mean CV", f"{quality_gate_result.cv_mean:.4f}")
        reporter.metric(
            "Quality gate — CV>threshold fraction",
            f"{quality_gate_result.cv_high_fraction*100:.1f}%",
        )
        reporter.metric(
            "Quality gate — calibrated horizons",
            f"{quality_gate_result.n_calibrated_horizons}/{quality_gate_result.n_total_horizons}",
        )
        if quality_gate_result.level == "WARN":
            reporter.log(
                "[PIPELINE][WARN] Final solution failed one or more quality gates -- "
                "see 'Quality gate reasons' in the report summary. The run still "
                "completed and all outputs were written; treat the solution with "
                "extra caution before relying on it."
            )

        # -----------------------------------------------------------------------
        # STEP 6 — WATER CONTENT VISUALISATIONS  (XZ always; XY in 2-D mode)
        # -----------------------------------------------------------------------
        reporter.end_step("Step 5")

        reporter.step("Step 6", "Rendering water content visualizations")
        reporter.log("\n" + "=" * 80)
        reporter.log(" [PIPELINE] STEP 6: RENDERING WATER CONTENT VISUALIZATIONS")
        reporter.log(f"            XZ cross-section : always")
        reporter.log(f"            XY plan view      : {'YES (2-D mode)' if mode == '2d' else 'NO (1-D mode)'}")
        reporter.log("=" * 80)

        pwc.plot_from_pipeline(
            selected_output=selected_output,
            x_transect=x_transect,
            depths0=DEPTHS0,
            probe_x_coords=PROBE_X_COORDS,
            probe_y_coords=PROBE_Y_COORDS,   # <-- new: needed for XY plan-view probe markers
            probe_depths=PROBE_DEPTHS,        # Z axis (unchanged)
            output_dir=OUTPUT_DIR,
            scenario_name=cfg.name if cfg else "default",
            timelapse=timelapse,
            survey_labels=None,
            utm_x=utm_x,
            utm_y=utm_y,
        )

        # Ground-truth counterpart of the plot above (water_content_xz_<scenario>_truth.png
        # / water_content_xy_<scenario>_truth.png) -- only available in synthetic mode,
        # where theta_true_realization is the known ground truth used to generate the
        # synthetic surveys. No truth exists for real field data, so this is skipped there.
        if data_source == "synthetic" and theta_true_realization is not None:
            try:
                pwc.plot_truth_from_pipeline(
                    theta_true=theta_true_realization,
                    x_transect=x_transect,
                    depths0=DEPTHS0,
                    probe_x_coords=PROBE_X_COORDS,
                    probe_y_coords=PROBE_Y_COORDS,
                    probe_depths=PROBE_DEPTHS,
                    output_dir=OUTPUT_DIR,
                    scenario_name=cfg.name if cfg else "default",
                    timelapse=timelapse,
                    survey_labels=None,
                    utm_x=utm_x,
                    utm_y=utm_y,
                )
                reporter.log(
                    f"[STEP 6] Ground-truth XZ (and XY, if 2-D) water-content "
                    f"plots -> {OUTPUT_DIR}/water_content_xz_"
                    f"{(cfg.name if cfg else 'default')}_truth.png"
                )
            except Exception as e:
                reporter.log(f"[STEP 6][WARN] plot_truth_from_pipeline failed: {e}")

            # % -difference between the calculated and ground-truth XZ
            # sections above -- (theta_calc - theta_true) / theta_true * 100,
            # signed (over- vs. under-estimation), masked where truth ≈ 0.
            try:
                pctdiff_path = pwc.plot_pct_diff_from_pipeline(
                    selected_output=selected_output,
                    theta_true=theta_true_realization,
                    x_transect=x_transect,
                    depths0=DEPTHS0,
                    probe_x_coords=PROBE_X_COORDS,
                    probe_depths=PROBE_DEPTHS,
                    output_dir=OUTPUT_DIR,
                    scenario_name=cfg.name if cfg else "default",
                    timelapse=timelapse,
                    survey_labels=None,
                )
                if pctdiff_path is not None:
                    reporter.log(f"[STEP 6] Calculated-vs-truth %% difference -> {pctdiff_path}")
            except Exception as e:
                reporter.log(f"[STEP 6][WARN] plot_pct_diff_from_pipeline failed: {e}")

            # Initial model (starting guess, conds0's layer MIN/MAX bounds) vs.
            # ground truth -- a QC check for whether the inversion's starting
            # point plausibly brackets reality, BEFORE any inversion runs.
            # Two flavours: depth-resolved EC_bulk (XZ heatmap, same style as
            # water_content_xz_*.png) and per-coil apparent conductivity (ECa),
            # which isn't depth-resolved so gets its own per-coil panel style.
            try:
                _ec_true_path = os.path.join(OUTPUT_DIR, "synthetic_time_lapse_ec.npy")
                _eca_true_path = os.path.join(OUTPUT_DIR, "synthetic_time_lapse_eca.npy")

                if os.path.exists(_ec_true_path):
                    _ec_bulk_true = np.load(_ec_true_path)[REALIZATION_INDEX]
                    ini_ec_path = pwc.plot_initial_model_vs_true_ec(
                        cond_ranges=COND_RANGES,
                        depths0=DEPTHS0,
                        x_coords=x_transect,
                        ec_bulk_true=_ec_bulk_true,
                        output_dir=OUTPUT_DIR,
                        scenario_name=cfg.name if cfg else "default",
                        probe_x=PROBE_X_COORDS,
                        probe_depths=PROBE_DEPTHS,
                    )
                    reporter.log(f"[STEP 6] Initial model vs. true EC_bulk -> {ini_ec_path}")
                else:
                    reporter.log(
                        "[STEP 6][WARN] 'synthetic_time_lapse_ec.npy' not found; "
                        "skipping initial-model-vs-true-EC_bulk plot."
                    )

                if os.path.exists(_eca_true_path):
                    _eca_true = np.load(_eca_true_path)[REALIZATION_INDEX]
                    ini_eca_path = pwc.plot_initial_eca_vs_true_eca(
                        cond_ranges=COND_RANGES,
                        depths0=DEPTHS0,
                        x_coords=x_transect,
                        eca_true=_eca_true,
                        coils=step0.COILS_LIST,
                        output_dir=OUTPUT_DIR,
                        scenario_name=cfg.name if cfg else "default",
                        forward_model=PARAM_GRID.get("forwardModel", ["FSlin"])[0],
                    )
                    if ini_eca_path is not None:
                        reporter.log(f"[STEP 6] Initial model vs. true ECa -> {ini_eca_path}")
                else:
                    reporter.log(
                        "[STEP 6][WARN] 'synthetic_time_lapse_eca.npy' not found; "
                        "skipping initial-model-vs-true-ECa plot."
                    )
            except Exception as e:
                reporter.log(f"[STEP 6][WARN] Initial-model-vs-truth diagnostics failed: {e}")

        # -----------------------------------------------------------------------
        # SUMMARY
        # -----------------------------------------------------------------------

        # -----------------------------------------------------------------------
        # STEP 6b — emagpy NATIVE DIAGNOSTIC PLOTS
        #   Two panels:
        #     (a) per_realisation — raw inversion for a sample of ensemble members
        #     (b) optimal_model   — weighted-average of top-K realisations (Step 4/5)
        # -----------------------------------------------------------------------
        reporter.step("Step 6b", "emagpy native diagnostic plots")
        reporter.log(" [PIPELINE] STEP 6b: emagpy DIAGNOSTIC PLOTS")
        reporter.log(f"            surveys        : {len(EM_FNAMES)}")
        reporter.log(f"            mode           : {mode.upper()}")
        reporter.log(f"            optimal K      : {selected_output.K}")
        reporter.log(f"            ensemble size  : {models_clean.shape[0]}")

        _fixed_bool = FIXED_DEPTHS if isinstance(FIXED_DEPTHS, bool) else all(FIXED_DEPTHS)
        _fixed_depths_list_6b = (
            [FIXED_DEPTHS] * len(DEPTHS0)
            if isinstance(FIXED_DEPTHS, bool) else list(FIXED_DEPTHS)
        )
        # FIX: same missing-bnds bug -- the "optimal model" re-inversion used
        # for Step 6b plotting must also see cond_ranges-derived bounds,
        # not EMagPy's hardcoded 2-100 mS/m default.
        _base_inv_kwargs = {
            "alpha":        PARAM_GRID.get("alpha",        [0.03])[0],
            "beta":         PARAM_GRID.get("beta",         [0.0])[0],
            "gamma":        GAMMA,
            "method":       PARAM_GRID.get("method",       ["L-BFGS-B"])[0],
            "forwardModel": PARAM_GRID.get("forwardModel", ["FSlin"])[0],
            "bnds":         _build_inversion_bnds(DEPTHS0, _fixed_depths_list_6b, COND_RANGES),
        }

        # -----------------------------------------------------------------------
        # STEP 6c — SWC PLOTS FOR ALL THREE PIPELINE STAGES
        #   Stage A: sample of raw realisations (models_clean)
        #   Stage B: averaged model at K=1, K_opt, K_max (averaged_models)
        #   Stage C: optimal model (selected_output) — same as Step 6 above
        # -----------------------------------------------------------------------
        reporter.step("Step 6c", "SWC plots for all pipeline stages")
        reporter.log(" [PIPELINE] STEP 6c: SWC STAGE PLOTS (A/B/C)")

        pwc.plot_swc_ensemble_stages(
            models_clean     = models_clean,
            averaged_models  = averaged_models,
            selected_output  = selected_output,
            best_fit         = fits_step4[selected_output.K - 1],
            x_transect       = x_transect,
            depths0          = DEPTHS0,
            output_dir       = OUTPUT_DIR,
            scenario_name    = cfg.name if cfg else "default",
            timelapse        = timelapse,
            survey_labels    = None,
            utm_x            = utm_x,
            utm_y            = utm_y,
            probe_x_coords   = PROBE_X_COORDS,
            probe_y_coords   = PROBE_Y_COORDS,
            probe_depths     = PROBE_DEPTHS,
            max_realisations = 4,
        )

        # -----------------------------------------------------------------------
        # STEP 6c-bis — 1:1 validation scatter (Stages A / B / C vs. TDR truth)
        # Called explicitly here so that the full sample_positions / sample_times /
        # probe_layer_indices vectors — which are available in this scope but are
        # not stored inside selected_output — are passed directly instead of
        # relying on the fallback reconstruction inside plot_swc_ensemble_stages.
        # -----------------------------------------------------------------------
        pwc.plot_one_to_one_stages(
            models_clean      = models_clean,
            averaged_models   = averaged_models,
            selected_output   = selected_output,
            best_fit          = fits_step4[selected_output.K - 1],
            theta_obs         = theta_obs,
            sample_positions  = sample_positions,
            sample_times      = sample_times,
            output_dir        = OUTPUT_DIR,
            scenario_name     = cfg.name if cfg else "default",
            probe_layer_indices = probe_layer_indices,
        )

        reporter.end_step("Step 6c")

        # (a) Per-realisation plots — a uniform sample of individual inversions
        pediag.run_emagpy_plots_per_realisation(
            csv_paths        = EM_FNAMES,
            tasks            = tasks,            # built in Step 1
            models_clean     = models_clean,
            depths_clean     = depths_clean,
            depths0          = DEPTHS0,
            output_dir       = OUTPUT_DIR,
            fixed_depths     = _fixed_bool,
            mode             = mode,
            max_realisations = 4,                # cap: 4 sample realisations rendered
            em_unit          = EM_UNIT,
        )

        # (b) Optimal model plots — weighted average of top-K realisations
        pediag.run_emagpy_plots_optimal(
            csv_paths        = EM_FNAMES,
            selected_output  = selected_output,  # SelectionOutput from Step 4/5
            averaged_models  = averaged_models,  # list from Step 3
            depths0          = DEPTHS0,
            output_dir       = OUTPUT_DIR,
            fixed_depths     = _fixed_bool,
            conds0           = None,
            invert_kwargs    = _base_inv_kwargs,
            mode             = mode,
            em_unit          = EM_UNIT,
        )

        reporter.end_step("Step 6b")

        # -----------------------------------------------------------------------
        # STEP 6d — SOLUTION SUMMARY PLOTS (SWC + EC_bulk, synthetic ground
        #           truth overlaid when available)
        #   Renders the pipeline's final solution -- the RMSE-weighted top-K
        #   averaged model selected in Step 5 -- as vertical soil-water-content
        #   and bulk-EC profiles. In synthetic mode the known ground truth
        #   (theta_true_realization / synthetic_time_lapse_ec.npy) is overlaid
        #   as a dashed reference so the recovered solution can be sanity-
        #   checked at a glance. Real-data runs get the same plots without the
        #   ground-truth overlay (none exists in the field).
        # -----------------------------------------------------------------------
        reporter.step("Step 6d", "Solution SWC / EC_bulk summary plots")
        reporter.log("\n" + "=" * 80)
        reporter.log(" [PIPELINE] STEP 6d: SOLUTION SUMMARY PLOTS (SWC + EC_bulk)")
        reporter.log("=" * 80)

        theta_true_for_plot   = None
        ec_bulk_true_for_plot = None
        if data_source == "synthetic":
            theta_true_for_plot = theta_true_realization
            _ec_true_path = os.path.join(OUTPUT_DIR, "synthetic_time_lapse_ec.npy")
            if os.path.exists(_ec_true_path):
                ec_bulk_true_for_plot = np.load(_ec_true_path)[REALIZATION_INDEX]
            else:
                reporter.log(
                    "[STEP 6d][WARN] 'synthetic_time_lapse_ec.npy' not found in "
                    f"'{OUTPUT_DIR}' — EC_bulk ground-truth overlay skipped."
                )

        swc_fig_path = psol.plot_solution_swc(
            theta_calc      = selected_output.theta_calc,
            depths0         = DEPTHS0,
            layer_centers   = synthetic_geometry["layer_centers"],
            output_dir      = OUTPUT_DIR,
            theta_true      = theta_true_for_plot,
            K               = selected_output.K,
            rmse            = selected_output.rmse,
            scenario_name   = cfg.name if cfg else "default",
            timelapse       = timelapse,
        )
        reporter.log(f"[STEP 6d] Solution SWC profiles      → {swc_fig_path}")

        ec_fig_path = psol.plot_solution_ec_bulk(
            ec_optimal      = averaged_models[selected_output.K - 1],
            depths0         = DEPTHS0,
            layer_centers   = synthetic_geometry["layer_centers"],
            output_dir      = OUTPUT_DIR,
            ec_bulk_true    = ec_bulk_true_for_plot,
            K               = selected_output.K,
            rmse            = selected_output.rmse,
            scenario_name   = cfg.name if cfg else "default",
            timelapse       = timelapse,
        )
        reporter.log(f"[STEP 6d] Solution EC_bulk profiles  → {ec_fig_path}")

        reporter.end_step("Step 6d")

        # -----------------------------------------------------------------------
        # STEP 6e — FORWARD-SIMULATE ECa FROM THE OPTIMAL MODEL
        #   Same "inject the weighted-average EC model" idea used by
        #   pediag.run_emagpy_plots_optimal() above, but here the injected
        #   model is forward-simulated (emagpy Problem.forward(), mirroring
        #   0_generate_synthetic_data.py's ground-truth generation) to obtain
        #   predicted ECa, which is then compared against the raw survey ECa
        #   that was actually fed to the inversion. This is a direct
        #   forward-model check of the full pipeline's final output.
        # -----------------------------------------------------------------------
        reporter.step("Step 6e", "Forward-simulate ECa from optimal model")
        reporter.log("\n" + "=" * 80)
        reporter.log(" [PIPELINE] STEP 6e: FORWARD-SIMULATE ECa FROM OPTIMAL MODEL")
        reporter.log("=" * 80)

        eca_forward_results = psol.forward_simulate_eca_optimal(
            csv_paths       = EM_FNAMES,
            averaged_models = averaged_models,
            selected_output = selected_output,
            depths0         = DEPTHS0,
            output_dir      = OUTPUT_DIR,
            coils_list      = step0.COILS_LIST if data_source == "synthetic" else None,
            x_transect      = x_transect,
            forward_model   = _base_inv_kwargs.get("forwardModel", "CS"),
            scenario_name   = cfg.name if cfg else "default",
        )
        for _t_idx, _rmse_eca in eca_forward_results:
            reporter.metric(f"Forward ECa RMSE (optimal, t={_t_idx})", f"{_rmse_eca:.4f} mS/m")

        reporter.end_step("Step 6e")

        reporter.end_step("Step 6")

        # Record key metrics and close report
        reporter.metric("Spatial mode",            mode.upper())
        reporter.metric("Ensemble size (Step 1)",  f"{ensemble_size} realisations  [axis 0 of ensemble matrix]")
        reporter.metric("Ensemble matrix shape",   str(models_clean.shape) + ("  (ensemble, time, points, layers)" if timelapse else "  (ensemble, points, layers)"))
        reporter.metric("Time steps (n_time)",     n_time)
        reporter.metric("Measured points / survey",num_measured_points)
        reporter.metric("Soil layers (n_layers)",  num_layers)
        reporter.metric("SWC sensor observations", len(theta_obs))
        reporter.metric("SWC spatial positions",   NUM_POSITIONS)
        reporter.metric("Optimal stack depth (K)", f"{selected_output.K}  (out of {len(averaged_models)} candidates)")
        reporter.metric("Final RMSE (Step 5)",     f"{selected_output.rmse:.6f} m³/m³")
        _cv_shape = selected_output.cv_map.shape
        reporter.metric(
            "CV map shape",
            str(_cv_shape) + (
                f"  (n_time={_cv_shape[0]}, n_points={_cv_shape[1]}, n_layers={_cv_shape[2]})"
                if len(_cv_shape) == 3 else
                f"  (n_points={_cv_shape[0]}, n_layers={_cv_shape[1]})"
            )
        )
        reporter.metric("CV map mean (relative uncertainty)", f"{np.nanmean(selected_output.cv_map):.4f}")
        reporter.metric("CV map max",                          f"{np.nanmax(selected_output.cv_map):.4f}")
        if selected_output.theta_calc is not None:
            _tc_shape = selected_output.theta_calc.shape
            reporter.metric(
                "theta_calc shape",
                str(_tc_shape) + (
                    f"  (n_time={_tc_shape[0]}, n_points={_tc_shape[1]}, n_layers={_tc_shape[2]})"
                    if len(_tc_shape) == 3 else ""
                )
            )
            reporter.metric("theta_calc mean [m³/m³]", f"{np.nanmean(selected_output.theta_calc):.4f}")
        reporter.metric("Timelapse mode",          timelapse)
        reporter.metric("Number of SWC probes",    len(PROBE_X_COORDS))
        reporter.metric("Probe placement mode",    "2D UTM grid" if mode == "2d" else "1D transect")

        reporter.log("\n" + "=" * 80)
        reporter.log(" [PIPELINE] METRIC REPORT SUMMARY")
        reporter.log("=" * 80)
        reporter.log(f"  Spatial mode             : {mode.upper()}")
        reporter.log(f"  Ensemble size            : {ensemble_size} realisations")
        reporter.log(f"  Ensemble matrix shape    : {models_clean.shape}"
                     + ("  (ensemble, time, points, layers)" if timelapse else "  (ensemble, points, layers)"))
        reporter.log(f"  Soil layers              : {num_layers}")
        reporter.log(f"  Measured points/survey   : {num_measured_points}")
        reporter.log(f"  SWC observations used    : {len(theta_obs)}  ({NUM_POSITIONS} positions × {n_time} time step(s))")
        reporter.log(f"  Optimal stack depth K    : {selected_output.K} / {len(averaged_models)}")
        reporter.log(f"  Final RMSE (Step 5)      : {selected_output.rmse:.6f} m³/m³")
        _cv_sh = selected_output.cv_map.shape
        reporter.log(f"  CV map shape             : {_cv_sh}"
                     + (f"  (n_time, n_points, n_layers)" if len(_cv_sh)==3 else "  (n_points, n_layers)"))
        reporter.log(f"  CV map mean / max        : {np.nanmean(selected_output.cv_map):.4f} / {np.nanmax(selected_output.cv_map):.4f}")
        reporter.log(f"  Quality gate             : {quality_gate_result.level}"
                     + (f"  ({'; '.join(quality_gate_result.reasons)})"
                        if quality_gate_result.reasons else ""))
        reporter.log("=" * 80 + "\n")

        reporter.finish(
            exit_ok=True,
            quality_level=quality_gate_result.level,
            quality_reasons=quality_gate_result.reasons,
        )

    except Exception as exc:
        reporter.failure(exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EA-EM ensemble pipeline (Loiseau et al., 2026) — Synthetic & Real Data."
    )
    parser.add_argument(
        "--data-source", default="synthetic", choices=["synthetic", "real"],
        help="Data source: 'synthetic' (default) or 'real' (field data)."
    )
    parser.add_argument("--scenario", default="default", metavar="NAME",
                        help="Scenario name (applies to both synthetic and real-data "
                             "mode — depths0 / cond_ranges / param_grid / SWC sensor "
                             "layout all carry over; e.g. use 'mfc2_real' for real data).")
    parser.add_argument("--config",   default=None,      metavar="PATH",
                        help="Custom scenario config JSON file.")
    parser.add_argument(
        "--mode", choices=["1d", "2d"], default="1d",
        help=(
            "Spatial mode. '1d' (default): single transect, XZ plots only. "
            "'2d': rectangular UTM grid anchored to the Alento survey; "
            "produces XZ cross-section AND XY plan-view plots."
        ),
    )
    parser.add_argument("--em-dir",   default=None,      metavar="PATH",
                        help="EM survey CSV directory (required for --data-source real).")
    parser.add_argument("--tdr-dir",  default=None,      metavar="PATH",
                        help="TDR Excel data directory (required for --data-source real).")
    parser.add_argument("--coord-file",
                        default="./data/SOILNET_MFC2_coordinates_sensors.csv", metavar="PATH",
                        help="Sensor coordinates CSV file (default: ./data/SOILNET_MFC2_coordinates_sensors.csv).")
    parser.add_argument(
        "--drop-coils", nargs="+", default=None, metavar="COIL",
        help="One or more coils/frequencies to exclude from every inversion "
             "(e.g. a noisy GEM-2 channel). Each value may be the exact "
             "emagpy coil name, a frequency in Hz (e.g. 425), or a unique "
             "coil-name substring (e.g. f425). Overrides any 'drop_coils' "
             "field in the scenario config. Example: --drop-coils f425",
    )
    parser.add_argument(
        "--homogeneous", action="store_true", default=False,
        help="Synthetic mode only. Disable spatial (XY/XZ) gradients in the "
             "generated ground truth so theta is spatially uniform. Temporal "
             "trend and noise are still applied. Overrides 'homogeneous' in "
             "the scenario config when passed.",
    )
    parser.add_argument(
        "--num-swc-positions", type=int, default=None, metavar="N",
        help="Number of horizontal SWC/TDR sensor positions. Total probes "
             "= N x n_layers (one per soil horizon at each position). "
             "Overrides 'num_swc_positions' in the scenario config when "
             "passed; falls back to the scenario value (or 4) otherwise.",
    )
    parser.add_argument(
        "--swc-depths", nargs="+", type=float, default=None, metavar="DEPTH",
        help="Explicit SWC/TDR sensor installation depths [m], e.g. "
             "--swc-depths 0.15 0.45 0.9. Overrides 'swc_depths' in the "
             "scenario config when passed; falls back to the scenario "
             "value, or auto-derived layer-centre depths, otherwise.",
    )
    parser.add_argument(
        "--window-m", type=float, default=None, metavar="M",
        help="Half-width [m] of the spatial averaging window used to match "
             "each SWC/TDR probe to nearby EM measurement indices. "
             "Overrides 'window_m' in the scenario config when passed; "
             "falls back to the scenario value (or 1.0) otherwise.",
    )
    parser.add_argument(
        "--em-unit", default=None, choices=["ppm", "ppt"],
        help="Physical unit of the quadrature/in-phase columns in the EM "
             "survey CSV. 'ppm' for GEM-2 / CMD-Explorer real-data "
             "instruments (this pipeline's real-data branch only ingests "
             "GEM-2 quadrature exports, i.e. 'ppm'); 'ppt' for GF "
             "Instruments CMD series. Overrides 'em_unit' in the scenario "
             "config when passed -- note that scenario_loader now REQUIRES "
             "em_unit to be set in the scenario JSON for data_source='real', "
             "so this flag is an ad-hoc override, not a substitute for "
             "setting it correctly there.",
    )
    parser.add_argument(
        "--beta", nargs="+", type=float, default=None, metavar="B",
        help="One or more beta values [lateral/spatial smoothing, emagpy "
             "inversion] to sweep, e.g. --beta 0.0 0.1 0.5. Overrides the "
             "'beta' entry in the scenario's param_grid when passed; falls "
             "back to the scenario value (or [0.0]) otherwise.",
    )
    parser.add_argument(
        "--gamma", type=float, default=None, metavar="G",
        help="Temporal regularisation between time steps (>= 0), only used "
             "for time-lapse (multi-epoch) inversions. Overrides 'gamma' in "
             "the scenario config when passed; falls back to the scenario "
             "value (or 0.0) otherwise.",
    )
    args = parser.parse_args() 
    
    # Validate real data requirements
    if args.data_source == "real":
        if not args.em_dir or not args.tdr_dir:
            print("[ERROR] --data-source real requires --em-dir and --tdr-dir")
            sys.exit(1)
        if not os.path.isdir(args.em_dir):
            print(f"[ERROR] EM directory not found: {args.em_dir}")
            sys.exit(1)
        if not os.path.isdir(args.tdr_dir):
            print(f"[ERROR] TDR directory not found: {args.tdr_dir}")
            sys.exit(1)
        if not os.path.isfile(args.coord_file):
            print(f"[ERROR] Coordinate file not found: {args.coord_file}")
            sys.exit(1)
        print("[VALIDATION] Real data mode: all paths verified ✓")

    # Load scenario config — applies to both data sources. depths0,
    # cond_ranges, param_grid, and the SWC sensor layout (num_swc_positions /
    # swc_depths / window_m) all still matter in real-data mode; use the
    # 'mfc2_real' scenario (or your own) to match your actual sensor network
    # instead of the generic synthetic-data defaults.
    cfg = load_scenario(args.scenario, config_path=args.config)

    # -----------------------------------------------------------------------
    # Resolve effective data source and paths — scenario JSON wins.
    # Rules:
    #   1. If the scenario specifies data_source='real', switch to real mode
    #      automatically and use the paths embedded in the JSON (unless the
    #      user explicitly overrode them on the CLI).
    #   2. If the user passed --data-source real but the scenario says
    #      'synthetic' (or vice versa), abort with a clear error — mixing is
    #      not supported and would silently produce wrong results.
    #   3. If both agree, proceed normally.
    # -----------------------------------------------------------------------
    scenario_ds = getattr(cfg, "data_source", "synthetic")
    cli_ds      = args.data_source   # what the user typed (default: 'synthetic')

    # Detect conflict between CLI flag and scenario JSON
    # Only raise if the user *explicitly* passed --data-source (i.e. it
    # differs from the scenario declaration).
    if cli_ds != scenario_ds:
        print(
            f"[ERROR] Data-source conflict: --data-source '{cli_ds}' was given "
            f"on the command line, but scenario '{cfg.name}' declares "
            f"data_source='{scenario_ds}'.\n"
            f"  Fix option A — remove --data-source and let the scenario decide.\n"
            f"  Fix option B — choose a scenario whose data_source matches '{cli_ds}'.\n"
            f"  Mixing synthetic generation with real field data is not supported."
        )
        sys.exit(1)

    # Resolve effective data source and paths.
    # For real-data scenarios, CLI path overrides > JSON paths > error.
    effective_ds         = scenario_ds
    effective_em_dir     = args.em_dir    or getattr(cfg, "em_data_dir",  None)
    effective_tdr_dir    = args.tdr_dir   or getattr(cfg, "tdr_data_dir", None)
    effective_coord_file = (args.coord_file
                            if args.coord_file != "./data/SOILNET_MFC2_coordinates_sensors.csv"
                            else None) or getattr(cfg, "coord_file", None) \
                           or "./data/SOILNET_MFC2_coordinates_sensors.csv"

    if effective_ds == "real":
        missing = []
        if not effective_em_dir:
            missing.append("em_data_dir (--em-dir)")
        if not effective_tdr_dir:
            missing.append("tdr_data_dir (--tdr-dir)")
        if missing:
            print(
                f"[ERROR] Scenario '{cfg.name}' uses data_source='real' but the "
                f"following required paths are not set in the JSON or on the CLI:\n"
                + "".join(f"  • {m}\n" for m in missing)
            )
            sys.exit(1)
        for label, path, checker in [
            ("EM directory",      effective_em_dir,     os.path.isdir),
            ("TDR directory",     effective_tdr_dir,    os.path.isdir),
            ("Coordinates file",  effective_coord_file, os.path.isfile),
        ]:
            if not checker(path):
                print(f"[ERROR] {label} not found: {path}")
                sys.exit(1)
        print("[VALIDATION] Real data mode: all paths verified ✓")

    # Startup banner
    data_source_label = "REAL DATA" if effective_ds == "real" else "SYNTHETIC"
    print("\n" + "=" * 80)
    print(f" [PIPELINE] Data source     : {data_source_label}")
    print(f" [PIPELINE] Loaded scenario : {cfg.name}")
    if effective_ds == "real":
        print(f" [PIPELINE] EM directory    : {effective_em_dir}")
        print(f" [PIPELINE] TDR directory   : {effective_tdr_dir}")
        print(f" [PIPELINE] Coordinates     : {effective_coord_file}")
    print(f" [PIPELINE] Spatial mode    : {args.mode.upper()}")
    print(cfg.summary())
    print("=" * 80 + "\n")

    try:
        main(cfg, mode=args.mode, data_source=effective_ds,
             em_dir=effective_em_dir, tdr_dir=effective_tdr_dir,
             coord_file=effective_coord_file, drop_coils=args.drop_coils,
             homogeneous=(True if args.homogeneous else None),
             num_swc_positions=args.num_swc_positions,
             swc_depths=args.swc_depths,
             window_m=args.window_m,
             em_unit=args.em_unit,
             beta=args.beta,
             gamma=args.gamma)
    except Exception:
        # reporter.failure() was already called inside main() and the report
        # was written there.  We just set the shell exit code to non-zero.
        sys.exit(1)