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
import re
import shutil
import sys
import numpy as np

# FIX ("stuck at 100% after Step 1 inversion"): pin a headless matplotlib
# backend BEFORE any of the plot_*/em_methodology modules below get a
# chance to `import matplotlib.pyplot`, which otherwise happens in this,
# the PARENT process, ahead of the ProcessPoolExecutor fork further down.
# If matplotlib had already picked an interactive backend (TkAgg/QtAgg --
# possible whenever a display is reachable, e.g. over X11/VNC on the
# host), every forked worker inherits that half-initialised GUI backend
# state. Workers still return their inversion results fine (which is why
# tqdm reaches 864/864), but then hang tearing down that inherited state
# at interpreter shutdown, which blocks the parent's implicit
# `executor.shutdown(wait=True)` forever with ~0% CPU on the stuck
# workers -- looks identical to "the pipeline is stuck" but isn't stuck
# doing any work. `Agg` is headless/file-only, which is all this pipeline
# ever needs (every plot here is `savefig`'d, never `show()`n).
import matplotlib
matplotlib.use("Agg")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import em_methodology as EMmethod
import em_subsample
import geometry_utils
import quick_preset
import stage_split
import train_val_split as tvs
from scenario_loader import load_scenario, ScenarioConfig
import plot_water_content as pwc
import plot_emagpy_diagnostics as pediag
import plot_em_diagnostics_real as pediag_real
import plot_tdr_diagnostics as ptdrdiag
import plot_solution_diagnostics as psol
import plot_ini_model as pim
import plot_probe_matching as ppm
import plot_calibration_density as pcd
import em_sensitivity as emsens
import plot_em_footprint as pef
import plot_pre_inversion_diagnostics as preinvdiag
import plot_train_val_diagnostics as ptv
from plots import pre_inversion, ws_distribution, initial_ec_ws_real
from plots.truth_tdr import plot_truth_tdr  # moved to plots/ (Phase 5); name kept for run_and_score

import data_synthetic as step0
import step2_petro as step2
import step3_average as step3
import step4_select as step4_5
from steps import step6_plots as steps_step6  # Phase 6: step functions extracted from main()


# ===========================================================================
# PIPELINE REPORTER
# ===========================================================================

import sys
import datetime
import platform


def _collect_npy_into_output_model(output_dir: str, output_model_dir: str) -> list[str]:
    """
    Move every ``*.npy`` file sitting directly in ``output_dir`` (not in
    subfolders) into ``output_model_dir``. Returns the list of moved files.

    This is used right after any step that writes raw numpy arrays (e.g.
    Step 0's synthetic-data generator) so all ``.npy`` outputs end up
    consolidated in ``output_model_dir`` instead of scattered alongside
    plots/CSVs in the top-level output directory.
    """
    import shutil
    os.makedirs(output_model_dir, exist_ok=True)
    moved = []
    for fname in os.listdir(output_dir):
        src = os.path.join(output_dir, fname)
        if os.path.isfile(src) and fname.endswith(".npy"):
            dst = os.path.join(output_model_dir, fname)
            shutil.move(src, dst)
            moved.append(dst)
    return moved


from steps._common import resolve_npy_path as _resolve_npy_path  # noqa: E402  (moved in Phase 6)


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
                 data_source: str = "synthetic", em_dir: str = None, tdr_dir: str = None,
                 run_id: str | None = None, train_val_split: str = "off") -> None:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = os.path.join(output_dir, f"pipeline_report_{ts}.md")
        self._f = open(self.report_path, "w", encoding="utf-8")
        self._step_times: dict[str, datetime.datetime] = {}
        self._step_durations: dict[str, float] = {}
        self._metrics: list[tuple[str, str]] = []
        self._start = datetime.datetime.now()

        # Unique identifier for this run, shared between the report header,
        # the run_id.json marker dropped in output_dir, and the run's row
        # in the shared run-log CSV (see append_run_log / write_run_marker).
        # Defaults to "<scenario>_<timestamp>" -- reuses the same timestamp
        # as the report filename so the two are trivially linkable.
        self.run_id = run_id or f"{scenario_name}_{ts}"

        # Populated by finish()/failure() so callers (main()) can build a
        # run-log row without re-deriving elapsed time / status themselves.
        self.exit_ok: bool | None = None
        self.elapsed_total: float | None = None
        self.quality_level: str | None = None
        self.error_message: str | None = None

        # What run_manifest.json needs besides metrics/timings (see run_log.py).
        self._cfg = cfg
        self._header = {
            "scenario": scenario_name, "mode": mode.upper(),
            "data_source": data_source, "train_val_split": train_val_split,
            "output_dir": output_dir,
        }
        self._generated = self._start.strftime('%Y-%m-%d %H:%M:%S')
        self._manifest_extra: dict = {}

        # Header
        self._write(f"# EA-EM Pipeline Run Report")
        self._write(f"")
        self._write(f"| Field | Value |")
        self._write(f"|-------|-------|")
        self._write(f"| Generated | {self._start.strftime('%Y-%m-%d %H:%M:%S')} |")
        self._write(f"| Run ID    | `{self.run_id}` |")
        self._write(f"| Data source | `{data_source.upper()}` |")
        if data_source == "real":
            self._write(f"| EM directory | `{em_dir}` |")
            self._write(f"| TDR directory | `{tdr_dir}` |")
        self._write(f"| Scenario  | `{scenario_name}` |")
        self._write(f"| Mode      | `{mode.upper()}` |")
        self._write(f"| Train/val split | `{train_val_split}` |")
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

    def get_metric(self, name: str, default=""):
        """
        Return the string value most recently recorded for *name* via
        ``.metric()``, or *default* if it was never recorded. Used to pull
        a handful of headline numbers (RMSE, CV, ensemble size, ...) into
        the shared run-log CSV row without re-deriving them from scratch.
        """
        for k, v in reversed(self._metrics):
            if k == name:
                return v
        return default


    def set_manifest_extra(self, **kwargs) -> None:
        """Add fields (seeds, CLI options, ...) to the run_manifest.json written at the end."""
        self._manifest_extra.update(kwargs)

    def _write_manifest(self, status: str, result_line: str, elapsed: float,
                        quality_reasons: list[str] | None = None,
                        error: str = "") -> None:
        """Write ``run_manifest.json`` next to the report (never raises)."""
        import dataclasses
        cfg_dict = None
        if self._cfg is not None and dataclasses.is_dataclass(self._cfg):
            cfg_dict = dataclasses.asdict(self._cfg)
        write_run_manifest(self._header["output_dir"], {
            "run_id": self.run_id,
            **self._header,
            "generated": self._generated,
            "status": status,
            "result_line": result_line,
            "quality_level": self.quality_level,
            "quality_reasons": list(quality_reasons or []),
            "error": error,
            "runtime_s": round(elapsed, 1),
            "report": os.path.basename(self.report_path),
            "git_commit": git_commit(),
            "python": sys.version.split()[0],
            "platform": f"{platform.system()} {platform.machine()}",
            "host": platform.node(),
            "metrics": dict(self._metrics),
            "step_timing": {k: round(v, 1) for k, v in self._step_durations.items()},
            "config": cfg_dict,
            **self._manifest_extra,
        })

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

        self.exit_ok = False
        self.elapsed_total = elapsed
        self.quality_level = "FAILED"
        self.error_message = str(exc)

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
        self._write_manifest("FAILED", "**❌ FAILED**", elapsed, error=str(exc)[:300])
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
        self.exit_ok = exit_ok
        self.elapsed_total = elapsed_total
        self.quality_level = quality_level if exit_ok else "FAILED"
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
        self._write_manifest(
            "FAILED" if not exit_ok else ("WARN" if quality_level == "WARN" else "SUCCESS"),
            f"**{status}**", elapsed_total, quality_reasons=quality_reasons)
        print(f"[REPORTER] Report written → {self.report_path}  (status: {status})")


# ===========================================================================
# RUN LOG — one row per pipeline run, shared across all scenarios
# ===========================================================================

# The shared run-log CSV (one row per run, all scenarios) and the per-run
# run_manifest.json live in run_log.py. DEFAULT_RUN_LOG / RUN_LOG_COLUMNS /
# append_run_log keep their old names here for the existing call sites.
from run_log import (DEFAULT_RUN_LOG, RUN_LOG_COLUMNS, RunLog, git_commit,  # noqa: E402,F401
                     write_run_manifest)

# Seed of the Step 1 inversion workers (task i uses DEFAULT_INVERSION_SEED + i).
DEFAULT_INVERSION_SEED = 1234


def append_run_log(log_path: str, row: dict) -> None:
    """
    Append one row to the shared run-log CSV (see ``run_log.RunLog.append``).
    Every ``run_pipeline.py`` invocation -- success, quality-gate warning, or
    failure -- adds exactly one row, keyed by ``run_id``. Never raises.
    """
    RunLog(log_path).append(row)


def _next_run_seq(base_dir: str, width: int = 4) -> int:
    """
    Scan *base_dir* for existing per-run subfolders named ``<seq>_...``
    (zero-padded to *width* digits) and return the next sequence number.

    Purely based on what's actually on disk, so it self-heals if folders
    are renamed or deleted -- there's no separate counter file to get out
    of sync.
    """
    pattern = re.compile(r"^(\d+)_")
    best = 0
    try:
        for entry in os.listdir(base_dir):
            if not os.path.isdir(os.path.join(base_dir, entry)):
                continue
            m = pattern.match(entry)
            if m:
                best = max(best, int(m.group(1)))
    except FileNotFoundError:
        pass
    return best + 1


def _sanitize_folder_component(name: str) -> str:
    """Make *name* safe to use as (part of) a directory name."""
    safe = re.sub(r"\s+", "_", name.strip())
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", safe)
    return safe or "scenario"


def next_run_output_dir(base_dir: str, scenario_name: str, width: int = 4) -> str:
    """
    Build (and create) this run's own numbered output folder:
    ``<base_dir>/<seq>_<scenario_name>/``.

    Every ``run_pipeline.py`` invocation now gets a fresh, uniquely
    numbered folder under *base_dir* (e.g. ``0001_field_survey_2d``,
    then ``0002_field_survey_2d`` on the next run of the same scenario)
    instead of writing into -- and silently overwriting -- the same
    directory every time. The shared run-log CSV (see ``append_run_log``)
    still accumulates one row per run across every scenario for a
    quick, flat history / ``--list`` view.
    """
    os.makedirs(base_dir, exist_ok=True)
    seq = _next_run_seq(base_dir, width=width)
    folder = f"{seq:0{width}d}_{_sanitize_folder_component(scenario_name)}"
    run_dir = os.path.join(base_dir, folder)
    # Extremely unlikely, but guard against a collision (e.g. two runs
    # launched in the same instant against a freshly-cleared directory)
    # by bumping the sequence rather than clobbering an existing folder.
    while os.path.exists(run_dir):
        seq += 1
        folder = f"{seq:0{width}d}_{_sanitize_folder_component(scenario_name)}"
        run_dir = os.path.join(base_dir, folder)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_run_marker(output_dir: str, run_id: str, report_path: str) -> None:
    """
    Drop a small ``run_id.json`` marker into *output_dir* identifying the
    run that produced it.

    Each run now gets its own numbered folder (see
    ``next_run_output_dir``), so this marker is mostly a convenience for
    identifying a folder out of context (e.g. after copying it
    elsewhere); the folder name itself (``<seq>_<scenario>``) is already
    unique per run. The full flat history across every scenario still
    lives in the shared run-log CSV (see ``append_run_log``).

    Never raises: same rationale as ``append_run_log``.
    """
    try:
        marker_path = os.path.join(output_dir, "run_id.json")
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id,
                "report_path": os.path.abspath(report_path),
                "written_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
    except OSError as e:
        print(f"[RUN LOG] WARNING: could not write run marker in {output_dir!r}: {e}")


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
        filename upstream by ``data_real.extract_date_from_filename``).
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
        ``_prepare_real_em_csv`` / ``preprocess_em_survey``.
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
    ``data_real.prepare_real_dataset`` uses for SWC, but for
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
        The ``load_tdr_dataframe`` function from ``data_real``,
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
                    filter_repeated_tolerance: float = 1e-10,
                    filter_range_vmin: float | None = 0.0,
                    filter_range_vmax: float | None = None,
                    filter_percentile_qmin: float | None = 5.0,
                    filter_percentile_qmax: float | None = 95.0,
                    rolling_mean_window: int | None = 5,
                    drop_coils=None,
                    em_subsample_cfg: dict | None = None,
                    quick_epochs: int | None = None,
                    quick: bool = False):
    """
    Load real EM and TDR field data.

    ``quick`` / ``quick_epochs`` (``--quick`` preset, see quick_preset.py):
    when ``quick_epochs`` is set, only that many evenly spaced survey epochs
    are kept (first and last always included) -- applied right after the
    dataset is discovered, so skipped epochs are never pre-processed and
    every per-epoch output (em_files, em_dates, raw_em_files, observations'
    ``time_index``, temp_by_epoch) stays mutually consistent. The kept
    original indices are returned in ``real_data_diagnostics['epoch_indices']``.

    Preprocessing step added: raw GEM-2 CSV files use semicolon delimiters and
    instrument-specific column names that emagpy cannot parse.  Each file is
    converted to a comma-separated, emagpy-compatible CSV before being returned
    as part of ``em_files``.  The processed files are written to a sub-directory
    ``<em_dir>/_emagpy_ready/`` so the originals are never modified.

    Each processed CSV also has the classic EMagPy raw-ECa cleaning chain
    applied in place via ``preprocess_em_survey`` — using emagpy's own
    ``Problem.filterRange`` / ``filterRepeated`` / ``filterPercentile`` /
    ``rollingMean`` — so this happens exactly once per survey here at load
    time, rather than redundantly inside every parallel inversion worker.
    Order: drop_coils (exclude channel(s) that will be excluded from
    inversion anyway, so they can't veto points on filterRange) ->
    filterRange (drop impossible negative ECa) -> filterRepeated (drop
    stationary/duplicate GPS fixes) -> filterPercentile (drop statistical
    outliers/spikes per coil) -> rollingMean (smooth the now-clean
    sequence). Any stage can be disabled by passing ``None`` for its
    threshold(s) (or ``rolling_mean_window=None``/``<=1``; ``drop_coils=None``).

    Parameters (new)
    -----------------
    filter_range_vmin, filter_range_vmax : float or None, optional
        ``filterRange`` bounds passed to ``preprocess_em_survey``. Default
        ``(0.0, None)`` — drop negative ECa, no upper cutoff.
    filter_percentile_qmin, filter_percentile_qmax : float or None, optional
        ``filterPercentile`` bounds (percent) passed to
        ``preprocess_em_survey``. Default ``(5.0, 95.0)``.
    rolling_mean_window : int or None, optional
        ``rollingMean`` window size passed to ``preprocess_em_survey``.
        Default 5.
    drop_coils : None, str, int, float, or sequence thereof, optional
        Coil(s)/frequency channel(s) to exclude (same spec as
        ``EMmethod.drop_coils_from_problem``), applied inside
        ``preprocess_em_survey`` before ``filterRange`` runs, so a
        soon-to-be-dropped channel's readings can't cause otherwise-good
        points to be discarded. This is purely an in-memory exclusion for
        the purpose of computing/applying the filter chain — it does not
        remove the coil's column from the CSV, and does not replace the
        per-task ``drop_coils_from_problem`` call still made before
        inversion. Default ``None``.

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
               filtering) in the same order — kept around solely so
               diagnostic plotting (Step 1b) can show the full per-epoch
               footprint before the row-count trim, instead of the already
               column-renamed and filtered processed files —
               and real_data_diagnostics is a dict with keys
               'coord_kinds' (list of per-file {'file','coord_kind',
               'epsg_used'} dicts), 'em_unit_checks' (list of per-file
               _check_em_unit_sanity() result dicts), 'crs_consistency'
               (single {'level','message'} verdict vs. probe_table),
               'n_geographic_reprojected' (int), 'filter_stats' (list of
               per-file ``preprocess_em_survey`` stats dicts — see that
               function's docstring for keys), and 'temp_by_epoch'
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

    # Import the real data loader (data_real.py, next to this script)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_real import prepare_real_dataset, preprocess_em_survey, load_tdr_dataframe

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

    # --quick epoch subset (None = all epochs, the normal behaviour).
    n_epochs_total = len(raw_em_files)
    epoch_indices = list(range(n_epochs_total))
    if quick and quick_epochs is not None:
        epoch_indices = quick_preset.select_epochs(n_epochs_total, quick_epochs)
        if len(epoch_indices) < n_epochs_total:
            raw_em_files, em_dates, observations = quick_preset.subset_epochs(
                raw_em_files, em_dates, observations, epoch_indices)
            print(f"[QUICK] Epoch subset: keeping {len(epoch_indices)}/{n_epochs_total} "
                  f"surveys (original indices {epoch_indices}): "
                  + ", ".join(str(d.date()) for d in em_dates))

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
    filter_stat_records: list[dict] = []
    for raw_path, survey_date in zip(raw_em_files, em_dates):
        try:
            proc_path, coord_kind, epsg_used = _prepare_real_em_csv(
                raw_path, processed_dir, survey_date=survey_date,
            )
            # Snapshot the pre-filter processed CSV (regenerated from the raw
            # file on every run, so this is always genuinely pre-filter) for
            # the before/after overlay in filtered_eca_transect.png.
            _prefilter_dir = os.path.join(processed_dir, "_prefilter")
            os.makedirs(_prefilter_dir, exist_ok=True)
            shutil.copy2(proc_path,
                         os.path.join(_prefilter_dir, os.path.basename(proc_path)))
            proc_path, filt_stats = preprocess_em_survey(
                proc_path,
                tolerance=filter_repeated_tolerance,
                unit=em_unit,
                range_vmin=filter_range_vmin,
                range_vmax=filter_range_vmax,
                percentile_qmin=filter_percentile_qmin,
                percentile_qmax=filter_percentile_qmax,
                rolling_window=rolling_mean_window,
                drop_coils=drop_coils,
            )
            if em_subsample_cfg:
                # Optional thinning AFTER the filter chain (see em_subsample.py).
                sub = em_subsample.subsample_em_csv(
                    proc_path, em_subsample_cfg,
                    probe_x=probe_table["utm_x"].to_numpy(dtype=float),
                    probe_y=probe_table["utm_y"].to_numpy(dtype=float),
                )
                filt_stats["subsample"] = sub
                filt_stats["n_final_before_subsample"] = filt_stats["n_final"]
                filt_stats["n_final"] = sub["n_after"]
                print(f"[+] em_subsample[{sub['mode']}] {os.path.basename(proc_path)}: "
                      f"{sub['n_before']} -> {sub['n_after']} pts")
            filter_stat_records.append(filt_stats)
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
        "filter_stats": filter_stat_records,      # one preprocess_em_survey() stats dict per file
        "temp_by_epoch": temp_by_epoch,           # list[float|None], one per em_dates entry
        "epoch_indices": epoch_indices,          # original survey indices kept (all unless --quick-epochs)
        "n_epochs_total": n_epochs_total,
    }

    return em_files, observations, probe_table, em_dates, raw_em_files, real_data_diagnostics


def _config_fingerprint(depths0, fixed_depths, param_grid, cond_ranges, gamma,
                          drop_coils=None, spatial_match_tolerance_m=None,
                          pin_layers=None, pin_source=None,
                          em_subsample=None, epochs=None, members=None) -> str:
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
    if pin_layers is not None:
        # Only added when pinning is on, so every existing cache keeps its hash.
        payload["pin_layers"] = [bool(v) for v in pin_layers]
        payload["pin_source"] = pin_source
    if em_subsample:
        # Only added when thinning is on, so every existing cache keeps its hash.
        payload["em_subsample"] = em_subsample
    if epochs is not None:
        # Only added for a --quick-epochs subset, so every existing cache keeps its hash.
        payload["epochs"] = list(epochs)
    if members is not None:
        # --stage map: only the calibration-selected members are inverted, so the
        # member set must be part of the key (param_grid alone describes the full grid).
        payload["members"] = [list(m) for m in members]
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
    # Reproducibility: ROPE/MCMC-family inversions sample randomly, and a spawned
    # worker starts with a fresh OS-entropy RNG. Seed per task (base seed + task
    # index) so results do not depend on worker scheduling order.
    import random as _random
    _seed = (int(invert_kwargs.get("inversion_seed", DEFAULT_INVERSION_SEED)) + int(task_idx)) % (2**32 - 1)
    _random.seed(_seed)
    np.random.seed(_seed)
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
        # (`data_real.preprocess_em_survey`), which runs the full
        # filterRange -> filterRepeated -> filterPercentile -> rollingMean
        # chain once per EM CSV at load time in `_load_real_data`, rather
        # than redundantly (and, for filterRepeated, incorrectly) inside
        # every parallel inversion task. That original per-task call also
        # had a bug: emagpy's `Problem.filterRepeated(self, tolerance=0.2)`
        # has never accepted an `index` keyword, so every call here was
        # raising `TypeError: filterRepeated() got an unexpected keyword
        # argument 'index'` and silently failing every single inversion
        # task.
        drop_coils = invert_kwargs.get("drop_coils")
        if drop_coils is not None:
            EMmethod.drop_coils_from_problem(k, drop_coils)
        # Optional layer pinning (e.g. condition the deeper layers on a layer
        # whose EC is known from TDR): ``fixed_conds`` is a per-layer bool list
        # and ``conds0_matrix`` an (n_points, n_layers) array of per-point
        # starting EC (the pinned layers keep exactly these values). Both
        # absent (default) -> all layers free, vector conds0, as before.
        _fixed_conds = invert_kwargs.get("fixed_conds")
        _conds0_matrix = invert_kwargs.get("conds0_matrix")
        _init_kwargs = {}
        if _fixed_conds is not None:
            _init_kwargs["fixedConds"] = [bool(f) for f in _fixed_conds]
        k.setInit(depths0=list(depths0),
                  conds0=(np.asarray(_conds0_matrix, dtype=float)
                          if _conds0_matrix is not None else list(conds0)),
                  fixedDepths=fixed_depths_list, **_init_kwargs)

        clean_kwargs = {key: v for key, v in invert_kwargs.items()
                        if key not in ("targetProjection", "em_unit", "drop_coils", "inversion_seed",
                                       "fixed_conds", "conds0_matrix")}
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
# data_synthetic.py (inline), and in run_pipeline_short.py, and
# a mismatch between those copies is exactly what caused the Step 6d
# "x and y must have same first dimension" crash this function's old
# docstring described. Kept as a thin alias so every existing call site
# (`_layer_centers(...)`) keeps working unchanged.
_layer_centers = geometry_utils.layer_centers


# ===========================================================================
# Explicit probe XYZ file loader (--probe-xyz)
# ===========================================================================

_PROBE_XYZ_COL_ALIASES = {
    "x": ("x", "utm_x", "easting"),
    "y": ("y", "utm_y", "northing"),
    "z": ("z", "depth", "depth_m"),
    "label": ("id", "name", "probe", "label"),
}


def _resolve_probe_xyz_column(columns_lower: dict, kind: str, path: str) -> str | None:
    """Return the actual column name matching one of the accepted aliases
    for X/Y/Z/label (case-insensitive), or None if 'label' isn't present."""
    for alias in _PROBE_XYZ_COL_ALIASES[kind]:
        if alias in columns_lower:
            return columns_lower[alias]
    if kind == "label":
        return None
    raise ValueError(
        f"[probe_xyz_file] '{path}': could not find a column for '{kind}'. "
        f"Accepted names: {_PROBE_XYZ_COL_ALIASES[kind]} (case-insensitive)."
    )


def load_probe_xyz_file(path: str):
    """
    Load explicit probe locations from a CSV, e.g. ``coord_TDR.csv``.

    One row per probe. Columns are matched case-insensitively against
    ``_PROBE_XYZ_COL_ALIASES``; an optional label column is carried
    through for plot legends but is not required.

    Returns
    -------
    (probe_x, probe_y, probe_z, probe_labels) : tuple of lists
        probe_labels is a list of str, falling back to "P0", "P1", ...
        when no label column is present.
    """
    import pandas as pd

    if not os.path.isfile(path):
        raise FileNotFoundError(f"--probe-xyz file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"--probe-xyz file '{path}' has no rows.")

    columns_lower = {c.lower(): c for c in df.columns}
    x_col = _resolve_probe_xyz_column(columns_lower, "x", path)
    y_col = _resolve_probe_xyz_column(columns_lower, "y", path)
    z_col = _resolve_probe_xyz_column(columns_lower, "z", path)
    label_col = _resolve_probe_xyz_column(columns_lower, "label", path)

    for col, kind in ((x_col, "x"), (y_col, "y"), (z_col, "z")):
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(
                f"--probe-xyz file '{path}': non-numeric or missing value(s) "
                f"in column '{col}' ({kind})."
            )

    probe_x = df[x_col].astype(float).tolist()
    probe_y = df[y_col].astype(float).tolist()
    probe_z = df[z_col].astype(float).tolist()
    probe_labels = (
        df[label_col].astype(str).tolist() if label_col is not None
        else [f"P{i}" for i in range(len(df))]
    )
    return probe_x, probe_y, probe_z, probe_labels


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
         gamma: float | None = None,
         probe_xyz_file: str | None = None,
         xy_render_mode: str = "scatter",
         full_range_gradient: bool | None = None,
         xy_z_blend: float | None = None,
         invert_z_gradient: bool | None = None,
         random_xy_field: bool | None = None,
         random_xy_n_blobs: int | None = None,
         random_xy_correlation_m: float | None = None,
         random_xy_seed: int | None = None,
         log_file: str | None = None,
         train_val_split_method: str | None = None,
         ws_localized_bandwidth: float | None = None,
         ws_localized_bandwidth_sweep: list[float] | None = None,
         em_footprint_diagnostic: bool = False,
         quick: bool = False,
         quick_epochs: int | None = None,
         stage: str = "full",
         calibration: str | None = None):
    """
    Run the full EA-EM pipeline.

    ``stage`` : ``"full"`` (default, one run does everything), ``"calibrate"`` or
    ``"map"`` (real data only; see stage_split.py). ``calibrate`` thins the EM
    survey to the probe neighbourhoods, inverts the FULL ensemble, runs Steps
    0-5 and writes ``stage_calibration.json`` (selected K + members + WS
    parameters + localized bandwidth) into the run folder, then stops before
    the map rendering. ``map`` reads that file (``calibration`` = file / run
    folder, default: newest ``*_calibrate`` run of the scenario), inverts ONLY
    the K selected members on the scenario's EM points (all, or grid-thinned),
    runs Steps 2-6 and reports how far K and the WS parameters drifted from the
    calibration. Own run folder suffix and cache file per stage.

    ``quick`` : smoke-test preset (see quick_preset.py): ONE alpha and ONE
    forward model out of the scenario's grid; with ``quick_epochs=k`` also
    only k evenly spaced survey epochs (real data). Cuts e.g. 280 Step-1
    tasks to 28 (12 with 3 epochs). Never touches the EM/TDR data itself,
    writes to its own ``<seq>_<scenario>_quick`` folder and its own
    ``*_quick*.pkl`` cache so a full-run cache is never overwritten.
    The ensemble is shrunk, so the result is a smoke test, not a result.

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
        scenario value (or 4 if unset / no scenario given). Ignored when
        ``probe_xyz_file`` is given.
    swc_depths : list of float or None, optional
        Overrides the scenario's ``swc_depths`` with explicit sensor
        installation depths [m]. ``None`` (default) uses the scenario
        value, or auto-derived layer-centre depths if the scenario also
        leaves it unset. Ignored when ``probe_xyz_file`` is given.
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
    probe_xyz_file : str or None, optional
        Path to a CSV of explicit SWC/TDR probe locations (one row per
        probe), e.g. ``coord_TDR.csv``. Accepted columns (case-insensitive,
        first match wins): X -> {'x', 'utm_x', 'easting'}; Y -> {'y',
        'utm_y', 'northing'}; Z -> {'z', 'depth', 'depth_m'}. When given,
        this completely replaces the auto-placement logic (the
        ``num_swc_positions`` × ``swc_depths`` grid) for BOTH 1-D and 2-D
        mode -- every row becomes one probe at its literal (x, y, z), no
        cartesian expansion. An optional label column ('id', 'name',
        'probe', 'label') is read for plot legends if present but is not
        required. ``None`` (default) keeps the existing auto-placement
        behaviour.
    full_range_gradient : bool or None, optional
        Synthetic generation only (ignored when ``homogeneous`` is True).
        When True, the static XZ (depth) and XY (plan-view) initial-model
        gradients each span the full ``[theta_dry, theta_wet]`` range
        independently (see ``data_synthetic.generate_time_lapse_
        dataset``), instead of the small fixed-amplitude default gradient.
        ``None`` (default) uses the scenario config's ``full_range_gradient``
        value (or False if unset).
    xy_z_blend : float or None, optional
        Only used when ``full_range_gradient`` is True. Weight in [0, 1]
        between the XY component (1.0) and the Z/depth component (0.0) of
        the initial model. ``None`` (default) uses the scenario value (or
        0.5, equal blend).
    invert_z_gradient : bool or None, optional
        Only used when ``full_range_gradient`` is True. When True, flips
        the depth gradient (wet at the surface, dry at depth). ``None``
        (default) uses the scenario value (or False, i.e. dry at surface).
    random_xy_field : bool or None, optional
        Only used when ``full_range_gradient`` is True. When True, replaces
        the deterministic linear XY ramp with a smooth, spatially-
        correlated RANDOM field (scattered wet/dry patches) instead of a
        single monotonic gradient direction -- see
        ``data_synthetic._generate_random_xy_field``. ``None``
        (default) uses the scenario config's ``random_xy_field`` value (or
        False if unset).
    random_xy_n_blobs : int or None, optional
        Only used when ``random_xy_field`` is True. Number of random
        Gaussian patches summed into the field. ``None`` (default) uses
        the scenario value (or 8).
    random_xy_correlation_m : float or None, optional
        Only used when ``random_xy_field`` is True. Patch radius [m]
        (1-sigma). ``None`` (default) uses the scenario value (or auto:
        ~1/4 of the survey's shorter horizontal extent).
    random_xy_seed : int or None, optional
        Only used when ``random_xy_field`` is True. Seeds the random field
        independently of the per-sample noise stream. ``None`` (default)
        uses the scenario value (or a seed derived from 42).
    log_file : str or None, optional
        Path to the shared run-log CSV that gets one appended row per run
        (see ``append_run_log``). ``None`` (default) uses
        ``DEFAULT_RUN_LOG`` (``mfc2_outputs/run_log.csv``).
    train_val_split_method : str or None, optional
        Split TDR/SWC observations into train (drives Step 2/4 fitting and
        Step 3/5 K-selection) and validation (held out, reported once in
        Step 5b) subsets — see ``train_val_split.py`` and Loiseau et al.
        (2026) Secs. 2.3/2.6. One of ``"alternate_sorted"``, ``"by_probe"``,
        ``"by_time_block"``, ``"random"``, or ``"off"``. ``None`` (default)
        falls back to the scenario config value, or ``"off"`` (legacy
        behaviour, no split) if the scenario doesn't set one either.
    ws_localized_bandwidth : float or None, optional
        Gaspari-Cohn localization radius 'c' for the Step 2 kernel-localized
        Waxman-Smits diagnostic (see the ``WS_LOCALIZED_BANDWIDTH`` block in
        Step 2). ``None`` (default) falls back to the scenario config's
        ``ws_localized_bandwidth`` field, or disables the diagnostic if that
        isn't set either. When ``ws_localized_bandwidth_sweep`` is also
        given (CLI or scenario), the sweep's recommended bandwidth takes
        precedence over this fixed value, same as scenario-only behaviour.
    ws_localized_bandwidth_sweep : list of float or None, optional
        Candidate bandwidths (c values) for the Step 2 LOCALIZED SWEEP
        leave-one-probe-out CV diagnostic. ``None`` (default) falls back to
        the scenario config's ``ws_localized_bandwidth_sweep`` field, or
        skips the sweep if that isn't set either. Overrides the scenario
        value when passed (CLI > scenario, same precedence as beta/gamma).
    em_footprint_diagnostic : bool, optional
        Compute a McNeill-type depth-weighted "effective EM theta" per
        probe (see ``em_sensitivity.py``) and report its discrepancy
        against ``theta_calc`` at the hard-routed ``layer_index`` and
        against ``theta_obs`` (Step 5c). Diagnostic only -- does not change
        Step 2/4 calibration or Step 3/5 K-selection. ``False`` by default
        (purely additive opt-in, no CLI/scenario fallback needed since
        there's no meaningful non-boolean scenario value for it).
    """
    mode = mode.lower().strip()

    # -----------------------------------------------------------------------
    # 0. PIPELINE CONFIGURATION
    # -----------------------------------------------------------------------
    if cfg is not None:
        _scenario_name    = cfg.name
        # cfg.output_dir is the configured *base* directory for this
        # scenario (and where scenario-level artifacts like the ensemble
        # cache live -- see _cache_path_override below); each run gets its
        # own fresh, numbered subfolder inside it (see next_run_output_dir)
        # instead of writing directly into -- and overwriting -- it.
        OUTPUT_DIR        = next_run_output_dir(
            cfg.output_dir,
            _scenario_name + ("_quick" if quick else "") + stage_split.stage_label(stage))
        OUTPUT_MODEL_DIR  = os.path.join(OUTPUT_DIR, "output_model")
        os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
        REALIZATION_INDEX = cfg.realization_index
        DEPTHS0           = cfg.depths0
        FIXED_DEPTHS      = cfg.fixed_depths
        PARAM_GRID        = cfg.param_grid
        COND_RANGES       = cfg.cond_ranges
        GAMMA             = cfg.gamma
        _cache_path_override = cfg.cache_path
        EM_UNIT           = em_unit if em_unit is not None else getattr(cfg, "em_unit", "ppt")
    else:
        _scenario_name    = "default"
        OUTPUT_DIR        = next_run_output_dir("mfc2_outputs", _scenario_name)
        OUTPUT_MODEL_DIR  = os.path.join(OUTPUT_DIR, "output_model")
        os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
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

    # Ensure 'gamma' lives in PARAM_GRID (mirrors 'beta') so it can be swept
    # like any other ensemble-grid parameter instead of being a single
    # fixed scalar applied to every combo. Seed it from the scenario's
    # scalar cfg.gamma (or the hardcoded default) if the scenario config
    # doesn't already provide a 'gamma' param_grid entry of its own.
    if "gamma" not in PARAM_GRID:
        PARAM_GRID = dict(PARAM_GRID)
        PARAM_GRID["gamma"] = [GAMMA]

    # --beta / --gamma CLI overrides (mirror the --window-m / --swc-depths
    # pattern: CLI value wins over whatever the scenario config set).
    # Both are now list-valued param_grid entries -- gamma is no longer a
    # single scalar applied uniformly to every ensemble combo; passing
    # multiple values sweeps gamma across combos exactly like beta does
    # (still only meaningful for time-lapse/multi-epoch inversions).
    if beta is not None:
        PARAM_GRID = dict(PARAM_GRID)
        PARAM_GRID["beta"] = list(beta)
    if gamma is not None:
        PARAM_GRID = dict(PARAM_GRID)
        PARAM_GRID["gamma"] = list(gamma)
    # GAMMA is kept only as a legacy single-value fallback for call sites
    # that still want "the" gamma (e.g. Step 6b's single "optimal model"
    # reconstruction) -- always the first value of the (possibly swept)
    # grid, same convention already used there for alpha/beta.
    GAMMA = PARAM_GRID["gamma"][0]

    # --quick: one alpha + one forward model (after the --beta/--gamma
    # overrides so those still apply). PARAM_GRID feeds the cache
    # fingerprint, so a quick run can never reuse a full-grid cache.
    QUICK_INFO = {}
    if quick:
        if quick_epochs is not None and data_source != "real":
            raise ValueError("--quick-epochs applies to real-data runs only "
                             "(synthetic epochs come from the generator).")
        PARAM_GRID, QUICK_INFO = quick_preset.reduce_param_grid(PARAM_GRID)
        print("[QUICK] Preset ON -- SMOKE TEST, the ensemble is deliberately shrunk:")
        for _axis, _i in QUICK_INFO.items():
            print(f"[QUICK]   {_axis}: {_i['n_full']} -> 1 (kept {_i['kept']})")
        if not QUICK_INFO:
            print("[QUICK]   (alpha / forwardModel already single-valued: nothing to reduce)")

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

    # Optional override of the Waxman-Smits optimizer bounds ({param: [lo, hi]},
    # subset of a/m/sigma_w/sigma_s). None keeps EMmethod.WS_BOUNDS.
    WS_BOUNDS_OVERRIDE = getattr(cfg, "ws_bounds", None) if cfg is not None else None

    # Step 1 layer pinning (condition the free layers on pinned ones). None =
    # every layer free (default). 'truth' source = ORACLE (synthetic only).
    PIN_LAYERS = getattr(cfg, "pin_layers", None) if cfg is not None else None
    PIN_SOURCE = getattr(cfg, "pin_source", "truth") if cfg is not None else "truth"

    # Homogeneous mode: disable spatial gradients in synthetic generation.
    # CLI flag (if given) overrides scenario config.
    HOMOGENEOUS = (
        homogeneous if homogeneous is not None
        else (getattr(cfg, "homogeneous", False) if cfg is not None else False)
    )

    # Full-range gradient mode: static XZ/XY initial-model gradients span
    # the full [theta_dry, theta_wet] range instead of the small default
    # offset. CLI flags (if given) override scenario config; ignored when
    # HOMOGENEOUS is True (mutually exclusive, same precedence as
    # step0.generate_time_lapse_dataset's own homogeneous/full_range_gradient
    # branching).
    FULL_RANGE_GRADIENT = (
        full_range_gradient if full_range_gradient is not None
        else (getattr(cfg, "full_range_gradient", False) if cfg is not None else False)
    )
    XY_Z_BLEND = (
        xy_z_blend if xy_z_blend is not None
        else (getattr(cfg, "xy_z_blend", 0.5) if cfg is not None else 0.5)
    )
    INVERT_Z_GRADIENT = (
        invert_z_gradient if invert_z_gradient is not None
        else (getattr(cfg, "invert_z_gradient", False) if cfg is not None else False)
    )
    RANDOM_XY_FIELD = (
        random_xy_field if random_xy_field is not None
        else (getattr(cfg, "random_xy_field", False) if cfg is not None else False)
    )
    RANDOM_XY_N_BLOBS = (
        random_xy_n_blobs if random_xy_n_blobs is not None
        else (getattr(cfg, "random_xy_n_blobs", 8) if cfg is not None else 8)
    )
    RANDOM_XY_CORRELATION_M = (
        random_xy_correlation_m if random_xy_correlation_m is not None
        else (getattr(cfg, "random_xy_correlation_m", None) if cfg is not None else None)
    )
    RANDOM_XY_SEED = (
        random_xy_seed if random_xy_seed is not None
        else (getattr(cfg, "random_xy_seed", None) if cfg is not None else None)
    )

    # Spatially-varying TRUTH Waxman-Smits params (ws_xy_variation). Unlike
    # random_xy_field above (which perturbs theta), this perturbs the WS
    # petrophysical params (a, m, sigma_w, sigma_s) themselves. Scenario-only
    # (no CLI override), same pattern as WS_FIXED_PARAMS below.
    WS_XY_VARIATION = getattr(cfg, "ws_xy_variation", False) if cfg is not None else False
    WS_XY_PARAMS = getattr(cfg, "ws_xy_params", None) if cfg is not None else None
    WS_XY_RANGES = getattr(cfg, "ws_xy_ranges", None) if cfg is not None else None
    WS_XY_PATTERN = getattr(cfg, "ws_xy_pattern", "linear") if cfg is not None else "linear"
    WS_XY_N_BLOBS = getattr(cfg, "ws_xy_n_blobs", 6) if cfg is not None else 6
    WS_XY_CORRELATION_M = getattr(cfg, "ws_xy_correlation_m", None) if cfg is not None else None
    WS_XY_SEED = getattr(cfg, "ws_xy_seed", None) if cfg is not None else None
    WS_LAYER_PARAMS = getattr(cfg, "ws_layer_params", None) if cfg is not None else None

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
    # Optional thinning of the real EM surveys (em_subsample.py); None = off.
    EM_SUBSAMPLE = getattr(cfg, "em_subsample", None) if cfg is not None else None

    # --stage calibrate / map (see stage_split.py)
    STAGE = stage or "full"
    CALIBRATION = None
    CALIBRATION_PATH = None
    if STAGE != "full":
        if data_source != "real":
            raise ValueError("--stage calibrate/map is for real-data runs only "
                             "(there is no probe<->EM thinning to split on synthetic data).")
        print(f"[STAGE] --stage {STAGE}")
    if STAGE == "calibrate":
        EM_SUBSAMPLE, _sub_replaced = stage_split.calibrate_em_subsample(EM_SUBSAMPLE)
        print(f"[STAGE] calibrate: EM points restricted to the probe neighbourhoods "
              f"({EM_SUBSAMPLE})" + (" -- replaces the scenario's em_subsample." if _sub_replaced else "."))
        print("[STAGE] calibrate: FULL ensemble; Steps 0-5, then stage_calibration.json; "
              "no map rendering (Step 6).")
    elif STAGE == "map":
        CALIBRATION_PATH = stage_split.resolve_calibration_path(
            calibration, cfg.output_dir if cfg is not None else None)
        CALIBRATION = stage_split.read_calibration(CALIBRATION_PATH)
        print(f"[STAGE] map: calibration '{CALIBRATION_PATH}' (scenario "
              f"'{CALIBRATION.get('scenario')}', created {CALIBRATION.get('created')}): "
              f"K={CALIBRATION['selected_K']} member(s) of {CALIBRATION.get('n_ensemble_total')} "
              f"will be inverted.")
        if EM_SUBSAMPLE and str(EM_SUBSAMPLE.get("mode", "off")).lower() == "probes":
            print("[STAGE][WARN] map stage with em_subsample.mode='probes': only the probe "
                  "neighbourhoods will be mapped. Use a scenario with no em_subsample (all "
                  "points) or mode 'grid' for the maps.")

    # Train/validation split (see train_val_split.py and Loiseau et al.,
    # 2026 Secs. 2.3/2.6). "off" preserves the legacy behaviour of fitting
    # and selecting K on the full observation set. Resolved here (CLI >
    # scenario > "off") so it's available both to the reporter header and
    # to the split logic right before Step 2.
    TRAIN_VAL_SPLIT = (
        train_val_split_method if train_val_split_method is not None
        else (getattr(cfg, "train_val_split", "off") if cfg is not None else "off")
    )

    # Initialise run reporter
    # (_scenario_name was already set above, alongside OUTPUT_DIR)
    LOG_FILE = log_file or DEFAULT_RUN_LOG
    # Announce the resolved output_dir on stdout so a wrapper script (e.g.
    # run_pipeline.sh) can pick it up and know exactly where this run's
    # files are landing, instead of re-deriving/guessing it independently
    # (which can drift out of sync with what scenario_loader actually
    # resolved -- see next_run_output_dir).
    print(f"[OUTPUT_DIR] {OUTPUT_DIR}")
    reporter = PipelineReporter(
        output_dir=OUTPUT_DIR,
        scenario_name=_scenario_name,
        mode=mode,
        cfg=cfg,
        data_source=data_source,
        em_dir=em_dir,
        tdr_dir=tdr_dir,
        train_val_split=TRAIN_VAL_SPLIT,
    )
    # Tag the output folder with this run's id immediately (before any
    # pipeline step runs) so it's identifiable even if the run crashes
    # early. See write_run_marker() for why this -- not the run_log CSV --
    # is what lives in the folder itself.
    write_run_marker(OUTPUT_DIR, reporter.run_id, reporter.report_path)
    reporter.set_manifest_extra(
        stage=STAGE,
        seeds={"inversion_seed": DEFAULT_INVERSION_SEED,
               "random_xy_seed": getattr(cfg, "random_xy_seed", None),
               "ws_xy_seed": getattr(cfg, "ws_xy_seed", None)},
    )
    reporter.args_table("Runtime Arguments", {
        "mode":               mode.upper(),
        "scenario":           _scenario_name,
        "output_dir":         OUTPUT_DIR,
        "eca_noise_level":    f"{ECA_NOISE_LEVEL*100:.1f}%",
        "tdr_noise_std":      f"{TDR_NOISE_STD*100:.1f}% vol",
        "realization_index":  REALIZATION_INDEX,
        "fixed_depths":       FIXED_DEPTHS,
        "gamma":              str(PARAM_GRID.get("gamma", [GAMMA])),
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
        "ws_bounds":          str(WS_BOUNDS_OVERRIDE) if WS_BOUNDS_OVERRIDE else "(defaults — EMmethod.WS_BOUNDS)",
        "lab_p0":             str(LAB_P0) if LAB_P0 else "(none — using DEFAULT_WS_P0 = [1.5, 1.8, 50.0, 5.0])",
        "homogeneous":        HOMOGENEOUS,
        "spatial_match_tolerance_m (real time-lapse)": SPATIAL_MATCH_TOLERANCE_M,
        "em_subsample (real)": str(EM_SUBSAMPLE) if EM_SUBSAMPLE else "(off)",
        "quick preset": (
            f"ON -- SMOKE TEST (alpha/forwardModel reduced: {QUICK_INFO or 'n/a'}; "
            f"epochs: {quick_epochs if quick_epochs else 'all'})"
            if quick else "(off)"
        ),
        "stage": (
            STAGE if STAGE == "full" else
            f"{STAGE}" + (f" (calibration: {CALIBRATION_PATH})" if CALIBRATION_PATH else "")
        ),
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
            # Raw-ECa preprocessing chain config (see preprocess_em_survey()) --
            # scenario-overridable, defined here so the same values can be
            # reused below when logging the per-file results to the report.
            FILTER_REPEATED_TOLERANCE = getattr(cfg, "filter_repeated_tolerance", 1e-10)
            FILTER_RANGE_VMIN = getattr(cfg, "filter_range_vmin", 0.0)
            FILTER_RANGE_VMAX = getattr(cfg, "filter_range_vmax", None)
            FILTER_PERCENTILE_QMIN = getattr(cfg, "filter_percentile_qmin", 5.0)
            FILTER_PERCENTILE_QMAX = getattr(cfg, "filter_percentile_qmax", 95.0)
            ROLLING_MEAN_WINDOW = getattr(cfg, "rolling_mean_window", 5)

            EM_FNAMES, observations_list, probe_table, EM_DATES, RAW_EM_FNAMES, REAL_DATA_DIAG = _load_real_data(
                em_dir, tdr_dir, coord_file,
                time_tolerance_days=getattr(cfg, "time_tolerance_days", 1.0),
                em_unit=EM_UNIT,
                filter_repeated_tolerance=FILTER_REPEATED_TOLERANCE,
                filter_range_vmin=FILTER_RANGE_VMIN,
                filter_range_vmax=FILTER_RANGE_VMAX,
                filter_percentile_qmin=FILTER_PERCENTILE_QMIN,
                filter_percentile_qmax=FILTER_PERCENTILE_QMAX,
                rolling_mean_window=ROLLING_MEAN_WINDOW,
                drop_coils=DROP_COILS,
                em_subsample_cfg=EM_SUBSAMPLE,
                quick=quick, quick_epochs=quick_epochs,
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

            # -----------------------------------------------------------------------
            # Fold the raw-ECa preprocessing chain (filterRange -> filterRepeated ->
            # filterPercentile -> rollingMean, see preprocess_em_survey()) into the
            # Markdown run report -- same "return structured, don't just print"
            # pattern as coord_kinds / em_unit_checks above, so the report shows
            # exactly how many points each stage dropped per survey file.
            # -----------------------------------------------------------------------
            _filter_rows = [
                [
                    s["file"],
                    ", ".join(s.get("dropped_coils") or []) or "—",
                    f"{s['n_initial']} → {s['n_after_range']} (-{s['range_removed']})",
                    f"{s['n_after_range']} → {s['n_after_repeated']} (-{s['repeated_removed']})",
                    f"{s['n_after_repeated']} → {s['n_after_percentile']} (-{s['percentile_removed']})",
                    f"window={s['rolling_window']}" if s["rolling_mean_applied"] else "skipped",
                    s["n_final"],
                ]
                for s in REAL_DATA_DIAG["filter_stats"]
            ]
            reporter.table(
                "Step 0 — EM raw-ECa preprocessing chain per file",
                ["File", "Dropped coil(s)", "filterRange", "filterRepeated",
                 "filterPercentile", "rollingMean", "Final pts"],
                _filter_rows,
                note=(
                    f"drop_coils={DROP_COILS if DROP_COILS else '(none)'} -> "
                    f"filterRange[{FILTER_RANGE_VMIN},{FILTER_RANGE_VMAX}] -> "
                    f"filterRepeated(tol={FILTER_REPEATED_TOLERANCE}) -> "
                    f"filterPercentile[{FILTER_PERCENTILE_QMIN},{FILTER_PERCENTILE_QMAX}] -> "
                    f"rollingMean(window={ROLLING_MEAN_WINDOW}), applied once per file at "
                    "load time (not redundantly inside every parallel inversion worker). "
                    "Dropped coil(s) are excluded before filterRange runs, so a "
                    "soon-to-be-dropped channel's own readings can't veto an otherwise-"
                    "good point on the remaining coils; the CSV itself is untouched, and "
                    "the per-worker drop before inversion still happens separately."
                ),
            )
            _total_removed = sum(
                s["range_removed"] + s["repeated_removed"] + s["percentile_removed"]
                for s in REAL_DATA_DIAG["filter_stats"]
            )
            reporter.metric("Total EM points removed by preprocessing chain", _total_removed)

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
            # FIX ("TDR noise weighting silently disabled for real-data runs"):
            # this used to unconditionally force TDR_NOISE_STD to 0.0 here,
            # discarding whatever value cfg.tdr_noise_std resolved to just
            # above (line ~1687) before it ever reached Step 2's
            # theta_obs_std weighting. A scenario that explicitly sets
            # tdr_noise_std (e.g. '2d_real': 0.01) now has that value flow
            # through for real data exactly as it already does for
            # synthetic data. A real-data scenario that does NOT set
            # tdr_noise_std still falls back to the 0.01 default from line
            # ~1687 (previously silently disabled instead) -- if that
            # default doesn't reflect a real per-probe noise estimate for a
            # given survey, set tdr_noise_std explicitly in that scenario
            # (or to 0.0/null) to disable weighting on purpose.

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
                full_range_gradient=FULL_RANGE_GRADIENT,
                xy_z_blend=XY_Z_BLEND,
                invert_z_gradient=INVERT_Z_GRADIENT,
                random_xy_field=RANDOM_XY_FIELD,
                random_xy_n_blobs=RANDOM_XY_N_BLOBS,
                random_xy_correlation_m=RANDOM_XY_CORRELATION_M,
                random_xy_seed=RANDOM_XY_SEED,
                ws_xy_variation=WS_XY_VARIATION,
                ws_xy_params=WS_XY_PARAMS,
                ws_xy_ranges=WS_XY_RANGES,
                ws_xy_pattern=WS_XY_PATTERN,
                ws_xy_n_blobs=WS_XY_N_BLOBS,
                ws_xy_correlation_m=WS_XY_CORRELATION_M,
                ws_xy_seed=WS_XY_SEED,
                ws_layer_params=WS_LAYER_PARAMS,
            )
            step0.plot_synthetic_results(output_dir=OUTPUT_DIR)
            # Truth vs initial WS parameters per horizon (never aborts the run).
            try:
                _ws_fig = ws_distribution.plot_ws_truth_vs_initial(
                    OUTPUT_DIR, lab_p0=LAB_P0, ws_fixed_params=WS_FIXED_PARAMS,
                    ws_truth_params=synthetic_geometry.get("ws_truth_params"),
                )
                reporter.log(f"[STEP 0] WS truth vs initial per horizon -> {_ws_fig}")
            except Exception as _exc:  # plotting only
                reporter.log(f"[STEP 0][WARN] ws_distribution plot failed: {_exc!r}")
            if WS_XY_VARIATION:
                _ws_xy_map_path = os.path.join(OUTPUT_DIR, "synthetic_ws_truth_xy_maps.png")
                reporter.log(
                    f"[STEP 0] WS truth XY map(s) -> {_ws_xy_map_path}"
                    if os.path.exists(_ws_xy_map_path) else
                    "[STEP 0][WARN] ws_xy_variation is on but "
                    "synthetic_ws_truth_xy_maps.png was not produced -- check "
                    "plot_ws_truth_xy_maps for errors."
                )

            # ------------------------------------------------------------
            # Detailed truth-generation report: how theta was built spatially/
            # temporally, and which Waxman-Smits parameters were used to turn
            # theta into EC_bulk (and, via emagpy's forward model, into ECa).
            # ------------------------------------------------------------
            _gradient_mode = synthetic_geometry.get("gradient_mode", "unknown")
            reporter.log(f"[STEP 0] Theta generation mode : {_gradient_mode}")
            if HOMOGENEOUS:
                reporter.log(
                    "[STEP 0]   theta is IDENTICAL across all positions and layers at "
                    "each timestep -- only a temporal wetting trend varies."
                )
            elif not FULL_RANGE_GRADIENT and "infiltration" not in _gradient_mode:
                reporter.log(
                    "[STEP 0]   theta varies via small fixed-amplitude XY/XZ gradients "
                    "(dry at surface/x=0, wetter with depth/x) plus a temporal trend."
                )

            _ws_truth = synthetic_geometry.get("ws_truth_params", {})
            if _ws_truth:
                _same_everywhere = len({tuple(sorted(p.items())) for p in _ws_truth.values()}) == 1
                _truth_rows: dict[str, str] = {}
                for _l, _p in _ws_truth.items():
                    _truth_rows[f"Layer {_l} — a / m / σ_w / σ_s"] = (
                        f"{_p['a']:.4f} / {_p['m']:.4f} / "
                        f"{_p['sigma_w']:.2f} mS/m / {_p['sigma_s']:.2f} mS/m"
                    )
                reporter.args_table(
                    "Step 0 — Synthetic ground-truth Waxman-Smits parameters "
                    "(EC_bulk = theta^m / a * (sigma_w + sigma_s))"
                    + ("  [layer baseline -- see note + XY map below]" if WS_XY_VARIATION else ""),
                    _truth_rows,
                )
                if WS_XY_VARIATION:
                    _wsxy = synthetic_geometry.get("ws_xy_variation_params", {})
                    reporter.log(
                        "[STEP 0]   ws_xy_variation is ON: the table above is only the "
                        f"per-layer BASELINE. {_wsxy.get('varying_params', '(?)')} actually "
                        f"vary continuously in XY per point (pattern="
                        f"'{_wsxy.get('pattern', '?')}', ranges={_wsxy.get('ranges', '?')}) -- "
                        "the true per-point-per-layer values used are saved to "
                        "synthetic_ws_truth_{a,m,sigma_w,sigma_s}.npy and mapped in "
                        "synthetic_ws_truth_xy_maps.png, NOT summarized by this table."
                    )
                else:
                    reporter.log(
                        "[STEP 0]   " + (
                            "Truth (a, m, sigma_w, sigma_s) are IDENTICAL across every layer, "
                            "spatial position, ensemble member, and time step -- ONLY theta "
                            "varies spatially/temporally (see theta generation mode above). "
                            "There is currently no spatial or per-horizon heterogeneity in the "
                            "truth petrophysics."
                            if _same_everywhere else
                            "Truth (a, m, sigma_w, sigma_s) DIFFER across layers -- see table above."
                        )
                    )
                # Flag mismatch against the calibration starting point (prior),
                # since Step 2/4 calibrate against LAB_P0 / DEFAULT_WS_P0, not
                # against these truth values.
                _prior = LAB_P0 if LAB_P0 else [1.5, 1.8, 50.0, 5.0]
                if isinstance(_prior[0], (list, tuple)):
                    _prior0 = _prior[0]
                else:
                    _prior0 = _prior
                _truth0 = _ws_truth.get(0, next(iter(_ws_truth.values())))
                reporter.log(
                    f"[STEP 0]   Calibration prior/start (a, m, sigma_w, sigma_s) = "
                    f"({_prior0[0]:.2f}, {_prior0[1]:.2f}, {_prior0[2]:.1f} mS/m, "
                    f"{_prior0[3]:.1f} mS/m) vs. "
                    f"{'layer-baseline TRUTH (NOT the true spatial value -- see note above)' if WS_XY_VARIATION else 'TRUTH'} "
                    f"({_truth0['a']:.2f}, "
                    f"{_truth0['m']:.2f}, {_truth0['sigma_w']:.1f} mS/m, "
                    f"{_truth0['sigma_s']:.1f} mS/m) "
                    f"{'-- MISMATCH: prior is far from truth, calibration must recover it via optimization' if tuple(round(x,2) for x in _prior0) != (round(_truth0['a'],2), round(_truth0['m'],2), round(_truth0['sigma_w'],2), round(_truth0['sigma_s'],2)) else '-- prior matches truth'}"
                )

            if synthetic_geometry.get("cond_ranges_input") is not None and not synthetic_geometry.get("cond_ranges_used", True):
                reporter.log(
                    f"[STEP 0][WARN] cond_ranges={synthetic_geometry['cond_ranges_input']} was "
                    f"passed but is NOT used by the truth generator (dead parameter, reserved "
                    f"for a future spatially-heterogeneous petrophysics mode)."
                )

            _gen_meta_rows = {
                "Ensemble size (truth realizations)": synthetic_geometry.get("ensemble_size"),
                "Time steps": synthetic_geometry.get("n_time"),
                "Point spacing [m]": synthetic_geometry.get("point_spacing"),
                "theta_dry / theta_wet [m³/m³]": (
                    f"{synthetic_geometry.get('theta_dry')} / {synthetic_geometry.get('theta_wet')}"
                    if FULL_RANGE_GRADIENT or "infiltration" in _gradient_mode else "(not used by this gradient mode)"
                ),
            }
            if mode == "2d":
                _gen_meta_rows["Grid size (num_x × num_y)"] = (
                    f"{synthetic_geometry.get('num_x')} × {synthetic_geometry.get('num_y')} "
                    f"= {synthetic_geometry.get('num_x', 0) * synthetic_geometry.get('num_y', 0)} points"
                )
            if FULL_RANGE_GRADIENT and "full_range_gradient_params" in synthetic_geometry:
                _frg = synthetic_geometry["full_range_gradient_params"]
                _gen_meta_rows["xy_z_blend (1=XY only, 0=Z only)"] = _frg["xy_z_blend"]
                _gen_meta_rows["invert_z_gradient"] = _frg["invert_z_gradient"]
                _gen_meta_rows["random_xy_field"] = _frg["random_xy_field"]
            if FULL_RANGE_GRADIENT and "random_xy_params" in synthetic_geometry:
                _rxy = synthetic_geometry["random_xy_params"]
                _gen_meta_rows["Random XY field: n_blobs / correlation [m] / seed"] = (
                    f"{_rxy['n_blobs']} / {_rxy['correlation_m']:.2f} / {_rxy['seed']}"
                )
            if WS_XY_VARIATION and "ws_xy_variation_params" in synthetic_geometry:
                _wsxy = synthetic_geometry["ws_xy_variation_params"]
                _gen_meta_rows["WS xy variation: params / ranges / pattern"] = (
                    f"{_wsxy['varying_params']} / {_wsxy['ranges']} / {_wsxy['pattern']}"
                )
                if _wsxy["pattern"] == "random":
                    _gen_meta_rows["WS xy field: n_blobs / correlation [m] / seed"] = (
                        f"{_wsxy['n_blobs']} / {_wsxy['correlation_m']:.2f} / {_wsxy['seed']}"
                    )
            if "infiltration_params" in synthetic_geometry:
                _inf = synthetic_geometry["infiltration_params"]
                _gen_meta_rows["Wetting front z0 / speed / width [m]"] = (
                    f"{_inf['front_z0']} / {_inf['front_speed']} per step / {_inf['front_width']}"
                )
            reporter.args_table("Step 0 — Synthetic dataset generation settings", _gen_meta_rows)

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

            _moved_npy = _collect_npy_into_output_model(OUTPUT_DIR, OUTPUT_MODEL_DIR)
            if _moved_npy:
                reporter.log(
                    f"[STEP 0] Moved {len(_moved_npy)} .npy array(s) into "
                    f"'{OUTPUT_MODEL_DIR}': "
                    + ", ".join(os.path.basename(p) for p in _moved_npy)
                )

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
                # FIX: no longer load/pass the absolute-UTM
                # synthetic_utm_x/y grid here -- probe placement inside
                # run_pre_inversion_diagnostics_synthetic_from_pipeline now
                # always uses the survey's own local coordinates, so the
                # separate (and previously mismatched) absolute-UTM grid is
                # not needed for this diagnostic.
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
                    # FIX: without this, the diagnostic always fabricated its
                    # own auto-placed probes (num_swc_positions x swc_depths)
                    # even when the real run was configured with an explicit
                    # --probe-xyz file -- so probe_em_matching_pre_inversion_
                    # synthetic.png showed the wrong probe COUNT and the
                    # wrong LOCATIONS relative to what Step 1/6 actually use.
                    probe_xyz_file=probe_xyz_file,
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
        if quick:
            # Own cache file: never clobber (or be mistaken for) the full-run cache.
            cache_path = quick_preset.quick_cache_path(cache_path, quick_epochs)
            print(f"[QUICK] Using separate cache file: '{cache_path}'")
        if STAGE != "full":
            cache_path = stage_split.stage_cache_path(cache_path, STAGE)
            print(f"[STAGE] Using separate cache file: '{cache_path}'")

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
            DEPTHS0, FIXED_DEPTHS, PARAM_GRID, COND_RANGES,
            sorted(PARAM_GRID.get("gamma", [GAMMA])), DROP_COILS,
            spatial_match_tolerance_m=_fingerprint_tolerance,
            pin_layers=PIN_LAYERS, pin_source=PIN_SOURCE,
            em_subsample=EM_SUBSAMPLE if data_source == "real" else None,
            epochs=(
                REAL_DATA_DIAG["epoch_indices"]
                if (quick and quick_epochs is not None and data_source == "real"
                    and REAL_DATA_DIAG is not None)
                else None
            ),
            members=(stage_split.record_member_keys(CALIBRATION) if CALIBRATION else None),
        )

        if STAGE == "map":
            # Refuse an incompatible calibration BEFORE any inversion (or cache reuse).
            _errs, _warns = stage_split.check_compatibility(
                CALIBRATION, depths0=DEPTHS0, cond_ranges=COND_RANGES,
                em_files=(list(EM_FNAMES) if timelapse else [EM_FNAMES]),
                scenario=_scenario_name)
            for _w in _warns:
                print(f"[STAGE][WARN] {_w}")
            if _errs:
                raise ValueError("--stage map: calibration is incompatible with this run: "
                                 + " | ".join(_errs))

        # Real-data time-lapse spatial reference grid coordinates (union
        # across all survey epochs -- see geometry_utils.match_epochs_to_
        # reference_grid). None for synthetic data and for real single-survey
        # (non-time-lapse) runs, where no cross-epoch spatial matching is
        # needed. Populated either from a cache hit below or freshly built
        # in the "no cache" branch further down.
        REAL_DATA_REFERENCE_XY = None
        # Companion to REAL_DATA_REFERENCE_XY: per-epoch mapping from each
        # epoch's own raw-CSV row order onto the reference-grid index
        # (matched_indices[t_idx][j] = reference-grid index for that
        # epoch's row j, or -1 if unmatched). Same lifetime/caching as
        # REAL_DATA_REFERENCE_XY -- see geometry_utils.
        # match_epochs_to_reference_grid. Threaded through to Step 6e
        # (forward_simulate_eca_optimal) so raw per-epoch ECa observations
        # can be gathered in reference-grid order for comparison against
        # averaged_models[K-1], which lives on the reference grid --
        # INDEXING_AUDIT_REPORT.md finding #5.
        REAL_DATA_MATCHED_INDICES = None

        cache_hit = False
        # FIX: the ensemble cache is now used ONLY for real-data runs. It
        # caused a stale-cache bug for synthetic runs (a 1D/10-point cache
        # was silently reused by a later 2D/50-point run because the cache
        # fingerprint didn't cover mode/grid size), and synthetic inversions
        # are already fast enough that re-running is cheap and always
        # correct. Real-data caching stays as-is (unaffected by this bug and
        # genuinely expensive to recompute).
        if data_source != "real":
            print("[CACHE] Skipped for synthetic data (cache is real-data-only "
                  "-- always running a fresh inversion).")
        elif os.path.exists(cache_path):
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
                REAL_DATA_MATCHED_INDICES = cache_data.get("matched_indices")
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
                elif (data_source == "real" and timelapse
                      and REAL_DATA_REFERENCE_XY is not None
                      and REAL_DATA_MATCHED_INDICES is None):
                    print(
                        "[CACHE][WARN] Cache predates matched_indices capture "
                        "(INDEXING_AUDIT_REPORT.md finding #5) and has no stored "
                        "epoch->reference-grid mapping. Step 6e "
                        "(forward_simulate_eca_optimal) will fall back to raw "
                        "per-epoch CSV row order for observed ECa, which is only "
                        "correct if that order already matches the reference "
                        "grid -- delete this cache and re-run to populate it."
                    )
                print(f"[CACHE] Loaded {models_clean.shape[0]} ensemble members "
                      f"(config fingerprint verified: {current_fingerprint}).")
                cache_hit = True

        if not cache_hit:
            print("[CACHE] No valid cache found. Running parallel inversion tasks...")

            # Pre-filter snapshot written by _load_real_data (real data only;
            # synthetic EM files have none -> None).
            def _prefilter_path(_f):
                _p = os.path.join(os.path.dirname(_f), "_prefilter", os.path.basename(_f))
                return _p if os.path.exists(_p) else None

            # RAW plot: genuinely pre-filter signal when a snapshot exists
            # (EM_FNAMES themselves are already filtered in place), with
            # dropped coils flagged in red rather than hidden.
            if timelapse:
                for t_idx, fname in enumerate(EM_FNAMES):
                    pre_inversion.plot_raw_apparent_conductivity(
                        _prefilter_path(fname) or fname,
                        os.path.join(OUTPUT_DIR, f"t{t_idx:02d}"),
                        em_unit=EM_UNIT, drop_coils=DROP_COILS,
                    )
            else:
                pre_inversion.plot_raw_apparent_conductivity(
                    _prefilter_path(EM_FNAMES) or EM_FNAMES, OUTPUT_DIR,
                    em_unit=EM_UNIT, drop_coils=DROP_COILS,
                )

            # Same transect view AFTER the load-time filter chain, with
            # dropped coils removed and the pre-filter signal in grey.

            if timelapse:
                for t_idx, fname in enumerate(EM_FNAMES):
                    pre_inversion.plot_filtered_apparent_conductivity(
                        fname, os.path.join(OUTPUT_DIR, f"t{t_idx:02d}"),
                        em_unit=EM_UNIT, drop_coils=DROP_COILS,
                        prefilter_fname=_prefilter_path(fname),
                    )
            else:
                pre_inversion.plot_filtered_apparent_conductivity(
                    EM_FNAMES, OUTPUT_DIR, em_unit=EM_UNIT,
                    drop_coils=DROP_COILS,
                    prefilter_fname=_prefilter_path(EM_FNAMES),
                )

            alphas         = PARAM_GRID.get("alpha",        [0.03])
            betas          = PARAM_GRID.get("beta",         [0.0])
            gammas         = PARAM_GRID.get("gamma",        [GAMMA])
            methods        = PARAM_GRID.get("method",       ["L-BFGS-B"])
            forward_models = PARAM_GRID.get("forwardModel", ["FSlin"])
            combinations   = [(a, b, g, m, f) for a in alphas for b in betas
                              for g in gammas for m in methods for f in forward_models]
            if len(gammas) > 1 and not timelapse:
                print(f"[WARN] Sweeping {len(gammas)} gamma values on a "
                      f"non-time-lapse (single-epoch) run -- gamma only "
                      f"affects time-lapse inversions, so these combos will "
                      f"produce identical results and just waste compute.")

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
            bnds_list = _build_inversion_bnds(DEPTHS0, fixed_depths_list, COND_RANGES,
                                              fixed_conds=PIN_LAYERS)
            print(f"[INFO] Inversion bnds (depths+conds, {len(bnds_list)} params): {bnds_list}")

            # ---- Optional layer pinning (oracle: pinned EC = synthetic truth) ----
            # Pinned layers keep a per-point EC (n_points, n_layers matrix passed
            # to EMagPy setInit) and are excluded from the optimisation, so the
            # free deeper layers are inverted conditioned on them. The starting-
            # conductivity grid is collapsed over pinned layers (identical
            # starts would only duplicate tasks).
            _pin_truth_ec = None
            if PIN_LAYERS is not None:
                if data_source != "synthetic":
                    raise ValueError("pin_layers with pin_source='truth' needs synthetic data "
                                     "(there is no ground-truth EC for real data).")
                _ec_truth_path = _resolve_npy_path("synthetic_time_lapse_ec.npy",
                                                   OUTPUT_MODEL_DIR, OUTPUT_DIR)
                if not os.path.exists(_ec_truth_path):
                    raise FileNotFoundError(f"pin_layers needs {_ec_truth_path} (written by Step 0).")
                _pin_truth_ec = np.load(_ec_truth_path)[REALIZATION_INDEX]
                if _pin_truth_ec.ndim == 2:
                    _pin_truth_ec = _pin_truth_ec[np.newaxis, ...]
                _n_layers_pin = len(DEPTHS0) + 1
                if _pin_truth_ec.shape[-1] != _n_layers_pin:
                    raise ValueError(f"Truth EC has {_pin_truth_ec.shape[-1]} layers, "
                                     f"expected {_n_layers_pin}.")
                _pinned_idx = [i for i, f in enumerate(PIN_LAYERS) if f]
                _seen, _dedup = set(), []
                for _cs in cond_sets:
                    _cs = list(_cs)
                    for _i in _pinned_idx:
                        _cs[_i] = float(COND_RANGES[_i][0])
                    if tuple(_cs) not in _seen:
                        _seen.add(tuple(_cs)); _dedup.append(_cs)
                print(f"[INFO] Layer pinning (ORACLE truth EC) on layers {_pinned_idx}: "
                      f"{len(cond_sets)} -> {len(_dedup)} starting-conductivity combinations.")
                cond_sets = _dedup

            combo_list = []
            for alpha, beta, gamma_val, method, f_model in combinations:
                for conds0 in cond_sets:
                    combo_list.append((alpha, beta, gamma_val, method, f_model, list(conds0)))

            files_to_process = EM_FNAMES if timelapse else [EM_FNAMES]
            if STAGE == "map":
                _n_grid = len(combo_list)
                combo_list, _missing = stage_split.select_members(combo_list, CALIBRATION)
                if _missing:
                    raise ValueError(
                        f"--stage map: {len(_missing)} calibrated member(s) are not in this "
                        f"scenario's parameter grid (grid/scenario changed since the "
                        f"calibration?): {_missing[:3]}{' ...' if len(_missing) > 3 else ''}")
                print(f"[STAGE] map: ensemble cut from {_n_grid} to {len(combo_list)} "
                      f"member(s) (the K={CALIBRATION['selected_K']} selected by the calibration).")
            tasks = []
            task_meta = {}          # task_idx -> dict describing the combo, for failure reporting
            global_task_idx = 0
            for c_idx, (alpha, beta, gamma_val, method, f_model, conds0) in enumerate(combo_list):
                for t_idx, file_path in enumerate(files_to_process):
                    invert_kwargs = {
                        "alpha": alpha, "beta": beta, "gamma": gamma_val,
                        "method": method,
                        "forwardModel": f_model, "targetProjection": None,
                        "em_unit": EM_UNIT, "drop_coils": DROP_COILS,
                        "bnds": bnds_list,
                    }
                    if _pin_truth_ec is not None:
                        _truth_t = _pin_truth_ec[min(t_idx, _pin_truth_ec.shape[0] - 1)]
                        _mat = np.tile(np.asarray(conds0, dtype=float), (_truth_t.shape[0], 1))
                        _mat[:, _pinned_idx] = _truth_t[:, _pinned_idx]
                        invert_kwargs["fixed_conds"] = list(PIN_LAYERS)
                        invert_kwargs["conds0_matrix"] = _mat
                    tasks.append((global_task_idx, file_path, DEPTHS0, conds0,
                                   fixed_depths_list, invert_kwargs))
                    task_meta[global_task_idx] = {
                        "combo_idx": c_idx, "t_idx": t_idx, "file": file_path,
                        "alpha": alpha, "beta": beta, "gamma": gamma_val,
                        "method": method,
                        "forwardModel": f_model,
                        "conds0": conds0,
                    }
                    global_task_idx += 1

            if quick:
                print(f"[QUICK] {len(tasks)} Step-1 tasks "
                      f"({len(combinations)} grid combo(s) x {len(cond_sets)} starting "
                      f"combo(s) x {len(files_to_process)} epoch(s)).")
            print(f"[INFO] Dispatched {len(tasks)} tasks to parallel pool...")

            unordered_results = []
            failed_tasks = []       # list of (task_idx, error_message)
            # FIX (same "stuck at 100%" issue): use a fresh `spawn` context
            # instead of the platform-default `fork` on Linux. `spawn`
            # starts each worker as a clean interpreter that re-imports
            # only what it needs, so it can never inherit half-initialised
            # state (matplotlib backend, BLAS/OpenMP thread pools, open
            # file handles, etc.) from this parent process -- eliminating
            # this whole class of "all tasks finished but the pool won't
            # shut down" hangs, not just the matplotlib instance of it.
            # Costs a bit more per-worker startup time; negligible next to
            # the ~11-minute inversion run itself.
            with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as executor:
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

            for c_idx, (alpha, beta, gamma_val, method, f_model, conds0) in enumerate(combo_list):
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
                        "alpha": alpha, "beta": beta, "gamma": gamma_val,
                        "method": method,
                        "forwardModel": f_model, "conds0": conds0,
                    })
                else:
                    discarded_combos.append({
                        "combo_idx": c_idx, "alpha": alpha, "beta": beta,
                        "gamma": gamma_val, "method": method,
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
            # step4_select.py) treats an unmatched cell as "no data"
            # rather than silently pairing it with an unrelated location.
            #
            # FIX: previously synthetic data used a SEPARATE, hand-rolled
            # `arr[:min_pts]` row-POSITION trim here instead of this
            # coordinate-based matcher, on the assumption that synthetic
            # epochs always have identical point count AND identical row
            # order (true today, by construction of 0_generate_synthetic_
            # data.py -- but a fragile assumption to keep re-verifying by
            # eye every time that generator changes). Synthetic epoch
            # coordinates ARE bit-identical across timesteps, so the same
            # coordinate matcher used for real data gives an EXACT,
            # provably-correct match for synthetic too (an
            # essentially-zero tolerance is enough, vs. the coarser
            # tolerance real GPS noise needs) -- so there is no longer any
            # reason to keep a second, weaker code path for synthetic runs.
            # ------------------------------------------------------------------
            if timelapse:
                _reference_grid_tolerance = (
                    SPATIAL_MATCH_TOLERANCE_M if data_source == "real"
                    # Synthetic coordinates are the same float array written
                    # to every epoch's CSV (see data_synthetic.py
                    # export_eca_surveys_to_csv) -- only float round-trip
                    # noise, not real positional uncertainty, needs covering.
                    else 1e-6
                )
                reporter.log(
                    f"[STEP1] Building spatial reference grid across {n_time} "
                    f"{'real-data' if data_source == 'real' else 'synthetic'} "
                    f"survey epochs (tolerance={_reference_grid_tolerance} m) "
                    f"instead of trimming to a common point COUNT -- see "
                    f"'spatial reference-grid matching, not row trimming'."
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
                    epoch_xy, tolerance_m=_reference_grid_tolerance,
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
                    "Spatial reference grid — tolerance (m)", _reference_grid_tolerance
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
                REAL_DATA_MATCHED_INDICES = matched_indices

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

            # Cache is real-data-only (see the [CACHE] skip above) -- never
            # write a synthetic-run cache that a later run could stumble
            # into loading.
            if data_source == "real":
                print(f"[CACHE] Saving matrix {models_clean.shape} to '{cache_path}' "
                      f"(config fingerprint: {current_fingerprint})")
                _cache_dir = os.path.dirname(cache_path)
                if _cache_dir:
                    os.makedirs(_cache_dir, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump({
                        "models_clean": models_clean,
                        "param_metadata": param_metadata,
                        "depths_clean": depths_clean,
                        "pstds_clean": pstds_clean,
                        "config_fingerprint": current_fingerprint,
                        "reference_xy": REAL_DATA_REFERENCE_XY,
                        "matched_indices": REAL_DATA_MATCHED_INDICES,
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
                    # .get() fallback for caches written before gamma became
                    # a per-combo swept value -- old param_metadata entries
                    # won't have a "gamma" key, so fall back to the current
                    # (first-value) GAMMA in that case.
                    "gamma": meta.get("gamma", GAMMA),
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
                pstd_npy_path = os.path.join(OUTPUT_MODEL_DIR, "mcmc_conductivity_pstd.npy")
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
        # EM DEPTH-SENSITIVITY WEIGHTS (McNeill 1980 closed-form kernel)
        #   Computed against EFFECTIVE_DEPTHS0, not the static DEPTHS0 --
        #   EFFECTIVE_DEPTHS0 is what TDR probe-to-horizon routing (below,
        #   `EMmethod.extract_sample_positions_from_survey` /
        #   `layer_index_from_depth`) actually uses, so the weights need to
        #   describe the SAME layer boundaries the hard-routed comparison
        #   they're diagnosed against is using. Using the unmoved DEPTHS0
        #   here would silently disagree with EFFECTIVE_DEPTHS0 whenever
        #   layer interfaces were free during Step 1.
        #
        #   Real vs. synthetic coil geometry are NOT the same (see
        #   em_sensitivity.py module docstring): real-data channels are one
        #   fixed geometry (HCP, 1.66 m) at six frequencies, so a single
        #   weight vector is exact. Synthetic scenarios instead sweep six
        #   distinct (orientation, spacing) coils feeding the SAME
        #   inversion (`step0.COILS_LIST`) -- there's no single physically
        #   correct way to collapse those into one kernel, so the synthetic
        #   branch uses `layer_weights_multi_coil`'s explicitly-labelled
        #   equal-weighted approximation rather than silently defaulting to
        #   the real-data HCP1.66 geometry (which would be a different,
        #   wrong, geometry for synthetic runs).
        # -----------------------------------------------------------------------
        if data_source == "real":
            _COIL_SPACING_M, _COIL_ORIENTATION = 1.66, "HCP"
            EM_LAYER_WEIGHTS = emsens.layer_weights(
                EFFECTIVE_DEPTHS0, coil_spacing_m=_COIL_SPACING_M, orientation=_COIL_ORIENTATION,
            )
            reporter.log(
                f"[EM_SENS] McNeill {_COIL_ORIENTATION} weights (spacing={_COIL_SPACING_M} m) "
                f"over layers {EFFECTIVE_DEPTHS0} + half-space: "
                f"{np.round(EM_LAYER_WEIGHTS, 4).tolist()}"
            )
        else:
            EM_LAYER_WEIGHTS = emsens.layer_weights_multi_coil(
                EFFECTIVE_DEPTHS0, step0.COILS_LIST,
            )
            reporter.log(
                f"[EM_SENS] McNeill multi-coil AVERAGE weights (equal-weighted across "
                f"{step0.COILS_LIST} -- an approximation, not the array's true joint "
                f"sensitivity; see em_sensitivity.py docstring) over layers "
                f"{EFFECTIVE_DEPTHS0} + half-space: {np.round(EM_LAYER_WEIGHTS, 4).tolist()}"
            )
        reporter.metric("EM_SENS — layer weights (McNeill)", np.round(EM_LAYER_WEIGHTS, 4).tolist())

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
            # Initial EC range (ECmin/ECmax) + initial WS values, one subplot
            # per horizon (real-data counterpart of ws_distribution.py).
            try:
                _ini_ec_ws = initial_ec_ws_real.plot_initial_ec_ws_real(
                    OUTPUT_DIR, cond_ranges=COND_RANGES, depths0=DEPTHS0,
                    lab_p0=LAB_P0, ws_fixed_params=WS_FIXED_PARAMS,
                    ws_bounds=EMmethod.resolve_ws_bounds(WS_BOUNDS_OVERRIDE),
                    observations=observations_list, probe_table=probe_table,
                )
                reporter.log(f"[STEP 1d] Initial EC range / WS per horizon -> "
                             f"{_ini_ec_ws['ec']}, {_ini_ec_ws['ws']}")
            except Exception as _ini_ws_exc:
                reporter.log(f"[STEP 1d][WARN] initial EC/WS plots failed (pipeline continues): "
                             f"{type(_ini_ws_exc).__name__}: {_ini_ws_exc}")
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
            "param_grid γ values (temporal smoothing, TL only)": str(PARAM_GRID.get("gamma", [GAMMA])),
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
        pre_inversion.plot_step1_ensemble_single_file(
            plot_models, DEPTHS0, OUTPUT_DIR,
            depths_are_fixed=FIXED_DEPTHS if isinstance(FIXED_DEPTHS, bool) else all(FIXED_DEPTHS),
            depths_ensemble=plot_depths,
        )
        # ADDED: per-spatial-point companion view (median +/- min/max per
        # point, instead of averaged across all points like the plot
        # above) -- see plot_step1_ensemble_by_point docstring for why
        # this matters: the plot above can visually manufacture a shared
        # "conductive layer" that isn't actually shared across points.
        pre_inversion.plot_step1_ensemble_by_point(
            plot_models, DEPTHS0, OUTPUT_DIR,
        )

        # -----------------------------------------------------------------------
        # UTM GRID LOADING  (needed for 2-D probe placement and XY plots)
        # -----------------------------------------------------------------------
        utm_x = utm_y = None
        if mode == "2d" and data_source == "synthetic":
            # Synthetic surveys use a LOCAL grid near (0, 0) for
            # x_transect/y_transect, so the absolute-UTM frame needed for
            # 2-D probe placement / XY plots has to come from a separate
            # grid written during synthetic data generation. Real data has
            # no such file -- its x_transect/y_transect ARE already
            # absolute UTM, and get assigned directly to utm_x/utm_y below
            # once they're computed (see "SAMPLE POSITION EXTRACTION").
            utm_x_path = _resolve_npy_path("synthetic_utm_x.npy", OUTPUT_MODEL_DIR, OUTPUT_DIR)
            utm_y_path = _resolve_npy_path("synthetic_utm_y.npy", OUTPUT_MODEL_DIR, OUTPUT_DIR)
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
            y_transect = ref_y
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
            y_transect = survey_df["y"].to_numpy(dtype=float)

        # FIX (real-data 2-D mode): unlike synthetic surveys, real EM survey
        # positions (x_transect/y_transect above, from either the
        # reference-grid or single-file branch) are already absolute UTM
        # coordinates -- there is no separate synthetic_utm_x/y.npy grid to
        # load for real data (see the UTM GRID LOADING block further up).
        # Reuse them directly as utm_x/utm_y so every downstream 2-D
        # consumer that gates on "utm_x is not None" -- probe UTM
        # placement, the Step 2 LOCALIZED kernel bandwidth's 2-D-vs-1-D
        # distance choice, and Step 6's XY plan-view plots -- actually
        # gets real 2-D behaviour instead of silently falling back to
        # 1-D/None just because --mode 2d was combined with real data.
        if data_source == "real" and mode == "2d":
            utm_x = x_transect
            utm_y = y_transect
            print(f"[PIPELINE] Real-data 2-D mode: using the survey's own "
                  f"{len(x_transect)}-point coordinates as the UTM XY grid "
                  f"(X {x_transect.min():.1f}\u2013{x_transect.max():.1f}, "
                  f"Y {y_transect.min():.1f}\u2013{y_transect.max():.1f}).")

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
        #   - probe_xyz_file given: explicit (x, y, z) per probe, no
        #     cartesian expansion, overrides num_swc_positions/swc_depths
        #     entirely (works in both 1-D and 2-D mode).
        #   - 1-D mode (auto): probes along the X transect only (y=0 for all)
        #   - 2-D mode (auto): probes spread across the UTM XY grid
        # ------------------------------------------------------------------
        # NOTE on PROBE_X_COORDS_LOCAL vs PROBE_X_COORDS_UTM:
        #   Two different downstream consumers need probe positions in two
        #   different, INCOMPATIBLE coordinate frames:
        #     - EMmethod.extract_sample_positions_from_survey() matches
        #       probes against survey_df, which is always in the synthetic
        #       survey's LOCAL grid (x_transect/y_transect, near (0, 0)).
        #     - plot_from_pipeline()/plot_truth_from_pipeline() (XY plan-view
        #       figures) pair probe markers against utm_x/utm_y, which are
        #       absolute UTM.
        #   Previously both consumers shared one PROBE_X_COORDS/PROBE_Y_COORDS
        #   variable, which the --probe-xyz branch below re-localized in
        #   place (correctly fixing the matching use, but silently
        #   corrupting the plotting use with local-frame values pretending
        #   to be UTM -- the t=0-panel bug: 50 UTM grid points collapsing to
        #   what looked like one point plus a probe marker mislocated at
        #   (0, 0)). Both frames are now tracked explicitly and never mixed.
        if probe_xyz_file is not None:
            PROBE_X_COORDS, PROBE_Y_COORDS, PROBE_DEPTHS, PROBE_LABELS = (
                load_probe_xyz_file(probe_xyz_file)
            )
            print(f"[PIPELINE] Loaded {len(PROBE_X_COORDS)} explicit probe "
                  f"locations from --probe-xyz {probe_xyz_file}")
            # The file's own values are the true absolute-UTM positions --
            # correct to use as-is for plotting ONLY when the survey itself
            # is also in real absolute UTM (data_source == "real"). See the
            # synthetic branch below for why this default gets overridden
            # there.
            PROBE_X_COORDS_UTM = list(PROBE_X_COORDS)
            PROBE_Y_COORDS_UTM = list(PROBE_Y_COORDS)
            if data_source == "synthetic":
                # FIX: --probe-xyz files are written for REAL field
                # deployments, so their x/y are real-world absolute
                # coordinates (e.g. actual UTM eastings/northings). The
                # synthetic EM survey's own x/y (x_transect/y_transect) are
                # always a LOCAL grid starting near (0, 0), never UTM. Using
                # the file's absolute coordinates as-is here put probes and
                # the synthetic survey in two disjoint coordinate frames --
                # the exact CRS-mismatch signature previously flagged at
                # Step 1a (span ratio / centroid offset check) and the
                # cause of probe<->EM matching collapsing onto a single
                # nearest point. Fix: re-localize the probe grid so it sits
                # in the SAME local frame as the synthetic survey, by
                # translating it so its own minimum x/y lands on the
                # survey's minimum x/y. This preserves the probe file's
                # internal relative layout (spacing/shape) exactly -- only
                # the absolute offset is removed. IMPORTANT: this
                # re-localized copy is for EM-matching ONLY; it is written
                # to PROBE_X_COORDS_LOCAL, never back onto PROBE_X_COORDS_UTM.
                _px = np.asarray(PROBE_X_COORDS, dtype=float)
                _py = np.asarray(PROBE_Y_COORDS, dtype=float)
                if _px.size and len(x_transect):
                    _dx = float(x_transect.min()) - float(_px.min())
                    _dy = float(y_transect.min()) - float(_py.min())
                    PROBE_X_COORDS = (_px + _dx).tolist()
                    PROBE_Y_COORDS = (_py + _dy).tolist()
                    print(f"[PIPELINE][FIX] Synthetic run: --probe-xyz coordinates "
                          f"were absolute (real-world) and did not share the "
                          f"synthetic survey's local coordinate frame. "
                          f"Re-localized by offset (dx={_dx:.2f}, dy={_dy:.2f}) so "
                          f"probes and survey now occupy the same local grid "
                          f"for EM-matching.")
                # FIX (part 2): the file's raw absolute coordinates captured
                # into PROBE_X_COORDS_UTM/PROBE_Y_COORDS_UTM above are the
                # probes' TRUE real-world location -- which, in synthetic
                # mode, has nothing to do with utm_x/utm_y (a synthetic grid
                # anchored at an arbitrary point, spanning a few metres).
                # Plotting the real-world probe coordinates against that
                # synthetic grid is exactly the t=0-panel bug: matplotlib
                # autoscales to fit both, so the 50 real survey points
                # collapse to a single dot while the probe marker floats far
                # outside the grid. Fix: express the SAME re-localized local
                # probe coordinates (already computed above) in utm_x/utm_y's
                # own frame, by applying the survey's local->UTM offset --
                # mirrors exactly what the auto-placement 2-D branch below
                # does. This preserves the probe file's real relative
                # layout/spacing, just placed inside the synthetic grid's
                # frame instead of at its (irrelevant here) true location.
                if utm_x is not None and utm_y is not None and _px.size and len(x_transect):
                    _offx = float(utm_x.min()) - float(x_transect.min())
                    _offy = float(utm_y.min()) - float(y_transect.min())
                    PROBE_X_COORDS_UTM = [x + _offx for x in PROBE_X_COORDS]
                    PROBE_Y_COORDS_UTM = [y + _offy for y in PROBE_Y_COORDS]
                    print(f"[PIPELINE][FIX] Synthetic run: re-expressed probe "
                          f"coordinates in the synthetic survey's own UTM frame "
                          f"for plotting (offset ex={_offx:.2f}, ey={_offy:.2f}) "
                          f"-- the probe file's real-world absolute coordinates "
                          f"are not meaningful against a synthetic grid.")
                else:
                    PROBE_X_COORDS_UTM = None
                    PROBE_Y_COORDS_UTM = None
            PROBE_X_COORDS_LOCAL = list(PROBE_X_COORDS)
            PROBE_Y_COORDS_LOCAL = list(PROBE_Y_COORDS)
        elif mode == "2d" and len(x_transect) and np.ptp(y_transect) > 0:
            # Sample NUM_POSITIONS points spread across the survey's own
            # LOCAL 2-D grid (x_transect/y_transect) -- NOT the absolute-UTM
            # synthetic_utm_x/y grid (utm_x/utm_y). The UTM grid is offset
            # by the real-world survey origin and lives in a different
            # frame than survey_df's local coordinates; mixing the two here
            # previously caused the same CRS-mismatch bug as the
            # --probe-xyz case above.
            step = max(1, len(x_transect) // NUM_POSITIONS)
            sample_indices = list(range(0, len(x_transect), step))[:NUM_POSITIONS]
            UNIQUE_XY_COORDS = [(float(x_transect[i]), float(y_transect[i])) for i in sample_indices]
            # Expand (x, y) × depth to get one probe entry per (position, layer)
            PROBE_X_COORDS = [xy[0] for xy in UNIQUE_XY_COORDS for _ in TDR_DEPTH]
            PROBE_Y_COORDS = [xy[1] for xy in UNIQUE_XY_COORDS for _ in TDR_DEPTH]
            PROBE_DEPTHS   = [d     for _  in UNIQUE_XY_COORDS for d in TDR_DEPTH]
            PROBE_LABELS   = [f"P{i}" for i in range(len(PROBE_X_COORDS))]
            PROBE_X_COORDS_LOCAL = list(PROBE_X_COORDS)
            PROBE_Y_COORDS_LOCAL = list(PROBE_Y_COORDS)
            if utm_x is not None and utm_y is not None and len(x_transect):
                # Re-express the same points in absolute UTM by applying the
                # local->UTM offset of the survey grid itself (mirror image
                # of the re-localization above), so plotting always has a
                # matching UTM pair -- never the raw local values.
                _offx = float(utm_x.min()) - float(x_transect.min())
                _offy = float(utm_y.min()) - float(y_transect.min())
                PROBE_X_COORDS_UTM = [x + _offx for x in PROBE_X_COORDS_LOCAL]
                PROBE_Y_COORDS_UTM = [y + _offy for y in PROBE_Y_COORDS_LOCAL]
            else:
                PROBE_X_COORDS_UTM = None
                PROBE_Y_COORDS_UTM = None
        else:
            # 1-D transect: distribute along X, fix Y=0
            unique_x = list(np.linspace(x_transect.min(), x_transect.max(), NUM_POSITIONS))
            UNIQUE_XY_COORDS = [(x, 0.0) for x in unique_x]
            # Expand (x, y) × depth to get one probe entry per (position, layer)
            PROBE_X_COORDS = [xy[0] for xy in UNIQUE_XY_COORDS for _ in TDR_DEPTH]
            PROBE_Y_COORDS = [xy[1] for xy in UNIQUE_XY_COORDS for _ in TDR_DEPTH]
            PROBE_DEPTHS   = [d     for _  in UNIQUE_XY_COORDS for d in TDR_DEPTH]
            PROBE_LABELS   = [f"P{i}" for i in range(len(PROBE_X_COORDS))]
            # 1-D mode has no meaningful Y/UTM position (y is always 0) --
            # downstream plotting code already suppresses XY markers in
            # this case via the has_real_y check, so there is no UTM frame
            # to populate here.
            PROBE_X_COORDS_LOCAL = list(PROBE_X_COORDS)
            PROBE_Y_COORDS_LOCAL = list(PROBE_Y_COORDS)
            PROBE_X_COORDS_UTM = None
            PROBE_Y_COORDS_UTM = None

        if data_source == "real" and probe_xyz_file is None:
            # Real-data mode (no explicit --probe-xyz override): the
            # placement above (PROBE_X_COORDS_UTM/PROBE_LABELS) is a
            # synthetic-style subsample of the survey grid ("P0", "P1", ...
            # at NUM_POSITIONS points along x_transect/y_transect) -- it
            # was never a real TDR station location. Real station
            # positions/IDs (probe_table['utm_x'/'utm_y'/'station_name'],
            # built at import time in data_real.py) are the
            # correct thing to plot on water_content_xy_{scenario}.png's
            # probe markers, so swap them in here. This does NOT affect
            # PROBE_X_COORDS_LOCAL/PROBE_Y_COORDS_LOCAL above (used only
            # for the placeholder `position_results` spatial match below,
            # which the real branch further down replaces with the
            # correctly-matched `station_position_results` anyway).
            PROBE_X_COORDS_UTM = probe_table['utm_x'].tolist()
            PROBE_Y_COORDS_UTM = probe_table['utm_y'].tolist()
            PROBE_LABELS = probe_table['station_name'].tolist()

        position_results = EMmethod.extract_sample_positions_from_survey(
            survey_df=survey_df, models=models_clean,
            probe_x_coords=PROBE_X_COORDS_LOCAL,
            probe_y_coords=PROBE_Y_COORDS_LOCAL,   # non-zero in 2D mode → XY distance auto-enabled
            probe_depths=PROBE_DEPTHS,        # <-- Z axis (unchanged)
            # Real data without --probe-xyz: PROBE_LABELS was just replaced by the
            # real station names (e.g. 20) while PROBE_X_COORDS_LOCAL is still the
            # synthetic-style placeholder (NUM_POSITIONS x layers, e.g. 8), so the
            # lengths differ and extract_sample_positions_from_survey raised
            # ValueError. The placeholder results are replaced below by
            # station_position_results (which carry the real IDs), so label them
            # only when the lengths agree.
            probe_ids=(PROBE_LABELS if len(PROBE_LABELS) == len(PROBE_X_COORDS_LOCAL) else None),  # sensor IDs -> plot_probe_matching / plot_calibration_density_* markers
            # Horizon routing uses EFFECTIVE_DEPTHS0 (== DEPTHS0 when fixed;
            # the ensemble-median FITTED interface depths otherwise -- see
            # "EFFECTIVE DEPTHS FOR TDR PROBE HORIZON ROUTING" above).
            depths0=EFFECTIVE_DEPTHS0, window_m=WINDOW_M,
        )
        sample_positions    = EMmethod.sample_positions_from_results(position_results)
        probe_layer_indices = EMmethod.sample_layer_indices_from_results(position_results)
        # Per-base-probe sensor ID (real station name / synthetic "P0",
        # "P1", ... -- see PROBE_LABELS above), one entry per
        # position_results/sample_positions element at this point (before
        # any timelapse expansion below). Threaded to parity_validation_
        # by_sensor.png / petro_calibration_localized_map_*.png so those
        # plots group/label by real sensor identity instead of by the
        # anonymous EM spatial index.
        sample_sensor_ids = [r.probe_id for r in position_results]

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
            if probe_xyz_file is not None:
                # Explicit (x, y, z) probes: no horizontal-position × depth
                # cartesian expansion happened (UNIQUE_XY_COORDS is never
                # built on this path), so describe the probes as-loaded.
                _probe_note = (
                    f"{len(position_results)} probe(s) loaded explicitly from "
                    f"--probe-xyz '{probe_xyz_file}' (literal x/y/z per row, "
                    f"no cartesian expansion)."
                )
            else:
                _probe_note = (
                    f"{len(UNIQUE_XY_COORDS)} horizontal position(s) × "
                    f"{len(TDR_DEPTH)} depth(s) = {len(position_results)} probes. "
                    f"Depths resolved from "
                    f"{'scenario swc_depths' if _swc_depths_cfg is not None else 'layer centres (swc_depths was None/auto)'}"
                    f": {list(TDR_DEPTH)}."
                )
            reporter.table(
                "SWC / TDR Probe Network — resolved x/y/z positions",
                ["Probe #", "x [m]", "y [m]", "z / depth [m]",
                 "Horizon (layer idx)", "Matched EM point idx"],
                _probe_rows,
                note=_probe_note,
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

            try:
                pcd.plot_calibration_density_hard(
                    survey_df=survey_df,
                    position_results=position_results,
                    output_dir=OUTPUT_DIR,
                    title="Calibration density — hard window (SWC/TDR probes)",
                )
            except Exception as _calib_hard_exc:
                reporter.log(
                    f"[PIPELINE][WARN] Hard-window calibration-density plot failed "
                    f"(pipeline continues): {type(_calib_hard_exc).__name__}: {_calib_hard_exc}"
                )

            # Branch-neutral alias so the kernel-localized diagnostics further
            # below (Step 2 LOCALIZED, gated on WS_LOCALIZED_BANDWIDTH) don't
            # need to know whether this run used synthetic `position_results`
            # or real-data `station_position_results` — mirrors `survey_df`,
            # which both branches already share under one name.
            CALIB_POSITION_RESULTS = position_results

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
            # data_real.prepare_real_dataset (which gathers ALL
            # historical readings per probe rather than one snapshot).
            n_probes_base       = len(sample_positions)
            sample_positions    = [pos for pos in sample_positions for _ in range(n_time)]
            probe_layer_indices = [li  for li  in probe_layer_indices for _ in range(n_time)]
            sample_times        = [t for _ in range(n_probes_base) for t in range(n_time)]
            sample_sensor_ids   = [pid for pid in sample_sensor_ids for _ in range(n_time)]
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
            # (0..n_stations-1) assigned by data_real.py — as if
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
                # Real TDR station IDs (the logger 'Unit' column, captured
                # at import time in data_real.py) -> labels the
                # probe markers on probe_em_matching_real_stations.png,
                # calibration_density_hard_real_stations.png, and
                # calibration_density_kernel.png (via CALIB_POSITION_RESULTS
                # below) with real station names instead of anonymous dots.
                probe_ids=probe_table['station_name'].tolist(),
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

            try:
                pcd.plot_calibration_density_hard(
                    survey_df=survey_df,
                    position_results=station_position_results,
                    output_dir=OUTPUT_DIR,
                    title="Calibration density — hard window (real TDR stations)",
                    filename="calibration_density_hard_real_stations.png",
                )
            except Exception as _calib_hard_exc:
                reporter.log(
                    f"[PIPELINE][WARN] Hard-window calibration-density plot failed "
                    f"(pipeline continues): {type(_calib_hard_exc).__name__}: {_calib_hard_exc}"
                )

            # Branch-neutral alias — see synthetic-branch comment above.
            CALIB_POSITION_RESULTS = station_position_results

            # For real data mode, extract sample metadata from observations,
            # routing each observation's station-ordinal probe_index through
            # the station -> EM-spatial-index map computed above.
            sample_positions = np.array([
                station_to_spatial_idx[obs['probe_index']] for obs in observations_list
            ])
            sample_times = np.array([obs['time_index'] for obs in observations_list])
            # Real TDR station name per observation, routed through the
            # same 'probe_index' (station-ordinal index into probe_table)
            # each observation already carries -- overwrites the
            # synthetic-branch sample_sensor_ids above with real station
            # identity for parity_validation_by_sensor.png /
            # petro_calibration_localized_map_*.png.
            _station_names = probe_table['station_name'].to_numpy()
            sample_sensor_ids = [
                _station_names[obs['probe_index']] for obs in observations_list
            ]

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
        # TRAIN / VALIDATION SPLIT (Loiseau et al., 2026 Secs. 2.3/2.6)
        # -----------------------------------------------------------------------
        # Split once, right here, over the flattened (position, time)
        # observation pairs -- before anything downstream (Step 2 fitting,
        # Step 3 ranking, Step 4 recalibration, Step 5 K-selection) ever
        # touches theta_obs. Training data drives fitting/selection;
        # validation data is held out untouched until the one-shot Step 5b
        # check after K is picked. "off" (default) preserves the legacy
        # behaviour: everything below sees the full observation set, same
        # as before this split existed.
        _probe_id_for_split = None
        if data_source == "real":
            # obs['probe_index'] is the real TDR station's ordinal index,
            # built in the exact same loop order as theta_obs_list above --
            # this is the only case where a genuine per-observation probe
            # identifier exists today, so it's the only case 'by_probe' can
            # actually run fully independent (see tvs.split_train_val's
            # fallback-to-alternate_sorted otherwise).
            _probe_id_for_split = np.array([obs['probe_index'] for obs in observations_list])

        sample_positions_val = sample_times_val = theta_obs_val = probe_layer_val = None
        split = None
        if TRAIN_VAL_SPLIT != "off":
            split = tvs.split_train_val(
                theta_obs=np.asarray(theta_obs),
                sample_positions=np.asarray(sample_positions),
                sample_times=np.asarray(sample_times),
                probe_layer_indices=(np.asarray(probe_layer_indices)
                                      if probe_layer_indices is not None else None),
                probe_id=_probe_id_for_split,
                method=TRAIN_VAL_SPLIT,
            )
            reporter.log(
                f"[SPLIT] Train/val split method='{split.method}' "
                f"(independent={split.independent}): "
                f"{len(split.train)} train / {len(split.val)} val observations "
                f"(of {len(theta_obs)} total)."
            )
            if not split.independent:
                reporter.log(
                    "[SPLIT][WARN] This split is NOT fully independent -- "
                    "train and validation subsets may share the same probe "
                    "and/or time step (Loiseau et al.'s Larzac caveat: "
                    "network too sparse for a fully independent split). "
                    "Treat Step 5b's validation RMSE as indicative, not a "
                    "strict generalisation test, until switching to "
                    "'by_probe' or 'by_time_block' once TDR coverage is "
                    "denser."
                )
            reporter.metric("Train/val split method", split.method)
            reporter.metric("Train observations", len(split.train))
            reporter.metric("Validation observations", len(split.val))
            reporter.metric("Split independent", str(split.independent))

            # Hold out the validation subset untouched, for the one-shot
            # Step 5b check after K is selected.
            sample_positions_val = tvs.subset(np.asarray(sample_positions), split.val)
            sample_times_val     = tvs.subset(np.asarray(sample_times), split.val)
            theta_obs_val        = tvs.subset(np.asarray(theta_obs), split.val)
            probe_layer_val      = (tvs.subset(np.asarray(probe_layer_indices), split.val)
                                     if probe_layer_indices is not None else None)
            sample_sensor_ids_val = tvs.subset(
                np.asarray(sample_sensor_ids, dtype=object), split.val
            )

            # Route ONLY the training subset into every downstream step
            # that currently drives fitting/selection (Step 2/3/4/5). This
            # reassigns the exact same variable names step2/step3/step4/
            # step4_5 already consume below, so nothing past this point
            # needs to change to become train-only.
            sample_positions    = tvs.subset(np.asarray(sample_positions), split.train)
            sample_times        = tvs.subset(np.asarray(sample_times), split.train)
            theta_obs           = tvs.subset(np.asarray(theta_obs), split.train)
            probe_layer_indices = (tvs.subset(np.asarray(probe_layer_indices), split.train)
                                    if probe_layer_indices is not None else None)
            sample_sensor_ids   = tvs.subset(
                np.asarray(sample_sensor_ids, dtype=object), split.train
            )

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
        # -- TDR_NOISE_STD is whatever cfg.tdr_noise_std resolved to at
        # Step 0 (synthetic and real-data runs alike; see the FIX note
        # there). None/0.0/negative disables weighting entirely inside
        # calibrate_petrophysical_relationship -- 1/0^2 weighting would be
        # nonsensical/unstable, so that case is passed through as None
        # rather than as a literal 0.0.
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
            sensor_ids=sample_sensor_ids,
            ws_bounds=WS_BOUNDS_OVERRIDE,
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
            ws_bounds=WS_BOUNDS_OVERRIDE,
        )

        # ------------------------------------------------------------------
        # Step 2 (diagnostic sibling) — kernel-localized Waxman-Smits
        # calibration (spatial_calibration_alternatives.md §2). Gated on
        # the scenario's ws_localized_bandwidth field; does NOT replace
        # fits_step2 and does NOT change Steps 3-5. In 2-D mode (utm_x/
        # utm_y both loaded) localization uses true 2-D Euclidean (x, y)
        # distance instead of 1-D along-transect x, so probes that are
        # close in y but far in x (or vice versa) aren't ignored.
        # ------------------------------------------------------------------
        WS_LOCALIZED_BANDWIDTH = getattr(cfg, "ws_localized_bandwidth", None) if cfg is not None else None
        WS_LOCALIZED_BANDWIDTH_SWEEP = getattr(cfg, "ws_localized_bandwidth_sweep", None) if cfg is not None else None
        # --ws-localized-bandwidth / --ws-localized-bandwidth-sweep CLI
        # overrides (mirror the --beta / --gamma pattern: CLI value wins
        # over whatever the scenario config set, including turning the
        # diagnostic ON for a scenario that doesn't set either field, or
        # OFF is not directly expressible here -- pass neither flag and
        # the scenario value (if any) is used as before).
        if ws_localized_bandwidth is not None:
            WS_LOCALIZED_BANDWIDTH = ws_localized_bandwidth
        if ws_localized_bandwidth_sweep is not None:
            WS_LOCALIZED_BANDWIDTH_SWEEP = list(ws_localized_bandwidth_sweep)
        if STAGE == "map":
            # The leave-one-probe-out bandwidth CV only reads probe-matched points, so it
            # belongs to the calibrate stage; the map run reuses the bandwidth it chose
            # (an explicit --ws-localized-bandwidth still wins).
            if WS_LOCALIZED_BANDWIDTH_SWEEP is not None:
                reporter.log("[STAGE] map: ws_localized_bandwidth_sweep skipped -- the bandwidth "
                             "is taken from the calibration run.")
            WS_LOCALIZED_BANDWIDTH_SWEEP = None
            if ws_localized_bandwidth is None and CALIBRATION.get("localized_bandwidth") is not None:
                WS_LOCALIZED_BANDWIDTH = CALIBRATION["localized_bandwidth"]
                reporter.log(f"[STAGE] map: ws_localized_bandwidth = {WS_LOCALIZED_BANDWIDTH:g} "
                             f"(from the calibration"
                             f"{', chosen there by the LOO-CV sweep' if CALIBRATION.get('bandwidth_from_sweep') else ''}).")
        _localized_y = y_transect if (mode == "2d" and utm_y is not None) else None
        _bandwidth_from_sweep = False

        # ------------------------------------------------------------------
        # Step 2 LOCALIZED — bandwidth (c) selection via leave-one-probe-out
        # CV sweep (EMmethod.sweep_localization_bandwidth /
        # step2.run_bandwidth_sweep). Opt-in via the scenario's
        # ws_localized_bandwidth_sweep field (a sequence of candidate c
        # values, e.g. np.linspace(1.0, 20.0, 10)). This is a one-off
        # tuning decision (step2.run_bandwidth_sweep's own docstring: run it
        # against a single reference model, not the full ensemble), so it is
        # evaluated once against the rank-1 (best-RMSE) Step 2 ensemble
        # member (fits_step2[0].model_index) rather than per ensemble
        # member. The recommended bandwidth then becomes
        # WS_LOCALIZED_BANDWIDTH for the diagnostic sibling call below,
        # overriding any fixed ws_localized_bandwidth in the scenario.
        # ------------------------------------------------------------------
        if WS_LOCALIZED_BANDWIDTH_SWEEP is not None:
            reporter.step(
                "Step 2 LOCALIZED SWEEP",
                "Bandwidth (c) selection via leave-one-probe-out CV",
            )
            _sweep_ref_idx = fits_step2[0].model_index
            reporter.log(
                f"[STEP 2 LOCALIZED SWEEP] ws_localized_bandwidth_sweep="
                f"{list(WS_LOCALIZED_BANDWIDTH_SWEEP)} set -- evaluating candidate "
                f"bandwidth(s) via leave-one-probe-out CV against the rank-1 "
                f"ensemble member (model_index={_sweep_ref_idx}, "
                f"RMSE={fits_step2[0].rmse:.5f} m³/m³) "
                f"({'2-D Euclidean (x, y)' if _localized_y is not None else '1-D along-transect (x)'} distance)."
            )
            try:
                _sigma_b_reference = models_clean[_sweep_ref_idx]
                sweep_results, _recommended_bandwidth = step2.run_bandwidth_sweep(
                    sigma_b_reference=_sigma_b_reference,
                    sample_positions=sample_positions,
                    sample_times=sample_times,
                    theta_obs=theta_obs,
                    point_x_coords=x_transect,
                    point_y_coords=_localized_y,
                    bandwidths=WS_LOCALIZED_BANDWIDTH_SWEEP,
                    output_dir=OUTPUT_DIR,
                    probe_layer_indices=probe_layer_indices,
                    lab_p0=LAB_P0,
                    fixed_params=WS_FIXED_PARAMS,
                    theta_obs_std=_theta_obs_std_step2,
                    ws_bounds=WS_BOUNDS_OVERRIDE,
                )
                _sweep_rows = [
                    [
                        f"{r.bandwidth:g}",
                        f"{r.loo_rmse:.5f}" if not np.isnan(r.loo_rmse) else "N/A",
                        r.n_folds,
                        r.n_fallback,
                    ]
                    for r in sweep_results
                ]
                reporter.table(
                    "Step 2 LOCALIZED — bandwidth sweep (leave-one-probe-out CV)",
                    ["Bandwidth c", "Pooled LOO-CV RMSE [m³/m³]", "Valid folds", "Folds fell back to prior"],
                    _sweep_rows,
                    note=(
                        f"Recommended bandwidth: c={_recommended_bandwidth:g} (minimum pooled "
                        f"LOO-CV RMSE). Evaluated once against the rank-1 ensemble member "
                        f"(model_index={_sweep_ref_idx}) -- a one-off tuning decision, not "
                        f"repeated per ensemble member."
                    ),
                )
                reporter.log(
                    f"[STEP 2 LOCALIZED SWEEP] Recommended bandwidth: c={_recommended_bandwidth:g} "
                    f"-> {OUTPUT_DIR}/bandwidth_sweep.png"
                )
                reporter.metric("Step 2 LOCALIZED — bandwidth selection method", "LOO-CV sweep")
                reporter.metric("Step 2 LOCALIZED — recommended bandwidth (c)", f"{_recommended_bandwidth:g}")
                reporter.metric("Step 2 LOCALIZED — sweep candidates evaluated", len(sweep_results))
                WS_LOCALIZED_BANDWIDTH = _recommended_bandwidth
                _bandwidth_from_sweep = True
            except Exception as e:
                # Sweep is a tuning aid only -- never let it fail a run that
                # would otherwise succeed; fall back to the fixed
                # ws_localized_bandwidth (if any) below, unchanged.
                reporter.log(
                    f"[STEP 2 LOCALIZED SWEEP][WARN] Bandwidth sweep failed: {e}. "
                    f"Falling back to fixed ws_localized_bandwidth="
                    f"{WS_LOCALIZED_BANDWIDTH!r} (unswept)."
                )
            reporter.end_step("Step 2 LOCALIZED SWEEP")

        if WS_LOCALIZED_BANDWIDTH is not None:
            reporter.log(
                f"[STEP 2 LOCALIZED] ws_localized_bandwidth="
                f"{WS_LOCALIZED_BANDWIDTH:g} "
                f"({'recommended by LOO-CV sweep above' if _bandwidth_from_sweep else 'fixed scenario value'}) "
                f"-- running kernel-localized calibration as a diagnostic sibling to Step 2 "
                f"({'2-D Euclidean (x, y)' if _localized_y is not None else '1-D along-transect (x)'} distance)."
            )
            try:
                fits_step2_localized = step2.run_petro_localized(
                    models=models_clean,
                    sample_positions=sample_positions,
                    sample_times=sample_times,
                    theta_obs=theta_obs,
                    point_x_coords=x_transect,
                    point_y_coords=_localized_y,
                    bandwidth=WS_LOCALIZED_BANDWIDTH,
                    depths0=DEPTHS0,
                    output_dir=OUTPUT_DIR,
                    probe_layer_indices=probe_layer_indices,
                    fixed_params=WS_FIXED_PARAMS,
                    lab_p0=LAB_P0,
                    theta_obs_std=_theta_obs_std_step2,
                    global_fit=fits_step2[0],
                    sensor_ids=sample_sensor_ids,
                    ws_bounds=WS_BOUNDS_OVERRIDE,
                )
                reporter.log(
                    f"[STEP 2 LOCALIZED] Diagnostic complete -> "
                    f"{OUTPUT_DIR}/petro_calibration_localized.png"
                )

                # Spatial diagnostics for this same localized fit: how many
                # (effective, kernel-weighted) TDR probes back each EM point,
                # and what the fit actually produced spatially, per horizon.
                pcd.plot_calibration_density_kernel(
                    survey_df=survey_df,
                    position_results=CALIB_POSITION_RESULTS,
                    bandwidth=WS_LOCALIZED_BANDWIDTH,
                    output_dir=OUTPUT_DIR,
                )
                _calib_probe_x = np.array([r.probe_x for r in CALIB_POSITION_RESULTS])
                # run_petro_localized returns a list of LocalizedPetroFit, one
                # per ensemble member (symmetric with fits_step2) -- [0] is the
                # rank-1 member, matching global_fit=fits_step2[0] passed in
                # above. NOT a single LocalizedPetroFit with its own
                # .horizon_fits directly.
                #
                # One combined XZ-style figure (horizon = depth axis) rather
                # than one flat XY scatter per horizon -- directly comparable
                # to water_content_xz_*.png. depths0=EFFECTIVE_DEPTHS0, not
                # the static DEPTHS0, for the same reason EM_LAYER_WEIGHTS
                # uses it above: it's what this fit's probe routing actually
                # used.
                pcd.plot_ws_params_xz(
                    horizon_fits=fits_step2_localized[0].horizon_fits,
                    depths0=EFFECTIVE_DEPTHS0,
                    output_dir=OUTPUT_DIR,
                    probe_x=_calib_probe_x,
                    # .probe_depth, not a re-derivation via probe_layer_indices:
                    # sample_positions/probe_layer_indices can be reordered or
                    # subset relative to CALIB_POSITION_RESULTS (train/val
                    # split, time-lapse expansion), so indexing one against
                    # the other isn't safe. SamplePositionResult carries its
                    # own installation depth directly.
                    probe_depths=np.array([r.probe_depth for r in CALIB_POSITION_RESULTS]),
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
                        output_dir=OUTPUT_DIR,
                        probe_x=_calib_probe_x,
                        probe_y=np.array([r.probe_y for r in CALIB_POSITION_RESULTS]),
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
            ws_bounds=WS_BOUNDS_OVERRIDE,
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

        # -----------------------------------------------------------------------
        # STEP 5c — EM FOOTPRINT DISCREPANCY (McNeill depth weighting)
        #   Diagnostic only, opt-in via --em-footprint-diagnostic. Does not
        #   change Step 2/4 calibration or Step 3/5 K-selection -- reuses
        #   selected_output.theta_calc (already computed, whole grid, at K)
        #   rather than re-deriving anything, and just adds the
        #   McNeill-weighted comparison alongside the existing hard-routed
        #   (nearest-layer) one.
        # -----------------------------------------------------------------------
        # ------------------------------------------------------------------
        # --stage hand-over (see stage_split.py)
        # ------------------------------------------------------------------
        if STAGE in ("calibrate", "map"):
            _ws_now = stage_split.ws_rows_from_fit(fits_step4[selected_output.K - 1].horizon_fits)
        if STAGE == "calibrate":
            _calib_record = stage_split.build_calibration_record(
                scenario=_scenario_name, run_id=reporter.run_id, output_dir=OUTPUT_DIR,
                ranking=ranking, param_metadata=param_metadata,
                rmse_by_member=[f.rmse for f in fits_step2],
                selected_K=selected_output.K, selected_rmse=selected_output.rmse,
                ws_by_horizon=_ws_now,
                localized_bandwidth=WS_LOCALIZED_BANDWIDTH,
                bandwidth_from_sweep=_bandwidth_from_sweep,
                depths0=DEPTHS0, cond_ranges=COND_RANGES,
                em_files=(list(EM_FNAMES) if timelapse else [EM_FNAMES]),
                em_subsample=EM_SUBSAMPLE, quick=quick, n_probe_obs=len(theta_obs),
            )
            _calib_path = stage_split.write_calibration(OUTPUT_DIR, _calib_record)
            reporter.log(
                f"[STAGE] calibrate: wrote '{_calib_path}' -- K={selected_output.K} of "
                f"{len(param_metadata)} member(s), RMSE={selected_output.rmse:.5f} m3/m3, "
                f"localized bandwidth={WS_LOCALIZED_BANDWIDTH!r}. Next: "
                f"--stage map --calibration '{OUTPUT_DIR}'")
            reporter.args_table("Stage calibrate — members handed to the map stage", {
                f"rank {m['rank']}": (f"alpha={m['alpha']:.4g}  {m['forwardModel']}  {m['method']}  "
                                      f"start={m['conds0']}  RMSE={m['rmse']:.5f}")
                for m in _calib_record["members"]
            })
        elif STAGE == "map":
            _drift = stage_split.drift_report(CALIBRATION, selected_output.K, _ws_now)
            reporter.table(
                "Stage map — drift of the re-derived calibration vs the calibrate run",
                ["Horizon", "Param", "Calibration", "Map run", "Rel. diff", "Drift"],
                [[r["horizon"], r["param"], f"{r['calibration']:.4g}", f"{r['map_run']:.4g}",
                  f"{r['rel_diff']*100:.1f}%", "YES" if r["drift"] else ""] for r in _drift["rows"]],
                note=(f"K: calibration {_drift['K_calibration']} vs map run {_drift['K_map']}. "
                      f"Drift threshold {_drift['rel_tol']*100:.0f}% of the calibrated value. A "
                      f"small drift is expected when the map run matches probes to a denser "
                      f"point set; a changed K or large drift means the calibration no longer "
                      f"describes this run."))
            reporter.metric("Stage map — K (calibration / map)",
                            f"{_drift['K_calibration']} / {_drift['K_map']}")
            reporter.metric("Stage map — max WS parameter drift", f"{_drift['max_rel']*100:.1f}%")
            reporter.log(
                f"[STAGE] map: K calibration={_drift['K_calibration']} vs map={_drift['K_map']}; "
                f"max WS parameter drift {_drift['max_rel']*100:.1f}%"
                + ("  [WARN] CALIBRATION DRIFT -- consider redoing --stage calibrate." if _drift["warn"] else "  (OK)"))

        if em_footprint_diagnostic:
            reporter.step("Step 5c", "EM footprint discrepancy (McNeill depth weighting)")

            # theta_calc is ALWAYS (n_time, num_measured_points, num_layers) --
            # SelectionOutput guarantees this (step4_select.py
            # normalises `if theta_calc.ndim == 2: theta_calc = theta_calc[np.newaxis, ...]`
            # right before returning), so t_idx = 0 covers the single-time
            # case directly -- no separate timelapse-conditional 2-D branch.
            _footprint_theta_calc = selected_output.theta_calc
            _eff_theta = emsens.effective_theta(_footprint_theta_calc, EM_LAYER_WEIGHTS)
            # _eff_theta shape: (n_time, num_measured_points) -- one depth-weighted
            # "what the coil actually saw" value per position per time step.

            _footprint_discrepancies = []
            for _fp_i, (_fp_pos, _fp_li, _fp_obs) in enumerate(
                zip(sample_positions, probe_layer_indices, theta_obs)
            ):
                _fp_t_idx = sample_times[_fp_i] if timelapse else 0
                _fp_hard_routed = _footprint_theta_calc[_fp_t_idx, _fp_pos, _fp_li]
                _fp_weighted = _eff_theta[_fp_t_idx, _fp_pos]
                _footprint_discrepancies.append({
                    "probe": _fp_i, "pos": int(_fp_pos), "obs": _fp_obs,
                    # 'horizon' -- the probe's own routed layer index, same
                    # as used for _fp_hard_routed above -- carried through
                    # so plot_footprint_spatial_discrepancy_xz can place
                    # this record on the depth axis without re-deriving it.
                    "horizon": int(_fp_li),
                    "hard_routed": _fp_hard_routed, "weighted": _fp_weighted,
                    "hard_routed_err": _fp_hard_routed - _fp_obs,
                    "weighted_err": _fp_weighted - _fp_obs,
                })

            _hr_rmse = np.sqrt(np.nanmean([d["hard_routed_err"] ** 2 for d in _footprint_discrepancies]))
            _w_rmse = np.sqrt(np.nanmean([d["weighted_err"] ** 2 for d in _footprint_discrepancies]))
            reporter.log(
                f"[STEP 5c] RMSE vs TDR — hard-routed (nearest layer): {_hr_rmse:.5f} m³/m³  "
                f"|  depth-weighted (McNeill {_COIL_ORIENTATION if data_source == 'real' else 'multi-coil avg'}): "
                f"{_w_rmse:.5f} m³/m³"
            )
            reporter.metric("Footprint RMSE — hard-routed layer", f"{_hr_rmse:.5f} m³/m³")
            reporter.metric("Footprint RMSE — McNeill-weighted", f"{_w_rmse:.5f} m³/m³")
            reporter.table(
                "Step 5c — EM footprint discrepancy (per probe)",
                ["Probe #", "TDR obs [m³/m³]", "Hard-routed θ", "Weighted θ",
                 "Hard-routed err", "Weighted err"],
                [
                    [d["probe"], f"{d['obs']:.4f}", f"{d['hard_routed']:.4f}", f"{d['weighted']:.4f}",
                     f"{d['hard_routed_err']:+.4f}", f"{d['weighted_err']:+.4f}"]
                    for d in _footprint_discrepancies
                ],
                note=(
                    "'Hard-routed' = theta_calc at the probe's single assigned layer "
                    "(current Step 2/4 calibration routing, unchanged). 'Weighted' = "
                    "McNeill depth-weighted blend across all layers (EM_LAYER_WEIGHTS). "
                    "Diagnostic only -- neither RMSE feeds back into calibration or "
                    "K-selection."
                ),
            )
            if _w_rmse < _hr_rmse:
                reporter.log(
                    "[STEP 5c] Depth-weighted comparison fits TDR better than the "
                    "hard-routed layer -- the current nearest-layer probe routing "
                    "may be discarding real cross-layer signal contribution. "
                    "See the explainer's Step D for what it would take to fold "
                    "this into calibration itself, and note the caveats there "
                    "before doing so."
                )
            else:
                reporter.log(
                    "[STEP 5c] Hard-routed layer already matches TDR at least as "
                    "well as the McNeill-weighted blend -- either your layers are "
                    "thick relative to the coil spacing used (weight concentrates in "
                    "one layer anyway) or CS/high-frequency LIN departure is "
                    "degrading the kernel here. Check EM_LAYER_WEIGHTS -- if one "
                    "layer already carries >90% of the weight, this result is "
                    "expected and the current routing is fine."
                )

            try:
                pef.plot_footprint_probe_comparison(
                    discrepancies=_footprint_discrepancies,
                    output_dir=OUTPUT_DIR,
                    hr_rmse=_hr_rmse, w_rmse=_w_rmse,
                )
                pef.plot_footprint_spatial_discrepancy(
                    discrepancies=_footprint_discrepancies,
                    survey_df=survey_df,
                    output_dir=OUTPUT_DIR,
                    probe_x=np.array([r.probe_x for r in CALIB_POSITION_RESULTS]),
                    probe_y=np.array([r.probe_y for r in CALIB_POSITION_RESULTS]),
                )
                # XZ (depth) counterpart to the XY map above -- same
                # per-(position, horizon) improvement metric
                # (|err_hard| - |err_weighted|), but placed on the depth
                # axis via each record's own 'horizon' instead of the
                # survey's y/utm_y coordinate. Directly comparable to
                # water_content_xz_*.png and plot_ws_params_xz's XZ
                # figures elsewhere in this pipeline. Uses
                # EFFECTIVE_DEPTHS0 (not the static DEPTHS0), same reason
                # as the Step 2 LOCALIZED XZ call above: it's what this
                # run's actual (possibly fitted) interfaces are.
                pef.plot_footprint_spatial_discrepancy_xz(
                    discrepancies=_footprint_discrepancies,
                    survey_df=survey_df,
                    depths0=EFFECTIVE_DEPTHS0,
                    output_dir=OUTPUT_DIR,
                    probe_x=np.array([r.probe_x for r in CALIB_POSITION_RESULTS]),
                    probe_depths=np.array([r.probe_depth for r in CALIB_POSITION_RESULTS]),
                )
            except Exception as _footprint_plot_exc:
                reporter.log(
                    f"[STEP 5c][WARN] Footprint diagnostic plots failed "
                    f"(pipeline continues): {type(_footprint_plot_exc).__name__}: {_footprint_plot_exc}"
                )

            reporter.end_step("Step 5c")

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
        # PERSIST STEP 4/5 RESULT (theta_calc, cv_map, K, rmse, coverage,
        # is_calibrated) TO OUTPUT_MODEL_DIR
        #   Previously this array was only ever plotted, then discarded --
        #   compare_scenarios.py / plot_truth_comparison.py need it on disk
        #   as 'theta_calc.npy' next to the synthetic_*.npy truth arrays to
        #   do calc-vs-truth comparisons. Saving it here makes every run
        #   self-sufficient for that comparison, synthetic or real, with no
        #   extra step.
        # -----------------------------------------------------------------------
        _step5_capture = {
            "K": np.array(selected_output.K),
            "rmse": np.array(selected_output.rmse),
            "cv_map": np.asarray(selected_output.cv_map),
        }
        if _tc is not None:
            _step5_capture["theta_calc"] = np.asarray(_tc)
        if getattr(selected_output, "coverage", None) is not None:
            _step5_capture["coverage"] = np.asarray(selected_output.coverage)
        if getattr(selected_output, "best_fit", None) is not None:
            _step5_capture["is_calibrated"] = np.array(
                [h.is_calibrated for h in selected_output.best_fit.horizon_fits]
            )
        # Optimal RMSE-weighted bulk-EC model (n_time, n_pts, n_layers) -- needed
        # by plot_validation_1to1.py for the per-horizon EC truth-vs-optimal 1:1.
        try:
            _step5_capture["ec_optimal"] = np.asarray(averaged_models[selected_output.K - 1])
        except Exception as _e:
            reporter.log(f"[STEP 5][WARN] could not capture ec_optimal: {_e}")
        for _key, _arr in _step5_capture.items():
            _path = os.path.join(OUTPUT_MODEL_DIR, f"{_key}.npy")
            np.save(_path, _arr)
        reporter.log(
            f"[STEP 5] Saved Step 4/5 result to '{OUTPUT_MODEL_DIR}': "
            f"{', '.join(f'{k}.npy' for k in _step5_capture)}"
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

        reporter.end_step("Step 5")

        # -----------------------------------------------------------------------
        # STEP 5b — HELD-OUT VALIDATION (Loiseau et al., 2026 Sec. 2.6)
        # -----------------------------------------------------------------------
        # Forward-only evaluation of the TRAIN-fitted Waxman-Smits parameters
        # against the VALIDATION subset, at the selected K. No re-fitting
        # happens here -- everything above (Step 2/3/4/5) only ever saw
        # `split.train`, so this is the first and only time the validation
        # observations are touched.
        if split is not None and theta_obs_val is not None and len(theta_obs_val):
            reporter.step("Step 5b", "Held-out validation (Sec. 2.6)")
            reporter.log("\n" + "=" * 80)
            reporter.log(" [PIPELINE] STEP 5b: HELD-OUT VALIDATION")
            reporter.log("=" * 80)

            # -----------------------------------------------------------
            # Validation RMSE at EVERY candidate stack depth K, not just
            # the selected one -- needed for the train-vs-val RMSE-vs-K
            # overfitting curve (train_val/rmse_vs_K.png). Same "pin every
            # WS param to its train-fitted value" trick as before, just
            # looped across all K instead of stopping at selected_output.K.
            # Cheap: fixed_params turns each L-BFGS-B call into a no-op
            # forward pass, so this is len(fits_step4) forward evaluations,
            # not len(fits_step4) real optimizations.
            # -----------------------------------------------------------
            val_rmse_by_k = []
            val_fit = None
            for _k in range(len(fits_step4)):
                _train_params_k = fits_step4[_k].params  # (num_layers, 4)
                _val_fit_k = step2.run_petro(
                    # averaged_models[_k] already has the ensemble axis
                    # collapsed (Step 3); wrap it back into a length-1
                    # ensemble since run_petro/calibrate_petrophysical_
                    # relationship call `.ndim` on `models` directly.
                    models=np.stack([averaged_models[_k]]),
                    sample_positions=sample_positions_val, sample_times=sample_times_val,
                    theta_obs=theta_obs_val, depths0=DEPTHS0, output_dir=OUTPUT_DIR,
                    generate_plots=False,   # diagnostic-only pass, skip Step 2's plot set
                    probe_layer_indices=probe_layer_val,
                    depths_are_fixed=FIXED_DEPTHS if isinstance(FIXED_DEPTHS, bool) else all(FIXED_DEPTHS),
                    fixed_params={"a": True, "m": True, "sigma_w": True, "sigma_s": True},
                    probe_depths_reflect_fitted_interfaces=PROBE_ROUTING_USES_FITTED_DEPTHS,
                    lab_p0=_train_params_k,   # pin to TRAIN-fitted values at this K, not LAB_P0
                    theta_obs_std=_theta_obs_std_step2,
                )[0]
                val_rmse_by_k.append(_val_fit_k.rmse)
                if _k == selected_output.K - 1:
                    # Keep the fit at the SELECTED K -- this is what the
                    # scalar metrics/parity plot below use, identical to
                    # the single-K evaluation this loop replaces.
                    val_fit = _val_fit_k
            val_rmse_by_k = np.array(val_rmse_by_k)

            if val_fit is None:
                # Defensive fallback -- should be unreachable since
                # selected_output.K is always in range(1, len(fits_step4)+1),
                # but avoids a confusing AttributeError below if that ever
                # changes upstream.
                reporter.log(
                    "[STEP 5b][WARN] selected_output.K was not covered by "
                    "the per-K validation loop -- re-evaluating it directly."
                )
                val_fit = step2.run_petro(
                    models=np.stack([averaged_models[selected_output.K - 1]]),
                    sample_positions=sample_positions_val, sample_times=sample_times_val,
                    theta_obs=theta_obs_val, depths0=DEPTHS0, output_dir=OUTPUT_DIR,
                    generate_plots=False,
                    probe_layer_indices=probe_layer_val,
                    depths_are_fixed=FIXED_DEPTHS if isinstance(FIXED_DEPTHS, bool) else all(FIXED_DEPTHS),
                    fixed_params={"a": True, "m": True, "sigma_w": True, "sigma_s": True},
                    probe_depths_reflect_fitted_interfaces=PROBE_ROUTING_USES_FITTED_DEPTHS,
                    lab_p0=fits_step4[selected_output.K - 1].params,
                    theta_obs_std=_theta_obs_std_step2,
                )[0]

            # Train RMSE per K was already computed above at Step 4
            # (`_rmse4_all = [f.rmse for f in fits_step4]`) -- reuse it
            # rather than recomputing, so both curves are guaranteed to
            # come from the exact same fits_step4 list, index for index.
            train_rmse_by_k = np.array(_rmse4_all)

            # `fits_step4` (Step 4's recalibration of the K-averaged models)
            # is what selected_output.K was actually picked on -- these are
            # the TRAIN-fitted parameters frozen for reporting below.
            train_fitted_params = fits_step4[selected_output.K - 1].params  # (num_layers, 4)

            train_rmse = selected_output.rmse
            val_rmse = val_fit.rmse
            reporter.log(
                f"[STEP 5b] Train RMSE (K={selected_output.K}, n={len(split.train)}): "
                f"{train_rmse:.5f} m³/m³"
            )
            reporter.log(
                f"[STEP 5b] Val   RMSE (K={selected_output.K}, n={len(split.val)}): "
                f"{val_rmse:.5f} m³/m³"
            )
            _gap = val_rmse - train_rmse
            reporter.metric("Train RMSE", f"{train_rmse:.5f} m³/m³")
            reporter.metric("Validation RMSE", f"{val_rmse:.5f} m³/m³")
            reporter.metric("Train↔Val RMSE gap", f"{_gap:+.5f} m³/m³")
            if not split.independent:
                reporter.log(
                    f"[STEP 5b][NOTE] split method='{split.method}' is not "
                    f"fully independent (see [SPLIT][WARN] above) -- a "
                    f"small train/val gap here is expected and does not by "
                    f"itself rule out overfitting the way it would for an "
                    f"independent split."
                )
            if abs(_gap) > 0.3 * train_rmse:
                reporter.log(
                    "[STEP 5b][WARN] Val RMSE diverges >30% from train RMSE "
                    "-- possible overfitting in K selection, or val subset "
                    "too small to be representative. Treat the selected K "
                    "with caution."
                )

            # -----------------------------------------------------------
            # Train/val diagnostic plots -> <OUTPUT_DIR>/train_val/
            # -----------------------------------------------------------
            _train_val_dir = os.path.join(OUTPUT_DIR, "train_val")
            try:
                os.makedirs(_train_val_dir, exist_ok=True)
                ptv.plot_rmse_vs_k(
                    train_rmse_by_k=train_rmse_by_k,
                    val_rmse_by_k=val_rmse_by_k,
                    selected_K=selected_output.K,
                    output_dir=_train_val_dir,
                )
                ptv.plot_parity_train_val(
                    train_obs=fits_step4[selected_output.K - 1].observed_theta,
                    train_pred=fits_step4[selected_output.K - 1].predicted_theta,
                    val_obs=val_fit.observed_theta,
                    val_pred=val_fit.predicted_theta,
                    output_dir=_train_val_dir,
                )
                reporter.log(f"[STEP 5b] Train/val diagnostic plots -> {_train_val_dir}")
            except Exception as _tv_plot_exc:
                reporter.log(
                    f"[STEP 5b][WARN] Train/val diagnostic plots failed "
                    f"(pipeline continues): {type(_tv_plot_exc).__name__}: {_tv_plot_exc}"
                )

            reporter.end_step("Step 5b")

        # -----------------------------------------------------------------------
        # STEP 6 — WATER CONTENT VISUALISATIONS  (XZ always; XY in 2-D mode)
        # -----------------------------------------------------------------------
        if STAGE == "calibrate":
            reporter.log("[STAGE] calibrate: stopping before Step 6 (no maps of the probe "
                         "neighbourhoods). Run '--stage map' for the maps.")
            reporter.metric("Ensemble size (Step 1)", f"{ensemble_size} realisations  [axis 0 of ensemble matrix]")
            reporter.metric("Optimal stack depth (K)", f"{selected_output.K}  (out of {len(averaged_models)} candidates)")
            reporter.metric("Final RMSE (Step 5)", f"{selected_output.rmse:.6f} m³/m³")
            reporter.finish(exit_ok=True, quality_level=quality_gate_result.level,
                            quality_reasons=quality_gate_result.reasons)
            append_run_log(LOG_FILE, {
                "run_id":        reporter.run_id,
                "timestamp":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "scenario":      _scenario_name,
                "mode":          mode.upper(),
                "data_source":   data_source,
                "status":        "WARN" if quality_gate_result.level == "WARN" else "SUCCESS",
                "output_dir":    OUTPUT_DIR,
                "report_path":   reporter.report_path,
                "final_rmse":    reporter.get_metric("Final RMSE (Step 5)"),
                "cv_mean":       "",
                "cv_max":        "",
                "ensemble_size": reporter.get_metric("Ensemble size (Step 1)"),
                "runtime_s":     f"{reporter.elapsed_total:.1f}" if reporter.elapsed_total is not None else "",
                "quality_level": quality_gate_result.level,
                "error":         "",
            })
            return

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
            probe_y_coords=PROBE_Y_COORDS,   # local frame -- XZ probe markers only
            probe_depths=PROBE_DEPTHS,        # Z axis (unchanged)
            output_dir=OUTPUT_DIR,
            scenario_name=cfg.name if cfg else "default",
            timelapse=timelapse,
            survey_labels=None,
            utm_x=utm_x,
            utm_y=utm_y,
            # FIX: without these, plot_from_pipeline's own fallback treated
            # the LOCAL probe_x_coords/probe_y_coords above as if they were
            # already absolute UTM, mislocating every probe marker on
            # water_content_xy_{scenario}.png relative to the real UTM
            # survey grid. PROBE_X_COORDS_UTM/PROBE_Y_COORDS_UTM are the
            # correct absolute-UTM counterparts computed earlier.
            probe_utm_x=PROBE_X_COORDS_UTM,
            probe_utm_y=PROBE_Y_COORDS_UTM,
            # Real TDR station names in real-data mode (overridden above),
            # synthetic "P0","P1",... labels otherwise.
            probe_ids=PROBE_LABELS,
            xy_render_mode=xy_render_mode,
            # Auto-sizes the extra water_content_xy_{scenario}_imshow.png
            # raster's pixel size to 2 * window_m -- see plot_from_pipeline.
            window_m=WINDOW_M,
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
                    # FIX: same probe_utm_x/y fix as plot_from_pipeline above --
                    # otherwise water_content_xy_{scenario}_truth.png mislocates
                    # every probe marker the same way.
                    probe_utm_x=PROBE_X_COORDS_UTM,
                    probe_utm_y=PROBE_Y_COORDS_UTM,
                    probe_ids=PROBE_LABELS,
                    xy_render_mode=xy_render_mode,
                    window_m=WINDOW_M,
                )
                reporter.log(
                    f"[STEP 6] Ground-truth XZ (and XY, if 2-D) water-content "
                    f"plots -> {OUTPUT_DIR}/water_content_xz_"
                    f"{(cfg.name if cfg else 'default')}_truth.png"
                )
            except Exception as e:
                reporter.log(f"[STEP 6][WARN] plot_truth_from_pipeline failed: {e}")

            # Calculated-vs-truth error maps: MAE and RMSE (%VWC), XZ always
            # and XY too in 2-D mode -- these supersede the old signed
            # %-difference plot (water_content_xy_{scenario}_pctdiff.png),
            # which is undefined/blows up wherever truth ~ 0. Also logs the
            # pooled overall MAE/RMSE into the report's Metrics Summary so
            # compare_scenarios.py can pull them into scenario comparisons.
            try:
                error_metrics = pwc.plot_error_metrics_from_pipeline(
                    selected_output=selected_output,
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
                    probe_utm_x=PROBE_X_COORDS_UTM,
                    probe_utm_y=PROBE_Y_COORDS_UTM,
                    xy_render_mode="imshow",
                    window_m=WINDOW_M,
                )
                reporter.log(f"[STEP 6] Calculated-vs-truth XZ RMSE map -> {error_metrics['xz_rmse_path']}")
                reporter.log(f"[STEP 6] Calculated-vs-truth XZ MAE map  -> {error_metrics['xz_mae_path']}")
                reporter.metric("Water Content MAE vs Truth (XZ)", f"{error_metrics['xz_mae_overall']:.3f} %VWC")
                reporter.metric("Water Content RMSE vs Truth (XZ)", f"{error_metrics['xz_rmse_overall']:.3f} %VWC")
                if "xy_rmse_path" in error_metrics:
                    reporter.log(f"[STEP 6] Calculated-vs-truth XY RMSE map -> {error_metrics['xy_rmse_path']}")
                    reporter.log(f"[STEP 6] Calculated-vs-truth XY MAE map  -> {error_metrics['xy_mae_path']}")
                    reporter.metric("Water Content MAE vs Truth (XY)", f"{error_metrics['xy_mae_overall']:.3f} %VWC")
                    reporter.metric("Water Content RMSE vs Truth (XY)", f"{error_metrics['xy_rmse_overall']:.3f} %VWC")
            except Exception as e:
                reporter.log(f"[STEP 6][WARN] plot_error_metrics_from_pipeline failed: {e}")

            # Initial model (starting guess, conds0's layer MIN/MAX bounds) vs.
            # ground truth -- a QC check for whether the inversion's starting
            # point plausibly brackets reality, BEFORE any inversion runs.
            # Two flavours: depth-resolved EC_bulk (XZ heatmap, same style as
            # water_content_xz_*.png) and per-coil apparent conductivity (ECa),
            # which isn't depth-resolved so gets its own per-coil panel style.
            try:
                _ec_true_path = _resolve_npy_path("synthetic_time_lapse_ec.npy", OUTPUT_MODEL_DIR, OUTPUT_DIR)
                _eca_true_path = _resolve_npy_path("synthetic_time_lapse_eca.npy", OUTPUT_MODEL_DIR, OUTPUT_DIR)

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
            "gamma":        PARAM_GRID.get("gamma",        [GAMMA])[0],
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
            max_realisations = 1,                # cap: 1 sample realisation rendered (was 4 —
                                                   # each realisation re-inverts the full survey
                                                   # from scratch per time step; with real-data
                                                   # point counts of 20k-47k this was the main
                                                   # source of multi-hour Step 6b runtimes)
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
        steps_step6.run_step6d_solution_plots(
            reporter,
            data_source=data_source,
            theta_true_realization=theta_true_realization,
            output_model_dir=OUTPUT_MODEL_DIR,
            output_dir=OUTPUT_DIR,
            realization_index=REALIZATION_INDEX,
            selected_output=selected_output,
            averaged_models=averaged_models,
            depths0=DEPTHS0,
            layer_centers=synthetic_geometry["layer_centers"],
            cfg=cfg,
            timelapse=timelapse,
        )

        # -----------------------------------------------------------------------
        # STEP 6e — FORWARD-SIMULATE ECa FROM THE OPTIMAL MODEL
        #   Same "inject the weighted-average EC model" idea used by
        #   pediag.run_emagpy_plots_optimal() above, but here the injected
        #   model is forward-simulated (emagpy Problem.forward(), mirroring
        #   data_synthetic.py's ground-truth generation) to obtain
        #   predicted ECa, which is then compared against the raw survey ECa
        #   that was actually fed to the inversion. This is a direct
        #   forward-model check of the full pipeline's final output.
        # -----------------------------------------------------------------------
        reporter.step("Step 6e", "Forward-simulate ECa from optimal model")
        reporter.log("\n" + "=" * 80)
        reporter.log(" [PIPELINE] STEP 6e: FORWARD-SIMULATE ECa FROM OPTIMAL MODEL")
        reporter.log("=" * 80)

        # INDEXING_AUDIT_REPORT.md finding #5: for real-data, time-lapse
        # runs, averaged_models[K-1]'s spatial axis is the reference grid
        # (geometry_utils.match_epochs_to_reference_grid), not any one
        # epoch's own raw-CSV row order/count. forward_simulate_eca_optimal
        # forward-simulates ECa from that reference-grid-ordered model, so
        # the observed ECa it's compared against must be gathered in the
        # SAME reference-grid order -- REAL_DATA_MATCHED_INDICES (built
        # alongside REAL_DATA_REFERENCE_XY, see Step 1 above) is exactly
        # that per-epoch row->reference-grid-index map, and
        # plot_solution_diagnostics.forward_simulate_eca_optimal now
        # accepts it and gathers eca_obs/xpos through it accordingly.
        #
        # None for synthetic data and for real single-survey (non-time-
        # lapse) runs, where raw CSV order already IS the model's own
        # order and no remap ever happened -- forward_simulate_eca_optimal
        # keeps reading raw CSV rows directly in that case.
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
            matched_indices = (
                REAL_DATA_MATCHED_INDICES
                if (data_source == "real" and timelapse) else None
            ),
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
        append_run_log(LOG_FILE, {
            "run_id":        reporter.run_id,
            "timestamp":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario":      _scenario_name,
            "mode":          mode.upper(),
            "data_source":   data_source,
            "status":        "WARN" if quality_gate_result.level == "WARN" else "SUCCESS",
            "output_dir":    OUTPUT_DIR,
            "report_path":   reporter.report_path,
            "final_rmse":    reporter.get_metric("Final RMSE (Step 5)"),
            "cv_mean":       reporter.get_metric("CV map mean (relative uncertainty)"),
            "cv_max":        reporter.get_metric("CV map max"),
            "ensemble_size": reporter.get_metric("Ensemble size (Step 1)"),
            "runtime_s":     f"{reporter.elapsed_total:.1f}" if reporter.elapsed_total is not None else "",
            "quality_level": quality_gate_result.level,
            "error":         "",
        })

    except Exception as exc:
        reporter.failure(exc)
        append_run_log(LOG_FILE, {
            "run_id":        reporter.run_id,
            "timestamp":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario":      _scenario_name,
            "mode":          mode.upper(),
            "data_source":   data_source,
            "status":        "FAILED",
            "output_dir":    OUTPUT_DIR,
            "report_path":   reporter.report_path,
            "final_rmse":    "",
            "cv_mean":       "",
            "cv_max":        "",
            "ensemble_size": "",
            "runtime_s":     f"{reporter.elapsed_total:.1f}" if reporter.elapsed_total is not None else "",
            "quality_level": "FAILED",
            "error":         str(exc)[:300],
        })
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
        "--mode", choices=["1d", "2d"], default=None,
        help=(
            "Spatial mode. '1d': single transect, XZ plots only. "
            "'2d': rectangular UTM grid anchored to the Alento survey; "
            "produces XZ cross-section AND XY plan-view plots. "
            "Default (when not passed): '2d' for --data-source real "
            "(real EM survey positions are already absolute UTM, so "
            "there's no reason to leave 2-D output on the table), '1d' "
            "for synthetic data (unchanged legacy default). Pass "
            "--mode explicitly to override either default, e.g. to force "
            "a 1-D run on real data for a quick/lightweight check."
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
        "--full-range-gradient", action="store_true", default=False,
        help="Synthetic mode only (ignored if --homogeneous is also given). "
             "Static XZ (depth) and XY (plan-view) initial-model gradients "
             "each span the full [theta_dry, theta_wet] range independently, "
             "instead of the small fixed-amplitude default gradient. "
             "Combine with --xy-z-blend / --invert-z-gradient. Overrides "
             "'full_range_gradient' in the scenario config when passed.",
    )
    parser.add_argument(
        "--xy-z-blend", type=float, default=None, metavar="W",
        help="Only used with --full-range-gradient. Weight in [0, 1] "
             "between the XY gradient (1.0 = XY only) and the Z/depth "
             "gradient (0.0 = Z only). Overrides 'xy_z_blend' in the "
             "scenario config when passed; falls back to the scenario "
             "value (or 0.5, equal blend) otherwise.",
    )
    parser.add_argument(
        "--invert-z-gradient", action="store_true", default=False,
        help="Only used with --full-range-gradient. Flip the depth "
             "gradient so the shallowest layer is wet and the deepest "
             "layer is dry (default: dry at surface, wet at depth). "
             "Overrides 'invert_z_gradient' in the scenario config when "
             "passed.",
    )
    parser.add_argument(
        "--random-xy-field", action="store_true", default=False,
        help="Only used with --full-range-gradient. Replace the linear XY "
             "ramp with a smooth, spatially-correlated RANDOM field "
             "(scattered wet/dry patches) instead of one monotonic "
             "direction. Combine with --random-xy-n-blobs / "
             "--random-xy-correlation-m / --random-xy-seed. Overrides "
             "'random_xy_field' in the scenario config when passed.",
    )
    parser.add_argument(
        "--random-xy-n-blobs", type=int, default=None, metavar="N",
        help="Only used with --random-xy-field. Number of random Gaussian "
             "patches summed into the field. Overrides 'random_xy_n_blobs' "
             "in the scenario config when passed; falls back to the "
             "scenario value (or 8) otherwise.",
    )
    parser.add_argument(
        "--random-xy-correlation-m", type=float, default=None, metavar="M",
        help="Only used with --random-xy-field. Patch radius [m] "
             "(1-sigma). Overrides 'random_xy_correlation_m' in the "
             "scenario config when passed; falls back to the scenario "
             "value (or auto: ~1/4 of the survey's shorter extent) "
             "otherwise.",
    )
    parser.add_argument(
        "--random-xy-seed", type=int, default=None, metavar="SEED",
        help="Only used with --random-xy-field. Seeds the random field "
             "independently of the per-sample noise stream. Overrides "
             "'random_xy_seed' in the scenario config when passed; falls "
             "back to the scenario value (or a seed derived from 42) "
             "otherwise.",
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
        "--em-footprint-diagnostic", action="store_true",
        help="Compute a McNeill-type depth-weighted 'effective EM theta' per "
             "probe (see em_sensitivity.py) and report its discrepancy "
             "against theta_calc at the hard-routed layer_index and against "
             "theta_obs (Step 5c). Diagnostic only -- does not change Step "
             "2/4 calibration or Step 3/5 K-selection. Off by default.",
    )
    parser.add_argument(
        "--probe-xyz", default=None, metavar="PATH",
        help="CSV of explicit SWC/TDR probe locations, one row per probe, "
             "e.g. --probe-xyz coord_TDR.csv. Columns (case-insensitive): "
             "x/utm_x/easting, y/utm_y/northing, z/depth/depth_m, and an "
             "optional label column (id/name/probe/label). When given, "
             "completely replaces the auto-placement grid derived from "
             "--num-swc-positions/--swc-depths for BOTH 1-D and 2-D mode -- "
             "each row becomes one probe at its literal (x, y, z), no "
             "position x depth expansion.",
    )
    parser.add_argument(
        "--xy-render-mode", default="scatter", choices=["scatter", "imshow"],
        help="Rendering for XY plan-view plots (2-D mode only). 'scatter' "
             "(default) draws one square per measurement point; 'imshow' "
             "interpolates onto a smooth raster (scipy.griddata).",
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
        "--log-file", default=None, metavar="PATH",
        help="Path to the shared run-log CSV that gets one row appended "
             "per run (run_id, scenario, mode, status, key metrics, "
             "path to this run's pipeline_report_*.md, ...). Default: "
             f"'{DEFAULT_RUN_LOG}'. compare_scenarios.py --ids reads this "
             "file to resolve a run_id to its report automatically.",
    )
    parser.add_argument(
        "--gamma", nargs="+", type=float, default=None, metavar="G",
        help="One or more temporal regularisation values [>= 0, time-lapse/"
             "multi-epoch inversions only] to sweep, e.g. --gamma 0 5 10. "
             "Each value gets combined with every alpha/beta/method/"
             "forwardModel/starting-conductivity combo, same as --beta -- "
             "so N gamma values means N x as many ensemble tasks. "
             "Overrides the 'gamma' entry in the scenario's param_grid "
             "when passed; falls back to the scenario value (or [0.0]) "
             "otherwise. No effect on non-time-lapse (single-epoch) runs "
             "regardless of value.",
    )
    parser.add_argument(
        "--ws-localized-bandwidth", type=float, default=None, metavar="C",
        help="Gaspari-Cohn localization radius 'c' (same units as your x/y "
             "coordinates, e.g. metres) for the Step 2 kernel-localized "
             "Waxman-Smits diagnostic -- a sibling to the normal Step 2 "
             "calibration that fits a separate WS parameter set at every "
             "EM point, weighting nearby TDR probes more heavily. Never "
             "replaces Step 2 and never changes Steps 3-5; only adds "
             "'petro_calibration_localized.png' (and, in 2-D mode, a "
             "sigma_s(x, y) map) to the output directory. Overrides "
             "'ws_localized_bandwidth' in the scenario config when passed; "
             "falls back to the scenario value (or disabled) otherwise. "
             "Rule of thumb: pick a value at least ~half the spacing "
             "between your closest adjacent SWC/TDR probes. Ignored if "
             "--ws-localized-bandwidth-sweep is also given (the sweep's "
             "recommended bandwidth takes precedence).",
    )
    parser.add_argument(
        "--ws-localized-bandwidth-sweep", nargs="+", type=float, default=None,
        metavar="C",
        help="Candidate bandwidths (c values) for the Step 2 LOCALIZED "
             "SWEEP diagnostic: leave-one-probe-out CV over these "
             "candidates against the rank-1 Step 2 ensemble member, e.g. "
             "--ws-localized-bandwidth-sweep 1.0 2.0 5.0 10.0 15.0 20.0. "
             "The recommended (minimum pooled LOO-CV RMSE) bandwidth then "
             "becomes the localized diagnostic's bandwidth, overriding any "
             "fixed --ws-localized-bandwidth / scenario value. Overrides "
             "'ws_localized_bandwidth_sweep' in the scenario config when "
             "passed; falls back to the scenario value (or disabled, i.e. "
             "no sweep) otherwise.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke-test preset: keep ONE alpha (the middle of the scenario's "
             "alpha list) and ONE forward model (the first) instead of the full "
             "grid -- e.g. 280 Step-1 tasks -> 28. Combine with --quick-epochs "
             "to also cut the survey epochs (-> 12 with 3 of 7). Does not touch "
             "the EM/TDR data. The ensemble is shrunk on purpose, so treat the "
             "output as a smoke test, not a result. Uses its own '_quick' "
             "output folder and cache file (the full-run cache is never "
             "overwritten).",
    )
    parser.add_argument(
        "--stage", choices=list(stage_split.STAGES), default="full",
        help="Real data only. 'full' (default): one run does everything. 'calibrate': "
             "EM points restricted to the probe neighbourhoods, FULL ensemble, Steps 0-5, "
             "writes stage_calibration.json (selected K, members, WS parameters, localized "
             "bandwidth) and stops before the maps. 'map': inverts ONLY the members the "
             "calibration selected on the scenario's EM points (all, or grid-thinned), runs "
             "Steps 2-6 and reports the drift vs the calibration. Own '_calibrate' / '_map' "
             "run folder and cache file.",
    )
    parser.add_argument(
        "--calibration", default=None, metavar="PATH",
        help="With --stage map: a stage_calibration.json or the '*_calibrate' run folder "
             "that holds it. Default: the newest '*_calibrate' run under the scenario's "
             "output_dir.",
    )
    parser.add_argument(
        "--quick-epochs", type=int, default=None, metavar="N",
        help="With --quick (real data only): keep only N evenly spaced survey "
             "epochs (first and last always included). Default: all epochs.",
    )
    parser.add_argument(
        "--train-val-split", default=None,
        choices=["alternate_sorted", "by_probe", "by_time_block", "random", "off"],
        help="Split TDR/SWC observations into train (fits the Waxman-Smits "
             "relationship in Step 2/4 and selects the stack depth K in "
             "Step 3/5) and validation (held out, reported once at the end "
             "in Step 5b) subsets, mirroring Loiseau et al. (2026) Secs. "
             "2.3/2.6. 'alternate_sorted' (paper's Larzac fallback, NOT "
             "fully independent), 'by_probe' / 'by_time_block' (fully "
             "independent splits), 'random' (comparison only). Overrides "
             "'train_val_split' in the scenario config when passed; falls "
             "back to the scenario value (or 'off', i.e. no split — legacy "
             "behaviour) otherwise.",
    )
    args = parser.parse_args()

    if args.calibration is not None and args.stage != "map":
        parser.error("--calibration requires --stage map")
    if args.stage == "map" and args.quick:
        parser.error("--stage map cannot be combined with --quick (the selected members must "
                     "exist in the full parameter grid)")
    if args.quick_epochs is not None:
        if not args.quick:
            parser.error("--quick-epochs requires --quick")
        if args.quick_epochs < 1:
            parser.error("--quick-epochs must be >= 1")

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

    # Resolve effective spatial mode.
    #   - Explicit --mode always wins, for either data source (e.g. to
    #     force a quick 1-D check on real data, or a 2-D synthetic run).
    #   - Otherwise, default by data source: real data's own survey
    #     positions are already absolute UTM (see the real-data 2-D fix
    #     in main()), so there's no reason to default it to 1-D and lose
    #     XY plan-view output/localized calibration for free -- '2d' is
    #     the sensible default there. Synthetic keeps its original '1d'
    #     default unchanged.
    if args.mode is not None:
        effective_mode = args.mode
    else:
        effective_mode = "2d" if effective_ds == "real" else "1d"
        print(f"[PIPELINE] --mode not given; defaulting to "
              f"'{effective_mode}' for data_source='{effective_ds}'.")

    # Startup banner
    data_source_label = "REAL DATA" if effective_ds == "real" else "SYNTHETIC"
    print("\n" + "=" * 80)
    print(f" [PIPELINE] Data source     : {data_source_label}")
    print(f" [PIPELINE] Loaded scenario : {cfg.name}")
    if effective_ds == "real":
        print(f" [PIPELINE] EM directory    : {effective_em_dir}")
        print(f" [PIPELINE] TDR directory   : {effective_tdr_dir}")
        print(f" [PIPELINE] Coordinates     : {effective_coord_file}")
    print(f" [PIPELINE] Spatial mode    : {effective_mode.upper()}")
    print(cfg.summary())
    print("=" * 80 + "\n")

    try:
        main(cfg, mode=effective_mode, data_source=effective_ds,
             em_dir=effective_em_dir, tdr_dir=effective_tdr_dir,
             coord_file=effective_coord_file, drop_coils=args.drop_coils,
             homogeneous=(True if args.homogeneous else None),
             num_swc_positions=args.num_swc_positions,
             swc_depths=args.swc_depths,
             window_m=args.window_m,
             em_unit=args.em_unit,
             beta=args.beta,
             gamma=args.gamma,
             probe_xyz_file=args.probe_xyz,
             xy_render_mode=args.xy_render_mode,
             full_range_gradient=(True if args.full_range_gradient else None),
             xy_z_blend=args.xy_z_blend,
             invert_z_gradient=(True if args.invert_z_gradient else None),
             random_xy_field=(True if args.random_xy_field else None),
             random_xy_n_blobs=args.random_xy_n_blobs,
             random_xy_correlation_m=args.random_xy_correlation_m,
             random_xy_seed=args.random_xy_seed,
             log_file=args.log_file,
             train_val_split_method=args.train_val_split,
             ws_localized_bandwidth=args.ws_localized_bandwidth,
             ws_localized_bandwidth_sweep=args.ws_localized_bandwidth_sweep,
             em_footprint_diagnostic=args.em_footprint_diagnostic,
             quick=args.quick, quick_epochs=args.quick_epochs,
             stage=args.stage, calibration=args.calibration)
    except Exception:
        # reporter.failure() was already called inside main() and the report
        # was written there.  We just set the shell exit code to non-zero.
        sys.exit(1)