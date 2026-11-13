"""Pipeline Steps 1b-1d: real-data pre-inversion diagnostic plots (Phase 6 of refactoring.md).

Each function takes what it reads as explicit arguments. Plot failures are
logged as warnings and never abort the pipeline. No import of ``run_pipeline``
(it would be circular). The caller decides whether to run them
(``data_source == "real"``).
"""
from __future__ import annotations

import os

import plot_em_diagnostics_real as pediag_real
import plot_ini_model as pim
import plot_tdr_diagnostics as ptdrdiag
from plots import initial_ec_ws_real


def run_step1b_em_diagnostics(reporter, *, raw_em_files, min_pts, probe_table,
                              output_dir, em_unit):
    """STEP 1b: common-point footprint + ECa plan-view maps. Returns #figures (0 on failure)."""
    reporter.step("Step 1b", "Real-data EM diagnostic plots")
    reporter.log(" [PIPELINE] STEP 1b: REAL-DATA EM DIAGNOSTICS")
    reporter.log(f"            Common points (min_pts) : {min_pts}")
    reporter.log(f"            Raw EM files            : {len(raw_em_files)}")
    n = 0
    try:
        saved = pediag_real.plot_real_em_diagnostics(
            raw_em_files=raw_em_files, min_pts=min_pts, probe_table=probe_table,
            output_dir=output_dir, em_unit=em_unit)
        n = len(saved)
        reporter.log(f"[STEP 1b] {n} diagnostic figure(s) written "
                     f"to '{os.path.join(output_dir, 'diagnostics')}'")
        reporter.metric("Real-data EM diagnostic figures", n)
    except Exception as exc:
        reporter.log(f"[STEP 1b][WARN] Diagnostic plots failed (pipeline continues): "
                     f"{type(exc).__name__}: {exc}")
    reporter.end_step("Step 1b")
    return n


def run_step1c_tdr_diagnostics(reporter, *, tdr_dir, coord_file, em_dates,
                               time_tolerance_days, output_dir):
    """STEP 1c: raw per-station TDR series + QC flags. Returns #files (0 on failure)."""
    reporter.step("Step 1c", "Real-data TDR import diagnostic plots")
    reporter.log(" [PIPELINE] STEP 1c: REAL-DATA TDR IMPORT DIAGNOSTICS")
    n = 0
    try:
        saved = ptdrdiag.plot_tdr_diagnostics(
            tdr_dir=tdr_dir, coord_file=coord_file, em_dates=em_dates,
            time_tolerance_days=time_tolerance_days, output_dir=output_dir)
        n = len(saved)
        reporter.log(f"[STEP 1c] {n} diagnostic file(s) written "
                     f"to '{os.path.join(output_dir, 'diagnostics')}'")
        reporter.metric("Real-data TDR diagnostic figures", n)
    except Exception as exc:
        reporter.log(f"[STEP 1c][WARN] TDR diagnostic plots failed (pipeline continues): "
                     f"{type(exc).__name__}: {exc}")
    reporter.end_step("Step 1c")
    return n


def run_step1d_initial_model(reporter, *, em_dir, tdr_dir, coord_file, output_dir,
                             em_unit, cond_ranges, depths0, lab_p0, ws_fixed_params,
                             ws_bounds, observations, probe_table):
    """STEP 1d: initial model (raw ECa / SWC maps) + initial EC range / WS per horizon.

    ``ws_bounds`` is the already-resolved (lo, hi) pair. Returns the dict of
    initial-model figure paths ({} on failure).
    """
    reporter.step("Step 1d", "Initial model diagnostics (raw field inputs)")
    reporter.log(" [PIPELINE] STEP 1d: INITIAL MODEL DIAGNOSTICS (raw field inputs)")
    paths = {}
    try:
        paths = pim.plot_initial_model(
            em_dir=em_dir, tdr_dir=tdr_dir, coord_file=coord_file,
            output_dir=output_dir, em_unit=em_unit, em_date_index=0)
        reporter.log(
            "[STEP 1d] Initial-model plots written -> "
            f"ECa: {paths['eca_plot']}, SWC: {paths['swc_plot']}, "
            f"combined: {paths['combined_plot']}")
        reporter.metric("Initial model diagnostic figures", len(paths))
    except Exception as exc:
        reporter.log(f"[STEP 1d][WARN] Initial model plots failed (pipeline continues): "
                     f"{type(exc).__name__}: {exc}")
    try:
        ec_ws = initial_ec_ws_real.plot_initial_ec_ws_real(
            output_dir, cond_ranges=cond_ranges, depths0=depths0, lab_p0=lab_p0,
            ws_fixed_params=ws_fixed_params, ws_bounds=ws_bounds,
            observations=observations, probe_table=probe_table)
        reporter.log(f"[STEP 1d] Initial EC range / WS per horizon -> "
                     f"{ec_ws['ec']}, {ec_ws['ws']}")
    except Exception as exc:
        reporter.log(f"[STEP 1d][WARN] initial EC/WS plots failed (pipeline continues): "
                     f"{type(exc).__name__}: {exc}")
    reporter.end_step("Step 1d")
    return paths
