#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_scenarios.py
=====================
Compares N EA-EM pipeline runs (``pipeline_report_*.md`` files produced by
``run_pipeline.py``) side by side: a wide comparison table (CSV + Markdown)
plus a set of PNG plots. Works for 2 scenarios or 10 -- nothing here is
hardcoded to a pair.

Nothing in the project is modified: this only READS report files and
(optionally) ``scoring_summary.csv`` files produced by
``score_against_truth.py``.

Usage
-----
    # Note: run_pipeline.py gives every run its own numbered output folder
    # -- mfc2_outputs/<seq>_<scenario>/ (e.g. mfc2_outputs/0001_field_survey_2d/,
    # then 0002_field_survey_2d/ on the next run of the same scenario) --
    # rather than reusing one folder per scenario. --ids (below) sidesteps
    # needing to know the exact folder/run number at all, so prefer it;
    # --reports still works with a glob if you'd rather point at paths
    # directly.

    # explicit list (a specific run number per scenario)
    python3 compare_scenarios.py \\
        --reports mfc2_outputs/0001_field_survey_2d/pipeline_report_*.md \\
                  mfc2_outputs/0002_field_survey_2d_dense_tdr/pipeline_report_*.md \\
        --labels baseline dense_tdr \\
        --out-dir ./comparison_output

    # glob (quote it so the shell doesn't expand -- this script expands it)
    python3 compare_scenarios.py --reports "mfc2_outputs/*/pipeline_report_*.md" \\
        --out-dir ./comparison_output

    # optionally fold in truth-scoring CSVs (one per scenario, same order
    # as --reports/--labels) produced by score_against_truth.py
    python3 compare_scenarios.py --reports r1.md r2.md --labels A B \\
        --scoring scoring_A.csv scoring_B.csv --out-dir ./comparison_output

    # by run number -- looked up in the shared run-log CSV written by
    # run_pipeline.py (mfc2_outputs/run_log.csv by default; every run
    # appends one row there, see run_pipeline.py's append_run_log()).
    # The run number is the <seq> in .../<seq>_<scenario>/ (e.g. 7 or
    # 0007) -- use --list (via compare_scenarios.sh) or just look at the
    # folder name, no need to know report paths, filenames, or the full
    # run_id at all.
    python3 compare_scenarios.py --ids 7 12 --out-dir ./comparison_output

    # if each scenario writes into its own base output dir
    # (mfc2_outputs/<scenario>/<seq>_<scenario>/), the <seq> counter is
    # local to that scenario and can repeat across scenarios -- e.g. both
    # ending up as run 1. A plain --ids 1 is then ambiguous and errors
    # out; scope it with 'scenario:number' instead:
    python3 compare_scenarios.py --ids field_survey_2d:1 field_survey_2d_dense_tdr:1 \\
        --out-dir ./comparison_output

    # a full run_id string still works too, and can be mixed with numbers
    python3 compare_scenarios.py --ids 7 field_survey_2d_dense_tdr_20260731_091500 \\
        --out-dir ./comparison_output

    # --ids and --reports can be combined; --ids entries come first
    python3 compare_scenarios.py --ids 7 \\
        --reports "mfc2_outputs/*_moisture_2d_free/pipeline_report_*.md" \\
        --out-dir ./comparison_output

    # point-level truth comparison (depth profile, spatial transect,
    # scatter/residual) -- needs each scenario's output_model/ directory,
    # with theta_calc.npy captured via capture_synthetic_theta.py, plus
    # the synthetic_*.npy truth arrays 0_generate_synthetic_data.py already
    # writes there. Enabled by passing --position; output_dirs default to
    # each report's own directory, or the run-log's output_dir column for
    # --ids entries -- override with --output-dirs if reports don't live
    # next to output_model/.
    python3 compare_scenarios.py --ids 7 12 \\
        --position 20 10 --transect-y 10 --out-dir ./comparison_output

If --labels is omitted, labels are taken from each report's own
``Scenario`` field (falls back to the filename if that's missing/blank);
for entries resolved via --ids, the run-log's own ``scenario`` column is
used instead. If multiple report files match one --reports glob entry
(e.g. re-runs), the most recently modified one is used and a note is
printed.

Output (written to --out-dir)
------------------------------
If --out-dir is omitted, it defaults to an auto-named subfolder of
./comparison_output built from the compared runs -- e.g. --ids 7 12
writes to ./comparison_output/7_12/ -- so comparisons of different runs
don't overwrite each other. Pass --out-dir explicitly to use a fixed path
instead.

* comparison_table.csv       -- metrics (rows) x scenarios (cols), wide format
* comparison_table.md        -- same, as a Markdown table for pasting into notes/papers
* fig_rmse_and_cv.png        -- Final RMSE and CV map mean/max per scenario
* fig_water_content_error.png-- Calculated-vs-truth water content MAE/RMSE
                                 per scenario (XZ always, XY in 2-D mode)
* fig_ensemble_and_probes.png-- ensemble size, SWC probes/positions, matching fraction
* fig_eca_rmse_by_epoch.png  -- forward ECa RMSE per epoch, grouped by scenario
* fig_step_timing.png        -- stacked step-timing breakdown per scenario
* fig_truth_scoring.png      -- (only if --scoring given) RMSE/R2 vs truth per horizon
* fig_depth_profile.png      -- (only if --position given) theta vs depth, per scenario + truth
* fig_spatial_transect.png   -- (only if --position given) theta vs x along --transect-y, per layer
* fig_scatter_vs_truth.png   -- (only if --position given) calc-vs-truth scatter + residual boxplot
* truth_error_summary.csv    -- (only if --position given) per-scenario RMSE/bias vs truth
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import warnings
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Markdown parsing helpers
# --------------------------------------------------------------------------

def _extract_section(text: str, heading: str) -> str:
    """Return the text between a '## heading' line and the next '## ' line."""
    pattern = re.compile(
        r"^" + re.escape(heading) + r"\s*$\n(.*?)(?=^## |\Z)",
        re.S | re.M,
    )
    m = pattern.search(text)
    return m.group(1) if m else ""


def _parse_pipe_table(block: str) -> "OrderedDict[str, str]":
    """Parse a two-column markdown pipe table into an ordered dict."""
    rows: "OrderedDict[str, str]" = OrderedDict()
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[\s:\-|]+\|$", line):
            continue  # separator row
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 2:
            continue
        key = cols[0].strip("` ").strip()
        val = cols[1].strip()
        if key.lower() in ("field", "key", "metric", "parameter", "step"):
            continue  # header row
        rows[key] = val
    return rows


def _first_number(raw: str) -> float:
    """Pull the first float/int out of a string like '0.045383 m3/m3' or '50.0%'."""
    if raw is None:
        return float("nan")
    m = re.search(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", raw)
    return float(m.group(0)) if m else float("nan")


def parse_report(path: str) -> dict:
    """Parse one pipeline_report_*.md into a flat dict of useful fields."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    out: dict = {"_path": path}

    # --- header table (before the first '## ' heading) --------------------
    header_block = text.split("\n## ", 1)[0]
    header = _parse_pipe_table(header_block)
    out["scenario_name"] = header.get("Scenario", "").strip("`")
    out["data_source"] = header.get("Data source", "").strip("`")
    out["mode"] = header.get("Mode", "").strip("`")
    out["generated"] = header.get("Generated", "")
    out["run_id"] = header.get("Run ID", "").strip("`")

    # --- Metrics Summary ----------------------------------------------------
    metrics = _parse_pipe_table(_extract_section(text, "## Metrics Summary"))
    out["_metrics_raw"] = metrics

    def m(key, default=float("nan")):
        return _first_number(metrics.get(key)) if key in metrics else default

    out["final_rmse"] = m("Final RMSE (Step 5)")
    out["cv_mean"] = m("CV map mean (relative uncertainty)")
    out["cv_max"] = m("CV map max")
    out["theta_calc_mean"] = m("theta_calc mean [m³/m³]")
    out["ensemble_size"] = m("Ensemble size (Step 1)")
    out["optimal_K"] = m("Optimal stack depth (K)")
    out["mcmc_pstd_mean"] = m("MCMC posterior std — mean [mS/m]")
    out["mcmc_pstd_median"] = m("MCMC posterior std — median [mS/m]")
    out["n_probes"] = m("Number of SWC probes")
    out["n_swc_positions"] = m("SWC spatial positions")
    out["n_swc_obs"] = m("SWC sensor observations")
    out["matching_distinct_em_points"] = m("Matching — distinct EM points used")
    out["matching_largest_group_frac_pct"] = m("Matching — largest group fraction")
    out["quality_gate_level"] = metrics.get("Quality gate level", "")
    out["quality_gate_calibrated"] = metrics.get("Quality gate — calibrated horizons", "")
    out["inversion_tasks_failed"] = m("Inversion tasks failed")
    out["inversion_combos_dropped"] = m("Ensemble combos dropped")

    # Calculated-vs-truth water content error (synthetic mode only) --
    # written by run_pipeline.py Step 6 (see plot_error_metrics_from_pipeline).
    # XY entries are only present in 2-D mode; m() defaults to NaN when a
    # key is absent, so 1-D reports just show NaN for the XY columns.
    out["wc_mae_xz"] = m("Water Content MAE vs Truth (XZ)")
    out["wc_rmse_xz"] = m("Water Content RMSE vs Truth (XZ)")
    out["wc_mae_xy"] = m("Water Content MAE vs Truth (XY)")
    out["wc_rmse_xy"] = m("Water Content RMSE vs Truth (XY)")

    # per-epoch forward ECa RMSE: keys look like
    # "Forward ECa RMSE (optimal, t=0)" .. t=N
    eca = {}
    for k, v in metrics.items():
        mt = re.match(r"Forward ECa RMSE \(optimal, t=(\d+)\)", k)
        if mt:
            eca[int(mt.group(1))] = _first_number(v)
    out["eca_rmse_by_epoch"] = OrderedDict(sorted(eca.items()))

    # --- Step Timing ---------------------------------------------------------
    timing_rows = _parse_pipe_table(_extract_section(text, "## Step Timing"))
    timing = OrderedDict()
    total = float("nan")
    for k, v in timing_rows.items():
        k_clean = k.strip("* ")
        val = _first_number(v.replace("**", ""))
        if k_clean.lower() == "total":
            total = val
        else:
            timing[k_clean] = val
    out["step_timing"] = timing
    out["total_runtime_s"] = total

    # --- Result ---------------------------------------------------------------
    result_block = _extract_section(text, "## Result").strip()
    out["result_line"] = result_block.splitlines()[0].strip() if result_block else ""
    out["result_ok"] = "SUCCESS" in out["result_line"] and "WARN" not in out["result_line"].upper()

    return out


# --------------------------------------------------------------------------
# Report file resolution
# --------------------------------------------------------------------------

def resolve_report_paths(patterns: list[str]) -> list[str]:
    resolved = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if not matches:
            # maybe it's a literal existing path with glob-looking chars
            if os.path.exists(pat):
                matches = [pat]
            else:
                raise FileNotFoundError(f"No file matches pattern: {pat!r}")
        if len(matches) > 1:
            matches.sort(key=os.path.getmtime)
            chosen = matches[-1]
            print(f"[compare_scenarios] NOTE: {len(matches)} files matched "
                  f"{pat!r}; using most recent -> {chosen}", file=sys.stderr)
            resolved.append(chosen)
        else:
            resolved.append(matches[0])
    return resolved


def _seq_from_output_dir(output_dir: str):
    """
    Extract the leading numeric sequence from a run's output_dir folder
    name, e.g. 1 from 'mfc2_outputs/0001_field_survey_2d'. Returns None
    if the folder name doesn't start with digits (older/custom runs).
    """
    if not output_dir:
        return None
    base = os.path.basename(os.path.normpath(str(output_dir)))
    m = re.match(r"^(\d+)", base)
    return int(m.group(1)) if m else None


def resolve_ids_to_reports(ids: list[str], log_file: str) -> "OrderedDict[str, dict]":
    """
    Look up one or more run identifiers in the shared run-log CSV written
    by ``run_pipeline.py`` (see its ``append_run_log`` / ``DEFAULT_RUN_LOG``)
    and return, in the same order as *ids*, an ``OrderedDict`` of
    ``run_id -> {"report_path": ..., "scenario": ..., "output_dir": ...}``.

    Each entry in *ids* may be any of:
      * a plain run number -- e.g. ``7`` or ``0007`` -- matched against the
        leading ``<seq>_`` in the run's output_dir folder name (the number
        run_pipeline.py itself names the output folder with, e.g.
        ``mfc2_outputs/0007_field_survey_2d/``). This number is only
        unique *within a scenario* if each scenario writes into its own
        base output dir (``mfc2_outputs/<scenario>/<seq>_<scenario>/``),
        so if it matches more than one run across different scenarios,
        resolution fails with a message telling you to disambiguate;
      * ``<scenario>:<number>`` -- the run number scoped to one scenario,
        e.g. ``field_survey_2d:1`` -- always unambiguous, use this when
        plain numbers collide across scenarios. ``<scenario>`` can be
        either the folder name you see on disk (the parent directory of
        output_dir, e.g. ``field_survey_2d``) or the log's own
        ``scenario`` column value (e.g. ``2d``) if that differs;
      * a full ``run_id`` string from the log's ``run_id`` column (still
        accepted, for scripts/history that already reference it).
    A plain number (or scenario:number) is tried first; if nothing matches
    it falls back to matching the literal run_id column, so styles can be
    mixed freely in the same ``--ids`` list.

    Raises ``FileNotFoundError``/``ValueError`` with a clear message if
    the log file is missing, malformed, doesn't contain a given id, a
    plain number is ambiguous, or a resolved report_path no longer exists
    on disk.
    """
    if not os.path.isfile(log_file):
        raise FileNotFoundError(
            f"--ids given but run-log CSV not found: {log_file!r} "
            f"(this file is written by run_pipeline.py; pass --log-file "
            f"to point at a different location)."
        )
    log_df = pd.read_csv(log_file, dtype=str)
    if "run_id" not in log_df.columns:
        raise ValueError(f"{log_file!r} has no 'run_id' column -- is this a run-log CSV?")

    seqs = log_df.get("output_dir", pd.Series([""] * len(log_df))).map(_seq_from_output_dir)
    scenario_col = log_df.get("scenario", pd.Series([""] * len(log_df)))
    # The log's 'scenario' column can be a short internal code (e.g. '2d')
    # that differs from the folder name a person actually sees on disk
    # (e.g. 'field_survey_2d', the parent dir of output_dir when each
    # scenario gets its own base output dir). Accept either as the
    # scenario part of 'scenario:number'.
    folder_scenario = log_df.get("output_dir", pd.Series([""] * len(log_df))).map(
        lambda p: os.path.basename(os.path.dirname(str(p))) if p else ""
    )

    out: "OrderedDict[str, dict]" = OrderedDict()
    for raw_id in ids:
        token = raw_id.strip()
        matches = pd.DataFrame()

        scenario_part, sep, num_part = token.rpartition(":")
        if sep and num_part.isdigit():
            # scenario:number -- always unambiguous
            matches = log_df[
                ((scenario_col == scenario_part) | (folder_scenario == scenario_part))
                & (seqs == int(num_part))
            ]
        elif token.isdigit():
            matches = log_df[seqs == int(token)]
            if not matches.empty and matches["run_id"].nunique() > 1:
                conflicts = ", ".join(
                    f"{fs!r} ({rid})" for fs, rid in
                    zip(folder_scenario[matches.index], matches["run_id"])
                )
                first_fs = folder_scenario[matches.index[0]]
                raise ValueError(
                    f"run number {token!r} is ambiguous -- it matches more than "
                    f"one scenario's run: {conflicts}. Disambiguate with "
                    f"'<scenario>:{token}', e.g. '{first_fs}:{token}', "
                    f"or pass the full run_id instead."
                )
        if matches.empty:
            matches = log_df[log_df["run_id"] == token]
        if matches.empty:
            raise ValueError(
                f"{raw_id!r} not found in {log_file!r} (tried it as a run "
                f"number, as scenario:number, and as a literal run_id)"
            )
        # If duplicated (shouldn't normally happen), take the most recent row.
        row = matches.iloc[-1]
        report_path = row.get("report_path", "")
        if not report_path or not os.path.isfile(report_path):
            raise FileNotFoundError(
                f"{raw_id!r} resolved to report_path {report_path!r}, "
                f"which does not exist on disk (was the output_dir moved, "
                f"cleaned, or re-run since?)."
            )
        out[row["run_id"]] = {
            "report_path": report_path,
            "scenario": row.get("scenario", "") or row["run_id"],
            "output_dir": row.get("output_dir", "") or os.path.dirname(report_path),
        }
    return out


# --------------------------------------------------------------------------
# Table assembly
# --------------------------------------------------------------------------

SCALAR_METRIC_ROWS = [
    ("scenario_name", "Scenario name"),
    ("data_source", "Data source"),
    ("mode", "Mode"),
    ("result_line", "Result"),
    ("quality_gate_level", "Quality gate"),
    ("final_rmse", "Final RMSE (Step 5) [m3/m3]"),
    ("cv_mean", "CV map mean"),
    ("cv_max", "CV map max"),
    ("theta_calc_mean", "theta_calc mean [m3/m3]"),
    ("ensemble_size", "Ensemble size"),
    ("optimal_K", "Optimal stack depth K"),
    ("mcmc_pstd_mean", "MCMC posterior std mean [mS/m]"),
    ("mcmc_pstd_median", "MCMC posterior std median [mS/m]"),
    ("n_probes", "Number of SWC probes"),
    ("n_swc_positions", "SWC spatial positions"),
    ("n_swc_obs", "SWC sensor observations"),
    ("matching_distinct_em_points", "Matching: distinct EM points used"),
    ("matching_largest_group_frac_pct", "Matching: largest group fraction [%]"),
    ("inversion_tasks_failed", "Inversion tasks failed"),
    ("inversion_combos_dropped", "Ensemble combos dropped"),
    ("wc_mae_xz", "Water content MAE vs truth, XZ [%VWC]"),
    ("wc_rmse_xz", "Water content RMSE vs truth, XZ [%VWC]"),
    ("wc_mae_xy", "Water content MAE vs truth, XY [%VWC]"),
    ("wc_rmse_xy", "Water content RMSE vs truth, XY [%VWC]"),
    ("total_runtime_s", "Total runtime [s]"),
]


def build_comparison_table(parsed: "OrderedDict[str, dict]") -> pd.DataFrame:
    data = OrderedDict()
    for label, rec in parsed.items():
        col = OrderedDict()
        for key, nice_name in SCALAR_METRIC_ROWS:
            col[nice_name] = rec.get(key, float("nan"))
        data[label] = col
    df = pd.DataFrame(data)
    df.index.name = "Metric"
    return df


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def _bar_grid(ax, labels, values, ylabel, title, fmt="{:.3f}"):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=plt.cm.tab10(np.linspace(0, 1, len(labels))))
    ax.set_xticks(x, labels=labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    fmt.format(v), ha="center", va="bottom", fontsize=8)


def plot_rmse_and_cv(parsed, out_dir):
    labels = list(parsed.keys())
    final_rmse = [parsed[l]["final_rmse"] for l in labels]
    cv_mean = [parsed[l]["cv_mean"] for l in labels]
    cv_max = [parsed[l]["cv_max"] for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(4.5 * len(labels) ** 0.3 + 6, 4.2))
    _bar_grid(axes[0], labels, final_rmse, "RMSE [m3/m3]", "Final RMSE (Step 5)")
    _bar_grid(axes[1], labels, cv_mean, "CV (mean)", "Mean CV (relative uncertainty)")
    _bar_grid(axes[2], labels, cv_max, "CV (max)", "Max CV")
    for ax, thresh in zip(axes[1:], [0.30, None]):
        if thresh:
            ax.axhline(thresh, color="red", ls="--", lw=1, label=f"warn threshold {thresh}")
            ax.legend(fontsize=8)
    fig.suptitle("Solution quality across scenarios")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_rmse_and_cv.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_water_content_error(parsed, out_dir):
    """
    Bar chart of calculated-vs-truth water content MAE/RMSE (%VWC) per
    scenario -- XZ always, XY too wherever a scenario ran in 2-D mode.
    Synthetic-mode only (needs ground truth); returns None if none of the
    parsed reports have these metrics (e.g. all real-data runs).
    """
    labels = list(parsed.keys())
    mae_xz = [parsed[l]["wc_mae_xz"] for l in labels]
    rmse_xz = [parsed[l]["wc_rmse_xz"] for l in labels]
    mae_xy = [parsed[l]["wc_mae_xy"] for l in labels]
    rmse_xy = [parsed[l]["wc_rmse_xy"] for l in labels]

    if not any(np.isfinite(v) for v in mae_xz + rmse_xz + mae_xy + rmse_xy):
        return None

    has_xy = any(np.isfinite(v) for v in mae_xy + rmse_xy)
    n_panels = 4 if has_xy else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * len(labels) ** 0.3 + 3 * n_panels, 4.2))
    if n_panels == 1:
        axes = [axes]

    _bar_grid(axes[0], labels, mae_xz, "MAE [%VWC]", "Water Content MAE vs Truth (XZ)")
    _bar_grid(axes[1], labels, rmse_xz, "RMSE [%VWC]", "Water Content RMSE vs Truth (XZ)")
    if has_xy:
        _bar_grid(axes[2], labels, mae_xy, "MAE [%VWC]", "Water Content MAE vs Truth (XY)")
        _bar_grid(axes[3], labels, rmse_xy, "RMSE [%VWC]", "Water Content RMSE vs Truth (XY)")

    fig.suptitle("Calculated-vs-truth water content error across scenarios")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_water_content_error.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ensemble_and_probes(parsed, out_dir):
    labels = list(parsed.keys())
    ensemble = [parsed[l]["ensemble_size"] for l in labels]
    probes = [parsed[l]["n_probes"] for l in labels]
    positions = [parsed[l]["n_swc_positions"] for l in labels]
    match_frac = [parsed[l]["matching_largest_group_frac_pct"] for l in labels]

    fig, axes = plt.subplots(1, 4, figsize=(4.2 * len(labels) ** 0.3 + 8, 4.2))
    _bar_grid(axes[0], labels, ensemble, "n realisations", "Ensemble size", fmt="{:.0f}")
    _bar_grid(axes[1], labels, probes, "n probes", "SWC/TDR probes", fmt="{:.0f}")
    _bar_grid(axes[2], labels, positions, "n positions", "SWC spatial positions", fmt="{:.0f}")
    _bar_grid(axes[3], labels, match_frac, "%", "Largest EM-match group fraction", fmt="{:.1f}")
    fig.suptitle("Sampling design & EM-TDR matching across scenarios")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_ensemble_and_probes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_eca_rmse_by_epoch(parsed, out_dir):
    labels = list(parsed.keys())
    all_epochs = sorted({e for l in labels for e in parsed[l]["eca_rmse_by_epoch"]})
    if not all_epochs:
        return None

    n_labels = len(labels)
    x = np.arange(len(all_epochs))
    width = 0.8 / max(n_labels, 1)

    fig, ax = plt.subplots(figsize=(1.4 * len(all_epochs) + 2.5, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_labels))
    for i, l in enumerate(labels):
        vals = [parsed[l]["eca_rmse_by_epoch"].get(e, np.nan) for e in all_epochs]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=l, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([f"t={e}" for e in all_epochs])
    ax.set_ylabel("Forward ECa RMSE [mS/m]")
    ax.set_title("Forward-simulated ECa RMSE by epoch (optimal model)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_eca_rmse_by_epoch.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_step_timing(parsed, out_dir):
    labels = list(parsed.keys())
    all_steps = sorted(
        {s for l in labels for s in parsed[l]["step_timing"]},
        key=lambda s: (s != "Step 0", s),
    )
    if not all_steps:
        return None

    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 3, 5.5))
    bottoms = np.zeros(len(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_steps)))
    for step, color in zip(all_steps, colors):
        vals = np.array([parsed[l]["step_timing"].get(step, 0.0) for l in labels])
        vals = np.nan_to_num(vals)
        ax.bar(labels, vals, bottom=bottoms, label=step, color=color)
        bottoms += vals
    for i, l in enumerate(labels):
        ax.text(i, bottoms[i], f"{bottoms[i]:.1f}s", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Duration [s]")
    ax.set_title("Step-by-step runtime breakdown")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=20, ha="right")
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_step_timing.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_truth_scoring(scoring_paths: "OrderedDict[str, str]", out_dir: str):
    """Optional: compare RMSE/R2 vs known ground truth, from score_against_truth.py CSVs."""
    frames = {}
    for label, path in scoring_paths.items():
        if not path:
            continue
        if not os.path.exists(path):
            print(f"[compare_scenarios] WARNING: scoring file not found for "
                  f"'{label}': {path} -- skipping.", file=sys.stderr)
            continue
        frames[label] = pd.read_csv(path)
    if not frames:
        return None

    # aggregate mean RMSE/R2 per horizon, pooled across epochs
    fig, axes = plt.subplots(1, 2, figsize=(1.6 * len(frames) + 6, 4.5))
    labels = list(frames.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for i, l in enumerate(labels):
        df = frames[l]
        agg = df.groupby("horizon", as_index=False)[["rmse", "r2"]].mean()
        axes[0].plot(agg["horizon"], agg["rmse"], marker="o", color=colors[i], label=l)
        axes[1].plot(agg["horizon"], agg["r2"], marker="o", color=colors[i], label=l)
    axes[0].set_xlabel("Horizon (layer idx)")
    axes[0].set_ylabel("Mean RMSE vs truth [m3/m3]")
    axes[0].set_title("RMSE vs known ground truth, per horizon")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Horizon (layer idx)")
    axes[1].set_ylabel("Mean R2 vs truth")
    axes[1].set_title("R2 vs known ground truth, per horizon")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_truth_scoring.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Point-level truth comparison (depth profile / spatial transect / scatter)
# --------------------------------------------------------------------------
# Everything above this reads pipeline_report_*.md -- scalar metrics only.
# These functions instead read the per-point arrays out of each scenario's
# <output_dir>/output_model/ directory:
#   synthetic_time_lapse_theta.npy / synthetic_x_coords.npy /
#   synthetic_y_coords.npy / synthetic_layer_centers.npy /
#   synthetic_depth_boundaries.npy   -- written by 0_generate_synthetic_data.py,
#                                        already present for every synthetic run.
#   theta_calc.npy                   -- the pipeline's final inverted model.
#                                        NOT written by run_pipeline.py itself
#                                        (Step 4/5's SelectionOutput is only
#                                        ever plotted, never saved) -- run
#                                        capture_synthetic_theta.py once per
#                                        scenario to produce it.
# A scenario missing output_model/ or the synthetic_*.npy truth arrays is
# skipped with a warning (same "don't abort the whole comparison" style as
# plot_truth_scoring above); a scenario missing just theta_calc.npy still
# contributes its truth curve, with a note that calc comparisons aren't
# available for it.

_TAB10 = plt.cm.tab10(np.linspace(0, 1, 10))


def load_truth_scenario(output_dir: str, label: str, realization: int) -> "dict | None":
    """Load one scenario's truth + (optional) inverted model from
    ``<output_dir>/output_model/``. Returns None (with a warning) if the
    truth arrays aren't there at all -- unlike the report-parsing path
    above, this is optional/best-effort, so it never raises."""
    om = os.path.join(output_dir, "output_model")
    required = {
        "theta_true": "synthetic_time_lapse_theta.npy",
        "x": "synthetic_x_coords.npy",
        "y": "synthetic_y_coords.npy",
        "layer_centers": "synthetic_layer_centers.npy",
        "depths0": "synthetic_depth_boundaries.npy",
    }
    missing = [fn for fn in required.values() if not os.path.isfile(os.path.join(om, fn))]
    if missing:
        print(f"[compare_scenarios] WARNING: truth comparison skipped for "
              f"'{label}' -- missing {missing} under {om!r} (not a synthetic "
              f"run, or output_model/ wasn't found).", file=sys.stderr)
        return None

    theta_true_ens = np.load(os.path.join(om, required["theta_true"]))
    if realization >= theta_true_ens.shape[0]:
        print(f"[compare_scenarios] WARNING: truth comparison skipped for "
              f"'{label}' -- --realization {realization} out of range "
              f"(truth ensemble has {theta_true_ens.shape[0]} members).", file=sys.stderr)
        return None

    x_coords = np.load(os.path.join(om, required["x"]))
    y_coords = np.load(os.path.join(om, required["y"]))
    layer_centers = np.load(os.path.join(om, required["layer_centers"]))
    depths0 = np.load(os.path.join(om, required["depths0"]))

    theta_calc_path = os.path.join(om, "theta_calc.npy")
    theta_calc = None
    if os.path.isfile(theta_calc_path):
        theta_calc = np.load(theta_calc_path)
        if theta_calc.shape[1] != len(x_coords):
            warnings.warn(
                f"[{label}] theta_calc has {theta_calc.shape[1]} points but "
                f"synthetic_x_coords has {len(x_coords)} -- point ordering "
                f"may not line up 1:1; skipping calc plots for this scenario."
            )
            theta_calc = None
    else:
        print(f"[compare_scenarios] NOTE: '{label}' has no theta_calc.npy -- "
              f"run capture_synthetic_theta.py for this scenario to get "
              f"calc-vs-truth comparisons. Truth-only curve still included.")

    return dict(
        label=label, x=np.asarray(x_coords, dtype=float), y=np.asarray(y_coords, dtype=float),
        layer_centers=np.asarray(layer_centers, dtype=float), depths0=np.asarray(depths0, dtype=float),
        theta_true=theta_true_ens[realization], theta_calc=theta_calc,
    )


def _z_for_layers(layer_centers: np.ndarray, depths0: np.ndarray, n_layer: int) -> np.ndarray:
    """Same half-space extrapolation convention as run_pipeline.plot_truth_tdr
    / plot_water_content._layer_centers, for theta arrays with
    n_layer > len(depths0)+1."""
    if n_layer <= len(layer_centers):
        return layer_centers[:n_layer]
    last_bottom = depths0[-1] if len(depths0) > 0 else 0.0
    return np.concatenate([layer_centers, [last_bottom + 0.4]])[:n_layer]


def _nearest_index(x, y, x0, y0) -> int:
    return int(np.argmin((x - x0) ** 2 + (y - y0) ** 2))


def plot_depth_profile(scenarios: list[dict], position: tuple[float, float],
                        time_index: int, out_dir: str) -> "str | None":
    x0, y0 = position
    ref = scenarios[0]
    fig, ax = plt.subplots(figsize=(6, 6.5))

    t_idx = time_index if time_index >= 0 else ref["theta_true"].shape[0] + time_index
    ref_pt = _nearest_index(ref["x"], ref["y"], x0, y0)
    z_true = _z_for_layers(ref["layer_centers"], ref["depths0"], ref["theta_true"].shape[-1])
    ax.plot(ref["theta_true"][t_idx, ref_pt, :], z_true, "o-", lw=3, color="black",
            label=f"Truth ({ref['label']}, pt @ x={ref['x'][ref_pt]:.1f}, y={ref['y'][ref_pt]:.1f})",
            zorder=10)

    for i, sc in enumerate(scenarios):
        if sc["theta_calc"] is None or sc["theta_calc"].shape[1] != len(sc["x"]):
            continue
        n_time_calc = sc["theta_calc"].shape[0]
        t = min(max(time_index if time_index >= 0 else n_time_calc + time_index, 0), n_time_calc - 1)
        pt = _nearest_index(sc["x"], sc["y"], x0, y0)
        z_calc = _z_for_layers(sc["layer_centers"], sc["depths0"], sc["theta_calc"].shape[-1])
        ax.plot(sc["theta_calc"][t, pt, :], z_calc, "s--", lw=2, color=_TAB10[i % 10],
                label=f"{sc['label']} (calc, pt @ x={sc['x'][pt]:.1f}, y={sc['y'][pt]:.1f})")

    ax.invert_yaxis()
    ax.set_xlabel(r"$\theta$ [m$^3$/m$^3$]")
    ax.set_ylabel("Depth [m]")
    ax.set_title(f"Depth profile @ ({x0:.1f}, {y0:.1f}) — inverted model vs. truth")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_depth_profile.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_spatial_transect(scenarios: list[dict], transect_y: float, tol: float,
                           time_index: int, out_dir: str) -> "str | None":
    ref = scenarios[0]
    n_layer = ref["theta_true"].shape[-1]
    z = _z_for_layers(ref["layer_centers"], ref["depths0"], n_layer)
    t_idx = time_index if time_index >= 0 else ref["theta_true"].shape[0] + time_index

    mask_ref = np.abs(ref["y"] - transect_y) <= tol
    if not mask_ref.any():
        print(f"[compare_scenarios] WARNING: spatial transect skipped -- no truth "
              f"points within tol={tol} of transect_y={transect_y} "
              f"(range [{ref['y'].min():.1f}, {ref['y'].max():.1f}]).", file=sys.stderr)
        return None
    order_ref = np.argsort(ref["x"][mask_ref])
    x_ref = ref["x"][mask_ref][order_ref]

    fig, axes = plt.subplots(1, n_layer, figsize=(4.5 * n_layer, 4.5), squeeze=False, sharey=True)
    axes = axes[0]
    for l in range(n_layer):
        ax = axes[l]
        ax.plot(x_ref, ref["theta_true"][t_idx, mask_ref, l][order_ref], "o-", lw=3,
                color="black", label="Truth", zorder=10)
        for i, sc in enumerate(scenarios):
            if sc["theta_calc"] is None or sc["theta_calc"].shape[1] != len(sc["x"]):
                continue
            n_time_calc = sc["theta_calc"].shape[0]
            t = min(max(time_index if time_index >= 0 else n_time_calc + time_index, 0), n_time_calc - 1)
            mask = np.abs(sc["y"] - transect_y) <= tol
            if not mask.any() or l >= sc["theta_calc"].shape[-1]:
                continue
            order = np.argsort(sc["x"][mask])
            ax.plot(sc["x"][mask][order], sc["theta_calc"][t, mask, l][order],
                    "s--", lw=1.5, color=_TAB10[i % 10], label=sc["label"])
        ax.set_title(f"Depth {z[l]:.2f} m")
        ax.set_xlabel("x [m]")
        ax.grid(alpha=0.3)
        if l == 0:
            ax.set_ylabel(r"$\theta$ [m$^3$/m$^3$]")
            ax.legend(fontsize=8)
    fig.suptitle(f"Spatial transect @ y≈{transect_y:.1f} (±{tol}) — inverted model vs. truth", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_spatial_transect.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _matched_calc_truth(sc: dict, time_index: int):
    theta_calc, theta_true = sc["theta_calc"], sc["theta_true"]
    n_time = min(theta_calc.shape[0], theta_true.shape[0])
    t = min(max(time_index if time_index >= 0 else n_time + time_index, 0), n_time - 1)
    calc_t, true_t = theta_calc[t], theta_true[t]
    n_layer = min(calc_t.shape[-1], true_t.shape[-1])
    idx = np.arange(calc_t.shape[0]) if calc_t.shape[0] == true_t.shape[0] else None
    if idx is None:
        # point counts disagree -- shouldn't reach here since callers only
        # pass scenarios where this already matched (see load_truth_scenario)
        return np.array([]), np.array([])
    return calc_t[:, :n_layer].ravel(), true_t[idx][:, :n_layer].ravel()


def plot_scatter_and_residual(scenarios: list[dict], time_index: int, out_dir: str):
    usable = [sc for sc in scenarios
              if sc["theta_calc"] is not None and sc["theta_calc"].shape[1] == len(sc["x"])]
    if not usable:
        print("[compare_scenarios] Scatter/residual truth plot skipped -- "
              "no scenario has a usable theta_calc.npy.", file=sys.stderr)
        return None, pd.DataFrame()

    fig, (ax_scatter, ax_resid) = plt.subplots(1, 2, figsize=(11, 5))
    rows, resid_data, labels = [], [], []
    for i, sc in enumerate(usable):
        calc, true = _matched_calc_truth(sc, time_index)
        ax_scatter.scatter(true, calc, s=10, alpha=0.5, color=_TAB10[i % 10], label=sc["label"])
        resid = calc - true
        rows.append(dict(scenario=sc["label"], rmse=float(np.sqrt(np.mean(resid ** 2))),
                          bias=float(np.mean(resid)), n=len(resid)))
        resid_data.append(resid)
        labels.append(sc["label"])

    lims = [min(ax_scatter.get_xlim()[0], ax_scatter.get_ylim()[0]),
            max(ax_scatter.get_xlim()[1], ax_scatter.get_ylim()[1])]
    ax_scatter.plot(lims, lims, "k--", lw=1, label="1:1")
    ax_scatter.set_xlabel(r"Truth $\theta$ [m$^3$/m$^3$]")
    ax_scatter.set_ylabel(r"Calc $\theta$ [m$^3$/m$^3$]")
    ax_scatter.set_title("Inverted vs. truth (all points, all layers)")
    ax_scatter.grid(alpha=0.3)
    ax_scatter.legend(fontsize=8)

    ax_resid.axhline(0, color="k", lw=1)
    ax_resid.boxplot(resid_data, tick_labels=labels, showmeans=True)
    ax_resid.set_ylabel(r"Residual (calc − truth) [m$^3$/m$^3$]")
    ax_resid.set_title("Residual distribution")
    ax_resid.tick_params(axis="x", rotation=20)
    ax_resid.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_scatter_vs_truth.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, pd.DataFrame(rows)


def run_truth_comparison(labels: list[str], output_dirs: list[str], realization: int,
                          time_index: int, position: tuple[float, float],
                          transect_y: "float | None", transect_tol: float, out_dir: str) -> list[str]:
    """Top-level entry point called from main() when --position is given.
    Loads what it can, skips what it can't, never raises -- consistent with
    how the rest of compare_scenarios.py treats optional inputs."""
    scenarios = []
    for label, output_dir in zip(labels, output_dirs):
        sc = load_truth_scenario(output_dir, label, realization)
        if sc is not None:
            scenarios.append(sc)
    if not scenarios:
        print("[compare_scenarios] WARNING: truth comparison skipped entirely -- "
              "no scenario had a usable output_model/ directory.", file=sys.stderr)
        return []

    made = []
    made.append(plot_depth_profile(scenarios, position, time_index, out_dir))
    ty = transect_y if transect_y is not None else float(np.median(scenarios[0]["y"]))
    made.append(plot_spatial_transect(scenarios, ty, transect_tol, time_index, out_dir))
    scatter_path, err_df = plot_scatter_and_residual(scenarios, time_index, out_dir)
    made.append(scatter_path)
    if not err_df.empty:
        csv_path = os.path.join(out_dir, "truth_error_summary.csv")
        err_df.to_csv(csv_path, index=False)
        made.append(csv_path)
        print(err_df.to_string(index=False))
    return [p for p in made if p]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _sanitize_for_dirname(token: str) -> str:
    """Turn a run id / label / filename into a safe, readable folder-name
    fragment: keep alphanumerics, dash, underscore, dot; collapse/replace
    everything else (spaces, colons, slashes, glob chars, ...) with '_'."""
    token = str(token).strip()
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token)
    return token.strip("_.") or "run"


def default_out_dir_from_tokens(id_tokens: list[str], report_only_names: list[str]) -> str:
    """Build the auto out-dir name (under ./comparison_output/) from the
    raw --ids tokens (in the order given) followed by names for any
    --reports-only entries (their default label / filename). Falls back to
    'comparison' if nothing usable is available, and truncates very long
    combinations (many ids) to keep the path sane."""
    parts = [_sanitize_for_dirname(t) for t in id_tokens] + \
            [_sanitize_for_dirname(t) for t in report_only_names]
    parts = [p for p in parts if p]
    if not parts:
        name = "comparison"
    else:
        name = "_".join(parts)
        if len(name) > 120:
            # Too long to be a friendly folder name (e.g. many runs) --
            # keep the first few ids and note the count instead.
            head = "_".join(parts[:4])
            name = f"{head}_and_{len(parts) - 4}_more"
    return os.path.join("./comparison_output", name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", nargs="+", default=None,
                     help="Paths or glob patterns to pipeline_report_*.md files, "
                          "one entry per scenario. Combine with --ids if given; "
                          "--ids entries come first, in order.")
    ap.add_argument("--ids", nargs="+", default=None,
                     help="One or more runs to compare, from the shared run-log CSV "
                          "(written by run_pipeline.py, default location "
                          "mfc2_outputs/run_log.csv) -- each is resolved to its "
                          "report_path automatically, no need to type report "
                          "paths/globs by hand. Give the plain run number (e.g. 7 "
                          "or 0007 -- the <seq> in .../<seq>_<scenario>/); if each "
                          "scenario writes to its own base output dir this number "
                          "can repeat across scenarios, so use 'scenario:number' "
                          "(e.g. field_survey_2d:7) to pin one, or pass the full "
                          "run_id string. Styles can be mixed. Combine with "
                          "--reports if given.")
    ap.add_argument("--log-file", default="mfc2_outputs/run_log.csv",
                     help="Path to the run-log CSV written by run_pipeline.py "
                          "(default: mfc2_outputs/run_log.csv). Only used when "
                          "--ids is given.")
    ap.add_argument("--labels", nargs="+", default=None,
                     help="Optional scenario labels, same order/count as the "
                          "combined --ids + --reports list. Defaults to the "
                          "run-log's scenario column (for --ids entries) or "
                          "the Scenario field parsed from each report.")
    ap.add_argument("--scoring", nargs="+", default=None,
                     help="Optional scoring_summary.csv files from score_against_truth.py, "
                          "same order/count as --reports.")
    ap.add_argument("--output-dirs", nargs="+", default=None,
                     help="Optional override: pipeline output_dir per scenario (each must "
                          "contain output_model/), same order/count as the combined "
                          "--ids + --reports list. Only needed for the point-level truth "
                          "comparison (--position). Defaults to the run-log's output_dir "
                          "column for --ids entries, or each report's own directory for "
                          "--reports entries.")
    ap.add_argument("--position", type=float, nargs=2, metavar=("X", "Y"), default=None,
                     help="(x, y) position for a depth-profile-vs-truth plot; nearest grid "
                          "point is used. Enables the point-level truth comparison plots "
                          "(depth profile, spatial transect, scatter/residual) -- needs each "
                          "scenario's output_model/ (theta_calc.npy via "
                          "capture_synthetic_theta.py + the synthetic_*.npy truth arrays).")
    ap.add_argument("--transect-y", type=float, default=None,
                     help="y-value for the spatial transect plot. Default: median y of the "
                          "first scenario. Only used with --position.")
    ap.add_argument("--transect-tol", type=float, default=1.0,
                     help="Tolerance [m] for selecting points near --transect-y. Default: %(default)s")
    ap.add_argument("--realization", type=int, default=0,
                     help="Ensemble truth realization index (must match each scenario's "
                          "config 'realization_index'). Default: %(default)s")
    ap.add_argument("--time-index", type=int, default=-1,
                     help="Time/epoch index for the truth-comparison plots (-1 = last). "
                          "Default: %(default)s")
    ap.add_argument("--out-dir", default=None,
                     help="Directory to write comparison_table.csv/.md and the "
                          "comparison PNGs into. Default: an auto-named subfolder "
                          "of ./comparison_output built from the compared run "
                          "ids/labels, e.g. ./comparison_output/7_12 for "
                          "--ids 7 12 (pass this explicitly to use a fixed path "
                          "instead).")
    args = ap.parse_args()

    if not args.reports and not args.ids:
        ap.error("at least one of --reports or --ids is required")

    # --ids entries resolve first (in order), then --reports globs, so
    # --labels (if given) lines up with this combined order.
    report_paths: list[str] = []
    default_labels: list[str] = []  # "" where no better-than-parsed default exists
    default_output_dirs: list[str] = []
    resolved_run_ids: list[str] = []  # only populated for --ids entries, used for --out-dir naming

    if args.ids:
        id_map = resolve_ids_to_reports(args.ids, args.log_file)
        for run_id, info in id_map.items():
            report_paths.append(info["report_path"])
            default_labels.append(info["scenario"])
            default_output_dirs.append(info["output_dir"])
            resolved_run_ids.append(run_id)

    if args.reports:
        explicit_paths = resolve_report_paths(args.reports)
        report_paths.extend(explicit_paths)
        default_labels.extend([""] * len(explicit_paths))
        # No run-log row for glob-resolved reports -- assume the report
        # lives directly under its scenario's output_dir (true for
        # run_pipeline.py's own reporter), overridable via --output-dirs.
        default_output_dirs.extend([os.path.dirname(p) for p in explicit_paths])

    if args.labels and len(args.labels) != len(report_paths):
        raise ValueError(f"--labels has {len(args.labels)} entries but --ids/--reports "
                          f"resolved to {len(report_paths)} files.")
    if args.scoring and len(args.scoring) != len(report_paths):
        raise ValueError(f"--scoring has {len(args.scoring)} entries but --ids/--reports "
                          f"resolved to {len(report_paths)} files.")
    if args.output_dirs and len(args.output_dirs) != len(report_paths):
        raise ValueError(f"--output-dirs has {len(args.output_dirs)} entries but --ids/--reports "
                          f"resolved to {len(report_paths)} files.")
    output_dirs = args.output_dirs or default_output_dirs

    if args.out_dir is None:
        # Auto-name the output folder from what's being compared, so runs
        # against different ids/reports don't overwrite each other's
        # comparison_output/ by default. --ids tokens are used as typed
        # (e.g. "7", "12", "field_survey_2d:7"); any --reports-only entries
        # fall back to their default label / filename. (Offset by the
        # number of *resolved* ids, not raw tokens, in case two tokens
        # ever resolve to the same run_id.)
        n_id_entries = len(resolved_run_ids)
        report_only_names = [
            name or os.path.splitext(os.path.basename(p))[0]
            for name, p in zip(default_labels[n_id_entries:], report_paths[n_id_entries:])
        ]
        args.out_dir = default_out_dir_from_tokens(
            id_tokens=list(args.ids) if args.ids else [],
            report_only_names=report_only_names,
        )
        print(f"[compare_scenarios] --out-dir not given, using: {args.out_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    parsed = OrderedDict()
    for i, path in enumerate(report_paths):
        rec = parse_report(path)
        if args.labels:
            label = args.labels[i]
        else:
            label = (default_labels[i]
                     or rec["scenario_name"]
                     or os.path.splitext(os.path.basename(path))[0])
        if label in parsed:
            label = f"{label} ({i})"
        parsed[label] = rec

    print(f"[compare_scenarios] Parsed {len(parsed)} scenario(s): {list(parsed.keys())}")

    # --- comparison table ---------------------------------------------------
    df = build_comparison_table(parsed)
    csv_path = os.path.join(args.out_dir, "comparison_table.csv")
    md_path = os.path.join(args.out_dir, "comparison_table.md")
    df.to_csv(csv_path)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Scenario comparison\n\n")
        f.write(df.to_markdown())
        f.write("\n")
    print(f"[compare_scenarios] Wrote {csv_path}")
    print(f"[compare_scenarios] Wrote {md_path}")

    # --- plots ---------------------------------------------------------------
    made = []
    made.append(plot_rmse_and_cv(parsed, args.out_dir))
    made.append(plot_water_content_error(parsed, args.out_dir))
    made.append(plot_ensemble_and_probes(parsed, args.out_dir))
    made.append(plot_eca_rmse_by_epoch(parsed, args.out_dir))
    made.append(plot_step_timing(parsed, args.out_dir))

    if args.scoring:
        scoring_paths = OrderedDict(zip(parsed.keys(), args.scoring))
        made.append(plot_truth_scoring(scoring_paths, args.out_dir))

    if args.position:
        # parsed was built in the same order as report_paths/output_dirs
        # (one dict insertion per loop iteration, no reordering), so
        # zipping them here lines labels up with the right output_dir.
        made.extend(run_truth_comparison(
            labels=list(parsed.keys()), output_dirs=output_dirs,
            realization=args.realization, time_index=args.time_index,
            position=tuple(args.position), transect_y=args.transect_y,
            transect_tol=args.transect_tol, out_dir=args.out_dir,
        ))

    for p in made:
        if p:
            print(f"[compare_scenarios] Wrote {p}")

    print("\n[compare_scenarios] Done.")
    print(df.to_string())


if __name__ == "__main__":
    main()
