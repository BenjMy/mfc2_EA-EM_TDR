#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_run_log.py -- read the run-log CSV and generate ALL the summary plots, one folder
per study type, so a study hypothesis can be checked without copying run ids by hand.

The log has a ``study_type`` column (hyp1_layers_ws_freedom, hyp2_train_val_split,
surface_vs_full, swc_design_study, ...; see ``run_log.STUDY_TYPE_RULES``). Old logs
without the column are handled on the fly (the type is inferred from the scenario
name); ``--upgrade`` rewrites the file once with the new columns.

Usage
-----
    python plot_run_log.py --list                       # runs per study type, nothing written
    python plot_run_log.py                              # all study types of the shared log
    python plot_run_log.py --study-type hyp1_layers_ws_freedom surface_vs_full
    python plot_run_log.py --log-file mfc2_outputs/run_log_swc_study.csv   # (see also plot_swc_study_log.py)
    python plot_run_log.py --compare                    # + compare_scenarios.py on each study
    python plot_run_log.py --upgrade                    # add the new columns to the CSV first

Output: ``<out-dir>/<study_type>/`` (default ``comparison_output/run_log_plots``), with

    rmse_by_scenario.png   headline RMSE in % vol per scenario (latest finished run), bars
                           coloured by the RMSE rating when the run logged RMSE/std (RSR)
    rmse_vs_rsr.png        RMSE [% vol] vs RSR = RMSE/std(observed SWC): is the error large
                           compared with the natural spread of the data? (RSR 0.5 / 0.7 / 1 lines)
    rmse_vs_k.png          RMSE vs number of free Waxman-Smits parameters K (names ``..free{K}``),
                           one line per scenario family (e.g. surface vs full)
    rmse_history.png       RMSE of every run over time, one line per scenario (did a change help?)
    runtime_vs_rmse.png    cost / accuracy trade-off
    runs.csv               the table behind the plots

Only the latest finished run of each scenario is used (failed runs are skipped), except
in rmse_history.png.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from EA_EM_config import OUTPUT_ROOT  # noqa: E402
from rmse_context import RSR_BANDS, parse_rmse, rating_from_rsr  # noqa: E402
from run_log import RunLog, infer_study_type  # noqa: E402

SHARED_LOG = os.path.join(OUTPUT_ROOT, "run_log.csv")
DEFAULT_OUT = os.path.join("comparison_output", "run_log_plots")
FAILED = {"FAIL", "FAILED", "ERROR"}
MAX_COMPARE = 12                                  # compare_scenarios gets unreadable above this
RATING_COLORS = {"very good": "#2e7d32", "good": "#7cb342", "satisfactory": "#f9a825",
                 "unsatisfactory": "#ef6c00", "no skill": "#c62828", "std too small": "#9e9e9e",
                 "": "#607d8b", "n/a": "#607d8b"}


# --------------------------------------------------------------------------
# Reading the log (pure helpers, unit-tested)
# --------------------------------------------------------------------------
def _num(value) -> float:
    return parse_rmse(value)


def load_rows(log_path: str) -> list[dict]:
    """Rows of the log with numeric columns parsed and ``study_type`` / ``rmse_pct`` filled in
    when the log predates them (the file itself is not modified)."""
    out = []
    for r in RunLog(log_path).rows():
        d = dict(r)
        d["study_type"] = r.get("study_type") or infer_study_type(r.get("scenario"))
        rmse = _num(r.get("final_rmse"))
        pct = _num(r.get("rmse_pct"))
        d["rmse"] = rmse
        d["rmse_pct"] = pct if np.isfinite(pct) else rmse * 100.0
        d["rsr"] = _num(r.get("rsr"))
        d["swc_obs_std"] = _num(r.get("swc_obs_std"))
        d["runtime"] = _num(r.get("runtime_s"))
        d["rating"] = r.get("rmse_rating") or (
            rating_from_rsr(d["rsr"], d["swc_obs_std"]) if np.isfinite(d["rsr"]) else "")
        d["status_ok"] = (r.get("status", "").upper() not in FAILED) and np.isfinite(rmse)
        out.append(d)
    return out


def latest_per_scenario(rows: list[dict]) -> dict[str, dict]:
    """scenario -> latest finished run (rows are oldest first, later rows win)."""
    out: dict[str, dict] = {}
    for r in rows:
        if r["status_ok"]:
            out[r["scenario"]] = r
    return out


def group_by_study(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["study_type"], []).append(r)
    return dict(sorted(groups.items()))


def split_free_k(scenario: str) -> tuple[str, int | None]:
    """``'1d_surface_hetero_free3'`` -> (``'1d_surface_hetero'``, 3); (name, None) without ``free{K}``."""
    m = re.match(r"^(.*?)_?free(\d+)(?:_(.*))?$", scenario)
    if not m:
        return scenario, None
    family = m.group(1) + (("_" + m.group(3)) if m.group(3) else "")
    return family, int(m.group(2))


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rmse_by_scenario(latest: dict[str, dict], out_dir: str, title: str) -> str | None:
    items = sorted(latest.values(), key=lambda r: r["rmse_pct"])
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(items) + 1.2)))
    y = np.arange(len(items))
    ax.barh(y, [r["rmse_pct"] for r in items], color=[RATING_COLORS.get(r["rating"], "#607d8b") for r in items])
    ax.set_yticks(y)
    ax.set_yticklabels([r["scenario"] for r in items], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE [% vol]  (= m³/m³ × 100)")
    ax.set_title(f"{title}: RMSE by scenario (latest run)")
    for yi, r in zip(y, items):
        txt = f"{r['rmse_pct']:.2f}" + (f"  (RSR {r['rsr']:.2f})" if np.isfinite(r["rsr"]) else "")
        ax.text(r["rmse_pct"], yi, " " + txt, va="center", fontsize=6)
    shown = sorted({r["rating"] for r in items if r["rating"]})
    if shown:
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=RATING_COLORS.get(k, "#607d8b")) for k in shown],
                  labels=[f"{k}" for k in shown], title="RMSE/std rating", fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, out_dir, "rmse_by_scenario.png")


def plot_rmse_vs_rsr(latest: dict[str, dict], out_dir: str, title: str) -> str | None:
    items = [r for r in latest.values() if np.isfinite(r["rsr"])]
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter([r["rmse_pct"] for r in items], [r["rsr"] for r in items],
               c=[RATING_COLORS.get(r["rating"], "#607d8b") for r in items], s=28)
    for r in items:
        ax.annotate(r["scenario"], (r["rmse_pct"], r["rsr"]), fontsize=5, alpha=0.8)
    for upper, label in RSR_BANDS:
        ax.axhline(upper, color="grey", lw=0.6, ls=":")
        ax.text(ax.get_xlim()[1], upper, f" {label}", va="center", fontsize=6, color="grey")
    ax.axhline(1.0, color="#c62828", lw=0.8, ls="--")
    ax.set_xlabel("RMSE [% vol]")
    ax.set_ylabel("RSR = RMSE / std(observed SWC)")
    ax.set_title(f"{title}: error relative to the spread of the data")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "rmse_vs_rsr.png")


def plot_rmse_vs_k(latest: dict[str, dict], out_dir: str, title: str) -> str | None:
    fam: dict[str, list[tuple[int, float]]] = {}
    for name, r in latest.items():
        f, k = split_free_k(name)
        if k is not None:
            fam.setdefault(f, []).append((k, r["rmse_pct"]))
    fam = {f: sorted(v) for f, v in fam.items() if len(v) >= 2}
    if not fam:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for f, v in sorted(fam.items()):
        ax.plot([k for k, _ in v], [e for _, e in v], "o-", label=f, lw=1.2)
    ax.set_xlabel("free Waxman-Smits parameters K")
    ax.set_ylabel("RMSE [% vol]")
    ax.set_xticks(sorted({k for v in fam.values() for k, _ in v}))
    ax.set_title(f"{title}: RMSE vs K")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "rmse_vs_k.png")


def plot_rmse_history(rows: list[dict], out_dir: str, title: str) -> str | None:
    by: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        if r["status_ok"]:
            by.setdefault(r["scenario"], []).append((r.get("timestamp", ""), r["rmse_pct"]))
    by = {k: v for k, v in by.items() if len(v) >= 2}
    if not by:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, v in sorted(by.items()):
        ax.plot(range(1, len(v) + 1), [e for _, e in v], "o-", lw=1, ms=3, label=name)
    ax.set_xlabel("run number of the scenario (oldest = 1)")
    ax.set_ylabel("RMSE [% vol]")
    ax.set_title(f"{title}: RMSE of repeated runs")
    ax.legend(fontsize=5)
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "rmse_history.png")


def plot_runtime_vs_rmse(latest: dict[str, dict], out_dir: str, title: str) -> str | None:
    items = [r for r in latest.values() if np.isfinite(r["runtime"])]
    if len(items) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter([r["runtime"] / 60 for r in items], [r["rmse_pct"] for r in items], s=25)
    for r in items:
        ax.annotate(r["scenario"], (r["runtime"] / 60, r["rmse_pct"]), fontsize=5, alpha=0.8)
    ax.set_xlabel("runtime [min]")
    ax.set_ylabel("RMSE [% vol]")
    ax.set_title(f"{title}: cost vs accuracy")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "runtime_vs_rmse.png")


def write_table(latest: dict[str, dict], out_dir: str) -> str:
    import csv
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "runs.csv")
    cols = ["scenario", "study_type", "run_id", "status", "rmse_pct", "swc_obs_std", "rsr", "rating", "runtime"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(latest.values(), key=lambda r: r["scenario"]):
            w.writerow([r.get("scenario"), r["study_type"], r.get("run_id"), r.get("status"),
                        f"{r['rmse_pct']:.3f}", "" if not np.isfinite(r["swc_obs_std"]) else f"{r['swc_obs_std']:.5f}",
                        "" if not np.isfinite(r["rsr"]) else f"{r['rsr']:.4f}", r["rating"],
                        "" if not np.isfinite(r["runtime"]) else f"{r['runtime']:.1f}"])
    return path


def plot_study(study: str, rows: list[dict], out_root: str) -> list[str]:
    """All plots of one study type; returns the files written."""
    out_dir = os.path.join(out_root, study)
    latest = latest_per_scenario(rows)
    if not latest:
        return []
    files = [write_table(latest, out_dir)]
    for fn, arg in ((plot_rmse_by_scenario, latest), (plot_rmse_vs_rsr, latest),
                    (plot_rmse_vs_k, latest), (plot_rmse_history, rows), (plot_runtime_vs_rmse, latest)):
        p = fn(arg, out_dir, study)
        if p:
            files.append(p)
    return files


def compare_study(study: str, rows: list[dict], log_path: str, out_root: str, dry_run: bool = False) -> int:
    """Run compare_scenarios.py on the latest run of each scenario of the study."""
    latest = latest_per_scenario(rows)
    if len(latest) < 2:
        print(f"[plot_run_log] {study}: need >= 2 finished scenarios to compare ({len(latest)}), skipped")
        return 0
    if len(latest) > MAX_COMPARE:
        print(f"[plot_run_log] {study}: {len(latest)} scenarios > {MAX_COMPARE}, too many for one compare_scenarios "
              f"figure; use auto_compare.py with filters. Skipped.")
        return 0
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_scenarios.py"),
           "--log-file", log_path, "--ids", *[r["run_id"] for r in latest.values()],
           "--labels", *latest.keys(), "--out-dir", os.path.join(out_root, study, "compare_scenarios")]
    print("[plot_run_log] " + " ".join(cmd))
    return 0 if dry_run else subprocess.call(cmd)


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-file", default=SHARED_LOG, help="run-log CSV (default: %(default)s)")
    ap.add_argument("--study-type", nargs="+", help="only these study types (default: all in the log)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--list", action="store_true", help="print the runs per study type and exit")
    ap.add_argument("--upgrade", action="store_true", help="add the new columns to the CSV first (backup .bak)")
    ap.add_argument("--compare", action="store_true", help="also run compare_scenarios.py per study type")
    ap.add_argument("--dry-run", action="store_true", help="with --compare: print the commands only")
    a = ap.parse_args(argv)

    if not os.path.isfile(a.log_file):
        print(f"[plot_run_log] run log not found: {a.log_file}", file=sys.stderr)
        return 2
    if a.upgrade:
        RunLog(a.log_file).ensure_schema()
    groups = group_by_study(load_rows(a.log_file))
    if a.study_type:
        unknown = [s for s in a.study_type if s not in groups]
        if unknown:
            print(f"[plot_run_log] no runs of study type {unknown}; available: {list(groups)}", file=sys.stderr)
            return 1
        groups = {s: groups[s] for s in a.study_type}
    print(f"[plot_run_log] {a.log_file}: {len(groups)} study type(s)")
    for s, rows in groups.items():
        print(f"   {s:<28s} {len(rows):4d} runs, {len(latest_per_scenario(rows)):3d} finished scenarios")
    if a.list:
        return 0
    for s, rows in groups.items():
        for f in plot_study(s, rows, a.out_dir):
            print(f"   wrote {f}")
        if a.compare:
            compare_study(s, rows, a.log_file, a.out_dir, a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
