#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
study_status.py -- what exists and what is missing to have a complete picture.

Compares, per study type, the scenarios EXPECTED by the study definitions (the
hyp* scripts, run_1d_surface_vs_full.sh, study_swc_design.py, the scenario JSONs)
with the runs in the run logs, and writes ``STUDY_STATUS.md``:

* per study: done / failed (no good run) / missing / extra (in the log, not in the matrix);
* caveats found in the run manifests: one realisation only (no repeats), WARN runs,
  parameters at bounds (``ws_start_warnings``), quick / subsampled runs (not comparable),
  several code versions mixed in one study, runs without RMSE/std (RSR), ratings
  ``no skill`` / ``unsatisfactory``;
* the exact commands to run what is missing.

    python study_status.py                      # shared log + SWC-study log -> STUDY_STATUS.md
    python study_status.py --log-file mfc2_outputs/run_log.csv --out status.md
    python study_status.py --study-type hyp1_layers_ws_freedom
    python study_status.py --swc-preset main    # expected matrix of the SWC study preset

Read-only: it never runs anything. The expected matrices live in ``expected_matrix()``;
add a family there when you add a study (and a rule in ``run_log.STUDY_TYPE_RULES``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import OrderedDict

from EA_EM_config import OUTPUT_ROOT
from run_log import RunLog, infer_study_type, read_run_manifest

SHARED_LOG = os.path.join(OUTPUT_ROOT, "run_log.csv")
SWC_LOG = os.path.join(OUTPUT_ROOT, "run_log_swc_study.csv")
FAILED = {"FAIL", "FAILED", "ERROR"}
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Expected matrices (pure; lazy imports keep this module light)
# --------------------------------------------------------------------------
def _real_json_names() -> list[str]:
    path = os.path.join(HERE, "ensemble_scenarios_real.json")
    try:
        with open(path, encoding="utf-8") as f:
            return [k for k in json.load(f) if not k.startswith("_")]
    except (OSError, ValueError):
        return []


def expected_matrix(swc_preset: str = "oat", npos: int = 5) -> "OrderedDict[str, dict]":
    """study_type -> {"scenarios": [...], "command": f(name) -> str, "note": str}."""
    one = lambda n: f"./run_one_scenario.sh -s {n}"
    one1d = lambda n: f"./run_one_scenario.sh -s {n} --mode 1d --num-swc-positions {npos}"
    out: "OrderedDict[str, dict]" = OrderedDict()
    out["surface_vs_full"] = dict(
        scenarios=[f"1d_{t}_free{k}" for k in (1, 2, 3, 4) for t in ("surface", "full")],
        command=one1d, note="run_1d_surface_vs_full.sh (uniform truth)")
    out["surface_vs_full_hetero"] = dict(
        scenarios=[f"1d_{t}_hetero_free{k}" for k in (1, 2, 3, 4) for t in ("surface", "full")],
        command=one1d, note="VARIANT=hetero ./run_1d_surface_vs_full.sh")
    try:
        import hyp1_layers_ws_freedom as h1
        designs = list(h1.DESIGNS)
        out["hyp1_layers_ws_freedom"] = dict(
            scenarios=[h1.scenario_name(d, k) for d in designs for k in (1, 2, 3, 4)],
            command=lambda n: "python hyp1_layers_ws_freedom.py --designs "
                              f"{n.split('_')[2]} --free-k {n.rsplit('free', 1)[1]}",
            note="python hyp1_layers_ws_freedom.py")
    except Exception:  # noqa: BLE001 - heavy import failed: study simply not listed
        pass
    try:
        import hyp2_train_val_split as h2
        out["hyp2_train_val_split"] = dict(
            scenarios=[h2.scenario_name(m) for m in h2.METHODS],
            command=lambda n: f"python hyp2_train_val_split.py --methods {n[len('real_tv_'):]}",
            note="python hyp2_train_val_split.py")
    except Exception:  # noqa: BLE001
        pass
    out["hyp3_localized_bandwidth"] = dict(
        scenarios=["real_bw_sweep"], command=lambda n: "python hyp3_localized_bandwidth.py",
        note="python hyp3_localized_bandwidth.py")
    real = _real_json_names()
    if real:
        out["real_scenarios"] = dict(scenarios=real, command=lambda n: one(n) + " --mode 2d",
                                     note="run_real_all_scenarios.sh (all scenarios of ensemble_scenarios_real.json)")
    try:
        import study_swc_design as sw
        out["swc_design_study"] = dict(
            scenarios=[d.name for d in sw.build_designs(swc_preset, sw.LEVELS)],
            command=lambda n: f"python study_swc_design.py --preset {swc_preset} --resume",
            note=f"python study_swc_design.py --preset {swc_preset}  (own log run_log_swc_study.csv)")
    except Exception:  # noqa: BLE001
        pass
    return out


# --------------------------------------------------------------------------
# Status (pure: rows in, dict out -> unit-tested)
# --------------------------------------------------------------------------
def _is_failed(row: dict) -> bool:
    return row.get("status", "").upper() in FAILED


def study_of(row: dict) -> str:
    return row.get("study_type") or infer_study_type(row.get("scenario"))


def classify(expected: list[str], rows: list[dict]) -> dict:
    """done / failed / missing / extra scenario names + the runs per scenario."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r.get("scenario", ""), []).append(r)
    done, failed, missing = [], [], []
    for name in expected:
        runs = by.get(name, [])
        if not runs:
            missing.append(name)
        elif all(_is_failed(r) for r in runs):
            failed.append(name)
        else:
            done.append(name)
    extra = sorted(n for n in by if n not in set(expected))
    return dict(done=done, failed=failed, missing=missing, extra=extra, runs=by)


def caveats(rows: list[dict], manifests: dict[str, dict | None], head: str | None) -> list[str]:
    """Human-readable caveats for the finished runs of one study."""
    out: list[str] = []
    good = [r for r in rows if not _is_failed(r)]
    if not good:
        return out
    n_by: dict[str, int] = {}
    for r in good:
        n_by[r["scenario"]] = n_by.get(r["scenario"], 0) + 1
    if all(v == 1 for v in n_by.values()):
        out.append("One run per scenario, one noise realisation: no repeats, so differences "
                   "smaller than the run-to-run spread cannot be interpreted.")
    warn = sorted({r["scenario"] for r in good if r.get("status", "").upper() == "WARN"})
    if warn:
        out.append(f"{len(warn)} scenario(s) finished with quality WARN: {', '.join(warn[:6])}"
                   + (" ..." if len(warn) > 6 else ""))
    no_rsr = sorted({r["scenario"] for r in good if not r.get("rsr")})
    if no_rsr:
        out.append(f"{len(no_rsr)} scenario(s) have no RSR (run before rmse_context existed): their RMSE "
                   "cannot be compared with the spread of the data. Re-run to get it.")
    bad = sorted({r["scenario"] for r in good if r.get("rmse_rating") in ("no skill", "unsatisfactory")})
    if bad:
        out.append(f"RMSE rating no skill / unsatisfactory (RMSE not clearly below the std of the SWC): "
                   f"{', '.join(bad[:6])}" + (" ..." if len(bad) > 6 else ""))
    commits, quick, sub, bounds, stage = set(), set(), set(), set(), set()
    for r in good:
        m = manifests.get(r["run_id"]) or {}
        if m.get("git_commit"):
            commits.add(str(m["git_commit"]))
        cfg = m.get("config") or {}
        if cfg.get("em_subsample") not in (None, "", "off", "none", False):
            sub.add(r["scenario"])
        if str(r.get("output_dir", "")).endswith("_quick") or "_quick" in str(r.get("output_dir", "")):
            quick.add(r["scenario"])
        if m.get("ws_start_warnings"):
            bounds.add(r["scenario"])
        if m.get("stage") and m["stage"] != "all":
            stage.add(r["scenario"])
    if len(commits) > 1:
        out.append(f"Runs come from {len(commits)} different code versions "
                   f"({', '.join(sorted(c[:8] for c in commits))}): results may not be comparable.")
    if any(c.endswith("+dirty") for c in commits):
        out.append("Some runs were made with uncommitted changes (+dirty): not reproducible from git.")
    if head and commits and not any(c.split("+")[0] == head for c in commits):
        out.append("No run was made with the current HEAD: re-run before drawing conclusions on the new code.")
    for label, names in (("quick preset (smoke tests, shrunken ensemble)", quick),
                         ("subsampled EM data (em_subsample)", sub),
                         ("WS start values outside the bounds / on a bound (ws_start_warnings)", bounds),
                         ("a partial stage (calibrate only): no maps", stage)):
        if names:
            out.append(f"{len(names)} scenario(s) use {label}: {', '.join(sorted(names)[:5])}"
                       + (" ..." if len(names) > 5 else ""))
    return out


def _head() -> str | None:
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_report(logs: list[str], matrix: "OrderedDict[str, dict]", only: list[str] | None = None,
                 head: str | None = None, read_manifests: bool = True) -> str:
    rows: list[dict] = []
    for p in logs:
        rows += [{**r, "_log": p} for r in RunLog(p).rows()]
    by_study: dict[str, list[dict]] = {}
    for r in rows:
        by_study.setdefault(study_of(r), []).append(r)
    studies = list(matrix) + [s for s in by_study if s not in matrix]
    if only:
        studies = [s for s in studies if s in only]
    L = ["# Study status", "",
         f"Logs: {', '.join(f'`{p}`' for p in logs)}  ({len(rows)} runs). "
         f"HEAD: `{(head or 'unknown')[:8]}`.", "",
         "| study | expected | done | failed | missing | extra |", "|---|---|---|---|---|---|"]
    details: list[str] = []
    for s in studies:
        exp = matrix.get(s, {}).get("scenarios", [])
        srows = by_study.get(s, [])
        c = classify(exp, srows)
        L.append(f"| {s} | {len(exp) or '-'} | {len(c['done'])} | {len(c['failed'])} | "
                 f"{len(c['missing'])} | {len(c['extra'])} |")
        details += ["", f"## {s}", ""]
        if s in matrix:
            details.append(f"Matrix: {matrix[s]['note']}")
        else:
            details.append("No expected matrix defined for this study type (add it in `expected_matrix()`).")
        if c["missing"] or c["failed"]:
            cmd = matrix[s]["command"]
            details += ["", "**To run** (missing, then failed):", "", "```bash"]
            seen = set()
            for n in c["missing"] + c["failed"]:
                line = cmd(n)
                if line not in seen:
                    seen.add(line)
                    details.append(line)
            details.append("```")
        if c["extra"]:
            details += ["", f"Extra scenarios in the log (not in the matrix): {', '.join(c['extra'][:10])}"
                        + (" ..." if len(c["extra"]) > 10 else "")]
        manifests = {}
        if read_manifests:
            for r in srows:
                manifests[r["run_id"]] = read_run_manifest(r.get("output_dir", "")) if r.get("output_dir") else None
        cav = caveats(srows, manifests, head)
        if cav:
            details += ["", "**Caveats**", ""] + [f"- {x}" for x in cav]
        elif srows:
            details += ["", "No caveats found."]
    return "\n".join(L + details) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    ap.add_argument("--log-file", nargs="+", default=None,
                    help=f"run logs (default: {SHARED_LOG} and {SWC_LOG} when they exist)")
    ap.add_argument("--out", default="STUDY_STATUS.md")
    ap.add_argument("--study-type", nargs="+")
    ap.add_argument("--swc-preset", default="oat", choices=["oat", "main", "full", "smoke"])
    ap.add_argument("--npos", type=int, default=5, help="SWC positions in the printed 1-D commands")
    a = ap.parse_args(argv)
    logs = a.log_file or [p for p in (SHARED_LOG, SWC_LOG) if os.path.isfile(p)]
    if not logs:
        print("[study_status] no run log found (pass --log-file)", file=sys.stderr)
        return 2
    text = build_report(logs, expected_matrix(a.swc_preset, a.npos), a.study_type, _head())
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"[study_status] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
