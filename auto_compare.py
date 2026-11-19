#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_compare.py -- run compare_scenarios.py on the LATEST run of each scenario,
looked up in the run-log CSV, so you never copy run ids by hand.

Usage
-----
    # swc_design_study (its own log: mfc2_outputs/run_log_swc_study.csv)
    python auto_compare.py --preset swc --list                 # what would be compared
    python auto_compare.py --preset swc --n 5 --dist spread --smc hetero --ws hetero --reg noreg
    python auto_compare.py --preset swc --k 1 2 3 4 --n 5 --dist spread --smc homo --ws homo --reg noreg

    # synthetic MFC2-like dataset (shared log mfc2_outputs/run_log.csv)
    python auto_compare.py --preset mirror --list
    python auto_compare.py --preset mirror

    # any scenarios (glob patterns on the scenario name), any log
    python auto_compare.py --match '1d_surface_*_free3' '1d_full_*_free3' --position 8 0
    python auto_compare.py --match 'real_study_*' --log-file mfc2_outputs/run_log.csv

Only the latest run of each scenario is used, runs whose status is a failure
are skipped. Everything not recognised here (--position, --transect-y,
--realization, --time-index, --scoring, --out-dir, --labels is built for you)
is forwarded to compare_scenarios.py unchanged.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys

from EA_EM_config import OUTPUT_ROOT
from run_log import RunLog

SHARED_LOG = os.path.join(OUTPUT_ROOT, "run_log.csv")
SWC_LOG = os.path.join(OUTPUT_ROOT, "run_log_swc_study.csv")
SWC_PREFIX = "swcstudy_"
FAILED = {"FAIL", "FAILED", "ERROR"}
MAX_RUNS = 12                       # more than this makes unreadable bar charts

PRESETS = {
    "swc": dict(log=SWC_LOG, match=[SWC_PREFIX + "*"], position=("8", "0"), tag="swc_design_study/auto"),
    # scenarios written by generate_synthetic_real_mirror.py and the hyp* scripts
    "mirror": dict(log=SHARED_LOG,
                   match=["mfc2_real_synthetic_mirror*", "real_study_*", "real_tv_*", "real_bw_sweep*"],
                   position=None, tag="mirror/auto"),
}


def latest_runs(log_path: str, patterns: list[str]) -> "dict[str, dict]":
    """scenario name -> latest non-failed row with an existing report."""
    out: dict[str, dict] = {}
    for r in RunLog(log_path).rows():                      # oldest first: later rows win
        name = r.get("scenario", "")
        if not any(fnmatch.fnmatch(name, p) for p in patterns):
            continue
        if r.get("status", "").upper() in FAILED or not os.path.isfile(r.get("report_path", "") or ""):
            continue                                       # failed run: keep the earlier good one, if any
        out[name] = r
    return out


def filter_swc(runs: "dict[str, dict]", args) -> "dict[str, dict]":
    """Apply --smc/--ws/--reg/--dist/--n/--k using study_swc_design.parse_name."""
    from study_swc_design import parse_name
    want = {"smc": args.smc, "ws": args.ws, "reg": args.reg, "dist": args.dist, "n": args.n, "k": args.k}
    out = {}
    for name, row in runs.items():
        d = parse_name(name)
        if d is None:
            continue
        if all(v is None or getattr(d, f) in v for f, v in want.items()):
            out[name] = row
    # order: n, then k, then the rest, so the legend reads naturally
    key = lambda kv: (lambda d: (d.smc, d.ws, d.reg, d.dist, d.n, d.k))(parse_name(kv[0]))
    return dict(sorted(out.items(), key=key))


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--preset", choices=list(PRESETS), help="swc | mirror: sensible log, patterns and defaults")
    ap.add_argument("--match", nargs="+", help="glob patterns on the scenario name (overrides the preset's)")
    ap.add_argument("--log-file", help="run-log CSV (default: the preset's, else the shared one)")
    ap.add_argument("--list", action="store_true", help="print the selected runs and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the compare_scenarios.py command, run nothing")
    ap.add_argument("--max-runs", type=int, default=MAX_RUNS, help="refuse to compare more runs (default %(default)s)")
    ap.add_argument("--no-position", action="store_true", help="skip the truth plots the swc preset enables")
    g = ap.add_argument_group("swc filters (only with --preset swc; several values allowed)")
    g.add_argument("--smc", nargs="+", choices=["homo", "hetero"])
    g.add_argument("--ws", nargs="+", choices=["homo", "hetero"])
    g.add_argument("--reg", nargs="+", choices=["noreg", "beta", "gamma", "both"])
    g.add_argument("--dist", nargs="+", choices=["spread", "clustered", "random"])
    g.add_argument("--n", nargs="+", type=int, help="number of SWC positions")
    g.add_argument("--k", nargs="+", type=int, help="number of free WS parameters")
    args, passthrough = ap.parse_known_args(argv)

    preset = PRESETS.get(args.preset or "", {})
    log = args.log_file or preset.get("log", SHARED_LOG)
    patterns = args.match or preset.get("match")
    if not patterns:
        ap.error("give --preset or --match")
    if not os.path.isfile(log):
        print(f"[auto_compare] run log not found: {log}  (run the scenarios first, or pass --log-file)",
              file=sys.stderr)
        return 2

    runs = latest_runs(log, patterns)
    if args.preset == "swc":
        runs = filter_swc(runs, args)
    if not runs:
        print(f"[auto_compare] no finished run matches {patterns} in {log}", file=sys.stderr)
        return 1

    prefix = SWC_PREFIX if args.preset == "swc" else ""
    labels = [n[len(prefix):] if prefix and n.startswith(prefix) else n for n in runs]
    print(f"[auto_compare] {len(runs)} run(s) from {log}:")
    for lab, (name, r) in zip(labels, runs.items()):
        print(f"   {r['run_id']:<55s} {r.get('status', ''):<8s} {lab}")
    if args.list:
        return 0
    if len(runs) < 2:
        print("[auto_compare] need at least 2 runs to compare.", file=sys.stderr)
        return 1
    if len(runs) > args.max_runs:
        print(f"[auto_compare] {len(runs)} runs is too many for one comparison (max {args.max_runs}). "
              f"Narrow it down (swc: --n/--k/--smc/--ws/--reg/--dist; else a tighter --match) "
              f"or raise --max-runs.", file=sys.stderr)
        return 1

    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_scenarios.py"),
           "--log-file", log, "--ids", *[r["run_id"] for r in runs.values()], "--labels", *labels]
    flags = " ".join(passthrough)
    if "--out-dir" not in passthrough:
        # one folder per selection, so successive comparisons do not overwrite each other
        sel = "_".join(f"{f}{'-'.join(map(str, v))}" for f in ("smc", "ws", "reg", "dist", "n", "k")
                       if (v := getattr(args, f)) is not None)
        tag = preset.get("tag", "auto") + (f"_{sel}" if sel else "")
        cmd += ["--out-dir", os.path.join("comparison_output", tag)]
    if preset.get("position") and "--position" not in flags and not args.no_position:
        cmd += ["--position", *preset["position"]]
    cmd += passthrough
    print("[auto_compare] " + " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
