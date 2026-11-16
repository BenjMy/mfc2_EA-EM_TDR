#!/usr/bin/env python3
"""
hyp_common.py -- shared engine of the hypothesis scripts (hyp1_*, hyp2_*, hyp3_*).

Every hypothesis script does the same four things; only step 1 differs:

1. GENERATE a few scenarios from a base scenario of ``ensemble_scenarios_real.json``
   (one parent = copy of the base, one child per variant that ``extends`` it).
   This is the only part a hypothesis script writes itself.
2. WRITE them to ``<out-dir>/<tag>_scenarios.json`` (the repo's JSON is never edited)
   and check that every one loads, so a typo fails now and not after hours of Step 1.
3. PRINT the plan (and the exact ``run_one_scenario.sh`` commands); ``--dry-run`` stops here.
4. RUN ``run_one_scenario.sh`` once per scenario and collect the rows of
   ``mfc2_outputs/run_log.csv`` into ``<out-dir>/<tag>_summary.csv``.

To add a hypothesis: copy ``hyp3_localized_bandwidth.py`` (the smallest one), change
``QUESTION`` and the scenario generator, and register it in ``run_all_hypotheses.py``.

Nothing here needs the field data or emagpy except the run itself.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import time

import scenario_loader as sl

BASE_CONFIG = "ensemble_scenarios_real.json"
BASE_SCENARIO = "real_single_base"
OUT_ROOT = os.path.join("mfc2_outputs", "hypotheses")
RUN_LOG = os.path.join("mfc2_outputs", "run_log.csv")

FAST_EPOCHS = 2
FAST_SUBSAMPLE = {"mode": "probes", "radius_m": 5.0}

# --stage: the hypotheses are about the CALIBRATION (RMSE at the TDR probes), which
# the 'calibrate' stage computes on the probe neighbourhoods only. 'none' = full run.
STAGES = ("calibrate", "none")

COMMON_COLUMNS = ["scenario", "status", "final_rmse", "cv_mean", "ensemble_size",
                  "runtime_s", "quality_level", "output_dir", "error"]


# --------------------------------------------------------------------------
# Pure helpers (no emagpy, no data needed -> unit-tested)
# --------------------------------------------------------------------------
def resolve_base(config_path: str, base: str) -> dict:
    """The raw dict of scenario ``base`` with ``extends`` already resolved."""
    with open(config_path, encoding="utf-8") as f:
        resolved = sl._resolve_extends(
            {k: v for k, v in json.load(f).items() if k not in sl._SKIP_KEYS})
    if base not in resolved:
        raise KeyError(f"base scenario {base!r} not in {config_path} "
                       f"(known: {sorted(resolved)})")
    return resolved[base]


def make_parent(base_raw: dict, *, cache_tag: str, out_dir: str, description: str,
                fast: bool = False) -> dict:
    """Copy of the base scenario that the children extend.

    ``em_subsample`` is dropped (a probe-only thinning would make later --stage map
    runs see only the probe neighbourhoods) unless ``fast``, the smoke test.
    """
    parent = copy.deepcopy(base_raw)
    parent["description"] = description
    parent["output_dir"] = os.path.join(out_dir, "parent")
    parent["cache_tag"] = cache_tag
    parent.pop("em_subsample", None)
    if fast:
        parent["em_subsample"] = dict(FAST_SUBSAMPLE)
    return parent


def pipeline_command(name: str, config_path: str, mode: str, fast: bool,
                     extra: list[str], stage: str = "calibrate") -> list[str]:
    cmd = ["./run_one_scenario.sh", "-s", name, "-c", config_path, "--mode", mode]
    if stage != "none":
        cmd += ["--stage", stage]
    if fast:
        cmd += ["--quick", "--quick-epochs", str(FAST_EPOCHS)]
    return cmd + list(extra)


def read_run_log(path: str = RUN_LOG) -> list[dict]:
    from run_log import RunLog
    return RunLog(path).rows()


def write_summary(rows: list[dict], path: str, extra_columns: list[str] | None = None) -> None:
    columns = list(extra_columns or []) + [c for c in COMMON_COLUMNS
                                           if c not in (extra_columns or [])]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def add_common_args(ap: argparse.ArgumentParser, *, default_out_dir: str) -> None:
    ap.add_argument("--config", default=BASE_CONFIG, help="real scenarios file (parent source)")
    ap.add_argument("--base", default=BASE_SCENARIO,
                    help="scenario copied as parent; it fixes the LAYERS of the run "
                         "(e.g. real_L3_base for 3 layers)")
    ap.add_argument("--mode", default="2d", choices=["1d", "2d"])
    ap.add_argument("--stage", default="calibrate", choices=STAGES,
                    help="calibrate (default): probe neighbourhoods only, Steps 0-5; "
                         "none: the full run")
    ap.add_argument("--out-dir", default=default_out_dir)
    ap.add_argument("--fast", action="store_true",
                    help=f"smoke test: --quick, {FAST_EPOCHS} epochs, probe points only")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the config and print the plan, run nothing")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue with the next scenario after a failure")


def split_extra(argv: list[str]) -> tuple[list[str], list[str]]:
    """Arguments after a bare ``--`` are forwarded to run_one_scenario.sh."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


# --------------------------------------------------------------------------
# Steps 2-4: write, check, print, run
# --------------------------------------------------------------------------
def execute(*, tag: str, cfg: dict, names: list[str], args: argparse.Namespace,
            extra: list[str], plan_lines: list[str],
            row_info: dict[str, dict] | None = None,
            extra_columns: list[str] | None = None) -> int:
    """Write ``cfg``, check it loads, print the plan, then run each scenario in ``names``.

    ``row_info[name]`` adds hypothesis-specific columns (e.g. design, k_free) to the
    summary row of that scenario. Returns a shell-style exit code (0 = all ran).
    """
    os.makedirs(args.out_dir, exist_ok=True)
    cfg_path = os.path.join(args.out_dir, f"{tag}_scenarios.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    for n in names:                       # fail now, not after hours of Step 1
        sl.load_scenario(n, config_path=cfg_path)

    print(f"[{tag}] config  : {cfg_path}  ({len(names)} scenarios, base '{args.base}', "
          f"stage '{args.stage}')")
    for line in plan_lines:
        print(f"[{tag}] {line}")
    if args.fast:
        print(f"[{tag}] >>> FAST smoke test: --quick --quick-epochs {FAST_EPOCHS}, "
              f"em_subsample={FAST_SUBSAMPLE}")
    for n in names:
        print("  $", " ".join(pipeline_command(n, cfg_path, args.mode, args.fast, extra, args.stage)))
    if args.dry_run:
        return 0

    rows, failed = [], []
    summary_path = os.path.join(args.out_dir, f"{tag}_summary.csv")
    for n in names:
        before = len(read_run_log())
        t0 = time.time()
        rc = subprocess.call(pipeline_command(n, cfg_path, args.mode, args.fast, extra, args.stage))
        new = [r for r in read_run_log()[before:] if r.get("scenario", "").startswith(n)]
        row = dict(new[-1]) if new else {"status": f"no run-log row (exit {rc})"}
        row.update({"scenario": n, **((row_info or {}).get(n, {}))})
        row.setdefault("runtime_s", round(time.time() - t0))
        rows.append(row)
        write_summary(rows, summary_path, extra_columns)   # after every run
        if rc != 0:
            failed.append(n)
            if not args.keep_going:
                print(f"!!! {n} failed (exit {rc}); stopping (use --keep-going to continue)")
                break

    print(f"\n[{tag}] summary: {summary_path}")
    for r in rows:
        print(f"  {r['scenario']:34s} {str(r.get('status', '')):10s} final_rmse={r.get('final_rmse', '')}")
    return 1 if failed else 0
