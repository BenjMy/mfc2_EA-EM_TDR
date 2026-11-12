#!/usr/bin/env python3
"""
run_real_study.py -- layer-design x Waxman-Smits-freedom study on the REAL data.

Replaces the hand-written ``run_real_PAPER.sh`` for this question:
"how many layers (horizons) can the real TDR sensors support, and how many
Waxman-Smits (WS) parameters can be freed in the calibration?"

What it does
------------
1. Takes a base scenario of ``ensemble_scenarios_real.json`` (default
   ``real_single_base``: data paths, forward models, alphas, drop_coils ...).
2. For every layer DESIGN (a list of interface depths ``depths0``) it prints
   which horizons contain a real TDR depth (SoilNet: 0.15 and 0.30 m). A
   horizon without a sensor can never be calibrated (it falls back to the
   prior WS parameters), so you see this BEFORE spending hours on Step 1.
3. For every design it loops over K = number of FREE WS parameters
   (freed in the same order as the synthetic ``1d_*_free{K}`` study:
   sigma_s, m, sigma_w, a; the others stay at ``lab_p0``).
4. Writes the resulting scenarios to ``<out-dir>/study_scenarios.json``
   (every scenario ``extends`` the base, so only the differences are listed)
   and runs ``./run_pipeline.sh`` once per scenario.
5. Collects the rows of ``mfc2_outputs/run_log.csv`` into
   ``<out-dir>/study_summary.csv``.

Step 1 (the expensive ensemble inversion) does not depend on the WS freedom,
so all K of one design share ONE ``output_dir`` / ``cache_tag``: the cache is
computed by the first K and reused (fingerprint-checked) by the others.

Usage
-----
    python run_real_study.py --fast --dry-run     # show the plan, run nothing
    python run_real_study.py --fast               # quick smoke test of everything
    python run_real_study.py                      # full study (long)
    python run_real_study.py --designs L2 --free-k 1 2   # a subset
    python run_real_study.py --design L3b=0.2,0.5 --designs L3b   # own design
    python run_real_study.py --fast -- --em-footprint-diagnostic  # extra flags
                                                  # (after '--') go to run_pipeline.sh

``--fast`` = ``--quick --quick-epochs 2`` (one alpha, one forward model, two
epochs) + ``em_subsample = probes`` (only EM points within 5 m of a TDR probe).
It is a SMOKE TEST (the calibration stage works, maps do not): never a result.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time

import scenario_loader as sl

# --------------------------------------------------------------------------
# Study definition (edit here)
# --------------------------------------------------------------------------
BASE_CONFIG = "ensemble_scenarios_real.json"
BASE_SCENARIO = "real_single_base"
OUT_DIR = os.path.join("mfc2_outputs", "real_study")
RUN_LOG = os.path.join("mfc2_outputs", "run_log.csv")

# Depths [m] of the real SoilNet TDR sensors (data_real.load_tdr_dataframe
# maps the '15' and '30' SWC columns to these two depths).
TDR_DEPTHS = (0.15, 0.30)

# Layer designs: name -> interface depths (n_layers = len + 1).
DESIGNS = {
    "L2": [0.225],             # 15 cm | 30 cm in different horizons (= real_single_*)
    "L3": [0.225, 0.6],        # + a deep horizon WITHOUT sensor
    "L4": [0.3, 0.6, 1.0],     # = real_multi_*: 30 cm falls in H1, H2-H3 without sensor
}

# Order in which WS parameters are freed (same as the synthetic free{K} study).
FREE_ORDER = ("sigma_s", "m", "sigma_w", "a")

FAST_EPOCHS = 2
FAST_SUBSAMPLE = {"mode": "probes", "radius_m": 5.0}


# --------------------------------------------------------------------------
# Pure helpers (no emagpy, no data needed -> unit-tested)
# --------------------------------------------------------------------------
def parse_design(text: str) -> tuple[str, list[float]]:
    """'L3b=0.2,0.5' -> ('L3b', [0.2, 0.5]); depths must be increasing."""
    name, sep, depths = text.partition("=")
    if not sep or not name or not depths:
        raise ValueError(f"design must look like NAME=d1,d2,... got {text!r}")
    values = [float(v) for v in depths.split(",")]
    if any(b <= a for a, b in zip(values, values[1:])) or values[0] <= 0:
        raise ValueError(f"design {name!r}: depths must be positive and increasing, got {values}")
    return name, values


def horizon_coverage(depths0: list[float], tdr_depths=TDR_DEPTHS) -> dict[int, int]:
    """{horizon index: number of TDR depths inside it} (same routing as the pipeline)."""
    counts = {i: 0 for i in range(len(depths0) + 1)}
    for d in tdr_depths:
        counts[sl._horizon_index_from_depth(d, depths0)] += 1
    return counts


def coverage_text(depths0: list[float], tdr_depths=TDR_DEPTHS) -> str:
    cov = horizon_coverage(depths0, tdr_depths)
    bounds = [0.0, *depths0, float("inf")]
    parts = []
    for i, n in cov.items():
        top = "inf" if bounds[i + 1] == float("inf") else f"{bounds[i + 1]:g}"
        parts.append(f"H{i} [{bounds[i]:g}-{top}] m: {n} TDR depth(s)" + ("" if n else "  <-- NO SENSOR"))
    return " | ".join(parts)


def ws_freedom(k: int, base_p0: list[float]) -> dict:
    """Scenario fields for K free WS parameters (K = 0..4)."""
    if not 0 <= k <= len(FREE_ORDER):
        raise ValueError(f"K must be between 0 and {len(FREE_ORDER)}, got {k}")
    free = set(FREE_ORDER[:k])
    fields = {"ws_fixed_params": {p: (p not in free) for p in ("a", "m", "sigma_w", "sigma_s")}}
    if "a" in free:
        # Same fix as the synthetic free4 scenarios: 'a' must not start ON its
        # lower bound (0.5), otherwise L-BFGS-B clips every member to it.
        p0 = list(base_p0)
        p0[0] = 1.0
        fields["lab_p0"] = p0
        fields["ws_bounds"] = {"a": [0.2, 5.0]}
    return fields


def scenario_name(design: str, k: int) -> str:
    return f"real_study_{design}_free{k}"


def build_study_config(base_raw: dict, designs: dict[str, list[float]], ks: list[int],
                       tdr_depths=TDR_DEPTHS, fast: bool = False) -> dict:
    """Study scenarios file content: one parent + one child per (design, K)."""
    parent = copy.deepcopy(base_raw)
    parent["description"] = "Parent of the real-data layer x WS-freedom study (see run_real_study.py)."
    parent["output_dir"] = os.path.join(OUT_DIR, "parent")
    parent["cache_tag"] = "real_study_parent"
    parent.pop("em_subsample", None)
    cond_hi = parent["cond_ranges"][0]
    base_p0 = parent.get("lab_p0") or [0.5, 3.0, 500.0, 100.0]
    if fast:
        parent["em_subsample"] = dict(FAST_SUBSAMPLE)

    cfg = {"_comment": [
        "GENERATED by run_real_study.py -- do not edit, re-run the script.",
        f"Parent '{parent_name()}' = copy of '{BASE_SCENARIO}'; children differ only by "
        "depths0 / cond_ranges (layer design) and the WS freedom (ws_fixed_params, lab_p0, ws_bounds).",
    ], parent_name(): parent}
    for dname, depths0 in designs.items():
        for k in ks:
            cov = horizon_coverage(depths0, tdr_depths)
            child = {
                "extends": parent_name(),
                "description": (f"Real-data study, design {dname} depths0={depths0}, K={k} free WS "
                                f"parameter(s) ({', '.join(FREE_ORDER[:k]) or 'none'}). "
                                f"Horizons without TDR: {[h for h, n in cov.items() if not n] or 'none'}."),
                "depths0": list(depths0),
                "cond_ranges": [list(cond_hi) for _ in range(len(depths0) + 1)],
                # one cache per DESIGN: Step 1 does not depend on the WS freedom
                "output_dir": os.path.join(OUT_DIR, dname),
                "cache_tag": f"real_study_{dname}",
                **ws_freedom(k, base_p0),
            }
            cfg[scenario_name(dname, k)] = child
    return cfg


def parent_name() -> str:
    return "real_study_parent"


def pipeline_command(name: str, config_path: str, mode: str, fast: bool, extra: list[str]) -> list[str]:
    cmd = ["./run_pipeline.sh", "-s", name, "-c", config_path, "--mode", mode]
    if fast:
        cmd += ["--quick", "--quick-epochs", str(FAST_EPOCHS)]
    return cmd + list(extra)


def read_run_log(path: str = RUN_LOG) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


SUMMARY_COLUMNS = ["scenario", "design", "k_free", "status", "final_rmse", "cv_mean",
                   "ensemble_size", "runtime_s", "quality_level", "output_dir", "error"]


def write_summary(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in SUMMARY_COLUMNS})


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    global OUT_DIR
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", nargs="+", help=f"designs to run (default: all of {list(DESIGNS)})")
    ap.add_argument("--design", action="append", default=[], metavar="NAME=d1,d2",
                    help="add or override a layer design (interface depths in m)")
    ap.add_argument("--free-k", nargs="+", type=int, default=None,
                    help="numbers of free WS parameters (0-4, order sigma_s, m, sigma_w, a); "
                         "default 1 2 3 4, or 1 4 (extremes only) with --fast")
    ap.add_argument("--tdr-depths", nargs="+", type=float, default=list(TDR_DEPTHS),
                    help="TDR sensor depths [m] used for the coverage report")
    ap.add_argument("--config", default=BASE_CONFIG, help="real scenarios file (parent source)")
    ap.add_argument("--base", default=BASE_SCENARIO, help="scenario copied as parent")
    ap.add_argument("--mode", default="2d", choices=["1d", "2d"])
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--fast", action="store_true", help=f"smoke test: --quick, {FAST_EPOCHS} epochs, probe points only")
    ap.add_argument("--dry-run", action="store_true", help="write the config and print the plan, run nothing")
    ap.add_argument("--keep-going", action="store_true", help="continue with the next scenario after a failure")
    args = ap.parse_args(argv)

    designs = dict(DESIGNS)
    for text in args.design:
        n, d = parse_design(text)
        designs[n] = d
    chosen = args.designs or list(designs)
    unknown = [d for d in chosen if d not in designs]
    if unknown:
        ap.error(f"unknown design(s) {unknown}; known: {list(designs)}")
    designs = {d: designs[d] for d in chosen}

    # base scenario with `extends` already resolved
    with open(args.config, encoding="utf-8") as f:
        resolved = sl._resolve_extends(
            {k: v for k, v in json.load(f).items() if k not in sl._SKIP_KEYS})
    if args.base not in resolved:
        ap.error(f"base scenario {args.base!r} not in {args.config}")

    OUT_DIR = args.out_dir
    ks = args.free_k or ([1, 4] if args.fast else [1, 2, 3, 4])
    cfg = build_study_config(resolved[args.base], designs, ks, tuple(args.tdr_depths), args.fast)
    cfg_path = os.path.join(OUT_DIR, "study_scenarios.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    names = [n for n in cfg if n.startswith("real_study_") and n != parent_name()]
    for n in names:                       # fail now, not after hours of Step 1
        sl.load_scenario(n, config_path=cfg_path)

    print(f"Study config : {cfg_path}  ({len(names)} scenarios, base '{args.base}')")
    print(f"TDR depths   : {args.tdr_depths} m")
    for dname, d in designs.items():
        print(f"  {dname:4s} depths0={d}:  {coverage_text(d, tuple(args.tdr_depths))}")
    if args.fast:
        print(f">>> FAST smoke test: --quick --quick-epochs {FAST_EPOCHS}, em_subsample={FAST_SUBSAMPLE}")
    for n in names:
        print("  $", " ".join(pipeline_command(n, cfg_path, args.mode, args.fast, extra)))
    if args.dry_run:
        return 0

    rows, failed = [], []
    for n in names:
        before = len(read_run_log())
        t0 = time.time()
        rc = subprocess.call(pipeline_command(n, cfg_path, args.mode, args.fast, extra))
        new = [r for r in read_run_log()[before:] if r.get("scenario", "").startswith(n)]
        row = dict(new[-1]) if new else {"status": f"no run-log row (exit {rc})"}
        design, k = n[len("real_study_"):].rsplit("_free", 1)
        row.update(scenario=n, design=design, k_free=k)
        row.setdefault("runtime_s", round(time.time() - t0))
        rows.append(row)
        write_summary(rows, os.path.join(OUT_DIR, "study_summary.csv"))   # after every run
        if rc != 0:
            failed.append(n)
            if not args.keep_going:
                print(f"!!! {n} failed (exit {rc}); stopping (use --keep-going to continue)")
                break

    print(f"\nSummary: {os.path.join(OUT_DIR, 'study_summary.csv')}")
    for r in rows:
        print(f"  {r['scenario']:28s} {r.get('status', ''):10s} final_rmse={r.get('final_rmse', '')}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
