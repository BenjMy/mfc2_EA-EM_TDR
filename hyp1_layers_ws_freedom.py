#!/usr/bin/env python3
"""
hyp1_layers_ws_freedom.py -- HYPOTHESIS 1 on the REAL data.

QUESTION: how many layers (horizons) can the real TDR sensors support, and how many
Waxman-Smits (WS) parameters can be freed in the calibration?

What it does (the shared steps are in hyp_common.py)
----------------------------------------------------
1. Takes a base scenario of ``ensemble_scenarios_real.json`` (default
   ``real_single_base``: data paths, forward models, alphas, drop_coils ...).
2. For every layer DESIGN (a list of interface depths ``depths0``) it prints which
   horizons contain a real TDR depth (SoilNet: 0.15 and 0.30 m). A horizon without a
   sensor can never be calibrated (it falls back to the prior WS parameters), so you
   see this BEFORE spending hours on Step 1.
3. For every design it loops over K = number of FREE WS parameters (freed in the
   order sigma_s, m, sigma_w, a; the others stay at ``lab_p0``).
4. Writes the generated scenarios to ``<out-dir>/hyp1_scenarios.json`` (every scenario
   ``extends`` the parent, so only the differences are listed), runs
   ``./run_one_scenario.sh`` once per scenario and collects the run-log rows into
   ``<out-dir>/hyp1_summary.csv``.

Step 1 (the expensive ensemble inversion) does not depend on the WS freedom, so all K
of one design share ONE ``output_dir`` / ``cache_tag``: the cache is computed by the
first K and reused (fingerprint-checked) by the others.

Usage
-----
    python hyp1_layers_ws_freedom.py --fast --dry-run     # show the plan, run nothing
    python hyp1_layers_ws_freedom.py --fast               # quick smoke test of everything
    python hyp1_layers_ws_freedom.py                      # full study (long)
    python hyp1_layers_ws_freedom.py --designs L3 --free-k 1 2   # a subset
    python hyp1_layers_ws_freedom.py --design L3b=0.2,0.5 --designs L3b   # own design
    python hyp1_layers_ws_freedom.py --fast -- --em-footprint-diagnostic  # flags after
                                                  # '--' go to run_one_scenario.sh

``--fast`` = ``--quick --quick-epochs 2`` + ``em_subsample = probes`` (only EM points
within 5 m of a TDR probe). It is a SMOKE TEST (the calibration works, maps do not):
never a result.
"""
from __future__ import annotations

import argparse
import os
import sys

import hyp_common as hc
import scenario_loader as sl

QUESTION = ("How many layers can the real TDR sensors support, and how many "
            "Waxman-Smits parameters can be freed?")
TAG = "hyp1"
OUT_DIR = os.path.join(hc.OUT_ROOT, "hyp1_layers_ws_freedom")

# Re-exported so existing imports / tests keep working.
BASE_CONFIG = hc.BASE_CONFIG
BASE_SCENARIO = hc.BASE_SCENARIO
RUN_LOG = hc.RUN_LOG
FAST_EPOCHS = hc.FAST_EPOCHS
FAST_SUBSAMPLE = hc.FAST_SUBSAMPLE
pipeline_command = hc.pipeline_command
read_run_log = hc.read_run_log

# Depths [m] of the real SoilNet TDR sensors (data_real.load_tdr_dataframe
# maps the '15' and '30' SWC columns to these two depths).
TDR_DEPTHS = (0.15, 0.30)

# Layer designs: name -> interface depths (n_layers = len + 1).
DESIGNS = {
    "L2": [0.225],             # 15 cm | 30 cm in different horizons (= real_single_*)
    "L3": [0.225, 0.6],        # + a deep horizon WITHOUT sensor
    "L4": [0.3, 0.6, 1.0],     # = real_multi_*: 15 cm in H0, 30 cm in H1, H2-H3 without sensor
}

# Order in which WS parameters are freed.
FREE_ORDER = ("sigma_s", "m", "sigma_w", "a")

SUMMARY_COLUMNS = ["scenario", "design", "k_free"]


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


def parent_name() -> str:
    return "real_study_parent"


def build_study_config(base_raw: dict, designs: dict[str, list[float]], ks: list[int],
                       tdr_depths=TDR_DEPTHS, fast: bool = False,
                       out_dir: str | None = None) -> dict:
    """Study scenarios file content: one parent + one child per (design, K)."""
    out_dir = out_dir or OUT_DIR
    parent = hc.make_parent(
        base_raw, cache_tag="real_study_parent", out_dir=out_dir, fast=fast,
        description="Parent of the real-data layer x WS-freedom study (see hyp1_layers_ws_freedom.py).")
    cond_hi = parent["cond_ranges"][0]
    base_p0 = parent.get("lab_p0") or [0.5, 3.0, 500.0, 100.0]

    cfg = {"_comment": [
        "GENERATED by hyp1_layers_ws_freedom.py -- do not edit, re-run the script.",
        f"Parent '{parent_name()}' = copy of the base scenario; children differ only by "
        "depths0 / cond_ranges (layer design) and the WS freedom (ws_fixed_params, lab_p0, ws_bounds).",
    ], parent_name(): parent}
    for dname, depths0 in designs.items():
        for k in ks:
            cov = horizon_coverage(depths0, tdr_depths)
            cfg[scenario_name(dname, k)] = {
                "extends": parent_name(),
                "description": (f"Real-data study, design {dname} depths0={depths0}, K={k} free WS "
                                f"parameter(s) ({', '.join(FREE_ORDER[:k]) or 'none'}). "
                                f"Horizons without TDR: {[h for h, n in cov.items() if not n] or 'none'}."),
                "depths0": list(depths0),
                "cond_ranges": [list(cond_hi) for _ in range(len(depths0) + 1)],
                # one cache per DESIGN: Step 1 does not depend on the WS freedom
                "output_dir": os.path.join(out_dir, dname),
                "cache_tag": f"real_study_{dname}",
                **ws_freedom(k, base_p0),
            }
    return cfg


def write_summary(rows: list[dict], path: str) -> None:
    hc.write_summary(rows, path, SUMMARY_COLUMNS)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    argv, extra = hc.split_extra(list(sys.argv[1:] if argv is None else argv))

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", nargs="+", help=f"designs to run (default: all of {list(DESIGNS)})")
    ap.add_argument("--design", action="append", default=[], metavar="NAME=d1,d2",
                    help="add or override a layer design (interface depths in m)")
    ap.add_argument("--free-k", nargs="+", type=int, default=None,
                    help="numbers of free WS parameters (0-4, order sigma_s, m, sigma_w, a); "
                         "default 0 1 2 3 4 (0 = nothing calibrated: the floor), or 0 1 4 with --fast")
    ap.add_argument("--tdr-depths", nargs="+", type=float, default=list(TDR_DEPTHS),
                    help="TDR sensor depths [m] used for the coverage report")
    hc.add_common_args(ap, default_out_dir=OUT_DIR)
    args = hc.parse_args(ap, argv)

    designs = dict(DESIGNS)
    for text in args.design:
        n, d = parse_design(text)
        designs[n] = d
    chosen = args.designs or list(designs)
    unknown = [d for d in chosen if d not in designs]
    if unknown:
        ap.error(f"unknown design(s) {unknown}; known: {list(designs)}")
    designs = {d: designs[d] for d in chosen}

    try:
        base_raw = hc.resolve_base(args.config, args.base)
    except KeyError as exc:
        ap.error(str(exc))
    ks = args.free_k or ([0, 1, 4] if args.fast else [0, 1, 2, 3, 4])      # K = 0: nothing calibrated (floor)
    cfg = build_study_config(base_raw, designs, ks, tuple(args.tdr_depths), args.fast, args.out_dir)

    names = [n for n in cfg if n.startswith("real_study_") and n != parent_name()]
    info = {}
    for n in names:
        design, k = n[len("real_study_"):].rsplit("_free", 1)
        info[n] = {"design": design, "k_free": k}
    plan = [f"TDR depths: {args.tdr_depths} m"] + [
        f"  {dname:4s} depths0={d}:  {coverage_text(d, tuple(args.tdr_depths))}"
        for dname, d in designs.items()]
    return hc.execute(tag=TAG, cfg=cfg, names=names, args=args, extra=extra,
                      plan_lines=plan, row_info=info, extra_columns=SUMMARY_COLUMNS)


if __name__ == "__main__":
    sys.exit(main())
