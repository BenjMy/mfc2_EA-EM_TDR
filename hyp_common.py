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

from EA_EM_config import OUTPUT_ROOT, resolve_real_paths  # central output root + data-path resolution (EA_EM_config.py)
import argparse
import copy
import csv
import json
import os
import subprocess
import time

import em_options
import scenario_loader as sl

BASE_CONFIG = "ensemble_scenarios_real.json"
BASE_SCENARIO = "real_single_base"
OUT_ROOT = os.path.join(OUTPUT_ROOT, "hypotheses")

# --data real|synthetic: ONE switch for the dataset (sets --config and --base, and keeps
# the outputs of the synthetic mirror apart from the real ones).
#   real       the field data in ./data/            (ensemble_scenarios_real.json)
#   synthetic  the synthetic twin of the field data (generate_synthetic_real_mirror.py)
SYNTHETIC_CONFIG = "mfc2_synthetic/mfc2_real_synthetic_mirror.json"
SYNTHETIC_BASE = "mfc2_real_synthetic_mirror"
DATASETS = {"real": (BASE_CONFIG, BASE_SCENARIO),
            "synthetic": (SYNTHETIC_CONFIG, SYNTHETIC_BASE)}
SYNTHETIC_OUT_ROOT = OUT_ROOT + "_synthetic"
RUN_LOG = os.path.join(OUTPUT_ROOT, "run_log.csv")

FAST_EPOCHS = 2
FAST_SUBSAMPLE = {"mode": "probes", "radius_m": 5.0}

# --stage: the hypotheses are about the CALIBRATION (RMSE at the TDR probes), which
# the 'calibrate' stage computes on the probe neighbourhoods only. 'none' = full run.
# 'both' = per scenario, --stage calibrate and then --stage map on that calibration
# (the map run inverts only the K selected members, on all EM points).
STAGES = ("calibrate", "map", "both", "none")   # 'map' alone: needs an existing calibrate run (newest one unless --calibration is forwarded after '--')

COMMON_COLUMNS = ["scenario", "stage", "status", "final_rmse", "cv_mean", "ensemble_size",
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


WS_PARAMS = ("a", "m", "sigma_w", "sigma_s")


def k0_child(parent_name: str, *, base: str, out_dir: str, cache_tag: str, what: str) -> dict:
    """K = 0 reference of a hypothesis: all four WS parameters pinned at ``lab_p0``, nothing calibrated.

    It extends the same parent and shares the hypothesis' Step 1 cache (``cache_tag`` / run folder), so it costs
    only Steps 2-6: it is the floor that every calibrated variant has to beat.
    """
    return {"extends": parent_name,
            "description": f"K = 0 reference of {what}: all WS parameters pinned at lab_p0 (nothing calibrated); "
                           f"otherwise = {base}.",
            "ws_fixed_params": {p: True for p in WS_PARAMS},
            "output_dir": os.path.join(out_dir, "runs"),
            "cache_tag": cache_tag}


def add_k0_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--no-k0", action="store_true",
                    help="skip the K = 0 reference run (all four WS parameters pinned, nothing calibrated; it shares "
                         "the Step 1 cache of the hypothesis, so it is cheap)")


def stages_to_run(stage: str, fast: bool = False) -> list[str]:
    """The ``run_one_scenario.sh --stage`` values executed for one scenario.

    ``both`` = calibrate, then map. ``--stage map`` cannot be combined with
    ``--quick``, so a ``--fast`` smoke test only does the calibrate half.
    """
    if stage == "both":
        return ["calibrate"] if fast else ["calibrate", "map"]
    return [stage]


def pipeline_command(name: str, config_path: str, mode: str, fast: bool,
                     extra: list[str], stage: str = "calibrate",
                     calibration: str | None = None) -> list[str]:
    """``stage`` is calibrate / map / none (``both`` is expanded by stages_to_run).

    ``calibration`` (map only): the calibrate run folder or its stage_calibration.json;
    without it run_one_scenario.sh takes the newest ``*_calibrate`` run of the scenario.
    """
    if stage == "both":
        raise ValueError("'both' is expanded with stages_to_run(); pass calibrate or map")
    cmd = ["./run_one_scenario.sh", "-s", name, "-c", config_path, "--mode", mode]
    if stage != "none":
        cmd += ["--stage", stage]
    if stage == "map" and calibration:
        cmd += ["--calibration", calibration]
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
    ap.add_argument("--data", choices=sorted(DATASETS), default=None,
                    help="dataset: 'real' = field data in ./data (default); 'synthetic' = the "
                         f"synthetic mirror ({SYNTHETIC_CONFIG}). Shortcut for --config/--base, "
                         f"and writes under {SYNTHETIC_OUT_ROOT}/ instead of {OUT_ROOT}/")
    ap.add_argument("--config", default=BASE_CONFIG, help="scenarios file (parent source)")
    ap.add_argument("--base", default=BASE_SCENARIO,
                    help="scenario copied as parent; it fixes the LAYERS of the run "
                         "(e.g. real_L3_base for 3 layers)")
    ap.add_argument("--mode", default="2d", choices=["1d", "2d"])
    ap.add_argument("--stage", default="calibrate", choices=STAGES,
                    help="calibrate (default): probe neighbourhoods only, Steps 0-5; "
                         "map: only the map run (Step 6, all EM points) on the newest calibrate run; "
                         "both: calibrate, then map for every scenario; none: the full run")
    ap.add_argument("--out-dir", default=default_out_dir)
    em_options.add_em_arg(ap)
    ap.add_argument("--fast", action="store_true",
                    help=f"smoke test: --quick, {FAST_EPOCHS} epochs, probe points only")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the config and print the plan, run nothing")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue with the next scenario after a failure")
    ap.add_argument("--plot-map", action="store_true",
                    help="run NOTHING: redraw the XY water-content maps of the newest finished "
                         "'--stage map' run of each scenario from its output_model/maps.nc "
                         "(written to <run folder>/replot/)")


def _plot_maps_only(cfg: dict, names: list[str], cfg_path: str) -> int:
    """``--plot-map``: redraw the maps of the newest map run of every scenario, no inversion."""
    import replot_maps
    failed = []
    for n in names:
        run_dir = replot_maps.latest_map_run(cfg[n]["output_dir"], n)
        if run_dir is None:
            print(f"!!! {n}: no finished map run (with output_model/maps.nc) under {cfg[n]['output_dir']}")
            failed.append(n)
            continue
        sc = sl.load_scenario(n, config_path=cfg_path)
        window_m = getattr(sc, "window_m", None)
        em_dir = coord_file = None
        if getattr(sc, "data_source", None) == "real":      # real field data or the synthetic mirror
            rp = resolve_real_paths(scenario_em_dir=getattr(sc, "em_data_dir", None),
                                    scenario_tdr_dir=getattr(sc, "tdr_data_dir", None),
                                    scenario_coord_file=getattr(sc, "coord_file", None))
            em_dir, coord_file = rp.em_dir, rp.coord_file
        try:
            for p in replot_maps.replot_map_run(run_dir, n, window_m=window_m,
                                                em_dir=em_dir, coord_file=coord_file):
                print(f"[plot-map] {p}")
        except (OSError, ValueError) as exc:
            print(f"!!! {n}: {exc}")
            failed.append(n)
    return 1 if failed else 0


def _to_synthetic_root(path: str | None) -> str | None:
    """mfc2_outputs/hypotheses[/x]  ->  mfc2_outputs/hypotheses_synthetic[/x] (other paths untouched)."""
    if path and (path == OUT_ROOT or path.startswith(OUT_ROOT + os.sep)):
        return SYNTHETIC_OUT_ROOT + path[len(OUT_ROOT):]
    return path


def parse_args(ap: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    """``ap.parse_args`` + resolve ``--data`` (use this instead of ``ap.parse_args``)."""
    args = ap.parse_args(argv)
    if args.data is None and args.config != BASE_CONFIG and "synthetic" in os.path.basename(args.config).lower():
        args.data = "synthetic"           # a mirror JSON given with --config: treat it as such
        print("[data] --config looks like the synthetic mirror: using --data synthetic "
              f"(outputs under {SYNTHETIC_OUT_ROOT}/)")
    if args.data:
        cfg, base = DATASETS[args.data]
        if args.config == BASE_CONFIG:
            args.config = cfg
        if args.base == BASE_SCENARIO:
            args.base = base
    if args.data == "synthetic":
        for attr in ("out_dir", "out_root"):
            if hasattr(args, attr):
                setattr(args, attr, _to_synthetic_root(getattr(args, attr)))
    return args


def describe_data(name: str, cfg_path: str) -> list[str]:
    """Lines saying WHICH data a scenario reads (real field data or the synthetic mirror)."""
    sc = sl.load_scenario(name, config_path=cfg_path)
    p = resolve_real_paths(scenario_em_dir=getattr(sc, "em_data_dir", None),
                           scenario_tdr_dir=getattr(sc, "tdr_data_dir", None),
                           scenario_coord_file=getattr(sc, "coord_file", None))
    src = getattr(sc, "data_source", None)
    if src != "real":
        kind = f"SYNTHETIC (generated in-process, data_source={src!r})"
    elif "synthetic" in (p.em_dir or "").lower():
        kind = "SYNTHETIC MIRROR (known truth) -- read through the real-data loader"
    else:
        kind = "REAL FIELD DATA"
    return [f"DATA: {kind}",
            f"      EM    : {p.em_dir}", f"      TDR   : {p.tdr_dir}", f"      coords: {p.coord_file}"]


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
    em = getattr(args, "em_method", em_options.DEFAULT_EM)
    ftag = tag
    if em != em_options.DEFAULT_EM:       # EM method of Step 1 (em_options.py): same scenario names, separate cache / run folders / files
        cfg = em_options.apply_to_config(cfg, em)
        ftag = f"{tag}_{em}"
        plan_lines = [*plan_lines, f"EM method: {em_options.method_name(em)} (param_grid.method; cache_tag / output_dir "
                                   f"suffixed '_{em}'; files '{ftag}_*')"]
    cfg_path = os.path.join(args.out_dir, f"{ftag}_scenarios.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    for n in names:                       # fail now, not after hours of Step 1
        sl.load_scenario(n, config_path=cfg_path)

    print(f"[{tag}] config  : {cfg_path}  ({len(names)} scenarios, base '{args.base}', "
          f"stage '{args.stage}')")
    for line in describe_data(names[0], cfg_path):
        print(f"[{tag}] {line}")
    for line in plan_lines:
        print(f"[{tag}] {line}")
    if args.fast:
        print(f"[{tag}] >>> FAST smoke test: --quick --quick-epochs {FAST_EPOCHS}, "
              f"em_subsample={FAST_SUBSAMPLE}")
    run_stages = stages_to_run(args.stage, args.fast)
    if args.stage == "both" and args.fast:
        print(f"[{tag}] >>> --stage both with --fast: map skipped (--stage map cannot be combined with --quick)")
    print(f"[{tag}] commands (the data paths above are read from {cfg_path}):")
    for n in names:
        for st in run_stages:
            cal = "<calibrate run folder>" if st == "map" else None
            print("  $", " ".join(pipeline_command(n, cfg_path, args.mode, args.fast, extra, st, cal)))
    if args.dry_run:
        return 0
    if getattr(args, "plot_map", False):
        return _plot_maps_only(cfg, names, cfg_path)

    rows, failed = [], []
    summary_path = os.path.join(args.out_dir, f"{ftag}_summary.csv")
    stop = False
    for n in names:
        calibration = None
        for st in run_stages:
            before = len(read_run_log())
            t0 = time.time()
            rc = subprocess.call(pipeline_command(n, cfg_path, args.mode, args.fast, extra, st, calibration))
            new = [r for r in read_run_log()[before:] if r.get("scenario", "").startswith(n)]
            row = dict(new[-1]) if new else {"status": f"no run-log row (exit {rc})"}
            row.update({"scenario": n, "stage": st if args.stage == "both" else args.stage,
                        **((row_info or {}).get(n, {}))})
            row.setdefault("runtime_s", round(time.time() - t0))
            rows.append(row)
            write_summary(rows, summary_path, extra_columns)   # after every run
            if rc != 0:
                failed.append(f"{n}[{st}]")
                if not args.keep_going:
                    print(f"!!! {n} ({st}) failed (exit {rc}); stopping (use --keep-going to continue)")
                    stop = True
                break                  # never map a scenario whose calibration failed
            if st == "calibrate":
                calibration = row.get("output_dir") or None
        if stop:
            break

    print(f"\n[{tag}] summary: {summary_path}")
    for r in rows:
        print(f"  {r['scenario']:34s} {str(r.get('stage', '')):10s} {str(r.get('status', '')):10s} "
              f"final_rmse={r.get('final_rmse', '')}")
    return 1 if failed else 0
