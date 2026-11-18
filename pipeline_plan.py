"""pipeline_plan -- what a run WOULD do (``run_pipeline.py --dry-run``).

Pure functions, no heavy imports: they only read the resolved scenario and the CLI choices, so
a dry run is fast, writes nothing (no output folder, no cache, no log row) and still catches the
mistakes that otherwise surface minutes into a run (bad quick/stage combinations, missing EM
files, unknown split).  The Step-1 task count follows ``quick_preset.count_tasks``; the number
of starting-conductivity combinations is the one ``scenario_loader`` already uses for its
``members_estimate`` (product of the lengths of the multi-valued ``cond_ranges`` entries), so it
is an estimate -- layer pinning can reduce it at run time.
"""
from __future__ import annotations

import glob
import os

import numpy as np

import quick_preset

TRAIN_VAL_METHODS = ("off", "alternate_sorted", "by_probe", "by_time_block", "random")


def effective_param_grid(cfg, beta=None, gamma=None, quick: bool = False):
    """The ensemble grid the run would use: ``(grid, quick_info)``.

    Mirrors ``run_pipeline.main``: ``alpha_range_log`` overrides ``param_grid['alpha']``, ``gamma``
    defaults to ``cfg.gamma``, ``--beta/--gamma`` win, and ``--quick`` keeps one alpha / one model.
    """
    grid = {k: list(v) for k, v in (cfg.param_grid or {}).items()}
    if getattr(cfg, "alpha_range_log", None) is not None:
        lo, hi, n = cfg.alpha_range_log
        grid["alpha"] = [float(a) for a in np.logspace(np.log10(lo), np.log10(hi), int(n))]
    if "gamma" not in grid:
        grid["gamma"] = [getattr(cfg, "gamma", 0.0)]
    if beta is not None:
        grid["beta"] = list(beta)
    if gamma is not None:
        grid["gamma"] = list(gamma)
    info = {}
    if quick:
        grid, info = quick_preset.reduce_param_grid(grid)
    return grid, info


def n_start_combos(cfg) -> int:
    """Starting-conductivity combinations implied by ``cond_ranges`` (1 when all are single-valued)."""
    n = 1
    for v in (cfg.cond_ranges or []):
        if len(v) > 1:
            n *= len(v)
    return n


def n_real_epochs(em_dir: str | None) -> int | None:
    """Number of EM survey CSV files in ``em_dir`` (None when the folder is not given)."""
    if not em_dir or not os.path.isdir(em_dir):
        return None
    return len(glob.glob(os.path.join(em_dir, "*.csv")))


def build_plan(cfg, *, mode, data_source, em_dir=None, quick=False, quick_epochs=None,
               stage="full", calibration=None, train_val_split=None, beta=None, gamma=None,
               ws_localized_bandwidth=None, ws_localized_bandwidth_sweep=None,
               em_footprint_diagnostic=False):
    """Return ``(rows, errors)``: ``rows`` = ``[(label, value), ...]`` to print, ``errors`` = strings.

    An empty ``errors`` means the run is expected to start; it does not guarantee it succeeds.
    """
    errors: list[str] = []
    grid, qinfo = effective_param_grid(cfg, beta=beta, gamma=gamma, quick=quick)
    if quick and quick_epochs is not None and data_source != "real":
        errors.append("--quick-epochs applies to real-data runs only (synthetic epochs come from the generator)")
    if quick_epochs is not None and not quick:
        errors.append("--quick-epochs requires --quick")
    if stage == "map" and quick:
        errors.append("--stage map cannot be combined with --quick")
    if calibration is not None and stage != "map":
        errors.append("--calibration requires --stage map")
    if stage == "map" and calibration is None:
        errors.append("--stage map needs --calibration <run folder or stage_calibration.json>")
    if stage == "map" and calibration is not None and not os.path.exists(calibration):
        errors.append(f"--calibration path not found: {calibration}")
    split = train_val_split or getattr(cfg, "train_val_split", None) or "off"
    if split not in TRAIN_VAL_METHODS:
        errors.append(f"unknown train_val_split '{split}' (choose from {', '.join(TRAIN_VAL_METHODS)})")

    n_epochs = n_real_epochs(em_dir) if data_source == "real" else getattr(cfg, "n_time", None)
    if data_source == "real" and not n_epochs:
        errors.append(f"no EM *.csv files in {em_dir}")
    if quick and quick_epochs is not None and n_epochs:
        n_epochs = len(quick_preset.select_epochs(n_epochs, quick_epochs))
    n_grid = 1
    for k in ("alpha", "beta", "gamma", "method", "forwardModel"):
        n_grid *= max(len(grid.get(k, [None])), 1)
    n_start = n_start_combos(cfg)
    n_tasks = quick_preset.count_tasks(grid, n_start, n_epochs) if n_epochs else None

    rows = [
        ("scenario", cfg.name),
        ("data source / mode", f"{data_source} / {mode}"),
        ("stage", stage + (f" (calibration: {calibration})" if calibration else "")),
        ("quick preset", ("ON: " + ", ".join(f"{k} {v['n_full']}->1 (kept {v['kept']})"
                                              for k, v in qinfo.items()) or "ON (nothing to reduce)")
         + (f"; epochs: {quick_epochs}" if quick_epochs else "") if quick else "off"),
        ("ensemble grid", ", ".join(f"{k}={len(v)}" for k, v in grid.items()) + f"  -> {n_grid} combination(s)"),
        ("starting-conductivity combinations (estimate)", n_start),
        ("epochs", n_epochs if n_epochs is not None else "from the generator"),
        ("Step-1 inversion tasks (estimate)", n_tasks if n_tasks is not None else "n/a"),
        ("train/validation split", split),
        ("localized WS bandwidth", ws_localized_bandwidth if ws_localized_bandwidth is not None
         else getattr(cfg, "ws_localized_bandwidth", None) or "off"),
        ("localized bandwidth sweep", ws_localized_bandwidth_sweep
         or getattr(cfg, "ws_localized_bandwidth_sweep", None) or "off"),
        ("EM footprint diagnostic (Step 5c)", "on" if em_footprint_diagnostic else "off"),
        ("output folder (not created)", f"{getattr(cfg, 'output_dir', None)}/<seq>_{cfg.name}"
         + ("_quick" if quick else "")),
    ]
    return rows, errors


def format_plan(rows, errors) -> str:
    """Printable block for the dry run."""
    w = max(len(k) for k, _ in rows)
    out = ["=" * 80, " [DRY RUN] nothing is executed, nothing is written", "=" * 80]
    out += [f" {k:<{w}} : {v}" for k, v in rows]
    if errors:
        out += ["-" * 80] + [f" [ERROR] {e}" for e in errors]
    else:
        out += ["-" * 80, " [DRY RUN] OK: scenario, paths and options are consistent."]
    out.append("=" * 80)
    return "\n".join(out)
