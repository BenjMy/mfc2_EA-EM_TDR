#!/usr/bin/env python3
"""
ws_recovery.py -- Waxman-Smits parameter recovery table of finished pipeline runs.

Pure parsing (no emagpy, no numpy): reads the ``pipeline_report_2*.md`` of a run folder and returns
one row per (scenario, horizon) with, for each of a, m, sigma_w, sigma_s:

    <p>_start   calibration prior/start      line "Calibration prior/start (a, m, sigma_w, sigma_s) = (...)"
    <p>_truth   synthetic ground truth       table "Step 0 - Synthetic ground-truth Waxman-Smits parameters"
    <p>_fit     fitted value                 table "Step 4 - Best recalibrated fit" (fallback:
                                             "Step 2 - Best-ranked Waxman-Smits fit")
    <p>_fixed   True when pinned             row ``ws_fixed_params`` of the run-arguments table
    <p>_lo/_hi  bounds                       row ``ws_bounds`` merged over the defaults (EMmethod.WS_BOUNDS)

and the derived  C = a / (sigma_w + sigma_s)  (``C_start``, ``C_truth``, ``C_fit``): theta depends on the
parameters only through C and m, so C is what the calibration really has to recover.

Notes
-----
* ``truth_baseline_only`` is True when the truth table is only the per-layer BASELINE (``ws_xy_variation``
  on: m / sigma_s then vary along x and no single truth number exists).
* Horizons without a TDR probe are reported on the prior (``calibrated`` False, ``<p>_fit`` = prior).
* The start is one value for all horizons (the pipeline uses one ``lab_p0``); ``a`` is reported as the
  pipeline logs it (the 'a'-free scenarios start at 1.0).

CLI:  python ws_recovery.py RUN_DIR_OR_REPORT [...] [-o ws_recovery.csv]
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import os
import re

PARAMS = ("a", "m", "sigma_w", "sigma_s")
# EMmethod.WS_BOUNDS (em_methodology.py), duplicated so that this module stays import-light.
DEFAULT_BOUNDS = {"a": (0.5, 5.0), "m": (1.0, 4.0), "sigma_w": (1.0, 1000.0), "sigma_s": (0.0, 200.0)}

COLUMNS = (["scenario", "horizon", "calibrated", "fit_source", "truth_baseline_only"]
           + [f"{p}_{s}" for p in PARAMS for s in ("start", "truth", "fit", "fixed", "lo", "hi")]
           + ["C_start", "C_truth", "C_fit", "report"])

_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_ROW = re.compile(r"^\|\s*`(?P<k>.*?)`\s*\|\s*`(?P<v>.*)`\s*\|\s*$")
_TITLE = re.compile(r"^\*\*(?P<t>.+)\*\*\s*$")
_START = re.compile(r"Calibration prior/start \(a, m, sigma_w, sigma_s\)\s*=\s*"
                    rf"\(\s*({_NUM}),\s*({_NUM}),\s*({_NUM})\s*mS/m,\s*({_NUM})\s*mS/m\s*\)")
_LAYER = re.compile(r"^Layer\s+(\d+)\b")
_HZ = re.compile(r"^Hz\s+(\d+)\b")


def read_tables(text: str) -> dict[str, dict[str, str]]:
    """``**title**`` + ``| `key` | `value` |`` markdown tables -> {title: {key: value}} (key/value tables only)."""
    tables: dict[str, dict[str, str]] = {}
    title = None
    for line in text.splitlines():
        m = _TITLE.match(line.strip())
        if m:
            title = m.group("t").strip()
            continue
        r = _ROW.match(line.strip())
        if r and title is not None:
            tables.setdefault(title, {})[r.group("k")] = r.group("v")
        elif title is not None and line.strip() and not line.startswith("|") and not line.startswith("_"):
            title = None
    return tables


def _parse_params(value: str):
    """'a / m / σ_w mS/m / σ_s mS/m [/ RMSE ...]' -> (dict of the 4 parameters, calibrated flag)."""
    calibrated = "UNCALIBRATED" not in value
    body = value.split("p0 only:")[-1] if not calibrated else value
    parts = [p.strip() for p in body.replace("mS/m", "mS_m").split("/")]      # unit contains a slash
    nums = []
    for p in parts[:4]:
        m = re.search(_NUM, p)
        if not m:
            return None, calibrated
        nums.append(float(m.group(0)))
    return dict(zip(PARAMS, nums)), calibrated


def _c(p: dict | None):
    if not p:
        return None
    den = p["sigma_w"] + p["sigma_s"]
    return p["a"] / den if den else None


def _first(tables: dict, prefix: str):
    for t, rows in tables.items():
        if t.startswith(prefix):
            return t, rows
    return None, None


def _args_value(tables: dict, key: str):
    for rows in tables.values():
        if key in rows:
            return rows[key]
    return None


def _literal(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None


def parse_report(text: str, scenario: str = "", report: str = "") -> list[dict]:
    """Rows (one per horizon) of one pipeline report; [] when the report has no fit table."""
    tables = read_tables(text)
    src, fit_rows = _first(tables, "Step 4 — Best recalibrated fit")
    source = "step4"
    if fit_rows is None:
        src, fit_rows = _first(tables, "Step 2 — Best-ranked Waxman-Smits fit")
        source = "step2"
    if fit_rows is None:
        return []

    truth_title, truth_rows = _first(tables, "Step 0 — Synthetic ground-truth Waxman-Smits")
    truth_layers: dict[int, dict] = {}
    for k, v in (truth_rows or {}).items():
        m = _LAYER.match(k)
        if m:
            p, _ = _parse_params(v)
            if p:
                truth_layers[int(m.group(1))] = p
    baseline_only = bool(truth_title and "layer baseline" in truth_title)

    sm = _START.search(text)
    start = dict(zip(PARAMS, map(float, sm.groups()))) if sm else None

    fixed = _literal(_args_value(tables, "ws_fixed_params") or "") or {}
    fixed = fixed if isinstance(fixed, dict) else {}
    bounds = {p: tuple(v) for p, v in DEFAULT_BOUNDS.items()}
    user = _literal(_args_value(tables, "ws_bounds") or "")
    if isinstance(user, dict):
        for p, v in user.items():
            if p in bounds and len(v) == 2:
                bounds[p] = (float(v[0]), float(v[1]))

    rows = []
    for k, v in fit_rows.items():
        m = _HZ.match(k)
        if not m:
            continue
        hz = int(m.group(1))
        fit, calibrated = _parse_params(v)
        if fit is None:
            continue
        truth = truth_layers.get(hz)
        row = {"scenario": scenario, "horizon": hz, "calibrated": calibrated, "fit_source": source,
               "truth_baseline_only": baseline_only, "report": report,
               "C_start": _c(start), "C_truth": _c(truth), "C_fit": _c(fit)}
        for p in PARAMS:
            row[f"{p}_start"] = None if start is None else start[p]
            row[f"{p}_truth"] = None if truth is None else truth[p]
            row[f"{p}_fit"] = fit[p]
            row[f"{p}_fixed"] = bool(fixed[p]) if p in fixed else None
            row[f"{p}_lo"], row[f"{p}_hi"] = bounds[p]
        rows.append(row)
    return rows


def latest_report(run_dir: str) -> str | None:
    """Newest ``pipeline_report_2*.md`` of a run folder (the shell companion report is ``pipeline_report_shell_*``)."""
    hits = sorted(glob.glob(os.path.join(run_dir, "pipeline_report_2*.md")))
    return hits[-1] if hits else None


def collect(run_dirs: dict[str, str]) -> list[dict]:
    """{scenario: run_dir} -> rows of all runs that have a report with a fit table."""
    rows: list[dict] = []
    for scenario, run_dir in run_dirs.items():
        path = run_dir if run_dir.endswith(".md") else latest_report(run_dir) if run_dir else None
        if not path or not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            rows += parse_report(f.read(), scenario, path)
    return rows


def write_csv(rows: list[dict], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: "" if r.get(c) is None else r[c] for c in COLUMNS})
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="run folders or pipeline_report_*.md files")
    ap.add_argument("-o", "--out", default="study_ws_recovery.csv")
    args = ap.parse_args(argv)
    runs = {os.path.basename(os.path.normpath(p if os.path.isdir(p) else os.path.dirname(p))): p for p in args.paths}
    rows = collect(runs)
    print(f"{len(rows)} rows from {len(runs)} run(s) -> {write_csv(rows, args.out)}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
