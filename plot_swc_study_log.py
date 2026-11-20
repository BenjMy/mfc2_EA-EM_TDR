#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_swc_study_log.py -- all summary plots of the SWC design study, from its own log
``mfc2_outputs/run_log_swc_study.csv`` (see study_swc_design.py).

Same idea as plot_run_log.py, but the scenario names ``swcstudy_smc*_ws*_<reg>_<dist>_n*_free*``
are parsed into the study factors (smc, ws, reg, dist, n, k), so the plots test the hypotheses
directly: RMSE vs number of SWC positions n, vs free parameters k, probe distribution,
and a heatmap n x k.

    python plot_swc_study_log.py                        # all plots -> comparison_output/swc_log_plots/
    python plot_swc_study_log.py --list
    python plot_swc_study_log.py --compare              # + auto_compare.py --preset swc (reference design)

Plots (RMSE in % vol; the RSR panel title is added when the runs logged RMSE/std):
    swc_rmse_vs_n.png       RMSE vs n positions, one line per probe distribution
    swc_rmse_vs_k.png       RMSE vs free WS parameters k, one line per truth (smc/ws)
    swc_heatmap_n_k.png     RMSE over n x k (reference truth, distribution, regularisation)
    swc_reg_effect.png      RMSE by regularisation (noreg/beta/gamma/both)
    swc_rmse_vs_rsr.png     (from plot_run_log) error vs spread of the data
    swc_runs.csv            table behind the plots
The final RMSE here is the one at the probes; the truth-based metrics of the full study
are made by ``study_swc_design.py analyze``.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import plot_run_log as prl  # noqa: E402
from EA_EM_config import OUTPUT_ROOT  # noqa: E402
from study_swc_design import FACTORS, REFERENCE, parse_name  # noqa: E402

SWC_LOG = os.path.join(OUTPUT_ROOT, "run_log_swc_study.csv")
DEFAULT_OUT = os.path.join("comparison_output", "swc_log_plots")


def design_table(latest: dict[str, dict]) -> list[dict]:
    """Latest runs with the study factors attached (scenarios that do not parse are dropped)."""
    out = []
    for name, r in latest.items():
        d = parse_name(name)
        if d is not None:
            out.append({**r, **{f: getattr(d, f) for f in FACTORS}})
    return out


def _select(table, **fixed):
    return [r for r in table if all(r[k] == v for k, v in fixed.items())]


def _lines(table, x, by, out_dir, fname, title, xlabel, **fixed):
    sub = _select(table, **fixed)
    groups = {}
    for r in sub:
        groups.setdefault(by(r), []).append((r[x], r["rmse_pct"]))
    groups = {g: sorted(v) for g, v in groups.items() if len(v) >= 2}
    if not groups:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for g, v in sorted(groups.items()):
        ax.plot([a for a, _ in v], [b for _, b in v], "o-", label=str(g))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE at probes [% vol]")
    ax.set_xticks(sorted({a for v in groups.values() for a, _ in v}))
    ax.set_title(title + ("  [" + ", ".join(f"{k}={v}" for k, v in fixed.items()) + "]" if fixed else ""), fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return prl._save(fig, out_dir, fname)


def plot_all(rows: list[dict], out_dir: str) -> list[str]:
    latest = prl.latest_per_scenario(rows)
    table = design_table(latest)
    if not table:
        return []
    ref = REFERENCE
    files = []
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "swc_runs.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", *FACTORS, "rmse_pct", "rsr", "rating"])
        for r in sorted(table, key=lambda r: r["scenario"]):
            w.writerow([r["scenario"], *[r[k] for k in FACTORS], f"{r['rmse_pct']:.3f}",
                        "" if not np.isfinite(r["rsr"]) else f"{r['rsr']:.3f}", r["rating"]])
    files.append(path)
    ref_truth = dict(smc=ref["smc"], ws=ref["ws"], reg=ref["reg"])
    files.append(_lines(table, "n", lambda r: r["dist"], out_dir, "swc_rmse_vs_n.png",
                        "RMSE vs number of SWC positions", "n positions", k=ref["k"], **ref_truth))
    files.append(_lines(table, "k", lambda r: f"smc={r['smc']} ws={r['ws']}", out_dir, "swc_rmse_vs_k.png",
                        "RMSE vs free WS parameters", "free parameters k",
                        n=ref["n"], dist=ref["dist"], reg=ref["reg"]))
    sub = _select(table, dist=ref["dist"], **ref_truth)
    ns, ks = sorted({r["n"] for r in sub}), sorted({r["k"] for r in sub})
    if len(ns) >= 2 and len(ks) >= 2:
        grid = np.full((len(ns), len(ks)), np.nan)
        for r in sub:
            grid[ns.index(r["n"]), ks.index(r["k"])] = r["rmse_pct"]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(ks)), ks)
        ax.set_yticks(range(len(ns)), ns)
        ax.set_xlabel("free parameters k")
        ax.set_ylabel("n positions")
        for i in range(len(ns)):
            for j in range(len(ks)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color="w", fontsize=7)
        fig.colorbar(im, label="RMSE [% vol]")
        ax.set_title("RMSE at probes, n x k", fontsize=9)
        files.append(prl._save(fig, out_dir, "swc_heatmap_n_k.png"))
    regs = {}
    for r in _select(table, n=ref["n"], k=ref["k"], dist=ref["dist"], smc=ref["smc"], ws=ref["ws"]):
        regs[r["reg"]] = r["rmse_pct"]
    if len(regs) >= 2:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.bar(list(regs), list(regs.values()))
        ax.set_ylabel("RMSE at probes [% vol]")
        ax.set_title("Regularisation effect", fontsize=9)
        files.append(prl._save(fig, out_dir, "swc_reg_effect.png"))
    files.append(prl.plot_rmse_vs_rsr({r["scenario"]: r for r in table}, out_dir, "swc study"))
    return [f for f in files if f]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    ap.add_argument("--log-file", default=SWC_LOG)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--compare", action="store_true", help="also run auto_compare.py --preset swc (reference design)")
    a = ap.parse_args(argv)
    if not os.path.isfile(a.log_file):
        print(f"[plot_swc_study_log] run log not found: {a.log_file}", file=sys.stderr)
        return 2
    rows = [r for r in prl.load_rows(a.log_file) if parse_name(r["scenario"]) is not None]
    latest = prl.latest_per_scenario(rows)
    print(f"[plot_swc_study_log] {a.log_file}: {len(rows)} runs, {len(latest)} finished scenarios")
    if a.list:
        return 0
    for f in plot_all(rows, a.out_dir):
        print("   wrote", f)
    if a.compare:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.call([sys.executable, os.path.join(here, "auto_compare.py"), "--preset", "swc",
                                "--log-file", a.log_file, "--n", "5", "--k", "1", "2", "3", "4",
                                "--dist", ref_dist(), "--smc", REFERENCE["smc"], "--ws", REFERENCE["ws"],
                                "--reg", REFERENCE["reg"]])
    return 0


def ref_dist() -> str:
    return REFERENCE["dist"]


if __name__ == "__main__":
    sys.exit(main())
