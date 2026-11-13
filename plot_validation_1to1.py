#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_validation_1to1.py
=======================
Truth-vs-optimal 1:1 validation plots for SYNTHETIC EA-EM runs, one column
(or row) per run, so surface-only vs full-depth runs can be read side by side.

Figures (written to --out-dir)
------------------------------
fig_eca_1to1.png              ECa truth (noise-free) vs ECa forward-simulated
                              from the optimal model, coloured by coil.
                              Also reports the noisy *observed* ECa vs truth,
                              i.e. the noise floor the fit cannot go below.
fig_ec_1to1_by_horizon.png    Bulk EC truth vs optimal, one panel per horizon.
fig_theta_1to1_by_horizon.png Water content truth vs calculated, per horizon.
validation_metrics.csv        RMSE / bias / R2 / n for every panel.

Why both ECa and EC per horizon?
--------------------------------
* ECa 1:1  = DATA-FIT check. ECa is what the inversion is fitted to, so a
  good ECa fit is necessary but NOT sufficient: many EC(z) models give the
  same ECa. It will look similar for surface-only and full-depth runs.
* EC (and theta) per horizon 1:1 = TRUTH-RECOVERY check (synthetic only).
  This is the one that discriminates surface-only vs full-depth, because
  horizons without TDR probes are only constrained by the prior.

Inputs read from each run's output_dir  (default: next to this script)
---------------------------------------------------------------------
output_model/synthetic_time_lapse_eca.npy    (ens, time, pts, coils)  truth
output_model/synthetic_time_lapse_ec.npy     (ens, time, pts, layers) truth
output_model/synthetic_time_lapse_theta.npy  (ens, time, pts, layers) truth
output_model/theta_calc.npy                  (time, pts, layers)  saved by run_pipeline
output_model/ec_optimal.npy                  (time, pts, layers)  needs save_ec_optimal.patch
output_model/is_calibrated.npy               (n_horizons,) bool   optional
solution/eca_forward_optimal/eca_forward_optimal_t??.csv   from Step 6e

Missing files are skipped with a warning, never fatal.

Usage
-----
    python plot_validation_1to1.py \\
        --ids 1d_surface_free2_20261112_101500 1d_full_free2_20261112_101700 \\
        --labels surface_free2 full_free2 \\
        --out-dir comparison_output/validation_free2

    python plot_validation_1to1.py --run-dirs <dir1> <dir2> --labels a b
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def _find(run_dir: str, name: str) -> Optional[str]:
    """output_model/<name> first, then <run_dir>/<name> (mirrors _resolve_npy_path)."""
    for d in (os.path.join(run_dir, "output_model"), run_dir):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def _load_npy(run_dir: str, name: str):
    p = _find(run_dir, name)
    if p is None:
        print(f"  [warn] {name} not found in {run_dir}", file=sys.stderr)
        return None
    return np.load(p, allow_pickle=False)


def _truth_slice(arr, realization: int, t: int):
    """(ens, time, pts, X) -> (pts, X) ; tolerates missing ens/time axes."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if a.ndim == 4:
        a = a[realization]
    if a.ndim == 3:
        a = a[t]
    return a


def _model_slice(arr, t: int):
    """(time, pts, X) or (pts, X) -> (pts, X)."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    return a[t] if a.ndim == 3 else a


def resolve_run_dirs(ids, log_file):
    log = pd.read_csv(log_file)
    base = os.path.dirname(os.path.dirname(os.path.abspath(log_file)))  # repo root
    out = []
    for i in ids:
        hit = log[log["run_id"] == i]
        if hit.empty:
            raise SystemExit(f"run_id '{i}' not found in {log_file}")
        d = hit.iloc[-1]["output_dir"]
        out.append(d if os.path.isabs(d) else os.path.join(base, d))
    return out


def load_run(run_dir: str, label: str, realization: int, time_index: int) -> dict:
    r = {"label": label, "dir": run_dir}
    eca_t = _load_npy(run_dir, "synthetic_time_lapse_eca.npy")
    n_time = (eca_t.shape[1] if eca_t is not None and eca_t.ndim == 4 else 1)
    t = time_index if time_index >= 0 else n_time + time_index
    r["t"] = t
    r["eca_truth"] = _truth_slice(eca_t, realization, t)
    r["ec_truth"] = _truth_slice(_load_npy(run_dir, "synthetic_time_lapse_ec.npy"), realization, t)
    r["theta_truth"] = _truth_slice(_load_npy(run_dir, "synthetic_time_lapse_theta.npy"), realization, t)
    r["theta_calc"] = _model_slice(_load_npy(run_dir, "theta_calc.npy"), t)
    r["ec_opt"] = _model_slice(_load_npy(run_dir, "ec_optimal.npy"), t)
    cal = _find(run_dir, "is_calibrated.npy")
    r["is_cal"] = np.load(cal).astype(bool) if cal else None

    csvs = sorted(glob.glob(os.path.join(run_dir, "solution", "eca_forward_optimal",
                                         "eca_forward_optimal_t*.csv")))
    r["eca_sim"] = r["eca_obs"] = r["coils"] = None
    if csvs:
        df = pd.read_csv(csvs[min(t, len(csvs) - 1)])
        coils = [c[:-len("_forward_optimal")] for c in df.columns if c.endswith("_forward_optimal")]
        r["coils"] = coils
        r["eca_sim"] = df[[f"{c}_forward_optimal" for c in coils]].to_numpy(float)
        r["eca_obs"] = df[[f"{c}_observed" for c in coils]].to_numpy(float)
    else:
        print(f"  [warn] no eca_forward_optimal CSV in {run_dir}/solution", file=sys.stderr)
    return r


# ----------------------------------------------------------------------------
# Metrics / panel
# ----------------------------------------------------------------------------
def metrics(x, y) -> dict:
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return {"n": int(x.size), "rmse": np.nan, "bias": np.nan, "r2": np.nan}
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((x - x.mean()) ** 2)
    return {"n": int(x.size),
            "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
            "bias": float(np.mean(y - x)),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan}


def _one_to_one(ax, x, y, color="tab:blue", label=None, unit="", annotate=True,
                lims=None, **kw):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    ax.scatter(x, y, s=18, alpha=0.7, color=color, label=label, **kw)
    if lims is None:
        v = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
        lims = (float(v.min()), float(v.max())) if v.size else (0, 1)
        pad = 0.05 * (lims[1] - lims[0] or 1)
        lims = (lims[0] - pad, lims[1] + pad)
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":", alpha=0.5)
    if annotate:
        m = metrics(x, y)
        ax.text(0.04, 0.96, f"RMSE {m['rmse']:.3g} {unit}\nbias {m['bias']:+.3g}\nR² {m['r2']:.2f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8, lw=0.3))


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_eca(runs, out_dir, rows):
    ok = [r for r in runs if r["eca_truth"] is not None and r["eca_sim"] is not None]
    if not ok:
        print("[skip] ECa 1:1: no run has truth + forward CSV")
        return
    fig, axes = plt.subplots(1, len(ok), figsize=(4.6 * len(ok), 4.8), squeeze=False)
    cmap = plt.cm.tab10
    for ax, r in zip(axes[0], ok):
        truth, sim, obs = r["eca_truth"], r["eca_sim"], r["eca_obs"]
        n = min(len(truth), len(sim))
        truth, sim, obs = truth[:n], sim[:n], obs[:n]
        allv = np.concatenate([truth.ravel(), sim.ravel(), obs.ravel()])
        lo, hi = np.nanmin(allv), np.nanmax(allv)
        pad = 0.05 * (hi - lo or 1)
        lims = (lo - pad, hi + pad)
        for c, coil in enumerate(r["coils"][:truth.shape[1]]):
            ax.scatter(truth[:, c], sim[:, c], s=20, alpha=0.8, color=cmap(c % 10),
                       label=coil)
            ax.scatter(truth[:, c], obs[:, c], s=22, facecolors="none",
                       edgecolors=cmap(c % 10), alpha=0.5, lw=0.8)
        _one_to_one(ax, truth, sim, color="none", unit="mS/m", annotate=False, lims=lims)
        m_sim, m_obs = metrics(truth, sim), metrics(truth, obs)
        ax.text(0.04, 0.96,
                f"optimal vs truth: RMSE {m_sim['rmse']:.3g} mS/m, R² {m_sim['r2']:.2f}\n"
                f"noisy obs vs truth (noise floor): RMSE {m_obs['rmse']:.3g}",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85, lw=0.3))
        ax.set_xlabel("ECa truth (noise-free) [mS/m]")
        ax.set_ylabel("ECa forward-simulated, optimal model [mS/m]")
        ax.set_title(r["label"], fontsize=10, fontweight="bold")
        for q, m in (("ECa_sim_vs_truth", m_sim), ("ECa_obs_vs_truth", m_obs)):
            rows.append({"run": r["label"], "quantity": q, "horizon": "all", **m})
    axes[0][0].legend(fontsize=7, loc="lower right", title="filled: optimal\nhollow: noisy obs")
    fig.suptitle("ECa validation: truth vs simulated optimal (1:1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_eca_1to1.png"), dpi=150)
    plt.close(fig)


def fig_by_horizon(runs, out_dir, rows, truth_key, model_key, quantity, unit, fname):
    ok = [r for r in runs if r[truth_key] is not None and r[model_key] is not None]
    if not ok:
        print(f"[skip] {quantity} 1:1: no run has both {truth_key} and {model_key}")
        return
    nh = max(r[truth_key].shape[1] for r in ok)
    fig, axes = plt.subplots(len(ok), nh, figsize=(3.2 * nh, 3.3 * len(ok)), squeeze=False)
    for i, r in enumerate(ok):
        truth, mod = r[truth_key], r[model_key]
        nl = min(truth.shape[1], mod.shape[1])
        for h in range(nh):
            ax = axes[i][h]
            if h >= nl:
                ax.axis("off")
                continue
            cal = None if r["is_cal"] is None or h >= len(r["is_cal"]) else bool(r["is_cal"][h])
            col = {True: "tab:green", False: "tab:red", None: "tab:blue"}[cal]
            _one_to_one(ax, truth[:, h], mod[:, h], color=col, unit=unit)
            tag = "" if cal is None else (" (calibrated)" if cal else " (uncalibrated)")
            ax.set_title(f"{r['label']} — horizon {h}{tag}", fontsize=8,
                         color={True: "darkgreen", False: "darkred", None: "black"}[cal])
            if h == 0:
                ax.set_ylabel(f"optimal [{unit}]")
            if i == len(ok) - 1:
                ax.set_xlabel(f"truth [{unit}]")
            rows.append({"run": r["label"], "quantity": quantity, "horizon": h,
                         "calibrated": cal, **metrics(truth[:, h], mod[:, h])})
    fig.suptitle(f"{quantity}: truth vs optimal per horizon (1:1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="+", help="run_id(s) from the run log")
    ap.add_argument("--run-dirs", nargs="+", help="pipeline output dirs (alternative to --ids)")
    ap.add_argument("--log-file", default="mfc2_outputs/run_log.csv")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--realization", type=int, default=0)
    ap.add_argument("--time-index", type=int, default=-1)
    ap.add_argument("--out-dir", default="comparison_output/validation_1to1")
    a = ap.parse_args(argv)

    if bool(a.ids) == bool(a.run_dirs):
        ap.error("give exactly one of --ids or --run-dirs")
    dirs = a.run_dirs or resolve_run_dirs(a.ids, a.log_file)
    labels = a.labels or [os.path.basename(d.rstrip("/")) for d in dirs]
    if len(labels) != len(dirs):
        ap.error("--labels must match the number of runs")

    os.makedirs(a.out_dir, exist_ok=True)
    runs = [load_run(d, l, a.realization, a.time_index) for d, l in zip(dirs, labels)]
    rows: list[dict] = []
    fig_eca(runs, a.out_dir, rows)
    fig_by_horizon(runs, a.out_dir, rows, "ec_truth", "ec_opt",
                   "EC_bulk", "mS/m", "fig_ec_1to1_by_horizon.png")
    fig_by_horizon(runs, a.out_dir, rows, "theta_truth", "theta_calc",
                   "theta", "m³/m³", "fig_theta_1to1_by_horizon.png")
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "validation_metrics.csv"), index=False)
    print(f"done -> {a.out_dir}  ({len(rows)} metric rows)")


if __name__ == "__main__":
    main()
