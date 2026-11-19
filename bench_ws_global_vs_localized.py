#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_ws_global_vs_localized.py -- global vs localized Waxman-Smits on synthetic scenarios.

One pipeline run already holds both calibrations: with the default ``--ws-map-calibration
localized`` Step 5 writes ``theta_calc.npy`` (localized map) AND ``theta_calc_global.npy``
(one WS set per horizon), next to the synthetic truth ``synthetic_time_lapse_theta.npy``.
So each scenario is run ONCE and both maps are scored against the same truth.

Three sub-commands (``all`` chains them)::

    python bench_ws_global_vs_localized.py generate   # write the scenario JSON (no run)
    python bench_ws_global_vs_localized.py run       # run_pipeline.py on each scenario
    python bench_ws_global_vs_localized.py score     # CSV + figure, global vs localized

    python bench_ws_global_vs_localized.py all --design oat --quick

Design of experiments (``--design``)
------------------------------------
Factors (edit ``FACTORS`` below to add levels):

* ``ws_truth``  : ``homogeneous`` (one WS set), ``per_horizon`` (depth trend), ``xy_linear``
                  (m and sigma_s drift smoothly in x/y), ``xy_random`` (m, sigma_s patchy,
                  random blobs of correlation length ``corr_m``).
* ``n_layers``  : 2, 3, 4 horizons.
* ``n_probes``  : TDR positions per layer.

``oat``  (default): one factor at a time around ``BASELINE`` (few runs).
``full``          : cartesian product (can be many long ensemble runs: check ``generate`` first).

Expectation to test: on ``homogeneous`` / ``per_horizon`` truth localized should be ~ global
(a small cost from fewer effective probes per fit); on ``xy_*`` truth localized should win
away from the probes up to roughly the correlation length, and lose where the nearest probe
is beyond 2c (it falls back to the global fit).

What ``score`` reports (per run, both maps vs the truth of ``--realization``)
-----------------------------------------------------------------------------
RMSE and bias over everything, per horizon, NEAR probes (<= c from the nearest probe) and
FAR from probes (> 2c, i.e. where localized has fallen back to the global fit), the WS
parameter error (when the truth parameter fields exist), K / K_global and the bandwidth.
Nothing is modified in the project; only ``<run>/output_model/*.npy`` is read.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG = os.path.join(HERE, "ensemble_scenarios.json")
OUT_ROOT = os.path.join("mfc2_outputs", "ws_bench")
PARAMS = ("a", "m", "sigma_w", "sigma_s")

# ------------------------------------------------------------------ design
FACTORS = {
    "ws_truth": ["homogeneous", "per_horizon", "xy_linear", "xy_random"],
    "n_layers": [2, 3, 4],
    "n_probes": [2, 4, 8],
}
BASELINE = {"ws_truth": "xy_random", "n_layers": 4, "n_probes": 4}
CORR_M = 4.0                               # xy_random patch size [m]
BANDWIDTH_SWEEP = [2.0, 5.0, 10.0, 20.0, 40.0]   # leave-one-probe-out picks c per run

# parent scenario each ws_truth level extends (already in ensemble_scenarios.json)
PARENT = {"homogeneous": "2d", "per_horizon": "2d_horizon_ws",
          "xy_linear": "2d_xy_waxman", "xy_random": "2d_xy_waxman"}


def _depths(n_layers: int) -> list[float]:
    return {2: [0.4], 3: [0.3, 0.7], 4: [0.3, 0.6, 1.0]}[n_layers]


def _horizon_p0(n_layers: int) -> list[list[float]]:
    """Depth trend as in '2d_horizon_ws': m rises, sigma_s falls with depth."""
    m = np.linspace(2.2, 3.4, n_layers)
    ss = np.linspace(150.0, 30.0, n_layers)
    return [[0.5, float(mi), 500.0, float(si)] for mi, si in zip(m, ss)]


def scenario_dict(ws_truth: str, n_layers: int, n_probes: int, *, corr_m: float = CORR_M,
                  seed: int = 21, sweep=BANDWIDTH_SWEEP) -> dict:
    """Overrides on top of the parent scenario (``extends``)."""
    s = {
        "extends": PARENT[ws_truth],
        "description": f"ws-bench: truth={ws_truth}, {n_layers} layers, {n_probes} probes/layer",
        "depths0": _depths(n_layers),
        "cond_ranges": [[0.01, 60.0]] * n_layers,
        "num_swc_positions": n_probes,
        "ws_localized_bandwidth_sweep": list(sweep),
    }
    if ws_truth == "per_horizon":
        s["lab_p0"] = _horizon_p0(n_layers)
    if ws_truth.startswith("xy_"):
        s["lab_p0"] = [0.5, 3.0, 500.0, 100.0]
        s["ws_xy_variation"] = True
        s["ws_xy_pattern"] = "linear" if ws_truth == "xy_linear" else "random"
        s["ws_xy_seed"] = seed
        if ws_truth == "xy_random":
            s["ws_xy_correlation_m"] = corr_m
    return s


def design(kind: str) -> list[dict]:
    if kind == "full":
        combos = itertools.product(*FACTORS.values())
        levels = [dict(zip(FACTORS, c)) for c in combos]
    else:                                              # one factor at a time
        levels, seen = [], set()
        for name, values in FACTORS.items():
            for v in values:
                lv = dict(BASELINE, **{name: v})
                key = tuple(lv.values())
                if key not in seen:
                    seen.add(key)
                    levels.append(lv)
    for lv in levels:
        lv["name"] = f"wsb_{lv['ws_truth']}_L{lv['n_layers']}_P{lv['n_probes']}"
    return levels


# ------------------------------------------------------------------ generate / run
def cmd_generate(args) -> str:
    base = json.load(open(BASE_CONFIG))
    scenarios = {k: v for k, v in base.items() if not k.startswith("_")}
    levels = design(args.design)
    for lv in levels:
        sc = scenario_dict(lv["ws_truth"], lv["n_layers"], lv["n_probes"], corr_m=args.corr_m)
        sc["output_dir"] = os.path.join(args.out_root, lv["name"])
        scenarios[lv["name"]] = sc
    path = os.path.join(args.out_root, "ws_bench_scenarios.json")
    os.makedirs(args.out_root, exist_ok=True)
    json.dump(scenarios, open(path, "w"), indent=1)
    json.dump(levels, open(os.path.join(args.out_root, "design.json"), "w"), indent=1)
    print(f"[generate] {len(levels)} scenarios -> {path}")
    try:                                               # fail now, not after hours of inversions
        import scenario_loader
        for lv in levels:
            scenario_loader.load_scenario(lv["name"], path)
        print("[generate] all scenarios validate with scenario_loader")
    except ImportError as exc:
        print(f"[generate][WARN] could not import scenario_loader to validate: {exc}")
    return path


def latest_run_dir(base: str) -> str | None:
    runs = sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)
                  and os.path.isdir(os.path.join(d, "output_model")))
    return runs[-1] if runs else None


def cmd_run(args) -> None:
    cfg = os.path.join(args.out_root, "ws_bench_scenarios.json")
    if not os.path.isfile(cfg):
        cfg = cmd_generate(args)
    levels = json.load(open(os.path.join(args.out_root, "design.json")))
    for lv in levels:
        if args.only and not any(o in lv["name"] for o in args.only):
            continue
        if not args.rerun and latest_run_dir(os.path.join(args.out_root, lv["name"])):
            print(f"[run] {lv['name']}: already has a run (use --rerun to redo)")
            continue
        cmd = [sys.executable, os.path.join(HERE, "run_pipeline.py"), "--data-source", "synthetic",
               "--scenario", lv["name"], "--config", cfg, "--mode", "2d",
               "--ws-map-calibration", "localized", "--k-selection", args.k_selection]
        if args.quick:
            cmd.append("--quick")
        print("[run]", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=False, cwd=HERE)


# ------------------------------------------------------------------ score
def _load(om: str, name: str):
    p = os.path.join(om, name)
    return np.load(p) if os.path.isfile(p) else None


def _metrics(err: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    e = err if mask is None else err[..., mask, :] if err.ndim == 3 else err
    e = e[np.isfinite(e)]
    return (float(np.sqrt(np.mean(e ** 2))), float(np.mean(e))) if e.size else (np.nan, np.nan)


def score_run(run_dir: str, realization: int = 0) -> dict | None:
    om = os.path.join(run_dir, "output_model")
    truth = _load(om, "synthetic_time_lapse_theta.npy")
    t_loc = _load(om, "theta_calc.npy")
    t_glob = _load(om, "theta_calc_global.npy")
    x, y = _load(om, "synthetic_x_coords.npy"), _load(om, "synthetic_y_coords.npy")
    if truth is None or t_loc is None or x is None:
        print(f"[score] {run_dir}: missing truth / theta_calc, skipped")
        return None
    if t_glob is None:
        print(f"[score] {run_dir}: no theta_calc_global.npy (localized maps were not used: "
              "no bandwidth or the fit failed), skipped")
        return None
    truth = truth[min(realization, truth.shape[0] - 1)]
    nl = min(truth.shape[-1], t_loc.shape[-1], t_glob.shape[-1])
    nt = min(truth.shape[0], t_loc.shape[0], t_glob.shape[0])
    truth, t_loc, t_glob = truth[:nt, :, :nl], t_loc[:nt, :, :nl], t_glob[:nt, :, :nl]

    # distance of every EM point to the nearest TDR probe
    pos = _load(om, "tdr_train_positions.npy")
    pos = np.unique(pos.astype(int)) if pos is not None else np.array([], dtype=int)
    if pos.size:
        d = np.sqrt((x[:, None] - x[pos][None, :]) ** 2 + (y[:, None] - y[pos][None, :]) ** 2).min(1)
    else:
        d = np.full(len(x), np.nan)
    bw = _load(om, "ws_localized_bandwidth.npy")
    c = float(bw) if bw is not None else np.nan
    near, far = d <= c, d > 2 * c

    row = {"run": os.path.basename(run_dir), "bandwidth": c,
           "K": int(_load(om, "K.npy")), "K_global": int(_load(om, "K_global.npy"))
           if _load(om, "K_global.npy") is not None else int(_load(om, "K.npy")),
           "n_probes_loc": int(pos.size)}
    fb = _load(om, "ws_localized_fallback.npy")
    row["fallback_frac"] = float(np.mean(fb)) if fb is not None else np.nan
    for tag, th in (("global", t_glob), ("local", t_loc)):
        err = th - truth
        row[f"rmse_{tag}"], row[f"bias_{tag}"] = _metrics(err)
        row[f"rmse_near_{tag}"], _ = _metrics(err, near) if near.any() else (np.nan, np.nan)
        row[f"rmse_far_{tag}"], _ = _metrics(err, far) if far.any() else (np.nan, np.nan)
        for l in range(nl):
            row[f"rmse_L{l}_{tag}"] = _metrics(err[..., l:l + 1])[0]
    row["gain"] = row["rmse_global"] - row["rmse_local"]          # > 0: localized better
    row["gain_far"] = (row["rmse_far_global"] - row["rmse_far_local"]
                       if np.isfinite(row["rmse_far_global"]) else np.nan)

    # WS parameter error (needs truth parameter fields, i.e. a ws_xy_variation run)
    p_loc, p_glob = _load(om, "ws_localized_params.npy"), _load(om, "ws_params.npy")
    for i, name in enumerate(PARAMS):
        tp = _load(om, f"synthetic_ws_truth_{name}.npy")
        if tp is None or p_loc is None or p_glob is None:
            continue
        tp = tp[min(realization, tp.shape[0] - 1)] if tp.ndim == 3 else tp          # (n_pts, n_layers)
        n = min(tp.shape[-1], p_loc.shape[1], p_glob.shape[0])
        loc = p_loc[i, :n, :].T                                                     # (n_pts, n_layers)
        glob = np.broadcast_to(p_glob[:n, i][None, :], loc.shape)
        row[f"param_{name}_rmse_global"] = float(np.sqrt(np.nanmean((glob - tp[:, :n]) ** 2)))
        row[f"param_{name}_rmse_local"] = float(np.sqrt(np.nanmean((loc - tp[:, :n]) ** 2)))
    return row


def cmd_score(args) -> None:
    import pandas as pd
    levels = {lv["name"]: lv for lv in json.load(open(os.path.join(args.out_root, "design.json")))}
    rows = []
    for name, lv in levels.items():
        run = latest_run_dir(os.path.join(args.out_root, name))
        if run is None:
            print(f"[score] {name}: no run found")
            continue
        r = score_run(run, args.realization)
        if r:
            rows.append({**{k: lv[k] for k in FACTORS}, "scenario": name, **r})
    if not rows:
        print("[score] nothing to score")
        return
    df = pd.DataFrame(rows)
    csv = os.path.join(args.out_root, "ws_global_vs_localized.csv")
    df.to_csv(csv, index=False)
    cols = ["scenario", "bandwidth", "K", "K_global", "rmse_global", "rmse_local", "gain",
            "rmse_far_global", "rmse_far_local", "fallback_frac"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(df[cols].to_string(index=False))
    print(f"\n[score] localized better in {(df['gain'] > 0).sum()}/{len(df)} runs; "
          f"mean gain {df['gain'].mean():+.4f} m3/m3 (>0 = localized better). CSV -> {csv}")
    _plot(df, os.path.join(args.out_root, "ws_global_vs_localized.png"))


def _plot(df, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5), dpi=130)
    lim = float(np.nanmax(df[["rmse_global", "rmse_local"]].values)) * 1.1
    a.plot([0, lim], [0, lim], "k--", lw=0.8)
    for tr, g in df.groupby("ws_truth"):
        a.scatter(g["rmse_global"], g["rmse_local"], s=30 + 12 * g["n_layers"], label=tr)
    a.set(xlabel="RMSE global map [m³/m³]", ylabel="RMSE localized map [m³/m³]",
          title="Below the line = localized better", xlim=(0, lim), ylim=(0, lim))
    a.legend(title="WS truth")
    order = df.sort_values(["ws_truth", "n_layers", "n_probes"])
    xs = np.arange(len(order))
    b.bar(xs - 0.2, order["gain"], 0.4, label="all points")
    b.bar(xs + 0.2, order["gain_far"].fillna(0), 0.4, label="far from probes (> 2c)")
    b.axhline(0, color="k", lw=0.8)
    b.set_xticks(xs)
    b.set_xticklabels(order["scenario"].str.replace("wsb_", ""), rotation=75, ha="right", fontsize=7)
    b.set(ylabel="RMSE gain = global − localized [m³/m³]", title=">0: localized better")
    b.legend()
    fig.tight_layout()
    fig.savefig(path)
    print(f"[score] figure -> {path}")


# ------------------------------------------------------------------ CLI
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("command", choices=["generate", "run", "score", "all"])
    p.add_argument("--design", choices=["oat", "full"], default="oat")
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--corr-m", type=float, default=CORR_M, help="xy_random correlation length [m]")
    p.add_argument("--k-selection", choices=["global", "localized-loo"], default="global")
    p.add_argument("--only", nargs="*", help="run only scenarios whose name contains one of these")
    p.add_argument("--quick", action="store_true", help="pass --quick to run_pipeline.py")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the run commands only")
    p.add_argument("--realization", type=int, default=0)
    a = p.parse_args()
    if a.command in ("generate", "all"):
        cmd_generate(a)
    if a.command in ("run", "all"):
        cmd_run(a)
    if a.command in ("score", "all"):
        cmd_score(a)


if __name__ == "__main__":
    main()
