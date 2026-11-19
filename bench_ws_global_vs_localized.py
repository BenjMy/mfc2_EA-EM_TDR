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
CORR_M = 4.0                               # xy_random patch size [m] (default level)
SEED = 21                                  # default ws_xy_seed (truth realization of the WS field)
DIST_BINS = [0.0, 2.0, 5.0, 10.0, 20.0, np.inf]   # distance-to-nearest-probe bins [m]
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
                  seed: int = SEED, sweep=BANDWIDTH_SWEEP, bandwidth: float | None = None) -> dict:
    """Overrides on top of the parent scenario (``extends``).

    ``bandwidth`` not None: FIXED localized bandwidth [m] (``ws_localized_bandwidth``, no
    cross-validated sweep), so a large c cannot hide behind the global fit and a small c is
    not half fallback. None: the leave-one-probe-out sweep ``sweep`` picks c per run.
    """
    s = {
        "extends": PARENT[ws_truth],
        "description": f"ws-bench: truth={ws_truth}, {n_layers} layers, {n_probes} probes/layer",
        "depths0": _depths(n_layers),
        "cond_ranges": [[0.01, 60.0]] * n_layers,
        "num_swc_positions": n_probes,
    }
    if bandwidth is None:
        s["ws_localized_bandwidth_sweep"] = list(sweep)
    else:
        s["ws_localized_bandwidth"] = float(bandwidth)
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


def _fmt(v: float) -> str:
    return f"{v:g}".replace(".", "p")


def design(kind: str, *, seeds=(SEED,), corr_ms=(CORR_M,), bandwidths=(None,)) -> list[dict]:
    """Factor levels x replicates (seeds) x correlation lengths (xy_random only) x bandwidths.

    Names stay ``wsb_<truth>_L<l>_P<p>`` for the defaults (existing runs are still found);
    a suffix ``_C<corr>`` / ``_bw<c>`` / ``_s<seed>`` is added only for non-default values.
    """
    if kind == "full":
        combos = itertools.product(*FACTORS.values())
        base_levels = [dict(zip(FACTORS, c)) for c in combos]
    else:                                              # one factor at a time
        base_levels, seen = [], set()
        for name, values in FACTORS.items():
            for v in values:
                lv = dict(BASELINE, **{name: v})
                key = tuple(lv.values())
                if key not in seen:
                    seen.add(key)
                    base_levels.append(lv)
    levels = []
    for lv in base_levels:
        corrs = corr_ms if lv["ws_truth"] == "xy_random" else (CORR_M,)   # c only matters there
        # homogeneous / per_horizon truths have no random field: one replicate is enough
        seed_list = seeds if lv["ws_truth"] in ("xy_linear", "xy_random") else seeds[:1]
        for corr, bw, seed in itertools.product(corrs, bandwidths, seed_list):
            n = dict(lv, corr_m=float(corr), bandwidth=bw, seed=int(seed))
            name = f"wsb_{n['ws_truth']}_L{n['n_layers']}_P{n['n_probes']}"
            if n["ws_truth"] == "xy_random" and corr != CORR_M:
                name += f"_C{_fmt(corr)}"
            if bw is not None:
                name += f"_bw{_fmt(bw)}"
            if seed != SEED:
                name += f"_s{seed}"
            n["name"] = name
            levels.append(n)
    return levels


# ------------------------------------------------------------------ generate / run
def cmd_generate(args) -> str:
    base = json.load(open(BASE_CONFIG))
    scenarios = {k: v for k, v in base.items() if not k.startswith("_")}
    levels = design(args.design, seeds=args.seeds, corr_ms=args.corr_m, bandwidths=args.bandwidth)
    for lv in levels:
        sc = scenario_dict(lv["ws_truth"], lv["n_layers"], lv["n_probes"], corr_m=lv["corr_m"],
                           seed=lv["seed"], bandwidth=lv["bandwidth"])
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


def score_run(run_dir: str, realization: int = 0, bins=DIST_BINS) -> tuple[dict, list[dict]] | None:
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

    # error by distance to the nearest probe: tests "localized wins away from the probes" directly
    bin_rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d >= lo) & (d < hi) if np.isfinite(hi) else (d >= lo)
        if not m.any():
            continue
        row_b = {"run": row["run"], "bin_lo": lo, "bin_hi": hi, "n_points": int(m.sum()),
                 "frac_fallback_bin": float(np.mean(fb[:, m])) if fb is not None and fb.ndim == 2
                 and fb.shape[1] == len(x) else np.nan}
        for tag, th in (("global", t_glob), ("local", t_loc)):
            row_b[f"rmse_{tag}"], row_b[f"bias_{tag}"] = _metrics(th - truth, m)
        row_b["gain"] = row_b["rmse_global"] - row_b["rmse_local"]
        bin_rows.append(row_b)

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
    return row, bin_rows


CONFIG_KEYS = ["ws_truth", "n_layers", "n_probes", "corr_m", "bandwidth_fixed"]


def _bin_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g} m" if np.isfinite(hi) else f">{lo:g} m"


def cmd_score(args) -> None:
    import pandas as pd
    levels = {lv["name"]: lv for lv in json.load(open(os.path.join(args.out_root, "design.json")))}
    rows, bin_all = [], []
    for name, lv in levels.items():
        run = latest_run_dir(os.path.join(args.out_root, name))
        if run is None:
            print(f"[score] {name}: no run found")
            continue
        res = score_run(run, args.realization, bins=args.bins)
        if res:
            r, brows = res
            # designs written before the replicate options have no corr_m / seed / bandwidth keys
            meta = {"corr_m": lv.get("corr_m", CORR_M), "seed": lv.get("seed", SEED),
                    "bandwidth_fixed": lv.get("bandwidth")}
            rows.append({**{k: lv[k] for k in FACTORS}, **meta, "scenario": name, **r})
            bin_all += [{"scenario": name, **{k: lv[k] for k in FACTORS}, **meta, **b} for b in brows]
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

    # replicates (ws_xy_seed): mean and standard deviation per configuration
    df["_cfg"] = df[CONFIG_KEYS].astype(str).agg(" | ".join, axis=1)
    agg = (df.groupby(CONFIG_KEYS, dropna=False)
             .agg(n_rep=("gain", "size"), rmse_global=("rmse_global", "mean"),
                  rmse_local=("rmse_local", "mean"), gain=("gain", "mean"),
                  gain_sd=("gain", "std"), gain_far=("gain_far", "mean"),
                  gain_far_sd=("gain_far", "std"), bandwidth=("bandwidth", "mean"),
                  fallback_frac=("fallback_frac", "mean")).reset_index())
    agg_csv = os.path.join(args.out_root, "ws_global_vs_localized_agg.csv")
    agg.to_csv(agg_csv, index=False)
    print(f"[score] per-configuration mean/sd over replicates -> {agg_csv}")
    if (agg["n_rep"] < 2).all():
        print("[score][NOTE] one replicate per configuration: no error bars. "
              "Use --seeds 21 22 23 ... on generate/run.")

    if bin_all:
        dfb = pd.DataFrame(bin_all)
        dfb["bin"] = [_bin_label(lo, hi) for lo, hi in zip(dfb["bin_lo"], dfb["bin_hi"])]
        bcsv = os.path.join(args.out_root, "ws_error_by_distance.csv")
        dfb.to_csv(bcsv, index=False)
        print(f"[score] error by distance to the nearest probe -> {bcsv}")
        _plot_distance(dfb, os.path.join(args.out_root, "ws_error_by_distance.png"))
    _plot(df, os.path.join(args.out_root, "ws_global_vs_localized.png"))


def _plot_distance(dfb, path: str) -> None:
    """RMSE (global and localized) per distance bin, mean +- sd over replicates, one panel per truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["ws_truth", "n_layers", "n_probes", "corr_m", "bandwidth_fixed"]
    dfb = dfb.copy()
    dfb["cfg"] = [f"{r.ws_truth} L{r.n_layers} P{r.n_probes} corr={r.corr_m:g} m, "
                  + ("c=sweep" if r.bandwidth_fixed is None or r.bandwidth_fixed != r.bandwidth_fixed
                     else f"c={r.bandwidth_fixed:g} m") for r in dfb.itertuples()]
    cfgs = list(dict.fromkeys(dfb["cfg"]))
    ncol = min(3, len(cfgs))
    nrow = int(np.ceil(len(cfgs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow), dpi=130, squeeze=False)
    for ax, cfg in zip(axes.ravel(), cfgs):
        g = dfb[dfb["cfg"] == cfg]
        order = g.sort_values("bin_lo")["bin"].drop_duplicates().tolist()
        xs = np.arange(len(order))
        for tag, col, off in (("global", "C0", -0.08), ("local", "C3", 0.08)):
            grp = g.groupby("bin")[f"rmse_{tag}"]
            mu = grp.mean().reindex(order)
            sd = grp.std().reindex(order).fillna(0.0)          # one replicate: no bar
            ax.errorbar(xs + off, mu, yerr=sd, marker="o", ms=4, capsize=3, color=col,
                        label="localized" if tag == "local" else "global")
        ax.set_xticks(xs)
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=7)
        ax.set_title(cfg, fontsize=7)
        ax.set_ylabel("RMSE [m³/m³]", fontsize=8)
        ax.set_xlabel("distance to the nearest probe", fontsize=8)
    for ax in axes.ravel()[len(cfgs):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("Error by distance to the nearest TDR probe (mean ± sd over replicates)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    print(f"[score] figure -> {path}")


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
    p.add_argument("--corr-m", type=float, nargs="+", default=[CORR_M],
                   help="xy_random correlation length(s) [m]; several values = one run each, "
                        "e.g. --corr-m 4 10 20")
    p.add_argument("--bandwidth", type=float, nargs="+", default=None,
                   help="FIXED localized bandwidth(s) [m], one run each, instead of the "
                        "leave-one-probe-out choice, e.g. --bandwidth 5 10")
    p.add_argument("--seeds", type=int, nargs="+", default=[SEED],
                   help="ws_xy_seed replicates of the xy_* truth (error bars), e.g. --seeds 21 22 23 24 25")
    p.add_argument("--bins", type=float, nargs="+", default=None,
                   help="distance-to-nearest-probe bin edges [m] for the error-by-distance score "
                        f"(default {[b for b in DIST_BINS[:-1]]} + inf)")
    p.add_argument("--k-selection", choices=["global", "localized-loo"], default="global")
    p.add_argument("--only", nargs="*", help="run only scenarios whose name contains one of these")
    p.add_argument("--quick", action="store_true", help="pass --quick to run_pipeline.py")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the run commands only")
    p.add_argument("--realization", type=int, default=0)
    a = p.parse_args()
    a.bandwidth = a.bandwidth or [None]
    edges = sorted(set(a.bins)) if a.bins else list(DIST_BINS[:-1])
    a.bins = [float(e) for e in edges] + [np.inf]
    if any(c <= 0 for c in a.corr_m) or any(b is not None and b <= 0 for b in a.bandwidth):
        p.error("--corr-m and --bandwidth must be positive")
    if a.command in ("generate", "all"):
        cmd_generate(a)
    if a.command in ("run", "all"):
        cmd_run(a)
    if a.command in ("score", "all"):
        cmd_score(a)


if __name__ == "__main__":
    main()
