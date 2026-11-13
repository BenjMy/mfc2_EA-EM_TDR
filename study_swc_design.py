#!/usr/bin/env python3
"""
study_swc_design.py -- how well does the EC -> SWC (EA-EM) algorithm recover the
soil water content as a function of the NUMBER and the DISTRIBUTION of the SWC
(TDR) probes used for the Waxman-Smits calibration?

Synthetic 1-D transect (10 survey points, 2 m spacing).  Factors:

  n      number of SWC probe positions            1, 2, 3, 5, 10
  dist   distribution of the probe positions      spread | clustered | random
  smc    truth soil moisture (theta)              homo   | hetero
  ws     truth Waxman-Smits parameters            homo   | hetero
  k      number of FREE WS parameters             1..4  (freed: sigma_s, m, sigma_w, a)
  reg    EMagPy inversion regularisation          noreg | beta | gamma | both

Meaning of the factors
----------------------
* smc  homo   : ``homogeneous = true``  (theta spatially uniform; temporal trend kept)
        hetero: the scenario's default spatial gradient (``--smc-full-range`` makes it
                a full-range gradient)
* ws   homo   : ``ws_xy_variation = false``  (true a, m, sigma_w, sigma_s = 0.5, 3, 500, 100
                everywhere)
        hetero: m and sigma_s vary along x (ranges of the base scenario) AND a, sigma_w
                change below horizon 1 (``sweeps.HETERO_LAYER_PARAMS``)
* reg  beta  : lateral/spatial smoothing of the EMagPy inversion (``param_grid.beta``)
        gamma: temporal smoothing (``gamma``; the synthetic data have 3 epochs)
* dist spread   : positions evenly spread over the transect (x = 0 ... 18 m)
        clustered: the n first survey points (x = 0, 2, 4, ...), all at one end
        random   : x = 0 plus n-1 random survey points (seeded)
  The pipeline re-anchors a ``--probe-xyz`` file at the survey origin (see
  run_pipeline.py, "--probe-xyz files are written for REAL field deployments"), so every
  design contains x = 0.  n = 1 (single probe at x = 0) and n = 10 (all points) are the
  same for the three distributions; they are run once and drawn in every panel.
* The calibration start values ``lab_p0`` are shifted by ``--p0-bias`` (default +20 %) for
  the free m, sigma_w, sigma_s.  With a homogeneous WS truth the repo default start
  (0.5, 3, 500, 100) IS the truth, which would make K meaningless; use ``--p0-bias 0``
  to get the repo convention back.

Design presets (``--preset``; levels can be restricted with --n-swc, --dist, ...)
  smoke  2 truths x n {1,5} x K {1,4}                                       (8 runs)
  oat    one factor at a time around the reference (hetero/hetero, noreg,
         spread, n = 5, K = 4)                                              (~16 runs)
  main   truth(4) x K(4) x n(5)  +  distribution at n = 5  +  reg at n = 5  (~160 runs)
  full   full factorial                                                      (~900 runs)

A synthetic run always repeats Step 1 (the ensemble cache is real-data only), so every
run costs a full pipeline run (~3-5 min; ``--fast`` = ``--quick`` ~ 3 min on one core).

Usage
-----
    python study_swc_design.py --dry-run                       # plan + validate, run nothing
    python study_swc_design.py --preset smoke --fast           # smoke test of the whole chain
    python study_swc_design.py --preset oat                    # default, ~1-2 h
    python study_swc_design.py --preset main --keep-going --resume
    python study_swc_design.py --analyze-only                  # re-collect + re-plot only
    python study_swc_design.py --smc homo hetero --ws hetero --free-k 2 4 --n-swc 1 3 5 10
    python study_swc_design.py --preset oat -- --em-footprint-diagnostic   # after '--' -> run_pipeline.sh

Outputs (``--out-dir``, default comparison_output/swc_design_study/)
    study_scenarios.json   generated scenarios (each ``extends`` one parent)
    probes/*.csv           explicit probe layouts passed with --probe-xyz
    study_summary.csv      one row per run (design factors + metrics vs truth)
    study_report.md        tables
    residuals/*.npz        theta_calc - theta_true per run (for the distribution plots)
    fig_*.png              comparison plots
Run folders: mfc2_outputs/swc_design_study/<scenario>/<NNNN>_<scenario>/
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np

import geometry_utils
import scenario_loader as sl
import sweeps
from run_real_study import ws_freedom

# --------------------------------------------------------------------------
# Study definition (edit here)
# --------------------------------------------------------------------------
BASE_CONFIG = "ensemble_scenarios.json"
BASE_SCENARIO = "1d_full_depth_ref"      # probes in every horizon; "1d_surface_tdr" = surface only
OUT_DIR = os.path.join("comparison_output", "swc_design_study")
RUNS_DIR = os.path.join("mfc2_outputs", "swc_design_study")
RUN_LOG = os.path.join("mfc2_outputs", "run_log.csv")

SURVEY_N = 10            # survey points of the synthetic 1-D transect
SURVEY_DX = 2.0          # spacing [m]

LEVELS = {
    "n": (1, 2, 3, 5, 10),
    "dist": ("spread", "clustered", "random"),
    "smc": ("homo", "hetero"),
    "ws": ("homo", "hetero"),
    "k": (1, 2, 3, 4),
    "reg": ("noreg", "beta", "gamma", "both"),
}
REFERENCE = {"n": 5, "dist": "spread", "smc": "hetero", "ws": "hetero", "k": 4, "reg": "noreg"}
FACTORS = ("smc", "ws", "reg", "dist", "n", "k")

BETA_VALUE = 0.5         # lateral smoothing used when reg in (beta, both)
GAMMA_VALUE = 5.0        # temporal smoothing used when reg in (gamma, both)
P0_BIAS = 0.2            # relative offset of the calibration start of free m, sigma_w, sigma_s
RANDOM_SEED = 7

SMC_FIELDS = {"homo": {"homogeneous": True}, "hetero": {"homogeneous": False}}
WS_FIELDS = {
    "homo": {"ws_xy_variation": False},
    "hetero": {"ws_xy_variation": True, "ws_layer_params": sweeps.HETERO_LAYER_PARAMS},
}
PARENT_NAME = "swcstudy_parent"
PREFIX = "swcstudy_"

SUMMARY_COLUMNS = [
    "scenario", *FACTORS, "n_probes", "pos_sig", "status", "quality_level",
    "final_rmse", "cv_mean", "runtime_s",
    "rmse_truth", "mae_truth", "bias_truth", "r2_truth",
    "rmse_calibrated_h", "rmse_uncalibrated_h", "rmse_near_probe", "rmse_far_probe",
    "rmse_by_layer", "run_dir", "error",
]


# --------------------------------------------------------------------------
# Design (pure helpers: no emagpy, no data -> unit-tested)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Design:
    smc: str
    ws: str
    reg: str
    dist: str
    n: int
    k: int

    @property
    def name(self) -> str:
        return f"{PREFIX}smc{self.smc}_ws{self.ws}_{self.reg}_{self.dist}_n{self.n}_free{self.k}"


def parse_name(name: str) -> Design | None:
    """Inverse of ``Design.name`` (None when ``name`` is not a study scenario)."""
    if not name.startswith(PREFIX):
        return None
    try:
        smc, ws, reg, dist, n, k = name[len(PREFIX):].split("_")
        return Design(smc[3:], ws[2:], reg, dist, int(n[1:]), int(k[4:]))
    except (ValueError, IndexError):
        return None


def probe_positions(dist: str, n: int, survey_x, seed: int = RANDOM_SEED) -> list[float]:
    """x positions [m] of ``n`` probes among the survey points (always includes x = 0)."""
    xs = np.asarray(survey_x, dtype=float)
    m = len(xs)
    n = int(min(max(n, 1), m))
    if dist == "spread":
        idx = np.unique(np.round(np.linspace(0, m - 1, n)).astype(int))
    elif dist == "clustered":
        idx = np.arange(n)
    elif dist == "random":
        rng = np.random.default_rng(seed + n)
        rest = rng.choice(np.arange(1, m), size=n - 1, replace=False) if n > 1 else []
        idx = np.sort(np.concatenate([[0], np.asarray(rest, dtype=int)]))
    else:
        raise ValueError(f"unknown distribution {dist!r}; choose from {LEVELS['dist']}")
    return [float(v) for v in xs[idx]]


def position_signature(xs) -> str:
    return "-".join(f"{v:g}" for v in xs)


def survey_x(n: int = SURVEY_N, dx: float = SURVEY_DX) -> np.ndarray:
    return np.arange(n, dtype=float) * dx


def preset_blocks(preset: str, levels: dict) -> list[dict]:
    """A preset is a list of blocks; each block maps factor -> levels (cartesian product)."""
    ref = {f: [REFERENCE[f]] if REFERENCE[f] in levels[f] else [levels[f][-1]] for f in FACTORS}
    if preset == "full":
        return [dict(levels)]
    if preset == "oat":
        return [{**ref, f: list(levels[f])} for f in FACTORS]
    if preset == "main":
        free = {"smc": list(levels["smc"]), "ws": list(levels["ws"]), "k": list(levels["k"])}
        return [{**ref, **free, "n": list(levels["n"])},
                {**ref, **free, "dist": list(levels["dist"])},
                {**ref, **free, "reg": list(levels["reg"])}]
    if preset == "smoke":
        two = lambda f: [levels[f][0], levels[f][-1]] if len(levels[f]) > 1 else list(levels[f])
        return [{**ref, "smc": two("smc"), "ws": ["hetero" if "hetero" in levels["ws"] else levels["ws"][0]],
                 "n": [1, 5] if {1, 5} <= set(levels["n"]) else two("n"), "k": two("k")}]
    raise ValueError(f"unknown preset {preset!r}")


def build_designs(preset: str, levels: dict, survey=None) -> list[Design]:
    """Unique designs of a preset; designs with identical probe positions are merged."""
    xs = survey_x() if survey is None else np.asarray(survey, dtype=float)
    seen: dict[tuple, Design] = {}
    for block in preset_blocks(preset, levels):
        for smc, ws, reg, dist, n, k in itertools.product(*(block[f] for f in FACTORS)):
            n_eff = min(int(n), len(xs))
            sig = position_signature(probe_positions(dist, n_eff, xs))
            key = (smc, ws, reg, n_eff, k, sig)
            # same positions under another distribution name (n = 1, n = all points): keep the first
            seen.setdefault(key, Design(smc, ws, reg, dist, n_eff, int(k)))
    order = {f: i for i, f in enumerate(LEVELS["dist"])}
    return sorted(seen.values(), key=lambda d: (d.smc, d.ws, d.reg, d.k, d.n, order.get(d.dist, 9)))


def reg_fields(reg: str, beta: float = BETA_VALUE, gamma: float = GAMMA_VALUE) -> dict:
    if reg not in LEVELS["reg"]:
        raise ValueError(f"unknown regularisation {reg!r}; choose from {LEVELS['reg']}")
    out = {"gamma": float(gamma) if reg in ("gamma", "both") else 0.0}
    if reg in ("beta", "both"):
        out["param_grid"] = {"beta": [float(beta)]}      # merged into the parent's param_grid
    return out


def biased_start(fields: dict, base_p0: list, k: int, bias: float) -> dict:
    """Shift the start of the free m, sigma_w, sigma_s by ``bias`` (relative); 'a' already starts at 1.0."""
    if not bias or k < 1:
        return fields
    idx = {"a": 0, "m": 1, "sigma_w": 2, "sigma_s": 3}
    p0 = list(fields.get("lab_p0", base_p0))
    for p in sweeps.FREE_ORDER[:k]:
        if p != "a":
            p0[idx[p]] = round(float(p0[idx[p]]) * (1.0 + bias), 6)
    return {**fields, "lab_p0": p0}


def probe_depths(parent: dict) -> list[float]:
    """Depths [m] of the probes: the scenario's ``swc_depths`` or the layer centres."""
    if parent.get("swc_depths"):
        return [float(d) for d in parent["swc_depths"]]
    return [float(d) for d in geometry_utils.layer_centers(parent["depths0"])]


def write_probe_csv(path: str, xs: list[float], depths: list[float]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["x", "y", "z", "label"])
        for i, x in enumerate(xs):
            for z in depths:
                w.writerow([f"{x:g}", "0", f"{z:g}", f"P{i}_{z:g}m"])
    return path


def build_study_config(parent_raw: dict, designs: list[Design], *, runs_dir: str = RUNS_DIR,
                       beta: float = BETA_VALUE, gamma: float = GAMMA_VALUE,
                       p0_bias: float = P0_BIAS, smc_full_range: bool = False) -> dict:
    """Scenario-file content: one parent (copy of the base) + one child per design."""
    parent = copy.deepcopy(parent_raw)
    parent["description"] = f"Parent of the SWC number/distribution study (see study_swc_design.py)."
    parent["output_dir"] = os.path.join(runs_dir, "parent")
    parent["cache_tag"] = PARENT_NAME
    base_p0 = list(parent.get("lab_p0") or [0.5, 3.0, 500.0, 100.0])
    cfg = {"_comment": ["GENERATED by study_swc_design.py -- do not edit, re-run the script.",
                        f"Every child extends '{PARENT_NAME}'; they differ by truth (smc, ws), "
                        "regularisation, WS freedom and (via --probe-xyz) the probe layout."],
           PARENT_NAME: parent}
    for d in designs:
        smc_fields = dict(SMC_FIELDS[d.smc])
        if d.smc == "hetero" and smc_full_range:
            smc_fields["full_range_gradient"] = True
        fields = {**smc_fields, **copy.deepcopy(WS_FIELDS[d.ws]), **reg_fields(d.reg, beta, gamma),
                  **ws_freedom(d.k, base_p0)}
        fields = biased_start(fields, base_p0, d.k, p0_bias)
        cfg[d.name] = {
            "extends": PARENT_NAME,
            "description": (f"SWC design study: smc={d.smc}, ws={d.ws}, reg={d.reg}, probes={d.dist} "
                            f"n={d.n}, K={d.k} free WS parameter(s) "
                            f"({', '.join(sweeps.FREE_ORDER[:d.k])})."),
            "output_dir": os.path.join(runs_dir, d.name),
            "cache_tag": d.name,
            "num_swc_positions": d.n,
            **fields,
        }
    return cfg


def pipeline_command(d: Design, config_path: str, probe_csv: str, mode: str, fast: bool,
                     extra: list[str]) -> list[str]:
    cmd = ["./run_pipeline.sh", "-s", d.name, "-c", config_path, "--mode", mode, "--probe-xyz", probe_csv]
    if fast:
        cmd.append("--quick")
    return cmd + list(extra)


# --------------------------------------------------------------------------
# Collecting results
# --------------------------------------------------------------------------
def _first_float(text) -> float:
    try:
        return float(str(text).split()[0])
    except (ValueError, IndexError):
        return float("nan")


def compute_metrics(run_dir: str, realization: int = 0, probe_x=None, residual_path: str | None = None) -> dict:
    """theta_calc - theta_true statistics of one synthetic run (empty dict when arrays are missing)."""
    om = os.path.join(run_dir, "output_model")
    p_calc, p_true = os.path.join(om, "theta_calc.npy"), os.path.join(om, "synthetic_time_lapse_theta.npy")
    if not (os.path.isfile(p_calc) and os.path.isfile(p_true)):
        return {}
    calc, truth_ens = np.load(p_calc), np.load(p_true)
    truth = truth_ens[realization if realization < truth_ens.shape[0] else 0]
    n_t, n_l = min(calc.shape[0], truth.shape[0]), min(calc.shape[-1], truth.shape[-1])
    if calc.shape[1] != truth.shape[1]:
        return {}
    resid = calc[:n_t, :, :n_l] - truth[:n_t, :, :n_l]
    rmse = lambda a: float(np.sqrt(np.mean(np.square(a)))) if np.size(a) else float("nan")
    ss_res, ss_tot = float(np.sum(resid ** 2)), float(np.sum((truth[:n_t, :, :n_l] - truth[:n_t, :, :n_l].mean()) ** 2))
    out = {
        "rmse_truth": rmse(resid), "mae_truth": float(np.mean(np.abs(resid))),
        "bias_truth": float(np.mean(resid)),
        "r2_truth": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse_by_layer": json.dumps([round(rmse(resid[:, :, l]), 6) for l in range(n_l)]),
    }
    p_cal = os.path.join(om, "is_calibrated.npy")
    if os.path.isfile(p_cal):
        cal = np.asarray(np.load(p_cal), dtype=bool)[:n_l]
        out["rmse_calibrated_h"] = rmse(resid[:, :, cal]) if cal.any() else float("nan")
        out["rmse_uncalibrated_h"] = rmse(resid[:, :, ~cal]) if (~cal).any() else float("nan")
    dist = None
    p_x = os.path.join(om, "synthetic_x_coords.npy")
    if probe_x is not None and os.path.isfile(p_x):
        x = np.load(p_x)
        px = np.asarray(probe_x, dtype=float) + float(x.min())     # pipeline re-anchors probes at the survey origin
        dist = np.min(np.abs(x[:, None] - px[None, :]), axis=1)
        near = dist <= 1e-6
        out["rmse_near_probe"] = rmse(resid[:, near, :]) if near.any() else float("nan")
        out["rmse_far_probe"] = rmse(resid[:, ~near, :]) if (~near).any() else float("nan")
    if residual_path:
        os.makedirs(os.path.dirname(residual_path) or ".", exist_ok=True)
        np.savez_compressed(residual_path, resid=resid, dist=np.full(resid.shape[1], np.nan) if dist is None else dist,
                            is_calibrated=np.load(p_cal) if os.path.isfile(p_cal) else np.array([]))
    return out


def summary_row(d: Design, log_row: dict | None, out_dir: str, survey, realization: int = 0) -> dict:
    xs = probe_positions(d.dist, d.n, survey)
    row = {"scenario": d.name, **{f: getattr(d, f) for f in FACTORS},
           "n_probes": len(xs), "pos_sig": position_signature(xs)}
    if not log_row:
        row["status"] = "missing"
        return row
    run_dir = log_row.get("output_dir", "")
    row.update(status=log_row.get("status", ""), quality_level=log_row.get("quality_level", ""),
               final_rmse=_first_float(log_row.get("final_rmse")), cv_mean=_first_float(log_row.get("cv_mean")),
               runtime_s=_first_float(log_row.get("runtime_s")), run_dir=run_dir, error=log_row.get("error", ""))
    if row["status"] not in ("FAIL", "ERROR", "FAILED"):
        row.update(compute_metrics(run_dir, realization, xs,
                                   os.path.join(out_dir, "residuals", d.name + ".npz")))
    return row


def latest_rows_by_scenario(log_path: str = RUN_LOG) -> dict[str, dict]:
    from run_log import RunLog
    out = {}
    for r in RunLog(log_path).rows():
        if r.get("scenario", "").startswith(PREFIX):
            out[r["scenario"]] = r
    return out


def write_summary(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in SUMMARY_COLUMNS})


# --------------------------------------------------------------------------
# Plots (pandas / matplotlib only; every plot tolerates missing levels)
# --------------------------------------------------------------------------
_DIST_COLORS = {"spread": "#1b9e77", "clustered": "#d95f02", "random": "#7570b3"}
_REG_COLORS = {"noreg": "#555555", "beta": "#e6ab02", "gamma": "#66a61e", "both": "#e7298a"}
_TRUTH_ORDER = [("homo", "homo"), ("homo", "hetero"), ("hetero", "homo"), ("hetero", "hetero")]


def load_summary(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    for c in ("n", "k", "n_probes"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["rmse_truth"] = pd.to_numeric(df["rmse_truth"], errors="coerce")
    return df


def _truth_label(smc: str, ws: str) -> str:
    return f"SMC {smc} / WS {ws}"


def _truths(df) -> list[tuple[str, str]]:
    have = set(zip(df["smc"], df["ws"]))
    return [t for t in _TRUTH_ORDER if t in have]


def expand_degenerate(df, n_survey: int = SURVEY_N):
    """Copy the n = 1 and n = n_survey rows (identical for all distributions) to every distribution."""
    import pandas as pd
    extra = []
    for dist in df["dist"].unique():
        for n_deg in (1, n_survey):
            have = df[(df["dist"] == dist) & (df["n"] == n_deg)]
            keys = set(zip(have["smc"], have["ws"], have["reg"], have["k"]))
            src = df[(df["n"] == n_deg) & (df["dist"] != dist)]
            for _, r in src.iterrows():
                if (r["smc"], r["ws"], r["reg"], r["k"]) not in keys:
                    r = r.copy()
                    r["dist"] = dist
                    extra.append(r)
                    keys.add((r["smc"], r["ws"], r["reg"], r["k"]))
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True) if extra else df


def _ok(df):
    return df[df["rmse_truth"].notna()]


def _empty(ax, text="no data"):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="0.5")
    ax.set_xticks([]); ax.set_yticks([])


def _save(fig, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _ref_reg(df, reg_ref: str) -> str:
    regs = list(df["reg"].unique())
    return reg_ref if reg_ref in regs else regs[0]


def plot_rmse_vs_nswc(df, out_dir, reg_ref="noreg"):
    """RMSE vs number of probes; lines = distribution; rows = truth; columns = K."""
    import matplotlib.pyplot as plt
    d = _ok(expand_degenerate(df)); d = d[d["reg"] == _ref_reg(d, reg_ref)]
    truths, ks = _truths(d), sorted(d["k"].unique())
    if d.empty:
        return None
    fig, axes = plt.subplots(len(truths), len(ks), figsize=(3.4 * len(ks), 2.8 * len(truths)),
                             sharex=True, sharey=True, squeeze=False)
    for i, (smc, ws) in enumerate(truths):
        for j, k in enumerate(ks):
            ax, sub = axes[i, j], d[(d["smc"] == smc) & (d["ws"] == ws) & (d["k"] == k)]
            if sub.empty:
                _empty(ax); continue
            for dist, g in sub.groupby("dist"):
                g = g.sort_values("n")
                ax.plot(g["n"], g["rmse_truth"], "o-", color=_DIST_COLORS.get(dist), label=dist, ms=4)
            ax.grid(alpha=0.3)
            if i == 0: ax.set_title(f"K = {k} free WS")
            if j == 0: ax.set_ylabel(f"{_truth_label(smc, ws)}\nRMSE θ [m³/m³]", fontsize=8)
            if i == len(truths) - 1: ax.set_xlabel("number of SWC positions")
    axes[0, 0].legend(fontsize=7, title="distribution", title_fontsize=7)
    fig.suptitle("EC→SWC error vs number / distribution of SWC probes", y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rmse_vs_nswc.png")


def plot_rmse_vs_k(df, out_dir, reg_ref="noreg"):
    """RMSE vs number of free WS parameters; lines = number of probes; rows = truth; columns = distribution."""
    import matplotlib.pyplot as plt
    d = _ok(expand_degenerate(df)); d = d[d["reg"] == _ref_reg(d, reg_ref)]
    truths, dists = _truths(d), [x for x in LEVELS["dist"] if x in set(d["dist"])]
    if d.empty:
        return None
    fig, axes = plt.subplots(len(truths), len(dists), figsize=(3.6 * len(dists), 2.8 * len(truths)),
                             sharex=True, sharey=True, squeeze=False)
    ns = sorted(d["n"].unique()); cmap = plt.get_cmap("viridis")
    for i, (smc, ws) in enumerate(truths):
        for j, dist in enumerate(dists):
            ax, sub = axes[i, j], d[(d["smc"] == smc) & (d["ws"] == ws) & (d["dist"] == dist)]
            if sub.empty:
                _empty(ax); continue
            for c, n in enumerate(ns):
                g = sub[sub["n"] == n].sort_values("k")
                if len(g):
                    ax.plot(g["k"], g["rmse_truth"], "o-", ms=4, color=cmap(c / max(len(ns) - 1, 1)), label=f"n={int(n)}")
            ax.set_xticks(sorted(d["k"].unique())); ax.grid(alpha=0.3)
            if i == 0: ax.set_title(f"probes: {dist}")
            if j == 0: ax.set_ylabel(f"{_truth_label(smc, ws)}\nRMSE θ [m³/m³]", fontsize=8)
            if i == len(truths) - 1: ax.set_xlabel("number of free WS parameters K")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("EC→SWC error vs number of free Waxman-Smits parameters", y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rmse_vs_k.png")


def plot_heatmap_n_k(df, out_dir, reg_ref="noreg", dist_ref="spread"):
    """RMSE heat map number of probes x K, one panel per truth."""
    import matplotlib.pyplot as plt
    d = _ok(expand_degenerate(df)); d = d[d["reg"] == _ref_reg(d, reg_ref)]
    dist_ref = dist_ref if dist_ref in set(d["dist"]) else (d["dist"].iloc[0] if len(d) else dist_ref)
    d = d[d["dist"] == dist_ref]; truths = _truths(d)
    if d.empty:
        return None
    fig, axes = plt.subplots(1, len(truths), figsize=(3.6 * len(truths) + 1, 3.4), squeeze=False)
    vmin, vmax = d["rmse_truth"].min(), d["rmse_truth"].max(); im = None
    for ax, (smc, ws) in zip(axes[0], truths):
        sub = d[(d["smc"] == smc) & (d["ws"] == ws)]
        ns, ks = sorted(sub["n"].unique()), sorted(sub["k"].unique())
        z = np.full((len(ns), len(ks)), np.nan)
        for _, r in sub.iterrows():
            z[ns.index(r["n"]), ks.index(r["k"])] = r["rmse_truth"]
        im = ax.imshow(z, origin="lower", aspect="auto", cmap="magma_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(ks)), [str(int(k)) for k in ks]); ax.set_yticks(range(len(ns)), [str(int(n)) for n in ns])
        for a, b in itertools.product(range(len(ns)), range(len(ks))):
            if np.isfinite(z[a, b]):
                ax.text(b, a, f"{z[a, b]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if z[a, b] > (vmin + vmax) / 2 else "black")
        ax.set_title(_truth_label(smc, ws), fontsize=9); ax.set_xlabel("K free WS"); ax.set_ylabel("n SWC positions")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="RMSE θ [m³/m³]", shrink=0.9)
    fig.suptitle(f"RMSE θ vs truth (probes: {dist_ref}, reg: {_ref_reg(d, reg_ref)})", y=1.02)
    return _save(fig, out_dir, "fig_heatmap_n_vs_k.png")


def plot_distribution_effect(df, out_dir, reg_ref="noreg", n_ref=None):
    """Grouped bars: probe distribution at fixed n; columns = truth; x = K."""
    import matplotlib.pyplot as plt
    d = _ok(expand_degenerate(df)); d = d[d["reg"] == _ref_reg(d, reg_ref)]
    cand = sorted(n for n in d["n"].unique() if 1 < n < SURVEY_N and d[d["n"] == n]["dist"].nunique() > 1)
    if not cand:
        return None
    n_ref = n_ref if n_ref in cand else (REFERENCE["n"] if REFERENCE["n"] in cand else cand[-1])
    d = d[d["n"] == n_ref]; truths = _truths(d); ks = sorted(d["k"].unique())
    fig, axes = plt.subplots(1, len(truths), figsize=(3.8 * len(truths), 3.2), sharey=True, squeeze=False)
    dists = [x for x in LEVELS["dist"] if x in set(d["dist"])]; w = 0.8 / max(len(dists), 1)
    for ax, (smc, ws) in zip(axes[0], truths):
        sub = d[(d["smc"] == smc) & (d["ws"] == ws)]
        for c, dist in enumerate(dists):
            vals = [sub[(sub["dist"] == dist) & (sub["k"] == k)]["rmse_truth"] for k in ks]
            ax.bar(np.arange(len(ks)) + (c - (len(dists) - 1) / 2) * w, [v.iloc[0] if len(v) else np.nan for v in vals],
                   w, color=_DIST_COLORS.get(dist), label=dist)
        ax.set_xticks(range(len(ks)), [str(int(k)) for k in ks]); ax.set_xlabel("K free WS")
        ax.set_title(_truth_label(smc, ws), fontsize=9); ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("RMSE θ [m³/m³]"); axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Distribution of the SWC probes (n = {int(n_ref)})", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_distribution_effect.png")


def plot_regularisation_effect(df, out_dir, n_ref=None, dist_ref="spread"):
    """Grouped bars: regularisation (none / beta / gamma / both); columns = truth; x = K."""
    import matplotlib.pyplot as plt
    d = _ok(expand_degenerate(df))
    if d["reg"].nunique() < 2:
        return None
    n_ref = n_ref if n_ref in set(d["n"]) else REFERENCE["n"] if REFERENCE["n"] in set(d["n"]) else d["n"].max()
    dist_ref = dist_ref if dist_ref in set(d["dist"]) else d["dist"].iloc[0]
    d = d[(d["n"] == n_ref) & (d["dist"] == dist_ref)]; truths = _truths(d); ks = sorted(d["k"].unique())
    regs = [r for r in LEVELS["reg"] if r in set(d["reg"])]; w = 0.8 / max(len(regs), 1)
    fig, axes = plt.subplots(1, len(truths), figsize=(3.8 * len(truths), 3.2), sharey=True, squeeze=False)
    for ax, (smc, ws) in zip(axes[0], truths):
        sub = d[(d["smc"] == smc) & (d["ws"] == ws)]
        for c, reg in enumerate(regs):
            vals = [sub[(sub["reg"] == reg) & (sub["k"] == k)]["rmse_truth"] for k in ks]
            ax.bar(np.arange(len(ks)) + (c - (len(regs) - 1) / 2) * w, [v.iloc[0] if len(v) else np.nan for v in vals],
                   w, color=_REG_COLORS.get(reg), label=reg)
        ax.set_xticks(range(len(ks)), [str(int(k)) for k in ks]); ax.set_xlabel("K free WS")
        ax.set_title(_truth_label(smc, ws), fontsize=9); ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("RMSE θ [m³/m³]"); axes[0, 0].legend(fontsize=7, title=f"β={BETA_VALUE:g}, γ={GAMMA_VALUE:g}", title_fontsize=7)
    fig.suptitle(f"Effect of the EMagPy regularisation (n = {int(n_ref)}, probes: {dist_ref})", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_regularisation_effect.png")


def plot_truth_effect(df, out_dir, reg_ref="noreg", dist_ref="spread", n_ref=None):
    """Effect of a homogeneous vs heterogeneous truth (SMC, WS) on the RMSE vs K."""
    import matplotlib.pyplot as plt
    d = _ok(expand_degenerate(df)); d = d[(d["reg"] == _ref_reg(d, reg_ref))]
    if len(_truths(d)) < 2:
        return None
    n_ref = n_ref if n_ref in set(d["n"]) else REFERENCE["n"] if REFERENCE["n"] in set(d["n"]) else d["n"].max()
    dist_ref = dist_ref if dist_ref in set(d["dist"]) else d["dist"].iloc[0]
    d = d[(d["n"] == n_ref) & (d["dist"] == dist_ref)]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for (smc, ws), mk in zip(_truths(d), "osD^"):
        g = d[(d["smc"] == smc) & (d["ws"] == ws)].sort_values("k")
        ax.plot(g["k"], g["rmse_truth"], f"{mk}-", label=_truth_label(smc, ws))
    ax.set_xlabel("K free WS"); ax.set_ylabel("RMSE θ [m³/m³]"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_xticks(sorted(d["k"].unique()))
    ax.set_title(f"Homogeneous vs heterogeneous truth (n = {int(n_ref)}, {dist_ref})", fontsize=9)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_truth_heterogeneity_effect.png")


def plot_calibrated_vs_uncalibrated(df, out_dir, reg_ref="noreg", dist_ref="spread"):
    """Error at points with a probe vs away from the probes, vs number of probes (K = max)."""
    import matplotlib.pyplot as plt
    d = _ok(expand_degenerate(df)); d = d[d["reg"] == _ref_reg(d, reg_ref)]
    d = d[d["dist"] == (dist_ref if dist_ref in set(d["dist"]) else d["dist"].iloc[0])]
    d = d[d["k"] == d["k"].max()]; truths = _truths(d)
    if d.empty or d["rmse_near_probe"].isna().all():
        return None
    fig, axes = plt.subplots(1, len(truths), figsize=(3.6 * len(truths), 3.1), sharey=True, squeeze=False)
    for ax, (smc, ws) in zip(axes[0], truths):
        g = d[(d["smc"] == smc) & (d["ws"] == ws)].sort_values("n")
        ax.plot(g["n"], g["rmse_near_probe"], "o-", label="at a probe position")
        ax.plot(g["n"], g["rmse_far_probe"], "s--", label="away from probes")
        ax.plot(g["n"], g["rmse_truth"], "k:", label="all points")
        ax.set_title(_truth_label(smc, ws), fontsize=9); ax.set_xlabel("number of SWC positions"); ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("RMSE θ [m³/m³]"); axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Error at / away from probes (K = {int(d['k'].max())})", y=1.03)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_error_at_vs_away_from_probes.png")


def plot_residual_distribution(df, out_dir, residual_dir, reg_ref="noreg", k_ref=None):
    """Box plots of theta_calc - theta_true by number of probes; rows = truth; columns = distribution."""
    import matplotlib.pyplot as plt
    d = _ok(expand_degenerate(df)); d = d[d["reg"] == _ref_reg(d, reg_ref)]
    if d.empty:
        return None
    k_ref = k_ref if k_ref in set(d["k"]) else d["k"].max()
    d = d[d["k"] == k_ref]; truths, dists = _truths(d), [x for x in LEVELS["dist"] if x in set(d["dist"])]
    fig, axes = plt.subplots(len(truths), len(dists), figsize=(3.6 * len(dists), 2.8 * len(truths)),
                             sharex=False, sharey=True, squeeze=False)
    any_data = False
    for i, (smc, ws) in enumerate(truths):
        for j, dist in enumerate(dists):
            ax, sub = axes[i, j], d[(d["smc"] == smc) & (d["ws"] == ws) & (d["dist"] == dist)].sort_values("n")
            data, labels = [], []
            for _, r in sub.iterrows():
                p = os.path.join(residual_dir, r["scenario"] + ".npz")
                if os.path.isfile(p):
                    data.append(np.load(p)["resid"].ravel()); labels.append(str(int(r["n"])))
            if not data:
                _empty(ax); continue
            any_data = True
            ax.axhline(0, color="k", lw=0.8); ax.boxplot(data, tick_labels=labels, showmeans=True, flierprops={"markersize": 2})
            ax.grid(axis="y", alpha=0.3)
            if i == 0: ax.set_title(f"probes: {dist}")
            if j == 0: ax.set_ylabel(f"{_truth_label(smc, ws)}\nθ_calc − θ_true", fontsize=8)
            if i == len(truths) - 1: ax.set_xlabel("number of SWC positions")
    if not any_data:
        import matplotlib.pyplot as plt2; plt2.close(fig); return None
    fig.suptitle(f"Residual distribution (K = {int(k_ref)})", y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_residual_distribution.png")


def make_plots(df, out_dir: str, reg_ref: str = "noreg") -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    ok = _ok(df)
    if ok.empty:
        print("[study] no run with results: nothing to plot")
        return []
    res = os.path.join(out_dir, "residuals")
    paths = [plot_rmse_vs_nswc(df, out_dir, reg_ref), plot_rmse_vs_k(df, out_dir, reg_ref),
             plot_heatmap_n_k(df, out_dir, reg_ref), plot_distribution_effect(df, out_dir, reg_ref),
             plot_regularisation_effect(df, out_dir), plot_truth_effect(df, out_dir, reg_ref),
             plot_calibrated_vs_uncalibrated(df, out_dir, reg_ref),
             plot_residual_distribution(df, out_dir, res, reg_ref)]
    return [p for p in paths if p]


def write_report(df, path: str, reg_ref: str = "noreg") -> None:
    """Markdown tables: RMSE per (truth, K) x n and the best design per truth."""
    lines = ["# SWC number / distribution study", "",
             f"Runs: {len(df)}, with results: {int(df['rmse_truth'].notna().sum())}. "
             "RMSE = rmse(theta_calc - theta_true) over all epochs, points, layers [m3/m3].", ""]
    d = _ok(expand_degenerate(df))
    if not d.empty:
        d = d[d["reg"] == _ref_reg(d, reg_ref)]
        for (smc, ws), dist in itertools.product(_truths(d), [x for x in LEVELS["dist"] if x in set(d["dist"])]):
            sub = d[(d["smc"] == smc) & (d["ws"] == ws) & (d["dist"] == dist)]
            if sub.empty:
                continue
            ns = sorted(sub["n"].unique())
            lines += [f"## {_truth_label(smc, ws)} - probes {dist} - reg {_ref_reg(d, reg_ref)}", "",
                      "| K \\ n | " + " | ".join(str(int(n)) for n in ns) + " |", "|---|" + "---|" * len(ns)]
            for k in sorted(sub["k"].unique()):
                cells = []
                for n in ns:
                    v = sub[(sub["k"] == k) & (sub["n"] == n)]["rmse_truth"]
                    cells.append(f"{v.iloc[0]:.4f}" if len(v) else "-")
                lines.append(f"| {int(k)} | " + " | ".join(cells) + " |")
            lines.append("")
        lines += ["## Best design per truth (all regularisations)", "", "| truth | scenario | RMSE |", "|---|---|---|"]
        for (smc, ws) in _truths(d):
            sub = _ok(df[(df["smc"] == smc) & (df["ws"] == ws)])
            if len(sub):
                b = sub.loc[sub["rmse_truth"].idxmin()]
                lines.append(f"| {_truth_label(smc, ws)} | {b['scenario']} | {b['rmse_truth']:.4f} |")
    failed = df[df["status"].astype(str).isin(["FAIL", "ERROR", "FAILED", "missing"])]
    if len(failed):
        lines += ["", "## Runs without result", ""] + [f"- {r['scenario']} ({r['status']}) {r.get('error', '')}" for _, r in failed.iterrows()]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def realization_from_config(cfg_path: str) -> int:
    """``realization_index`` of the study parent (truth member compared with theta_calc); 0 if unknown."""
    try:
        with open(cfg_path, encoding="utf-8") as f:
            return int(json.load(f)[PARENT_NAME].get("realization_index", 0))
    except (OSError, KeyError, ValueError):
        return 0


def analyze(designs: list[Design], out_dir: str, survey, realization: int, reg_ref: str) -> list[dict]:
    """Collect the latest run of every design from the run log, write summary/report/plots."""
    latest = latest_rows_by_scenario(RUN_LOG)
    rows = [summary_row(d, latest.get(d.name), out_dir, survey, realization) for d in designs]
    write_summary(rows, os.path.join(out_dir, "study_summary.csv"))
    df = load_summary(os.path.join(out_dir, "study_summary.csv"))
    write_report(df, os.path.join(out_dir, "study_report.md"), reg_ref)
    for p in make_plots(df, out_dir, reg_ref):
        print(f"[study] figure: {p}")
    return rows


def main(argv: list[str] | None = None) -> int:
    global RANDOM_SEED
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=["smoke", "oat", "main", "full"], default="oat")
    ap.add_argument("--n-swc", nargs="+", type=int, help=f"numbers of SWC positions (default {LEVELS['n']})")
    ap.add_argument("--dist", nargs="+", choices=LEVELS["dist"], help="probe distributions")
    ap.add_argument("--smc", nargs="+", choices=LEVELS["smc"], help="truth soil moisture")
    ap.add_argument("--ws", nargs="+", choices=LEVELS["ws"], help="truth Waxman-Smits parameters")
    ap.add_argument("--free-k", nargs="+", type=int, choices=[1, 2, 3, 4], help="numbers of free WS parameters")
    ap.add_argument("--reg", nargs="+", choices=LEVELS["reg"], help="regularisation: noreg beta gamma both")
    ap.add_argument("--beta", type=float, default=BETA_VALUE, help="beta used by reg beta/both")
    ap.add_argument("--gamma", type=float, default=GAMMA_VALUE, help="gamma used by reg gamma/both")
    ap.add_argument("--p0-bias", type=float, default=P0_BIAS,
                    help="relative offset of the calibration start of free m, sigma_w, sigma_s (0 = repo default)")
    ap.add_argument("--smc-full-range", action="store_true", help="hetero SMC = full-range gradient")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED, help="seed of the 'random' distribution")
    ap.add_argument("--survey-n", type=int, default=SURVEY_N)
    ap.add_argument("--survey-dx", type=float, default=SURVEY_DX)
    ap.add_argument("--config", default=BASE_CONFIG, help="scenarios file holding the base scenario")
    ap.add_argument("--base", default=BASE_SCENARIO, help="base scenario (1d_full_depth_ref | 1d_surface_tdr)")
    ap.add_argument("--mode", default="1d", choices=["1d", "2d"])
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--runs-dir", default=RUNS_DIR)
    ap.add_argument("--reg-ref", default="noreg", help="regularisation shown in the n / K plots")
    ap.add_argument("--fast", action="store_true", help="pass --quick to every run (smoke test, not a result)")
    ap.add_argument("--dry-run", action="store_true", help="write config + probes, validate, print the plan")
    ap.add_argument("--resume", action="store_true", help="skip designs already finished OK in the run log")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed run")
    ap.add_argument("--analyze-only", action="store_true", help="only collect + plot from the run log")
    args = ap.parse_args(argv)

    RANDOM_SEED = args.seed
    levels = {f: tuple(v) for f, v in LEVELS.items()}
    for f, v in (("n", args.n_swc), ("dist", args.dist), ("smc", args.smc), ("ws", args.ws),
                 ("k", args.free_k), ("reg", args.reg)):
        if v:
            levels[f] = tuple(v)
    survey = survey_x(args.survey_n, args.survey_dx)
    designs = build_designs(args.preset, levels, survey)
    if not designs:
        ap.error("empty design")

    cfg_path = os.path.join(args.out_dir, "study_scenarios.json")
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.analyze_only:
        raw = sweeps.with_generated({k: v for k, v in json.load(open(args.config, encoding="utf-8")).items()
                                     if k not in sl._SKIP_KEYS})
        resolved = sl._resolve_extends(raw)
        if args.base not in resolved:
            ap.error(f"base scenario {args.base!r} not in {args.config} (nor generated by sweeps.py)")
        parent = resolved[args.base]
        cfg = build_study_config(parent, designs, runs_dir=args.runs_dir, beta=args.beta, gamma=args.gamma,
                                 p0_bias=args.p0_bias, smc_full_range=args.smc_full_range)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        depths = probe_depths(parent)
        realization = realization_from_config(cfg_path)
        probe_csv = {}
        for d in designs:
            probe_csv[d.name] = write_probe_csv(
                os.path.join(args.out_dir, "probes", f"{d.dist}_n{d.n}.csv"), probe_positions(d.dist, d.n, survey), depths)
        for d in designs:                       # fail now, not after hours of runs
            sl.load_scenario(d.name, config_path=cfg_path)

        print(f"Study config : {cfg_path}  ({len(designs)} designs, preset '{args.preset}', base '{args.base}')")
        print(f"Probe depths : {depths} m   survey x = {survey.tolist()}")
        for d in designs:
            print(f"  $ {' '.join(pipeline_command(d, cfg_path, probe_csv[d.name], args.mode, args.fast, extra))}")
        if args.dry_run:
            return 0

        done = latest_rows_by_scenario(RUN_LOG) if args.resume else {}
        failed = []
        for d in designs:
            if args.resume and done.get(d.name, {}).get("status", "") not in ("", "FAIL", "ERROR", "FAILED"):
                print(f"[study] skip {d.name} (already run)")
                continue
            t0 = time.time()
            rc = subprocess.call(pipeline_command(d, cfg_path, probe_csv[d.name], args.mode, args.fast, extra))
            analyze(designs, args.out_dir, survey, realization, args.reg_ref)      # summary/plots after every run
            if rc != 0:
                failed.append(d.name)
                print(f"!!! {d.name} failed (exit {rc}) after {time.time() - t0:.0f} s")
                if not args.keep_going:
                    print("stopping (use --keep-going to continue)")
                    break
        analyze(designs, args.out_dir, survey, realization, args.reg_ref)
        return 1 if failed else 0

    analyze(designs, args.out_dir, survey, realization_from_config(cfg_path), args.reg_ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
