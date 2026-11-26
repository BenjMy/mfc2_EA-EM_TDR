#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_ec_inversion.py  --  Patch 2b of swc_study_known_ws_plan.md
=========================================================================

Why is the inverted EC (``ec_optimal.npy``) so far from the true EC even when
all four Waxman-Smits parameters are pinned (K = 0 floor 0.13 m3/m3)?

The plan lists three candidate causes taken from the scenario settings:

  C1  ``fixed_depths = false``       interfaces can move, truth layers do not
  C2  ``forwardModel = [FSlin, CS]`` data are generated with CS only
  C3  ``method = [ROPE]``            stochastic optimiser, cond. bounds 0.01-300

This script tests them **outside the pipeline** (EMagPy only, ~1 min), one
factor at a time, on the same synthetic truth as the swc study
(``smc = hetero``, ``ws = homo``, layers 0.3 / 0.6 / 1.0 m, 1 % ECa noise):

  T0  controls    : does the truth model reproduce the data? (generator vs
                    survey/inversion conventions); noise-free; start at truth.
  T1  C1          : fixed_depths  False -> True
  T2  C2          : forwardModel  {FSlin, CS} -> CS only / FSlin only
  T3  C3          : method        ROPE -> L-BFGS-B, TNC, Nelder-Mead
  T4  extras      : narrower cond. bounds, stronger regularisation (alpha)
  T5  combination : fixed depths + CS + L-BFGS-B
  T6  (slow, opt-in): ROPE with more samples (rep = 1000, 5000)

For each variant the same ensemble as step 1 is run (alpha x forwardModel x
two start models = the two ends of ``cond_ranges``) and compared with the true
EC layer by layer: ratio of medians (inverted / true), Pearson correlation of
log10 EC, RMSE of log10 EC and the data misfit of the inverted model.
"Mean" is the plain mean of the ensemble members (what step 3 averages with
K = 1); "best" is the member with the smallest data misfit.

Usage
-----
    python diagnose_ec_inversion.py                    # all tests
    python diagnose_ec_inversion.py --tests T0 T1      # a subset
    python diagnose_ec_inversion.py --time 1 --seed 3  # other epoch / seed
    python diagnose_ec_inversion.py --out ec_cause_tests

Writes ``<out>/ec_cause_summary.csv`` (one row per variant x layer),
``<out>/ec_cause_report.md`` and ``<out>/fig_ec_cause_ratio.png``.
Only reads the repo (``data_synthetic.py``); needs ``emagpy``.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import data_synthetic as ds  # noqa: E402

DEPTHS0 = [0.3, 0.6, 1.0]
ZMAX = 1.5                      # lower limit of the last truth layer (data_synthetic: 1.0 * 1.5)
COND_RANGE = (0.01, 300.0)      # study base scenario 1d_full_depth_ref
ALPHAS = (0.03, 0.1, 0.3)       # study base scenario
SILENT = True

# ---------------------------------------------------------------------------
# Variants.  BASE reproduces step 1 of the study base scenario.
# ---------------------------------------------------------------------------
BASE = dict(
    fixed_depths=False,
    forward_models=("FSlin", "CS"),
    methods=("ROPE",),
    alphas=ALPHAS,
    bounds=COND_RANGE,
    regularization="l2",        # EMagPy default (the pipeline does not set it)
    noise=0.01,                 # eca_noise_level of the base scenario
    start="range",              # the two ends of the bounds, as step 1
    rep=None,                   # ROPE samples (EMagPy default 100; the pipeline never sets it)
    note="",
)


def _v(**kw):
    d = copy.deepcopy(BASE)
    d.update(kw)
    return d


TESTS = {
    "T0": [
        ("T0a baseline (as the study)", _v(note="reference")),
        ("T0b noise-free data", _v(noise=0.0)),
        ("T0c start at truth, L-BFGS-B, fixed depths",
         _v(start="truth", methods=("L-BFGS-B",), fixed_depths=True,
            forward_models=("CS",), alphas=(0.03,))),
    ],
    "T1": [
        ("T1  C1 fixed_depths=True", _v(fixed_depths=True)),
    ],
    "T2": [
        ("T2a C2 forwardModel=CS only", _v(forward_models=("CS",))),
        ("T2b C2 forwardModel=FSlin only", _v(forward_models=("FSlin",))),
    ],
    "T3": [
        ("T3a C3 method=L-BFGS-B", _v(methods=("L-BFGS-B",))),
        ("T3b C3 method=TNC", _v(methods=("TNC",))),
        ("T3c C3 method=Nelder-Mead", _v(methods=("Nelder-Mead",))),
    ],
    # slow (minutes each): not in the default selection, run with --tests T6
    "T6": [
        ("T6a ROPE rep=1000", _v(rep=1000)),
        ("T6b ROPE rep=5000", _v(rep=5000)),
    ],
    "T4": [
        ("T4a bounds 0.01-100", _v(bounds=(0.01, 100.0))),
        ("T4b bounds 0.5-100", _v(bounds=(0.5, 100.0))),
        ("T4c alpha 1.0", _v(alphas=(1.0,))),
    ],
    "T5": [
        ("T5  fixed + CS + L-BFGS-B", _v(fixed_depths=True, forward_models=("CS",),
                                        methods=("L-BFGS-B",))),
        ("T5b fixed + CS + L-BFGS-B, bounds 0.5-100",
         _v(fixed_depths=True, forward_models=("CS",), methods=("L-BFGS-B",),
            bounds=(0.5, 100.0))),
    ],
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def quiet(on: bool = True):
    if not on:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def make_truth(workdir: str, seed: int, noise: float, t_idx: int):
    """Truth (swc study, smc=hetero, ws=homo) and the survey CSV of epoch ``t_idx``."""
    gen = os.path.join(workdir, "gen")
    np.random.seed(seed)
    with quiet(SILENT):
        ds.generate_time_lapse_dataset(
            output_dir=gen, mode="1d", depths0=DEPTHS0, homogeneous=False,
            ws_xy_variation=False)
        csvs, _, _ = ds.export_eca_surveys_to_csv(
            output_dir=gen, eca_noise_level=noise, csv_subdir=f"csv_{noise}")
    ec = np.load(os.path.join(gen, "synthetic_time_lapse_ec.npy"))[0]     # (t, pts, layers)
    eca = np.load(os.path.join(gen, "synthetic_time_lapse_eca.npy"))[0]   # (t, pts, coils)
    th = np.load(os.path.join(gen, "synthetic_time_lapse_theta.npy"))[0]  # (t, pts, layers)
    make_truth.theta = th[t_idx]          # read back by main(); see theta_rmse()
    return csvs[t_idx], ec[t_idx], eca[t_idx], gen


def csv_for_noise(gen: str, noise: float, t_idx: int) -> str:
    with quiet(SILENT):
        csvs, _, _ = ds.export_eca_surveys_to_csv(
            output_dir=gen, eca_noise_level=noise, csv_subdir=f"csv_{noise}")
    return csvs[t_idx]


# ---------------------------------------------------------------------------
# EMagPy
# ---------------------------------------------------------------------------
def forward_eca(models: np.ndarray, depths: np.ndarray, fm: str) -> np.ndarray:
    """ECa [mS/m] (pts, coils) of (pts, layers) models with (pts, layers-1) depths."""
    from emagpy import Problem
    k = Problem()
    with quiet(SILENT):
        out = k.forward(forwardModel=fm, coils=ds.COILS_LIST,
                        models=[np.atleast_2d(models)], depths=[np.atleast_2d(depths)])
    return np.asarray(out[0].values, dtype=float)


def bounds_list(fixed_depths: bool, lo: float, hi: float, n_layers: int):
    """Same rule as geometry_utils.build_inversion_bnds, written out here."""
    if fixed_depths:
        return [(lo, hi)] * n_layers
    import geometry_utils
    return geometry_utils.build_inversion_bnds(
        DEPTHS0, [False] * len(DEPTHS0), [[lo, hi]] * n_layers)


def run_members(csv: str, v: dict, truth_ec: np.ndarray):
    """All ensemble members of one variant -> list of dicts (models, depths, label)."""
    from emagpy import Problem
    n_layers = len(DEPTHS0) + 1
    lo, hi = v["bounds"]
    bnds = bounds_list(v["fixed_depths"], lo, hi, n_layers)
    starts = ([[lo] * n_layers, [hi] * n_layers] if v["start"] == "range" else ["truth"])
    members = []
    for fm in v["forward_models"]:
        for method in v["methods"]:
            for alpha in v["alphas"]:
                for s_i, conds0 in enumerate(starts):
                    k = Problem()
                    with quiet(SILENT):
                        k.createSurvey(csv, unit="ppt")
                        n_pts = len(k.surveys[0].df)
                        if conds0 == "truth":
                            # one start model per point is not supported by setInit:
                            # use the median truth profile
                            c0 = list(np.median(truth_ec, axis=0))
                        else:
                            c0 = conds0
                        k.setInit(depths0=list(DEPTHS0), conds0=c0,
                                  fixedDepths=[v["fixed_depths"]] * len(DEPTHS0))
                        extra = {"rep": v["rep"]} if v["rep"] else {}
                        k.invert(forwardModel=fm, method=method, alpha=alpha,
                                 regularization=v["regularization"], bnds=bnds, **extra)
                    members.append(dict(
                        label=f"{fm}/{method}/a{alpha}/s{s_i}", fm=fm,
                        models=np.asarray(k.models[0], dtype=float),
                        depths=np.asarray(k.depths[0], dtype=float)
                        if k.depths and k.depths[0] is not None
                        else np.tile(DEPTHS0, (n_pts, 1))))
    return members


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def resample_to_truth_layers(ec: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Depth-weighted mean of each inverted profile over the TRUTH intervals.

    ec (pts, L), depths (pts, L-1) interface depths; returns (pts, len(DEPTHS0)+1).
    With fixed depths this is the identity, with free depths it removes the
    'different depth interval' effect of candidate cause C1.
    """
    edges_t = np.r_[0.0, DEPTHS0, ZMAX]
    out = np.zeros((ec.shape[0], len(edges_t) - 1))
    for p in range(ec.shape[0]):
        edges_i = np.r_[0.0, np.sort(depths[p]), max(ZMAX, depths[p].max() + 0.1)]
        for j in range(len(edges_t) - 1):
            a, b = edges_t[j], edges_t[j + 1]
            acc = 0.0
            for i in range(len(edges_i) - 1):
                ov = max(0.0, min(b, edges_i[i + 1]) - max(a, edges_i[i]))
                acc += ov * ec[p, i]
            out[p, j] = acc / (b - a)
    return out


def layer_metrics(inv: np.ndarray, true: np.ndarray) -> list[dict]:
    rows = []
    for l in range(true.shape[1]):
        t, i = true[:, l], np.clip(inv[:, l], 1e-3, None)
        lt, li = np.log10(np.clip(t, 1e-3, None)), np.log10(i)
        corr = np.corrcoef(lt, li)[0, 1] if np.std(lt) > 0 and np.std(li) > 0 else np.nan
        rows.append(dict(layer=l, true_med=np.median(t), inv_med=np.median(i),
                         ratio=np.median(i) / np.median(t), corr_log=corr,
                         rmse_log10=float(np.sqrt(np.mean((li - lt) ** 2)))))
    return rows


WS_TRUE = (0.5, 3.0, 500.0, 100.0)     # a, m, sigma_w, sigma_s of the swc study baseline


def theta_from_ec(ec: np.ndarray, ws=WS_TRUE) -> np.ndarray:
    a, m, sw, ss = ws
    return (np.clip(ec, 1e-6, None) * a / (sw + ss)) ** (1.0 / m)


def theta_rmse(ec: np.ndarray, theta_true: np.ndarray, ws=WS_TRUE) -> float:
    """RMSE [m3/m3] of theta obtained from ``ec`` with the TRUE WS parameters.

    This is the K = 0 floor of the swc study as seen from step 1 alone (nothing is
    calibrated: all four WS parameters are known), so it isolates the EC error.
    """
    return float(np.sqrt(np.mean((theta_from_ec(ec, ws) - theta_true) ** 2)))


def data_misfit_pct(models, depths, fm, eca_ref):
    """RMS of the relative ECa misfit [%] between forward(inverted) and the reference ECa."""
    sim = forward_eca(models, depths, fm)
    return 100.0 * float(np.sqrt(np.mean(((sim - eca_ref) / np.clip(eca_ref, 1e-6, None)) ** 2)))


def evaluate(name: str, members: list, truth_ec: np.ndarray, eca_data: np.ndarray,
             theta_true: np.ndarray | None = None) -> list[dict]:
    """Rows for 'mean' and 'best' (smallest data misfit) of one variant."""
    res, mis = [], []
    for m in members:
        m["res"] = resample_to_truth_layers(m["models"], m["depths"])
        m["misfit"] = data_misfit_pct(m["models"], m["depths"], m["fm"], eca_data)
        res.append(m["res"])
        mis.append(m["misfit"])
    mean_res = np.mean(res, axis=0)
    best = int(np.argmin(mis))
    out = []
    for kind, arr, mf in (("mean", mean_res, float(np.mean(mis))),
                          ("best", res[best], float(mis[best]))):
        th_r = theta_rmse(arr, theta_true) if theta_true is not None else np.nan
        for r in layer_metrics(arr, truth_ec):
            r.update(variant=name, kind=kind, n_members=len(members), misfit_pct=mf,
                     theta_rmse=th_r)
            out.append(r)
    # spread between members (how much does the ensemble disagree?)
    spread = float(np.mean(np.std(np.log10(np.clip(np.array(res), 1e-3, None)), axis=0)))
    for r in out:
        r["member_spread_log10"] = spread
    return out


# ---------------------------------------------------------------------------
# T0: conventions
# ---------------------------------------------------------------------------
def control_truth_forward(truth_ec, eca_clean, eca_noisy_csv):
    """Truth model -> forward -> compare with the stored ECa and with the CSV."""
    pts = truth_ec.shape[0]
    depths = np.tile(DEPTHS0, (pts, 1))
    sim = forward_eca(truth_ec, depths, "CS")
    d_store = 100 * np.sqrt(np.mean(((sim - eca_clean) / eca_clean) ** 2))
    csv = pd.read_csv(eca_noisy_csv)[ds.COILS_LIST].values
    d_csv = 100 * np.sqrt(np.mean(((sim - csv) / csv) ** 2))
    d_fs = data_misfit_pct(truth_ec, depths, "FSlin", eca_clean)
    return dict(truth_CS_vs_stored_pct=d_store, truth_CS_vs_csv_pct=d_csv,
                truth_FSlin_vs_stored_pct=d_fs)


def floor_cap_share(truth_ec: np.ndarray, hi: float) -> dict:
    return dict(share_at_1mS_m_floor=float(np.mean(truth_ec <= 1.0 + 1e-9)),
                share_above_hi=float(np.mean(truth_ec > hi)),
                ec_min=float(truth_ec.min()), ec_max=float(truth_ec.max()))


# ---------------------------------------------------------------------------
# Report / figure
# ---------------------------------------------------------------------------
def write_report(df: pd.DataFrame, controls: dict, shares: dict, truth_ec, path: str):
    lines = ["# EC inversion: candidate-cause tests (Patch 2b)", "",
             f"Truth: smc=hetero, ws=homo (0.5, 3, 500, 100), layers {DEPTHS0}, "
             f"{truth_ec.shape[0]} points, one epoch.", "",
             "## Controls (conventions)", ""]
    for k, v in controls.items():
        lines.append(f"* {k}: {v:.3g}" + ("" if k.startswith("theta") else " %"))
    lines += ["", f"True EC: min {shares['ec_min']:.2f}, max {shares['ec_max']:.1f} mS/m; "
              f"{100 * shares['share_at_1mS_m_floor']:.1f} % at the 1 mS/m floor, "
              f"{100 * shares['share_above_hi']:.1f} % above the 300 mS/m bound.", ""]
    for kind in ("mean", "best"):
        lines += [f"## Ensemble {kind}: ratio inverted / true median EC per layer", "",
                  "| variant | n | misfit % | theta RMSE | spread log10 | ratio L0 | L1 | L2 | L3 | "
                  "corr L0 | L1 | L2 | L3 |", "|" + "---|" * 13]
        sub = df[df.kind == kind]
        for name, g in sub.groupby("variant", sort=False):
            g = g.sort_values("layer")
            lines.append(
                f"| {name} | {int(g.n_members.iloc[0])} | {g.misfit_pct.iloc[0]:.1f} | "
                f"{g.theta_rmse.iloc[0]:.3f} | {g.member_spread_log10.iloc[0]:.2f} | "
                + " | ".join(f"{x:.2f}" for x in g.ratio) + " | "
                + " | ".join(f"{x:.2f}" for x in g.corr_log) + " |")
        lines.append("")
    lines += ["Ratio = 1 and corr = 1 would be a perfect recovery. Misfit % is the RMS "
              "relative ECa error of forward(inverted); the 1 % noise level is the "
              "floor a correct inversion should reach. theta RMSE [m3/m3] = theta from the "
              "inverted EC with the TRUE WS parameters, i.e. the K = 0 floor of the swc study "
              "(compare with 0.03).", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_figure(df: pd.DataFrame, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = df[df.kind == "mean"]
    names = list(dict.fromkeys(sub.variant))
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(names) + 1.8))
    w = 0.18
    for l in range(4):
        vals = [sub[(sub.variant == n) & (sub.layer == l)].ratio.iloc[0] for n in names]
        ax.barh(np.arange(len(names)) + (l - 1.5) * w, vals, height=w, label=f"layer {l}")
    ax.axvline(1.0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("median inverted EC / median true EC  (1 = unbiased)")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tests", nargs="+", default=[t for t in TESTS if t != "T6"],
                    choices=list(TESTS), help="default: all except the slow T6")
    ap.add_argument("--time", type=int, default=0, help="epoch index (0-2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ec_cause_tests")
    ap.add_argument("--verbose", action="store_true", help="show EMagPy output")
    args = ap.parse_args(argv)

    global SILENT
    SILENT = not args.verbose
    os.makedirs(args.out, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        csv_noisy, truth_ec, eca_clean, gen = make_truth(tmp, args.seed, BASE["noise"], args.time)
        theta_true = make_truth.theta
        csv_clean = csv_for_noise(gen, 0.0, args.time)
        controls = control_truth_forward(truth_ec, eca_clean, csv_noisy)
        # theta RMSE of the TRUE EC through the true WS: only the generator's 1 mS/m floor
        controls["theta_rmse_of_true_EC_m3m3"] = theta_rmse(truth_ec, theta_true)
        shares = floor_cap_share(truth_ec, COND_RANGE[1])
        print("Controls:", {k: round(float(v), 3) for k, v in controls.items()})
        print("True EC :", {k: round(float(v), 3) for k, v in shares.items()})

        np.random.seed(args.seed)
        rows = []
        for tid in args.tests:
            for name, v in TESTS[tid]:
                csv = csv_clean if v["noise"] == 0.0 else csv_noisy
                eca_ref = eca_clean if v["noise"] == 0.0 else pd.read_csv(csv)[ds.COILS_LIST].values
                print(f"-> {name} ...", flush=True)
                members = run_members(csv, v, truth_ec)
                rows += evaluate(name, members, truth_ec, eca_ref, theta_true)
                last = [r for r in rows if r["variant"] == name and r["kind"] == "mean"]
                print("   mean ratio per layer:",
                      [round(float(r["ratio"]), 2) for r in last],
                      " corr:", [round(float(r["corr_log"]), 2) for r in last],
                      " misfit %:", round(float(last[0]["misfit_pct"]), 1),
                      " theta RMSE:", round(float(last[0]["theta_rmse"]), 3))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "ec_cause_summary.csv"), index=False)
    write_report(df, controls, shares, truth_ec, os.path.join(args.out, "ec_cause_report.md"))
    write_figure(df, os.path.join(args.out, "fig_ec_cause_ratio.png"))
    print(f"\nWrote {args.out}/ec_cause_summary.csv, ec_cause_report.md, fig_ec_cause_ratio.png")


if __name__ == "__main__":
    main()
