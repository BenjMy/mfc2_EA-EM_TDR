#!/usr/bin/env python3
"""Level 0 of the "does surface-only TDR still help the deep layers?" study.

No new inversion: re-uses the saved ensemble of ONE synthetic run
(``output_model/ensemble_members.npz`` + ``member_ranking.npy`` +
``member_rmse.npy`` + truth ``synthetic_time_lapse_ec.npy``) and compares, per
layer, the EC error vs truth of four stacks of the SAME size K (K = the
optimal stack depth the pipeline chose):

  all      equal-weight mean of ALL members (no selection)
  tdr      the pipeline's choice: top-K by TDR misfit, RMSE-weighted
  oracle   top-K by TRUE layer-0 EC error, equal weights  (the "layer 0
           pinned/known" ceiling that selection alone can reach)
  random   K random members, equal weights; mean and 5-95 % over many draws

and checks the trade-off hypothesis directly: across members, is the layer-0
error anti-correlated with the layer-1..3 errors?  (Pooled over every
(time, point) cell, correlation across members of the signed errors, then
averaged over cells; plus the correlation of the per-member mean errors.)

How to read it
--------------
* oracle << tdr at layers 1-3  -> knowing layer 0 would help; pinning is worth
  building (Level 1/2).  oracle ~ tdr ~ random -> selection on layer 0 buys
  nothing and the pinned-layer idea needs the actual conditioning, not just
  better member selection.
* corr(e0, e_l) clearly < 0 -> members that overshoot layer 0 undershoot the
  deeper layers (non-uniqueness trade-off).  ~0 or > 0 -> no such trade-off
  in this ensemble.
Selection is a weaker test than pinning inside the inversion (the members
never saw a pinned layer), so a null result here does not rule pinning out.

Usage:  python analysis_pinned_ceiling.py RUN_DIR [--realization 0] [--n-random 2000]
Writes  RUN_DIR/analysis_pinned_ceiling/{stack_errors.csv,correlations.csv,
        summary.md,fig_pinned_ceiling.png}
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import swc_uncertainty as su


def _err(stack, truth):
    """Per-layer RMSE and bias of a (T,P,L) stack vs truth (T,P,L)."""
    d = stack - truth
    return (np.sqrt(np.nanmean(d ** 2, axis=(0, 1))), np.nanmean(d, axis=(0, 1)))


def member_layer_errors(models, truth):
    """(N, L) per-member RMSE and signed mean error per layer."""
    d = models - truth[None]
    return np.sqrt(np.nanmean(d ** 2, axis=(1, 2))), np.nanmean(d, axis=(1, 2))


def stack_comparison(models, truth, ranking, member_rmse, K, n_random=2000, seed=0,
                     layer0=0):
    """Return {name: (rmse_L, bias_L)}, plus the random-draw 5/95 % bands."""
    N = models.shape[0]
    K = int(min(max(K, 1), N))
    out = {}
    out["all"] = _err(np.nanmean(models, axis=0), truth)
    top = np.asarray(ranking)[:K]
    w = su.rmse_weights(np.asarray(member_rmse)[top])
    out["tdr"] = _err(np.tensordot(w, models[top], axes=1), truth)
    rm0, _ = member_layer_errors(models, truth)
    orc = np.argsort(rm0[:, layer0])[:K]
    out["oracle"] = _err(np.nanmean(models[orc], axis=0), truth)
    rng = np.random.default_rng(seed)
    r_rmse, r_bias = [], []
    for _ in range(n_random):
        sel = rng.choice(N, K, replace=False)
        e = _err(np.nanmean(models[sel], axis=0), truth)
        r_rmse.append(e[0]); r_bias.append(e[1])
    r_rmse, r_bias = np.array(r_rmse), np.array(r_bias)
    out["random"] = (r_rmse.mean(0), r_bias.mean(0))
    band = (np.percentile(r_rmse, 5, axis=0), np.percentile(r_rmse, 95, axis=0))
    return out, band, top, orc


def layer0_vs_deep_correlation(models, truth, layer0=0):
    """Correlation of layer-0 error with each deeper layer's error across members.

    cell_corr[l]  : mean over (t,p) cells of corr_over_members(e0, e_l), signed errors
    member_corr[l]: corr over members of the per-member mean signed error
    """
    d = models - truth[None]                      # (N,T,P,L)
    N, T, P, L = d.shape
    e0 = d[..., layer0]
    cell, memb = {}, {}
    for l in range(L):
        if l == layer0:
            continue
        el = d[..., l]
        a = e0 - e0.mean(0, keepdims=True)
        b = el - el.mean(0, keepdims=True)
        den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
        with np.errstate(invalid="ignore", divide="ignore"):
            c = (a * b).sum(0) / den
        cell[l] = float(np.nanmean(c))
        x, y = np.nanmean(e0, axis=(1, 2)), np.nanmean(el, axis=(1, 2))
        memb[l] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return cell, memb


def run(run_dir, realization=0, n_random=2000, seed=0):
    om = os.path.join(run_dir, "output_model")
    z = np.load(os.path.join(om, "ensemble_members.npz"))
    models = np.asarray(z["models_clean"], float)
    if models.ndim == 3:                          # single survey -> add time axis
        models = models[:, None]
    truth = np.load(os.path.join(om, "synthetic_time_lapse_ec.npy"))[realization]
    ranking = np.load(os.path.join(om, "member_ranking.npy"))
    mrmse = np.load(os.path.join(om, "member_rmse.npy"))
    K = int(np.load(os.path.join(om, "K.npy")))
    res, band, top, orc = stack_comparison(models, truth, ranking, mrmse, K, n_random, seed)
    cell, memb = layer0_vs_deep_correlation(models, truth)
    overlap = len(set(top) & set(orc)) / max(K, 1)
    return dict(models=models, truth=truth, K=K, stacks=res, band=band, top=top, oracle=orc,
                overlap=overlap, cell_corr=cell, member_corr=memb)


def write_outputs(r, run_dir):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(run_dir, "analysis_pinned_ceiling")
    os.makedirs(out, exist_ok=True)
    L = r["truth"].shape[-1]
    names = ["all", "tdr", "oracle", "random"]
    with open(os.path.join(out, "stack_errors.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stack", "layer", "ec_rmse_mS_m", "ec_bias_mS_m", "K"])
        for n in names:
            for l in range(L):
                w.writerow([n, l, f"{r['stacks'][n][0][l]:.3f}", f"{r['stacks'][n][1][l]:.3f}", r["K"]])
    with open(os.path.join(out, "correlations.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "corr_cellwise_with_layer0", "corr_member_mean_with_layer0"])
        for l in r["cell_corr"]:
            w.writerow([l, f"{r['cell_corr'][l]:.3f}", f"{r['member_corr'][l]:.3f}"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(L); wd = 0.2
    for i, n in enumerate(names):
        ax[0].bar(x + (i - 1.5) * wd, r["stacks"][n][0], wd, label=n)
    ax[0].errorbar(x + 1.5 * wd, r["stacks"]["random"][0],
                   yerr=[r["stacks"]["random"][0] - r["band"][0], r["band"][1] - r["stacks"]["random"][0]],
                   fmt="none", ecolor="k", capsize=2)
    ax[0].set_xticks(x); ax[0].set_xlabel("layer"); ax[0].set_ylabel("EC RMSE vs truth [mS/m]")
    ax[0].set_title(f"Same-size stacks (K={r['K']})"); ax[0].legend(fontsize=8)
    rm0, b0 = member_layer_errors(r["models"], r["truth"])
    _, bl = member_layer_errors(r["models"], r["truth"])
    for l in range(1, L):
        ax[1].scatter(b0[:, 0], bl[:, l], s=12, alpha=0.6,
                      label=f"layer {l} (r={r['member_corr'][l]:+.2f})")
    ax[1].axhline(0, c="0.6", lw=0.8); ax[1].axvline(0, c="0.6", lw=0.8)
    ax[1].set_xlabel("member mean error, layer 0 [mS/m]")
    ax[1].set_ylabel("member mean error, deeper layer [mS/m]")
    ax[1].set_title("Layer-0 vs deep-layer error across members"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_pinned_ceiling.png"), dpi=130); plt.close(fig)

    lines = [f"# Level 0: pinned-layer ceiling ({os.path.basename(run_dir.rstrip('/'))})", "",
             f"K = {r['K']} members; overlap of TDR-selected and oracle-selected sets: {r['overlap']:.0%}", "",
             "| stack | " + " | ".join(f"L{l} RMSE (bias)" for l in range(L)) + " |",
             "|---|" + "---|" * L]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(
            f"{r['stacks'][n][0][l]:.1f} ({r['stacks'][n][1][l]:+.1f})" for l in range(L)) + " |")
    lines += ["", "Random 5-95 % RMSE band per layer: " +
              ", ".join(f"L{l} {r['band'][0][l]:.1f}-{r['band'][1][l]:.1f}" for l in range(L)), "",
              "Correlation of layer-0 error with deeper-layer error across members "
              "(negative = trade-off):", ""]
    for l in r["cell_corr"]:
        lines.append(f"- layer {l}: cellwise {r['cell_corr'][l]:+.2f}, member-mean {r['member_corr'][l]:+.2f}")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--realization", type=int, default=0)
    ap.add_argument("--n-random", type=int, default=2000)
    a = ap.parse_args()
    res = run(a.run_dir, a.realization, a.n_random)
    print(write_outputs(res, a.run_dir))
    print(open(os.path.join(a.run_dir, "analysis_pinned_ceiling", "summary.md")).read())
