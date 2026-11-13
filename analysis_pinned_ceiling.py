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


INTRO = """## Hypothesis being checked

Surface-only TDR calibrates only horizon 0, yet the question is whether this still improves the
DEEP layers (1-3) of the EC / SWC model. The EM inversion never sees TDR data, so the ECa data
can be explained by trading error between layer 0 and layers 1-3 (non-uniqueness). If layer 0
were known (pinned), that trade-off would disappear and the deep layers should improve.

Level 0 tests this WITHOUT any new inversion, on the ensemble already computed:
(a) if members that fit layer 0 better ALSO fit layers 1-3 better, then picking members by a
layer-0 criterion (which is all surface TDR can offer) helps the deep layers; (b) if layer-0
error anti-correlates with layer-1..3 error across members, that is the trade-off itself.

## The four stacks (all of the same size K = optimal stack depth chosen by the pipeline)

| stack | what it is | what it tells us |
|---|---|---|
| `all` | equal-weight mean of ALL ensemble members, no selection | reference: no TDR information used |
| `tdr` | the pipeline's own result: top-K members ranked by misfit to the surface TDR (Step 2/3), RMSE-weighted | what surface-only TDR actually delivers |
| `oracle` | top-K members ranked by their TRUE layer-0 EC error vs the synthetic truth, equal weights | ceiling of what selection on a perfectly known layer 0 can reach ("pinned layer 0" by selection) |
| `random` | K members drawn at random (mean over many draws; 5-95 % band in the table) | null model: any gain over `random` is due to the selection criterion, not to stacking |

Reading the table: `tdr` < `random` at layers 1-3 means surface TDR helps the deep layers;
`oracle` < `tdr` there means a known layer 0 would help more (pinning is worth building);
`oracle` ~ `tdr` ~ `random` or `oracle` worse means selection cannot help and only conditioning
INSIDE the inversion (a pinned layer) could. Errors are bulk EC in mS/m, bias = stack - truth.

## How to reproduce (synthetic data only, one realisation per run)

Needs the code with `ensemble_members.npz`, `member_ranking.npy` and `member_rmse.npy` saved
(commits "Save raw ensemble members ..." and "Level 0 analysis ..."), and `emagpy` 1.4.6.

```bash
# 1. run the scenario (about 7 min on one CPU; 5 surface probes at 0.2 m, K=2 free WS parameters)
./run_pipeline.sh -s 1d_surface_hetero_free2 --mode 1d --num-swc-positions 5   # two-zone vertical truth
./run_pipeline.sh -s 1d_surface_free2        --mode 1d --num-swc-positions 5   # uniform-in-depth truth
# 2. analyse the run folder (the latest numbered folder under mfc2_outputs/field_survey_<scenario>/)
python analysis_pinned_ceiling.py mfc2_outputs/field_survey_1d_surface_hetero_free2/<seq>_1d_surface_hetero_free2
python analysis_pinned_ceiling.py mfc2_outputs/field_survey_1d_surface_free2/<seq>_1d_surface_free2
```
Outputs land in `<run>/analysis_pinned_ceiling/` (`summary.md`, `stack_errors.csv`,
`correlations.csv`, `fig_pinned_ceiling.png`). `--n-random N` sets the number of random draws
(default 2000, seed fixed). Tests: `pytest tests/test_analysis_pinned_ceiling.py`.
Run-to-run reproducibility of the inversion itself has not been confirmed (seeding patch
untested), so numbers can change slightly on a rerun.

## Limits of this test

One truth realisation per run; K can be small (a few members), which makes `oracle` and `random`
noisy; members never saw a pinned layer, so selection is a weaker test than pinning inside the
inversion (a null result here does NOT rule pinning out); a bias shared by all members (e.g. from
the 300 mS/m EC upper bound) cannot be removed by any selection.
"""

OUTRO = """
## Perspectives: more complex procedures to test the hypothesis

1. **Oracle pin inside the inversion (Level 1).** Fix layer 0 to its TRUE EC in the Step 1
   inversion (EMagPy `fixedConds`); scenarios `1d_surface_free4_pin0` and
   `1d_surface_hetero_free4_pin0` already exist. Upper bound of what pinning can give.
2. **Pin to a TDR-derived EC (Level 2).** TDR theta at the probes -> calibrated Waxman-Smits ->
   EC of horizon 0, interpolated along x (per epoch for time-lapse); pin layer 0 to it
   (new `pin_source: "tdr"`, horizon 0 first, then any calibrated horizon).
3. **Honest uncertainty (Level 3).** Give each member its own EC draw for the pinned layer
   (spread = calibration + interpolation error) so pinning does not make the deep layers look
   more certain than they are.
4. **Full comparison (Level 4).** Per horizon, EC and theta RMSE/bias vs truth for: free (now),
   pinned-to-TDR (realistic), pinned-to-truth (oracle), full-depth TDR (reference). Success =
   pinned-to-TDR beats free at H1-H3 and sits between free and oracle.
5. **Robustness.** Repeat over several truth seeds / realisations and report mean and spread;
   vary K, the 300 mS/m EC upper bound (try ~150), the number of probes, and the uniform vs
   hetero truth; confirm run-to-run reproducibility first.
6. **Beyond 1-D synthetic.** 2-D grids and time-lapse (one pinned profile per epoch), then the
   real data, where horizon 0 may be thicker than the ~0.2 m TDR footprint (representativeness
   error) and horizons without sensors still need WS parameters (prior or borrowed from H0).
"""


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
    body = "\n".join(lines)
    lines2 = body.split("\n", 1)
    text = lines2[0] + "\n" + INTRO + "\n## Results\n" + (lines2[1] if len(lines2) > 1 else "") + "\n" + OUTRO
    open(os.path.join(out, "summary.md"), "w").write(text)
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
