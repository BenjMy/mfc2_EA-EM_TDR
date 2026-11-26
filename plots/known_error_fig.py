"""Error on the known values (``fig_known_error.png`` of study_swc_design.py).

One column per truth (smc, ws), two rows against ``err`` (error of the pinned values, in %):

    top     RMSE of theta vs truth, one line per "what is known" (pin)
    bottom  (fitted - truth) / truth of the FREE parameters, median over the calibrated horizons,
            one line per (pin, parameter): colour = pin, marker = parameter

Inputs: the study summary (DataFrame) and the rows of ``ws_recovery.collect``.  Only the reference
n / distribution / regularisation is drawn (or the first one present).
"""
from __future__ import annotations

import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from plots._common import ensure_dir, save_fig  # noqa: E402

_PIN_COLORS = {"sw": "#2a78d6", "sws": "#eb6834", "am": "#1baf7a"}
_PARAM_MARKERS = {"a": "o", "m": "s", "sigma_w": "^", "sigma_s": "D"}
_PARAM_LABELS = {"a": "a", "m": "m", "sigma_w": "σw", "sigma_s": "σs"}


def free_param_errors(rec: list[dict], scenario: str, free: list[str]) -> dict[str, float]:
    """{parameter: median over calibrated horizons of (fit - truth) / truth [%]} of one run."""
    out = {}
    for p in free:
        vals = []
        for r in rec:
            if r.get("scenario") != scenario or r.get("calibrated") in (False, "False"):
                continue
            fit, truth = r.get(f"{p}_fit"), r.get(f"{p}_truth")
            if fit in (None, "") or truth in (None, "") or float(truth) == 0.0:
                continue
            vals.append(100.0 * (float(fit) - float(truth)) / float(truth))
        if vals:
            out[p] = statistics.median(vals)
    return out


def build_figure(df, rec: list[dict], err_pins, pin_known: dict, params, reference: dict):
    """Figure (None when no run with a non-zero error has a result)."""
    d = df[df["pin"].isin(err_pins) & df["rmse_truth"].notna()]
    for f in ("n", "dist", "reg"):
        if len(d):
            d = d[d[f] == (reference[f] if (d[f] == reference[f]).any() else d[f].iloc[0])]
    if d.empty or not (d["err"] != 0).any():
        return None
    # a truth is drawn only when it has at least one perturbed run
    truths = sorted(set(zip(d.loc[d["err"] != 0, "smc"], d.loc[d["err"] != 0, "ws"])),
                    key=lambda t: (t[1] != "homo", t))        # homogeneous WS (primary result) first
    pins = [p for p in err_pins if p in set(d["pin"])]
    errs = sorted(int(e) for e in d["err"].unique())
    baseline_only = {r.get("scenario") for r in rec if r.get("truth_baseline_only") in (True, "True")}

    fig, axes = plt.subplots(2, len(truths), figsize=(max(4.2 * len(truths), 6.4), 6.6), sharex=True, sharey="row",
                             squeeze=False)
    used_params: set[str] = set()
    for j, (smc, ws) in enumerate(truths):
        top, bot = axes[0, j], axes[1, j]
        sub = d[(d["smc"] == smc) & (d["ws"] == ws)]
        for pin in pins:
            g = sub[sub["pin"] == pin].sort_values("err")
            if g.empty:
                continue
            col = _PIN_COLORS.get(pin)
            top.plot(g["err"], g["rmse_truth"], "o-", color=col, lw=2, ms=6)
            free = [p for p in params if p not in pin_known[pin]]
            per_run = [(int(r["err"]), free_param_errors(rec, r["scenario"], free)) for _, r in g.iterrows()]
            for p in free:
                pts = [(e, v[p]) for e, v in per_run if p in v]
                if pts:
                    used_params.add(p)
                    bot.plot(*zip(*pts), marker=_PARAM_MARKERS[p], color=col, lw=1.4, ms=6, alpha=0.9)
        note = " (vs layer baseline)" if baseline_only & set(sub["scenario"]) else ""
        top.set_title(f"SMC {smc} / WS {ws}", fontsize=9)
        bot.set_title(f"free parameters{note}", fontsize=8)
        bot.axhline(0, color="0.4", lw=0.8)
        bot.set_xlabel("error of the known values [%]")
        bot.set_xticks(errs)
        for ax in (top, bot):
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].set_ylabel("RMSE θ [m³/m³]")
    axes[1, 0].set_ylabel("(fitted − truth) / truth [%]")
    handles = [Line2D([], [], color=_PIN_COLORS.get(p), lw=2,
                      label="known: " + ", ".join(_PARAM_LABELS[q] for q in pin_known[p])) for p in pins]
    handles += [Line2D([], [], color="0.3", marker=_PARAM_MARKERS[p], ls="none", label=f"fitted {_PARAM_LABELS[p]}")
                for p in params if p in used_params]
    fig.suptitle("Effect of an error on the known Waxman-Smits values", fontsize=10)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False)
    return fig


def plot_known_error(df, rec, out_dir: str, err_pins, pin_known, params, reference,
                     fname: str = "fig_known_error.png") -> str | None:
    fig = build_figure(df, rec, err_pins, pin_known, params, reference)
    if fig is None:
        return None
    path = os.path.join(ensure_dir(out_dir), fname)
    save_fig(fig, path, dpi=140, tag="STUDY")
    return path
