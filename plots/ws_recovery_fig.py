"""Waxman-Smits parameter recovery figure (``fig_ws_recovery.png`` of study_swc_design.py).

Input: the rows of ``ws_recovery.collect`` (one per scenario x horizon).  One panel per parameter
(a, m, sigma_w, sigma_s) plus one for C = a / (sigma_w + sigma_s); x = scenario; per scenario one small
column per horizon (colour = horizon):

    o  fitted value, FILLED when the parameter was free, HOLLOW when it was pinned
    -  truth of that horizon (faded when it is only the per-layer baseline of a spatially varying truth)
    x  calibration start
    s  grey square: horizon without TDR probe (prior only, nothing was fitted)
    shaded rectangle  parameter bounds of that run (not for C)
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from plots._common import ensure_dir, save_fig  # noqa: E402

PANELS = (("a", "a [-]"), ("m", "m [-]"), ("sigma_w", "σw [mS/m]"), ("sigma_s", "σs [mS/m]"),
          ("C", "C = a/(σw+σs) [m/mS]"))
_HZ_COLORS = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02")
_PREFIX = "swcstudy_"


def short_label(name: str) -> str:
    return name[len(_PREFIX):] if name.startswith(_PREFIX) else name


def _val(r: dict, key: str):
    v = r.get(key)
    return None if v in (None, "") else v


def build_figure(rows: list[dict]):
    """Figure of the recovery rows (None when there is nothing to draw)."""
    rows = [r for r in rows if r.get("scenario") != ""]
    if not rows:
        return None
    scenarios = sorted({r["scenario"] for r in rows})
    horizons = sorted({int(r["horizon"]) for r in rows})
    xi = {s: i for i, s in enumerate(scenarios)}
    width = 0.8
    step = width / max(len(horizons), 1)
    off = {h: -width / 2 + step * (j + 0.5) for j, h in enumerate(horizons)}

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(max(7.0, 0.55 * len(scenarios) + 3.0), 2.3 * len(PANELS)),
                             sharex=True)
    drawn_bounds: set[tuple] = set()
    for ax, (p, ylabel) in zip(axes, PANELS):
        ys: list[float] = []
        for r in rows:
            x = xi[r["scenario"]] + off[int(r["horizon"])]
            col = _HZ_COLORS[horizons.index(int(r["horizon"])) % len(_HZ_COLORS)]
            if p != "C":
                lo, hi = _val(r, f"{p}_lo"), _val(r, f"{p}_hi")
                key = (r["scenario"], p)
                if lo is not None and hi is not None and key not in drawn_bounds:
                    drawn_bounds.add(key)
                    ax.add_patch(Rectangle((xi[r["scenario"]] - width / 2, lo), width, hi - lo,
                                           color="0.85", alpha=0.6, lw=0, zorder=0))
            fit, truth, start = _val(r, f"{p}_fit"), _val(r, f"{p}_truth"), _val(r, f"{p}_start")
            fixed = False if p == "C" else bool(_val(r, f"{p}_fixed"))
            calibrated = r.get("calibrated") not in (False, "False")
            ys += [float(v) for v in (fit, truth, start) if v is not None]
            if truth is not None:
                faded = r.get("truth_baseline_only") in (True, "True")
                ax.plot([x - step * 0.45, x + step * 0.45], [truth, truth], color=col, lw=2.2,
                        alpha=0.35 if faded else 1.0, zorder=2, solid_capstyle="butt")
            if start is not None:
                ax.plot(x, start, marker="x", ms=4, color=col, ls="none", alpha=0.8, zorder=2)
            if fit is not None:
                if not calibrated:
                    ax.plot(x, fit, marker="s", ms=4.5, mfc="0.7", mec="0.4", ls="none", zorder=3)
                else:
                    ax.plot(x, fit, marker="o", ms=5, mfc="none" if fixed else col, mec=col, mew=1.3,
                            ls="none", zorder=3)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=7)
        if ys:                       # zoom on the data; the bounds rectangles are clipped at the panel edge
            lo_y, hi_y = min(ys), max(ys)
            pad = 0.12 * (hi_y - lo_y) if hi_y > lo_y else 0.1 * abs(hi_y) or 1.0
            ax.set_ylim(lo_y - pad, hi_y + pad)
    axes[-1].set_xticks(range(len(scenarios)))
    axes[-1].set_xticklabels([short_label(s) for s in scenarios], rotation=70, ha="right", fontsize=6)
    axes[-1].set_xlim(-0.6, len(scenarios) - 0.4)

    handles = [Line2D([], [], marker="o", color="k", mfc="k", ls="none", label="fitted (free)"),
               Line2D([], [], marker="o", color="k", mfc="none", ls="none", label="fitted (pinned)"),
               Line2D([], [], color="k", lw=2.2, label="truth"),
               Line2D([], [], marker="x", color="k", ls="none", label="start"),
               Line2D([], [], marker="s", mfc="0.7", mec="0.4", ls="none", label="prior only (no probe)"),
               Rectangle((0, 0), 1, 1, color="0.85", label="bounds")]
    handles += [Line2D([], [], color=_HZ_COLORS[j % len(_HZ_COLORS)], lw=3, label=f"horizon {h}")
                for j, h in enumerate(horizons)]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 7), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Waxman-Smits parameter recovery: start, fitted and truth", y=1.04, fontsize=10)
    fig.tight_layout()
    return fig


def plot_ws_recovery(rows: list[dict], out_dir: str, fname: str = "fig_ws_recovery.png") -> str | None:
    fig = build_figure(rows)
    if fig is None:
        return None
    path = os.path.join(ensure_dir(out_dir), fname)
    save_fig(fig, path, dpi=140, tag="STUDY")
    return path
