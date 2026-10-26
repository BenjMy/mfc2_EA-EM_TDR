#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_probe_matching.py
=======================
Diagnostic visualisation of the TDR-probe <-> EM-measurement-point spatial
matching performed by ``em_methodology.extract_sample_positions_from_survey``.

Why this exists
----------------
``extract_sample_positions_from_survey`` can map many TDR probes onto the
same EM ``representative_index`` (e.g. 11 of 20 stations collapsing onto a
single EM point in a real-data run), and can also silently fall back to a
"nearest point" match when no EM measurement falls within ``window_m`` of a
probe. Both situations are known internally (``SamplePositionResult
.representative_index`` / ``.used_fallback``) but previously only surfaced
as a per-probe ``print()`` that's easy to miss in a long pipeline log.

This module renders that matching directly:

  * every EM measurement point, coloured by how many probes were matched
    to it (a "collapse hotspot" map),
  * every probe location, connected by a thin line to the EM point it was
    actually matched to,
  * fallback (no-window-match) probes drawn with a distinct marker/colour
    so they're visually distinguishable from genuine window matches.

Public API
----------
    plot_probe_matching(survey_df, position_results, output_dir, ...)

Typical call (from run_pipeline.py, right after
``EMmethod.extract_sample_positions_from_survey`` / after building
``position_results`` and before/after ``EMmethod.summarize_probe_matching``)::

    import plot_probe_matching as ppm

    ppm.plot_probe_matching(
        survey_df=survey_df,
        position_results=position_results,
        output_dir=OUTPUT_DIR,
        summary=matching_summary,   # optional EMmethod.ProbeMatchingSummary
        title="SWC/TDR probe -> EM point matching",
    )
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_probe_matching(
    survey_df,
    position_results: Sequence[Any],
    output_dir: str,
    summary: Any | None = None,
    title: str = "TDR probe -> EM measurement point matching",
    filename: str = "probe_em_matching.png",
    cmap: str = "Reds",
) -> str:
    """
    Plot every EM measurement point coloured by how many TDR probes were
    matched to it, with each probe connected by a thin line to its matched
    EM point. Window-fallback matches (no EM point found within
    ``window_m``) are drawn with a distinct marker/colour.

    Parameters
    ----------
    survey_df : pd.DataFrame
        ``k.surveys[0].df`` (or the ``survey_df`` used elsewhere in the
        pipeline) — must contain an ``'x'`` column and, optionally, a
        ``'y'`` column (treated as all-zero for pure 1-D transects).
    position_results : sequence of em_methodology.SamplePositionResult
        Output of ``EMmethod.extract_sample_positions_from_survey``.
    output_dir : str
        Figures are written to ``<output_dir>/diagnostics/``.
    summary : em_methodology.ProbeMatchingSummary, optional
        When supplied, its ``.message`` is embedded in the figure subtitle
        so the plot is self-explanatory without cross-referencing the log.
    title : str, optional
        Figure title.
    filename : str, optional
        Output PNG filename (default ``"probe_em_matching.png"``).
    cmap : str, optional
        Colormap for the "probes matched per EM point" scale.

    Returns
    -------
    str
        Path to the saved figure. Empty string if there was nothing to plot
        (e.g. no probes).
    """
    from collections import Counter

    diag_dir = _ensure_dir(os.path.join(output_dir, "diagnostics"))

    if not position_results:
        print("[MATCHING PLOT][WARN] No position_results supplied — skipping plot.")
        return ""

    x_em = survey_df["x"].to_numpy(dtype=float)
    y_em = (survey_df["y"].to_numpy(dtype=float)
            if "y" in survey_df.columns else np.zeros_like(x_em))

    rep_indices = [int(r.representative_index) for r in position_results]
    counts = Counter(rep_indices)

    match_count = np.zeros(len(x_em), dtype=int)
    for idx, c in counts.items():
        if 0 <= idx < len(match_count):
            match_count[idx] = c

    matched_mask = match_count > 0
    n_matched_points = int(matched_mask.sum())

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # All EM points, unmatched ones shown faint/grey for spatial context
    ax.scatter(
        x_em[~matched_mask], y_em[~matched_mask],
        s=8, color="lightgrey", zorder=1, linewidths=0,
        label="EM point (unmatched)",
    )

    vmax = max(1, int(match_count.max()))
    sc = ax.scatter(
        x_em[matched_mask], y_em[matched_mask],
        c=match_count[matched_mask], cmap=cmap, vmin=1, vmax=vmax,
        s=55, edgecolor="k", linewidth=0.4, zorder=3,
        label="EM point (matched)",
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probes matched to this EM point", fontsize=9)

    # Connector lines + probe markers, split by genuine vs fallback match
    _genuine_labelled = False
    _fallback_labelled = False
    for r in position_results:
        rep_idx = int(r.representative_index)
        if not (0 <= rep_idx < len(x_em)):
            continue
        ex, ey = x_em[rep_idx], y_em[rep_idx]

        is_fallback = bool(getattr(r, "used_fallback", False))
        line_color = "tab:orange" if is_fallback else "grey"
        marker = "x" if is_fallback else "v"
        marker_color = "tab:orange" if is_fallback else "black"

        ax.plot([r.probe_x, ex], [r.probe_y, ey],
                color=line_color, linewidth=0.7, alpha=0.65, zorder=2)

        label = None
        if is_fallback and not _fallback_labelled:
            label = "Probe — window fallback (nearest point)"
            _fallback_labelled = True
        elif (not is_fallback) and not _genuine_labelled:
            label = "Probe — genuine window match"
            _genuine_labelled = True

        ax.scatter([r.probe_x], [r.probe_y], marker=marker, s=55,
                   color=marker_color, edgecolor="white", linewidth=0.6,
                   zorder=4, label=label)

        probe_id = getattr(r, "probe_id", None)
        if probe_id is not None:
            ax.annotate(
                str(probe_id), (r.probe_x, r.probe_y),
                xytext=(4, 4), textcoords="offset points",
                fontsize=6.5, color=marker_color, zorder=5,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            )

    ax.set_xlabel("x [m]" if y_em.any() else "Distance along transect [m]", fontsize=10)
    ax.set_ylabel("y [m]", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)

    n_probes = len(position_results)
    subtitle = (
        f"{n_probes} probe(s)  ->  {n_matched_points} distinct EM point(s) used  |  "
        f"largest group: {int(match_count.max())} probe(s)"
    )
    if summary is not None and getattr(summary, "message", None):
        subtitle += f"\n{summary.message}"

    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    fig.tight_layout()
    out_path = os.path.join(diag_dir, filename)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[MATCHING PLOT] Saved -> {out_path}")
    return out_path
