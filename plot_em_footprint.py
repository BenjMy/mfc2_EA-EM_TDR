#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_em_footprint.py
=====================
Plots for the Step 5c EM footprint discrepancy diagnostic (McNeill depth
weighting, see ``em_sensitivity.py`` and run_pipeline.py's
``--em-footprint-diagnostic``). Takes the same per-probe discrepancy
records Step 5c already builds and logs to the report -- no new
computation, just visualising what's already there.

Public API
----------
    plot_footprint_probe_comparison(discrepancies, output_dir, ...)
    plot_footprint_spatial_discrepancy(discrepancies, survey_df, output_dir, ...)
    plot_footprint_spatial_discrepancy_xz(discrepancies, survey_df, depths0, output_dir, ...)

Typical call (from run_pipeline.py, right after the Step 5c RMSE/table
reporting, inside the same ``if em_footprint_diagnostic:`` block)::

    import plot_em_footprint as pef

    pef.plot_footprint_probe_comparison(
        discrepancies=_footprint_discrepancies,
        output_dir=OUTPUT_DIR,
        hr_rmse=_hr_rmse, w_rmse=_w_rmse,
    )
    pef.plot_footprint_spatial_discrepancy(
        discrepancies=_footprint_discrepancies,
        survey_df=survey_df,
        output_dir=OUTPUT_DIR,
    )

Each ``discrepancies`` entry is one dict per probe observation, with keys
``probe`` (int), ``pos`` (int -- index into the EM survey grid / theta_calc's
``num_measured_points`` axis), ``obs``, ``hard_routed``, ``weighted``,
``hard_routed_err``, ``weighted_err`` (all float, m3/m3) -- exactly the
records Step 5c already builds for its own RMSE/table reporting.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_footprint_probe_comparison(
    discrepancies: Sequence[dict],
    output_dir: str,
    hr_rmse: float | None = None,
    w_rmse: float | None = None,
    title: str = "EM footprint discrepancy — per-probe comparison",
    filename: str = "em_footprint_probe_comparison.png",
) -> str:
    """
    Grouped bar chart, one probe per x-tick: hard-routed error vs.
    McNeill-weighted error (both ``calc - obs``, m3/m3), so over/under-
    prediction and which method is closer to zero are both visible at a
    glance per probe.

    Parameters
    ----------
    discrepancies : sequence of dict
        Step 5c's own per-probe records (see module docstring for keys).
    output_dir : str
        Figures are written to ``<output_dir>/diagnostics/``.
    hr_rmse, w_rmse : float, optional
        Overall RMSEs (already computed by Step 5c) to annotate in the
        subtitle, so the figure is self-explanatory without cross-
        referencing the report text.
    title : str, optional
        Figure title.
    filename : str, optional
        Output PNG filename.

    Returns
    -------
    str
        Path to the saved figure. Empty string if there was nothing to plot.
    """
    diag_dir = _ensure_dir(os.path.join(output_dir, "diagnostics"))

    if not discrepancies:
        print("[EM FOOTPRINT][WARN] No discrepancy records supplied — skipping probe comparison plot.")
        return ""

    probes = [d["probe"] for d in discrepancies]
    hard_err = np.array([d["hard_routed_err"] for d in discrepancies], dtype=float)
    weighted_err = np.array([d["weighted_err"] for d in discrepancies], dtype=float)

    n = len(probes)
    x = np.arange(n)
    bar_w = 0.38

    fig, ax = plt.subplots(figsize=(max(8.0, 0.5 * n), 5.5))
    ax.bar(x - bar_w / 2, hard_err, width=bar_w, color="tab:orange",
           edgecolor="k", linewidth=0.3, label="Hard-routed (nearest layer)")
    ax.bar(x + bar_w / 2, weighted_err, width=bar_w, color="tab:blue",
           edgecolor="k", linewidth=0.3, label="McNeill-weighted (all layers)")
    ax.axhline(0.0, color="grey", linewidth=0.8, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in probes], fontsize=8)
    ax.set_xlabel("Probe #", fontsize=10)
    ax.set_ylabel(r"$\theta_{calc} - \theta_{obs}$  [m$^3$/m$^3$]", fontsize=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)

    subtitle = f"{n} probe observation(s)"
    if hr_rmse is not None and w_rmse is not None:
        subtitle += f"  |  RMSE: hard-routed {hr_rmse:.5f}, weighted {w_rmse:.5f} m³/m³"
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(diag_dir, filename)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[EM FOOTPRINT] Saved per-probe comparison plot -> {out_path}")
    return out_path


def plot_footprint_spatial_discrepancy(
    discrepancies: Sequence[dict],
    survey_df,
    output_dir: str,
    probe_x: np.ndarray | None = None,
    probe_y: np.ndarray | None = None,
    title: str = "EM footprint discrepancy — spatial map",
    filename: str = "em_footprint_spatial_discrepancy.png",
    cmap: str = "RdBu_r",
) -> str:
    """
    Map, over the EM survey grid, whether the hard-routed or the
    McNeill-weighted comparison sits closer to the TDR observation at each
    matched position: ``mean(|hard_routed_err|) - mean(|weighted_err|)``,
    averaged when more than one discrepancy record shares the same ``pos``
    (e.g. multiple horizons/time steps routed through the same EM point).

    Diverging colour scale centred at 0: positive (one colour) = the
    McNeill-weighted comparison is closer to TDR at that point; negative
    (the other colour) = hard-routed is closer. EM points with no matched
    probe observation are drawn faint/grey for spatial context, same
    convention as ``plot_calibration_density``.

    Parameters
    ----------
    discrepancies : sequence of dict
        Step 5c's own per-probe records; must include ``pos`` (index into
        ``survey_df`` / theta_calc's ``num_measured_points`` axis).
    survey_df : pd.DataFrame
        Same frame used elsewhere in the pipeline for EM point coordinates
        -- must contain ``'x'`` and, optionally, ``'y'``.
    output_dir : str
        Figures are written to ``<output_dir>/diagnostics/``.
    probe_x, probe_y : np.ndarray, optional
        TDR probe locations, for overlay only (e.g. from
        ``CALIB_POSITION_RESULTS``). Omit to skip the overlay.
    title : str, optional
        Figure title.
    filename : str, optional
        Output PNG filename.
    cmap : str, optional
        Diverging colormap for the improvement scale.

    Returns
    -------
    str
        Path to the saved figure. Empty string if there was nothing to plot.
    """
    diag_dir = _ensure_dir(os.path.join(output_dir, "diagnostics"))

    if not discrepancies:
        print("[EM FOOTPRINT][WARN] No discrepancy records supplied — skipping spatial discrepancy plot.")
        return ""

    x_em = survey_df["x"].to_numpy(dtype=float)
    y_em = (survey_df["y"].to_numpy(dtype=float)
            if "y" in survey_df.columns else np.zeros_like(x_em))

    per_pos: dict[int, list[float]] = {}
    for d in discrepancies:
        pos = int(d["pos"])
        improvement = abs(d["hard_routed_err"]) - abs(d["weighted_err"])
        per_pos.setdefault(pos, []).append(improvement)

    improvement_map = np.full(len(x_em), np.nan, dtype=float)
    for pos, values in per_pos.items():
        if 0 <= pos < len(improvement_map):
            improvement_map[pos] = float(np.mean(values))

    matched_mask = ~np.isnan(improvement_map)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.scatter(
        x_em[~matched_mask], y_em[~matched_mask],
        s=8, color="lightgrey", zorder=1, linewidths=0,
        label="EM point (no matched probe observation)",
    )

    vmax = float(np.nanmax(np.abs(improvement_map))) if matched_mask.any() else 1e-9
    vmax = max(vmax, 1e-9)
    sc = ax.scatter(
        x_em[matched_mask], y_em[matched_mask],
        c=improvement_map[matched_mask], cmap=cmap, vmin=-vmax, vmax=vmax,
        s=55, edgecolor="k", linewidth=0.4, zorder=2,
        label="EM point (matched probe observation)",
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        r"$|err_{hard}| - |err_{weighted}|$  [m$^3$/m$^3$]"
        "\n(+) weighted closer to TDR   (-) hard-routed closer to TDR",
        fontsize=8,
    )

    if probe_x is not None and probe_y is not None:
        ax.scatter(
            probe_x, probe_y, marker="v", s=55, color="black",
            edgecolor="white", linewidth=0.6, zorder=3, label="TDR probe",
        )

    ax.set_xlabel("x [m]" if y_em.any() else "Distance along transect [m]", fontsize=10)
    ax.set_ylabel("y [m]", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)

    n_matched = int(matched_mask.sum())
    subtitle = f"{n_matched}/{len(x_em)} EM point(s) with a matched probe observation"
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    fig.tight_layout()
    out_path = os.path.join(diag_dir, filename)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[EM FOOTPRINT] Saved spatial discrepancy map -> {out_path}")
    return out_path


def plot_footprint_spatial_discrepancy_xz(
    discrepancies: Sequence[dict],
    survey_df,
    depths0: Sequence[float],
    output_dir: str,
    probe_x: np.ndarray | None = None,
    probe_depths: np.ndarray | None = None,
    title: str = "EM footprint discrepancy — XZ (depth) map",
    filename: str = "em_footprint_spatial_discrepancy_xz.png",
    cmap: str = "RdBu_r",
) -> str:
    """
    XZ (x, depth) counterpart to ``plot_footprint_spatial_discrepancy``'s
    XY (x, y) map. Same improvement metric
    (``mean(|hard_routed_err|) - mean(|weighted_err|)``, diverging
    colour scale centred at 0), but placed on the depth axis using each
    record's own routed horizon instead of the survey's y/utm_y
    coordinate -- directly comparable to this pipeline's other XZ-style
    figures (``water_content_xz_*.png``, ``plot_ws_params_xz``).

    Unlike the XY version, which plots one point per EM survey point
    (including unmatched ones, drawn faint/grey for context), this XZ
    version only ever has as many points as there are matched
    (position, horizon) pairs -- there is no equivalent of "every EM grid
    point" on the depth axis, since depth positions only exist where a
    probe/observation actually routed to a horizon.

    Parameters
    ----------
    discrepancies : sequence of dict
        Step 5c's own per-probe records (see module docstring for the
        base keys). Each entry must ALSO carry a ``'horizon'`` key (the
        probe's own routed 0-based layer index, the same value used to
        compute ``hard_routed`` upstream) -- added in ``run_pipeline.py``
        alongside ``'pos'``. Records missing ``'horizon'`` cause this
        function to skip plotting entirely (see Returns) rather than
        guess a depth.
    survey_df : pd.DataFrame
        Same frame used elsewhere in the pipeline for EM point
        coordinates -- must contain ``'x'`` (only the x-coordinate is
        used here; depth comes from ``depths0``, not ``survey_df``).
    depths0 : sequence of float
        Layer BOTTOM depths [m], same convention as this pipeline's
        DEPTHS0/EFFECTIVE_DEPTHS0 (e.g. the caller's already-resolved
        fitted interfaces, not necessarily the static starting depths).
        An implicit 0.0 top boundary is prepended; each horizon's plotted
        depth is that layer's CENTRE (midpoint of its bottom and the
        previous layer's bottom), same convention as ``plot_ws_params_xz``.
    output_dir : str
        Figures are written to ``<output_dir>/diagnostics/``.
    probe_x, probe_depths : np.ndarray, optional
        TDR probe installation locations, for overlay only (e.g. from
        ``CALIB_POSITION_RESULTS``' ``probe_x`` / ``probe_depth``). Omit
        to skip the overlay. These are independent of the discrepancy
        records' own (x, depth) placement -- purely visual context for
        where probes physically sit relative to the improvement map.
    title : str, optional
        Figure title.
    filename : str, optional
        Output PNG filename.
    cmap : str, optional
        Diverging colormap for the improvement scale.

    Returns
    -------
    str
        Path to the saved figure. Empty string if there was nothing to
        plot (no discrepancy records, or none carry a ``'horizon'`` key).
    """
    diag_dir = _ensure_dir(os.path.join(output_dir, "diagnostics"))

    if not discrepancies:
        print("[EM FOOTPRINT][WARN] No discrepancy records supplied — skipping XZ spatial discrepancy plot.")
        return ""

    if any("horizon" not in d for d in discrepancies):
        print(
            "[EM FOOTPRINT][WARN] discrepancy record(s) missing 'horizon' -- "
            "cannot place points on the depth axis. Skipping XZ spatial "
            "discrepancy plot. (Pass each probe's routed layer index as "
            "'horizon' alongside 'pos' when building the discrepancy list.)"
        )
        return ""

    depths_arr = np.asarray(depths0, dtype=float)
    if depths_arr.size == 0:
        print("[EM FOOTPRINT][WARN] Empty depths0 -- skipping XZ spatial discrepancy plot.")
        return ""

    # Layer centres from layer BOTTOM boundaries, implicit 0.0 top --
    # same convention as plot_ws_params_xz (boundaries in, centres out).
    edges = np.concatenate(([0.0], depths_arr))
    layer_centers = (edges[:-1] + edges[1:]) / 2.0
    num_layers = len(layer_centers)

    x_em = survey_df["x"].to_numpy(dtype=float)

    # Average duplicate (position, horizon) pairs (e.g. multiple time
    # steps routed through the same EM point/horizon), same convention
    # as the XY version's per-pos averaging.
    per_cell: dict[tuple[int, int], list[float]] = {}
    for d in discrepancies:
        pos = int(d["pos"])
        layer = int(d["horizon"])
        improvement = abs(d["hard_routed_err"]) - abs(d["weighted_err"])
        per_cell.setdefault((pos, layer), []).append(improvement)

    xs: list[float] = []
    zs: list[float] = []
    improvements: list[float] = []
    for (pos, layer), values in per_cell.items():
        if not (0 <= pos < len(x_em)):
            continue
        if not (0 <= layer < num_layers):
            continue
        xs.append(x_em[pos])
        zs.append(layer_centers[layer])
        improvements.append(float(np.mean(values)))

    if not xs:
        print(
            "[EM FOOTPRINT][WARN] No discrepancy record resolved to a valid "
            "(position, horizon) cell -- skipping XZ spatial discrepancy plot."
        )
        return ""

    xs_arr = np.array(xs)
    zs_arr = np.array(zs)
    improvements_arr = np.array(improvements)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    # Horizon boundary lines for depth context, same convention as this
    # pipeline's other XZ figures.
    for boundary in depths_arr:
        ax.axhline(boundary, color="grey", linestyle=":", linewidth=0.7, zorder=0)

    vmax = float(np.max(np.abs(improvements_arr)))
    vmax = max(vmax, 1e-9)
    sc = ax.scatter(
        xs_arr, zs_arr, c=improvements_arr, cmap=cmap, vmin=-vmax, vmax=vmax,
        s=90, edgecolor="k", linewidth=0.5, zorder=2,
        label="Matched (position, horizon) cell",
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        r"$|err_{hard}| - |err_{weighted}|$  [m$^3$/m$^3$]"
        "\n(+) weighted closer to TDR   (-) hard-routed closer to TDR",
        fontsize=8,
    )

    if probe_x is not None and probe_depths is not None:
        ax.scatter(
            probe_x, probe_depths, marker="v", s=40, color="black",
            edgecolor="white", linewidth=0.5, zorder=3, alpha=0.5,
            label="TDR probe installation depth",
        )

    ax.set_xlabel("x [m]", fontsize=10)
    ax.set_ylabel("Depth [m]", fontsize=10)
    ax.invert_yaxis()
    ax.grid(True, linestyle=":", alpha=0.4)

    n_cells = len(xs_arr)
    subtitle = f"{n_cells} matched (position, horizon) cell(s)"
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    fig.tight_layout()
    out_path = os.path.join(diag_dir, filename)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[EM FOOTPRINT] Saved XZ spatial discrepancy map -> {out_path}")
    return out_path
