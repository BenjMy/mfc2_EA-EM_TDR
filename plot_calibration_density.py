#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_calibration_density.py
============================
Spatial diagnostics for the kernel-localized Waxman-Smits calibration
diagnostic sibling (``em_methodology.calibrate_petrophysical_relationship_
localized``) and its un-localized window-matching counterpart. Two families
of plot live here:

  (1) "How much TDR calibration support does each EM measurement point
      have" — how densely each point on the EM survey is backed by real
      TDR/SWC data, either via the hard ``window_m`` match
      (``plot_calibration_density_hard``) or the Gaspari-Cohn kernel weight
      (``plot_calibration_density_kernel``).

  (2) "What did the kernel-localized fit actually produce, spatially" — the
      four fitted Waxman-Smits parameters (``a``, ``m``, ``sigma_w``,
      ``sigma_s``) plus per-target RMSE, each mapped over the EM survey
      (``plot_ws_params_localized_spatial``).

Metric (1) has two independent public functions rather than one with a
``mode=`` switch. They are
DELIBERATELY not merged behind a single ``mode=`` switch:

  * ``plot_calibration_density_hard`` reuses the exact window-matching
    logic ``em_methodology.extract_sample_positions_from_survey`` already
    computed (each ``SamplePositionResult.indices`` is "the EM points
    within ``window_m`` of this probe") — it is always available, since
    every run has a ``window_m``, and requires no new distance metric.

  * ``plot_calibration_density_kernel`` reuses ``em_methodology
    .gaspari_cohn`` — the SAME taper ``calibrate_petrophysical_relationship
    _localized`` weights TDR observations with — and is only meaningful
    when a localization ``bandwidth`` exists. ``--ws-localized-bandwidth``
    defaults to ``None`` (disabled), so this is opt-in: callers should only
    invoke it when a bandwidth was actually resolved (fixed or from the
    sweep's recommendation), exactly mirroring the existing
    ``ws_localized_bandwidth`` gating in run_pipeline.py. There is no
    "auto-fallback to window_m" here — an absent bandwidth means the
    localized diagnostic wasn't run, so plotting it would show a metric no
    calibration on this run actually used.

Both target the SAME location set — every point in ``survey_df`` (i.e. the
EM measurement array), matching where ``calibrate_petrophysical_
relationship[_localized]`` themselves operate — so the two maps are
directly comparable panel-for-panel.

Public API
----------
    plot_calibration_density_hard(survey_df, position_results, output_dir, ...)
    plot_calibration_density_kernel(survey_df, position_results, bandwidth, output_dir, ...)
    plot_ws_params_localized_spatial(fit, probe_x, probe_y, output_dir, ...)

Typical call (from run_pipeline.py, right after ``ppm.plot_probe_matching``,
using the same ``position_results`` already built there)::

    import plot_calibration_density as pcd

    pcd.plot_calibration_density_hard(
        survey_df=survey_df,
        position_results=position_results,
        output_dir=OUTPUT_DIR,
    )

    if resolved_bandwidth is not None:   # only when localized WS was requested
        pcd.plot_calibration_density_kernel(
            survey_df=survey_df,
            position_results=position_results,
            bandwidth=resolved_bandwidth,
            output_dir=OUTPUT_DIR,
        )

        # fits_step2_localized: em_methodology.LocalizedPetroFit
        for hz_fit in fits_step2_localized.horizon_fits:
            pcd.plot_ws_params_localized_spatial(
                fit=hz_fit,
                probe_x=np.array([r.probe_x for r in position_results]),
                probe_y=np.array([r.probe_y for r in position_results]),
                output_dir=OUTPUT_DIR,
            )
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

from em_methodology import gaspari_cohn
# Reused rather than reimplemented so the WS-params XZ plot below uses the
# EXACT same depth/x bound conventions as water_content_xz_*.png -- private
# (underscore) helpers, but pure functions with no side effects, and this
# guarantees the two figure families stay pixel-grid-compatible if
# plot_water_content.py's bound logic ever changes.
from plot_water_content import _z_bounds, _x_edges, _add_hatching, _interpolate_to_grid


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _survey_xy(survey_df) -> tuple[np.ndarray, np.ndarray]:
    x_em = survey_df["x"].to_numpy(dtype=float)
    y_em = (survey_df["y"].to_numpy(dtype=float)
            if "y" in survey_df.columns else np.zeros_like(x_em))
    return x_em, y_em


def _probe_xy(position_results: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    probe_x = np.array([float(r.probe_x) for r in position_results], dtype=float)
    probe_y = np.array([float(r.probe_y) for r in position_results], dtype=float)
    return probe_x, probe_y


def _probe_ids(position_results: Sequence[Any]) -> list[str] | None:
    """
    Extract each result's ``.probe_id`` (set when ``extract_sample_positions_
    from_survey`` was called with ``probe_ids=...``). Returns None -- not a
    list of Nones -- when no result carries an id, so callers can cheaply
    skip annotation for older/unlabeled position_results.
    """
    ids = [getattr(r, "probe_id", None) for r in position_results]
    return ids if any(i is not None for i in ids) else None


def _render_density_map(
    x_em: np.ndarray,
    y_em: np.ndarray,
    density: np.ndarray,
    probe_x: np.ndarray,
    probe_y: np.ndarray,
    title: str,
    subtitle: str,
    cbar_label: str,
    cmap: str,
    out_path: str,
    probe_ids: Sequence[str] | None = None,
) -> str:
    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    covered_mask = density > 0
    ax.scatter(
        x_em[~covered_mask], y_em[~covered_mask],
        s=8, color="lightgrey", zorder=1, linewidths=0,
        label="EM point (no calibration support)",
    )

    vmax = max(float(density.max()), 1e-9)
    sc = ax.scatter(
        x_em[covered_mask], y_em[covered_mask],
        c=density[covered_mask], cmap=cmap, vmin=0, vmax=vmax,
        s=45, edgecolor="k", linewidth=0.3, zorder=2,
        label="EM point (calibration support)",
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)

    ax.scatter(
        probe_x, probe_y, marker="v", s=60, color="black",
        edgecolor="white", linewidth=0.6, zorder=3, label="TDR probe",
    )
    if probe_ids is not None:
        for px, py, pid in zip(probe_x, probe_y, probe_ids):
            if pid is None:
                continue
            ax.annotate(
                str(pid), (px, py), xytext=(4, 4), textcoords="offset points",
                fontsize=6.5, color="black", zorder=4,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            )

    ax.set_xlabel("x [m]" if y_em.any() else "Distance along transect [m]", fontsize=10)
    ax.set_ylabel("y [m]", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_calibration_density_hard(
    survey_df,
    position_results: Sequence[Any],
    output_dir: str,
    title: str = "Calibration density — hard window",
    filename: str = "calibration_density_hard.png",
    cmap: str = "viridis",
) -> str:
    """
    Map, for every EM measurement point, how many TDR probes' hard
    ``window_m`` windows cover it — i.e. how many of the EM points already
    fed into ``calibrate_petrophysical_relationship`` as "this probe's
    data" include this location.

    Built purely by inverting ``SamplePositionResult.indices`` (already
    computed by ``extract_sample_positions_from_survey``); no new distance
    computation. Symmetric with the window match itself: EM point ``p`` is
    within ``window_m`` of probe ``j`` iff probe ``j`` is within
    ``window_m`` of point ``p``, so counting how many probes' ``indices``
    contain ``p`` is exactly "how many probes calibrate against this
    point".

    Parameters
    ----------
    survey_df : pd.DataFrame
        Same frame passed to ``extract_sample_positions_from_survey`` —
        must contain ``'x'`` and, optionally, ``'y'``.
    position_results : sequence of em_methodology.SamplePositionResult
        Output of ``EMmethod.extract_sample_positions_from_survey``.
    output_dir : str
        Figures are written to ``<output_dir>/diagnostics/``.
    title : str, optional
        Figure title.
    filename : str, optional
        Output PNG filename.
    cmap : str, optional
        Colormap for the density scale.

    Returns
    -------
    str
        Path to the saved figure. Empty string if there was nothing to plot.
    """
    diag_dir = _ensure_dir(os.path.join(output_dir, "diagnostics"))

    if not position_results:
        print("[CALIB DENSITY][WARN] No position_results supplied — skipping hard-window plot.")
        return ""

    x_em, y_em = _survey_xy(survey_df)
    probe_x, probe_y = _probe_xy(position_results)

    density = np.zeros(len(x_em), dtype=int)
    for r in position_results:
        for idx in r.indices:
            if 0 <= idx < len(density):
                density[idx] += 1

    window_vals = {float(r.window_m) for r in position_results}
    window_label = (f"{next(iter(window_vals)):.2f} m" if len(window_vals) == 1
                     else f"{min(window_vals):.2f}–{max(window_vals):.2f} m (mixed)")
    n_covered = int((density > 0).sum())
    subtitle = (
        f"{len(position_results)} probe(s), window_m = {window_label}  ->  "
        f"{n_covered}/{len(x_em)} EM point(s) calibrated (max {int(density.max())} probes/point)"
    )

    out_path = os.path.join(diag_dir, filename)
    _render_density_map(
        x_em, y_em, density.astype(float), probe_x, probe_y,
        title=title, subtitle=subtitle,
        cbar_label="TDR probes calibrating this EM point (hard count)",
        cmap=cmap, out_path=out_path,
        probe_ids=_probe_ids(position_results),
    )
    print(f"[CALIB DENSITY] Saved hard-window density map -> {out_path}")
    return out_path


def plot_calibration_density_kernel(
    survey_df,
    position_results: Sequence[Any],
    bandwidth: float,
    output_dir: str,
    title: str = "Calibration density — kernel-localized",
    filename: str = "calibration_density_kernel.png",
    cmap: str = "viridis",
) -> str:
    """
    Map, for every EM measurement point, the Gaspari-Cohn-weighted "effective
    number" of TDR probes contributing to a kernel-localized fit at that
    point — the same weighting ``calibrate_petrophysical_relationship_
    localized`` uses, so this is the continuous (tapered-falloff) analogue
    of ``plot_calibration_density_hard``'s step-function count.

    Requires an actual ``bandwidth``. This is intentionally NOT optional
    with a fallback to ``window_m``: ``--ws-localized-bandwidth`` (and its
    scenario-config equivalent) defaults to disabled, so an absent
    bandwidth means the localized diagnostic wasn't run for this pipeline
    call, and plotting a metric under a bandwidth nothing was actually
    calibrated with would be misleading. Callers should only invoke this
    when a fixed or swept bandwidth was actually resolved.

    Parameters
    ----------
    survey_df : pd.DataFrame
        Same frame passed to ``extract_sample_positions_from_survey`` —
        must contain ``'x'`` and, optionally, ``'y'``.
    position_results : sequence of em_methodology.SamplePositionResult
        Output of ``EMmethod.extract_sample_positions_from_survey``. Used
        here only for probe (x, y) locations — the kernel weighting itself
        is recomputed at every EM point rather than reusing ``indices``
        (which encode the hard window, not the taper).
    bandwidth : float
        The Gaspari-Cohn localization radius ``c`` — must match (or be the
        one actually used by) ``calibrate_petrophysical_relationship_
        localized`` for this map to describe the run's real calibration.
    output_dir : str
        Figures are written to ``<output_dir>/diagnostics/``.
    title : str, optional
        Figure title.
    filename : str, optional
        Output PNG filename.
    cmap : str, optional
        Colormap for the density scale.

    Returns
    -------
    str
        Path to the saved figure. Empty string if there was nothing to plot.
    """
    diag_dir = _ensure_dir(os.path.join(output_dir, "diagnostics"))

    if not position_results:
        print("[CALIB DENSITY][WARN] No position_results supplied — skipping kernel plot.")
        return ""
    if bandwidth is None:
        print("[CALIB DENSITY][WARN] No bandwidth given — kernel-localized diagnostic "
              "was not run for this pipeline call. Skipping kernel density plot rather "
              "than guessing a bandwidth.")
        return ""

    x_em, y_em = _survey_xy(survey_df)
    probe_x, probe_y = _probe_xy(position_results)

    # Same 1-D vs 2-D decision rule as em_methodology._probe_distances /
    # gaspari_cohn's own callers: use Euclidean XY only if there's real
    # cross-transect spread, otherwise |dx| alone.
    has_y_spread = np.any(probe_y != 0.0) or np.any(y_em != 0.0)

    density = np.zeros(len(x_em), dtype=float)
    for px, py in zip(probe_x, probe_y):
        if has_y_spread:
            dist = np.sqrt((x_em - px) ** 2 + (y_em - py) ** 2)
        else:
            dist = np.abs(x_em - px)
        density += gaspari_cohn(dist, bandwidth)

    subtitle = (
        f"{len(position_results)} probe(s), bandwidth c = {bandwidth:.2f} m "
        f"(support 2c = {2 * bandwidth:.2f} m)  ->  "
        f"max effective weight {density.max():.2f} probes/point"
    )

    out_path = os.path.join(diag_dir, filename)
    _render_density_map(
        x_em, y_em, density, probe_x, probe_y,
        title=title, subtitle=subtitle,
        cbar_label="Effective TDR probes (Gaspari-Cohn-weighted)",
        cmap=cmap, out_path=out_path,
        probe_ids=_probe_ids(position_results),
    )
    print(f"[CALIB DENSITY] Saved kernel-localized density map -> {out_path}")
    return out_path


def plot_ws_params_localized_spatial(
    fit: Any,
    probe_x: np.ndarray,
    probe_y: np.ndarray,
    output_dir: str,
    title: str | None = None,
    filename: str | None = None,
    cmap: str = "viridis",
) -> str:
    """
    Spatial map of a kernel-localized Waxman-Smits fit's four parameters
    (``a``, ``m``, ``sigma_w``, ``sigma_s``) plus per-target RMSE, one panel
    each, over the EM survey's target locations.

    Takes a SINGLE ``em_methodology.LocalizedHorizonWaxmanSmitsParams``
    (i.e. one soil horizon) — for a multi-horizon fit, call once per
    ``fit in LocalizedPetroFit.horizon_fits`` at the call site rather than
    looping internally, so each horizon gets its own independent colour
    scales (the four params and RMSE have physically different ranges
    across horizons, so a shared scale across horizons would be
    misleading).

    Targets where ``fit.is_calibrated`` is ``False`` (no TDR observation of
    this horizon fell within the kernel's support, ``2*bandwidth``, so the
    entry is just the unfit prior) are drawn with a distinct hollow marker
    in every panel, so it's visually obvious at a glance which stretches of
    the map are genuinely locally fit vs. falling back to the prior.

    Parameters
    ----------
    fit : em_methodology.LocalizedHorizonWaxmanSmitsParams
        One horizon's localized fit — ``target_x``/``target_y``, ``a``,
        ``m``, ``sigma_w``, ``sigma_s``, ``rmse``, ``is_calibrated``,
        ``bandwidth``, ``horizon`` are all read from this.
    probe_x, probe_y : np.ndarray
        TDR probe locations, for overlay only (this fit's own dataclass
        only carries EM target coordinates, not probe coordinates).
    output_dir : str
        Figures are written to ``<output_dir>/diagnostics/``.
    title : str, optional
        Figure suptitle. Defaults to a string built from ``fit.horizon``
        and ``fit.bandwidth``.
    filename : str, optional
        Output PNG filename. Defaults to ``f"ws_params_spatial_hz{fit.horizon}.png"``.
    cmap : str, optional
        Colormap for each panel (each panel gets its own colorbar — the
        five fields are not on comparable scales).

    Returns
    -------
    str
        Path to the saved figure. Empty string if there was nothing to plot.
    """
    diag_dir = _ensure_dir(os.path.join(output_dir, "diagnostics"))

    target_x = np.asarray(fit.target_x, dtype=float)
    if target_x.size == 0:
        print("[WS SPATIAL][WARN] Empty fit (no targets) — skipping WS-parameter spatial plot.")
        return ""

    target_y = (np.asarray(fit.target_y, dtype=float) if fit.target_y is not None
                else np.zeros_like(target_x))
    is_calibrated = np.asarray(fit.is_calibrated, dtype=bool)
    y_has_spread = bool(np.any(target_y != 0.0) or np.any(probe_y != 0.0))

    fields = [
        ("a", fit.a, "a  [-]"),
        ("m", fit.m, "m  [-]"),
        ("sigma_w", fit.sigma_w, "\u03c3\u1d64  [mS/m]"),
        ("sigma_s", fit.sigma_s, "\u03c3\u209b  [mS/m]"),
        ("rmse", fit.rmse, "RMSE  [m\u00b3/m\u00b3]"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.ravel()

    for ax, (key, values, cbar_label) in zip(axes_flat, fields):
        values = np.asarray(values, dtype=float)

        ax.scatter(
            target_x[~is_calibrated], target_y[~is_calibrated],
            marker="o", s=35, facecolors="none", edgecolors="grey",
            linewidths=0.8, alpha=0.6, zorder=1,
            label="Target — fell back to prior (no local TDR support)",
        )
        finite = np.isfinite(values) & is_calibrated
        if np.any(finite):
            sc = ax.scatter(
                target_x[finite], target_y[finite],
                c=values[finite], cmap=cmap, s=40, edgecolor="k",
                linewidth=0.3, zorder=2, label="Target — kernel-localized fit",
            )
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label, fontsize=8)
        else:
            ax.text(0.5, 0.5, "No calibrated targets", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="grey")

        ax.scatter(
            probe_x, probe_y, marker="v", s=45, color="black",
            edgecolor="white", linewidth=0.5, zorder=3, label="TDR probe",
        )

        ax.set_title(cbar_label, fontsize=10)
        ax.set_xlabel("x [m]" if y_has_spread else "Distance along transect [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle=":", alpha=0.4)

    # Unused 6th slot -> shared legend instead of repeating it 5x.
    legend_ax = axes_flat[-1]
    legend_ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        legend_ax.legend(handles, labels, loc="center", fontsize=9, framealpha=0.9)

    n_calibrated = int(is_calibrated.sum())
    resolved_title = title or (
        f"Kernel-localized Waxman-Smits parameters — horizon {fit.horizon}"
    )
    subtitle = (
        f"bandwidth c = {fit.bandwidth:.2f} m (support 2c = {2 * fit.bandwidth:.2f} m)  ->  "
        f"{n_calibrated}/{len(target_x)} target(s) locally calibrated"
    )
    fig.suptitle(f"{resolved_title}\n{subtitle}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    resolved_filename = filename or f"ws_params_spatial_hz{fit.horizon}.png"
    out_path = os.path.join(diag_dir, resolved_filename)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[WS SPATIAL] Saved horizon {fit.horizon} WS-parameter spatial map -> {out_path}")
    return out_path


def plot_ws_params_xz(
    horizon_fits: Sequence[Any],
    depths0: Sequence[float],
    output_dir: str,
    probe_x: np.ndarray | None = None,
    probe_depths: np.ndarray | None = None,
    title: str = "Kernel-localized Waxman-Smits parameters — XZ cross-section",
    filename: str = "ws_params_xz.png",
    cmap: str = "viridis",
    rmse_cmap: str = "inferno",
) -> str:
    """
    Single-figure XZ cross-section (distance along transect vs. depth) of
    the four kernel-localized Waxman-Smits parameters (a, m, sigma_w,
    sigma_s) plus per-target RMSE -- ALL horizons in one figure, one panel
    per field, horizon = depth row (same layout family as
    ``plot_water_content.py``'s ``water_content_xz_*.png``, reusing its
    exact ``_z_bounds``/``_x_edges``/``_add_hatching`` helpers).

    Replaces calling ``plot_ws_params_localized_spatial`` once per horizon
    (4 separate ``ws_params_spatial_hz{N}.png`` files, one flat XY scatter
    each) with ONE file where horizon becomes the depth axis -- directly
    comparable to the water-content XZ panel instead of a scattered set of
    per-horizon maps.

    Targets where ``is_calibrated`` is ``False`` (no TDR observation of
    that horizon fell within the kernel's support, ``2*bandwidth`` -- the
    entry is just the unfit prior) are hatched, same convention
    ``_plot_xz_panel`` uses for "no coverage" cells.

    Parameters
    ----------
    horizon_fits : sequence of em_methodology.LocalizedHorizonWaxmanSmitsParams
        ``LocalizedPetroFit.horizon_fits`` for ONE ensemble member (e.g.
        ``fits_step2_localized[0].horizon_fits`` -- the rank-1 member).
        All horizons must share the same ``target_x``/``target_indices``
        (true by construction -- ``calibrate_petrophysical_relationship_
        localized`` computes ``target_x_all`` once, before the per-horizon
        loop, so every horizon's fit copies the same array).
    depths0 : sequence of float
        Layer BOTTOM depths [m] -- same convention as this pipeline's
        DEPTHS0/EFFECTIVE_DEPTHS0 -- length must be
        ``len(horizon_fits) - 1`` (finite layers; the last ``horizon_fits``
        entry is the implicit half-space, exactly like
        ``em_sensitivity.layer_weights``'s output).
    output_dir : str
        Figures are written to ``<output_dir>/diagnostics/``.
    probe_x, probe_depths : np.ndarray, optional
        TDR probe (x, depth) locations, for overlay only. Omit either to
        skip the overlay.
    title : str, optional
        Figure suptitle.
    filename : str, optional
        Output PNG filename.
    cmap : str, optional
        Colormap for the four WS-parameter panels (a, m, sigma_w, sigma_s;
        each panel gets its own colorbar -- the fields are not on
        comparable scales).
    rmse_cmap : str, optional
        Separate colormap for the RMSE panel. RMSE is an error/quality
        metric rather than a physical WS parameter, so it gets its own
        (sequential, "hot = worse") scale distinct from the ``cmap`` used
        for a/m/sigma_w/sigma_s -- keeps a glance at the figure from
        conflating "high sigma_s" with "high RMSE" just because both are
        rendered in the same colours.

    Returns
    -------
    str
        Path to the saved figure. Empty string if there was nothing to plot.
    """
    diag_dir = _ensure_dir(os.path.join(output_dir, "diagnostics"))

    if not horizon_fits:
        print("[WS XZ][WARN] No horizon_fits supplied — skipping WS-parameter XZ plot.")
        return ""

    n_layers = len(horizon_fits)
    if len(depths0) != n_layers - 1:
        print(
            f"[WS XZ][WARN] len(depths0)={len(depths0)} does not match "
            f"len(horizon_fits)-1={n_layers - 1} (finite layers only, half-space "
            "implicit) -- depth axis may not line up with the actual horizons. "
            "Proceeding anyway."
        )

    x_ref = np.asarray(horizon_fits[0].target_x, dtype=float)
    if x_ref.size == 0:
        print("[WS XZ][WARN] Empty target_x on horizon 0 — skipping WS-parameter XZ plot.")
        return ""
    order = np.argsort(x_ref)
    x_sorted = x_ref[order]

    fields = [
        ("a", "a  [-]"),
        ("m", "m  [-]"),
        ("sigma_w", "\u03c3\u1d64  [mS/m]"),
        ("sigma_s", "\u03c3\u209b  [mS/m]"),
        ("rmse", "RMSE  [m\u00b3/m\u00b3]"),
    ]
    n_x = len(x_sorted)

    grids = {key: np.full((n_x, n_layers), np.nan) for key, _ in fields}
    calibrated_grid = np.zeros((n_x, n_layers), dtype=bool)

    for h, fit in enumerate(horizon_fits):
        fit_x = np.asarray(fit.target_x, dtype=float)
        if fit_x.shape != x_ref.shape or not np.allclose(fit_x, x_ref):
            print(
                f"[WS XZ][WARN] horizon_fits[{h}].target_x differs from "
                "horizon_fits[0].target_x -- this shouldn't happen given how "
                "calibrate_petrophysical_relationship_localized builds target_x_all "
                "once, shared across horizons. Skipping this horizon's column."
            )
            continue
        for key, _ in fields:
            grids[key][:, h] = np.asarray(getattr(fit, key), dtype=float)[order]
        calibrated_grid[:, h] = np.asarray(fit.is_calibrated, dtype=bool)[order]

    zb = _z_bounds(list(depths0)) if len(depths0) == n_layers - 1 else np.arange(n_layers + 1, dtype=float)
    xe = _x_edges(x_sorted)
    X_mesh, Z_mesh = np.meshgrid(xe, zb)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.ravel()
    uncalibrated_mask = ~calibrated_grid

    for ax, (key, cbar_label) in zip(axes_flat, fields):
        grid = grids[key]
        finite_vals = grid[calibrated_grid & np.isfinite(grid)]
        if finite_vals.size == 0:
            ax.text(0.5, 0.5, "No calibrated targets", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="grey")
            ax.set_title(cbar_label, fontsize=10)
            continue

        panel_cmap = rmse_cmap if key == "rmse" else cmap
        norm = plt.Normalize(vmin=float(finite_vals.min()), vmax=float(finite_vals.max()))
        sm = ax.pcolormesh(X_mesh, Z_mesh, grid.T, cmap=panel_cmap, norm=norm, shading="flat")
        _add_hatching(ax, uncalibrated_mask, xe, zb,
                      hatch="///", facecolor="lightgrey", edgecolor="grey")

        if probe_x is not None and probe_depths is not None:
            ax.scatter(probe_x, probe_depths, marker="v", s=35, color="black",
                       edgecolor="white", linewidth=0.5, zorder=5, label="TDR probe")

        ax.invert_yaxis()
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(zb[-1], zb[0])
        ax.set_xlabel("Distance along transect (m)", fontsize=8)
        ax.set_ylabel("Depth (m)", fontsize=8)
        ax.set_title(cbar_label, fontsize=10)
        ax.tick_params(labelsize=7)

        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # Unused 6th slot -> legend (hatch + probe marker), instead of repeating
    # it in every panel.
    legend_ax = axes_flat[-1]
    legend_ax.axis("off")
    _legend_handles = [
        mpatches.Patch(facecolor="lightgrey", edgecolor="grey", hatch="///",
                       label="Fell back to prior (no local TDR support)"),
    ]
    if probe_x is not None and probe_depths is not None:
        _legend_handles.append(
            plt.Line2D([0], [0], marker="v", color="black", linestyle="None",
                       markersize=7, markeredgecolor="white", label="TDR probe")
        )
    legend_ax.legend(handles=_legend_handles, loc="center", fontsize=9, framealpha=0.9)

    n_calibrated_total = int(calibrated_grid.sum())
    n_total = calibrated_grid.size
    bandwidth = getattr(horizon_fits[0], "bandwidth", None)
    subtitle = (
        f"bandwidth c = {bandwidth:.2f} m (support 2c = {2 * bandwidth:.2f} m)  ->  "
        if bandwidth is not None else ""
    ) + f"{n_calibrated_total}/{n_total} (x, horizon) cell(s) locally calibrated"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(diag_dir, filename)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[WS XZ] Saved WS-parameter XZ cross-section -> {out_path}")
    return out_path


def plot_ws_params_xy(
    horizon_fits: Sequence[Any],
    output_dir: str,
    probe_x: np.ndarray | None = None,
    probe_y: np.ndarray | None = None,
    layer: int | None = None,
    title: str = "Kernel-localized Waxman-Smits parameters — XY plan view",
    filename: str = "ws_params_xy.png",
    cmap: str = "viridis",
    rmse_cmap: str = "inferno",
    render_mode: str = "scatter",
    grid_resolution: int = 200,
    interp_method: str = "linear",
) -> str:
    """
    Plan-view (easting vs. northing) counterpart of ``plot_ws_params_xz``,
    for the same kernel-localized Waxman-Smits parameters (a, m, sigma_w,
    sigma_s) plus per-target RMSE -- ALL horizons collapsed onto one map
    per field, one panel per field, same 2x3 layout family (reuses this
    module's own ``plot_ws_params_localized_spatial`` marker conventions
    and ``plot_water_content.py``'s ``_interpolate_to_grid`` for the
    optional smooth-raster render mode). Directly comparable to
    ``water_content_xy_*.png`` / ``plot_footprint_spatial_discrepancy``'s
    XY map, the same way ``plot_ws_params_xz`` is comparable to
    ``water_content_xz_*.png``.

    Where ``plot_ws_params_xz`` puts horizon on the depth axis,
    this function collapses horizons into ONE value per target by
    averaging across whichever horizons are calibrated at that target
    (``layer=None``, default) -- or, if ``layer`` is given, uses that
    single horizon's values directly instead of averaging.

    Targets with no calibrated horizon to draw from (nothing to average,
    or the requested ``layer`` itself fell back to the prior) are drawn as
    hollow grey markers, matching ``plot_ws_params_localized_spatial``'s
    "fell back to prior" convention.

    Parameters
    ----------
    horizon_fits : sequence of em_methodology.LocalizedHorizonWaxmanSmitsParams
        Same object as passed to ``plot_ws_params_xz`` -- typically
        ``fits_step2_localized[0].horizon_fits`` (the rank-1 member). All
        horizons must share the same ``target_x``/``target_y`` (true by
        construction, see ``plot_ws_params_xz``'s docstring).
    output_dir : str
        Figures are written to ``<output_dir>/diagnostics/``.
    probe_x, probe_y : np.ndarray, optional
        TDR probe (x, y) locations, for overlay only. Omit either to skip
        the overlay.
    layer : int, optional
        If given, plot that single horizon's values instead of averaging
        across horizons (index into ``horizon_fits``, same convention as
        ``plot_ws_params_xz``'s depth axis). Default ``None`` averages
        over all calibrated horizons at each target.
    title : str, optional
        Figure suptitle.
    filename : str, optional
        Output PNG filename.
    cmap : str, optional
        Colormap for the four WS-parameter panels (a, m, sigma_w,
        sigma_s; each panel gets its own colorbar).
    rmse_cmap : str, optional
        Separate colormap for the RMSE panel -- same rationale as
        ``plot_ws_params_xz``'s ``rmse_cmap``: RMSE is an error metric,
        not a physical WS parameter, so it's kept visually distinct from
        the other four fields.
    render_mode : {"scatter", "imshow"}, optional
        "scatter" (default) draws one marker per target -- exact values,
        but gappy for sparse/irregular layouts. "imshow" interpolates
        onto a regular grid via ``_interpolate_to_grid`` (same helper
        ``plot_water_content.py``'s XY panels use) for a smooth
        continuous raster; only calibrated targets feed the
        interpolation. Requires >=3 calibrated targets to interpolate --
        falls back to "scatter" for that panel otherwise.
    grid_resolution, interp_method : passed straight through to
        ``_interpolate_to_grid`` when ``render_mode="imshow"``.

    Returns
    -------
    str
        Path to the saved figure. Empty string if there was nothing to plot.
    """
    diag_dir = _ensure_dir(os.path.join(output_dir, "diagnostics"))

    if not horizon_fits:
        print("[WS XY][WARN] No horizon_fits supplied — skipping WS-parameter XY plot.")
        return ""

    n_layers = len(horizon_fits)
    if layer is not None and not (0 <= layer < n_layers):
        print(
            f"[WS XY][WARN] layer={layer} out of range for "
            f"{n_layers} horizon_fits — skipping WS-parameter XY plot."
        )
        return ""

    x_ref = np.asarray(horizon_fits[0].target_x, dtype=float)
    if x_ref.size == 0:
        print("[WS XY][WARN] Empty target_x on horizon 0 — skipping WS-parameter XY plot.")
        return ""
    y_ref = (np.asarray(horizon_fits[0].target_y, dtype=float)
             if horizon_fits[0].target_y is not None else np.zeros_like(x_ref))

    fields = [
        ("a", "a  [-]"),
        ("m", "m  [-]"),
        ("sigma_w", "\u03c3\u1d64  [mS/m]"),
        ("sigma_s", "\u03c3\u209b  [mS/m]"),
        ("rmse", "RMSE  [m\u00b3/m\u00b3]"),
    ]
    n_x = x_ref.size

    grids = {key: np.full((n_x, n_layers), np.nan) for key, _ in fields}
    calibrated_grid = np.zeros((n_x, n_layers), dtype=bool)

    for h, fit in enumerate(horizon_fits):
        fit_x = np.asarray(fit.target_x, dtype=float)
        if fit_x.shape != x_ref.shape or not np.allclose(fit_x, x_ref):
            print(
                f"[WS XY][WARN] horizon_fits[{h}].target_x differs from "
                "horizon_fits[0].target_x -- skipping this horizon's column."
            )
            continue
        for key, _ in fields:
            grids[key][:, h] = np.asarray(getattr(fit, key), dtype=float)
        calibrated_grid[:, h] = np.asarray(fit.is_calibrated, dtype=bool)

    if layer is not None:
        agg_calibrated = calibrated_grid[:, layer]
        agg_values = {key: grids[key][:, layer] for key, _ in fields}
        agg_label = f"horizon {layer}"
    else:
        agg_calibrated = calibrated_grid.any(axis=1)
        agg_values = {}
        for key, _ in fields:
            masked = np.where(calibrated_grid, grids[key], np.nan)
            with np.errstate(all="ignore"):
                agg_values[key] = np.nanmean(masked, axis=1)
        agg_label = f"mean over {n_layers} horizons (calibrated only)"

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.ravel()

    for ax, (key, cbar_label) in zip(axes_flat, fields):
        panel_cmap = rmse_cmap if key == "rmse" else cmap
        values = agg_values[key]

        ax.scatter(
            x_ref[~agg_calibrated], y_ref[~agg_calibrated],
            marker="o", s=35, facecolors="none", edgecolors="grey",
            linewidths=0.8, alpha=0.6, zorder=1,
            label="Target — fell back to prior (no local TDR support)",
        )

        finite = np.isfinite(values) & agg_calibrated
        if not np.any(finite):
            ax.text(0.5, 0.5, "No calibrated targets", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="grey")
            ax.set_title(cbar_label, fontsize=10)
            continue

        if render_mode == "imshow" and int(np.sum(finite)) >= 3:
            grid_values, extent = _interpolate_to_grid(
                x_ref[finite], y_ref[finite], values[finite],
                grid_resolution=grid_resolution, interp_method=interp_method,
            )
            sm = ax.imshow(
                grid_values, extent=extent, origin="lower",
                cmap=panel_cmap, aspect="auto", interpolation="nearest",
            )
        else:
            sm = ax.scatter(
                x_ref[finite], y_ref[finite],
                c=values[finite], cmap=panel_cmap, s=40, edgecolor="k",
                linewidth=0.3, zorder=2, label="Target — kernel-localized fit",
            )

        if probe_x is not None and probe_y is not None:
            ax.scatter(
                probe_x, probe_y, marker="v", s=45, color="black",
                edgecolor="white", linewidth=0.5, zorder=3, label="TDR probe",
            )

        ax.set_title(cbar_label, fontsize=10)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.tick_params(labelsize=7)

        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # Unused 6th slot -> shared legend, same convention as
    # plot_ws_params_localized_spatial / plot_ws_params_xz.
    legend_ax = axes_flat[-1]
    legend_ax.axis("off")
    _legend_handles = [
        plt.Line2D([0], [0], marker="o", color="grey", linestyle="None",
                   markersize=7, markerfacecolor="none",
                   label="Fell back to prior (no local TDR support)"),
    ]
    if probe_x is not None and probe_y is not None:
        _legend_handles.append(
            plt.Line2D([0], [0], marker="v", color="black", linestyle="None",
                       markersize=7, markeredgecolor="white", label="TDR probe")
        )
    legend_ax.legend(handles=_legend_handles, loc="center", fontsize=9, framealpha=0.9)

    n_calibrated_total = int(agg_calibrated.sum())
    bandwidth = getattr(horizon_fits[0], "bandwidth", None)
    subtitle = (
        f"bandwidth c = {bandwidth:.2f} m (support 2c = {2 * bandwidth:.2f} m)  ->  "
        if bandwidth is not None else ""
    ) + f"{n_calibrated_total}/{n_x} target(s) locally calibrated, {agg_label}"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(diag_dir, filename)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[WS XY] Saved WS-parameter XY plan view -> {out_path}")
    return out_path
