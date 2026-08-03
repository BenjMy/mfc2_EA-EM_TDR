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

from em_methodology import gaspari_cohn


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
