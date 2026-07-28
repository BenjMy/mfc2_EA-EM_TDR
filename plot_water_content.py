#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_water_content.py
=====================
Produces soil water content (θ, % vol.) visualisations from the EA-EM
pipeline output.

Figures produced
----------------
XZ panels  — classic cross-section: distance along transect (x-axis) vs.
              depth (z-axis).  Available in single-epoch and time-lapse
              grid layouts.

XY panels  — plan-view map: easting (x-axis) vs. northing (y-axis),
              coloured by θ averaged over all layers (or a chosen layer).
              Only meaningful in 2-D mode when utm_x / utm_y coordinates
              are available.

Public API
----------
    plot_water_content_section(...)        Single-epoch XZ section.
    plot_water_content_timelapse(...)      Grid of XZ panels, one per epoch.
    plot_water_content_xy(...)             Plan-view XY map (2-D mode).
    plot_water_content_xy_timelapse(...)   Grid of XY panels, one per epoch.
    plot_water_content_paired(...)         Paired θ / CV XZ panels.
    plot_from_pipeline(...)                Convenience wrapper called from
                                           run_pipeline.main().
    plot_truth_from_pipeline(...)          Ground-truth counterpart of
                                           plot_from_pipeline (synthetic mode
                                           only) — same XZ/XY layout, filenames
                                           tagged '_truth'.
    plot_one_to_one_stages(...)            3-panel 1:1 scatter of predicted vs
                                           TDR-observed θ for stages A, B, C.
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import BoundaryNorm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _layer_centers(depths0: list[float]) -> np.ndarray:
    bounds = [0.0] + list(depths0) + [depths0[-1] * 1.5]
    return np.array([(bounds[i] + bounds[i + 1]) / 2 for i in range(len(bounds) - 1)])


def _z_bounds(depths0: list[float]) -> np.ndarray:
    return np.array(
        [0.0] + list(depths0)
        + [depths0[-1] + (depths0[-1] - depths0[-2] if len(depths0) > 1 else depths0[-1])]
    )


def _x_edges(x_coords: np.ndarray) -> np.ndarray:
    n_x = len(x_coords)
    if n_x > 1:
        half = np.diff(x_coords) / 2
        return np.concatenate([
            [x_coords[0] - half[0]],
            x_coords[:-1] + half,
            [x_coords[-1] + half[-1]],
        ])
    return np.array([x_coords[0] - 0.5, x_coords[0] + 0.5])


def _add_hatching(
    ax: plt.Axes,
    mask: np.ndarray,
    x_edges: np.ndarray,
    z_bounds: np.ndarray,
    hatch: str = "///",
    facecolor: str = "white",
    edgecolor: str = "grey",
    linewidth: float = 0.4,
    alpha: float = 0.6,
) -> None:
    n_x, n_layers = mask.shape
    for ix in range(n_x):
        for iz in range(n_layers):
            if mask[ix, iz]:
                x0, x1 = x_edges[ix], x_edges[ix + 1]
                z0, z1 = z_bounds[iz], z_bounds[iz + 1]
                rect = mpatches.FancyBboxPatch(
                    (x0, z0), x1 - x0, z1 - z0,
                    linewidth=linewidth,
                    edgecolor=edgecolor,
                    facecolor=facecolor,
                    hatch=hatch,
                    alpha=alpha,
                    boxstyle="square,pad=0",
                )
                ax.add_patch(rect)


def _extract_theta_calc(selected_output) -> np.ndarray:
    """
    Robustly pull the calculated volumetric water content array (m^3/m^3)
    off a ``SelectionOutput``-like object, trying the conventional
    attribute names before falling back to a fuzzy ``"theta"`` search.
    Shared by ``plot_from_pipeline`` and ``plot_pct_diff_from_pipeline`` so
    both always agree on which array counts as "the calculated theta".
    """
    if hasattr(selected_output, "theta_calc"):
        return np.asarray(selected_output.theta_calc)
    elif hasattr(selected_output, "calculated_theta"):
        return np.asarray(selected_output.calculated_theta)
    elif hasattr(selected_output, "theta"):
        return np.asarray(selected_output.theta)
    else:
        theta_attrs = [a for a in dir(selected_output) if "theta" in a.lower()]
        if theta_attrs:
            return np.asarray(getattr(selected_output, theta_attrs[0]))
        raise AttributeError(
            f"No theta attribute found. Available: "
            f"{[a for a in dir(selected_output) if not a.startswith('_')]}"
        )


# ---------------------------------------------------------------------------
# Core XZ single-panel (cross-section: distance vs depth)
# ---------------------------------------------------------------------------

def _plot_xz_panel(
    ax: plt.Axes,
    theta: np.ndarray,           # (n_x, n_layers)  θ in % vol.
    cv: np.ndarray,              # (n_x, n_layers)  CV in %
    coverage: np.ndarray | None, # (n_x, n_layers) or None
    x_coords: np.ndarray,        # (n_x,) positions along transect [m]
    depths0: list[float],
    probe_x: Sequence[float] | None = None,
    probe_depths: Sequence[float] | None = None,
    theta_vmin: float = 0.0,
    theta_vmax: float = 45.0,
    cv_threshold: float = 100.0,
    coverage_threshold: float = 0.0,
    cmap: str = "Blues",
    title: str = "",
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_colorbar: bool = False,
    cbar_ax: plt.Axes | None = None,
    xlabel: str = "Distance along transect (m)",
    value_label: str = "θ (% vol.)",
) -> plt.cm.ScalarMappable:
    """Draw one XZ cross-section panel on *ax*. Returns the ScalarMappable."""
    zb = _z_bounds(depths0)
    xe = _x_edges(x_coords)

    norm     = plt.Normalize(vmin=theta_vmin, vmax=theta_vmax)
    cmap_obj = plt.get_cmap(cmap)

    X_mesh, Z_mesh = np.meshgrid(xe, zb)
    sm = ax.pcolormesh(X_mesh, Z_mesh, theta.T, cmap=cmap_obj, norm=norm, shading="flat")

    if coverage is not None:
        _add_hatching(ax, coverage <= coverage_threshold, xe, zb,
                      hatch="///", facecolor="lightgrey", edgecolor="grey")

    _add_hatching(ax, cv > cv_threshold, xe, zb,
                  hatch="xxx", facecolor="white", edgecolor="white", alpha=0.8)

    if probe_x is not None and probe_depths is not None:
        ax.scatter(probe_x, probe_depths, marker="v", s=40,
                   color="black", zorder=5, label="Probe")

    ax.invert_yaxis()
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(zb[-1], 0.0)

    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Depth (m)", fontsize=8)

    ax.set_title(title, fontsize=8, pad=3)
    ax.tick_params(labelsize=7)

    if show_colorbar and cbar_ax is not None:
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label(value_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    return sm


# ---------------------------------------------------------------------------
# Core XY single-panel (plan view: easting vs northing)
# ---------------------------------------------------------------------------

def _plot_xy_panel(
    ax: plt.Axes,
    theta_layer_mean: np.ndarray,  # (n_points,) mean θ over layers, % vol.
    utm_x: np.ndarray,             # (n_points,) UTM easting [m]
    utm_y: np.ndarray,             # (n_points,) UTM northing [m]
    theta_vmin: float = 0.0,
    theta_vmax: float = 45.0,
    cmap: str = "Blues",
    title: str = "",
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_colorbar: bool = False,
    cbar_ax: plt.Axes | None = None,
    probe_utm_x: Sequence[float] | None = None,
    probe_utm_y: Sequence[float] | None = None,
    scatter_size: float = 60.0,
) -> plt.cm.ScalarMappable:
    """Draw one XY plan-view panel on *ax*. Returns the ScalarMappable."""
    norm     = plt.Normalize(vmin=theta_vmin, vmax=theta_vmax)
    cmap_obj = plt.get_cmap(cmap)

    sc = ax.scatter(
        utm_x, utm_y,
        c=theta_layer_mean, cmap=cmap_obj, norm=norm,
        s=scatter_size, marker="s", linewidths=0,
    )

    if probe_utm_x is not None and probe_utm_y is not None:
        ax.scatter(probe_utm_x, probe_utm_y,
                   marker="v", s=50, color="black", zorder=5, label="Probe")

    if show_xlabel:
        ax.set_xlabel("UTM Easting (m)", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("UTM Northing (m)", fontsize=8)

    ax.set_title(title, fontsize=8, pad=3)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")

    if show_colorbar and cbar_ax is not None:
        cbar = plt.colorbar(sc, cax=cbar_ax)
        cbar.set_label("θ (% vol.)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    return sc


# ---------------------------------------------------------------------------
# Public: single-epoch XZ cross-section
# ---------------------------------------------------------------------------

def plot_water_content_section(
    theta_calc: np.ndarray,
    cv_map: np.ndarray,
    x_coords: np.ndarray,
    depths0: list[float],
    coverage: np.ndarray | None = None,
    probe_x: Sequence[float] | None = None,
    probe_depths: Sequence[float] | None = None,
    theta_vmin: float = 0.0,
    theta_vmax: float = 45.0,
    cv_threshold: float = 100.0,
    coverage_threshold: float = 0.0,
    cmap: str = "Blues",
    title: str = "Soil Water Content",
    output_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot a single XZ soil-water-content section (one time step)."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])

    sm = _plot_xz_panel(
        ax=ax, theta=theta_calc, cv=cv_map, coverage=coverage,
        x_coords=x_coords, depths0=depths0,
        probe_x=probe_x, probe_depths=probe_depths,
        theta_vmin=theta_vmin, theta_vmax=theta_vmax,
        cv_threshold=cv_threshold, coverage_threshold=coverage_threshold,
        cmap=cmap, title=title,
        show_colorbar=True, cbar_ax=cbar_ax,
    )

    legend_elements = [
        mpatches.Patch(facecolor="lightgrey", edgecolor="grey", hatch="///",
                       label="Coverage ≤ 0 (poorly constrained)"),
        mpatches.Patch(facecolor="white", edgecolor="white", hatch="xxx",
                       label=f"CV > {cv_threshold:.0f}% (high uncertainty)"),
    ]
    if probe_x is not None:
        legend_elements.append(
            plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="black",
                       markersize=6, label="Water content probe")
        )
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6, framealpha=0.8)
    fig.suptitle(title, fontsize=10, y=1.01)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Public: time-lapse XZ grid (Fig. 6 layout)
# ---------------------------------------------------------------------------

def plot_water_content_timelapse(
    theta_calc: np.ndarray,
    cv_map: np.ndarray,
    x_coords: np.ndarray,
    depths0: list[float],
    coverage: np.ndarray | None = None,
    probe_x: Sequence[float] | None = None,
    probe_depths: Sequence[float] | None = None,
    survey_labels: Sequence[str] | None = None,
    theta_vmin: float = 0.0,
    theta_vmax: float = 45.0,
    cv_threshold: float = 100.0,
    coverage_threshold: float = 0.0,
    cmap: str = "Blues",
    n_cols: int = 2,
    output_dir: str = ".",
    scenario_name: str = "scenario",
    dpi: int = 150,
) -> plt.Figure:
    """Grid of XZ water-content panels, one per survey time step."""
    if theta_calc.ndim == 2:
        theta_calc = theta_calc[np.newaxis]
        cv_map     = cv_map[np.newaxis]
        if coverage is not None:
            coverage = coverage[np.newaxis]

    n_time, n_x, n_layers = theta_calc.shape

    if survey_labels is None:
        survey_labels = [f"t = {i}" for i in range(n_time)]

    n_rows = int(np.ceil(n_time / n_cols))
    fig_w  = 6.0 * n_cols + 1.2
    fig_h  = 2.5 * n_rows
    fig    = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        n_rows, n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.04],
        hspace=0.45, wspace=0.22,
        left=0.06, right=0.96, top=0.91, bottom=0.10,
    )
    cbar_ax = fig.add_subplot(gs[:, -1])
    sm_ref  = None

    for t in range(n_time):
        row, col = divmod(t, n_cols)
        ax = fig.add_subplot(gs[row, col])
        cov_t  = coverage[t] if coverage is not None else None
        show_y = (col == 0)
        show_x = (row == n_rows - 1) or (t >= n_time - n_cols)

        sm = _plot_xz_panel(
            ax=ax, theta=theta_calc[t], cv=cv_map[t], coverage=cov_t,
            x_coords=x_coords, depths0=depths0,
            probe_x=probe_x if t == 0 else None,
            probe_depths=probe_depths if t == 0 else None,
            theta_vmin=theta_vmin, theta_vmax=theta_vmax,
            cv_threshold=cv_threshold, coverage_threshold=coverage_threshold,
            cmap=cmap, title=survey_labels[t],
            show_xlabel=show_x, show_ylabel=show_y,
        )
        sm_ref = sm

    for t in range(n_time, n_rows * n_cols):
        row, col = divmod(t, n_cols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    if sm_ref is not None:
        cbar = fig.colorbar(sm_ref, cax=cbar_ax)
        cbar.set_label("θ (% vol.)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    legend_elements = [
        mpatches.Patch(facecolor="lightgrey", edgecolor="grey", hatch="///",
                       label="Coverage ≤ 0"),
        mpatches.Patch(facecolor="white", edgecolor="lightgrey", hatch="xxx",
                       label=f"CV > {cv_threshold:.0f}%"),
    ]
    if probe_x is not None:
        legend_elements.append(
            plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="black",
                       markersize=6, label="Water content probe")
        )
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=len(legend_elements), fontsize=7, framealpha=0.85,
               bbox_to_anchor=(0.46, 0.01))

    fig.suptitle(f"Soil Water Content (XZ) — scenario: {scenario_name}",
                 fontsize=11, y=0.97)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"water_content_xz_{scenario_name}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved XZ timelapse: {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Public: single-epoch XY plan view (2-D mode)
# ---------------------------------------------------------------------------

def plot_water_content_xy(
    theta_calc: np.ndarray,   # (n_points, n_layers) or (n_points,) % vol.
    utm_x: np.ndarray,        # (n_points,) UTM easting [m]
    utm_y: np.ndarray,        # (n_points,) UTM northing [m]
    layer_index: int | None = None,
    theta_vmin: float = 0.0,
    theta_vmax: float = 45.0,
    cmap: str = "Blues",
    title: str = "Soil Water Content — Plan View (XY)",
    output_path: str | None = None,
    dpi: int = 150,
    probe_utm_x: Sequence[float] | None = None,
    probe_utm_y: Sequence[float] | None = None,
) -> plt.Figure:
    """
    Plot a plan-view XY map of θ at a chosen layer or averaged over all
    layers.

    Parameters
    ----------
    theta_calc : ndarray, shape (n_points, n_layers) or (n_points,)
        Volumetric water content in % vol.
    utm_x, utm_y : ndarray, shape (n_points,)
        Absolute UTM coordinates of each measurement point.
    layer_index : int or None
        If None (default), the map shows the depth-average over all layers.
        Pass an integer (0-based) to show a single layer.
    """
    if theta_calc.ndim == 2:
        if layer_index is not None:
            theta_map = theta_calc[:, layer_index]
            layer_str = f"Layer {layer_index}"
        else:
            theta_map = theta_calc.mean(axis=1)
            layer_str = "mean over layers"
    else:
        theta_map = theta_calc
        layer_str = ""

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.025, 0.70])

    _plot_xy_panel(
        ax=ax, theta_layer_mean=theta_map,
        utm_x=utm_x, utm_y=utm_y,
        theta_vmin=theta_vmin, theta_vmax=theta_vmax,
        cmap=cmap,
        title=f"{title}\n({layer_str})" if layer_str else title,
        show_colorbar=True, cbar_ax=cbar_ax,
        probe_utm_x=probe_utm_x, probe_utm_y=probe_utm_y,
    )

    if probe_utm_x is not None:
        ax.legend(fontsize=7, loc="upper right")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved XY plan view: {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Public: time-lapse XY grid
# ---------------------------------------------------------------------------

def plot_water_content_xy_timelapse(
    theta_calc: np.ndarray,   # (n_time, n_points, n_layers) or (n_time, n_points)
    utm_x: np.ndarray,        # (n_points,)
    utm_y: np.ndarray,        # (n_points,)
    survey_labels: Sequence[str] | None = None,
    layer_index: int | None = None,
    theta_vmin: float = 0.0,
    theta_vmax: float = 45.0,
    cmap: str = "Blues",
    n_cols: int = 2,
    output_dir: str = ".",
    scenario_name: str = "scenario",
    dpi: int = 150,
    probe_utm_x: Sequence[float] | None = None,
    probe_utm_y: Sequence[float] | None = None,
) -> plt.Figure:
    """Grid of XY plan-view panels, one per survey time step (2-D mode)."""
    if theta_calc.ndim == 3:
        if layer_index is not None:
            theta_maps = theta_calc[:, :, layer_index]
            layer_str  = f"Layer {layer_index}"
        else:
            theta_maps = theta_calc.mean(axis=2)
            layer_str  = "mean over layers"
    else:
        theta_maps = theta_calc
        layer_str  = ""

    n_time = theta_maps.shape[0]
    if survey_labels is None:
        survey_labels = [f"t = {i}" for i in range(n_time)]

    n_rows = int(np.ceil(n_time / n_cols))
    fig_w  = 5.5 * n_cols + 1.0
    fig_h  = 5.0 * n_rows
    fig    = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        n_rows, n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.04],
        hspace=0.40, wspace=0.25,
        left=0.08, right=0.96, top=0.91, bottom=0.06,
    )
    cbar_ax = fig.add_subplot(gs[:, -1])
    sc_ref  = None

    for t in range(n_time):
        row, col = divmod(t, n_cols)
        ax = fig.add_subplot(gs[row, col])
        show_y = (col == 0)
        show_x = (row == n_rows - 1) or (t >= n_time - n_cols)

        sc = _plot_xy_panel(
            ax=ax, theta_layer_mean=theta_maps[t],
            utm_x=utm_x, utm_y=utm_y,
            theta_vmin=theta_vmin, theta_vmax=theta_vmax,
            cmap=cmap,
            title=f"{survey_labels[t]}" + (f" ({layer_str})" if layer_str else ""),
            show_xlabel=show_x, show_ylabel=show_y,
            probe_utm_x=probe_utm_x if t == 0 else None,
            probe_utm_y=probe_utm_y if t == 0 else None,
        )
        sc_ref = sc

    for t in range(n_time, n_rows * n_cols):
        row, col = divmod(t, n_cols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    if sc_ref is not None:
        cbar = fig.colorbar(sc_ref, cax=cbar_ax)
        cbar.set_label("θ (% vol.)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f"Soil Water Content (XY Plan View) — scenario: {scenario_name}",
                 fontsize=11, y=0.97)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"water_content_xy_{scenario_name}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved XY timelapse: {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Public: paired θ / CV XZ panels per time step (uncertainty figure)
# ---------------------------------------------------------------------------

def plot_water_content_paired(
    theta_calc: np.ndarray,
    cv_map: np.ndarray,
    x_coords: np.ndarray,
    depths0: list[float],
    coverage: np.ndarray | None = None,
    probe_x: Sequence[float] | None = None,
    probe_depths: Sequence[float] | None = None,
    survey_labels: Sequence[str] | None = None,
    theta_vmin: float = 0.0,
    theta_vmax: float = 45.0,
    cv_threshold: float = 30.0,
    cv_vmax: float = 150.0,
    coverage_threshold: float = 0.0,
    cmap_theta: str = "Blues",
    output_dir: str = ".",
    scenario_name: str = "scenario",
    dpi: int = 150,
) -> plt.Figure:
    """Paired XZ panels: θ cross-section (left) + CV cross-section (right)."""
    if theta_calc.ndim == 2:
        theta_calc = theta_calc[np.newaxis]
        cv_map     = cv_map[np.newaxis]
        if coverage is not None:
            coverage = coverage[np.newaxis]

    n_time, n_x, n_layers = theta_calc.shape
    cv_pct = cv_map * 100.0

    if survey_labels is None:
        survey_labels = [f"t = {i}" for i in range(n_time)]

    zb = _z_bounds(depths0)
    xe = _x_edges(x_coords)

    from matplotlib.colors import LinearSegmentedColormap
    cv_cmap = LinearSegmentedColormap.from_list(
        "cv_uncertainty", ["white", "#FF8C00", "#CC0000"]
    )
    cv_norm = plt.Normalize(vmin=0.0, vmax=cv_vmax)

    fig   = plt.figure(figsize=(14.0, 2.8 * n_time))
    gs    = fig.add_gridspec(
        n_time, 5,
        width_ratios=[1.0, 0.03, 0.06, 1.0, 0.03],
        hspace=0.50, wspace=0.12,
        left=0.06, right=0.97, top=0.93, bottom=0.08,
    )

    sm_theta_ref = sm_cv_ref = None

    for t in range(n_time):
        ax_th = fig.add_subplot(gs[t, 0])
        ax_cv = fig.add_subplot(gs[t, 3])

        theta_t = theta_calc[t]
        cv_t    = cv_pct[t]
        cov_t   = coverage[t] if coverage is not None else None

        X_mesh, Z_mesh = np.meshgrid(xe, zb)

        # Left: θ
        norm_th  = plt.Normalize(vmin=theta_vmin, vmax=theta_vmax)
        sm_theta = ax_th.pcolormesh(X_mesh, Z_mesh, theta_t.T,
                                    cmap=cmap_theta, norm=norm_th, shading="flat")
        sm_theta_ref = sm_theta

        if cov_t is not None:
            _add_hatching(ax_th, cov_t <= coverage_threshold, xe, zb,
                          hatch="///", facecolor="lightgrey", edgecolor="grey")
        _add_hatching(ax_th, cv_t > 100.0, xe, zb,
                      hatch="xxx", facecolor="white", edgecolor="white", alpha=0.8)

        if probe_x is not None and probe_depths is not None:
            ax_th.scatter(probe_x, probe_depths, marker="v", s=40,
                          color="black", zorder=5)

        ax_th.invert_yaxis()
        ax_th.set_xlim(xe[0], xe[-1])
        ax_th.set_ylim(zb[-1], 0.0)
        ax_th.set_title(f"{survey_labels[t]} — θ (% vol.)", fontsize=8, pad=3)
        ax_th.set_ylabel("Depth (m)", fontsize=8)
        ax_th.tick_params(labelsize=7)
        if t == n_time - 1:
            ax_th.set_xlabel("Distance along transect (m)", fontsize=8)

        # Right: CV
        sm_cv = ax_cv.pcolormesh(X_mesh, Z_mesh, cv_t.T,
                                  cmap=cv_cmap, norm=cv_norm, shading="flat")
        sm_cv_ref = sm_cv

        x_centers = (xe[:-1] + xe[1:]) / 2
        z_centers = (zb[:-1] + zb[1:]) / 2
        Xc, Zc   = np.meshgrid(x_centers, z_centers)
        try:
            ax_cv.contour(Xc, Zc, cv_t.T, levels=[cv_threshold],
                          colors=["#FF4500"], linewidths=1.2, linestyles="--")
        except Exception:
            pass
        try:
            theta_levels = np.linspace(theta_vmin + 5, theta_vmax - 5, 4)
            cs = ax_cv.contour(Xc, Zc, theta_t.T, levels=theta_levels,
                               colors=["#333333"], linewidths=0.7, linestyles="-", alpha=0.6)
            ax_cv.clabel(cs, inline=True, fontsize=5, fmt="%.0f%%")
        except Exception:
            pass

        _add_hatching(ax_cv, cv_t > 100.0, xe, zb,
                      hatch="xxx", facecolor="white", edgecolor="white", alpha=0.7)

        if probe_x is not None and probe_depths is not None:
            ax_cv.scatter(probe_x, probe_depths, marker="v", s=40,
                          color="black", zorder=5)

        ax_cv.invert_yaxis()
        ax_cv.set_xlim(xe[0], xe[-1])
        ax_cv.set_ylim(zb[-1], 0.0)
        ax_cv.set_title(f"{survey_labels[t]} — CV (%)", fontsize=8, pad=3)
        ax_cv.tick_params(labelsize=7)
        if t == n_time - 1:
            ax_cv.set_xlabel("Distance along transect (m)", fontsize=8)

    if sm_theta_ref is not None:
        cbar_th = fig.colorbar(sm_theta_ref, cax=fig.add_subplot(gs[:, 1]))
        cbar_th.set_label("θ (% vol.)", fontsize=8)
        cbar_th.ax.tick_params(labelsize=7)
    if sm_cv_ref is not None:
        cbar_cv = fig.colorbar(sm_cv_ref, cax=fig.add_subplot(gs[:, 4]))
        cbar_cv.set_label("CV (%)", fontsize=8)
        cbar_cv.ax.tick_params(labelsize=7)

    legend_elements = [
        mpatches.Patch(facecolor="lightgrey", edgecolor="grey", hatch="///",
                       label="Coverage ≤ 0 (poorly constrained)"),
        mpatches.Patch(facecolor="white", edgecolor="lightgrey", hatch="xxx",
                       label="CV > 100 % (high uncertainty)"),
        plt.Line2D([0], [0], color="#FF4500", linewidth=1.2, linestyle="--",
                   label=f"CV = {cv_threshold:.0f}% threshold"),
        plt.Line2D([0], [0], color="#333333", linewidth=0.9, linestyle="-",
                   label="θ contours (% vol.)"),
    ]
    if probe_x is not None:
        legend_elements.append(
            plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="black",
                       markersize=6, label="Water content probe")
        )
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=min(len(legend_elements), 3), fontsize=7, framealpha=0.85,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(f"Soil Water Content & Uncertainty (XZ) — scenario: {scenario_name}",
                 fontsize=11, y=0.97)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"water_content_paired_{scenario_name}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved paired θ/CV (XZ): {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Public: pipeline convenience wrapper
# ---------------------------------------------------------------------------

def plot_from_pipeline(
    selected_output,
    x_transect: np.ndarray,
    depths0: list[float],
    probe_x_coords: list[float],
    probe_depths: list[float],
    output_dir: str,
    scenario_name: str = "default",
    timelapse: bool = False,
    survey_labels: Sequence[str] | None = None,
    # 2-D mode extras (optional — passed through from run_pipeline when in 2d mode)
    utm_x: np.ndarray | None = None,
    utm_y: np.ndarray | None = None,
    probe_y_coords: list[float] | None = None,   # UTM northing of each probe — for XY markers
    y_transect: np.ndarray | None = None,
) -> None:
    """
    Render all water-content figures from pipeline output.

    XZ figures are always produced (cross-section view).
    XY figures are produced when ``utm_x`` and ``utm_y`` are supplied
    (i.e. in 2-D mode).

    Parameters
    ----------
    utm_x, utm_y : ndarray or None
        UTM coordinates of each measurement point; loaded by run_pipeline
        from ``synthetic_utm_x.npy`` / ``synthetic_utm_y.npy`` when
        ``--mode 2d`` was used.
    probe_y_coords : list of float or None
        UTM northing of each TDR probe, same order as ``probe_x_coords``.
        When supplied together with ``utm_x``/``utm_y``, probe markers are
        drawn on XY plan-view plots.  In 1-D mode all values are 0.0 and
        markers are omitted from XY plots automatically.
    y_transect : ndarray or None
        Local y-offsets (northing) used as a second spatial axis in XZ plots
        when the transect is truly 2-D.  Usually not needed; pass None.
    """
    # --- Robust attribute extraction ---
    theta_calc = _extract_theta_calc(selected_output)

    # m³/m³ → % vol.
    theta_calc = theta_calc * 100.0

    if hasattr(selected_output, "cv_map"):
        cv_map = np.asarray(selected_output.cv_map)
    elif hasattr(selected_output, "cv"):
        cv_map = np.asarray(selected_output.cv)
    else:
        cv_map = np.zeros_like(theta_calc)
        print("[WARNING] 'cv_map' missing. Using zero-filled array.")

    coverage = (
        np.asarray(selected_output.coverage)
        if hasattr(selected_output, "coverage") and selected_output.coverage is not None
        else None
    )

    vmax = float(np.nanpercentile(theta_calc, 98))
    vmax = max(5.0, np.ceil(vmax / 5) * 5)

    has_utm = (utm_x is not None and utm_y is not None)

    # Resolve probe UTM coords for XY marker overlay.
    # In 2-D mode probe_x_coords and probe_y_coords are real UTM values.
    # In 1-D mode probe_y_coords are all 0.0 — suppress XY markers in that case.
    _probe_utm_x: list[float] | None = None
    _probe_utm_y: list[float] | None = None
    if has_utm and probe_y_coords is not None:
        has_real_y = any(y != 0.0 for y in probe_y_coords)
        if has_real_y:
            _probe_utm_x = list(probe_x_coords)
            _probe_utm_y = list(probe_y_coords)

    # -----------------------------------------------------------------------
    # XZ plots (always)
    # -----------------------------------------------------------------------
    if timelapse and theta_calc.ndim == 3:
        plot_water_content_timelapse(
            theta_calc=theta_calc, cv_map=cv_map,
            x_coords=x_transect, depths0=depths0,
            coverage=coverage,
            probe_x=probe_x_coords, probe_depths=probe_depths,
            survey_labels=survey_labels,
            theta_vmin=0.0, theta_vmax=vmax,
            output_dir=output_dir, scenario_name=scenario_name,
        )
        plot_water_content_paired(
            theta_calc=theta_calc, cv_map=cv_map,
            x_coords=x_transect, depths0=depths0,
            coverage=coverage,
            probe_x=probe_x_coords, probe_depths=probe_depths,
            survey_labels=survey_labels,
            theta_vmin=0.0, theta_vmax=vmax,
            output_dir=output_dir, scenario_name=scenario_name,
        )
    else:
        tc  = theta_calc[0] if theta_calc.ndim == 3 else theta_calc
        cv  = cv_map[0]     if cv_map.ndim == 3     else cv_map
        cov = (coverage[0]  if (coverage is not None and coverage.ndim == 3)
               else coverage)

        plot_water_content_section(
            theta_calc=tc, cv_map=cv,
            x_coords=x_transect, depths0=depths0,
            coverage=cov,
            probe_x=probe_x_coords, probe_depths=probe_depths,
            theta_vmin=0.0, theta_vmax=vmax,
            title=f"Soil Water Content (XZ) — {scenario_name}",
            output_path=os.path.join(output_dir, f"water_content_xz_{scenario_name}.png"),
        )
        plot_water_content_paired(
            theta_calc=tc, cv_map=cv,
            x_coords=x_transect, depths0=depths0,
            coverage=cov,
            probe_x=probe_x_coords, probe_depths=probe_depths,
            theta_vmin=0.0, theta_vmax=vmax,
            output_dir=output_dir, scenario_name=scenario_name,
        )

    # -----------------------------------------------------------------------
    # XY plots (only when UTM coords are available — 2-D mode)
    # -----------------------------------------------------------------------
    if has_utm:
        if timelapse and theta_calc.ndim == 3:
            plot_water_content_xy_timelapse(
                theta_calc=theta_calc,
                utm_x=utm_x, utm_y=utm_y,
                survey_labels=survey_labels,
                theta_vmin=0.0, theta_vmax=vmax,
                output_dir=output_dir, scenario_name=scenario_name,
                probe_utm_x=_probe_utm_x,   # None in 1-D mode, real coords in 2-D
                probe_utm_y=_probe_utm_y,
            )
        else:
            tc = theta_calc[0] if theta_calc.ndim == 3 else theta_calc
            plot_water_content_xy(
                theta_calc=tc,
                utm_x=utm_x, utm_y=utm_y,
                theta_vmin=0.0, theta_vmax=vmax,
                title=f"Soil Water Content (XY) — {scenario_name}",
                output_path=os.path.join(
                    output_dir, f"water_content_xy_{scenario_name}.png"
                ),
                probe_utm_x=_probe_utm_x,   # None in 1-D mode, real coords in 2-D
                probe_utm_y=_probe_utm_y,
            )


# ---------------------------------------------------------------------------
# Public: ground-truth counterpart of plot_from_pipeline
# ---------------------------------------------------------------------------

def plot_truth_from_pipeline(
    theta_true: np.ndarray,      # (n_time, n_pos, n_layers) or (n_pos, n_layers), m^3/m^3
    x_transect: np.ndarray,
    depths0: list[float],
    probe_x_coords: list[float],
    probe_depths: list[float],
    output_dir: str,
    scenario_name: str = "default",
    timelapse: bool = False,
    survey_labels: Sequence[str] | None = None,
    theta_vmax: float | None = None,
    # 2-D mode extras (optional — same as plot_from_pipeline)
    utm_x: np.ndarray | None = None,
    utm_y: np.ndarray | None = None,
    probe_y_coords: list[float] | None = None,
) -> None:
    """
    Render the ground-truth counterpart of ``plot_from_pipeline``'s XZ (and,
    in 2-D mode, XY) water-content figures — from the synthetic ground-truth
    array (``theta_true_realization``) rather than the inverted
    ``selected_output.theta_calc``.

    Only meaningful when synthetic data with a known truth is available
    (``data_source == "synthetic"``); the caller is responsible for skipping
    this when running on real field data (no truth to plot).

    Deliberately thin: reuses the same public ``plot_water_content_*``
    functions as ``plot_from_pipeline`` rather than duplicating the panel
    logic, so styling always stays in sync. Ground truth has no inversion
    uncertainty, so CV/coverage hatching is disabled and ``plot_water_content_
    paired`` (θ+CV) is skipped — there is no CV to pair with.

    Filenames are tagged with a ``_truth`` suffix on the scenario name
    (e.g. ``water_content_xz_2d_truth.png``), so they sit alongside and are
    directly comparable to the inverted ``water_content_xz_2d.png``.

    Parameters
    ----------
    theta_true : ndarray
        Volumetric water content in m^3/m^3 (as produced by
        ``0_generate_synthetic_data.py`` / ``theta_true_realization``).
        Converted to % vol. internally to match the inverted plots' units.
    theta_vmax : float or None
        Pass the SAME vmax used for the inverted plot (from
        ``plot_from_pipeline``) so truth and inverted panels share one
        color scale and are visually comparable side by side. If None
        (default), it is computed independently from the truth data's
        98th percentile, same convention as plot_from_pipeline.
    """
    theta_true = np.asarray(theta_true, dtype=float)
    if theta_true.ndim not in (2, 3):
        raise ValueError(
            f"[plot_truth_from_pipeline] Expected theta_true with 2 or 3 dims "
            f"(n_pos, n_layers) or (n_time, n_pos, n_layers); got shape {theta_true.shape}."
        )
    n_pos_truth = theta_true.shape[-2]
    if n_pos_truth != len(x_transect):
        print(
            f"[plot_truth_from_pipeline][WARN] theta_true has {n_pos_truth} spatial "
            f"points but x_transect has {len(x_transect)}; truth and inverted plots "
            "may not be spatially aligned. Skipping truth XZ/XY rendering."
        )
        return

    theta_true_pct = theta_true * 100.0  # m^3/m^3 -> % vol., matches plot_from_pipeline

    if theta_vmax is None:
        theta_vmax = float(np.nanpercentile(theta_true_pct, 98))
        theta_vmax = max(5.0, np.ceil(theta_vmax / 5) * 5)

    has_utm = (utm_x is not None and utm_y is not None)

    _probe_utm_x: list[float] | None = None
    _probe_utm_y: list[float] | None = None
    if has_utm and probe_y_coords is not None:
        has_real_y = any(y != 0.0 for y in probe_y_coords)
        if has_real_y:
            _probe_utm_x = list(probe_x_coords)
            _probe_utm_y = list(probe_y_coords)

    truth_scenario_name = f"{scenario_name}_truth"

    # -----------------------------------------------------------------------
    # XZ plots (always)
    # -----------------------------------------------------------------------
    if timelapse and theta_true_pct.ndim == 3:
        cv_zero = np.zeros_like(theta_true_pct)
        plot_water_content_timelapse(
            theta_calc=theta_true_pct, cv_map=cv_zero,
            x_coords=x_transect, depths0=depths0,
            coverage=None,
            probe_x=probe_x_coords, probe_depths=probe_depths,
            survey_labels=survey_labels,
            theta_vmin=0.0, theta_vmax=theta_vmax,
            output_dir=output_dir, scenario_name=truth_scenario_name,
        )
    else:
        tc = theta_true_pct[0] if theta_true_pct.ndim == 3 else theta_true_pct
        plot_water_content_section(
            theta_calc=tc, cv_map=np.zeros_like(tc),
            x_coords=x_transect, depths0=depths0,
            coverage=None,
            probe_x=probe_x_coords, probe_depths=probe_depths,
            theta_vmin=0.0, theta_vmax=theta_vmax,
            title=f"Soil Water Content (XZ) — Ground Truth — {scenario_name}",
            output_path=os.path.join(output_dir, f"water_content_xz_{truth_scenario_name}.png"),
        )

    # -----------------------------------------------------------------------
    # XY plots (only when UTM coords are available — 2-D mode)
    # -----------------------------------------------------------------------
    if has_utm:
        if timelapse and theta_true_pct.ndim == 3:
            plot_water_content_xy_timelapse(
                theta_calc=theta_true_pct,
                utm_x=utm_x, utm_y=utm_y,
                survey_labels=survey_labels,
                theta_vmin=0.0, theta_vmax=theta_vmax,
                output_dir=output_dir, scenario_name=truth_scenario_name,
                probe_utm_x=_probe_utm_x,
                probe_utm_y=_probe_utm_y,
            )
        else:
            tc = theta_true_pct[0] if theta_true_pct.ndim == 3 else theta_true_pct
            plot_water_content_xy(
                theta_calc=tc,
                utm_x=utm_x, utm_y=utm_y,
                theta_vmin=0.0, theta_vmax=theta_vmax,
                title=f"Soil Water Content (XY) — Ground Truth — {scenario_name}",
                output_path=os.path.join(
                    output_dir, f"water_content_xy_{truth_scenario_name}.png"
                ),
                probe_utm_x=_probe_utm_x,
                probe_utm_y=_probe_utm_y,
            )


# ---------------------------------------------------------------------------
# Core XZ single-panel — % DIFFERENCE (diverging colormap centred at 0)
# ---------------------------------------------------------------------------

def _plot_xz_diff_panel(
    ax: plt.Axes,
    pct_diff: np.ndarray,         # (n_x, n_layers)  signed % difference
    x_coords: np.ndarray,
    depths0: list[float],
    probe_x: Sequence[float] | None = None,
    probe_depths: Sequence[float] | None = None,
    diff_vmax: float = 25.0,
    cmap: str = "RdBu_r",
    title: str = "",
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_colorbar: bool = False,
    cbar_ax: plt.Axes | None = None,
    xlabel: str = "Distance along transect (m)",
) -> plt.cm.ScalarMappable:
    """
    Draw one XZ % -difference panel on *ax*, colour-symmetric about zero
    (blue = calculated wetter than truth, red = calculated drier than
    truth, white = agreement) so over- and under-estimation are visually
    distinguishable at a glance. Returns the ScalarMappable.
    """
    zb = _z_bounds(depths0)
    xe = _x_edges(x_coords)

    norm     = plt.Normalize(vmin=-diff_vmax, vmax=diff_vmax)
    cmap_obj = plt.get_cmap(cmap)

    X_mesh, Z_mesh = np.meshgrid(xe, zb)
    masked = np.ma.masked_invalid(pct_diff.T)
    sm = ax.pcolormesh(X_mesh, Z_mesh, masked, cmap=cmap_obj, norm=norm, shading="flat")
    cmap_obj.set_bad(color="lightgrey")

    if probe_x is not None and probe_depths is not None:
        ax.scatter(probe_x, probe_depths, marker="v", s=40,
                   color="black", zorder=5, label="Probe")

    ax.invert_yaxis()
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(zb[-1], 0.0)

    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Depth (m)", fontsize=8)

    ax.set_title(title, fontsize=8, pad=3)
    ax.tick_params(labelsize=7)

    if show_colorbar and cbar_ax is not None:
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Δθ vs. truth (%)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    return sm


def _compute_pct_diff(theta_calc: np.ndarray, theta_true: np.ndarray,
                       zero_floor: float = 1e-6) -> np.ndarray:
    """
    Signed relative percent difference of *theta_calc* vs. *theta_true*,
    both in the SAME units (m^3/m^3 or % vol., doesn't matter as long as
    they match — the ratio cancels the unit):

        pct_diff = (theta_calc - theta_true) / theta_true * 100

    Positive  → calculated wetter than truth.
    Negative  → calculated drier than truth.
    Cells where ``theta_true`` is (near) zero are masked to NaN rather
    than producing a divide-by-zero blow-up, since soil water content is
    never truly zero in these synthetic scenarios and a near-zero truth
    cell is far more likely a numerical/edge artefact than a meaningful
    reference value.
    """
    theta_calc = np.asarray(theta_calc, dtype=float)
    theta_true = np.asarray(theta_true, dtype=float)
    if theta_calc.shape != theta_true.shape:
        raise ValueError(
            f"[_compute_pct_diff] Shape mismatch: theta_calc {theta_calc.shape} "
            f"vs. theta_true {theta_true.shape}."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_diff = np.where(
            np.abs(theta_true) > zero_floor,
            (theta_calc - theta_true) / theta_true * 100.0,
            np.nan,
        )
    return pct_diff


def plot_water_content_pct_diff(
    theta_calc: np.ndarray,
    theta_true: np.ndarray,
    x_coords: np.ndarray,
    depths0: list[float],
    probe_x: Sequence[float] | None = None,
    probe_depths: Sequence[float] | None = None,
    diff_vmax: float | None = None,
    cmap: str = "RdBu_r",
    title: str = "Water Content % Difference (Calculated vs. Truth)",
    output_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot a single XZ panel of the signed % difference between a calculated
    water-content section and its ground-truth counterpart (one epoch).
    Both inputs must be in the same units and shaped (n_x, n_layers).
    """
    pct_diff = _compute_pct_diff(theta_calc, theta_true)

    if diff_vmax is None:
        finite = pct_diff[np.isfinite(pct_diff)]
        diff_vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 25.0
        diff_vmax = max(5.0, np.ceil(diff_vmax / 5) * 5)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])

    _plot_xz_diff_panel(
        ax=ax, pct_diff=pct_diff,
        x_coords=x_coords, depths0=depths0,
        probe_x=probe_x, probe_depths=probe_depths,
        diff_vmax=diff_vmax, cmap=cmap, title=title,
        show_colorbar=True, cbar_ax=cbar_ax,
    )

    legend_elements = [
        mpatches.Patch(facecolor="lightgrey", edgecolor="grey",
                       label="Truth ≈ 0 (undefined % diff, masked)"),
    ]
    if probe_x is not None:
        legend_elements.append(
            plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="black",
                       markersize=6, label="Water content probe")
        )
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6, framealpha=0.8)

    n_finite = int(np.isfinite(pct_diff).sum())
    if n_finite:
        mean_abs = float(np.nanmean(np.abs(pct_diff)))
        fig.suptitle(f"{title}\nmean |Δ| = {mean_abs:.1f} %  (n={n_finite} cells)",
                     fontsize=10, y=1.03)
    else:
        fig.suptitle(title, fontsize=10, y=1.01)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved: {output_path}")

    return fig


def plot_water_content_pct_diff_timelapse(
    theta_calc: np.ndarray,
    theta_true: np.ndarray,
    x_coords: np.ndarray,
    depths0: list[float],
    probe_x: Sequence[float] | None = None,
    probe_depths: Sequence[float] | None = None,
    survey_labels: Sequence[str] | None = None,
    diff_vmax: float | None = None,
    cmap: str = "RdBu_r",
    n_cols: int = 2,
    output_dir: str = ".",
    scenario_name: str = "scenario",
    dpi: int = 150,
) -> plt.Figure:
    """Grid of XZ % -difference panels, one per survey time step."""
    if theta_calc.ndim == 2:
        theta_calc = theta_calc[np.newaxis]
    if theta_true.ndim == 2:
        theta_true = theta_true[np.newaxis]

    n_time, n_x, n_layers = theta_calc.shape

    pct_diff_all = _compute_pct_diff(theta_calc, theta_true)

    if diff_vmax is None:
        finite = pct_diff_all[np.isfinite(pct_diff_all)]
        diff_vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 25.0
        diff_vmax = max(5.0, np.ceil(diff_vmax / 5) * 5)

    if survey_labels is None:
        survey_labels = [f"t = {i}" for i in range(n_time)]

    n_rows = int(np.ceil(n_time / n_cols))
    fig_w  = 6.0 * n_cols + 1.2
    fig_h  = 2.5 * n_rows
    fig    = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        n_rows, n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.04],
        hspace=0.45, wspace=0.22,
        left=0.06, right=0.96, top=0.91, bottom=0.10,
    )
    cbar_ax = fig.add_subplot(gs[:, -1])
    sm_ref  = None

    for t in range(n_time):
        row, col = divmod(t, n_cols)
        ax = fig.add_subplot(gs[row, col])
        show_y = (col == 0)
        show_x = (row == n_rows - 1) or (t >= n_time - n_cols)

        sm = _plot_xz_diff_panel(
            ax=ax, pct_diff=pct_diff_all[t],
            x_coords=x_coords, depths0=depths0,
            probe_x=probe_x if t == 0 else None,
            probe_depths=probe_depths if t == 0 else None,
            diff_vmax=diff_vmax, cmap=cmap, title=survey_labels[t],
            show_xlabel=show_x, show_ylabel=show_y,
        )
        sm_ref = sm

    for t in range(n_time, n_rows * n_cols):
        row, col = divmod(t, n_cols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    if sm_ref is not None:
        cbar = fig.colorbar(sm_ref, cax=cbar_ax)
        cbar.set_label("Δθ vs. truth (%)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    legend_elements = [
        mpatches.Patch(facecolor="lightgrey", edgecolor="grey",
                       label="Truth ≈ 0 (undefined % diff, masked)"),
    ]
    if probe_x is not None:
        legend_elements.append(
            plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="black",
                       markersize=6, label="Water content probe")
        )
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=len(legend_elements), fontsize=7, framealpha=0.85,
               bbox_to_anchor=(0.46, 0.01))

    fig.suptitle(f"Water Content % Difference (Calculated vs. Truth) — scenario: {scenario_name}",
                 fontsize=11, y=0.97)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"water_content_xz_{scenario_name}_pctdiff.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved XZ % -difference timelapse: {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Public: % difference between plot_from_pipeline and plot_truth_from_pipeline
# ---------------------------------------------------------------------------

def plot_pct_diff_from_pipeline(
    selected_output,
    theta_true: np.ndarray,      # (n_time, n_pos, n_layers) or (n_pos, n_layers), m^3/m^3
    x_transect: np.ndarray,
    depths0: list[float],
    probe_x_coords: list[float],
    probe_depths: list[float],
    output_dir: str,
    scenario_name: str = "default",
    timelapse: bool = False,
    survey_labels: Sequence[str] | None = None,
    diff_vmax: float | None = None,
) -> str | None:
    """
    Render the XZ % -difference figure between ``plot_from_pipeline``'s
    calculated water content and ``plot_truth_from_pipeline``'s ground
    truth — i.e. the figure requested to sit alongside
    ``water_content_xz_{scenario_name}.png`` and
    ``water_content_xz_{scenario_name}_truth.png``, saved as
    ``water_content_xz_{scenario_name}_pctdiff.png``.

    Call this the same way as ``plot_truth_from_pipeline`` (same
    ``theta_true``, ``x_transect``, etc.) immediately after both of those
    have run, so all three files share one scenario/epoch layout. Only
    meaningful in synthetic mode (a ground truth must exist).

    Returns the output file path, or ``None`` if rendering was skipped
    (spatial-point mismatch between ``theta_true`` and ``x_transect``).
    """
    theta_calc = _extract_theta_calc(selected_output)  # m^3/m^3
    theta_true = np.asarray(theta_true, dtype=float)   # m^3/m^3

    if theta_true.ndim not in (2, 3):
        raise ValueError(
            f"[plot_pct_diff_from_pipeline] Expected theta_true with 2 or 3 dims "
            f"(n_pos, n_layers) or (n_time, n_pos, n_layers); got shape {theta_true.shape}."
        )
    n_pos_truth = theta_true.shape[-2]
    if n_pos_truth != len(x_transect):
        print(
            f"[plot_pct_diff_from_pipeline][WARN] theta_true has {n_pos_truth} spatial "
            f"points but x_transect has {len(x_transect)}; truth and calculated arrays "
            "may not be spatially aligned. Skipping % -difference rendering."
        )
        return None
    if theta_calc.shape != theta_true.shape:
        print(
            f"[plot_pct_diff_from_pipeline][WARN] theta_calc shape {theta_calc.shape} "
            f"!= theta_true shape {theta_true.shape}. Skipping % -difference rendering."
        )
        return None

    if timelapse and theta_calc.ndim == 3:
        fig = plot_water_content_pct_diff_timelapse(
            theta_calc=theta_calc, theta_true=theta_true,
            x_coords=x_transect, depths0=depths0,
            probe_x=probe_x_coords, probe_depths=probe_depths,
            survey_labels=survey_labels,
            diff_vmax=diff_vmax,
            output_dir=output_dir, scenario_name=scenario_name,
        )
        return os.path.join(output_dir, f"water_content_xz_{scenario_name}_pctdiff.png")
    else:
        tc = theta_calc[0] if theta_calc.ndim == 3 else theta_calc
        tt = theta_true[0] if theta_true.ndim == 3 else theta_true
        output_path = os.path.join(output_dir, f"water_content_xz_{scenario_name}_pctdiff.png")
        plot_water_content_pct_diff(
            theta_calc=tc, theta_true=tt,
            x_coords=x_transect, depths0=depths0,
            probe_x=probe_x_coords, probe_depths=probe_depths,
            diff_vmax=diff_vmax,
            title=f"Water Content % Difference (Calculated vs. Truth) — {scenario_name}",
            output_path=output_path,
        )
        return output_path


# ---------------------------------------------------------------------------
# Public: INITIAL MODEL (starting guess) vs. TRUTH — two flavours
#
#   (1) plot_initial_model_vs_true_ec   -- depth-resolved EC_bulk (mS/m),
#       same XZ-heatmap style as water_content_xz_*.png. The initial model
#       (conds0) has no spatial variation by construction (it's the same
#       starting guess at every position), so it's rendered as flat
#       horizontal bands -- one panel for the layer-MIN guess, one for the
#       layer-MAX guess (the two extremes of cond_ranges) -- next to the
#       real, spatially-varying true EC_bulk panel. If the guess brackets
#       reality, the true panel's colours should sit "between" the two
#       flat-band panels.
#
#   (2) plot_initial_eca_vs_true_eca    -- per-coil apparent conductivity
#       (ECa), which is NOT depth-resolved (each coil has an overlapping,
#       not a discrete, depth sensitivity), so it can't use the XZ-heatmap
#       style at all. Instead: one subplot per coil, x = position along
#       transect, true (noisy, spatially-varying) ECa as points/line, and
#       the initial model's min/max guesses forward-modelled through
#       EMagPy (no inversion) into flat reference bands for comparison.
# ---------------------------------------------------------------------------

def plot_initial_model_vs_true_ec(
    cond_ranges: Sequence[Sequence[float]],
    depths0: list[float],
    x_coords: np.ndarray,
    ec_bulk_true: np.ndarray,      # (n_pos, n_layers) or (n_time, n_pos, n_layers), mS/m
    output_dir: str,
    scenario_name: str = "default",
    probe_x: Sequence[float] | None = None,
    probe_depths: Sequence[float] | None = None,
    epoch_index: int = 0,
    dpi: int = 150,
) -> str:
    """
    Three-panel XZ comparison, same visual style/colour-scale as
    ``water_content_xz_{scenario}.png``: the initial (starting-guess)
    conductivity model's layer-MIN and layer-MAX bounds (flat bands,
    since the starting guess doesn't vary spatially) next to the real,
    spatially-varying true EC_bulk. Saved as
    ``initial_model_vs_true_ec_xz_{scenario_name}.png``.

    If ``ec_bulk_true`` carries a time-lapse axis, only ``epoch_index``
    is shown (the starting guess is identical at every epoch anyway, so
    there's nothing new to gain from a full multi-epoch grid here).
    """
    ec_bulk_true = np.asarray(ec_bulk_true, dtype=float)
    if ec_bulk_true.ndim == 3:
        if ec_bulk_true.shape[0] <= epoch_index:
            raise ValueError(
                f"[plot_initial_model_vs_true_ec] epoch_index={epoch_index} out of range "
                f"for ec_bulk_true with {ec_bulk_true.shape[0]} epoch(s)."
            )
        ec_true_epoch = ec_bulk_true[epoch_index]
    elif ec_bulk_true.ndim == 2:
        ec_true_epoch = ec_bulk_true
    else:
        raise ValueError(
            f"[plot_initial_model_vs_true_ec] Expected 2 or 3 dims, got shape {ec_bulk_true.shape}."
        )

    n_pos, n_layers = ec_true_epoch.shape
    if n_pos != len(x_coords):
        raise ValueError(
            f"[plot_initial_model_vs_true_ec] ec_bulk_true has {n_pos} spatial points "
            f"but x_coords has {len(x_coords)}."
        )
    if len(cond_ranges) != n_layers:
        raise ValueError(
            f"[plot_initial_model_vs_true_ec] cond_ranges has {len(cond_ranges)} layer(s) "
            f"but ec_bulk_true has {n_layers}."
        )

    min_vec = np.array([r[0]  for r in cond_ranges], dtype=float)
    max_vec = np.array([r[-1] for r in cond_ranges], dtype=float)

    min_band = np.tile(min_vec, (n_pos, 1))   # flat across x -- (n_pos, n_layers)
    max_band = np.tile(max_vec, (n_pos, 1))

    vmax = max(float(np.nanpercentile(ec_true_epoch, 98)), float(max_vec.max()))
    vmin = min(float(np.nanpercentile(ec_true_epoch, 2)),  float(min_vec.min()))
    vmin = max(0.0, vmin)

    fig = plt.figure(figsize=(16, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05],
                          left=0.05, right=0.94, wspace=0.30)
    ax_min, ax_max, ax_true = (fig.add_subplot(gs[0, i]) for i in range(3))
    cbar_ax = fig.add_subplot(gs[0, 3])

    zero_cv_min  = np.zeros_like(min_band)
    zero_cv_max  = np.zeros_like(max_band)
    zero_cv_true = np.zeros_like(ec_true_epoch)

    _plot_xz_panel(
        ax=ax_min, theta=min_band, cv=zero_cv_min, coverage=None,
        x_coords=x_coords, depths0=depths0,
        cmap="viridis", title="Initial model — layer MIN guess",
        show_ylabel=True, value_label="EC_bulk (mS/m)",
        theta_vmin=vmin, theta_vmax=vmax,
    )
    _plot_xz_panel(
        ax=ax_max, theta=max_band, cv=zero_cv_max, coverage=None,
        x_coords=x_coords, depths0=depths0,
        cmap="viridis", title="Initial model — layer MAX guess",
        show_ylabel=False, value_label="EC_bulk (mS/m)",
        theta_vmin=vmin, theta_vmax=vmax,
    )
    sm = _plot_xz_panel(
        ax=ax_true, theta=ec_true_epoch, cv=zero_cv_true, coverage=None,
        x_coords=x_coords, depths0=depths0,
        probe_x=probe_x, probe_depths=probe_depths,
        cmap="viridis", title="True EC_bulk",
        show_ylabel=False, show_colorbar=True, cbar_ax=cbar_ax,
        value_label="EC_bulk (mS/m)",
        theta_vmin=vmin, theta_vmax=vmax,
    )

    fig.suptitle(
        f"Initial Model vs. Ground Truth — EC_bulk — scenario: {scenario_name}"
        + (f" (epoch {epoch_index})" if ec_bulk_true.ndim == 3 else ""),
        fontsize=11, y=1.04,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"initial_model_vs_true_ec_xz_{scenario_name}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {out_path}")
    return out_path


def _forward_model_single_point(
    ec_profile: Sequence[float],
    depths0: Sequence[float],
    coils: Sequence[str],
    forward_model: str = "CS",
) -> np.ndarray:
    """
    Forward-model ONE conductivity profile (no inversion) through EMagPy
    to get the ECa it would produce on the given coil set -- the same
    ``Problem.forward(...)`` call pattern used by
    ``0_generate_synthetic_data.py`` to build the synthetic truth ECa in
    the first place, so the two are directly comparable.
    """
    from emagpy import Problem

    models_input = [np.array([list(ec_profile)])]
    depths_input = [np.array([list(depths0)])]

    k_point = Problem()
    k_point.models = models_input
    k_point.depths = depths_input
    res_forward = k_point.forward(
        forwardModel=forward_model, coils=list(coils),
        models=models_input, depths=depths_input,
    )
    raw = res_forward[0]
    values = raw.values if hasattr(raw, "values") else np.asarray(raw)
    return np.asarray(values).flatten()


def plot_initial_eca_vs_true_eca(
    cond_ranges: Sequence[Sequence[float]],
    depths0: list[float],
    x_coords: np.ndarray,
    eca_true: np.ndarray,          # (n_pos, n_coils) or (n_time, n_pos, n_coils)
    coils: Sequence[str],
    output_dir: str,
    scenario_name: str = "default",
    forward_model: str = "CS",
    epoch_index: int = 0,
    n_cols: int = 3,
    dpi: int = 150,
) -> str | None:
    """
    Per-coil comparison of true (measured) apparent conductivity against
    what the initial model's layer-MIN and layer-MAX guesses would
    forward-predict for the same coil set -- ECa isn't depth-resolved so
    this uses per-coil x-vs-ECa panels (à la
    ``0_generate_synthetic_data.plot_synthetic_results``'s "Apparent EC"
    figure), not an XZ heatmap. Saved as
    ``initial_eca_vs_true_eca_{scenario_name}.png``.

    Returns ``None`` (and logs a warning) if EMagPy isn't importable or
    the forward call fails, rather than raising -- this is a QC extra,
    not a pipeline-critical step.
    """
    eca_true = np.asarray(eca_true, dtype=float)
    if eca_true.ndim == 3:
        eca_epoch = eca_true[epoch_index]
    elif eca_true.ndim == 2:
        eca_epoch = eca_true
    else:
        raise ValueError(
            f"[plot_initial_eca_vs_true_eca] Expected 2 or 3 dims, got shape {eca_true.shape}."
        )

    n_pos, n_coils = eca_epoch.shape
    if n_pos != len(x_coords):
        raise ValueError(
            f"[plot_initial_eca_vs_true_eca] eca_true has {n_pos} spatial points "
            f"but x_coords has {len(x_coords)}."
        )
    if n_coils != len(coils):
        raise ValueError(
            f"[plot_initial_eca_vs_true_eca] eca_true has {n_coils} coil(s) "
            f"but coils has {len(coils)}."
        )

    min_vec = [r[0]  for r in cond_ranges]
    max_vec = [r[-1] for r in cond_ranges]

    try:
        eca_min = _forward_model_single_point(min_vec, depths0, coils, forward_model)
        eca_max = _forward_model_single_point(max_vec, depths0, coils, forward_model)
    except Exception as e:
        print(
            f"[plot_initial_eca_vs_true_eca][WARN] Forward-modelling the initial model "
            f"failed ({type(e).__name__}: {e}); skipping this plot."
        )
        return None

    n_rows = int(np.ceil(n_coils / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows),
                              squeeze=False, sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, n_coils))
    x_idx = np.arange(1, n_pos + 1)

    for c in range(n_coils):
        row, col = divmod(c, n_cols)
        ax = axes[row][col]
        ax.plot(x_idx, eca_epoch[:, c], marker="o", markersize=4, linewidth=1.3,
                color=colors[c], label="True ECa (measured)")
        ax.axhline(eca_min[c], linestyle="--", linewidth=1.5, color="tab:red",
                   label="Initial model — MIN guess")
        ax.axhline(eca_max[c], linestyle="--", linewidth=1.5, color="tab:green",
                   label="Initial model — MAX guess")
        ax.set_title(coils[c], fontsize=9, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel("$EC_a$ (mS/m)", fontsize=8)
        if row == n_rows - 1:
            ax.set_xlabel("Position index along transect", fontsize=8)
        if c == 0:
            ax.legend(fontsize=6, loc="best")

    for c in range(n_coils, n_rows * n_cols):
        row, col = divmod(c, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"Initial Model vs. True Apparent Conductivity ($EC_a$) — scenario: {scenario_name}"
        + (f" (epoch {epoch_index})" if eca_true.ndim == 3 else ""),
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"initial_eca_vs_true_eca_{scenario_name}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Public: SWC plots for all three pipeline stages
#   Stage A — raw ensemble realisations (models_clean, sampled)
#   Stage B — averaged model at every stack depth K (averaged_models)
#   Stage C — optimal K model (selected_output)   ← same as plot_from_pipeline
# ---------------------------------------------------------------------------

def _ec_to_theta_grid(ec_grid: np.ndarray, best_fit) -> np.ndarray:
    """
    Convert a (..., n_layers) EC array [mS/m] to θ [m³/m³] using the
    per-horizon Waxman-Smits parameters stored in *best_fit*.

    Works for any leading shape: (n_pts, n_layers),
    (n_time, n_pts, n_layers), (ensemble, n_pts, n_layers), etc.
    """
    original_shape = ec_grid.shape
    n_layers = original_shape[-1]
    ec_flat  = ec_grid.reshape(-1, n_layers)
    theta_flat = np.empty_like(ec_flat)

    for l_idx in range(n_layers):
        if l_idx < len(best_fit.horizon_fits):
            hf = best_fit.horizon_fits[l_idx]
            ec_col = np.clip(ec_flat[:, l_idx], 1e-6, None)
            ratio  = hf.a * ec_col / (hf.sigma_w + hf.sigma_s)
            theta_flat[:, l_idx] = np.power(np.clip(ratio, 0.0, None), 1.0 / hf.m)
        else:
            theta_flat[:, l_idx] = 0.0

    return theta_flat.reshape(original_shape)


def plot_swc_ensemble_stages(
    models_clean: np.ndarray,
    averaged_models: list,
    selected_output,
    best_fit,
    x_transect: np.ndarray,
    depths0: list[float],
    output_dir: str,
    scenario_name: str = "default",
    timelapse: bool = False,
    survey_labels: Sequence[str] | None = None,
    utm_x: np.ndarray | None = None,
    utm_y: np.ndarray | None = None,
    probe_x_coords: list[float] | None = None,
    probe_y_coords: list[float] | None = None,
    probe_depths: list[float] | None = None,
    max_realisations: int = 4,
    cv_threshold: float = 100.0,
    dpi: int = 150,
) -> None:
    """
    Produce SWC (θ) visualisations for every stage of the EA-EM pipeline so
    that the evolution from raw realisations → ensemble average → optimal model
    can be inspected side-by-side.

    Stage A  —  raw ensemble realisations (``models_clean``)
        A uniform sample of up to *max_realisations* individual inversion
        results is converted to θ and plotted as XZ (and XY) panels.
        Useful for seeing the spread before any averaging.

    Stage B  —  averaged model at every stack depth K (``averaged_models``)
        The RMSE-weighted mean EC is converted to θ for K = 1, 2, …, K_opt
        and for K = K_max (all models).  Gives a visual tour of the
        convergence trajectory.

    Stage C  —  optimal K model (``selected_output``)
        Reuses the ``theta_calc`` already stored in *selected_output* (already
        converted inside Step 4/5), so no extra EC→θ conversion is needed
        here.  This matches exactly what ``plot_from_pipeline`` produces.

    All outputs land in ``<output_dir>/swc_stages/``.

    Parameters
    ----------
    models_clean      : (ensemble_size, [n_time,] n_pts, n_layers)  EC [mS/m]
    averaged_models   : list of arrays, one per K.
                        Each element: ([n_time,] n_pts, n_layers)  EC [mS/m]
    selected_output   : SelectionOutput from Step 4/5
    best_fit          : PetroFit with .horizon_fits used for EC→θ conversion
    x_transect        : (n_pts,) distance along transect [m]
    depths0           : layer bottom depths [m]
    output_dir        : root pipeline output directory
    scenario_name     : used in filenames and titles
    timelapse         : True when more than one survey time step
    survey_labels     : optional list of time-step labels
    utm_x, utm_y      : UTM coordinates for XY plan-view (2-D mode only)
    probe_x_coords    : probe easting / transect-x positions
    probe_y_coords    : probe northing positions (2-D mode, else all 0.0)
    probe_depths      : probe depths [m]
    max_realisations  : cap on how many Stage-A realisations to plot
    cv_threshold      : CV (%) above which cells are hatched as unreliable
    dpi               : PNG resolution
    """
    import os
    stages_dir = os.path.join(output_dir, "swc_stages")
    os.makedirs(stages_dir, exist_ok=True)

    has_utm = utm_x is not None and utm_y is not None
    _probe_utm_x: list[float] | None = None
    _probe_utm_y: list[float] | None = None
    if has_utm and probe_y_coords is not None:
        if any(y != 0.0 for y in probe_y_coords):
            _probe_utm_x = list(probe_x_coords) if probe_x_coords else None
            _probe_utm_y = list(probe_y_coords)

    # shape details
    timelapse_models = models_clean.ndim == 4   # (ens, n_time, n_pts, n_layers)
    ensemble_size    = models_clean.shape[0]
    K_opt            = selected_output.K
    K_max            = len(averaged_models)

    # dummy zero CV for stages A & B (no per-realisation uncertainty estimate)
    def _zero_cv(theta: np.ndarray) -> np.ndarray:
        return np.zeros_like(theta)

    def _vmax(theta_pct: np.ndarray) -> float:
        v = float(np.nanpercentile(theta_pct, 98))
        return max(5.0, np.ceil(v / 5) * 5)

    def _ensure_3d(arr: np.ndarray) -> np.ndarray:
        """Guarantee shape is (n_time, n_pts, n_layers)."""
        if arr.ndim == 2:
            return arr[np.newaxis]
        return arr

    # ------------------------------------------------------------------
    # STAGE A — raw realisations
    # ------------------------------------------------------------------
    print(f"\n[SWC STAGES] Stage A — raw realisations  (sample {min(max_realisations, ensemble_size)} of {ensemble_size})")
    stage_a_dir = os.path.join(stages_dir, "A_raw_realisations")
    os.makedirs(stage_a_dir, exist_ok=True)

    step = max(1, ensemble_size // max_realisations)
    sample_ens = list(range(0, ensemble_size, step))[:max_realisations]

    for ens_idx in sample_ens:
        ec_ens = models_clean[ens_idx]              # ([n_time,] n_pts, n_layers)
        theta  = _ec_to_theta_grid(ec_ens, best_fit) * 100.0   # → % vol.
        theta3 = _ensure_3d(theta)
        cv3    = _zero_cv(theta3)
        vm     = _vmax(theta3)
        tag    = f"{scenario_name}_ens{ens_idx:03d}"

        if timelapse and theta3.shape[0] > 1:
            plot_water_content_timelapse(
                theta_calc=theta3, cv_map=cv3,
                x_coords=x_transect, depths0=depths0,
                probe_x=probe_x_coords, probe_depths=probe_depths,
                survey_labels=survey_labels,
                theta_vmin=0.0, theta_vmax=vm,
                cv_threshold=cv_threshold,
                output_dir=stage_a_dir, scenario_name=tag, dpi=dpi,
            )
        else:
            tc = theta3[0]
            cv = cv3[0]
            plot_water_content_section(
                theta_calc=tc, cv_map=cv,
                x_coords=x_transect, depths0=depths0,
                probe_x=probe_x_coords, probe_depths=probe_depths,
                theta_vmin=0.0, theta_vmax=vm,
                cv_threshold=cv_threshold,
                title=f"SWC Stage A — realisation {ens_idx}  [{scenario_name}]",
                output_path=os.path.join(stage_a_dir, f"water_content_xz_{tag}.png"),
                dpi=dpi,
            )

        if has_utm:
            if timelapse and theta3.shape[0] > 1:
                plot_water_content_xy_timelapse(
                    theta_calc=theta3,
                    utm_x=utm_x, utm_y=utm_y,
                    survey_labels=survey_labels,
                    theta_vmin=0.0, theta_vmax=vm,
                    output_dir=stage_a_dir, scenario_name=tag, dpi=dpi,
                    probe_utm_x=_probe_utm_x, probe_utm_y=_probe_utm_y,
                )
            else:
                plot_water_content_xy(
                    theta_calc=theta3[0],
                    utm_x=utm_x, utm_y=utm_y,
                    theta_vmin=0.0, theta_vmax=vm,
                    title=f"SWC Stage A — realisation {ens_idx} XY  [{scenario_name}]",
                    output_path=os.path.join(stage_a_dir, f"water_content_xy_{tag}.png"),
                    probe_utm_x=_probe_utm_x, probe_utm_y=_probe_utm_y,
                    dpi=dpi,
                )

    # ------------------------------------------------------------------
    # STAGE B — averaged models at K = 1, K_opt, K_max
    # ------------------------------------------------------------------
    k_indices_to_plot = sorted(set([0, K_opt - 1, K_max - 1]))
    print(f"\n[SWC STAGES] Stage B — averaged models at K = {[k+1 for k in k_indices_to_plot]}")
    stage_b_dir = os.path.join(stages_dir, "B_averaged_models")
    os.makedirs(stage_b_dir, exist_ok=True)

    for k_idx in k_indices_to_plot:
        K = k_idx + 1
        ec_avg = np.asarray(averaged_models[k_idx])     # ([n_time,] n_pts, n_layers)
        theta  = _ec_to_theta_grid(ec_avg, best_fit) * 100.0
        theta3 = _ensure_3d(theta)
        cv3    = _zero_cv(theta3)
        vm     = _vmax(theta3)
        tag    = f"{scenario_name}_K{K:03d}"
        label  = f"K={K}" + (" (optimal)" if K == K_opt else "") + (" (all)" if K == K_max else "")

        if timelapse and theta3.shape[0] > 1:
            plot_water_content_timelapse(
                theta_calc=theta3, cv_map=cv3,
                x_coords=x_transect, depths0=depths0,
                probe_x=probe_x_coords, probe_depths=probe_depths,
                survey_labels=survey_labels,
                theta_vmin=0.0, theta_vmax=vm,
                cv_threshold=cv_threshold,
                output_dir=stage_b_dir, scenario_name=tag, dpi=dpi,
            )
            plot_water_content_paired(
                theta_calc=theta3, cv_map=cv3,
                x_coords=x_transect, depths0=depths0,
                probe_x=probe_x_coords, probe_depths=probe_depths,
                survey_labels=survey_labels,
                theta_vmin=0.0, theta_vmax=vm,
                cv_threshold=cv_threshold,
                output_dir=stage_b_dir, scenario_name=tag, dpi=dpi,
            )
        else:
            tc = theta3[0]
            cv = cv3[0]
            plot_water_content_section(
                theta_calc=tc, cv_map=cv,
                x_coords=x_transect, depths0=depths0,
                probe_x=probe_x_coords, probe_depths=probe_depths,
                theta_vmin=0.0, theta_vmax=vm,
                cv_threshold=cv_threshold,
                title=f"SWC Stage B — {label}  [{scenario_name}]",
                output_path=os.path.join(stage_b_dir, f"water_content_xz_{tag}.png"),
                dpi=dpi,
            )

        if has_utm:
            if timelapse and theta3.shape[0] > 1:
                plot_water_content_xy_timelapse(
                    theta_calc=theta3,
                    utm_x=utm_x, utm_y=utm_y,
                    survey_labels=survey_labels,
                    theta_vmin=0.0, theta_vmax=vm,
                    output_dir=stage_b_dir, scenario_name=tag, dpi=dpi,
                    probe_utm_x=_probe_utm_x, probe_utm_y=_probe_utm_y,
                )
            else:
                plot_water_content_xy(
                    theta_calc=theta3[0],
                    utm_x=utm_x, utm_y=utm_y,
                    theta_vmin=0.0, theta_vmax=vm,
                    title=f"SWC Stage B — {label} XY  [{scenario_name}]",
                    output_path=os.path.join(stage_b_dir, f"water_content_xy_{tag}.png"),
                    probe_utm_x=_probe_utm_x, probe_utm_y=_probe_utm_y,
                    dpi=dpi,
                )

    # ------------------------------------------------------------------
    # STAGE C — optimal model  (theta_calc already in selected_output)
    # ------------------------------------------------------------------
    print(f"\n[SWC STAGES] Stage C — optimal model  (K={K_opt}, RMSE={selected_output.rmse:.5f})")
    stage_c_dir = os.path.join(stages_dir, "C_optimal_model")
    os.makedirs(stage_c_dir, exist_ok=True)

    plot_from_pipeline(
        selected_output=selected_output,
        x_transect=x_transect,
        depths0=depths0,
        probe_x_coords=probe_x_coords or [],
        probe_y_coords=probe_y_coords,
        probe_depths=probe_depths or [],
        output_dir=stage_c_dir,
        scenario_name=scenario_name,
        timelapse=timelapse,
        survey_labels=survey_labels,
        utm_x=utm_x,
        utm_y=utm_y,
    )

    # ------------------------------------------------------------------
    # 1:1 validation scatter — Stages A / B / C vs. TDR truth
    # ------------------------------------------------------------------
    # ``best_fit`` already carries .observed_theta (the TDR truth used for
    # Step-4 recalibration).  We recover sample_positions / sample_times
    # from the per-horizon observed arrays stored inside best_fit so the
    # caller does not need to pass extra arguments.
    #
    # NOTE: plot_one_to_one_stages needs the raw theta_obs vector and the
    # probe routing arrays.  These are embedded in best_fit.horizon_fits as
    # .observed_theta / .predicted_theta.  We reconstruct theta_obs by
    # concatenating across horizons (same order as calibrate used).
    _theta_obs_concat = best_fit.observed_theta  # shape (n_probes,)

    # Reconstruct minimal sample_positions / sample_times from selected_output
    # if they are stored there, otherwise build a simple sequential fallback
    # compatible with the shapes already in selected_output.theta_calc.
    if hasattr(selected_output, "sample_positions"):
        _sp = list(selected_output.sample_positions)
        _st = list(selected_output.sample_times)
    else:
        # Fallback: one probe per point in theta_obs, time=0
        n_p = len(_theta_obs_concat)
        n_pts = models_clean.shape[-2]   # works for 3-D and 4-D
        _sp = [int(round(i * (n_pts - 1) / max(n_p - 1, 1))) for i in range(n_p)]
        _st = [0] * n_p

    plot_one_to_one_stages(
        models_clean      = models_clean,
        averaged_models   = averaged_models,
        selected_output   = selected_output,
        best_fit          = best_fit,
        theta_obs         = _theta_obs_concat,
        sample_positions  = _sp,
        sample_times      = _st,
        output_dir        = output_dir,
        scenario_name     = scenario_name,
        dpi               = dpi,
    )

    print(f"\n[SWC STAGES] All stage plots written to '{stages_dir}'")
    print(f"  A_raw_realisations/  — {len(sample_ens)} sampled realisations")
    print(f"  B_averaged_models/   — K = {[k+1 for k in k_indices_to_plot]}")
    print(f"  C_optimal_model/     — K_opt = {K_opt}")
    print(f"  one_to_one_ABC_{scenario_name}.png  — 1:1 scatter (stages A/B/C)")


# ---------------------------------------------------------------------------
# Public: 1:1 scatter validation — Stages A, B, C vs. TDR truth
# ---------------------------------------------------------------------------

def plot_one_to_one_stages(
    models_clean: np.ndarray,
    averaged_models: list,
    selected_output,
    best_fit,
    theta_obs: np.ndarray,
    sample_positions: list[int],
    sample_times: list[int],
    output_dir: str,
    scenario_name: str = "default",
    probe_layer_indices: Sequence[int] | None = None,
    dpi: int = 180,
) -> None:
    """
    Produce a 3-panel 1:1 scatter plot comparing predicted θ against TDR truth
    for the three main pipeline stages:

    Panel A — Raw ensemble realisations (``models_clean``)
        All ensemble members are converted to θ at the TDR probe locations
        and plotted together.  Point colour encodes ensemble-member index so
        the spread across realisations is visible.  The cloud centroid is
        highlighted with a larger black marker.

    Panel B — Averaged model at optimal K (``averaged_models[K_opt-1]``)
        The RMSE-weighted average EC at the optimal stack depth is converted
        to θ at probe locations and compared with ``theta_obs``.

    Panel C — Final optimal model (``selected_output``)
        Uses ``selected_output.theta_calc`` (already converted inside Step 4/5)
        sampled at the probe locations.

    In every panel the diagonal 1:1 line, RMSE, bias, and R² statistics are
    shown so the improvement from A → B → C can be read at a glance.

    Parameters
    ----------
    models_clean : ndarray, shape (ens, [n_time,] n_pts, n_layers)
        Raw inverted EC models [mS/m] from Step 1.
    averaged_models : list of ndarray
        RMSE-weighted cumulative averages from Step 3, one per stack depth K.
        Each element has shape ([n_time,] n_pts, n_layers).
    selected_output : SelectionOutput
        Output of Step 4/5.  Must have ``.K``, ``.theta_calc``.
    best_fit : PetroFit
        The Step-4 recalibrated fit used for EC→θ in stages A and B.
    theta_obs : ndarray, shape (n_probes,)
        Observed TDR volumetric water content [m³/m³].
    sample_positions : list of int
        Spatial index into the n_pts axis for each TDR probe (same order as
        ``theta_obs``).
    sample_times : list of int
        Time-step index for each TDR probe observation (all-zeros for single
        surveys).
    output_dir : str
        Root pipeline output directory.  The figure is saved under
        ``<output_dir>/swc_stages/one_to_one_ABC_<scenario_name>.png``.
    scenario_name : str
        Used in the filename and figure suptitle.
    probe_layer_indices : list of int, optional
        0-based horizon index for each TDR probe, same order as
        ``sample_positions`` / ``theta_obs``.  When supplied each EC value is
        converted via the matching horizon's Waxman-Smits parameters.  When
        ``None``, horizon 0 parameters are used for all probes (safe fallback).
    dpi : int
        PNG resolution (default 180).

    Output
    ------
    Saves one PNG:
        ``<output_dir>/swc_stages/one_to_one_ABC_<scenario_name>.png``

    Also prints the per-stage RMSE / bias / R² to stdout.
    """
    import em_methodology as EMmethod

    os.makedirs(os.path.join(output_dir, "swc_stages"), exist_ok=True)
    out_path = os.path.join(
        output_dir, "swc_stages",
        f"one_to_one_ABC_{scenario_name}.png",
    )

    timelapse = models_clean.ndim == 4   # (ens, n_time, n_pts, n_layers)
    n_probes  = len(theta_obs)
    K_opt     = selected_output.K

    # ------------------------------------------------------------------
    # Helper: extract θ_pred [m³/m³] at probe locations from an EC array.
    # ec_arr shape: ([n_time,] n_pts, n_layers)
    # ------------------------------------------------------------------
    def _probe_theta(ec_arr: np.ndarray) -> np.ndarray:
        """Return predicted θ at each TDR probe location [m³/m³]."""
        ec_arr = np.asarray(ec_arr, dtype=float)
        ec_at_probes = np.empty(n_probes, dtype=float)
        for i, pos in enumerate(sample_positions):
            t_idx = sample_times[i]
            if ec_arr.ndim == 3:          # (n_time, n_pts, n_layers)
                ec_profile = ec_arr[t_idx, pos, :]
            else:                          # (n_pts, n_layers)
                ec_profile = ec_arr[pos, :]
            # Use mean EC across layers as the representative scalar, then
            # convert per-horizon if probe_layer_indices is available.
            l_idx = probe_layer_indices[i] if probe_layer_indices is not None else 0
            l_idx = int(np.clip(l_idx, 0, len(best_fit.horizon_fits) - 1))
            hf    = best_fit.horizon_fits[l_idx]
            ec_val = float(np.clip(ec_profile[l_idx], 1e-6, None))
            ratio  = hf.a * ec_val / max(hf.sigma_w + hf.sigma_s, 1e-12)
            ec_at_probes[i] = float(np.power(max(ratio, 0.0), 1.0 / hf.m))
        return ec_at_probes

    # ------------------------------------------------------------------
    # Stage A: all ensemble members at probe locations
    # ------------------------------------------------------------------
    ensemble_size = models_clean.shape[0]
    theta_pred_A_all = np.empty((ensemble_size, n_probes), dtype=float)
    for ens_idx in range(ensemble_size):
        ec_member = models_clean[ens_idx]  # ([n_time,] n_pts, n_layers)
        theta_pred_A_all[ens_idx] = _probe_theta(ec_member)

    # Flatten all ensemble members against repeated theta_obs for scatter
    theta_obs_A  = np.tile(theta_obs, ensemble_size)
    theta_pred_A = theta_pred_A_all.ravel()
    ens_labels   = np.repeat(np.arange(ensemble_size), n_probes)

    # ------------------------------------------------------------------
    # Stage B: averaged model at K_opt
    # ------------------------------------------------------------------
    ec_avg_kopt  = np.asarray(averaged_models[K_opt - 1], dtype=float)
    theta_pred_B = _probe_theta(ec_avg_kopt)

    # ------------------------------------------------------------------
    # Stage C: optimal model — sample theta_calc at probe locations
    # theta_calc shape: (n_time, n_pts, n_layers) after Step 4/5
    # ------------------------------------------------------------------
    theta_calc = np.asarray(selected_output.theta_calc, dtype=float)
    # Ensure 3-D: (n_time, n_pts, n_layers)
    if theta_calc.ndim == 2:
        theta_calc = theta_calc[np.newaxis, ...]

    theta_pred_C = np.empty(n_probes, dtype=float)
    for i, pos in enumerate(sample_positions):
        t_idx = sample_times[i]
        l_idx = probe_layer_indices[i] if probe_layer_indices is not None else 0
        l_idx = int(np.clip(l_idx, 0, theta_calc.shape[-1] - 1))
        theta_pred_C[i] = float(theta_calc[t_idx, pos, l_idx])

    # ------------------------------------------------------------------
    # Scalar metrics helper
    # ------------------------------------------------------------------
    def _metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
        """Return RMSE, bias, and R² for a flat pair of arrays."""
        residuals = pred - obs
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        bias = float(np.mean(residuals))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((obs - obs.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")
        return dict(rmse=rmse, bias=bias, r2=r2)

    m_a = _metrics(theta_obs_A,  theta_pred_A)
    m_b = _metrics(theta_obs,    theta_pred_B)
    m_c = _metrics(theta_obs,    theta_pred_C)

    for stage, m in zip(("A", "B", "C"), (m_a, m_b, m_c)):
        print(
            f"[1:1 Stage {stage}]  RMSE={m['rmse']:.4f}  "
            f"bias={m['bias']:+.4f}  R²={m['r2']:.4f}"
        )

    # ------------------------------------------------------------------
    # Figure layout: 1 row × 3 panels, shared axis range
    # ------------------------------------------------------------------
    all_vals = np.concatenate([theta_obs_A, theta_pred_A,
                                theta_obs,   theta_pred_B,
                                theta_obs,   theta_pred_C])
    global_min = max(0.0, float(np.nanmin(all_vals)) * 0.95)
    global_max = float(np.nanmax(all_vals)) * 1.05

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)
    fig.suptitle(
        f"1:1 Predicted vs. Observed TDR θ — Stages A / B / C\n"
        f"Scenario: {scenario_name}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    STAGE_COLORS = ("steelblue", "darkorange", "mediumseagreen")
    STAGE_LABELS = ("A — Raw realisations", "B — Averaged (K_opt)", "C — Optimal model")

    # ---- Panel A -------------------------------------------------------
    ax = axes[0]
    sc = ax.scatter(
        theta_obs_A, theta_pred_A,
        c=ens_labels, cmap="viridis", alpha=0.4,
        s=20, linewidths=0.3, edgecolors="none",
        label=f"All {ensemble_size} realisations",
    )
    # Centroid across ensemble members
    ax.scatter(
        theta_obs, theta_pred_A_all.mean(axis=0),
        color="black", s=60, zorder=5, marker="D",
        edgecolors="white", linewidths=0.8,
        label="Ensemble mean at probes",
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Ensemble index", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.plot([global_min, global_max], [global_min, global_max],
            "r--", linewidth=1.5, label="1:1 line")
    ax.set_title(
        f"{STAGE_LABELS[0]}\n"
        f"RMSE={m_a['rmse']:.4f}  bias={m_a['bias']:+.4f}  R²={m_a['r2']:.3f}",
        fontsize=9,
    )

    # ---- Panel B -------------------------------------------------------
    ax = axes[1]
    ax.scatter(
        theta_obs, theta_pred_B,
        color=STAGE_COLORS[1], edgecolors="black",
        linewidths=0.5, s=70, zorder=4,
    )
    ax.plot([global_min, global_max], [global_min, global_max],
            "r--", linewidth=1.5, label="1:1 line")
    ax.set_title(
        f"{STAGE_LABELS[1]}\n"
        f"RMSE={m_b['rmse']:.4f}  bias={m_b['bias']:+.4f}  R²={m_b['r2']:.3f}",
        fontsize=9,
    )

    # ---- Panel C -------------------------------------------------------
    ax = axes[2]
    ax.scatter(
        theta_obs, theta_pred_C,
        color=STAGE_COLORS[2], edgecolors="black",
        linewidths=0.5, s=70, zorder=4,
    )
    ax.plot([global_min, global_max], [global_min, global_max],
            "r--", linewidth=1.5, label="1:1 line")
    ax.set_title(
        f"{STAGE_LABELS[2]}\n"
        f"RMSE={m_c['rmse']:.4f}  bias={m_c['bias']:+.4f}  R²={m_c['r2']:.3f}",
        fontsize=9,
    )

    # ---- Shared aesthetics for panels B and C -------------------------
    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"Observed TDR $\theta$ [m³/m³]", fontsize=10)
        ax.set_ylabel(r"Predicted $\theta$ [m³/m³]", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[1:1 PLOT] 3-panel 1:1 figure saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI — quick test with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Quick test: plot synthetic water content section."
    )
    parser.add_argument("--out", default=".", help="Output directory")
    parser.add_argument("--timelapse", action="store_true",
                        help="Generate multi-panel time-lapse figure")
    parser.add_argument(
        "--mode", choices=["1d", "2d"], default="1d",
        help=(
            "'1d' — XZ cross-section only (default). "
            "'2d' — XZ + XY plan-view using synthetic UTM coords."
        ),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(0)

    depths0  = [0.3, 0.6, 1.0]
    n_layers = len(depths0) + 1
    n_x      = 30
    x_coords = np.linspace(0, 25.2, n_x)

    base  = np.array([32, 25, 18, 10], dtype=float)
    theta = base[None, :] + rng.normal(0, 2, (n_x, n_layers))
    theta = np.clip(theta, 0, 45)
    cv       = rng.uniform(20, 120, (n_x, n_layers))
    coverage = rng.uniform(-0.5, 2.0, (n_x, n_layers))

    # Synthetic UTM grid anchored to the Alento field survey origin
    ORIGIN_X = 515487.23   # UTM easting  SW corner of Alento survey
    ORIGIN_Y = 4468201.66  # UTM northing SW corner of Alento survey
    utm_x_test = ORIGIN_X + x_coords
    utm_y_test = ORIGIN_Y + rng.uniform(0, 50, n_x)

    if args.timelapse:
        n_t    = 6
        labels = [f"EM{i+1}" for i in range(n_t)]
        th_tl  = theta[None] * np.linspace(1.0, 0.6, n_t)[:, None, None]
        cv_tl  = cv[None]    * np.ones((n_t, 1, 1))
        cov_tl = coverage[None] * np.ones((n_t, 1, 1))

        plot_water_content_timelapse(
            theta_calc=th_tl, cv_map=cv_tl,
            x_coords=x_coords, depths0=depths0,
            coverage=cov_tl,
            probe_x=[5.0, 10.0, 15.0, 20.0],
            probe_depths=[0.15, 0.45, 0.8, 1.25],
            survey_labels=labels,
            output_dir=args.out, scenario_name="test_xz_timelapse",
        )
        if args.mode == "2d":
            plot_water_content_xy_timelapse(
                theta_calc=th_tl,
                utm_x=utm_x_test, utm_y=utm_y_test,
                survey_labels=labels,
                output_dir=args.out, scenario_name="test_xy_timelapse",
            )
    else:
        plot_water_content_section(
            theta_calc=theta, cv_map=cv,
            x_coords=x_coords, depths0=depths0,
            coverage=coverage,
            probe_x=[5.0, 10.0, 15.0, 20.0],
            probe_depths=[0.15, 0.45, 0.8, 1.25],
            title="Test — Soil Water Content (XZ)",
            output_path=os.path.join(args.out, "water_content_xz_test.png"),
        )
        if args.mode == "2d":
            plot_water_content_xy(
                theta_calc=theta,
                utm_x=utm_x_test, utm_y=utm_y_test,
                title="Test — Soil Water Content (XY Plan View)",
                output_path=os.path.join(args.out, "water_content_xy_test.png"),
            )
