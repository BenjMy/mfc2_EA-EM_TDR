#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_em_diagnostics_real.py
===========================
Diagnostic figures for the real-data branch of the EA-EM pipeline.

Figures produced
----------------
common_points_footprint.png
    One panel per survey epoch (up to 8 shown).  Grey = all measurement
    points in that epoch; coloured = the ``min_pts`` rows retained by the
    pipeline's Step 1 row-count trim (the trim that ``models_clean`` still
    needs for its dense-array ensemble maths).  This is the ONLY figure
    that still visualises that ensemble-side trim — it's shown here purely
    for reference/debugging, it is not applied to any other figure below.
    TDR probe locations are overlaid as black crosses when ``probe_table``
    is supplied.

eca_map_epoch_NN_<stem>.png
    Plan-view scatter of apparent quadrature ECa for every available
    frequency, using the FULL raw per-epoch point set — NOT trimmed to
    ``min_pts``.  Every point the GEM-2 logger recorded that day is shown.
    One figure per survey epoch.

eca_map_emagpy_epoch_NN_<stem>.png
    Plan-view ECa map for every coil, rendered with emagpy's OWN ``show()``
    plotting module after dropping repeated/duplicate points with emagpy's
    ``filterRepeated()``.  Independent of the manual-matplotlib figure
    above: here emagpy owns both the duplicate-point filtering and the
    rendering, so it's a cross-check that nothing in the raw point
    structure looks wrong.  Also untrimmed relative to ``min_pts``.  One
    figure per survey epoch.

eca_map_temporal_median.png
    Temporal median ECa across all epochs for each frequency — the stable
    spatial pattern independent of moisture dynamics.  Because different
    epochs can have different point counts/orderings, this figure is
    necessarily a cross-epoch comparison: it uses the epoch with the most
    raw points as the spatial reference grid, and nearest-neighbour-matches
    every other epoch onto that grid (within ``match_tolerance_m``),
    leaving unmatched reference points as NaN rather than silently
    dropping or misaligning them.  This is the only figure that performs
    any cross-epoch point correspondence; every other figure is fully
    per-epoch and untrimmed.  TDR probe locations are overlaid as black
    crosses when ``probe_table`` is supplied.

Public API
----------
    plot_real_em_diagnostics(raw_em_files, processed_em_files,
                             min_pts, probe_table, output_dir, em_unit)

Called from run_pipeline.main() immediately after _load_real_data() returns
and survey_df has been trimmed, so both the per-epoch raw footprints and
the authoritative min_pts are available.  A try/except wrapper in the
pipeline prevents a plot failure from aborting the run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import geometry_utils


# ---------------------------------------------------------------------------
# Column aliases — kept in sync with _COLUMN_VARIANTS in run_pipeline.py
# ---------------------------------------------------------------------------

_COLUMN_VARIANTS: dict[str, list[str]] = {
    "x":                    ["X", "Longitude", "lon", "LON", "LONGITUDE", "Easting"],
    "y":                    ["Y", "Latitude",  "lat", "LAT", "LATITUDE",  "Northing"],
    "elevation":            ["GPSalt", "Altitude", "Alt", "ALT", "altitude", "ALTITUDE", "Height"],
    "HCP1.66f425h1_quad":   ["Q_425Hz",   "Q425Hz",   "Q_425",   "Q425"],
    "HCP1.66f1525h1_quad":  ["Q_1525Hz",  "Q1525Hz",  "Q_1525",  "Q1525"],
    "HCP1.66f5325h1_quad":  ["Q_5325Hz",  "Q5325Hz",  "Q_5325",  "Q5325"],
    "HCP1.66f18325h1_quad": ["Q_18325Hz", "Q18325Hz", "Q_18325", "Q18325"],
    "HCP1.66f63025h1_quad": ["Q_63025Hz", "Q63025Hz", "Q_63025", "Q63025"],
    "HCP1.66f92775h1_quad": ["Q_92775Hz", "Q92775Hz", "Q_92775", "Q92775"],
}

_FREQ_LABELS: dict[str, str] = {
    "HCP1.66f425h1_quad":   "425 Hz",
    "HCP1.66f1525h1_quad":  "1 525 Hz",
    "HCP1.66f5325h1_quad":  "5 325 Hz",
    "HCP1.66f18325h1_quad": "18 325 Hz",
    "HCP1.66f63025h1_quad": "63 025 Hz",
    "HCP1.66f92775h1_quad": "92 775 Hz",
}

# Colormap used across all ECa maps — matches viridis used in plot_water_content
_CMAP_ECA    = "viridis"
_CMAP_MEDIAN = "plasma"
_DPI         = 200


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_raw_csv(path: str) -> pd.DataFrame:
    """
    Load one raw GEM-2 CSV and rename columns to canonical emagpy names.
    Mirrors _prepare_real_em_csv() in run_pipeline.py but returns a DataFrame
    instead of writing a file, so we can work directly with raw coordinates.

    FIX (TDR<->EM CRS mismatch): like run_pipeline.py's own
    _prepare_real_em_csv, the x/y columns here may resolve via either a
    metric alias (X/Easting, Y/Northing) or a geographic alias
    (Longitude/lon, Latitude/lat). Every figure in this module overlays TDR
    probe locations (``probe_table['utm_x'/'utm_y']``, always metric UTM)
    directly onto these coordinates (``_overlay_probes``) -- if left in
    decimal degrees, that overlay would be meaningless (probes plotting
    nowhere near the EM footprint, or the whole EM point cloud collapsing
    to a speck at plot scale). This now classifies the matched alias
    (``geometry_utils.classify_coordinate_alias``) and reprojects geographic
    coordinates to metric UTM (``geometry_utils.reproject_lonlat_to_utm``,
    auto-detecting the UTM zone from this file's own mean coordinates)
    before returning, so every plot built from this DataFrame is guaranteed
    metric.
    """
    with open(path, "r", errors="replace") as fh:
        first = next((ln.strip() for ln in fh if ln.strip()), "")
    delim = ";" if first.count(";") >= first.count(",") else ","

    df = pd.read_csv(path, delimiter=delim)
    df.columns = df.columns.str.strip()
    available = set(df.columns)

    rename: dict[str, str] = {}
    for canon, aliases in _COLUMN_VARIANTS.items():
        hit = next((a for a in aliases if a in available), None)
        if hit:
            rename[hit] = canon

    # x and y are mandatory; freq cols are optional (some epochs may miss freqs)
    missing_xy = [c for c in ("x", "y") if c not in rename.values() and
                  not any(a in available for a in _COLUMN_VARIANTS[c])]
    if missing_xy:
        raise ValueError(
            f"Cannot parse '{os.path.basename(path)}': "
            f"no recognised alias for {missing_xy} "
            f"(found: {sorted(available)})"
        )

    keep = list(rename.keys())
    out_df = df[keep].rename(columns=rename).reset_index(drop=True)

    if "x" in out_df.columns and "y" in out_df.columns:
        import geometry_utils
        x_raw = next(raw for raw, canon in rename.items() if canon == "x")
        y_raw = next(raw for raw, canon in rename.items() if canon == "y")
        x_kind = geometry_utils.classify_coordinate_alias(x_raw, "x")
        y_kind = geometry_utils.classify_coordinate_alias(y_raw, "y")
        if x_kind != y_kind or x_kind == "unknown":
            raise ValueError(
                f"Cannot parse '{os.path.basename(path)}': x resolved via '{x_raw}' "
                f"('{x_kind}') but y resolved via '{y_raw}' ('{y_kind}') -- "
                f"inconsistent/unrecognised coordinate columns."
            )
        if x_kind == "geographic":
            x_m, y_m, epsg_used = geometry_utils.reproject_lonlat_to_utm(
                out_df["x"].to_numpy(dtype=float), out_df["y"].to_numpy(dtype=float),
            )
            print(f"[DIAG] '{os.path.basename(path)}': geographic x/y ('{x_raw}'/'{y_raw}') "
                  f"reprojected to metric UTM ({epsg_used}, auto-detected).")
            out_df["x"] = x_m
            out_df["y"] = y_m

    return out_df


def _freq_cols_in(df: pd.DataFrame) -> list[str]:
    """Return ordered list of frequency columns present in df."""
    return [c for c in _COLUMN_VARIANTS if c.startswith("HCP") and c in df.columns]


def _robust_clim(vals: np.ndarray) -> tuple[float, float]:
    """2nd–98th percentile colour limits, with a guard against flat data."""
    vlo = float(np.nanpercentile(vals, 2))
    vhi = float(np.nanpercentile(vals, 98))
    if vlo == vhi:
        vlo -= 1.0
        vhi += 1.0
    return vlo, vhi


def _unit_label(dfs: list[pd.DataFrame], freq_cols: list[str],
                em_unit: str) -> str:
    """
    Return the display unit string.  When em_unit is 'auto' (shouldn't happen
    in normal pipeline use, but defensive), infer from data magnitude.
    """
    if em_unit in ("ppm", "ppt"):
        return em_unit.upper()
    # Fallback: infer from median absolute value
    sample = dfs[0][freq_cols].to_numpy(float)
    return "PPM" if np.nanmedian(np.abs(sample)) > 100 else "PPT"


def _match_epochs_to_reference(
    dfs: list[pd.DataFrame],
    freq_cols: list[str],
    tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Build a common spatial reference grid for cross-epoch comparison
    WITHOUT trimming any epoch's own point set.

    The epoch with the most raw points is used as the reference grid
    (x_ref, y_ref) — using the densest survey means we lose as little
    spatial resolution as possible. Every other epoch is nearest-neighbour
    matched onto that grid: for each reference point, the closest raw
    point in that epoch is used if it falls within ``tolerance_m``,
    otherwise the value is NaN for that epoch/point. No epoch's raw file
    is ever subsetted by row position — matching is purely by coordinate
    distance, and a coordinate with no near neighbour in a given epoch
    simply contributes NaN to that epoch's slice rather than forcing
    every epoch down to whatever the sparsest epoch covers.

    Returns
    -------
    x_ref, y_ref : np.ndarray
        Reference-grid coordinates (from the densest epoch).
    matched : list of np.ndarray
        One (n_ref, n_freq) array per epoch, NaN where no match was found
        within tolerance.
    """
    ref_idx = int(np.argmax([len(df) for df in dfs]))
    x_ref = dfs[ref_idx]["x"].to_numpy(float)
    y_ref = dfs[ref_idx]["y"].to_numpy(float)
    ref_xy = np.column_stack([x_ref, y_ref])

    matched: list[np.ndarray] = []
    for df in dfs:
        xy = df[["x", "y"]].to_numpy(float)
        vals = df[freq_cols].to_numpy(float)

        # Brute-force nearest neighbour (point counts here are small —
        # tens to low hundreds — so an O(n_ref * n_epoch) distance matrix
        # is cheap and avoids adding a scipy dependency just for this).
        d2 = ((ref_xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
        nearest = d2.argmin(axis=1)
        nearest_dist = np.sqrt(d2[np.arange(len(ref_xy)), nearest])

        epoch_vals = vals[nearest]
        epoch_vals[nearest_dist > tolerance_m] = np.nan
        matched.append(epoch_vals)

    return x_ref, y_ref, matched


def _overlay_probes(ax: plt.Axes, probe_table: pd.DataFrame | None) -> None:
    """Draw TDR probe locations as black crosses on an existing axes."""
    if probe_table is None or probe_table.empty:
        return
    # probe_table columns: probe_index, station_name, utm_x, utm_y
    # These are UTM coordinates — same CRS as the processed EM survey x/y
    if "utm_x" not in probe_table.columns or "utm_y" not in probe_table.columns:
        return
    px = probe_table["utm_x"].to_numpy(float)
    py = probe_table["utm_y"].to_numpy(float)
    # Filter out (0, 0) placeholders that arise from missing coordinates
    valid = (px != 0.0) | (py != 0.0)
    if valid.any():
        ax.scatter(px[valid], py[valid], marker="+", s=60, linewidths=1.2,
                   color="black", zorder=5, label="TDR probes")


# ---------------------------------------------------------------------------
# Figure 1 — Common-point footprint
# ---------------------------------------------------------------------------

def _plot_common_points(
    raw_dfs: list[pd.DataFrame],
    names: list[str],
    tolerance_m: float,
    probe_table: pd.DataFrame | None,
    out_dir: str,
) -> str:
    """
    One panel per epoch (max 8 shown).
    Grey = full epoch footprint; colour = this epoch's own points that
    fall within ``tolerance_m`` of a reference-grid cell EVERY epoch also
    covers -- i.e. true spatial commonality across all ``n_surveys``
    epochs, built with ``geometry_utils.match_epochs_to_reference_grid``,
    the SAME coordinate-based matcher ``run_pipeline.py`` Step 1 now uses
    to assemble ``models_clean``/``depths_clean`` ("spatial reference-grid
    matching, not row trimming").

    FIX ("common ≠ same row index"): this used to colour ``arr[:min_pts]``
    -- the first ``min_pts`` rows of each epoch's raw CSV, purely by row
    POSITION. Real GEM-2 epochs can differ both in point COUNT and in
    walk ORDER (the rover need not retrace an identical path/sampling
    rate each campaign), so "row i of epoch A" and "row i of epoch B" are
    only the same physical location by coincidence -- with epochs of
    different density this doesn't even hold loosely, since row index
    isn't even a consistent fraction of distance travelled between
    files. That trim is still used ONLY for the ensemble's dense-array
    packing upstream (unrelated to this plot); it is no longer used here.
    Black crosses = TDR probe locations (if probe_table supplied).
    """
    n_surveys = len(raw_dfs)
    n_show    = min(n_surveys, 8)
    n_cols    = min(n_show, 4)
    n_rows    = int(np.ceil(n_show / n_cols))

    colors = cm.tab10(np.linspace(0, 0.9, n_surveys))

    # Union spatial reference grid + per-epoch matches, exactly like
    # run_pipeline.py Step 1's _remap_combo_to_reference machinery.
    epoch_xy = [
        (df["x"].to_numpy(float), df["y"].to_numpy(float)) for df in raw_dfs
    ]
    ref_x, ref_y, matched_indices = geometry_utils.match_epochs_to_reference_grid(
        epoch_xy, tolerance_m=tolerance_m,
    )
    n_ref = len(ref_x)

    # A reference cell only counts as "common" if EVERY epoch has a point
    # within tolerance_m of it (matched_indices[e][r] >= 0 for all e).
    coverage_count = np.zeros(n_ref, dtype=int)
    for idx_map in matched_indices:
        coverage_count += (idx_map >= 0).astype(int)
    common_ref_mask = coverage_count == n_surveys
    n_common = int(common_ref_mask.sum())

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4.2 * n_rows),
        squeeze=False,
    )

    for panel_idx in range(n_rows * n_cols):
        row, col = divmod(panel_idx, n_cols)
        ax = axes[row][col]

        if panel_idx >= n_show:
            ax.set_visible(False)
            continue

        df   = raw_dfs[panel_idx]
        x    = df["x"].to_numpy(float)
        y    = df["y"].to_numpy(float)
        n_pt = len(df)

        # This epoch's own rows that map to a common reference cell.
        idx_map = matched_indices[panel_idx]
        common_rows = idx_map[common_ref_mask]
        common_rows = common_rows[common_rows >= 0]
        n_common_this_epoch = len(common_rows)

        # Full footprint
        ax.scatter(x, y, s=2, color="lightgrey", zorder=1, rasterized=True,
                   label=f"all ({n_pt})")
        # Spatially common rows
        if n_common_this_epoch:
            ax.scatter(x[common_rows], y[common_rows], s=3, color=colors[panel_idx],
                       zorder=2, rasterized=True,
                       label=f"common ({n_common_this_epoch})")
        # TDR probes
        _overlay_probes(ax, probe_table)

        ax.set_title(
            f"{names[panel_idx]}\n{n_pt} pts total → {n_common_this_epoch} spatially common",
            fontsize=8,
        )
        ax.set_xlabel("X / Easting", fontsize=7)
        ax.set_ylabel("Y / Northing", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", "datalim")
        ax.legend(fontsize=6, markerscale=2, loc="upper right")

    surplus = n_surveys - n_show
    subtitle = ""
    if surplus > 0:
        subtitle = f"  (first {n_show} of {n_surveys} epochs shown)"

    fig.suptitle(
        f"Common-point footprint — {n_surveys} GEM-2 survey epochs\n"
        f"Spatial reference-grid match (tolerance={tolerance_m} m): "
        f"{n_common}/{n_ref} reference cell(s) covered by every epoch{subtitle}",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(out_dir, "common_points_footprint.png")
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[DIAG] Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 2 — Per-epoch ECa plan-view maps
# ---------------------------------------------------------------------------

def _plot_eca_epoch(
    df_full: pd.DataFrame,
    epoch_idx: int,
    name: str,
    freq_cols: list[str],
    unit_lbl: str,
    probe_table: pd.DataFrame | None,
    out_dir: str,
) -> str:
    """
    One figure per epoch with one subplot per frequency.

    ``df_full`` is the epoch's COMPLETE raw point set — this function no
    longer trims to ``min_pts``. Every point the logger recorded that day
    is plotted.
    """
    n_freqs = len(freq_cols)
    x = df_full["x"].to_numpy(float)
    y = df_full["y"].to_numpy(float)

    fig, axes = plt.subplots(
        1, n_freqs,
        figsize=(4.2 * n_freqs, 4.5),
        squeeze=False,
    )

    for fi, col in enumerate(freq_cols):
        ax   = axes[0][fi]
        vals = df_full[col].to_numpy(float)
        vlo, vhi = _robust_clim(vals)

        sc = ax.scatter(x, y, c=vals, cmap=_CMAP_ECA, s=4,
                        vmin=vlo, vmax=vhi, rasterized=True)
        plt.colorbar(sc, ax=ax, label=f"ECa [{unit_lbl}]",
                     fraction=0.046, pad=0.04)

        _overlay_probes(ax, probe_table)

        ax.set_title(_FREQ_LABELS.get(col, col), fontsize=9)
        ax.set_xlabel("X / Easting", fontsize=8)
        ax.set_ylabel("Y / Northing", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", "datalim")

        if fi == n_freqs - 1 and probe_table is not None and not probe_table.empty:
            ax.legend(fontsize=7, loc="upper right")

    stem = Path(name).stem
    fig.suptitle(
        f"ECa plan-view — Epoch {epoch_idx + 1}  |  {name}\n"
        f"({len(df_full)} pts, full raw footprint — no trim,  unit: {unit_lbl})",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = os.path.join(out_dir, f"eca_map_epoch_{epoch_idx + 1:02d}_{stem}.png")
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[DIAG] Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 4 — emagpy-native ECa 2D maps (show() + filterRepeated())
# ---------------------------------------------------------------------------

def _plot_eca_emagpy_show(
    raw_em_files: list[str],
    names: list[str],
    probe_table: "pd.DataFrame | None",
    out_dir: str,
    em_unit: str,
) -> list[str]:
    """
    Per-epoch ECa 2D maps rendered with emagpy's OWN ``show()`` plotting
    module, after dropping repeated/duplicate points with emagpy's
    ``filterRepeated()``.

    This is independent of and complementary to ``_plot_eca_epoch`` above
    (manual matplotlib scatter on a Step-1 row-count trim): here emagpy
    owns both the duplicate-point filtering and the rendering, so it's a
    useful cross-check that the manual ``min_pts`` trim isn't masking
    something the field data's own repeated-point structure would flag
    (e.g. the logger sitting stationary while overlapping GPS fixes pile
    up at one corner of the grid).

    ``raw_em_files`` must already be emagpy-ready (comma-separated,
    canonical coil column names) — i.e. the processed CSVs produced by
    ``_prepare_real_em_csv()`` in run_pipeline.py, not the truly-raw
    semicolon-delimited GEM-2 exports. In the current pipeline wiring,
    the parameter the caller names ``raw_em_files`` is in fact already
    set to those processed paths (see ``EM_FNAMES`` in run_pipeline.py),
    so no extra preprocessing is needed here.
    """
    from emagpy import Problem

    saved: list[str] = []

    for idx, (path, name) in enumerate(zip(raw_em_files, names)):
        try:
            k = Problem()
            k.createSurvey(path, targetProjection=None, unit=em_unit)
        except Exception as exc:
            print(f"[DIAG][WARN] emagpy could not read '{name}' as an "
                  f"emagpy-ready survey CSV ({exc}) — skipping show() map. "
                  f"Pass the processed (canonical-column) CSVs, not the "
                  f"truly-raw GEM-2 export.")
            continue

        n_before = len(k.surveys[0].df)
        try:
            k.filterRepeated(index=0)
        except Exception as exc:
            print(f"[DIAG][WARN] filterRepeated() failed for '{name}': {exc}")
        n_after = len(k.surveys[0].df)

        coils = list(getattr(k.surveys[0], "coils", []))
        if not coils:
            print(f"[DIAG][WARN] No coils detected for '{name}' — skipping emagpy show().")
            continue

        n_coils = len(coils)
        fig, axes = plt.subplots(1, n_coils, figsize=(4.2 * n_coils, 4.5), squeeze=False)

        for ci, coil in enumerate(coils):
            ax = axes[0][ci]
            try:
                k.show(index=0, coil=coil, ax=ax)
            except TypeError:
                # Older/newer emagpy signature without a `coil` kwarg —
                # fall back to its default single-coil rendering.
                try:
                    k.show(index=0, ax=ax)
                except Exception as exc:
                    ax.text(0.5, 0.5, f"show() failed:\n{exc}", ha="center",
                            va="center", transform=ax.transAxes,
                            fontsize=7, color="red")
            except Exception as exc:
                ax.text(0.5, 0.5, f"show() failed:\n{exc}", ha="center",
                        va="center", transform=ax.transAxes,
                        fontsize=7, color="red")
            ax.set_title(_FREQ_LABELS.get(coil, coil), fontsize=9)
            ax.tick_params(labelsize=7)
            ax.set_aspect("equal", "datalim")
            _overlay_probes(ax, probe_table)

        stem = Path(name).stem
        fig.suptitle(
            f"emagpy show() ECa map — Epoch {idx + 1}  |  {name}\n"
            f"filterRepeated(): {n_before} → {n_after} pts  (unit: {em_unit.upper()})",
            fontsize=10,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        out_path = os.path.join(out_dir, f"eca_map_emagpy_epoch_{idx + 1:02d}_{stem}.png")
        fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[DIAG] Saved → {out_path}")
        saved.append(out_path)

    return saved


# ---------------------------------------------------------------------------
# Figure 3 — Temporal-median ECa
# ---------------------------------------------------------------------------

def _plot_eca_temporal_median(
    raw_dfs: list[pd.DataFrame],
    freq_cols: list[str],
    unit_lbl: str,
    probe_table: pd.DataFrame | None,
    out_dir: str,
    n_epochs: int,
    match_tolerance_m: float = 2.0,
) -> str:
    """
    Median ECa across all epochs per frequency.

    Comparing epochs point-by-point inherently requires some notion of
    "the same location in every epoch" — unlike the other figures in this
    module, this one can't be made fully trim-free. Instead of the old
    row-count trim (which silently assumed row `i` in every epoch's file
    was the same physical spot), this now nearest-neighbour matches every
    epoch onto the densest epoch's own point grid (see
    ``_match_epochs_to_reference``), within ``match_tolerance_m`` metres.
    No epoch's raw points are dropped from Figures 1/2/4 because of this —
    the matching/NaN-fill happens only inside this function, on a copy.
    """
    n_freqs = len(freq_cols)
    x_ref, y_ref, matched = _match_epochs_to_reference(
        raw_dfs, freq_cols, tolerance_m=match_tolerance_m
    )

    fig, axes = plt.subplots(
        1, n_freqs,
        figsize=(4.5 * n_freqs, 4.5),
        squeeze=False,
    )

    for fi, col in enumerate(freq_cols):
        stack = np.stack(
            [epoch_vals[:, fi] for epoch_vals in matched], axis=0
        )  # shape (n_epochs, n_ref_pts), NaN where unmatched
        med_vals = np.nanmedian(stack, axis=0)
        n_covered = int(np.isfinite(med_vals).sum())

        ax   = axes[0][fi]
        vlo, vhi = _robust_clim(med_vals)

        sc = ax.scatter(x_ref, y_ref, c=med_vals, cmap=_CMAP_MEDIAN, s=4,
                        vmin=vlo, vmax=vhi, rasterized=True)
        plt.colorbar(sc, ax=ax, label=f"Median ECa [{unit_lbl}]",
                     fraction=0.046, pad=0.04)

        _overlay_probes(ax, probe_table)

        ax.set_title(
            f"{_FREQ_LABELS.get(col, col)}  ({n_covered}/{len(x_ref)} pts w/ data)",
            fontsize=9,
        )
        ax.set_xlabel("X / Easting", fontsize=8)
        ax.set_ylabel("Y / Northing", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", "datalim")

    if probe_table is not None and not probe_table.empty:
        axes[0][-1].legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Temporal-median ECa — {n_epochs} epochs  |  {len(x_ref)} reference pts "
        f"(densest epoch's own grid, ±{match_tolerance_m} m match)  [unit: {unit_lbl}]",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = os.path.join(out_dir, "eca_map_temporal_median.png")
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[DIAG] Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Public entry point — called from run_pipeline.main()
# ---------------------------------------------------------------------------

def plot_real_em_diagnostics(
    raw_em_files: list[str],
    min_pts: int,
    probe_table: "pd.DataFrame | None",
    output_dir: str,
    em_unit: str = "ppt",
) -> list[str]:
    """
    Generate common-point footprint and ECa plan-view maps for real GEM-2 data.

    Parameters
    ----------
    raw_em_files : list of str
        Paths to the *raw* (pre-processing) GEM-2 survey CSVs in chronological
        order — the same list returned by ``prepare_real_dataset()`` and stored
        in ``_load_real_data()``.  Raw files are used so the full per-epoch
        footprint (before the row-count trim) can be shown alongside the common
        points.
    min_pts : int
        The authoritative ``min_pts`` established by the Step 1 row-count trim
        in ``run_pipeline.main()``.  Passed in explicitly so this function
        doesn't have to recompute it and stays consistent with the pipeline.
    probe_table : pd.DataFrame or None
        Station table returned by ``prepare_real_dataset()``, with columns
        ``probe_index``, ``station_name``, ``utm_x``, ``utm_y``.  When not
        None, TDR probe locations are overlaid on every map as black crosses.
    output_dir : str
        Root output directory of the current run (``cfg.output_dir``).
        All diagnostic figures are written to ``<output_dir>/diagnostics/``.
    em_unit : str
        Physical unit of the EM quadrature columns: ``'ppm'`` for GEM-2 /
        CMD-Explorer, ``'ppt'`` for GF Instruments (default).  Used only for
        the colour-bar label.

    Returns
    -------
    list of str
        Paths of all figures written.  Empty when no files could be loaded.
    """
    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    print(f"\n[DIAG] Real-data EM diagnostics — loading {len(raw_em_files)} raw files…")

    # ------------------------------------------------------------------
    # Load raw CSVs (original coordinates, before preprocessing)
    # ------------------------------------------------------------------
    raw_dfs: list[pd.DataFrame] = []
    names:   list[str]          = []
    for path in raw_em_files:
        try:
            df = _parse_raw_csv(path)
            raw_dfs.append(df)
            names.append(os.path.basename(path))
            print(f"  [OK]  {os.path.basename(path):50s}  {len(df):>6d} pts")
        except Exception as exc:
            print(f"  [WARN] Skipping '{os.path.basename(path)}': {exc}")

    if not raw_dfs:
        print("[DIAG][WARN] No raw EM files could be parsed — diagnostics skipped.")
        return []

    # ------------------------------------------------------------------
    # Determine common frequency columns (intersection across all epochs)
    # ------------------------------------------------------------------
    freq_cols = [
        c for c in _COLUMN_VARIANTS
        if c.startswith("HCP") and all(c in df.columns for df in raw_dfs)
    ]
    unit_lbl = _unit_label(raw_dfs, freq_cols, em_unit)

    print(f"[DIAG] Common frequency columns : {[_FREQ_LABELS.get(c, c) for c in freq_cols]}")
    print(f"[DIAG] min_pts (Step 1 trim)    : {min_pts}")
    print(f"[DIAG] ECa unit                 : {unit_lbl}")

    saved: list[str] = []

    # ------------------------------------------------------------------
    # Figure 1 — Common-point footprint
    # ------------------------------------------------------------------
    try:
        p = _plot_common_points(raw_dfs, names, min_pts, probe_table, diag_dir)
        saved.append(p)
    except Exception as exc:
        print(f"[DIAG][WARN] common_points_footprint failed: {exc}")

    # ------------------------------------------------------------------
    # NOTE: raw_dfs is used AS-IS below — no min_pts row trim. Every point
    # each epoch's logger actually recorded is plotted in Figures 2 and 4.
    # Only Figure 3 (temporal median) performs any cross-epoch matching,
    # and it does so by coordinate (see _match_epochs_to_reference), not
    # by slicing rows off the front of each file.
    # ------------------------------------------------------------------
    if not freq_cols:
        print("[DIAG][WARN] No common frequency columns found — ECa maps skipped.")
        return saved

    # ------------------------------------------------------------------
    # Figure 2 — Per-epoch ECa maps (full raw footprint, untrimmed)
    # ------------------------------------------------------------------
    for idx, (df_full, name) in enumerate(zip(raw_dfs, names)):
        try:
            p = _plot_eca_epoch(df_full, idx, name, freq_cols, unit_lbl,
                                probe_table, diag_dir)
            saved.append(p)
        except Exception as exc:
            print(f"[DIAG][WARN] eca_map epoch {idx + 1} failed: {exc}")

    # ------------------------------------------------------------------
    # Figure 4 — emagpy-native ECa 2D maps (show() + filterRepeated())
    # ------------------------------------------------------------------
    try:
        saved.extend(
            _plot_eca_emagpy_show(raw_em_files, names, probe_table, diag_dir, em_unit)
        )
    except Exception as exc:
        print(f"[DIAG][WARN] emagpy show() ECa maps failed: {exc}")

    # ------------------------------------------------------------------
    # Figure 3 — Temporal-median ECa
    # ------------------------------------------------------------------
    try:
        p = _plot_eca_temporal_median(raw_dfs, freq_cols, unit_lbl,
                                      probe_table, diag_dir, len(raw_dfs))
        saved.append(p)
    except Exception as exc:
        print(f"[DIAG][WARN] eca_map_temporal_median failed: {exc}")

    print(f"[DIAG] Done — {len(saved)} figures written to '{diag_dir}'\n")
    return saved
