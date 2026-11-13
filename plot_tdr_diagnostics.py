#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_tdr_diagnostics.py
========================
Diagnostic check-plot for raw TDR (SoilNet/MFC2) data *right after import*,
before any EM-date alignment or averaging.

Why this exists
----------------
``data_real.py`` aligns every TDR reading to its nearest EM survey
date and averages within the time-tolerance window before Step 2 ever sees
the data. If something is wrong upstream (a stuck/flat-lined sensor, a
parsing artefact that leaves a logger reporting raw % instead of fraction,
a coordinate merge that silently drops a station, big timestamp gaps, ...)
it is invisible by the time you reach the petrophysical calibration plot.
This module renders the raw per-station time series so those issues are
visible *before* Step 2, and runs a few cheap automated checks.

It deliberately reuses ``load_tdr_dataframe()`` from
``data_real.py`` (no re-implementation of the parsing logic) so the
diagnostic always reflects exactly what the pipeline itself loaded.
"""

from __future__ import annotations

import os
import importlib.util
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# --------------------------------------------------------------------------
# Import load_tdr_dataframe from data_real.py (next to this script).
# --------------------------------------------------------------------------
def _import_real_data_loader():
    from data_real import load_tdr_dataframe
    return load_tdr_dataframe


load_tdr_dataframe = _import_real_data_loader()


# ==========================================================================
# QC HELPERS
# ==========================================================================
def _flatline_runs(values: np.ndarray, min_run: int = 5) -> int:
    """
    Count how many runs of >= `min_run` consecutive, *identical*, non-NaN
    readings exist in `values`. A stuck/frozen TDR sensor typically repeats
    the exact same raw value for many readings in a row — real soil moisture
    essentially never does this exactly, even when nearly static, because of
    sensor noise.
    """
    v = np.asarray(values, dtype=float)
    mask = ~np.isnan(v)
    if mask.sum() < min_run:
        return 0
    runs = 0
    run_len = 1
    for i in range(1, len(v)):
        if mask[i] and mask[i - 1] and v[i] == v[i - 1]:
            run_len += 1
        else:
            if run_len >= min_run:
                runs += 1
            run_len = 1
    if run_len >= min_run:
        runs += 1
    return runs


def _qc_row(station: str, depth_label: str, raw: np.ndarray) -> dict:
    raw = np.asarray(raw, dtype=float)
    n = len(raw)
    n_nan = int(np.isnan(raw).sum())
    valid = raw[~np.isnan(raw)]
    n_over_1 = int(np.sum(valid > 1.0))          # still in % units, not yet /100
    frac = np.where(valid > 1.0, valid / 100.0, valid)
    n_out_of_range = int(np.sum((frac < 0.0) | (frac > 0.65)))  # implausible VWC
    n_flat_runs = _flatline_runs(raw, min_run=5)
    return {
        "station": station,
        "depth": depth_label,
        "n_readings": n,
        "n_nan": n_nan,
        "pct_nan": 100.0 * n_nan / n if n else float("nan"),
        "n_pct_units_(>1.0)": n_over_1,
        "n_implausible_vwc": n_out_of_range,
        "n_flatline_runs(>=5)": n_flat_runs,
    }


# ==========================================================================
# MAIN DIAGNOSTIC ENTRY POINT
# ==========================================================================
def plot_tdr_diagnostics(
    tdr_dir: str,
    coord_file: str,
    em_dates: Sequence | None = None,
    time_tolerance_days: float = 1.0,
    output_dir: str = "mfc2_outputs",
    max_stations_per_figure: int = 8,
) -> list[str]:
    """
    Load raw TDR data (reusing the exact same parser as the production
    pipeline) and render one check-plot per station showing the unaligned,
    unaveraged SWC time series at every depth column found, with EM survey
    dates (and the matching tolerance window) overlaid for context.

    Also prints + saves a small QC table flagging:
      * missing readings (NaNs),
      * values still in raw "%" units (>1.0, i.e. not yet through the
        `/100` normalization applied downstream),
      * physically implausible VWC after normalization (<0 or >0.65 m3/m3),
      * flat-lined/stuck-sensor runs (>=5 identical consecutive readings).

    Parameters
    ----------
    tdr_dir, coord_file : str
        Same paths passed to ``data_real.prepare_real_dataset``.
    em_dates : sequence of datetime, optional
        EM survey dates to overlay as vertical reference lines (pass the
        ``em_dates`` returned by ``prepare_real_dataset`` /
        ``extract_date_from_filename``-derived list). If omitted, no survey
        markers are drawn.
    time_tolerance_days : float, optional
        Shading half-width around each EM date, mirroring the alignment
        tolerance used during Step 0 matching. Only used if `em_dates`
        is given.
    output_dir : str, optional
        Figures are written to ``<output_dir>/diagnostics/``.
    max_stations_per_figure : int, optional
        Stations are split across multiple figures if there are more than
        this many, so each figure stays readable.

    Returns
    -------
    list of str
        Paths to all figures written (always includes the QC table CSV path
        as the diagnostics report companion file too).
    """
    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    print(f"[TDR DIAG] Loading raw TDR data from {tdr_dir} (unaligned, unaveraged)...")
    df_probes, target_depth_cols = load_tdr_dataframe(tdr_dir, coord_file)

    if df_probes.empty:
        raise ValueError("[TDR DIAG] load_tdr_dataframe returned an empty dataframe — "
                          "check tdr_dir / coord_file contents.")

    stations = sorted(df_probes["Unit"].unique())
    print(f"[TDR DIAG] {len(stations)} station(s), depth columns: {target_depth_cols}")

    depth_colors = {0.15: "tab:orange", 0.30: "tab:blue"}
    default_colors = plt.cm.tab10(np.linspace(0, 1, max(len(target_depth_cols), 1)))

    qc_rows = []
    saved_paths: list[str] = []

    # Split stations across figures so each stays readable
    chunks = [stations[i:i + max_stations_per_figure]
              for i in range(0, len(stations), max_stations_per_figure)]

    for chunk_idx, chunk_stations in enumerate(chunks):
        n_rows = len(chunk_stations)
        fig, axes = plt.subplots(
            n_rows, 1, figsize=(10, 2.6 * n_rows), sharex=False, squeeze=False
        )
        axes = axes[:, 0]

        for ax, station in zip(axes, chunk_stations):
            df_st = df_probes[df_probes["Unit"] == station].sort_values("Timestamp")
            t = df_st["Timestamp"]

            for depth_idx, (depth_val, col_name) in enumerate(sorted(target_depth_cols.items())):
                if col_name not in df_st.columns:
                    continue
                raw_vals = df_st[col_name].to_numpy(dtype=float)

                qc_rows.append(_qc_row(station, f"{depth_val*100:.0f}cm", raw_vals))

                # Apply the same >1.0 -> /100 normalization used downstream,
                # purely for plotting on a consistent VWC axis.
                plot_vals = np.where(raw_vals > 1.0, raw_vals / 100.0, raw_vals)
                color = depth_colors.get(depth_val, default_colors[depth_idx])

                ax.plot(t, plot_vals, '.-', ms=3, lw=0.8, color=color,
                        label=f"{depth_val*100:.0f} cm")

                # Flag NaN gaps and out-of-range points explicitly
                nan_mask = np.isnan(raw_vals)
                if nan_mask.any():
                    ax.scatter(t[nan_mask], np.zeros(nan_mask.sum()),
                               marker='x', color='red', s=15, zorder=5)

                bad_mask = (~nan_mask) & ((plot_vals < 0.0) | (plot_vals > 0.65))
                if bad_mask.any():
                    ax.scatter(t[bad_mask], plot_vals[bad_mask],
                               marker='o', facecolor='none', edgecolor='red',
                               s=60, linewidth=1.3, zorder=6,
                               label="out-of-range" if depth_idx == 0 else None)

            # Overlay EM survey dates + tolerance window
            if em_dates is not None:
                tol = pd.Timedelta(days=time_tolerance_days)
                for j, em_dt in enumerate(em_dates):
                    em_dt = pd.Timestamp(em_dt)
                    ax.axvline(em_dt, color='k', linestyle='--', linewidth=0.8,
                               alpha=0.6, zorder=1,
                               label="EM survey" if j == 0 else None)
                    ax.axvspan(em_dt - tol, em_dt + tol, color='grey', alpha=0.12, zorder=0)

            ax.set_title(f"Station: {station}", fontsize=9, loc='left')
            ax.set_ylabel("VWC [m$^3$/m$^3$]", fontsize=8)
            ax.tick_params(axis='both', labelsize=7)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.legend(loc='upper right', fontsize=6, ncol=4, frameon=False)

        fig.suptitle("TDR Import Diagnostic — Raw Per-Station Time Series "
                     "(before EM-date alignment/averaging)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        suffix = f"_part{chunk_idx + 1}" if len(chunks) > 1 else ""
        out_path = os.path.join(diag_dir, f"tdr_timeseries_check{suffix}.png")
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)
        print(f"[TDR DIAG] Wrote {out_path}")

    # ----------------------------------------------------------------
    # QC summary table
    # ----------------------------------------------------------------
    df_qc = pd.DataFrame(qc_rows)
    qc_path = os.path.join(diag_dir, "tdr_import_qc.csv")
    df_qc.to_csv(qc_path, index=False)
    saved_paths.append(qc_path)

    print("\n[TDR DIAG] === Import QC summary ===")
    flagged = df_qc[
        (df_qc["n_pct_units_(>1.0)"] > 0)
        | (df_qc["n_implausible_vwc"] > 0)
        | (df_qc["n_flatline_runs(>=5)"] > 0)
        | (df_qc["pct_nan"] > 20.0)
    ]
    if flagged.empty:
        print("[TDR DIAG] No stations/depths flagged — raw TDR series look clean.")
    else:
        print("[TDR DIAG] Flagged station/depth combinations:")
        print(flagged.to_string(index=False))
    print(f"[TDR DIAG] Full QC table written to {qc_path}\n")

    return saved_paths


# ==========================================================================
# STANDALONE CLI
# ==========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot raw TDR time series check-plots right after import."
    )
    parser.add_argument("--tdr-dir", required=True)
    parser.add_argument("--coord-file", required=True)
    parser.add_argument("--em-dir", default=None,
                         help="Optional EM directory to derive survey dates "
                              "for vertical reference lines (uses the same "
                              "DDMMYY filename pattern as data_real.py).")
    parser.add_argument("--time-tolerance-days", type=float, default=1.0)
    parser.add_argument("--output-dir", default="mfc2_outputs")
    args = parser.parse_args()

    em_dates = None
    if args.em_dir:
        import glob as _glob
        import data_real as rdl
        em_dates = sorted(
            d for f in _glob.glob(os.path.join(args.em_dir, "*.csv"))
            if (d := rdl.extract_date_from_filename(f)) is not None
        )

    plot_tdr_diagnostics(
        tdr_dir=args.tdr_dir,
        coord_file=args.coord_file,
        em_dates=em_dates,
        time_tolerance_days=args.time_tolerance_days,
        output_dir=args.output_dir,
    )
