#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_ini_model.py
==================
Quick diagnostic plots of the RAW field inputs to the MFC2 real-data
pipeline, produced BEFORE any inversion/calibration step runs — i.e. the
"initial model" the rest of the pipeline starts from.
"""

from __future__ import annotations

import os
import sys
import glob
import argparse
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_real_data_loader():
    import data_real
    return data_real


def _compute_survey_eca(raw_csv_path: str, header_map: dict, unit: str = "ppt") -> pd.DataFrame:
    from emagpy import Problem

    with open(raw_csv_path, "r", errors="replace") as fh:
        first_line = next((line.strip() for line in fh if line.strip()), "")
    delimiter = ";" if first_line.count(";") >= first_line.count(",") else ","

    df = pd.read_csv(raw_csv_path, delimiter=delimiter)
    df.columns = df.columns.str.strip()

    present_raw_cols = [c for c in header_map if c in df.columns]
    if not present_raw_cols:
        raise ValueError(
            f"No recognised EM columns found in {raw_csv_path}. "
            f"Expected one of: {list(header_map.keys())}"
        )

    out_df = df[present_raw_cols].rename(columns=header_map)
    if "x" not in out_df.columns or "y" not in out_df.columns:
        raise ValueError(f"Missing x/y coordinate columns after renaming {raw_csv_path}")

    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(raw_csv_path)), "_ini_model_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, "tmp_" + os.path.basename(raw_csv_path))
    out_df.to_csv(tmp_path, index=False)

    k = Problem()
    k.createSurvey(tmp_path, unit=unit)
    df_out = k.surveys[0].df

    coil_cols = [c for c in df_out.columns if c.startswith("HCP") and not c.endswith("_quad")]
    if not coil_cols:
        raise ValueError(f"emagpy did not produce any ECa coil columns for {raw_csv_path}")

    eca_mean = df_out[coil_cols].mean(axis=1)

    return pd.DataFrame({
        "x": df_out["x"].values,
        "y": df_out["y"].values,
        "eca": eca_mean.values,
    })


def _load_swc_table(rdl_mod, tdr_dir: str, coord_file: str) -> pd.DataFrame:
    df_probes, target_depth_cols = rdl_mod.load_tdr_dataframe(tdr_dir, coord_file)
    if not target_depth_cols:
        raise ValueError("No SWC depth columns (e.g. 'SWC_15cm_Avg') detected in TDR files.")

    rows = []
    for station, df_station in df_probes.groupby("Unit"):
        row0 = df_station.iloc[0]
        if "X_UTM" in df_probes.columns and "Y_UTM" in df_probes.columns:
            ux = float(str(row0["X_UTM"]).replace(",", "."))
            uy = float(str(row0["Y_UTM"]).replace(",", "."))
        elif "x" in df_probes.columns and "y" in df_probes.columns:
            ux = float(str(row0["x"]).replace(",", "."))
            uy = float(str(row0["y"]).replace(",", "."))
        else:
            raise KeyError(
                f"Merged probe/coordinate table has columns {df_probes.columns.tolist()}; "
                "expected one of ('X_UTM','Y_UTM') or ('x','y')."
            )

        depth_vals = df_station[list(target_depth_cols.values())].mean(numeric_only=True).values.astype(float)
        depth_vals = np.where(depth_vals > 1.0, depth_vals / 100.0, depth_vals)
        mean_swc = float(np.nanmean(depth_vals))

        rows.append({"station": station, "utm_x": ux, "utm_y": uy, "swc": mean_swc})

    return pd.DataFrame(rows)


def _scatter_map(ax, x, y, c, cmap, cbar_label, title, vmin=None, vmax=None):
    sc = ax.scatter(x, y, c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                     s=45, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("UTM Easting [m]")
    ax.set_ylabel("UTM Northing [m]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    return sc


def plot_initial_model(em_dir: str, tdr_dir: str, coord_file: str, 
                       output_dir: str = "mfc2_outputs", em_unit: str = "ppt", 
                       em_date_index: int = 0) -> dict[str, str]:
    """
    Exposed API function to be called from the run_pipeline pipeline.
    """
    rdl_mod = _load_real_data_loader()
    out_dir = os.path.join(output_dir, "initial_model")
    os.makedirs(out_dir, exist_ok=True)

    em_files = sorted(glob.glob(os.path.join(em_dir, "*.csv")))
    dated = [(rdl_mod.extract_date_from_filename(f), f) for f in em_files]
    dated = sorted((d for d in dated if d[0] is not None), key=lambda t: t[0])
    if not dated:
        raise FileNotFoundError(f"No dated EM survey CSVs found in {em_dir}")
    if not (0 <= em_date_index < len(dated)):
        raise IndexError(f"em_date_index {em_date_index} out of range (found {len(dated)} surveys)")
        
    em_date, em_path = dated[em_date_index]
    
    df_eca = _compute_survey_eca(em_path, rdl_mod.EM_HEADER_MAP, unit=em_unit)
    df_swc = _load_swc_table(rdl_mod, tdr_dir, coord_file)
    swc_vmax = max(0.45, float(df_swc["swc"].max()))

    # --- Plotting ---
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    _scatter_map(ax1, df_eca["x"], df_eca["y"], df_eca["eca"], cmap="plasma",
                 cbar_label="ECa [mS/m]", title=f"Apparent Conductivity (ECa)\n{em_date.date()}")
    fig1.tight_layout()
    p1 = os.path.join(out_dir, f"eca_xy_{em_date.strftime('%Y%m%d')}.png")
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    _scatter_map(ax2, df_swc["utm_x"], df_swc["utm_y"], df_swc["swc"], cmap="Blues",
                 cbar_label="SWC [m3/m3]", title="Soil Water Content (TDR, mean over depth & time)",
                 vmin=0.0, vmax=swc_vmax)
    fig2.tight_layout()
    p2 = os.path.join(out_dir, "swc_xy.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    fig3, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6))
    _scatter_map(axA, df_eca["x"], df_eca["y"], df_eca["eca"], cmap="plasma",
                 cbar_label="ECa [mS/m]", title=f"ECa — {em_date.date()}")
    _scatter_map(axB, df_swc["utm_x"], df_swc["utm_y"], df_swc["swc"], cmap="Blues",
                 cbar_label="SWC [m3/m3]", title="SWC (TDR, mean over depth & time)",
                 vmin=0.0, vmax=swc_vmax)
    fig3.suptitle("Initial Model — Raw Field Inputs", fontsize=13)
    fig3.tight_layout()
    p3 = os.path.join(out_dir, "eca_swc_xy_combined.png")
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    return {"eca_plot": p1, "swc_plot": p2, "combined_plot": p3}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--em-dir", required=True)
    parser.add_argument("--tdr-dir", required=True)
    parser.add_argument("--coord-file", required=True)
    parser.add_argument("--em-unit", default="ppt", choices=["ppt", "ppm"])
    parser.add_argument("--em-date-index", type=int, default=0)
    parser.add_argument("--output-dir", default="mfc2_outputs")
    args = parser.parse_args()

    saved_paths = plot_initial_model(
        em_dir=args.em_dir, tdr_dir=args.tdr_dir, coord_file=args.coord_file,
        output_dir=args.output_dir, em_unit=args.em_unit, em_date_index=args.em_date_index
    )
    print(f"\n[DONE] Initial-model plots written to: {os.path.abspath(os.path.dirname(saved_paths['swc_plot']))}")


if __name__ == "__main__":
    main()