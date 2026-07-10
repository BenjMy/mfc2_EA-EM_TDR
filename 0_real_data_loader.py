#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0_real_data_loader.py
=====================
Unified data loader engine for real field data from the MFC2 catchment.
Adapts raw EM files and multi-sensor TDR datasets into a compatible interface
matching the Step 0 synthetic API, while outputting real matched spatial-temporal
soil moisture observations.
"""

from __future__ import annotations

import os
import glob
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Header mapping for multi-frequency GEM-2 Quadrature columns
EM_HEADER_MAP = {
    'X': 'x',
    'Y': 'y',
    'GPSalt': 'elevation',
    'Q_425Hz': 'HCP1.66f425h1_quad',
    'Q_1525Hz': 'HCP1.66f1525h1_quad',
    'Q_5325Hz': 'HCP1.66f5325h1_quad',
    'Q_18325Hz': 'HCP1.66f18325h1_quad',
    'Q_63025Hz': 'HCP1.66f63025h1_quad',
    'Q_92775Hz': 'HCP1.66f92775h1_quad'
}

def extract_date_from_filename(filename: str) -> datetime | None:
    """Extracts DDMMYY date patterns from file names."""
    match = re.search(r'(?:^|\D)(\d{6})(?:\D|$)', os.path.basename(filename))
    if match:
        try:
            return datetime.strptime(match.group(1), "%d%m%y")
        except ValueError:
            return None
    return None

def load_tdr_dataframe(tdr_dir: str, coord_file: str):
    """
    Load and merge raw TDR sensor logs with sensor coordinates, WITHOUT
    aligning to any EM survey date.

    Split out of ``prepare_real_dataset`` (unchanged logic, just lifted into
    its own function) so diagnostic tooling — see ``plot_tdr_diagnostics.py``
    — can inspect the raw, per-timestamp TDR time series independently of
    the EM-date matching/averaging step. ``prepare_real_dataset`` below calls
    this function internally, so its behaviour and return values are
    unchanged.

    Returns
    -------
    df_probes : pd.DataFrame
        One row per logger timestamp, merged with sensor coordinates.
    target_depth_cols : dict[float, str]
        Mapping of nominal SWC depth [m] -> source column name
        (e.g. ``{0.15: 'SWC_15cm_Avg', 0.30: 'SWC_30cm_Avg'}``).
    """
    print(f"[+] Loading sensor coordinates from {coord_file}...")
    if not os.path.exists(coord_file):
        raise FileNotFoundError(f"Missing coordinate file: {coord_file}")
    df_coords = pd.read_csv(coord_file, delimiter=';', comment='[')
    df_coords.columns = df_coords.columns.str.strip()
    if 'Pos' in df_coords.columns:
        df_coords['Pos'] = df_coords['Pos'].str.strip()

    # Discover and load Excel TDR streams
    if not os.path.exists(tdr_dir):
        raise FileNotFoundError(f"TDR directory not found: {tdr_dir}")

    tdr_files = [f for f in os.listdir(tdr_dir) if f.endswith('.xlsx')]
    if not tdr_files:
        raise FileNotFoundError(f"No TDR data spreadsheets found in {tdr_dir}")

    df_tdr_list = []
    for f in tdr_files:
        # Real SoilNet/MFC2 TDR loggers are named e.g. 'MTX04...xlsx',
        # 'MTX020...xlsx' — NOT 'Unit_<n>...xlsx'. This mirrors the unit-ID
        # extraction already proven against the real files in
        # 0_realTDR_data_loader.py.
        match = re.search(r'MTX(\d+)', f, re.IGNORECASE)
        if not match:
            print(f"[-] Warning: Could not parse Unit ID (MTX##) from filename '{f}'. Skipping.")
            continue
        unit_number = int(match.group(1))
        unit_id = f"Unit {unit_number}"

        fpath = os.path.join(tdr_dir, f)
        df_sheet = pd.read_excel(fpath)
        df_sheet.columns = df_sheet.columns.str.strip()

        # These loggers store the timestamp split across six columns rather
        # than a single Date/Time column.
        required_cols = ['Year', 'Month', 'Day', 'Hour', 'Min', 'Sec']
        if all(col in df_sheet.columns for col in required_cols):
            timestamp_str = (
                df_sheet['Year'].astype(str) + '-' +
                df_sheet['Month'].astype(str).str.zfill(2) + '-' +
                df_sheet['Day'].astype(str).str.zfill(2) + ' ' +
                df_sheet['Hour'].astype(str).str.zfill(2) + ':' +
                df_sheet['Min'].astype(str).str.zfill(2) + ':' +
                df_sheet['Sec'].astype(str).str.zfill(2)
            )
            df_sheet['Timestamp'] = pd.to_datetime(timestamp_str, errors='coerce')
            df_sheet = df_sheet.drop(columns=required_cols)
        else:
            # Fall back to a single combined date/time column, if present.
            time_col = [c for c in df_sheet.columns if 'time' in c.lower() or 'date' in c.lower()]
            if not time_col:
                print(f"[-] Warning: No usable datetime columns in '{f}'. Skipping.")
                continue
            df_sheet['Timestamp'] = pd.to_datetime(df_sheet[time_col[0]], errors='coerce')

        df_sheet['Unit'] = unit_id
        df_tdr_list.append(df_sheet)

    if not df_tdr_list:
        raise ValueError("Could not extract any structural TDR sensor rows.")

    df_tdr_all = pd.concat(df_tdr_list, ignore_index=True)

    # Re-map coordinates to TDR dataframe
    df_probes = pd.merge(df_tdr_all, df_coords, left_on='Unit', right_on='Pos', how='inner')

    swc_cols = [c for c in df_probes.columns if 'swc' in c.lower()]
    target_depth_cols = {}
    for c in swc_cols:
        if '15' in c:
            target_depth_cols[0.15] = c
        elif '30' in c:
            target_depth_cols[0.30] = c

    return df_probes, target_depth_cols


def filter_repeated_em_points(csv_path: str, tolerance: float = 0.2,
                               unit: str = "ppt") -> str:
    """
    Remove consecutive, near-duplicate spatial points (e.g. logger left
    stationary, overlapping GPS fixes) from a processed, emagpy-ready EM
    survey CSV, using emagpy's own ``Problem.filterRepeated`` so the result
    matches exactly what every downstream consumer (inversion workers,
    diagnostic plotting) would get if they re-filtered the same CSV
    themselves. The CSV is overwritten in place when points are dropped.

    This used to be called once per parallel inversion task in
    ``run_pipeline.py`` as ``k.filterRepeated(index=0)``. That call passed
    an ``index`` keyword which ``Problem.filterRepeated`` has never
    accepted (its real signature is ``filterRepeated(self,
    tolerance=0.2)``), so every single inversion task raised
    ``TypeError: filterRepeated() got an unexpected keyword argument
    'index'`` and failed. Filtering is now done once here, at load time,
    with the correct keyword.

    Parameters
    ----------
    csv_path : str
        Path to an already emagpy-ready (comma-separated, correctly
        named coil columns) EM survey CSV. Modified in place.
    tolerance : float, optional
        Minimum distance from the previous point required to be retained
        (passed straight through to emagpy). Default 0.2.
    unit : str, optional
        Unit of the EM readings in the CSV (e.g. "ppt" or "ppm"), needed
        by emagpy to parse the survey correctly. Default "ppt".

    Returns
    -------
    str
        The same ``csv_path`` (for convenient call chaining).
    """
    from emagpy import Problem

    # Capture the original column set BEFORE createSurvey runs, since
    # createSurvey()/readDF() mutates the dataframe in place: for every
    # "_quad" column it computes and appends a derived plain-name ECa
    # column (e.g. 'HCP1.66f425h1' alongside 'HCP1.66f425h1_quad'), plus
    # bookkeeping columns from filterRepeated's computeStat() (interdist,
    # azimuth, bearing, surveyTime, elapsed(sec)). Writing the *full*
    # post-createSurvey dataframe back to disk — as an earlier version of
    # this function did — leaves both the raw '_quad' column and the
    # derived plain-name column in the CSV. The next time createSurvey()
    # runs on that same file (e.g. inside the parallel inversion worker),
    # it recomputes the derived column AGAIN and appends it to k.coils a
    # second time, so k.coils ends up with duplicate entries like
    # ['HCP1.66f425h1', 'HCP1.66f425h1'] — which is exactly what broke
    # drop_coils_from_problem's substring matching downstream. Only the
    # original columns are written back out here to keep the file
    # idempotent across repeated createSurvey() calls.
    original_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()

    k = Problem()
    k.createSurvey(csv_path, unit=unit)
    n_before = len(k.surveys[0].df)
    k.filterRepeated(tolerance=tolerance)
    n_after = len(k.surveys[0].df)

    if n_after < n_before:
        k.surveys[0].df[original_columns].to_csv(csv_path, index=False)
        print(f"[+] filterRepeated: {os.path.basename(csv_path)} "
              f"{n_before} → {n_after} points (tolerance={tolerance})")

    return csv_path


def prepare_real_dataset(
    em_dir: str,
    tdr_dir: str,
    coord_file: str,
    time_tolerance_days: float = 1.0
):
    """
    Scans and merges real EM surveys and TDR time-series logs.

    CRITICAL ARCHITECTURAL CHANGE: Unlike the round-robin synthetic snap generator,
    this implementation gathers ALL available historical TDR readings matched to its
    nearest EM survey date (within a configurable tolerance, e.g. ±1 day) rather than
    the synthetic code's round-robin one-observation-per-probe — this gives Steps 2/4/5
    the full real time series instead of one arbitrary snapshot per sensor.
    """
    # Discover and sort EM files dynamically
    em_files = sorted(glob.glob(os.path.join(em_dir, "*.csv")))
    valid_em_entries = []
    for f in em_files:
        dt = extract_date_from_filename(f)
        if dt:
            valid_em_entries.append((dt, f))

    valid_em_entries = sorted(valid_em_entries, key=lambda x: x[0])
    if not valid_em_entries:
        raise FileNotFoundError(f"No valid dated EM files found in {em_dir}")

    em_dates = [item[0] for item in valid_em_entries]
    em_fnames = [item[1] for item in valid_em_entries]

    print(f"[+] Found {len(em_fnames)} EM survey baselines ranging from {em_dates[0].date()} to {em_dates[-1].date()}")

    # TDR loading + coordinate merge (shared with the diagnostic plotting tool)
    df_probes, target_depth_cols = load_tdr_dataframe(tdr_dir, coord_file)

    observations = []
    tolerance = timedelta(days=time_tolerance_days)

    print(f"[+] Aligning TDR logs to EM datelines (Tolerance: ±{time_tolerance_days} days)...")

    stations = df_probes['Unit'].unique()
    station_table_rows = []

    for idx_pos, station in enumerate(stations):
        df_station = df_probes[df_probes['Unit'] == station]
        row_sample = df_station.iloc[0]

        # SOILNET_MFC2_coordinates_sensors.csv ships UTM eastings/northings
        # under the plain headers 'x' / 'y' (see file header:
        # "x;y;z;Pos;Notes"), NOT 'X_UTM' / 'Y_UTM'. The old .get('X_UTM', 0)
        # lookup silently missed every row and defaulted every station to
        # (0.0, 0.0), which made every TDR probe equidistant from the EM
        # survey grid and collapsed the nearest-point spatial match onto a
        # single EM index for all stations. Fail loudly instead of
        # defaulting, so a future header rename surfaces immediately.
        try:
            if 'X_UTM' in df_probes.columns and 'Y_UTM' in df_probes.columns:
                ux = float(str(row_sample['X_UTM']).replace(',', '.'))
                uy = float(str(row_sample['Y_UTM']).replace(',', '.'))
            elif 'x' in df_probes.columns and 'y' in df_probes.columns:
                ux = float(str(row_sample['x']).replace(',', '.'))
                uy = float(str(row_sample['y']).replace(',', '.'))
            else:
                raise KeyError(
                    f"Merged probe/coordinate table has columns "
                    f"{df_probes.columns.tolist()}; expected one of "
                    "('X_UTM','Y_UTM') or ('x','y')."
                )
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"Could not parse UTM coordinates for station '{station}': {exc}"
            ) from exc

        station_table_rows.append({
            'probe_index': idx_pos,
            'station_name': station,
            'utm_x': ux,
            'utm_y': uy,
        })

        for idx_time, em_dt in enumerate(em_dates):
            df_matched_time = df_station[
                (df_station['Timestamp'] >= em_dt - tolerance) &
                (df_station['Timestamp'] <= em_dt + tolerance)
            ]

            if df_matched_time.empty:
                continue

            mean_vals = df_matched_time[list(target_depth_cols.values())].mean()

            for depth_val, col_name in target_depth_cols.items():
                swc_val = mean_vals[col_name]
                if not np.isnan(swc_val):
                    if swc_val > 1.0:
                        swc_val /= 100.0

                    observations.append({
                        'probe_index': idx_pos,
                        'time_index': idx_time,
                        'depth': depth_val,
                        'swc': swc_val
                    })

    df_probe_table = pd.DataFrame(station_table_rows)

    print(f"[ SUCCESS ] Extracted {len(observations)} matched empirical observations.")
    return em_fnames, em_dates, df_probe_table, observations
