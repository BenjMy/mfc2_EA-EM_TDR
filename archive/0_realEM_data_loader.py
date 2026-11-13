import os
import glob
import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from emagpy import Problem

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
EM_DATA_DIR = "./data/EMI_MFC2/"
OUTPUT_PROCESSED_DIR = "./data/EMI_MFC2_processed/"
OUTPUT_SUMMARY_PLOT = "./data/em_timelapse_space_time_heatmap.png"

# Header translation map for multifrequency GEM-2 Quadrature columns
HEADER_MAP = {
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

def extract_date_from_filename(filename):
    """Extracts date matching DDMMYY pattern from file names."""
    # Looks for a 6-digit sequence bounded by non-digits or string edges
    match = re.search(r'(?:^|\D)(\d{6})(?:\D|$)', filename)
    if match:
        date_str = match.group(1)
        try:
            # Parse as Day/Month/Year (2-digit year)
            return datetime.strptime(date_str, "%d%m%y")
        except ValueError:
            return None
    return None

def process_and_register_timelapse(data_dir):
    """Scans, parses dates, chronologically sorts, and instantiates emagpy surveys."""
    if not os.path.exists(data_dir):
        print(f"[-] Error: Data directory not found at {data_dir}")
        return []

    raw_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not raw_files:
        print(f"[-] No CSV files found in {data_dir}")
        return []

    timelapse_registry = []

    # Step 1: Parse and link file dates
    for file_path in raw_files:
        filename = os.path.basename(file_path)
        file_date = extract_date_from_filename(filename)

        if file_date:
            timelapse_registry.append({
                'date': file_date,
                'date_str': file_date.strftime('%Y-%m-%d'),
                'raw_path': file_path,
                'filename': filename
            })
        else:
            print(f"[-] Warning: Could not parse DDMMYY date from '{filename}'. Skipping.")

    # Step 2: Sort chronologically by date
    timelapse_registry.sort(key=lambda x: x['date'])

    print(f"\n[+] Found {len(timelapse_registry)} time-lapse records in chronological order:")
    for idx, record in enumerate(timelapse_registry):
        print(f"  [{idx + 1}] Date: {record['date_str']} | File: {record['filename']}")
    print("-" * 60)

    os.makedirs(OUTPUT_PROCESSED_DIR, exist_ok=True)

    # Step 3: Loop through, re-map headers, and build emagpy problems
    for record in timelapse_registry:
        print(f"[+] Step: {record['date_str']} | Processing: {record['filename']}...")
        processed_path = os.path.join(OUTPUT_PROCESSED_DIR, f"processed_{record['filename']}")

        try:
            # Load raw data (assumes ',' delimiter based on header line)
            df = pd.read_csv(record['raw_path'], delimiter=',')
            df.columns = df.columns.str.strip()

            # Select and map specified columns
            df_processed = df[list(HEADER_MAP.keys())].rename(columns=HEADER_MAP)
            df_processed.to_csv(processed_path, index=False)

            # Instantiate emagpy Survey
            problem = Problem()
            problem.createSurvey(processed_path, unit="ppm")

            # Store structural info for visualization stage
            record['processed_path'] = processed_path
            record['emagpy_problem'] = problem

        except Exception as e:
            print(f"[-] Failed processing day {record['date_str']}: {e}")

    return timelapse_registry

def plot_em_timelapse_heatmap(timelapse_registry, target_coil='HCP1.66f5325h1_quad', save_path=OUTPUT_SUMMARY_PLOT):
    """Aggregates all time-lapse periods and builds an absolute Space-Time

    Hovmöller Evolution Heatmap to view spatial anomalies over time.
    """
    all_days_data = []

    for record in timelapse_registry:
        if 'processed_path' in record:
            df = pd.read_csv(record['processed_path'])
            df['Date'] = record['date_str']
            df['Point_Index'] = df.index  # Track coordinate along index/transect line
            all_days_data.append(df)

    if not all_days_data:
        print("[-] No data available to build summary time-lapse matrix.")
        return

    df_all = pd.concat(all_days_data, ignore_index=True)
    sns.set_theme(style="white")

    # Build 2D matrix: Rows = Dates, Columns = Positions, Values = Quadrature Signal
    heatmap_matrix = df_all.pivot(index="Date", columns="Point_Index", values=target_coil)
    heatmap_matrix = heatmap_matrix.sort_index(ascending=True) # Ascending chronological dates

    fig, ax = plt.subplots(figsize=(13, 6))

    # Generate Heatmap (Using standard Earth/Seawater color ramp)
    sns.heatmap(
        heatmap_matrix,
        cmap="YlGnBu",
        cbar_kws={'label': f'Quadrature Response [ppm] ({target_coil})'},
        linewidths=0.05,
        ax=ax
    )

    ax.set_title("Time-Lapse Subsurface Spatial-Temporal Matrix Plot", fontsize=14, fontweight='bold')
    ax.set_xlabel("Transect/Measurement Station Index", fontsize=12)
    ax.set_ylabel("Observation Baseline Date", fontsize=12)

    plt.tight_layout()

    # Secure the visual representation BEFORE rendering plot context
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[ SUCCESS ] Space-Time Heatmap exported to: {os.path.abspath(save_path)}")

    plt.show()

# ==========================================
# MAIN ROUTINE EXECUTION
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print(" STARTING REAL TIME-LAPSE EM PIPE ENGINE ".center(60, "="))
    print("="*60)

    # Process files and register emagpy parameters
    registry = process_and_register_timelapse(EM_DATA_DIR)

    # Generate the global time-lapse summary graphic
    if registry:
        print("-" * 60)
        # Choosing a mid-range frequency (5325 Hz) for structural monitoring
        plot_em_timelapse_heatmap(registry, target_coil='HCP1.66f5325h1_quad', save_path=OUTPUT_SUMMARY_PLOT)

    print("="*60)
    print(" PROCESSING COMPLETED ".center(60, "="))
    print("="*60)
