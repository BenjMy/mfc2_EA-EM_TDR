import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
COORDINATES_FILE = "./data/SOILNET_MFC2_coordinates_sensors.csv"
TDR_DATA_DIR = "./data/daily_MFC/"
OUTPUT_PLOT_FILE = "tdr_swc_trends.png"

def load_sensor_coordinates(coord_file):
    """Loads and cleans sensor coordinates from the CSV file."""
    if not os.path.exists(coord_file):
        print(f"[-] Error: Coordinate file not found at {coord_file}")
        return pd.DataFrame()

    print(f"[+] Loading sensor coordinates from {coord_file}...")
    df_coords = pd.read_csv(coord_file, delimiter=';', comment='[')
    df_coords.columns = df_coords.columns.str.strip()

    if 'Pos' in df_coords.columns:
        df_coords['Pos'] = df_coords['Pos'].str.strip()

    return df_coords

def discover_and_load_tdr_data(data_dir):
    """Dynamically finds all XLSX files in the directory, maps them to Unit IDs,

    and securely handles timestamp conversions.
    """
    if not os.path.exists(data_dir):
        print(f"[-] Error: Data directory not found at {data_dir}")
        return pd.DataFrame()

    # List all Excel files
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    print(f"[+] Found {len(all_files)} XLSX files in {data_dir}:")
    for f in sorted(all_files):
        print(f"  - {f}")

    combined_data = []

    for filename in sorted(all_files):
        file_path = os.path.join(data_dir, filename)

        # Extract the number after 'MTX' (e.g., MTX04 -> Unit 4, MTX020 -> Unit 20)
        match = re.search(r'MTX(\d+)', filename)
        if not match:
            print(f"[-] Warning: Could not parse Unit ID from filename '{filename}'. Skipping.")
            continue

        unit_number = int(match.group(1))
        unit_name = f"Unit {unit_number}"

        print(f"[+] Processing {unit_name} from {filename}...")

        try:
            # Load Excel file
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip()

            # Required core datetime columns
            required_cols = ['Year', 'Month', 'Day', 'Hour', 'Min', 'Sec']
            if not all(col in df.columns for col in required_cols):
                print(f"[-] Warning: Missing datetime columns in {filename}. Skipping.")
                continue

            # Robust datetime assembly via string format to prevent "extra keys" exceptions
            timestamp_str = (
                df['Year'].astype(str) + '-' +
                df['Month'].astype(str).str.zfill(2) + '-' +
                df['Day'].astype(str).str.zfill(2) + ' ' +
                df['Hour'].astype(str).str.zfill(2) + ':' +
                df['Min'].astype(str).str.zfill(2) + ':' +
                df['Sec'].astype(str).str.zfill(2)
            )
            df['Timestamp'] = pd.to_datetime(timestamp_str, errors='coerce')

            # Clean up old temporal components
            df = df.drop(columns=required_cols)

            # Add identifier metadata
            df['Sensor_Unit'] = unit_name

            combined_data.append(df)

        except Exception as e:
            print(f"[-] Error reading {filename}: {e}")

    if combined_data:
        return pd.concat(combined_data, ignore_index=True)
    else:
        return pd.DataFrame()

def load_em_data(em_dir_or_file=None):
    """Placeholder function for loading Electromagnetic (EM) real data."""
    print("[+] Scaffold: Ready to load EM Data...")
    return pd.DataFrame()

def plot_swc_data(df, save_path=OUTPUT_PLOT_FILE):
    """Generates and explicitly saves time-series plots for SWC."""
    if df.empty:
        print("[-] No data available to plot.")
        return

    print(f"[+] Generating SWC time-series plots...")
    sns.set_theme(style="darkgrid")

    # Force creation of a unique target figure object
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot SWC at 15cm depth
    if 'SWC_15cm' in df.columns:
        sns.lineplot(data=df, x='Timestamp', y='SWC_15cm', hue='Sensor_Unit', ax=axes[0], marker='o', alpha=0.8)
        axes[0].set_title('Soil Water Content at 15cm Depth ($SWC_{15cm}$)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('SWC (%)', fontsize=12)
        axes[0].legend(title="Sensors", bbox_to_anchor=(1.02, 1), loc='upper left')

    # Plot SWC at 30cm depth
    if 'SWC_30cm' in df.columns:
        sns.lineplot(data=df, x='Timestamp', y='SWC_30cm', hue='Sensor_Unit', ax=axes[1], marker='o', alpha=0.8)
        axes[1].set_title('Soil Water Content at 30cm Depth ($SWC_{30cm}$)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('SWC (%)', fontsize=12)
        axes[1].set_xlabel('Date / Time', fontsize=12)
        axes[1].legend(title="Sensors", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()

    # CRITICAL FIX: Save the file FIRST before calling plt.show()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[ SUCCESS ] Figure successfully exported to: {os.path.abspath(save_path)}")

    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print(" STARTING EM & TDR REAL DATA LOADER ".center(60, "="))
    print("="*60)

    # 1. Load the spatial coordinates
    df_coords = load_sensor_coordinates(COORDINATES_FILE)
    print("-" * 60)

    # 2. Dynamically scan folder and load TDR data
    df_tdr_all = discover_and_load_tdr_data(TDR_DATA_DIR)
    print("-" * 60)

    # 3. Load EM Data (Ready for expansion)
    df_em_all = load_em_data()
    print("-" * 60)

    # 4. Optional: Merge time-series data with coordinates on Unit Name
    if not df_tdr_all.empty and not df_coords.empty:
        df_merged = pd.merge(df_tdr_all, df_coords, left_on='Sensor_Unit', right_on='Pos', how='left')
        print(f"[+] Combined TDR spatial dataset shape: {df_merged.shape}")

    # 5. Create and save the visual plots
    if not df_tdr_all.empty:
        plot_swc_data(df_tdr_all)
        print("="*60)
        print(" PROCESS COMPLETED SUCCESSFULLY ".center(60, "="))
        print("="*60)
