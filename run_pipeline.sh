#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — EA-EM Ensemble Pipeline launcher (Synthetic & Real Data)
# =============================================================================
# Usage:
#   ./run_pipeline.sh                                           # synthetic, default scenario, 1-D mode
#   ./run_pipeline.sh --data-source real --em-dir ./data/EMI_MFC2/ --tdr-dir ./data/daily_MFC/
#   ./run_pipeline.sh --em-dir ./data/EMI_MFC2/ --tdr-dir ./data/daily_MFC/ --coord-file ./data/SOILNET_MFC2_coordinates_sensors.csv
#   ./run_pipeline.sh --mode 2d                                # synthetic, 2-D UTM grid
#   ./run_pipeline.sh -s mixed_middle_free                     # synthetic, named scenario, 1-D
#   ./run_pipeline.sh -s mixed_middle_free --mode 2d           # synthetic, named scenario, 2-D
#   ./run_pipeline.sh -s fine_layers_fixed -c my.json          # synthetic, custom config file
#   ./run_pipeline.sh -s free_all -e /path/to/venv             # synthetic, custom virtualenv
#   ./run_pipeline.sh --mode 2d --nx 15 --ny 8                 # synthetic, 2-D grid 15×8 points
#   ./run_pipeline.sh --list                                    # list scenarios
#   ./run_pipeline.sh --validate                               # validate all scenarios
#   ./run_pipeline.sh --help                                   # show this message
#
# Localization diagnostic (Step 2 kernel-localized WS calibration)
# ------------------------------------------------------------------
#   OFF by default for every scenario. Controlled only by the scenario's
#   own 'ws_localized_bandwidth' field in ensemble_scenarios.json -- no
#   CLI flag on this wrapper turns it on/off. See `--help` for details,
#   current per-scenario status, and how to enable/disable it.
#
# Data sources
# -----------
#   synthetic (default)   Generate synthetic forward models and add noise.
#                         Use --scenario to select inversion configuration.
#
#   real                  Load empirical EM and TDR time-series from field data.
#                         Required args: --em-dir, --tdr-dir, --coord-file.
#                         --scenario also applies here (depths0, cond_ranges,
#                         param_grid, SWC sensor layout); use a real-data
#                         scenario such as 'mfc2_real', or 'default' if
#                         omitted.
#
# Spatial modes
# -------------
#   1d (default)  Single transect. Measurement points placed along a straight
#                 line; no geographic coordinates. Produces XZ cross-section
#                 plots only.
#
#   2d            Rectangular UTM grid anchored to the Alento field survey
#                 bounding box (515487 E, 4468201 N — UTM Zone 33N).
#                 Grid size is controlled by --nx / --ny (default 10×5).
#                 Survey CSVs gain utm_x / utm_y columns.
#                 Produces XZ cross-section plots AND XY plan-view maps.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATA_SOURCE=""
SCENARIO="default"
CONFIG_FILE=""
VENV_PATH=""
PYTHON="python3"
MODE="1d"
NX=10
NY=5
EM_DIR=""
TDR_DIR=""
COORD_FILE="./data/SOILNET_MFC2_coordinates_sensors.csv"
DROP_COILS=()
HOMOGENEOUS=false
FULL_RANGE_GRADIENT=false
XY_Z_BLEND=""
INVERT_Z_GRADIENT=false
PROBE_XYZ=""
XY_RENDER_MODE=""
NUM_SWC_POSITIONS=""
SWC_DEPTHS=()
WINDOW_M=""
BETA=()
GAMMA=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_pipeline.py"
SCENARIO_LOADER="${SCRIPT_DIR}/scenario_loader.py"
SCENARIO_JSON="${SCRIPT_DIR}/ensemble_scenarios.json"
REAL_DATA_LOADER="${SCRIPT_DIR}/0_real_data_loader.py"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF

Usage: $(basename "$0") [OPTIONS]

Data source options:
      --data-source      Data source: 'synthetic' (default) or 'real'.
                          real mode requires --em-dir, --tdr-dir, --coord-file.

Real data options (when --data-source real):
      --em-dir PATH      Path to directory containing EM survey CSV files
                          (e.g., ./data/EMI_MFC2/)
      --tdr-dir PATH     Path to directory containing TDR Excel spreadsheets
                          (e.g., ./data/daily_MFC/)
      --coord-file PATH  Path to sensor coordinates CSV file
                          (default: ./data/SOILNET_MFC2_coordinates_sensors.csv)

EM channel options:
      --drop-coils COIL [COIL ...]
                          One or more coils/frequencies to exclude from every
                          inversion (e.g. a noisy GEM-2 channel). Each value
                          may be the exact emagpy coil name, a frequency in
                          Hz (e.g. 425), or a unique coil-name substring
                          (e.g. f425). Overrides any 'drop_coils' field in
                          the scenario config. Example: --drop-coils f425

Spatial options:
      --mode 1d|2d        Spatial mode (default: 1d).
                          1d — single transect, XZ plots only.
                          2d — UTM grid from Alento survey, XZ + XY plots.
      --nx INT            Grid columns in 2-D mode (default: 10).
      --ny INT            Grid rows    in 2-D mode (default: 5).
      --homogeneous       Synthetic mode only. Disable spatial (XY/XZ)
                          gradients in the generated ground truth so theta
                          is spatially uniform; temporal trend and noise
                          are still applied. Overrides 'homogeneous' in the
                          scenario JSON when passed.
      --full-range-gradient
                          Synthetic mode only (ignored if --homogeneous is
                          also given). Static XZ (depth) and XY (plan-view)
                          initial-model gradients each span the FULL
                          [theta_dry, theta_wet] range independently,
                          instead of the small fixed-amplitude default
                          gradient (which only adds ~0.12-0.18 vol.
                          fraction on top of a 0.05 baseline). Overrides
                          'full_range_gradient' in the scenario JSON.
      --xy-z-blend W      Only used with --full-range-gradient. Weight in
                          [0, 1] between the XY component (1.0 = XY only)
                          and the Z/depth component (0.0 = Z only) of the
                          initial model. Default: 0.5 (equal blend).
                          Overrides 'xy_z_blend' in the scenario JSON.
      --invert-z-gradient Only used with --full-range-gradient. Flip the
                          depth gradient so the shallowest layer is wet and
                          the deepest layer is dry (default: dry at
                          surface, wet at depth). Overrides
                          'invert_z_gradient' in the scenario JSON.

Probe placement and XY plot rendering:
      --probe-xyz PATH    CSV of explicit SWC/TDR probe locations, one row
                          per probe, e.g. --probe-xyz coord_TDR.csv.
                          Accepted columns (case-insensitive): x/utm_x/
                          easting, y/utm_y/northing, z/depth/depth_m, and
                          an optional label column (id/name/probe/label).
                          When given, completely replaces the auto-
                          placement grid derived from --num-swc-positions/
                          --swc-depths for BOTH 1-D and 2-D mode -- each
                          row becomes one probe at its literal (x, y, z).
      --xy-render-mode {scatter,imshow}
                          Rendering for XY plan-view plots (2-D mode only).
                          'scatter' (default) draws one square per
                          measurement point; 'imshow' interpolates onto a
                          smooth raster (scipy.griddata).

SWC / TDR sensor options (override the scenario JSON's sensor network):
      --num-swc-positions N
                          Number of horizontal SWC/TDR sensor positions.
                          Total probes = N x n_layers (one per soil horizon
                          at each position). Overrides 'num_swc_positions'
                          in the scenario config.
      --swc-depths D [D ...]
                          Explicit sensor installation depths [m], e.g.
                          --swc-depths 0.15 0.45 0.9. Overrides
                          'swc_depths' in the scenario config (which
                          otherwise auto-derives depths from layer centres).
      --window-m M        Half-width [m] of the spatial averaging window
                          used to match each sensor to nearby EM
                          measurement indices. Overrides 'window_m' in the
                          scenario config.

Inversion options (override the scenario JSON's param_grid / gamma):
      --beta B [B ...]    One or more beta values (lateral/spatial
                          smoothing, emagpy inversion) to sweep, e.g.
                          --beta 0.0 0.1 0.5. Overrides 'param_grid.beta'
                          in the scenario config.
      --gamma G           Temporal regularisation between time steps
                          (>= 0), only used for time-lapse (multi-epoch)
                          inversions. Overrides 'gamma' in the scenario
                          config.

Scenario options (apply in both synthetic and real-data mode):
  -s, --scenario NAME     Scenario name from ensemble_scenarios.json
                          (default: 'default'; use e.g. 'mfc2_real' for
                          real-data runs to match your actual sensor depths)
  -c, --config PATH       Path to a custom JSON scenario file
                          (default: ensemble_scenarios.json next to this script)

Localization diagnostic (Step 2 kernel-localized WS calibration):
      This runs a DIAGNOSTIC SIBLING to the normal Step 2 petrophysical
      calibration: instead of one flat Waxman-Smits parameter set per
      soil horizon, it fits a separate set at every EM point, weighting
      nearby TDR probes more heavily (Gaspari-Cohn kernel, radius 'c').
      It never replaces Step 2 and never changes Steps 3-5 -- it only
      adds 'petro_calibration_localized.png' (and, in 2-D mode, a
      sigma_s(x, y) map) to the output directory.

      DEFAULT: OFF. There is no --localized-bandwidth flag on this shell
      wrapper -- localization is controlled entirely by the scenario's
      own 'ws_localized_bandwidth' field in ensemble_scenarios.json
      (or your --config file), so the bandwidth stays tied to that
      scenario's own probe geometry instead of being set blindly from
      the command line. A scenario that omits the field (the default
      for every scenario except '2d') runs with localization disabled,
      exactly like before this feature existed.

      To turn it ON for a scenario: add
          "ws_localized_bandwidth": <radius in the same units as your
                                      x/y coordinates, e.g. metres>
      to that scenario's block in the JSON. Rule of thumb: pick a value
      at least ~half the spacing between your closest adjacent SWC/TDR
      probes (kernel support is exactly zero beyond 2×c, so anything
      smaller and most EM points won't see any probe at all).

      To turn it OFF for a scenario that currently has it on: delete
      the 'ws_localized_bandwidth' key, or set it to null.

      Current status in ensemble_scenarios.json (re-check after editing):
          2d                      -> ON  (c = 5.0)
          2d_dense_tdr            -> off (add the field to enable)
          2d_real                 -> off
          moisture_2d_free        -> off
          moisture_2d_dense_tdr   -> off
          mfc2_real_synthetic_mirror -> off
      NOTE: '2d''s c=5.0 is under-provisioned for its own default 2-probe
      layout (--num-swc-positions 2 on the default 10x5 grid places the
      two probes ~10.8 m apart, so the rule-of-thumb minimum is ~5.4 m).
      Either raise c to >= 6 in the JSON, or run with more probes
      (--num-swc-positions, e.g. 6-10) so a bandwidth of 5.0 has more
      than two points to localize between.

Environment options:
  -e, --env PATH          Path to a Python virtualenv to activate before running
                          (e.g. /path/to/venv or /path/to/conda/envs/myenv)
  -p, --python BIN        Python interpreter to use (default: python3)

Utility:
      --list              List all available scenario names and exit
      --validate          Validate all scenarios in the JSON file and exit
  -h, --help              Show this help message and exit

Examples:

  # --- Synthetic 1-D (classic single transect) ---
  ./run_pipeline.sh
  ./run_pipeline.sh -s free_all
  ./run_pipeline.sh -s mixed_middle_free -c /data/my_scenarios.json
  ./run_pipeline.sh -s fine_layers_fixed -e ~/venvs/eaem

  # --- Synthetic 2-D (UTM grid, XZ + XY plots) ---
  ./run_pipeline.sh --mode 2d
  ./run_pipeline.sh --mode 2d -s free_all
  ./run_pipeline.sh --mode 2d --nx 15 --ny 8
  ./run_pipeline.sh --mode 2d -s mixed_middle_free --nx 20 --ny 10 -e ~/venvs/eaem
  ./run_pipeline.sh --mode 2d -s 2d --homogeneous       # spatially uniform ground truth
  ./run_pipeline.sh --mode 2d -s 2d --full-range-gradient   # full dry-to-wet XZ+XY gradient
  ./run_pipeline.sh --mode 2d -s 2d --full-range-gradient --xy-z-blend 0.3 --invert-z-gradient

  # --- Override SWC/TDR sensor network from the command line ---
  ./run_pipeline.sh --mode 2d -s 2d --num-swc-positions 6
  ./run_pipeline.sh --mode 2d -s 2d --swc-depths 0.15 0.45 0.9
  ./run_pipeline.sh --mode 2d -s 2d --num-swc-positions 6 --swc-depths 0.15 0.45 0.9 --window-m 1.5
  ./run_pipeline.sh --mode 2d -s 2d --probe-xyz coord_TDR.csv   # explicit probe (x,y,z) from CSV

  # --- Control the XY plan-view plot rendering (2-D mode) ---
  ./run_pipeline.sh --mode 2d -s 2d --xy-render-mode imshow

  # --- Real data 1-D (field observations) ---
  ./run_pipeline.sh --data-source real --em-dir ./data/EMI_MFC2/ --tdr-dir ./data/daily_MFC/
  ./run_pipeline.sh --data-source real \
      --em-dir ./data/EMI_MFC2/ \
      --tdr-dir ./data/daily_MFC/ \
      --coord-file ./data/coordinates.csv
  # Use the dedicated real-data scenario (matches real 15cm/30cm TDR depths):
  ./run_pipeline.sh --data-source real -s mfc2_real \
      --em-dir ./data/EMI_MFC2/ \
      --tdr-dir ./data/daily_MFC/

  # --- Real data 2-D (field observations on UTM grid) ---
  ./run_pipeline.sh --data-source real --mode 2d -s mfc2_real \
      --em-dir ./data/EMI_MFC2/ \
      --tdr-dir ./data/daily_MFC/ \
      --nx 15 --ny 8

  # --- Drop a noisy coil/frequency channel before inversion ---
  ./run_pipeline.sh --data-source real -s mfc2_real \
      --em-dir ./data/EMI_MFC2/ \
      --tdr-dir ./data/daily_MFC/ \
      --drop-coils f425
  ./run_pipeline.sh --drop-coils f425 18325   # multiple channels

  # --- Utility ---
  ./run_pipeline.sh --list
  ./run_pipeline.sh --validate

Available scenarios (synthetic mode only):
  default                 Classic fixed-depth run
  fixed_dense             Fixed-depth with richer parameter sweep
  free_all                All interfaces free
  mixed_middle_free       Outer interfaces fixed, middle interface free
  mixed_deep_free         Shallow interface fixed, two deeper interfaces free
  free_timelapse_smooth   All free with temporal regularisation (gamma=0.2)
  fine_layers_fixed       Five-layer fixed discretisation
  deep_3m                 5 layers, fixed, max depth 3 m  ← use for deep profiles
  deep_3m_free            5 layers, free interfaces, max depth 3 m
  high_variance_sweep
  geometric_uncertainty
  extreme_stress_test

EOF
}

log() { echo "[$(date '+%H:%M:%S')] $*"; }
err() { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; }
die() { err "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
LIST_ONLY=false
VALIDATE_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-source)  DATA_SOURCE="${2:?'--data-source requires synthetic or real'}"; shift 2 ;;
        --em-dir)       EM_DIR="${2:?'--em-dir requires a path'}";                      shift 2 ;;
        --tdr-dir)      TDR_DIR="${2:?'--tdr-dir requires a path'}";                    shift 2 ;;
        --coord-file)   COORD_FILE="${2:?'--coord-file requires a path'}";              shift 2 ;;
        --drop-coils)
            [[ $# -ge 2 ]] || die "--drop-coils requires at least one coil/frequency value"
            shift # consume the flag itself
            DROP_COILS=()
            # Greedily consume one-or-more following tokens, stopping at the
            # next recognised flag (mirrors Python argparse's nargs='+').
            while [[ $# -gt 0 && "$1" != -* ]]; do
                DROP_COILS+=("$1")
                shift
            done
            [[ ${#DROP_COILS[@]} -ge 1 ]] || die "--drop-coils requires at least one coil/frequency value"
            ;;
        -s|--scenario)  SCENARIO="${2:?'--scenario requires a value'}";                 shift 2 ;;
        -c|--config)    CONFIG_FILE="${2:?'--config requires a value'}";                shift 2 ;;
        -e|--env)       VENV_PATH="${2:?'--env requires a value'}";                     shift 2 ;;
        -p|--python)    PYTHON="${2:?'--python requires a value'}";                     shift 2 ;;
        --mode)
            MODE="${2:?'--mode requires 1d or 2d'}"
            [[ "${MODE}" == "1d" || "${MODE}" == "2d" ]] \
                || die "--mode must be '1d' or '2d', got '${MODE}'"
            shift 2 ;;
        --nx)
            NX="${2:?'--nx requires an integer'}"
            [[ "${NX}" =~ ^[0-9]+$ ]] || die "--nx must be a positive integer"
            shift 2 ;;
        --ny)
            NY="${2:?'--ny requires an integer'}"
            [[ "${NY}" =~ ^[0-9]+$ ]] || die "--ny must be a positive integer"
            shift 2 ;;
        --num-swc-positions)
            NUM_SWC_POSITIONS="${2:?'--num-swc-positions requires an integer'}"
            [[ "${NUM_SWC_POSITIONS}" =~ ^[0-9]+$ && "${NUM_SWC_POSITIONS}" -ge 1 ]] \
                || die "--num-swc-positions must be a positive integer"
            shift 2 ;;
        --swc-depths)
            [[ $# -ge 2 ]] || die "--swc-depths requires at least one depth value"
            shift # consume the flag itself
            SWC_DEPTHS=()
            # Greedily consume one-or-more following tokens, stopping at the
            # next recognised flag (mirrors Python argparse's nargs='+').
            while [[ $# -gt 0 && "$1" != -* ]]; do
                SWC_DEPTHS+=("$1")
                shift
            done
            [[ ${#SWC_DEPTHS[@]} -ge 1 ]] || die "--swc-depths requires at least one depth value"
            ;;
        --window-m)
            WINDOW_M="${2:?'--window-m requires a number'}"
            shift 2 ;;
        --beta)
            [[ $# -ge 2 ]] || die "--beta requires at least one value"
            shift # consume the flag itself
            BETA=()
            # Greedily consume one-or-more following tokens, stopping at the
            # next recognised flag (mirrors Python argparse's nargs='+').
            while [[ $# -gt 0 && "$1" != -* ]]; do
                BETA+=("$1")
                shift
            done
            [[ ${#BETA[@]} -ge 1 ]] || die "--beta requires at least one value"
            ;;
        --gamma)
            GAMMA="${2:?'--gamma requires a number'}"
            shift 2 ;;
        --list)     LIST_ONLY=true;     shift ;;
        --validate) VALIDATE_ONLY=true; shift ;;
        --homogeneous) HOMOGENEOUS=true; shift ;;
        --full-range-gradient) FULL_RANGE_GRADIENT=true; shift ;;
        --xy-z-blend)
            XY_Z_BLEND="${2:?'--xy-z-blend requires a number'}"
            shift 2 ;;
        --invert-z-gradient) INVERT_Z_GRADIENT=true; shift ;;
        --probe-xyz)    PROBE_XYZ="${2:?'--probe-xyz requires a path'}";         shift 2 ;;
        --xy-render-mode)
            XY_RENDER_MODE="${2:?'--xy-render-mode requires scatter or imshow'}"
            [[ "${XY_RENDER_MODE}" == "scatter" || "${XY_RENDER_MODE}" == "imshow" ]] \
                || die "--xy-render-mode must be 'scatter' or 'imshow', got '${XY_RENDER_MODE}'"
            shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *) die "Unknown option: $1  (run with --help for usage)" ;;
    esac
done

# Warn when --nx / --ny are passed but mode is 1d
if [[ "${MODE}" == "1d" && ( "${NX}" != "10" || "${NY}" != "5" ) ]]; then
    log "WARNING: --nx / --ny are ignored in 1-D mode. Pass --mode 2d to use them."
fi

# Warn when --homogeneous is passed for real-data runs (synthetic-only flag)
if ${HOMOGENEOUS} && [[ "${DATA_SOURCE}" == "real" ]]; then
    log "WARNING: --homogeneous is ignored in real-data mode (synthetic generation only)."
fi

# Warn when --full-range-gradient / --xy-z-blend / --invert-z-gradient are
# passed for real-data runs (synthetic-generation-only flags)
if ${FULL_RANGE_GRADIENT} && [[ "${DATA_SOURCE}" == "real" ]]; then
    log "WARNING: --full-range-gradient is ignored in real-data mode (synthetic generation only)."
fi
if [[ -n "${XY_Z_BLEND}" && "${DATA_SOURCE}" == "real" ]]; then
    log "WARNING: --xy-z-blend is ignored in real-data mode (synthetic generation only)."
fi
if ${INVERT_Z_GRADIENT} && [[ "${DATA_SOURCE}" == "real" ]]; then
    log "WARNING: --invert-z-gradient is ignored in real-data mode (synthetic generation only)."
fi

# Warn when --full-range-gradient's blend/invert flags are passed without it
if [[ -n "${XY_Z_BLEND}" ]] && ! ${FULL_RANGE_GRADIENT}; then
    log "WARNING: --xy-z-blend has no effect without --full-range-gradient."
fi
if ${INVERT_Z_GRADIENT} && ! ${FULL_RANGE_GRADIENT}; then
    log "WARNING: --invert-z-gradient has no effect without --full-range-gradient."
fi

# ---------------------------------------------------------------------------
# Resolve data source and paths from scenario JSON
# ---------------------------------------------------------------------------
# Read data_source, em_data_dir, tdr_data_dir, and coord_file from the chosen
# scenario if they are declared there.  CLI flags take precedence over JSON
# values when explicitly supplied, but the two must not disagree.
_SCENARIO_JSON_PATH="${CONFIG_FILE:-${SCENARIO_JSON}}"

_read_scenario_field() {
    local field="$1"
    "${PYTHON}" - <<PYEOF 2>/dev/null
import json, sys
try:
    data = json.load(open('${_SCENARIO_JSON_PATH}'))
    scenarios = {k: v for k, v in data.items() if k not in ('_comment', '_schema')}
    s = scenarios.get('${SCENARIO}', {})
    val = s.get('${field}', '')
    if val:
        print(val)
except Exception:
    pass
PYEOF
}

SCENARIO_DS="$(_read_scenario_field data_source)"
SCENARIO_EM_DIR="$(_read_scenario_field em_data_dir)"
SCENARIO_TDR_DIR="$(_read_scenario_field tdr_data_dir)"
SCENARIO_COORD_FILE="$(_read_scenario_field coord_file)"
SCENARIO_COORD_FILE="${SCENARIO_COORD_FILE:-./data/SOILNET_MFC2_coordinates_sensors.csv}"

# Default scenario data_source to 'synthetic' when not declared in JSON
SCENARIO_DS="${SCENARIO_DS:-synthetic}"

# Detect conflict: CLI --data-source vs scenario JSON data_source
if [[ -n "${DATA_SOURCE}" && "${DATA_SOURCE}" != "${SCENARIO_DS}" ]]; then
    die "Data-source conflict: --data-source '${DATA_SOURCE}' was given on the
  command line, but scenario '${SCENARIO}' declares data_source='${SCENARIO_DS}'.
  Fix option A — remove --data-source and let the scenario decide.
  Fix option B — choose a scenario whose data_source matches '${DATA_SOURCE}'.
  Mixing synthetic generation with real field data is not supported."
fi

# Effective data source = CLI value if supplied, else scenario JSON
EFFECTIVE_DS="${DATA_SOURCE:-${SCENARIO_DS}}"

# Validate data source value
if [[ "${EFFECTIVE_DS}" != "synthetic" && "${EFFECTIVE_DS}" != "real" ]]; then
    die "--data-source must be 'synthetic' or 'real', got '${EFFECTIVE_DS}'"
fi

# For real mode: CLI paths override JSON paths; both must resolve
if [[ "${EFFECTIVE_DS}" == "real" ]]; then
    EFFECTIVE_EM_DIR="${EM_DIR:-${SCENARIO_EM_DIR}}"
    EFFECTIVE_TDR_DIR="${TDR_DIR:-${SCENARIO_TDR_DIR}}"
    EFFECTIVE_COORD_FILE="${COORD_FILE:-${SCENARIO_COORD_FILE}}"

    [[ -z "${EFFECTIVE_EM_DIR}" ]]  && die "Real data mode requires an EM directory.
  Set 'em_data_dir' in the scenario JSON or pass --em-dir."
    [[ -z "${EFFECTIVE_TDR_DIR}" ]] && die "Real data mode requires a TDR directory.
  Set 'tdr_data_dir' in the scenario JSON or pass --tdr-dir."
    [[ ! -d "${EFFECTIVE_EM_DIR}" ]]  && die "EM directory not found: ${EFFECTIVE_EM_DIR}"
    [[ ! -d "${EFFECTIVE_TDR_DIR}" ]] && die "TDR directory not found: ${EFFECTIVE_TDR_DIR}"
    [[ ! -f "${EFFECTIVE_COORD_FILE}" ]] && die "Coordinate file not found: ${EFFECTIVE_COORD_FILE}"
    log "Real data mode — EM=${EFFECTIVE_EM_DIR}, TDR=${EFFECTIVE_TDR_DIR}, Coords=${EFFECTIVE_COORD_FILE}"
fi

# ---------------------------------------------------------------------------
# Activate virtualenv if requested
# ---------------------------------------------------------------------------
if [[ -n "${VENV_PATH}" ]]; then
    activate="${VENV_PATH}/bin/activate"
    [[ -f "${activate}" ]] || die "No activate script found at '${activate}'"
    # shellcheck source=/dev/null
    source "${activate}"
    log "Activated virtualenv: ${VENV_PATH}"
fi

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
command -v "${PYTHON}" &>/dev/null \
    || die "'${PYTHON}' not found. Use -p to specify the interpreter."
[[ -f "${PIPELINE_SCRIPT}" ]] || die "Pipeline script not found: ${PIPELINE_SCRIPT}"
[[ -f "${SCENARIO_LOADER}" ]] || die "scenario_loader.py not found: ${SCENARIO_LOADER}"
if [[ "${DATA_SOURCE}" == "real" ]]; then
    [[ -f "${REAL_DATA_LOADER}" ]] || die "Real data loader not found: ${REAL_DATA_LOADER}"
fi

CONFIG_ARG=()
if [[ -n "${CONFIG_FILE}" ]]; then
    [[ -f "${CONFIG_FILE}" ]] || die "Config file not found: ${CONFIG_FILE}"
    CONFIG_ARG=(--config "${CONFIG_FILE}")
fi

# ---------------------------------------------------------------------------
# --list mode
# ---------------------------------------------------------------------------
if ${LIST_ONLY}; then
    log "Available scenarios in '${CONFIG_FILE:-${SCENARIO_JSON}}':"
    "${PYTHON}" "${SCENARIO_LOADER}" --list "${CONFIG_ARG[@]+"${CONFIG_ARG[@]}"}"
    exit 0
fi

# ---------------------------------------------------------------------------
# --validate mode
# ---------------------------------------------------------------------------
if ${VALIDATE_ONLY}; then
    log "Validating all scenarios in '${CONFIG_FILE:-${SCENARIO_JSON}}'..."
    "${PYTHON}" -c "
from scenario_loader import validate_all
validate_all($([ -n "${CONFIG_FILE}" ] && echo "\"${CONFIG_FILE}\"" || echo ""))
"
    exit 0
fi

# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
XY_NOTE="XZ cross-section only"
[[ "${MODE}" == "2d" ]] && XY_NOTE="XZ cross-section + XY plan-view  (grid: ${NX}×${NY})"

DATA_SOURCE_UPPER="${EFFECTIVE_DS^^}"
if [[ "${EFFECTIVE_DS}" == "real" ]]; then
    DATA_SOURCE_UPPER="REAL DATA (${EFFECTIVE_EM_DIR})"
fi

log "=================================================="
log " EA-EM Ensemble Pipeline"
log "  Data source : ${DATA_SOURCE_UPPER}"
log "  Scenario    : ${SCENARIO}"
log "  Mode        : ${MODE^^}  —  ${XY_NOTE}"
log "  Config      : ${CONFIG_FILE:-${SCENARIO_JSON} (default)}"
log "  Drop coils  : ${DROP_COILS[*]:-(none)}"
log "  Homogeneous : ${HOMOGENEOUS}"
log "  Python      : $(command -v "${PYTHON}") ($(${PYTHON} --version 2>&1))"
log "  Script      : ${PIPELINE_SCRIPT}"
log "=================================================="

# ---------------------------------------------------------------------------
# Build the argument list for run_pipeline.py.
# --nx / --ny are forwarded so run_pipeline can pass them to
# generate_time_lapse_dataset(); they are silently ignored by
# run_pipeline.py in 1-D mode.
#
# Real data args are only forwarded when --data-source real is active.
# ---------------------------------------------------------------------------
REAL_DATA_ARGS=()
if [[ "${EFFECTIVE_DS}" == "real" ]]; then
    REAL_DATA_ARGS=(
        --data-source "real"
        --em-dir "${EFFECTIVE_EM_DIR}"
        --tdr-dir "${EFFECTIVE_TDR_DIR}"
        --coord-file "${EFFECTIVE_COORD_FILE}"
    )
fi

DROP_COILS_ARGS=()
if [[ ${#DROP_COILS[@]} -gt 0 ]]; then
    DROP_COILS_ARGS=(--drop-coils "${DROP_COILS[@]}")
fi

HOMOGENEOUS_ARGS=()
if ${HOMOGENEOUS}; then
    HOMOGENEOUS_ARGS=(--homogeneous)
fi

GRADIENT_ARGS=()
if ${FULL_RANGE_GRADIENT}; then
    GRADIENT_ARGS+=(--full-range-gradient)
fi
if [[ -n "${XY_Z_BLEND}" ]]; then
    GRADIENT_ARGS+=(--xy-z-blend "${XY_Z_BLEND}")
fi
if ${INVERT_Z_GRADIENT}; then
    GRADIENT_ARGS+=(--invert-z-gradient)
fi

SWC_OVERRIDE_ARGS=()
if [[ -n "${NUM_SWC_POSITIONS}" ]]; then
    SWC_OVERRIDE_ARGS+=(--num-swc-positions "${NUM_SWC_POSITIONS}")
fi
if [[ ${#SWC_DEPTHS[@]} -gt 0 ]]; then
    SWC_OVERRIDE_ARGS+=(--swc-depths "${SWC_DEPTHS[@]}")
fi
if [[ -n "${WINDOW_M}" ]]; then
    SWC_OVERRIDE_ARGS+=(--window-m "${WINDOW_M}")
fi
if [[ -n "${PROBE_XYZ}" ]]; then
    SWC_OVERRIDE_ARGS+=(--probe-xyz "${PROBE_XYZ}")
fi

PLOT_ARGS=()
if [[ -n "${XY_RENDER_MODE}" ]]; then
    PLOT_ARGS+=(--xy-render-mode "${XY_RENDER_MODE}")
fi

INVERSION_OVERRIDE_ARGS=()
if [[ ${#BETA[@]} -gt 0 ]]; then
    INVERSION_OVERRIDE_ARGS+=(--beta "${BETA[@]}")
fi
if [[ -n "${GAMMA}" ]]; then
    INVERSION_OVERRIDE_ARGS+=(--gamma "${GAMMA}")
fi

# ---------------------------------------------------------------------------
# Shell-level report header (written to a temp file for now).
#
# run_pipeline.py (via scenario_loader) is the sole authority on where this
# run's output actually lands -- it now numbers each run's own folder as
# "<seq>_<scenario>" under the scenario's configured output_dir. Rather
# than re-deriving/guessing that path here (which can silently drift out
# of sync -- see next_run_output_dir()'s docstring), we let Python run
# first, capture the "[OUTPUT_DIR] <path>" line it prints, and only then
# write this header (plus the captured console log) into the real folder.
# ---------------------------------------------------------------------------
HEADER_TMP="$(mktemp)"
trap 'rm -f "${HEADER_TMP}" "${RUN_LOG_TMP:-}"' EXIT

cat > "${HEADER_TMP}" << MDEOF
# EA-EM Pipeline Run — Shell Launch Summary

_Companion to this run's main \`pipeline_report_*.md\` (written by
run_pipeline.py itself) -- this file just records how it was launched._

| Field | Value |
|-------|-------|
| Generated | $(date '+%Y-%m-%d %H:%M:%S') |
| Launched by | \`run_pipeline.sh\` |
| Data source | \`${EFFECTIVE_DS^^}\` |
| Scenario | \`${SCENARIO}\` |
| Mode | \`${MODE^^}\` |
| Config file | \`${CONFIG_FILE:-${SCENARIO_JSON} (default)}\` |
| Python | \`$(${PYTHON} --version 2>&1)\` |
| Python path | \`$(command -v "${PYTHON}")\` |
| Script | \`${PIPELINE_SCRIPT}\` |
| Host | \`$(hostname)\` |
| User | \`$(whoami)\` |
| Git commit | \`$(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo "n/a")\` |
| Grid (2D) | \`${NX}×${NY}\` |

MDEOF

if [[ "${EFFECTIVE_DS}" == "real" ]]; then
    cat >> "${HEADER_TMP}" << MDEOF
## Real Data Sources

| Path | Value |
|------|-------|
| EM directory | \`${EFFECTIVE_EM_DIR}\` |
| TDR directory | \`${EFFECTIVE_TDR_DIR}\` |
| Coordinates file | \`${EFFECTIVE_COORD_FILE}\` |

MDEOF
fi

cat >> "${HEADER_TMP}" << MDEOF

## Shell Launch Arguments

| Argument | Effective Value |
|----------|----------------|
| data-source | \`${EFFECTIVE_DS}\` |
| scenario | \`${SCENARIO}\` |
| mode | \`${MODE}\` |
| nx | \`${NX}\` |
| ny | \`${NY}\` |
| config | \`${CONFIG_FILE:-default}\` |
| drop-coils | \`${DROP_COILS[*]:-none}\` |
| homogeneous | \`${HOMOGENEOUS}\` |
| num-swc-positions | \`${NUM_SWC_POSITIONS:-scenario default}\` |
| swc-depths | \`${SWC_DEPTHS[*]:-scenario default}\` |
| window-m | \`${WINDOW_M:-scenario default}\` |
| beta | \`${BETA[*]:-scenario default}\` |
| gamma | \`${GAMMA:-scenario default}\` |
| env | \`${VENV_PATH:-none}\` |
| python | \`${PYTHON}\` |

---

MDEOF

# ---------------------------------------------------------------------------
# Run the pipeline, streaming output live while also capturing it (so we
# can pull the "[OUTPUT_DIR] <path>" line out afterwards).
# ---------------------------------------------------------------------------
RUN_LOG_TMP="$(mktemp)"

set +e
"${PYTHON}" "${PIPELINE_SCRIPT}" \
    --data-source "${EFFECTIVE_DS}" \
    --scenario "${SCENARIO}" \
    --mode     "${MODE}"     \
    "${REAL_DATA_ARGS[@]+"${REAL_DATA_ARGS[@]}"}" \
    "${DROP_COILS_ARGS[@]+"${DROP_COILS_ARGS[@]}"}" \
    "${HOMOGENEOUS_ARGS[@]+"${HOMOGENEOUS_ARGS[@]}"}" \
    "${GRADIENT_ARGS[@]+"${GRADIENT_ARGS[@]}"}" \
    "${SWC_OVERRIDE_ARGS[@]+"${SWC_OVERRIDE_ARGS[@]}"}" \
    "${PLOT_ARGS[@]+"${PLOT_ARGS[@]}"}" \
    "${INVERSION_OVERRIDE_ARGS[@]+"${INVERSION_OVERRIDE_ARGS[@]}"}" \
    "${CONFIG_ARG[@]+"${CONFIG_ARG[@]}"}" \
    2>&1 | tee "${RUN_LOG_TMP}"
EXIT_CODE=${PIPESTATUS[0]}
set -e

# ---------------------------------------------------------------------------
# Figure out where this run's output actually landed, and finalise the
# shell-level report there.
# ---------------------------------------------------------------------------
OUTPUT_DIR="$(grep -m1 '^\[OUTPUT_DIR\] ' "${RUN_LOG_TMP}" | sed -E 's/^\[OUTPUT_DIR\] //')"
if [[ -z "${OUTPUT_DIR}" || ! -d "${OUTPUT_DIR}" ]]; then
    err "Could not determine this run's output_dir from the pipeline's console output (crashed before it was announced?). Saving the shell-level report/log under 'mfc2_outputs/' instead."
    OUTPUT_DIR="mfc2_outputs"
    mkdir -p "${OUTPUT_DIR}"
fi

REPORT_TS="$(date '+%Y%m%d_%H%M%S')"
REPORT_PATH="${OUTPUT_DIR}/pipeline_report_shell_${REPORT_TS}.md"
mv "${HEADER_TMP}" "${REPORT_PATH}"
trap - EXIT

{
    echo "## Full Console Output"
    echo ""
    echo '```'
    cat "${RUN_LOG_TMP}"
    echo '```'
    echo ""
    echo "## Shell Exit Status"
    echo ""
    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "**✅ SUCCESS** — exited with code \`0\` at $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo "**❌ FAILED** — exited with code \`${EXIT_CODE}\` at $(date '+%Y-%m-%d %H:%M:%S')"
    fi
    echo ""
} >> "${REPORT_PATH}"
rm -f "${RUN_LOG_TMP}"

log "=================================================="
if [[ ${EXIT_CODE} -eq 0 ]]; then
    log " Pipeline completed successfully  (data: ${EFFECTIVE_DS}, scenario: ${SCENARIO}, mode: ${MODE^^})"
else
    err " Pipeline exited with code ${EXIT_CODE}  (data: ${EFFECTIVE_DS}, scenario: ${SCENARIO}, mode: ${MODE^^})"
fi
log " Output dir → ${OUTPUT_DIR}"
log " Shell launch summary → ${REPORT_PATH}"
log " (see ${OUTPUT_DIR}/pipeline_report_*.md for the main pipeline report)"
log "=================================================="

exit ${EXIT_CODE}