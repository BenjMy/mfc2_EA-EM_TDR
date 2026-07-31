#!/usr/bin/env bash
# =============================================================================
# compare_scenarios.sh — EA-EM scenario comparison launcher
# =============================================================================
# Thin wrapper around compare_scenarios.py -- same role for comparisons as
# run_pipeline.sh plays for a single pipeline run: sane defaults, --help,
# and a virtualenv switch, without needing to remember Python flag names.
#
# Usage:
#   ./compare_scenarios.sh --ids <run_number> <run_number> ...  # by run number (recommended)
#   ./compare_scenarios.sh --reports <glob or path> ...         # by report path/glob
#   ./compare_scenarios.sh --ids <run_number> --reports <glob>  # combine both
#   ./compare_scenarios.sh --list                                # show recent runs from the log
#   ./compare_scenarios.sh --help                                # show this message
#
# A run number is the <seq> in each run's output folder, e.g. 7 for
# mfc2_outputs/0007_field_survey_2d/. --ids looks it up in the shared
# run-log CSV written by run_pipeline.py (mfc2_outputs/run_log.csv by
# default -- one row per run, of every scenario). --list prints the tail
# of that file (including the output_dir column, where the number lives)
# so you can copy it straight out. A full run_id string still works too,
# and the two styles can be mixed.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
IDS=()
REPORTS=()
LABELS=()
SCORING=()
OUTPUT_DIRS=()
POSITION=()
TRANSECT_Y=""
TRANSECT_TOL="1.0"
REALIZATION="0"
TIME_INDEX="-1"
OUT_DIR="./comparison_output"
LOG_FILE="mfc2_outputs/run_log.csv"
VENV_PATH=""
PYTHON="python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_SCRIPT="${SCRIPT_DIR}/compare_scenarios.py"
LIST_ONLY=false

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF

Usage: $(basename "$0") [OPTIONS]

Selecting runs to compare (give --ids and/or --reports; at least one required):
      --ids ID [ID ...]   One or more runs to compare, from the shared
                          run-log CSV (see --log-file). Each is resolved
                          to its pipeline_report_*.md automatically -- no
                          need to know paths or filenames. Give the plain
                          run number (e.g. 7 or 0007 -- the <seq> in
                          .../<seq>_<scenario>/); if each scenario writes
                          to its own base output dir this number can
                          repeat across scenarios, so use
                          'scenario:number' (e.g. field_survey_2d:7) to
                          pin one, or pass the full run_id string. Use
                          --list to see available run numbers/scenarios.
      --reports PATH [PATH ...]
                          Paths or glob patterns to pipeline_report_*.md
                          files, one entry per scenario (quote globs so
                          this script expands them, not your shell).
                          Combine with --ids: --ids entries come first.
      --log-file PATH     Path to the run-log CSV written by
                          run_pipeline.py (default: ${LOG_FILE}).
                          Only used when --ids is given.

Labelling and scoring:
      --labels LABEL [LABEL ...]
                          Optional scenario labels, same order/count as the
                          combined --ids + --reports list. Defaults to the
                          run-log's scenario column (--ids entries) or the
                          Scenario field parsed from each report.
      --scoring PATH [PATH ...]
                          Optional scoring_summary.csv files from
                          score_against_truth.py, same order/count as the
                          combined --ids + --reports list.

Point-level truth comparison (needs each scenario's output_model/ --
theta_calc.npy via capture_synthetic_theta.py, plus the synthetic_*.npy
truth arrays 0_generate_synthetic_data.py already writes there):
      --position X Y      Enables depth-profile / spatial-transect /
                          scatter-residual comparison plots against
                          truth, at the grid point nearest (X, Y).
      --transect-y Y      y-value for the spatial transect plot.
                          Default: median y of the first scenario.
      --transect-tol M    Tolerance [m] around --transect-y (default: ${TRANSECT_TOL}).
      --realization N     Ensemble truth realization index -- must match
                          each scenario's config 'realization_index'
                          (default: ${REALIZATION}).
      --time-index N      Time/epoch index for these plots, -1 = last
                          (default: ${TIME_INDEX}).
      --output-dirs PATH [PATH ...]
                          Optional override: pipeline output_dir per
                          scenario, same order/count as the combined
                          --ids + --reports list. Defaults to the run-log's
                          output_dir column (--ids) or each report's own
                          directory (--reports).

Output:
      --out-dir PATH      Directory to write comparison_table.csv/.md and
                          the comparison PNGs into (default: ${OUT_DIR}).

Environment options:
  -e, --env PATH          Path to a Python virtualenv to activate before
                          running (e.g. /path/to/venv or
                          /path/to/conda/envs/myenv)
  -p, --python BIN        Python interpreter to use (default: python3)

Utility:
      --list              Print the tail of the run-log CSV -- including
                          output_dir, whose folder name starts with the
                          run number -- so you can copy one, then exit.
                          Respects --log-file.
  -h, --help              Show this help message and exit

Examples:

  # --- Compare two runs by run number (the usual case) ---
  ./compare_scenarios.sh --ids 7 12

  # --- See which run numbers/scenarios are available first ---
  ./compare_scenarios.sh --list

  # --- Same run number in two scenarios (each scenario has its own
  #     output dir, so its <seq> counter starts over) -- scope with
  #     scenario:number to disambiguate ---
  ./compare_scenarios.sh --ids field_survey_2d:1 field_survey_2d_dense_tdr:1

  # --- Custom labels and output directory ---
  ./compare_scenarios.sh --ids 7 12 \\
      --labels baseline dense_tdr \\
      --out-dir ./comparison_baseline_vs_dense_tdr

  # --- By report path/glob instead of run number ---
  # (each run lives in its own <seq>_<scenario>/ folder -- e.g.
  # 0001_field_survey_2d/ -- so a glob is still the easiest way to
  # sweep every run of every scenario at once; --ids remains the most
  # robust way to pin an *exact* run)
  ./compare_scenarios.sh --reports "mfc2_outputs/*/pipeline_report_*.md"

  # --- Mix a run number and a glob ---
  ./compare_scenarios.sh --ids 7 \\
      --reports "mfc2_outputs/*_moisture_2d_free/pipeline_report_*.md"

  # --- Fold in truth-scoring CSVs from score_against_truth.py ---
  ./compare_scenarios.sh --ids 7 12 \\
      --scoring scoring_A.csv scoring_B.csv

  # --- Point-level truth comparison (depth profile / transect / scatter) ---
  ./compare_scenarios.sh --ids 7 12 \\
      --position 20 10 --transect-y 10

  # --- Custom virtualenv ---
  ./compare_scenarios.sh --ids 7 12 -e ~/venvs/eaem

  # --- A full run_id string still works too, and can be mixed in ---
  ./compare_scenarios.sh --ids 7 field_survey_2d_dense_tdr_20260731_091500

EOF
}

log() { echo "[$(date '+%H:%M:%S')] $*"; }
err() { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; }
die() { err "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ids)
            [[ $# -ge 2 ]] || die "--ids requires at least one run_id value"
            shift # consume the flag itself
            IDS=()
            # Greedily consume one-or-more following tokens, stopping at the
            # next recognised flag (mirrors Python argparse's nargs='+').
            while [[ $# -gt 0 && "$1" != -* ]]; do
                IDS+=("$1")
                shift
            done
            [[ ${#IDS[@]} -ge 1 ]] || die "--ids requires at least one run_id value"
            ;;
        --reports)
            [[ $# -ge 2 ]] || die "--reports requires at least one path/glob"
            shift
            REPORTS=()
            while [[ $# -gt 0 && "$1" != -* ]]; do
                REPORTS+=("$1")
                shift
            done
            [[ ${#REPORTS[@]} -ge 1 ]] || die "--reports requires at least one path/glob"
            ;;
        --labels)
            [[ $# -ge 2 ]] || die "--labels requires at least one value"
            shift
            LABELS=()
            while [[ $# -gt 0 && "$1" != -* ]]; do
                LABELS+=("$1")
                shift
            done
            [[ ${#LABELS[@]} -ge 1 ]] || die "--labels requires at least one value"
            ;;
        --scoring)
            [[ $# -ge 2 ]] || die "--scoring requires at least one path"
            shift
            SCORING=()
            while [[ $# -gt 0 && "$1" != -* ]]; do
                SCORING+=("$1")
                shift
            done
            [[ ${#SCORING[@]} -ge 1 ]] || die "--scoring requires at least one path"
            ;;
        --output-dirs)
            [[ $# -ge 2 ]] || die "--output-dirs requires at least one path"
            shift
            OUTPUT_DIRS=()
            while [[ $# -gt 0 && "$1" != -* ]]; do
                OUTPUT_DIRS+=("$1")
                shift
            done
            [[ ${#OUTPUT_DIRS[@]} -ge 1 ]] || die "--output-dirs requires at least one path"
            ;;
        --position)
            [[ $# -ge 3 ]] || die "--position requires two values: X Y"
            POSITION=("$2" "$3")
            shift 3
            ;;
        --transect-y)    TRANSECT_Y="${2:?'--transect-y requires a value'}";   shift 2 ;;
        --transect-tol)  TRANSECT_TOL="${2:?'--transect-tol requires a value'}"; shift 2 ;;
        --realization)   REALIZATION="${2:?'--realization requires a value'}"; shift 2 ;;
        --time-index)    TIME_INDEX="${2:?'--time-index requires a value'}";   shift 2 ;;
        --log-file)  LOG_FILE="${2:?'--log-file requires a path'}"; shift 2 ;;
        --out-dir)   OUT_DIR="${2:?'--out-dir requires a path'}";   shift 2 ;;
        -e|--env)    VENV_PATH="${2:?'--env requires a value'}";    shift 2 ;;
        -p|--python) PYTHON="${2:?'--python requires a value'}";   shift 2 ;;
        --list)      LIST_ONLY=true; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) die "Unknown option: $1  (run with --help for usage)" ;;
    esac
done

# ---------------------------------------------------------------------------
# --list: show recent run_ids from the log, then exit
# ---------------------------------------------------------------------------
if ${LIST_ONLY}; then
    [[ -f "${LOG_FILE}" ]] || die "Run-log CSV not found: ${LOG_FILE}  (pass --log-file, or run run_pipeline.py at least once first)"
    log "Recent runs in ${LOG_FILE}:"
    # Portable CSV -> tab-separated preview (avoids depending on the
    # optional 'column' utility). Shows header + last 20 data rows.
    # (head/tail each read the file directly -- piping one into the
    # other off a single stream can silently drop rows to head's
    # internal read buffer.)
    { head -n 1 "${LOG_FILE}"; tail -n +2 "${LOG_FILE}" | tail -n 20; } | awk -F, 'BEGIN{OFS="\t"} {$1=$1; print}'
    exit 0
fi

if [[ ${#IDS[@]} -eq 0 && ${#REPORTS[@]} -eq 0 ]]; then
    die "At least one of --ids or --reports is required  (run with --help for usage, or --list to see available run_ids)"
fi

# ---------------------------------------------------------------------------
# Activate virtualenv if requested
# ---------------------------------------------------------------------------
if [[ -n "${VENV_PATH}" ]]; then
    activate="${VENV_PATH}/bin/activate"
    [[ -f "${activate}" ]] || die "No activate script found at '${activate}'"
    # shellcheck disable=SC1090
    source "${activate}"
    log "Activated virtualenv: ${VENV_PATH}"
fi

[[ -f "${COMPARE_SCRIPT}" ]] || die "compare_scenarios.py not found at ${COMPARE_SCRIPT}"

# ---------------------------------------------------------------------------
# Build the argument list for compare_scenarios.py
# ---------------------------------------------------------------------------
IDS_ARGS=()
[[ ${#IDS[@]} -gt 0 ]] && IDS_ARGS=(--ids "${IDS[@]}")

REPORTS_ARGS=()
[[ ${#REPORTS[@]} -gt 0 ]] && REPORTS_ARGS=(--reports "${REPORTS[@]}")

LABELS_ARGS=()
[[ ${#LABELS[@]} -gt 0 ]] && LABELS_ARGS=(--labels "${LABELS[@]}")

SCORING_ARGS=()
[[ ${#SCORING[@]} -gt 0 ]] && SCORING_ARGS=(--scoring "${SCORING[@]}")

OUTPUT_DIRS_ARGS=()
[[ ${#OUTPUT_DIRS[@]} -gt 0 ]] && OUTPUT_DIRS_ARGS=(--output-dirs "${OUTPUT_DIRS[@]}")

POSITION_ARGS=()
[[ ${#POSITION[@]} -gt 0 ]] && POSITION_ARGS=(--position "${POSITION[@]}")

TRANSECT_Y_ARGS=()
[[ -n "${TRANSECT_Y}" ]] && TRANSECT_Y_ARGS=(--transect-y "${TRANSECT_Y}")

log "Comparing runs → out-dir: ${OUT_DIR}"
[[ ${#IDS[@]} -gt 0 ]] && log "  by run_id : ${IDS[*]}"
[[ ${#REPORTS[@]} -gt 0 ]] && log "  by report : ${REPORTS[*]}"
[[ ${#POSITION[@]} -gt 0 ]] && log "  truth comparison @ position: ${POSITION[*]}"

"${PYTHON}" "${COMPARE_SCRIPT}" \
    "${IDS_ARGS[@]+"${IDS_ARGS[@]}"}" \
    "${REPORTS_ARGS[@]+"${REPORTS_ARGS[@]}"}" \
    --log-file "${LOG_FILE}" \
    "${LABELS_ARGS[@]+"${LABELS_ARGS[@]}"}" \
    "${SCORING_ARGS[@]+"${SCORING_ARGS[@]}"}" \
    "${OUTPUT_DIRS_ARGS[@]+"${OUTPUT_DIRS_ARGS[@]}"}" \
    "${POSITION_ARGS[@]+"${POSITION_ARGS[@]}"}" \
    "${TRANSECT_Y_ARGS[@]+"${TRANSECT_Y_ARGS[@]}"}" \
    --transect-tol "${TRANSECT_TOL}" \
    --realization "${REALIZATION}" \
    --time-index "${TIME_INDEX}" \
    --out-dir "${OUT_DIR}"

EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
    log "Comparison complete → ${OUT_DIR}"
else
    err "compare_scenarios.py exited with code ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
