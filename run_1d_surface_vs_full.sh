#!/usr/bin/env bash
# =============================================================================
# run_1d_surface_vs_full.sh -- surface-only vs full-depth TDR on a 1-D transect,
# with 1..4 free Waxman-Smits parameters in calibration.
#
#   1d_surface_free{K} : probes only at 0.2 m (horizon 0 calibrated)
#   1d_full_free{K}    : probes in all 4 horizons (reference)
#   K = number of FREE WS params; order freed: 1=sigma_s, 2=+m, 3=+sigma_w, 4=+a
#   (truth m and sigma_s vary along x; a and sigma_w are constant in the truth)
#
# Usage:  ./run_1d_surface_vs_full.sh [NPOS]     (default NPOS=5 probe positions)
#         SKIP_RUNS=1 ./run_1d_surface_vs_full.sh  -> only redo the comparisons
# Comparisons use the LATEST run of each scenario in the shared run log.
# =============================================================================
set -euo pipefail

NPOS="${1:-5}"
POSITION=(8 0)                       # truth point for depth-profile plots (x y)
LOG="mfc2_outputs/run_log.csv"

latest_id() {   # latest run_id of a scenario in the run log (cols: run_id,...,scenario)
    awk -F, -v s="$1" 'NR>1 && $3==s {id=$1} END{print id}' "${LOG}"
}

if [[ "${SKIP_RUNS:-0}" != "1" ]]; then
    for K in 1 2 3 4; do
        ./run_pipeline.sh -s "1d_surface_free${K}" --mode 1d --num-swc-positions "${NPOS}"
        ./run_pipeline.sh -s "1d_full_free${K}"    --mode 1d --num-swc-positions "${NPOS}"
    done
fi

# --- one comparison per K: surface-only vs full-depth -------------------------
for K in 1 2 3 4; do
    S="$(latest_id "1d_surface_free${K}")"; F="$(latest_id "1d_full_free${K}")"
    [[ -n "${S}" && -n "${F}" ]] || { echo "Missing run for K=${K} (see ${LOG})" >&2; continue; }
    ./compare_scenarios.sh --ids "${S}" "${F}" \
        --labels "surface_free${K}" "full_free${K}" \
        --position "${POSITION[@]}" --out-dir "./comparison_output/1d_surface_vs_full_free${K}"
done

# --- all 8 runs together ------------------------------------------------------
IDS=(); LABELS=()
for K in 1 2 3 4; do
    for T in surface full; do
        id="$(latest_id "1d_${T}_free${K}")"
        [[ -n "${id}" ]] && { IDS+=("${id}"); LABELS+=("${T}_free${K}"); }
    done
done
./compare_scenarios.sh --ids "${IDS[@]}" --labels "${LABELS[@]}" \
    --position "${POSITION[@]}" --out-dir ./comparison_output/1d_surface_vs_full_all
