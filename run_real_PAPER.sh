#!/usr/bin/env bash
# =============================================================================
# run_real_PAPER.sh — EA-EM Ensemble Pipeline launcher (REAL MFC2 field data)
# =============================================================================
#
# Companion to run_synthetic_PAPER.sh, but for the real-data scenarios in
# ensemble_scenarios_real.json (14 scenarios, see that file's header for the
# full naming scheme). Every call below passes -c ensemble_scenarios_real.json
# so run_pipeline.sh reads from that file instead of the default
# ensemble_scenarios.json.
#
# AXES COVERED AND HOW THEY'RE SET
# -----------------------------------------------------------------------
#   forward model (FSlin+CS / FSlin only / CS only)   -> scenario name
#                                                          (_base / _fslin / _cs)
#   depth-varying (fixed vs free interfaces)            -> scenario name
#                                                          (_base / _depth_free)
#   ws_fixed_params (all fixed / free sigma_w / free    -> scenario name
#     sigma_s / free sigma_w & sigma_s)                    (_ws_* / _base)
#   layer geometry (single interface / multi-layer)     -> scenario name
#                                                          (real_single_* / real_multi_*)
#   beta (lateral smoothing)                             -> CLI --beta
#   gamma (temporal smoothing, only matters if the real  -> CLI --gamma
#     survey has multiple epochs / is time-lapse)
#   train/validation split                               -> CLI --train-val-split
#
# NOTE on ws_localized_bandwidth_sweep:
#   EVERY scenario in ensemble_scenarios_real.json sets
#   ws_localized_bandwidth_sweep = [0.5, 1, 3, 5, 8, 12, 20], so the
#   leave-one-probe-out bandwidth CV + localized Step-2 diagnostic run
#   AUTOMATICALLY on every single call below -- no extra flag needed
#   (same convention as '2d'/'2d_intermediate'/'2d_dense_tdr' in the
#   synthetic scenarios file). To try DIFFERENT candidate bandwidths for a
#   one-off run without editing the JSON, pass
#   --ws-localized-bandwidth-sweep <c1> <c2> ... on the CLI -- it overrides
#   the JSON value for that run only (see section 8 below for an example).
#
# NOTE: --em-footprint-diagnostic, --probe-xyz, --mode/--nx/--ny work exactly
# as in run_synthetic_PAPER.sh; see that script's header for details.
#
# NOTE: adjust REAL_CFG below if you rename/move ensemble_scenarios_real.json.
#
# SKIPPING FREQUENCIES (noisy GEM-2 channels)
# -----------------------------------------------------------------------
#   The default lives in ensemble_scenarios_real.json: every scenario there
#   carries  "drop_coils": [425]  (single source of truth), so a plain
#       ./run_real_PAPER.sh
#   drops 425 Hz everywhere. To change it for ALL scenarios, edit the JSON.
#
#   One-off override from the command line (replaces, does not add to, the
#   JSON value for this whole run; forwarded as --drop-coils):
#       ./run_real_PAPER.sh 425 5325
#   Each value may be a frequency in Hz (425), a coil-name substring (f425),
#   or the exact emagpy coil name (HCP1.66f425h1_quad).
#   The inversion cache key includes drop_coils, so changing the set never
#   reuses results computed with a different set of channels.
# =============================================================================

set -euo pipefail

REAL_CFG="ensemble_scenarios_real.json"

# Optional CLI override of the scenario files' drop_coils (see header).
DROP_ARGS=()
if [ "$#" -gt 0 ]; then
    DROP_ARGS=(--drop-coils "$@")
    echo ">>> CLI override: dropping coils/frequencies: $*"
else
    echo ">>> drop_coils taken from $REAL_CFG (per-scenario)"
fi

# Wrapper: every inversion call goes through here so the optional CLI
# --drop-coils override is applied uniformly (the empty-array expansion is
# safe under 'set -u' on old bash).
run_pipeline() {
    ./run_pipeline.sh "$@" ${DROP_ARGS[@]+"${DROP_ARGS[@]}"}
}

# --- Sanity check: list + validate the real scenarios before doing anything --
./run_pipeline.sh --list     --config "$REAL_CFG"
./run_pipeline.sh --validate --config "$REAL_CFG"


# =============================================================================
# 1) BASELINE RUNS — one per scenario, no sweeps, just to confirm each of the
#    14 axis combinations runs cleanly end-to-end.
# =============================================================================
for s in real_single_base real_single_fslin real_single_cs real_single_depth_free \
         real_single_ws_all_fixed real_single_ws_free_sigmaw real_single_ws_free_sigmas \
         real_multi_base real_multi_fslin real_multi_cs real_multi_depth_free \
         real_multi_ws_all_fixed real_multi_ws_free_sigmaw real_multi_ws_free_sigmas
do
    run_pipeline -s "$s" -c "$REAL_CFG" --mode 2d --em-footprint-diagnostic
done

./compare_scenarios.sh --list
# Copy the real run numbers from --list above, then e.g.:
# ./compare_scenarios.sh --ids real_single_base:1 real_multi_base:1 --position 5 0


# =============================================================================
# 2) FORWARD MODEL COMPARISON (single vs single vs both) — 'single' layer group
# =============================================================================
run_pipeline -s real_single_base  -c "$REAL_CFG" --mode 2d
run_pipeline -s real_single_fslin -c "$REAL_CFG" --mode 2d
run_pipeline -s real_single_cs    -c "$REAL_CFG" --mode 2d

./compare_scenarios.sh --list
# ./compare_scenarios.sh --ids real_single_base:2 real_single_fslin:1 real_single_cs:1 --position 5 0

# ... and the 'multi' layer group
run_pipeline -s real_multi_base  -c "$REAL_CFG" --mode 2d
run_pipeline -s real_multi_fslin -c "$REAL_CFG" --mode 2d
run_pipeline -s real_multi_cs    -c "$REAL_CFG" --mode 2d

./compare_scenarios.sh --list


# =============================================================================
# 3) DEPTH-VARYING (fixed vs free interfaces) — both layer groups
# =============================================================================
run_pipeline -s real_single_base       -c "$REAL_CFG" --mode 2d
run_pipeline -s real_single_depth_free -c "$REAL_CFG" --mode 2d

run_pipeline -s real_multi_base        -c "$REAL_CFG" --mode 2d
run_pipeline -s real_multi_depth_free  -c "$REAL_CFG" --mode 2d

./compare_scenarios.sh --list


# =============================================================================
# 4) WS_FIXED_PARAMS SWEEP (all fixed / free sigma_w / free sigma_s /
#    free sigma_w & sigma_s) — both layer groups
# =============================================================================
for s in real_single_ws_all_fixed real_single_ws_free_sigmaw \
         real_single_ws_free_sigmas real_single_base
do
    run_pipeline -s "$s" -c "$REAL_CFG" --mode 2d
done

for s in real_multi_ws_all_fixed real_multi_ws_free_sigmaw \
         real_multi_ws_free_sigmas real_multi_base
do
    run_pipeline -s "$s" -c "$REAL_CFG" --mode 2d
done

./compare_scenarios.sh --list


# =============================================================================
# 5) BETA SWEEP (lateral/within-epoch smoothing) — space-separated, one call
# =============================================================================
run_pipeline -s real_single_base -c "$REAL_CFG" --mode 2d --beta 0 0.5 1
run_pipeline -s real_multi_base  -c "$REAL_CFG" --mode 2d --beta 0 0.5 1

./compare_scenarios.sh --list


# =============================================================================
# 6) GAMMA SWEEP (temporal smoothing — only affects time-lapse / multi-epoch
#    real surveys; no-op if your real dataset here is a single EM survey date)
# =============================================================================
run_pipeline -s real_single_base -c "$REAL_CFG" --mode 2d --gamma 0 10
run_pipeline -s real_multi_base  -c "$REAL_CFG" --mode 2d --gamma 0 10

./compare_scenarios.sh --list


# =============================================================================
# 7) COMBINED BETA + GAMMA SWEEP
#    len(beta) x len(gamma) x ... ensemble tasks -- not a sum.
# =============================================================================
#run_pipeline -s real_single_base -c "$REAL_CFG" --mode 2d --beta 0 0.5 1 --gamma 0 10
#run_pipeline -s real_multi_base  -c "$REAL_CFG" --mode 2d --beta 0 0.5 1 --gamma 0 10


# =============================================================================
# 8) WS-LOCALIZED-BANDWIDTH SWEEP — already runs automatically on every call
#    above (all 14 scenarios set ws_localized_bandwidth_sweep in the JSON).
#    This block is only for testing DIFFERENT candidate bandwidths than the
#    JSON default, ad hoc, without editing ensemble_scenarios_real.json --
#    the CLI flag overrides the JSON value for that run only.
# =============================================================================
run_pipeline -s real_single_base -c "$REAL_CFG" --mode 2d \
    --ws-localized-bandwidth-sweep 1 2 5 10 15 20 --em-footprint-diagnostic

run_pipeline -s real_multi_base -c "$REAL_CFG" --mode 2d \
    --ws-localized-bandwidth-sweep 1 2 5 10 15 20 --em-footprint-diagnostic

./compare_scenarios.sh --list


# =============================================================================
# 9) TRAIN/VAL SPLIT — real data supports 'by_probe' in addition to the
#    generic 'alternate_sorted' / 'by_time_block' / 'random' methods.
# =============================================================================
#run_pipeline -s real_single_base -c "$REAL_CFG" --mode 2d --train-val-split by_probe
#run_pipeline -s real_multi_base  -c "$REAL_CFG" --mode 2d --train-val-split by_probe

#./compare_scenarios.sh --list
