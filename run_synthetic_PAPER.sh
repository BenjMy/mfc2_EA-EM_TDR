#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — EA-EM Ensemble Pipeline launcher (Synthetic & Real Data)
# =============================================================================
#
# NOTE on things you CANNOT sweep from the command line:
#   - (nothing left in this category -- ws_localized_bandwidth /
#     ws_localized_bandwidth_sweep now also have CLI flags, see below.)
#
# NOTE on ws_localized_bandwidth / ws_localized_bandwidth_sweep:
#   Two ways to set these, CLI wins if both are given for a run:
#     - JSON (persistent, tied to the scenario): already set for '2d',
#       '2d_intermediate', and '2d_dense_tdr' in ensemble_scenarios.json
#       (the '2d_intermediate' typo'd key -- ws_localized_bandwidth_sweep_NULL
#       -- has been fixed), so the sweep runs automatically on every call
#       below for those three scenarios -- no extra block needed.
#     - CLI (ad hoc, no JSON edit): --ws-localized-bandwidth <c> or
#       --ws-localized-bandwidth-sweep <c1> <c2> ..., e.g.
#         ./run_pipeline.sh -s 2d --ws-localized-bandwidth-sweep 1 2 5 10 15 20
#       useful for scenarios that don't set either field in the JSON, or
#       to try different candidate bandwidths without editing the JSON.
#   - gamma only accepts ONE value per run_pipeline.sh call (it's a single
#     float, not a list) -- sweeping it means one call per value.
#   - beta DOES accept multiple values, but space-separated, no brackets/
#     commas: --beta 0 0.5 1
#
# NOTE on run numbers used below with compare_scenarios.sh:
#   '2d' and '2d_intermediate' currently share the same output_dir in
#   ensemble_scenarios.json (mfc2_outputs/field_survey_2d), so their <seq>
#   counters interleave and a bare number like "1" is ambiguous between
#   them. Use `scenario:number` (or --list) to disambiguate until that's
#   fixed in the JSON. Numbers below are illustrative -- run --list after
#   each block and copy the real ones.
# =============================================================================

set -euo pipefail

# --- Baseline runs -----------------------------------------------------------
./run_pipeline.sh -s 2d
./run_pipeline.sh -s 2d_intermediate
./run_pipeline.sh -s 2d_dense_tdr

./compare_scenarios.sh --list   # copy the 3 run numbers just produced, e.g.:
./compare_scenarios.sh --ids 2d:1 2d_intermediate:1 2d_dense_tdr:1 --position 0 0 
./compare_scenarios.sh --ids 2d:1 2d_intermediate:1 2d_dense_tdr:1 --position 5 0

# --- Localized-bandwidth sweep diagnostic ------------------------------------
# Already runs automatically above for '2d', '2d_intermediate', and
# '2d_dense_tdr' because ws_localized_bandwidth_sweep is set in the JSON for
# all three. To sweep DIFFERENT candidate bandwidths (rather than reuse the
# baseline runs' sweep), pass --ws-localized-bandwidth-sweep directly --
# it overrides the scenario's JSON value for that run, no config edit needed:

./run_pipeline.sh -s 2d              --ws-localized-bandwidth-sweep 0.5 1 3 5 8 12 20
./run_pipeline.sh -s 2d_intermediate --ws-localized-bandwidth-sweep 0.5 1 3 5 8 12 20
./run_pipeline.sh -s 2d_dense_tdr    --ws-localized-bandwidth-sweep 0.5 1 3 5 8 12 20

./compare_scenarios.sh --list
./compare_scenarios.sh --ids 2d:2 2d_intermediate:2 2d_dense_tdr:2

# --- Beta sweep ---------------------------------------------------------------
./run_pipeline.sh -s 2d              --beta 0 0.5 1
./run_pipeline.sh -s 2d_intermediate --beta 0 0.5 1
./run_pipeline.sh -s 2d_dense_tdr    --beta 0 0.5 1

./compare_scenarios.sh --list
./compare_scenarios.sh --ids 2d:3 2d_intermediate:3 2d_dense_tdr:3

# --- Gamma sweep ---------------------------------------------------------------
# gamma takes exactly one value per call -- loop over the sweep values.
# (Also: gamma only affects time-lapse/multi-epoch inversions -- confirm
# your synthetic setup actually has multiple epochs, otherwise this has
# no effect regardless of value.)
for scenario in 2d 2d_intermediate 2d_dense_tdr; do
    for g in 0 0.5 1; do
        ./run_pipeline.sh -s "${scenario}" --gamma "${g}"
    done
done

./compare_scenarios.sh --list
# ./compare_scenarios.sh --ids <scenario:number> <scenario:number> <scenario:number> ...

# ./run_pipeline.sh -s 2d --train-val-split alternate_sorted
