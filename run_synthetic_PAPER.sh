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
#   - gamma now accepts multiple values in ONE run_pipeline.sh call, same
#     as beta -- space-separated, no brackets/commas: --gamma 0 5 10.
#     Each gamma value is combined with every alpha/beta/method/
#     forwardModel/starting-conductivity combo (N gammas = N x the
#     ensemble tasks), and it's still a no-op on non-time-lapse
#     (single-epoch) runs regardless of value.
#   - beta DOES accept multiple values, but space-separated, no brackets/
#     commas: --beta 0 0.5 1
#
# NOTE on --em-footprint-diagnostic:
#   CLI-only, no scenario JSON field (unlike ws_localized_bandwidth above)
#   -- it's a read-only comparison against theta_calc that's already been
#   computed by the time Step 5c runs, so there's nothing scenario-specific
#   to persist. Add it to any of the calls below to get the Step 5c
#   McNeill depth-weighting comparison in that run's report; see the block
#   further down for a dedicated 3-scenario sweep.
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
./run_pipeline.sh -s 2d --em-footprint-diagnostic
./run_pipeline.sh -s 2d_intermediate --em-footprint-diagnostic
./run_pipeline.sh -s 2d_dense_tdr --em-footprint-diagnostic

./compare_scenarios.sh --list   # copy the 3 run numbers just produced, e.g.:
#./compare_scenarios.sh --ids 2d:1 2d_intermediate:1 2d_dense_tdr:1 --position 0 0 
./compare_scenarios.sh --ids 2d:1 2d_intermediate:1 2d_dense_tdr:1 --position 5 0

# --- Localized-bandwidth sweep diagnostic ------------------------------------
# Already runs automatically above for '2d', '2d_intermediate', and
# '2d_dense_tdr' because ws_localized_bandwidth_sweep is set in the JSON for
# all three. To sweep DIFFERENT candidate bandwidths (rather than reuse the
# baseline runs' sweep), pass --ws-localized-bandwidth-sweep directly --
# it overrides the scenario's JSON value for that run, no config edit needed:


./run_pipeline.sh -s 2d              --ws-localized-bandwidth 20 --em-footprint-diagnostic



./run_pipeline.sh -s 2d              --ws-localized-bandwidth-sweep 0.5 1 3 5 8 12 20 --em-footprint-diagnostic
./run_pipeline.sh -s 2d_intermediate --ws-localized-bandwidth-sweep 0.5 1 3 5 8 12 20 --em-footprint-diagnostic
./run_pipeline.sh -s 2d_dense_tdr    --ws-localized-bandwidth-sweep 0.5 1 3 5 8 12 20 --em-footprint-diagnostic

./compare_scenarios.sh --list
./compare_scenarios.sh --ids 2d:2 2d_intermediate:2 2d_dense_tdr:2 --position 5 0



./run_pipeline.sh -s 2d_random_xy_drying --mode 2d
./run_pipeline.sh -s 2d_random_xy_drying --mode 2d --ws-localized-bandwidth-sweep 0.5 1 3 5 8 12 20 --probe-xyz ./data/coord_TDR_mode2d_2sensors.csv
./run_pipeline.sh -s 2d_random_xy_drying --mode 2d --ws-localized-bandwidth-sweep 0.5 1 3 5 8 12 20 --probe-xyz ./data/coord_TDR_mode2d_8sensors.csv
./compare_scenarios.sh --ids 2d_random_xy_drying:1 2d_random_xy_drying:1 --position 5 0


./run_pipeline.sh --mode 1d -s 2d_horizon_ws

./run_pipeline.sh --mode 2d -s 2d_horizon_ws --ws-localized-bandwidth-sweep 0.5 1 3 5 8 12 20 --probe-xyz ./data/coord_TDR_mode2d_2sensors.csv
./run_pipeline.sh --mode 2d -s 2d_horizon_ws --ws-localized-bandwidth-sweep 0.5 1 3 5 8 12 20 --probe-xyz ./data/coord_TDR_mode2d_8sensors.csv



./run_pipeline.sh -s 2d_xy_waxman --mode 2d \
    --ws-localized-bandwidth-sweep 0.5 1 3 5 8 12 20 \
    --probe-xyz ./data/coord_TDR_mode2d_8sensors.csv \

    
    


# --- Beta sweep ---------------------------------------------------------------
#./run_pipeline.sh -s 2d              --beta 0 0.5 1
#./run_pipeline.sh -s 2d_intermediate --beta 0 0.5 1
#./run_pipeline.sh -s 2d_dense_tdr    --beta 0 0.5 1

# --- Gamma sweep ---------------------------------------------------------------
# gamma now sweeps in a single call, same as beta -- no more looping one
# call per value. (Still: gamma only affects time-lapse/multi-epoch
# inversions -- confirm your synthetic setup actually has multiple
# epochs, otherwise this has no effect regardless of value.)
./run_pipeline.sh -s 2d              --gamma 0 10
./run_pipeline.sh -s 2d_intermediate --gamma 0 10
./run_pipeline.sh -s 2d_dense_tdr    --gamma 0 10

./compare_scenarios.sh --list
# ./compare_scenarios.sh --ids <scenario:number> <scenario:number> <scenario:number> ...

# --- Combined beta + gamma sweep -----------------------------------------------
# beta and gamma can be swept together in the same call -- every beta
# value gets crossed with every gamma value (plus alpha/method/
# forwardModel/starting-conductivities), so this is
# len(beta) x len(gamma) x ... ensemble tasks, not len(beta) + len(gamma).
#./run_pipeline.sh -s 2d              --beta 0 0.5 1 --gamma 0 10
#./run_pipeline.sh -s 2d_intermediate --beta 0 0.5 1 --gamma 0 10
#./run_pipeline.sh -s 2d_dense_tdr    --beta 0 0.5 1 --gamma 0 10

# --- train-val-split ------------------------------------
#./run_pipeline.sh -s 2d --train-val-split alternate_sorted
#./run_pipeline.sh -s 2d_intermediate --train-val-split alternate_sorted
#./run_pipeline.sh -s 2d_dense_tdr --train-val-split alternate_sorted

#./compare_scenarios.sh --list
#./compare_scenarios.sh --ids 2d:3 2d_intermediate:3 2d_dense_tdr:3


# --- Surface-only TDR test (1-D transect) ------------------------------------
# Hypothesis: with TDR only at 0.2 m (horizon 0 calibrated, horizons 1-3 on the
# lab_p0 prior), does the inversion -- and theta at depth -- still improve?
# 1d_surface_tdr    : probes only at 0.2 m (allow_uncovered_horizons=true)
# 1d_full_depth_ref : identical setup, probes in all 4 horizons (reference)
# Truth m/sigma_s vary along x, identically at every depth. Probe count is swept
# with --num-swc-positions; probes are spread along the 18 m transect.
for N in 2 3 5 10; do
    ./run_pipeline.sh -s 1d_surface_tdr    --mode 1d --num-swc-positions "$N"
    ./run_pipeline.sh -s 1d_full_depth_ref --mode 1d --num-swc-positions "$N"
done

./compare_scenarios.sh --list
# Run numbers follow the loop order: <scenario>:1 -> N=2, :2 -> N=3, :3 -> N=5, :4 -> N=10
# Example: N=5, surface-only vs full-depth, truth point at x=8 m
#./compare_scenarios.sh --ids 1d_surface_tdr:3 1d_full_depth_ref:3 \
#    --labels surface_only full_depth --position 8 0
