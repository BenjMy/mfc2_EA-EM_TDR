#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — EA-EM Ensemble Pipeline launcher (Synthetic & Real Data)
# =============================================================================
# Usage:
#   ./run_pipeline.sh                                           # synthetic, default scenario, 1-D mode

#   ./run_pipeline.sh --mode 2d -s 2d_real       --em-dir ./data/EMI_MFC2/       --tdr-dir ./data/daily_MFC/

#   ./run_pipeline.sh --em-dir ./data/EMI_MFC2/ --tdr-dir ./data/daily_MFC/ --coord-file ./data/SOILNET_MFC2_coordinates_sensors.csv
#   ./run_pipeline.sh --mode 2d                                # synthetic, 2-D UTM grid
#   ./run_pipeline.sh -s mixed_middle_free                     # synthetic, named scenario, 1-D
#   ./run_pipeline.sh -s mixed_middle_free --mode 2d           # synthetic, named scenario, 2-D
#   ./run_pipeline.sh -s fine_layers_fixed -c my.json          # synthetic, custom config file
#   ./run_pipeline.sh -s free_all -e /path/to/venv             # synthetic, custom virtualenv
#   ./run_pipeline.sh --mode 2d --nx 15 --ny 8                 # synthetic, 2-D grid 15×8 points
#   ./run_pipeline.sh --train-val-split by_probe               # held-out-probe validation (see below)
#   ./run_pipeline.sh --list                                    # list scenarios
#   ./run_pipeline.sh --validate                               # validate all scenarios
#   ./run_pipeline.sh --help                                   # show this message
#
# Localization diagnostic (Step 2 kernel-localized WS calibration)
# ------------------------------------------------------------------
#   OFF by default for every scenario. Controlled by the scenario's own
#   'ws_localized_bandwidth' / 'ws_localized_bandwidth_sweep' fields in
#   ensemble_scenarios.json, OR by --ws-localized-bandwidth /
#   --ws-localized-bandwidth-sweep on this wrapper (CLI wins when both
#   are set). See `--help` for details, current per-scenario status, and
#   how to enable/disable it either way.
#
# EM footprint diagnostic (Step 5c McNeill depth-weighted comparison)
# ---------------------------------------------------------------------
#   OFF by default. CLI-only (no scenario JSON field) -- add
#   --em-footprint-diagnostic to any run. Compares each TDR probe's
#   hard-routed (single nearest layer) theta_calc against a McNeill
#   depth-weighted blend across all layers, reporting both RMSEs in
#   Step 5c. Diagnostic only -- never changes Step 2/4 calibration or
#   Step 3/5 K-selection. See `--help` for details.
#
# Train/validation split (Loiseau et al. 2026, Secs. 2.3 & 2.6)
# ---------------------------------------------------------------
#   OFF by default (legacy behaviour: all TDR/SWC observations both fit
#   and select K, so reported RMSE is not out-of-sample). Turn on with
#   --train-val-split {alternate_sorted, by_probe, by_time_block, random}
#   or by setting 'train_val_split' in the scenario JSON. See `--help`
#   for what each method does and which ones are fully independent.
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
MODE=""
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
RANDOM_XY=false
XY_PATCH_SIGMA=""
DEPTH_DRYING_STRENGTH=""
PROBE_XYZ=""
XY_RENDER_MODE=""
NUM_SWC_POSITIONS=""
SWC_DEPTHS=()
WINDOW_M=""
BETA=()
GAMMA=()
WS_LOCALIZED_BANDWIDTH=""
WS_LOCALIZED_BANDWIDTH_SWEEP=()
EM_FOOTPRINT_DIAGNOSTIC=false
QUICK=false
QUICK_EPOCHS=""
STAGE=""
CALIBRATION=""
TRAIN_VAL_SPLIT=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_pipeline.py"
SCENARIO_LOADER="${SCRIPT_DIR}/scenario_loader.py"
SCENARIO_JSON="${SCRIPT_DIR}/ensemble_scenarios.json"
REAL_DATA_LOADER="${SCRIPT_DIR}/data_real.py"

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
      --mode 1d|2d        Spatial mode (default: '2d' for real data,
                          '1d' for synthetic data).
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
      --random-xy         Synthetic mode only (ignored if --homogeneous,
                          --infiltration, or --full-range-gradient is also
                          given). Plan-view (XY) soil moisture is a
                          spatially RANDOM, optionally patch-smoothed field
                          per point, while EVERY point dries out with depth
                          by the same uniform fraction. Overrides
                          'random_xy' in the scenario JSON.
      --xy-patch-sigma S  Only used with --random-xy. Gaussian smoothing
                          sigma (grid cells) applied to the random XY field
                          so nearby points form correlated wet/dry patches
                          instead of point-to-point white noise. 0 = no
                          smoothing. Default: 1.5. Overrides
                          'xy_patch_sigma' in the scenario JSON.
      --depth-drying-strength F
                          Only used with --random-xy. Fraction (0-1) of
                          each point's surface theta lost by the deepest
                          layer, applied identically everywhere. Default:
                          0.6. Overrides 'depth_drying_strength' in the
                          scenario JSON.

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
      --gamma G [G ...]   One or more temporal regularisation values
                          (>= 0, time-lapse/multi-epoch inversions only)
                          to sweep, e.g. --gamma 0 5 10. Overrides
                          'param_grid.gamma' (or scalar 'gamma') in the
                          scenario config. No effect on non-time-lapse
                          (single-epoch) runs regardless of value.
      --quick             Smoke-test preset: ONE alpha (middle of the
                          scenario's alpha list) and ONE forward model
                          (the first) instead of the full grid, e.g.
                          280 Step-1 tasks -> 28. Does not touch the
                          EM/TDR data. The ensemble is shrunk on purpose:
                          output is a smoke test, not a result. Writes
                          to '<seq>_<scenario>_quick' and its own
                          '*_quick*.pkl' cache, so a full-run cache is
                          never overwritten.
      --stage S           Real data only. full (default): one run does
                          everything. calibrate: EM points restricted to
                          the probe neighbourhoods, FULL ensemble, Steps
                          0-5, writes stage_calibration.json (selected K,
                          members, WS parameters, localized bandwidth) and
                          stops before the maps. map: inverts ONLY the
                          members the calibration selected, on the
                          scenario's EM points (all or grid-thinned), runs
                          Steps 2-6 and reports drift vs the calibration.
                          Each stage has its own run folder and cache file.
      --calibration PATH  With --stage map: stage_calibration.json or the
                          '*_calibrate' run folder. Default: newest
                          '*_calibrate' run under the scenario's output_dir.
      --quick-epochs N    With --quick (real data only): also keep only
                          N evenly spaced survey epochs (first and last
                          always included), e.g. 3 of 7 -> 12 tasks.
      --ws-localized-bandwidth C
                          Gaspari-Cohn localization radius 'c' (same
                          units as your x/y coordinates, e.g. metres)
                          for the Step 2 kernel-localized Waxman-Smits
                          diagnostic. Overrides 'ws_localized_bandwidth'
                          in the scenario config; see the "Localization
                          diagnostic" section below for what this does
                          and how to pick a value. Ignored if
                          --ws-localized-bandwidth-sweep is also given.
      --ws-localized-bandwidth-sweep C [C ...]
                          Candidate bandwidths (c values) for the Step 2
                          LOCALIZED SWEEP leave-one-probe-out CV
                          diagnostic, e.g.
                          --ws-localized-bandwidth-sweep 1 2 5 10 15 20.
                          The recommended bandwidth (min pooled LOO-CV
                          RMSE) then overrides any fixed
                          --ws-localized-bandwidth / scenario value.
                          Overrides 'ws_localized_bandwidth_sweep' in
                          the scenario config.
      --em-footprint-diagnostic
                          Compute a McNeill-type depth-weighted "effective
                          EM theta" per probe and report its discrepancy
                          against theta_calc at the hard-routed
                          (nearest-layer) assignment vs. theta_obs (Step
                          5c). Diagnostic only -- never changes Step 2/4
                          calibration or Step 3/5 K-selection. See the "EM
                          footprint diagnostic" section below. OFF by
                          default; no scenario JSON equivalent (CLI-only).
      --train-val-split METHOD
                          Split TDR/SWC observations into a train subset
                          (drives Step 2/4 Waxman-Smits fitting AND Step
                          3/5 ensemble-K selection) and a validation
                          subset (held out untouched, reported once at
                          the end in Step 5b). OFF by default -- the
                          legacy behaviour, where every observation both
                          fits the model and picks K, so the reported
                          RMSE is in-sample, not a fair generalisation
                          estimate. Overrides 'train_val_split' in the
                          scenario config when passed. METHOD is one of:
                            off              No split (default).
                            alternate_sorted Sort by theta_obs, take
                                             every other value. Matches
                                             the paper's Larzac fallback.
                                             NOT fully independent -- the
                                             same probe/time step can
                                             appear in both subsets.
                            by_probe         Hold out whole probes/pits
                                             for validation. Fully
                                             independent. Falls back to
                                             alternate_sorted (with a
                                             loud warning) if the run
                                             can't identify per-
                                             observation probe IDs.
                            by_time_block    Hold out whole time steps
                                             for validation (paper's
                                             Avignon setup: "every other
                                             time step"). Independent
                                             w.r.t. time, not probes.
                            random           Uniform random 50/50 split.
                                             Comparison baseline only --
                                             not recommended as your
                                             primary split since it does
                                             NOT guard against pseudo-
                                             replication the way the
                                             other methods do.

Scenario options (apply in both synthetic and real-data mode):
  -s, --scenario NAME     Scenario name from ensemble_scenarios.json
                          (default: 'default'; use e.g. '2d_real' for
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

      DEFAULT: OFF. Set it either in the scenario JSON (bandwidth stays
      tied to that scenario's own probe geometry, reproducible across
      runs without remembering a flag) or from the command line with
      --ws-localized-bandwidth / --ws-localized-bandwidth-sweep (see
      "Inversion options" above) -- CLI wins if both are set for a given
      run. A scenario/run that sets neither (the default for every
      scenario except '2d') runs with localization disabled, exactly
      like before this feature existed.

      To turn it ON for a scenario (persistent, JSON): add
          "ws_localized_bandwidth": <radius in the same units as your
                                      x/y coordinates, e.g. metres>
      to that scenario's block in the JSON. Rule of thumb: pick a value
      at least ~half the spacing between your closest adjacent SWC/TDR
      probes (kernel support is exactly zero beyond 2×c, so anything
      smaller and most EM points won't see any probe at all).

      To turn it ON for a single run (ad hoc, no JSON edit): add
          --ws-localized-bandwidth <c>
      to your ./run_pipeline.sh call, e.g. --ws-localized-bandwidth 6.
      Use --ws-localized-bandwidth-sweep <c1> <c2> ... instead to pick
      c automatically via leave-one-probe-out CV over those candidates.

      To turn it OFF for a scenario that currently has it on: delete
      the 'ws_localized_bandwidth' key (and/or 'ws_localized_bandwidth_
      sweep'), or set it to null -- and don't pass the CLI flag, which
      would re-enable it for that run regardless of the JSON.

      Current status in ensemble_scenarios.json (re-check after editing):
          2d                      -> ON  (c = 5.0, sweep [1,2,5,10,15,20])
          2d_intermediate         -> ON  (c = 5.0, sweep [1,2,5,10,15,20])
          2d_dense_tdr            -> ON  (sweep only [1,2,5,10,15,20], no fixed c)
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

EM footprint diagnostic (Step 5c McNeill depth-weighted comparison):
      This runs a DIAGNOSTIC SIBLING to Step 2/4 calibration and Step 5
      K-selection, comparing two ways of predicting a TDR probe's water
      content from theta_calc:
        - hard-routed: the single layer the probe is geometrically
          assigned to (current default routing, unchanged by this flag),
        - McNeill-weighted: a depth-weighted blend across ALL layers,
          using the closed-form McNeill (1980) cumulative EM response
          curve for the coil geometry actually in use (see
          em_sensitivity.py) -- narrow near the surface, wider at depth,
          the EM analogue of ERT's depth-varying-footprint problem, but
          tractable here because EM's kernel is a fixed function of coil
          geometry rather than an emergent property of an inversion mesh.

      DEFAULT: OFF. CLI-only -- add --em-footprint-diagnostic to any run
      (synthetic or real). No scenario JSON field for this one (unlike
      ws_localized_bandwidth above); it's a read-only comparison of
      already-computed theta_calc, not something that needs per-scenario
      tuning to be meaningful.

      Real vs. synthetic coil geometry are handled differently
      internally (see em_sensitivity.py's module docstring): real-data
      runs use the single fixed HCP/1.66 m geometry exactly; synthetic
      runs average across all six (orientation, spacing) pairs in
      COILS_LIST, labelled explicitly as an approximation in the log --
      there is no single physically-correct kernel when six different
      coil geometries feed one inversion.

      Output: a Step 5c section in pipeline_report_*.md with both RMSEs,
      a per-probe table, and a one-line interpretation (whether the
      weighted comparison fits TDR better than hard-routed, and what
      that would or wouldn't imply about the current probe-to-layer
      routing) -- plus two PNGs in <output_dir>/diagnostics/:
      em_footprint_probe_comparison.png (per-probe hard-routed vs.
      weighted error bars) and em_footprint_spatial_discrepancy.png
      (which comparison sits closer to TDR at each matched EM point,
      mapped spatially). Nothing else in the pipeline changes when this
      is on.

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
  ./run_pipeline.sh --mode 2d -s moisture_2d_random_xy               # random/patchy XY, uniform depth-drying
  ./run_pipeline.sh --mode 2d -s 2d --random-xy --xy-patch-sigma 2 --depth-drying-strength 0.5

  # --- Override SWC/TDR sensor network from the command line ---
  ./run_pipeline.sh --mode 2d -s 2d --num-swc-positions 6
  ./run_pipeline.sh --mode 2d -s 2d --swc-depths 0.15 0.45 0.9
  ./run_pipeline.sh --mode 2d -s 2d --num-swc-positions 6 --swc-depths 0.15 0.45 0.9 --window-m 1.5
  ./run_pipeline.sh --mode 2d -s 2d --probe-xyz coord_TDR.csv   # explicit probe (x,y,z) from CSV

  # --- Control the XY plan-view plot rendering (2-D mode) ---
  ./run_pipeline.sh --mode 2d -s 2d --xy-render-mode imshow

  # --- Real data 2-D (field observations on UTM grid) ---
  ./run_pipeline.sh --data-source real --mode 2d -s 2d_real \
      --em-dir ./data/EMI_MFC2/ \
      --tdr-dir ./data/daily_MFC/ \
      --drop-coils f425 f1525 \
      --beta 0.5 20 \
      --gamma 0.5 20 \
      --ws-localized-bandwidth-sweep 0.5 1 3 5 8 12 20

  # --- Train/validation split (held-out RMSE, Loiseau et al. 2026) ---
  ./run_pipeline.sh --train-val-split alternate_sorted
  ./run_pipeline.sh --train-val-split by_probe             # fully independent
  ./run_pipeline.sh --train-val-split by_time_block         # fully independent (time)
  ./run_pipeline.sh --data-source real -s 2d_real \
      --em-dir ./data/EMI_MFC2/ \
      --tdr-dir ./data/daily_MFC/ \
      --train-val-split by_probe

  # --- EM footprint diagnostic (Step 5c, McNeill depth weighting) ---
  ./run_pipeline.sh -s 2d --em-footprint-diagnostic
  ./run_pipeline.sh --data-source real -s 2d_real \
      --em-dir ./data/EMI_MFC2/ \
      --tdr-dir ./data/daily_MFC/ \
      --em-footprint-diagnostic

  # --- Drop a noisy coil/frequency channel before inversion ---
  ./run_pipeline.sh --data-source real -s 2d_real \
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
            [[ $# -ge 2 ]] || die "--gamma requires at least one value"
            shift # consume the flag itself
            GAMMA=()
            # Greedily consume one-or-more following tokens, stopping at the
            # next recognised flag (mirrors Python argparse's nargs='+').
            while [[ $# -gt 0 && "$1" != -* ]]; do
                GAMMA+=("$1")
                shift
            done
            [[ ${#GAMMA[@]} -ge 1 ]] || die "--gamma requires at least one value"
            ;;
        --ws-localized-bandwidth)
            WS_LOCALIZED_BANDWIDTH="${2:?'--ws-localized-bandwidth requires a number'}"
            shift 2 ;;
        --ws-localized-bandwidth-sweep)
            [[ $# -ge 2 ]] || die "--ws-localized-bandwidth-sweep requires at least one value"
            shift # consume the flag itself
            WS_LOCALIZED_BANDWIDTH_SWEEP=()
            # Greedily consume one-or-more following tokens, stopping at the
            # next recognised flag (mirrors Python argparse's nargs='+').
            while [[ $# -gt 0 && "$1" != -* ]]; do
                WS_LOCALIZED_BANDWIDTH_SWEEP+=("$1")
                shift
            done
            [[ ${#WS_LOCALIZED_BANDWIDTH_SWEEP[@]} -ge 1 ]] || die "--ws-localized-bandwidth-sweep requires at least one value"
            ;;
        --train-val-split)
            TRAIN_VAL_SPLIT="${2:?'--train-val-split requires a method'}"
            case "${TRAIN_VAL_SPLIT}" in
                off|alternate_sorted|by_probe|by_time_block|random) ;;
                *) die "--train-val-split must be one of: off, alternate_sorted, by_probe, by_time_block, random (got '${TRAIN_VAL_SPLIT}')" ;;
            esac
            shift 2 ;;
        --list)     LIST_ONLY=true;     shift ;;
        --validate) VALIDATE_ONLY=true; shift ;;
        --homogeneous) HOMOGENEOUS=true; shift ;;
        --em-footprint-diagnostic) EM_FOOTPRINT_DIAGNOSTIC=true; shift ;;
        --quick) QUICK=true; shift ;;
        --stage)
            STAGE="${2:?'--stage requires one of: full calibrate map'}"
            case "${STAGE}" in full|calibrate|map) ;; *) die "--stage must be full, calibrate or map, got '${STAGE}'" ;; esac
            shift 2 ;;
        --calibration)
            CALIBRATION="${2:?'--calibration requires a path'}"; shift 2 ;;
        --quick-epochs)
            QUICK_EPOCHS="${2:?'--quick-epochs requires a positive integer'}"
            [[ "${QUICK_EPOCHS}" =~ ^[1-9][0-9]*$ ]] \
                || die "--quick-epochs must be a positive integer, got '${QUICK_EPOCHS}'"
            shift 2 ;;
        --full-range-gradient) FULL_RANGE_GRADIENT=true; shift ;;
        --xy-z-blend)
            XY_Z_BLEND="${2:?'--xy-z-blend requires a number'}"
            shift 2 ;;
        --invert-z-gradient) INVERT_Z_GRADIENT=true; shift ;;
        --random-xy) RANDOM_XY=true; shift ;;
        --xy-patch-sigma)
            XY_PATCH_SIGMA="${2:?'--xy-patch-sigma requires a number'}"
            shift 2 ;;
        --depth-drying-strength)
            DEPTH_DRYING_STRENGTH="${2:?'--depth-drying-strength requires a number'}"
            shift 2 ;;
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

# Warn when --random-xy / --xy-patch-sigma / --depth-drying-strength are
# passed for real-data runs (synthetic-generation-only flags)
if ${RANDOM_XY} && [[ "${DATA_SOURCE}" == "real" ]]; then
    log "WARNING: --random-xy is ignored in real-data mode (synthetic generation only)."
fi
if [[ -n "${XY_PATCH_SIGMA}" && "${DATA_SOURCE}" == "real" ]]; then
    log "WARNING: --xy-patch-sigma is ignored in real-data mode (synthetic generation only)."
fi
if [[ -n "${DEPTH_DRYING_STRENGTH}" && "${DATA_SOURCE}" == "real" ]]; then
    log "WARNING: --depth-drying-strength is ignored in real-data mode (synthetic generation only)."
fi

# Warn when --random-xy's sigma/strength flags are passed without it
if [[ -n "${XY_PATCH_SIGMA}" ]] && ! ${RANDOM_XY}; then
    log "WARNING: --xy-patch-sigma has no effect without --random-xy."
fi
if [[ -n "${DEPTH_DRYING_STRENGTH}" ]] && ! ${RANDOM_XY}; then
    log "WARNING: --depth-drying-strength has no effect without --random-xy."
fi

# --random-xy and --full-range-gradient are mutually exclusive branches in
# the generator's if/elif chain; --full-range-gradient wins, so warn.
if ${RANDOM_XY} && ${FULL_RANGE_GRADIENT}; then
    log "WARNING: --full-range-gradient takes precedence over --random-xy; random_xy will be ignored."
fi

# ---------------------------------------------------------------------------
# Resolve data source and paths from scenario JSON
# ---------------------------------------------------------------------------
# Read data_source, em_data_dir, tdr_data_dir, and coord_file from the chosen
# scenario if they are declared there.  CLI flags take precedence over JSON
# values when explicitly supplied, but the two must not disagree.
_SCENARIO_JSON_PATH="${CONFIG_FILE:-${SCENARIO_JSON}}"

# Fields are read through scenario_loader.py, so 'extends', generated sweeps and
# defaults are resolved exactly as in the pipeline. A scenario that cannot be
# loaded (typically: a real-data scenario run without -c ensemble_scenarios_real.json)
# stops here with the loader's reason instead of silently falling back to
# data_source='synthetic' / '1d'.
if ! _SCENARIO_ERR="$("${PYTHON}" "${SCRIPT_DIR}/scenario_loader.py" "${SCENARIO}" \
        --config "${_SCENARIO_JSON_PATH}" --field name 2>&1 >/dev/null)"; then
    die "${_SCENARIO_ERR%% Available scenarios:*}
  Config used : ${_SCENARIO_JSON_PATH}
  List the scenarios of a config with: python scenario_loader.py --list-scenarios --config <file>"
fi

_read_scenario_field() {
    "${PYTHON}" "${SCRIPT_DIR}/scenario_loader.py" "${SCENARIO}" \
        --config "${_SCENARIO_JSON_PATH}" --field "$1" 2>/dev/null
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
# Resolve effective spatial mode
# ---------------------------------------------------------------------------
# Explicit --mode always wins, for either data source. Otherwise default by
# data source: real EM survey positions are already absolute UTM coordinates
# (see run_pipeline.py's own real-data 2-D fix), so there's no reason to
# leave 2-D output on the table -- '2d' is the sensible default there.
# Synthetic keeps its original '1d' default, unchanged.
if [[ -z "${MODE}" ]]; then
    if [[ "${EFFECTIVE_DS}" == "real" ]]; then
        MODE="2d"
    else
        MODE="1d"
    fi
    log "--mode not given; defaulting to '${MODE}' for data_source='${EFFECTIVE_DS}'."
fi

# Warn when --nx / --ny are passed but the (now-resolved) mode is 1d
if [[ "${MODE}" == "1d" && ( "${NX}" != "10" || "${NY}" != "5" ) ]]; then
    log "WARNING: --nx / --ny are ignored in 1-D mode. Pass --mode 2d to use them."
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
log "  Random XY   : ${RANDOM_XY}"
log "  Train/val   : ${TRAIN_VAL_SPLIT:-scenario default (or off)}"
log "  EM footprint diagnostic : ${EM_FOOTPRINT_DIAGNOSTIC}"
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
if ${RANDOM_XY}; then
    GRADIENT_ARGS+=(--random-xy)
fi
if [[ -n "${XY_PATCH_SIGMA}" ]]; then
    GRADIENT_ARGS+=(--xy-patch-sigma "${XY_PATCH_SIGMA}")
fi
if [[ -n "${DEPTH_DRYING_STRENGTH}" ]]; then
    GRADIENT_ARGS+=(--depth-drying-strength "${DEPTH_DRYING_STRENGTH}")
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
if [[ ${#GAMMA[@]} -gt 0 ]]; then
    INVERSION_OVERRIDE_ARGS+=(--gamma "${GAMMA[@]}")
fi
if [[ -n "${WS_LOCALIZED_BANDWIDTH}" ]]; then
    INVERSION_OVERRIDE_ARGS+=(--ws-localized-bandwidth "${WS_LOCALIZED_BANDWIDTH}")
fi
if [[ ${#WS_LOCALIZED_BANDWIDTH_SWEEP[@]} -gt 0 ]]; then
    INVERSION_OVERRIDE_ARGS+=(--ws-localized-bandwidth-sweep "${WS_LOCALIZED_BANDWIDTH_SWEEP[@]}")
fi

TRAIN_VAL_ARGS=()
if [[ -n "${TRAIN_VAL_SPLIT}" ]]; then
    TRAIN_VAL_ARGS+=(--train-val-split "${TRAIN_VAL_SPLIT}")
fi

if [[ -n "${QUICK_EPOCHS}" ]] && ! ${QUICK}; then
    die "--quick-epochs requires --quick"
fi
QUICK_ARGS=()
if ${QUICK}; then
    QUICK_ARGS+=(--quick)
    [[ -n "${QUICK_EPOCHS}" ]] && QUICK_ARGS+=(--quick-epochs "${QUICK_EPOCHS}")
fi

if [[ -n "${CALIBRATION}" && "${STAGE}" != "map" ]]; then
    die "--calibration requires --stage map"
fi
if [[ "${STAGE}" == "map" ]] && ${QUICK}; then
    die "--stage map cannot be combined with --quick"
fi
STAGE_ARGS=()
[[ -n "${STAGE}" ]] && STAGE_ARGS+=(--stage "${STAGE}")
[[ -n "${CALIBRATION}" ]] && STAGE_ARGS+=(--calibration "${CALIBRATION}")

EM_FOOTPRINT_ARGS=()
if ${EM_FOOTPRINT_DIAGNOSTIC}; then
    EM_FOOTPRINT_ARGS=(--em-footprint-diagnostic)
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
| random_xy | \`${RANDOM_XY}\` |
| num-swc-positions | \`${NUM_SWC_POSITIONS:-scenario default}\` |
| swc-depths | \`${SWC_DEPTHS[*]:-scenario default}\` |
| window-m | \`${WINDOW_M:-scenario default}\` |
| beta | \`${BETA[*]:-scenario default}\` |
| gamma | \`${GAMMA[*]:-scenario default}\` |
| ws-localized-bandwidth | \`${WS_LOCALIZED_BANDWIDTH:-scenario default (or disabled)}\` |
| ws-localized-bandwidth-sweep | \`${WS_LOCALIZED_BANDWIDTH_SWEEP[*]:-scenario default (or disabled)}\` |
| train-val-split | \`${TRAIN_VAL_SPLIT:-scenario default (or off)}\` |
| em-footprint-diagnostic | \`${EM_FOOTPRINT_DIAGNOSTIC}\` |
| quick | \`${QUICK}${QUICK_EPOCHS:+ (epochs=${QUICK_EPOCHS})}\` |
| stage | \`${STAGE:-full}${CALIBRATION:+ (calibration=${CALIBRATION})}\` |
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
    "${TRAIN_VAL_ARGS[@]+"${TRAIN_VAL_ARGS[@]}"}" \
    "${EM_FOOTPRINT_ARGS[@]+"${EM_FOOTPRINT_ARGS[@]}"}" \
    "${QUICK_ARGS[@]+"${QUICK_ARGS[@]}"}" \
    "${STAGE_ARGS[@]+"${STAGE_ARGS[@]}"}" \
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
