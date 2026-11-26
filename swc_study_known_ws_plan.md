# SWC study: does knowing some WS parameters help retrieve the others?

Commit as me as author
also I dont like the fact that for the WS heteregeneous case some parameters varies with depth other varies with lateral distance. Create both cases. 

Plan for extending `study_swc_design.py` (branch `feat/localized-ws-maps`).
Each step below is one patch (`git am`), small, tested, and independent of the later ones.

## 0. Why the current study cannot show the message

Facts from your own `study_summary.csv` and from the K=1 / K=2 reports of
`swcstudy_smchomo_wshomo_beta_spread_n5_free{1,2}`:

| | K=1 (σs free; a, m, σw pinned at truth) | K=2 (σs, m free) |
|---|---|---|
| fitted σs | 200 in all 4 horizons (upper bound, truth 100) | 122-123 (start 120) |
| fitted m | 3.0 (pinned = truth) | 1.60 / 1.35 / 1.17 / 1.38 (truth 3.0) |
| calibration RMSE at probes | 0.18-0.30 m3/m3 | 0.037-0.065 m3/m3 |
| median `rmse_truth` / `bias_truth` (homo/homo) | 0.227 / +0.206 | 0.025 / -0.001 |

Reading:

1. With three parameters at their true values the fit is still bad, so the limit is **not** the
   number of unknowns. The inverted EC that enters Waxman-Smits is not consistent with the truth EC
   (cause not yet identified: EM inversion bias, EC range, or synthetic generation).
2. K=2 "works" by pushing m to 1.2-1.6 (truth 3.0): the free parameter absorbs the EC error. It is
   not retrieved, it is distorted.
3. θ = (a·σ_b / (σw+σs))^(1/m) depends only on `C = a/(σw+σs)` and `m`. With σw = 500 and σs ∈ [0, 200],
   σs moves C by about ±17 % and θ by about -5 % / +6 %. σs is nearly unidentifiable, and K freeing
   σs first starts with the weakest lever.
4. K is the wrong axis for "what is known": it frees parameters in a fixed order
   (σs, m, σw, a) and always starts the free ones +20 % off.

Target message: *"if some WS parameters are known from sampling, the others are retrieved better"*.
That message is only demonstrable if the inverted EC is good enough, which Patch 1 tests first.

## 1. Constraints for all patches

* Old run folders and `run_log_swc_study.csv` rows must keep parsing: new factors are added to the
  scenario name **only when they differ from the default**
  (`swcstudy_smchomo_wshomo_beta_spread_n5_free1` stays valid).
* No change to the pipeline itself: everything is expressed through existing scenario fields
  (`ws_fixed_params`, `lab_p0`, `ws_bounds`, `swc_depths`, `allow_uncovered_horizons`).
* Existing presets (`smoke`, `oat`, `main`, `full`) keep the same designs; new studies are new presets.
* Each patch adds tests to `test_study_swc_design.py` that fail without it.
* Cost: ~3-5 min per run (`--fast` for smoke tests only; its results are not comparable).

## 2. Patch sequence

### Patch 1 - K = 0 floor (item 1)

Goal: all four WS parameters pinned. The calibration has nothing to fit, so the remaining RMSE is
the **floor imposed by the inverted EC**. If the floor is large, no amount of known parameters helps
until the EC inversion is fixed.

* `LEVELS["k"] = (0, 1, 2, 3, 4)`; `--free-k` accepts 0. `ws_freedom(0, ...)` already exists.
* K=0 does not depend on `n` or `dist` (nothing is fitted), so `build_designs` keeps **one** K=0
  design per (smc, ws, reg): the smallest `n`, `spread`. Saves ~24 runs per truth.
* `biased_start` is skipped for K=0 (already true, `k < 1`).
* `study_status.py` picks K=0 up automatically through `build_designs`.
* Heat map, RMSE-vs-K plots: K=0 appears as the first x tick; no code change expected.
* Note: for `ws=hetero` the pinned values are the per-layer baseline, not the per-point truth, so
  the K=0 floor there is not "perfect knowledge". Use `ws=homo` for the clean floor.

Run: `python study_swc_design.py --smc homo hetero --ws homo hetero --reg noreg --free-k 0 --n-swc 5 --dist spread`
(4 runs, ~20 min).

Acceptance: you can read the floor per truth. Decision rule:
floor ≲ 0.03 m3/m3 → the message is demonstrable (go on with Patches 3-6);
floor ≫ 0.03 → first diagnose the EC (Patch 2b), then continue.

### Patch 2a - Parameter-recovery table (item 4, data side)

* New `ws_recovery.py` (pure parsing, unit-tested): reads each run folder's `pipeline_report_2*.md`
  and returns one row per (scenario, horizon): start (line "Calibration prior/start ..."), truth
  ("Step 0 ... ground-truth" table), fitted (table "Step 4 - Best recalibrated fit", fallback
  "Step 2 - Best-ranked"), fixed/free flags (`ws_fixed_params` row), bounds, and derived
  `C = a/(σw+σs)`.
* `study_swc_design.py --analyze-only` also writes `study_ws_recovery.csv`.
* Works on all runs you already have (no rerun).

### Patch 2b - EC diagnostics (only if the K=0 floor is large)

* Needs one listing from you: `ls mfc2_outputs/swc_design_study/<run>/0001_*/output_model | head -60`
  to find the inverted-EC array.
* Add `ec_rmse_truth`, `ec_bias_rel` (inverted vs true EC, per layer) to `compute_metrics`.
* Purpose: show directly how large the EC error is, before any WS step.

### Patch 2c - Parameter-recovery figure with truth (items 4 and "truth value in the plots")

* `fig_ws_recovery.png`: one panel per parameter (a, m, σw, σs) plus one for `C`; x = scenario;
  markers = start / fitted (per horizon); horizontal line = truth; shaded band = bounds;
  pinned parameters drawn hollow.
* Truth curve added to the Step 2 `petro_calibration*.png` (fitted vs true Waxman-Smits curve),
  read from the run's own truth table.

### Patch 3 - "What is known" as a factor (item 2)

New factor `pin` (default `k` = legacy behaviour, so old names are unchanged):

| pin | known (pinned) | free |
|---|---|---|
| `none` | nothing | a, m, σw, σs |
| `sw` | σw | a, m, σs |
| `sws` | σw, σs | a, m |
| `am` | a, m | σw, σs |
| `all` | all four | none (= K=0) |

* `ws_pinned(known_set, ...)` replaces `ws_freedom` for non-legacy pins; same rule as before when
  `a` is free (start 1.0, bounds [0.2, 5], because the truth 0.5 sits on the default lower bound).
* Name: `..._pinsw` appended only when `pin != k`.
* Free parameters start at truth × (1 + `--p0-bias`), as now; pinned ones at the known value.
* New preset `known` (reference n = 5, spread; pins × truths); see sizes in section 3.

### Patch 4 - Error on the known values (item 3)

New factor `err` (default 0): relative error of the pinned values, `p10` = +10 %, `m10` = -10 %.

* Pinned value = truth × (1 + err); name `..._errp10` only when `err != 0`.
* Levels: 0, ±10 %, ±20 %. Both signs are needed (a one-signed error hides asymmetric effects).
* For `ws=hetero` the pinned value is the per-layer baseline (`lab_p0` given per layer); the m and σs
  that vary along x cannot be pinned to one number, so primary results use `ws=homo`.
* Output: `fig_known_error.png`, RMSE and fitted-vs-truth of the free parameters against `err`.

### Patch 5 - Surface-only vs buried probes (your extra question)

New factor `probes` (default `full`; I read "electrodes" as the SWC/TDR probes, not ERT):

| probes | `swc_depths` [m] | horizons calibrated |
|---|---|---|
| `full` | layer centres 0.15, 0.45, 0.80, 1.25 | all 4 |
| `surface` | 0.2 (as `1d_surface_tdr`) | horizon 0 only |
| `buried` | 0.45, 0.80, 1.25 | horizons 1-3 (horizon 0 uncovered) |

* Child scenarios set `swc_depths` and `allow_uncovered_horizons: true`; the probe CSV in
  `probes/` is written with those depths (today `probe_depths(parent)` is global, it becomes per design).
* Name: `..._prsurface` only when `probes != full`.
* Uncovered horizons run on the prior (`lab_p0`) only. That is exactly the trade-off: what is known
  or biased in `lab_p0` decides the accuracy of the uncovered layers.
* `compute_metrics` already splits `rmse_calibrated_h` / `rmse_uncalibrated_h`; the plot reports
  both, per layer (`rmse_by_layer`).
* Question this answers: how much of the benefit of known parameters survives when only the surface
  is sampled, and which depths matter most (surface vs deep).

### Patch 6 - Synthesis figures (the message)

* `fig_known_summary.png`: RMSE (and bias) vs "what is known", one line per `err`, one panel per
  `probes`; floor (K=0 / `all`) as a dashed line.
* `fig_known_vs_free_recovery.png`: error of each free parameter (fit - truth, normalised by its
  range) against what is known.
* `study_report.md` gets the tables.

## 3. Sizes (reference n = 5, spread, noreg, smc hetero)

| preset | runs | time at ~4 min |
|---|---|---|
| Patch 1 floor (4 truths) | 4 | 0.3 h |
| Patch 3 `known` (5 pins × 2 truths) | 10 | 0.7 h |
| + Patch 4 (err 0, ±10, ±20; pins ≠ none/all: 3) | 10 + 3×4×2 = 34 | 2.3 h |
| + Patch 5 (× 3 probes) | ~100 | 7 h |

Run `--dry-run` first (it prints the plan and validates every scenario) and use `--resume --keep-going`.

## 4. Order of work and decision points

1. Patch 1 and run the 4 floor runs. **Decision**: floor small or large.
2. Patch 2a (+2c) on the runs you already have; this also gives the "truth" lines in the plots.
3. If the floor is large: Patch 2b and fix/understand the EC bias before Patches 3-6.
4. Patches 3, 4, 5, 6 in this order; run each preset before writing the next.

## 5. Assumptions to confirm

* Inverted-EC file name in `output_model/` (Patch 2b).
* "Electrodes" = SWC/TDR probes (Patch 5). If you meant ERT electrodes, that is a different
  factor (EM/ERT geometry) and needs its own plan.
* Report table titles are stable ("Step 2 - Best-ranked...", "Step 4 - Best recalibrated...").
  Old reports use the same format (checked on your K=1 / K=2 runs).

## 6. Observations (2026-11-26)

### Status of the patches

| Patch | Status |
|---|---|
| 1, 2a, 3 | done |
| 2c | done (figure `fig_ws_recovery.png` + truth curve on the Step 2 EC-theta plots; needed a pipeline change) |
| 4 | done (`--err`, `fig_known_error.png`) |
| 2b | not written; now the next one (see below) |
| 5, 6 | not started |

Patch 4 notes: a pinned value outside the WS bounds (e.g. a = 0.40) is not clipped, its bound is
(value, value). `err` applies to the pins `sw`, `sws`, `am` only. `--analyze-only` needs the same
`--preset` / `--err` as the runs.

### K = 0 floor: large

`smc = hetero`, noreg, spread, n = 5, all four WS parameters pinned:

| WS truth | rmse_truth [m3/m3] | mae_truth | final_rmse (probes) | quality |
|---|---|---|---|---|
| homo | **0.132** | 0.109 | 0.123 | WARN |
| hetero | 0.171 | 0.149 | 0.167 | OK |

Decision rule of Patch 1: 0.132 is far above 0.03, so diagnose the EC before running the `known` and
error studies. For `hetero` the pinned values are only the layer baseline; read the `homo` row.

### Inverted vs true EC

Run `smchetero_wshomo_..._free0`, `ec_optimal.npy` vs `synthetic_time_lapse_ec.npy[0]`, both
(3 times, 10 points, 4 layers); medians and correlation per layer over all times and points:

| layer | true median [mS/m] | inverted median [mS/m] | ratio | correlation |
|---|---|---|---|---|
| 0 | 5.17 | 24.74 | 4.8 | -0.05 |
| 1 | 10.80 | 51.44 | 4.8 | +0.20 |
| 2 | 19.02 | 71.84 | 3.8 | +0.40 |
| 3 | 29.80 | 32.53 | 1.1 | +0.60 |

EC is overestimated in the top three layers, and the surface layer shows no relation to the truth.
With m = 3 an EC 4.8 times too high gives theta about 1.7 times too high. The ratio is not uniform,
so a simple unit error is unlikely. This is one run.

### Candidate causes (from the scenario settings, none tested)

1. `fixed_depths = false`: the interfaces can move while the truth layers stay at 0.3 / 0.6 / 1.0 m,
   so the layer-by-layer comparison may not compare the same depth intervals.
2. `param_grid.forwardModel = [FSlin, CS]`: the data are generated with `CS` only.
3. `param_grid.method = [ROPE]` with `cond_ranges` 0.01-300 mS/m in every layer.

Next test: change one at a time on `smchetero_wshomo` K = 0 (fixed depths; `CS` only; a gradient
method such as `L-BFGS-B`) and compare the same per-layer table. Nothing is calibrated at K = 0.

### Synthetic EC floor

`data_synthetic.py` has `ec_bulk = max(1.0, ec_bulk + noise)`. With the baseline truth
(0.5, 3, 500, 100), any theta below about 0.094 is below 1 mS/m and gets raised to it.
Share of true EC at the floor (one run sampled per truth type):

| smc / ws | at 1.0 mS/m | median true EC [mS/m] |
|---|---|---|
| homo / homo | 66.7 % | 1.00 |
| homo / hetero | 35.6 % | 2.50 |
| hetero / hetero | 9.0 % | 39.67 |
| hetero / homo | 4.0 % | 14.66 |

Do not draw conclusions from `smc = homo` runs (this includes the K = 1 / K = 2 numbers of section 0)
until the floor is removed or the homogeneous theta is kept above it. Either changes the synthetic
truth, so those runs must be redone without `--resume`.

### Practical

* Inverted-EC file (section 5, first assumption): `output_model/ec_optimal.npy`.
* A K = 0 run took about 8.5 min here, so section 3 times (4 min per run) are about half: the 17-run
  error study (`--ws homo`) is about 2.4 h.
* Quick test (`--fast`, smc homo, so affected by the floor): with a, m known, sigma_w stayed on its
  upper bound 1000, and a -20 % error lowered the RMSE (0.19 -> 0.11). An error on a known value can
  offset the EC error and look like an improvement.
* Tests: `test_study_swc_design.py` and `test_ws_start_values.py` at the repo root expect to live in
  `tests/`, which holds older copies. `tests/data/scenarios_golden.json` is missing.
