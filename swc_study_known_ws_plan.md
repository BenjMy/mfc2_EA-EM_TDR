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

### Patch 2b - EC diagnostics (only if the K=0 floor is large)  -- DONE as stand-alone scripts

* `diagnose_ec_inversion.py` (EMagPy only, ~8 min): tests the three candidate causes one at a time on
  the study truth and reports data misfit, inverted/true EC per layer and the implied theta RMSE
  (= the K = 0 floor seen from Step 1). `ws_param_check.py` (numpy only): EC range, floor,
  a / sigma_w / sigma_s degeneracy and sensitivity of the truth parameters.
  Tests: `tests/test_diagnose_ec_inversion.py`.
* Result: see section 6 ("EC diagnosis"). The cause is `method = ROPE`.
* `ec_rmse_truth` / `ec_bias_rel` in `compute_metrics` are NOT added (the scripts answer the question
  outside the pipeline); add them only if you want the EC error in every study summary.

### Patch 2d - EM method (`--em-method`) and soil-like truth (`--ws soil*`)

* `em_options.py`: one switch for the EM inversion method, as an ordinary scenario field
  (`param_grid.method`), no pipeline change. `base` (default) = the scenario's own method (ROPE today),
  `lbfgsb` = L-BFGS-B. Nelder-Mead is not offered (EMagPy ignores its bounds), TNC did not fit.
* `study_swc_design.py`: new factor `em` (`--em-method base lbfgsb`; the name gets `_em<tag>` only when
  not base, so every existing run name / cache / log row is unchanged; figures per method in
  `em_<tag>/` when several methods are present).
* New truth cases `--ws soilhomo soilheterox soilheteroz`: same structure as homo / heterox / heteroz
  with the baseline (a, m, sigma_w, sigma_s) = (0.8, 2, 100, 100) instead of (0.5, 3, 500, 100);
  the m / sigma_s ranges scale with the baseline; `lab_p0` (calibration prior) is the baseline.
* Hypothesis runs (`hyp1/2/3_*.py`, via `hyp_common`) and `run_real_study.py`: `--em-method lbfgsb`
  writes `<tag>_lbfgsb_scenarios.json` / `<tag>_lbfgsb_summary.csv` with `_lbfgsb` appended to
  `cache_tag` / `output_dir` (same scenario names).
* Any other scenarios file (real or synthetic): `python em_options.py --config ensemble_scenarios_real.json
  --em-method lbfgsb` writes `ensemble_scenarios_real_lbfgsb.json`; then
  `REAL_CFG=ensemble_scenarios_real_lbfgsb.json ./run_real_all_scenarios.sh` (the only shell change: `REAL_CFG`
  can now be set from the environment) or `./run_one_scenario.sh -c <file> -s <name>`.
* Tests: `tests/test_em_options.py`.

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

1. Patch 1 and run the 4 floor runs. **Decision**: floor small or large.  -> large (0.13-0.15)
2. Patch 2a (+2c) on the runs you already have; this also gives the "truth" lines in the plots.
3. Patch 2b / 2d: cause = ROPE. Redo the floor runs with `--em-method lbfgsb` (legacy truth, and soil truth):
   `python study_swc_design.py --smc hetero --ws homo soilhomo --reg noreg --free-k 0 --n-swc 5 --dist spread --em-method base lbfgsb`
   (the `base` rows are your existing ROPE runs; `--resume` skips them if their folders exist).
   **Decision**: floor < 0.03 with L-BFGS-B -> continue; otherwise look at the top layer / fixed depths.
4. Patches 3, 4, 5, 6 in this order (all accept `--em-method`); run each preset before writing the next.
   Real-case and hypothesis runs: switch with `--em-method lbfgsb` (hyp1/2/3, run_real_study) or a derived
   scenarios file (`em_options.py`).

## 5. Assumptions to confirm

* Inverted-EC file name in `output_model/` (Patch 2b): `ec_optimal.npy` (confirmed).
* Soil truth values (0.8, 2, 100, 100) are my choice from one published sandy-loam case; replace them
  (`SOIL_WS_BASELINE` in `study_swc_design.py`) with values from your site / lab data if you have them.
* What to do with `smc = homo` (floor still 32-33 % with the soil truth): raise the homogeneous theta, or
  keep `smc = hetero` only.
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
| 2b | done (stand-alone scripts; cause = ROPE) |
| 2d | done (`--em-method`, soil truth cases); the 4 K = 0 floor runs have to be redone with `--em-method lbfgsb` |
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

### EC diagnosis (2026-11-26, `diagnose_ec_inversion.py`, smc hetero / ws homo, one realization, EMagPy 1.4.6)

Ensemble mean over the step-1 ensemble; misfit = RMS relative ECa error of forward(inverted);
theta RMSE = theta from the inverted EC with the TRUE WS parameters (the K = 0 floor from Step 1 alone).

| variant | misfit % | theta RMSE | EC ratio inverted / true, layers 0-3 |
|---|---|---|---|
| baseline (as the study: ROPE, free depths, FSlin + CS) | 604 | 0.153 | 8.4, 6.6, 6.2, 2.1 |
| fixed_depths = true | 678 | 0.158 | 10.6, 6.8, 5.5, 2.3 |
| forwardModel CS only / FSlin only | 625 / 552 | 0.160 / 0.148 | 7.4, 6.9, 7.2, 1.7 / 7.7, 5.6, 6.0, 2.1 |
| **method L-BFGS-B** | **5.4** | **0.022** | 1.25, 1.0, 0.82, 1.02 |
| method Nelder-Mead / TNC | 5.5 / 195 | 0.031 / 0.068 | |
| fixed depths + CS + L-BFGS-B | 7.8 | 0.022 | 1.4, 1.18, 1.27, 0.83 |
| start at the truth (L-BFGS-B) | 2.5 | 0.016 | about 1 |

* Cause: `method = ROPE`. With `rep = 100` random samples (EMagPy default, never set by the pipeline) it
  does not fit the data (misfit ~600 %). Best member only: still ~440 %. `rep = 1000`: 220 %
  (`--tests T6`, slow). Narrower bounds help only partly (0.074-0.078).
* Not causes: fixed depths and the forward model (no effect); conventions (truth reproduces the data to
  0.9 % = noise level; unit `ppt` only applies to quad / inph columns).
* L-BFGS-B: top layer still 25-40 % too conductive; theta RMSE 0.022 is below 0.03 but not by much.
  With free depths the members spread by a factor ~2 in EC (0.3 in log10) and the best-misfit member
  (theta RMSE 0.042) is worse than the mean (0.022); fixed depths spread 0.05 and best = 0.016.
* The pipeline's own default method is L-BFGS-B (`run_pipeline.py`); ROPE comes from the scenario files.
  alpha / beta / gamma work with L-BFGS-B (EMagPy applies them in every `minimize` method); the schema
  comment saying ROPE is the only optimiser that supports them is wrong for this. With L-BFGS-B the file
  `mcmc_conductivity_pstd.npy` is not written (nothing reads it).
* To check on a pipeline run: `ec_optimal` is the K-cluster average chosen in Step 4 by the TDR probes
  (not by data misfit); redo the 4 K = 0 floor runs with `--em-method lbfgsb` and compare.

### Waxman-Smits truth parameters (`ws_param_check.py`)

* In the generator's form `EC = theta^m / a * (sigma_w + sigma_s)` only `C = a / (sigma_w + sigma_s)` and `m`
  matter: a, sigma_w, sigma_s are exactly degenerate (same C -> identical EC). Judge recovery on C and m.
* theta error at theta = 0.30 for a +-10 % wrong value (EC exact): m +-0.037, a +-0.010, sigma_w -+0.008,
  sigma_s -+0.002.
* Realism: sigma_s = 100 and F(0.44) = 5.9 are in the range of a published sandy-loam case (F 4.5,
  sigma_s 109 mS/m, saturation exponent 1.9); sigma_w = 500 mS/m (saline) and m = 3 are on the high side.
  Hence the soil cases: (a, m, sigma_w, sigma_s) = (0.8, 2, 100, 100); `a` = 0.8 (not 1.0) so that the
  start of a free `a` (1.0) is 25 % off the truth.
* Share of true EC at the 1 mS/m floor (one realization each, measured with the generator):

| smc / ws | legacy truth | soil truth |
|---|---|---|
| homo / homo | 66.7 % (median 1.00) | 33.3 % (median 1.59) |
| homo / heterox | 60.0 % | 31.7 % |
| hetero / homo | 3.3 % | 1.7 % |
| hetero / heterox | 0.0 % | 0.0 % |

  The soil truth halves the floor but does NOT remove it for `smc = homo`: conclusions from `smc = homo`
  stay compromised (decision below). `smc = hetero` is fine with either truth.

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
