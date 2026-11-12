# mfc2_EA-EM_TDR

EA-EM pipeline: ensemble EM (ECa) inversion + TDR-calibrated Waxman-Smits petrophysics → soil water content. Works on synthetic scenarios and on real MFC2 field data.

Outputs go to `mfc2_outputs/<scenario output_dir>/<NNNN>_<scenario>/`; every run is appended to `mfc2_outputs/run_log.csv`.

## Run one scenario

```bash
./run_pipeline.sh --list                          # list scenarios
./run_pipeline.sh --validate                      # validate ensemble_scenarios.json
./run_pipeline.sh -s 1d_surface_tdr --mode 1d --num-swc-positions 5   # synthetic, 1-D transect
./run_pipeline.sh -s 2d --mode 2d                 # synthetic, 2-D grid
./run_pipeline.sh --mode 2d -s 2d_real \
    --em-dir ./data/EMI_MFC2/ --tdr-dir ./data/daily_MFC/             # real data
./run_pipeline.sh --help                          # all options
```

Scenarios are defined in `ensemble_scenarios.json` (synthetic) and `ensemble_scenarios_real.json` (real). A scenario can start from another one with `"extends": "<name>"` (nested objects are merged, lists/scalars replaced); `ws_layer_params` sets per-layer true Waxman-Smits parameters for synthetic data.

Tests: `pytest tests`.

## Compare runs

```bash
./compare_scenarios.sh --list                     # recent runs (run_id, output_dir)
./compare_scenarios.sh --ids <run_id_A> <run_id_B> \
    --labels A B --position 8 0 --out-dir comparison_output/A_vs_B
```

Writes `comparison_table.csv/.md` and the `fig_*.png` comparison figures into `--out-dir`.

## Surface-only vs full-depth TDR study (1-D)

```bash
./run_1d_surface_vs_full.sh            # 8 runs (K = 1..4 free WS params) + comparisons, ~20 min
SKIP_RUNS=1 ./run_1d_surface_vs_full.sh   # redo only the comparisons
VARIANT=hetero ./run_1d_surface_vs_full.sh   # same study, two-zone vertical truth (a, sigma_w change below horizon 1)
```

Results: `comparison_output/1d_surface_vs_full_free{K}/` and `comparison_output/1d_surface_vs_full_all/`.

## Truth-vs-optimal 1:1 validation (synthetic runs)

```bash
python plot_validation_1to1.py --ids <surface_run_id> <full_run_id> \
    --labels surface full --out-dir comparison_output/validation
```

## Main plots

| Plot | Where |
|---|---|
| ECa observed vs forward-simulated optimal (1:1 and per coil) | `<run>/solution/eca_forward_optimal/eca_forward_1to1_t??.png`, `eca_forward_vs_observed_t??.png` |
| WS parameters (a, m, sigma_w, sigma_s) truth vs calibration initial value, per horizon (synthetic runs) | `<run>/synthetic_ws_truth_vs_initial.png` (`plots/ws_distribution.py`) |
| Initial EC range (ECmin/ECmax) vs TDR-implied EC, histogram + XY probe map, one column per horizon (real runs) | `<run>/initial_model/initial_ec_range_by_horizon.png` (`plots/initial_ec_ws_real.py`, Step 1d) |
| Initial WS values (a, m, sigma_w, sigma_s) per horizon, position inside the bounds, fixed/free, on-bound flag (real runs) | `<run>/initial_model/initial_ws_by_horizon.png` (same module) |
| Optimal EC_bulk profiles vs truth | `<run>/solution/solution_ec_bulk_profiles.png` |
| Optimal water content (θ) profiles vs truth | `<run>/solution/solution_swc_profiles.png` |
| Cross-run RMSE/CV, θ error, depth profile, transect, scatter vs truth | `comparison_output/<name>/fig_*.png` |
| ECa 1:1 vs noise-free truth; EC and θ 1:1 per horizon | `comparison_output/validation/fig_eca_1to1.png`, `fig_ec_1to1_by_horizon.png`, `fig_theta_1to1_by_horizon.png` |

Saved arrays for synthetic runs (`<run>/output_model/`): `theta_calc.npy`, `K.npy`, `rmse.npy`, `cv_map.npy`, `is_calibrated.npy`, plus the `synthetic_*` truth arrays. `ec_optimal.npy` is written only by runs made after `mfc2_all_changes.patch` was applied (older runs don't have it).

See `refactoring.md` for the code map and cleanup plan.

## Real-data runs: initial EC range, initial WS, and how big the ensemble is

### Initial ECmin/ECmax and initial WS for real runs (implemented)

The synthetic plot `synthetic_ws_truth_vs_initial.png` needs a truth, so it is not drawn for real data. For real runs Step 1d now also writes, in `<run>/initial_model/`:

* `initial_ec_range_by_horizon.png`: one column per horizon. Top row: histogram of the EC_bulk implied by the TDR theta of that horizon and the **initial** WS values, `EC_init = (sigma_w + sigma_s) * theta^m / a`, with ECmin/ECmax (`cond_ranges`) as vertical lines. Bottom row: XY map of the TDR probes coloured by EC_init (time mean) on the scale [ECmin, ECmax], a red "x" marks a probe outside the range. The title counts how many TDR-implied values fall outside the range.
* `initial_ws_by_horizon.png`: one panel per horizon, the four initial WS values as a position inside their calibration bounds, `[fixed]`/`[free]` from `ws_fixed_params`, red edge + "ON BOUND" when a value sits on a bound.

Initial WS values and ECmin/ECmax are constants per horizon, so an XY map of them would be a single colour; the XY map shows what varies in space (the TDR-implied EC). Horizon of a TDR depth = position among `depths0` (e.g. `depths0=[0.225]`: 15 cm -> H0, 30 cm -> H1). Tests: `tests/test_initial_ec_ws_real.py`. The plot code is pure numpy/matplotlib and never aborts the run.

Observation for `real_*_base` (`lab_p0 = [0.5, 3.0, 500, 100]`): `a = 0.5` sits exactly on the default lower bound of `a` (`WS_BOUNDS`). It is harmless while `a` is fixed (`real_single_base`), but in any real scenario that frees `a` it would pile up on the bound (see the free4 case in `to-do.md`); the new WS figure flags it.

### Is the ensemble generated at the TDR points or at all points? (answered)

At **all** EM points. Step 1 dispatches one inversion task per (alpha, forward model, starting-conductivity combination, epoch), and each task inverts the whole filtered survey file. For `real_single_base`: 5 alphas x 2 forward models x 4 starting combinations (2 layers, ECmin/ECmax) x 7 epochs = 280 tasks, and the 7 filtered surveys hold 83 827 / 29 048 / 64 409 / 55 770 / 21 191 / 45 513 / 41 319 points (341 077 per combination). The TDR probes (20) enter only afterwards: they are matched to the nearest EM points (`window_m`) to calibrate Waxman-Smits in Steps 2/4; the inverted EC at all other points is converted to theta with the calibrated parameters. That is why the run is slow, and why the 20 probes matter for calibration but not for the cost.

Whether the ensemble can be restricted to the TDR points without changing the result at those points: with `beta = 0` and `gamma = 0` (what the real scenarios use) I expect every EM point to be inverted independently, so the inversion at a probe's matched point should be identical in a full and in a reduced run. I did not run this; the acceptance test below checks it.

### Skimming the dataset for testing: `em_subsample` (implemented)

Scenario field `em_subsample` (module `em_subsample.py`), `null`/`{"mode": "off"}` = every filtered point. It thins each EM survey **after** `preprocess_em_survey` (the rolling mean needs the full sequence) and **before** the inversion and `match_epochs_to_reference_grid`; the processed CSV (regenerated from the raw file on every run) is rewritten in place, rows keep their order.

| mode | keys | what it keeps |
|---|---|---|
| `grid` | `cell_m` | one point per `cell_m` x `cell_m` cell, the one closest to the cell centre. The grid is absolute (origin 0 in the survey CRS), so the same cells are chosen in every epoch and the epochs still overlap (random thinning would not). Keep `cell_m` < `spatial_match_tolerance_m` (3 m). Best general choice. |
| `probes` | `radius_m` | points within `radius_m` of a TDR probe. Fastest; enough to test calibration (Steps 2/4/5), not the maps. The nearest EM point of 16 of the 20 probes was 1.1-3.2 m away in the last run, so use `radius_m >= 5`. |
| `stride` | `stride` | every n-th point (smoke test). |
| `random` | `max_points`, `seed` | random subset (smoke test). |

Common key `keep_probe_nearest` (default `true`): the EM point nearest to each TDR probe is always kept. The setting is part of the Step 1 cache fingerprint (added only when thinning is on, so existing caches keep their hash), is printed in the scenario banner and the report config table, and the per-file point counts appear in the preprocessing table ("Final pts" = after thinning). Ready-made scenarios: `real_single_test_grid` (`grid`, 3 m) and `real_single_test_probes` (`probes`, 5 m), both copies of `real_single_base` with their own `output_dir`/`cache_tag`.

```bash
./run_pipeline.sh -s real_single_test_grid   -c ensemble_scenarios_real.json
./run_pipeline.sh -s real_single_test_probes -c ensemble_scenarios_real.json
```

Caveat: with a lateral-smoothing `beta > 0` (CLI `--beta`) thinning changes the neighbourhoods the smoothing acts on; for `beta = 0` points are independent. Not implemented, on purpose: a `--quick` preset that cuts alphas / forward models / epochs. The ensemble (alpha x forward model x starting conductivities) is the scientific object of the method, so shrinking it changes what is being tested; thinning the points does not.

### Why does the pipeline currently use all points? (my reading of the code, not a documented decision)

There is no step that needs all points for the calibration itself, so I would call it a consequence of how the code grew, not a requirement. Plausible reasons:

* The pipeline was built on synthetic 1-D/2-D grids where every point is cheap, and its products are full maps (`water_content_xz/xy`, `cv_map`, `is_calibrated`, per-horizon EC/theta maps): those need every point.
* Step 1 produces one `models_clean` array on the union reference grid, and Steps 2-5 read the probe-matched points out of it, so calibration and mapping share one run and one cache.
* The probe match averages EM points inside `window_m` (1 m) around each probe (nearest-point fallback otherwise), so a few neighbours of every probe are needed; that argues for `radius_m >= window_m`, not for all 341 077 points.
* Diagnostics (matching plots, footprint, transects) are drawn on the whole survey.

The cost, not the science, is what pushes toward a split (next section).

### Recommended split for the final runs (future)

Calibration only needs the probe-matched points; the maps need all points. Plan: `--stage calibrate` (`em_subsample.mode = "probes"`, all alphas/forward models) to select K and WS parameters, then `--stage map` (all or grid-thinned points, selected alpha/forward model/K only) applying them. Today both are one run.

### Optional: calibrate Waxman-Smits by zone (future, clustering method to be discussed)

For real data it would be convenient to optionally cluster the field into zones and run one WS calibration per zone (instead of one set of parameters per horizon for the whole field). How to cluster is open. Candidates to discuss:

* **Geographic / management zones** (plots, crops, user-drawn polygons): simplest, easy to explain, needs a polygon file.
* **ECa-based clustering** (k-means / Gaussian mixture on time-mean ECa per coil, as in precision agriculture): data-driven, but circular, because the zones are defined by the same quantity that is then converted to theta.
* **Soil-map / texture-based** (external soil map, clay content): physically closest to what a, m and sigma_s represent.
* **Temporal-stability based** (zones with persistent ECa pattern across the 7 epochs).

**Possibly redundant with `ws_localized_bandwidth_sweep`.** The localized Step 2 fit (`run_petro_localized`) already lets the WS parameters vary in space: it fits a separate parameter set at every target location, weighting every TDR observation of the horizon by a Gaspari-Cohn kernel of its distance (`bandwidth -> inf` = the global per-horizon fit, `bandwidth -> 0` = nearest probe), and the bandwidth is chosen by leave-one-probe-out CV. Zones are the hard-partition version of the same idea. Where they differ:

* Localized fit: no clustering decision, one hyperparameter chosen by CV, smooth variation. But it only knows distance: it smooths across an abrupt soil boundary and cannot pool probes from two separate areas that share the same soil.
* Zones: can use a covariate (ECa, soil map, texture) so non-adjacent areas share parameters and sharp boundaries are respected, with more probes pooled per fit and parameters that are easy to interpret. But they add the clustering choice and need enough probes per zone.
* Both can be combined (kernel weights multiplied by a same-zone indicator, or a distance that includes the covariate), at the price of more complexity.

Suggested decision rule before building anything: look at what the bandwidth CV selects on the real data. If it picks a bandwidth much larger than the field, there is no spatial structure to exploit from 20 probes and zones are not worth it; if it picks a small one, there is structure, and zones are worth it only if a covariate or a boundary explains it that distance alone cannot. I have not seen that CV output for the real runs.

Things to settle with it: minimum number of TDR probes per zone (20 probes in total, so only a few zones; free parameters per zone must stay below what the probes can constrain); what a zone without a probe falls back to (global parameters or nearest zone); leave-one-probe-out CV must respect the zones; relation to the existing continuous alternatives (`ws_xy_variation`, localized Gaspari-Cohn fit `ws_localized_bandwidth_sweep`). Sketch: scenario field `ws_zones` (`null` | `{"method": "polygons"|"kmeans_eca"|"file", ...}`) producing a zone label per reference-grid point, and Steps 2/4 calibrating per (horizon, zone).

### Acceptance tests for the above

* A `grid` or `probes` run reproduces, at the probe-matched points, the optimal EC and theta of the full run (same `ec_optimal`, `theta_calc` within solver tolerance).
* Probe-to-EM matching report is unchanged (same matched indices/distances) or only gets closer.
* Cache fingerprint changes when `em_subsample` changes.

### Notes from the 2026-11-12 real run log

* Banner prints "--mode not given; defaulting to '1d' for data_source='synthetic'" (twice) in a real run, although the run is 2D and real: cosmetic, to remove.
* The scenario banner shows "Forward model : CS" while the parameter grid sweeps FSlin and CS.
* Epoch 4 (`5_ALENTO_300421.csv`) covers only 25.8 % of the union reference grid, epoch 1 36.3 %; 50.2 % of the EC cells in `models_clean` become NaN before layers are considered. Thinning on a common grid should improve this.

