# Showing that the real-case dataset works with the pipeline (quick path)

Status: written from the repo (README, scenarios, plotting code) on 2026-11-13. The field data was not available where this was written, so **none of the commands below has been run on the real data yet** (`--quick` is documented as "not yet run on the real data" in `to-do.md`).

## 1. Command lines, cheapest first

Run from the repo root with the field data in `./data/`. The real scenarios in `ensemble_scenarios_real.json` already set `em_data_dir ./data/EMI_MFC2/`, `tdr_data_dir ./data/daily_MFC/` and the coordinate file, so `--em-dir` / `--tdr-dir` / `--coord-file` are only needed to override them (the examples below show them for completeness).

```bash
# 1. Plumbing check: 1 alpha, 1 forward model, 3 epochs  ->  12 inversion tasks
./run_pipeline.sh --data-source real -s real_single_test_probes \
    -c ensemble_scenarios_real.json --mode 2d --quick --quick-epochs 3 \
    --em-dir ./data/EMI_MFC2/ --tdr-dir ./data/daily_MFC/ \
    --coord-file ./data/SOILNET_MFC2_coordinates_sensors.csv

# 2. Same, with a regularly thinned map (use if you also want to show maps)
./run_pipeline.sh --data-source real -s real_single_test_grid \
    -c ensemble_scenarios_real.json --mode 2d --quick --quick-epochs 3 \
    --em-dir ./data/EMI_MFC2/ --tdr-dir ./data/daily_MFC/ \
    --coord-file ./data/SOILNET_MFC2_coordinates_sensors.csv

# 3. Whole real-case script, reduced
./run_real_PAPER.sh --quick --quick-epochs 3
```

- `real_single_test_probes` keeps only the EM points within a radius of the TDR probes: fastest, enough to test calibration (Steps 2/4/5), not the maps.
- `real_single_test_grid` thins on an absolute regular grid (same cells in every epoch); better for maps.
- Check the log for the `[QUICK]` banner: the task count should be **12** with `--quick-epochs 3` (**28** with `--quick` alone).
- `--quick` is a smoke test: the ensemble itself is shrunk, so do not quote its numbers as results. It writes its own `_quick` run folder and cache and never overwrites a full-run cache.
- Add `--train-val-split by_probe` if you want the held-out validation figure (see 2.3).

For presentable results use the two-stage run (README, "Split for the final runs"): `--stage calibrate` (probe points, full ensemble), then `--stage map` (selected alpha / forward model / K only). Also not yet run on real data.

## 2. Figures to look at, in order

1. **`swc_stages/one_to_one_ABC_<scenario>.png`** - estimated SWC vs the TDR sensors at the probe locations, for three stages: A raw inverted members, B averaged models, C optimal model. RMSE / bias / R2 per stage. This is the direct "estimated SMC (optimal model) vs TDR" comparison and already exists.
2. **`petro_calibration_optimal.png`** - EC of the optimal model at the probes vs TDR theta, with the fitted Waxman-Smits curve.
3. **`parity_train_vs_val.png`** - train / validation parity at the chosen K. Only written with `--train-val-split` (e.g. `by_probe`). This is the honest test, because the validation probes are not used for calibration.
4. **`figure_3_rmse_curve.png`** - RMSE vs number of stacked members, shows the chosen K.
5. **`solution/solution_swc_profiles.png`** and **`solution/solution_swc_uncertainty.png`** - final SWC profiles and the ensemble-spread uncertainty band (new, Step 6f).
6. **`water_content_xy_*.png`** - plan-view SWC maps.
7. Input checks: `probe_em_matching*.png` (probe to EM point matching), the Step 1d plots (raw ECa, initial EC / WS ranges), `plot_tdr_diagnostics.py` (raw TDR time series).

## 3. Caveats

- **`parity_validation.png` is not the optimal-model comparison.** It uses only the single best-ranked member's fit (`best_fit.predicted_theta`). Use `one_to_one_ABC` instead.
- `plot_tdr_diagnostics.py` still plots the **raw** TDR, not the filtered series (open item in `to-do.md`).
- Horizons L3 / L4 have no TDR sensor (only 15 / 30 cm), so they fall back to the prior Waxman-Smits parameters; none of the figures above checks them.
- The SWC uncertainty band is an ensemble spread, not a confidence interval. On a synthetic surface-only run the truth was inside +-1 / +-2 sigma only 27 % / 60 % of the time.
- Acceptance check still open (README / `to-do.md`): a `grid` / `probes` run should reproduce the full run's optimal EC and theta at the probe-matched points, and the matching report should not get worse.

## 4. `run_real_study.py` and `compare_scenarios.py` on the real data

### `run_real_study.py` - layer design x Waxman-Smits freedom (real data only)
Answers "how many layers can the real TDR sensors support, and how many WS parameters can be freed?". It copies `real_single_base`, loops over layer designs (`L2` = [0.225], `L3` = [0.225, 0.6], `L4` = [0.3, 0.6, 1.0]) and K = 1..4 free WS parameters, writes `mfc2_outputs/real_study/study_scenarios.json`, runs `./run_pipeline.sh` once per scenario and collects `run_log.csv` rows into `study_summary.csv`. All K of one design share one output dir and Step 1 cache, so the expensive inversion runs once per design.

```bash
python run_real_study.py --fast --dry-run     # plan + sensor coverage per horizon, runs nothing (works without data)
python run_real_study.py --fast               # smoke test: --quick, 2 epochs, EM points within 5 m of probes, K = 1 and 4
python run_real_study.py --designs L2 --free-k 1 4   # a small real subset
python run_real_study.py                      # full study (long)
```

- Use `--fast --dry-run` first: it checked out here (6 scenarios: L2/L3/L4 x K=1,4) and shows which horizons have no sensor - L3 H2 and L4 H2, H3 have none, so those fall back to the prior WS parameters.
- `--fast` is a smoke test (calibration works, maps do not): not a result.
- It needs `./data` and `emagpy`; per `to-do.md` it has not been run end to end on the real data.
- For a quick demonstration that the real data works, `run_real_study.py --fast --designs L2 --free-k 1` is a one-scenario alternative to section 1.

### `compare_scenarios.py` - compare finished runs
It only reads the `pipeline_report_*.md` files (plus optional CSVs) of runs already made, so it works on real runs too, but only part of it applies:

| works on real runs | synthetic only (needs truth) |
|---|---|
| `comparison_table.csv/.md` (final RMSE, CV, K, ensemble size, probes, matching fraction, runtime) | `fig_water_content_error.png` (returns None for real runs) |
| `fig_rmse_and_cv.png`, `fig_ensemble_and_probes.png` | `--position`: `fig_depth_profile.png`, `fig_spatial_transect.png`, `fig_scatter_vs_truth.png`, `truth_error_summary.csv` |
| `fig_eca_rmse_by_epoch.png`, `fig_step_timing.png` | `--scoring`: `fig_truth_scoring.png` |

```bash
python compare_scenarios.py --ids real_study_L2_free1:1 real_study_L2_free4:1 \
    --labels L2_free1 L2_free4 --out-dir ./comparison_output/real_L2
```
(`--ids` accepts run numbers, `scenario:number`, or full run ids from `mfc2_outputs/run_log.csv`; or use `./compare_scenarios.sh`.)

Real-data caveats: with no truth, "Final RMSE (Step 5)" is the fit to the TDR probes the run was given, so it is not a fair way to compare designs with different probe coverage (the same warning as in the 1-D surface-vs-full notes). Prefer the held-out validation RMSE (`--train-val-split by_probe`) when comparing real runs. Under `--quick`, scenarios that differ only by forward model collapse to one model, so such comparisons are only a plumbing check.
