# Showing that the real-case dataset works with the pipeline (quick path)

Status: written from the repo (README, scenarios, plotting code) on 2026-11-13. The field data was not available where this was written, so **none of the commands below has been run on the real data yet** (`--quick` is documented as "not yet run on the real data" in `to-do.md`).

## 1. Command lines, cheapest first

Replace the data paths with yours (defaults in the README: `./data/EMI_MFC2/`, `./data/daily_MFC/`, `./data/SOILNET_MFC2_coordinates_sensors.csv`).

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
