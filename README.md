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
| Optimal EC_bulk profiles vs truth | `<run>/solution/solution_ec_bulk_profiles.png` |
| Optimal water content (θ) profiles vs truth | `<run>/solution/solution_swc_profiles.png` |
| Cross-run RMSE/CV, θ error, depth profile, transect, scatter vs truth | `comparison_output/<name>/fig_*.png` |
| ECa 1:1 vs noise-free truth; EC and θ 1:1 per horizon | `comparison_output/validation/fig_eca_1to1.png`, `fig_ec_1to1_by_horizon.png`, `fig_theta_1to1_by_horizon.png` |

Saved arrays for synthetic runs (`<run>/output_model/`): `theta_calc.npy`, `K.npy`, `rmse.npy`, `cv_map.npy`, `is_calibrated.npy`, plus the `synthetic_*` truth arrays. `ec_optimal.npy` is written only by runs made after `mfc2_all_changes.patch` was applied (older runs don't have it).

See `refactoring.md` for the code map and cleanup plan.
