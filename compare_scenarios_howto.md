# Comparing scenario performance with `auto_compare.py`

Branch: `feat/localized-ws-maps` of `BenjMy/mfc2_EA-EM_TDR`. Written from reading the code; **no command below was run** (no field data, no `emagpy` env here; `auto_compare.py` was only tested with `--dry-run` on a fake run log).

## How it works

`comparison_output/` is the default output folder of `compare_scenarios.py`, which reads finished runs (`pipeline_report_*.md`, plus `output_model/*.npy` for truth plots).
`auto_compare.py` (repo root) removes the manual step: it takes the **latest finished run of each scenario** matching your patterns in a run-log CSV and calls `compare_scenarios.py` with the ids and labels.

```bash
python auto_compare.py --list ...      # show the selected runs, run nothing
python auto_compare.py --dry-run ...   # print the compare_scenarios.py command
python auto_compare.py ...             # run it
```

| Option | Meaning |
|---|---|
| `--preset swc\|mirror` | log file, scenario patterns and defaults for these two cases |
| `--match PATTERN ...` | glob patterns on the scenario name (overrides the preset's) |
| `--log-file FILE` | run-log CSV (default: shared `mfc2_outputs/run_log.csv`) |
| `--max-runs N` | refuse larger comparisons (default 12) |
| anything else (`--position`, `--transect-y`, `--out-dir`, ...) | forwarded to `compare_scenarios.py` |

Failed runs are skipped. Setup: activate the `eaem` env, run from the repo root.

## Optional: pick the runs yourself with ids

If you want a specific run (not the latest) or a hand-picked set, skip `auto_compare.py` and give the ids to `compare_scenarios.sh`:

```bash
./compare_scenarios.sh --list                              # run numbers, scenarios, output dirs
./compare_scenarios.sh --ids <idA> <idB> --labels A B --out-dir comparison_output/A_vs_B
./compare_scenarios.sh --ids <idA> <idB> --labels A B --position 8 0    # synthetic runs only: truth plots
```

An id can be a run number (`7`), `scenario:number` (`1d_surface_tdr:3`, needed when the same number exists in several scenarios), or a full `run_id` from the log. For the swc study add `--log-file mfc2_outputs/run_log_swc_study.csv`. The per-case sections below show the `auto_compare.py` form; the ids form works for each of them.

## What each case gives

| Case | `auto_compare` | Truth plots (`--position`) |
|---|---|---|
| 1. Real dataset, hypotheses H1-H3 | `--match` | No (no truth). "Final RMSE" is in-sample unless `--train-val-split by_probe`. |
| 2. Synthetic MFC2-like | `--preset mirror` | **No**: runs as `data_source: real`; truth in `mfc2_synthetic/ground_truth/` is never read. I found no script that scores against it. |
| 3. Synthetic 1-D | `--match` | **Yes** (depth profile, transect, scatter/residual vs truth). |
| 4. Benchmark `bench_ws_global_vs_localized.py` | not applicable | Own scoring (below). |
| 5. `swc_design_study` | `--preset swc` | Yes (preset adds `--position 8 0`). |

Report-level outputs for every case: `comparison_table.csv/.md`, `fig_rmse_and_cv.png`, `fig_ensemble_and_probes.png`, `fig_eca_rmse_by_epoch.png`, `fig_step_timing.png`. With `--position` on synthetic runs also: `fig_water_content_error.png`, `fig_depth_profile.png`, `fig_spatial_transect.png`, `fig_scatter_vs_truth.png`, `truth_error_summary.csv`.

## 1. Real dataset, hypotheses

```bash
python run_all_hypotheses.py --fast --dry-run              # plan first; --fast = smoke test, not a result
python run_all_hypotheses.py --only 1 2 3
# per-hypothesis tables: mfc2_outputs/hypotheses/<hyp>/hyp<N>_summary.csv

# compare (scenario names from the README: real_study_<design>_free<K>, real_tv_<method>, real_bw_sweep)
python auto_compare.py --match 'real_study_*' --list       # H1 (3 designs x 4 K = 12 runs = the cap)
python auto_compare.py --match 'real_study_L2_free*' --out-dir comparison_output/real_L2
python auto_compare.py --match 'real_tv_*'                 # H2
python auto_compare.py --match 'real_bw_sweep*'            # H3
```

## 2. Synthetic MFC2-like dataset

```bash
python generate_synthetic_real_mirror.py --standin-stations 20 --depths0 0.225 \
    --max-points 120 --outdir mfc2_synthetic
python run_all_hypotheses.py --config mfc2_synthetic/mfc2_real_synthetic_mirror.json \
    --base mfc2_real_synthetic_mirror --fast

python auto_compare.py --preset mirror --list
python auto_compare.py --preset mirror
```

The preset matches `mfc2_real_synthetic_mirror*`, `real_study_*`, `real_tv_*`, `real_bw_sweep*`. Do not pass `--position`: it needs `output_model/synthetic_*.npy`, which this dataset does not write.

## 3. Synthetic 1-D

```bash
./run_one_scenario.sh -s 1d_surface_tdr    --mode 1d --num-swc-positions 5
./run_one_scenario.sh -s 1d_full_depth_ref --mode 1d --num-swc-positions 5
python auto_compare.py --match 1d_surface_tdr 1d_full_depth_ref --position 8 0 \
    --out-dir comparison_output/1d_surface_vs_full

# whole study (8 runs, K = 1..4), ~20 min; SKIP_RUNS=1 redoes only the comparisons
./run_1d_surface_vs_full.sh
python auto_compare.py --match '1d_surface_free*' '1d_full_free*' --position 8 0 \
    --out-dir comparison_output/1d_all

# optional 1:1 validation against truth
python plot_validation_1to1.py --ids <surface_id> <full_id> --labels surface full \
    --out-dir comparison_output/validation
```

Only the latest run of each scenario is used: if you looped over several probe counts, only the last one is compared.

## 4. Benchmark study (global vs localized WS maps)

Not a `compare_scenarios` job: one run already holds both maps (`theta_calc.npy` localized, `theta_calc_global.npy`) next to the truth, and the script scores them itself.

```bash
python bench_ws_global_vs_localized.py all --design oat --quick \
    --only wsb_xy_random_L4_P4 --n-free 1 2 3 4 --bandwidth 5 \
    --seeds 21 22 23 \
    --out-root mfc2_outputs/ws_bench_nfree           # --quick = smoke test; 4 K x 3 seeds = 12 runs

# re-score existing runs: repeat the SAME options and --out-root (scenario names and design.json depend on them)
python bench_ws_global_vs_localized.py score \
    --only wsb_xy_random_L4_P4 --n-free 1 2 3 4 --bandwidth 5 \
    --seeds 21 22 23 \
    --out-root mfc2_outputs/ws_bench_nfree
```

`--seeds` sets `ws_xy_seed` (a new realisation of the random WS field per seed); `score` then reports mean and sd over the seeds in `ws_global_vs_localized_agg.csv`. Use one `--out-root` per batch and a different one for `--quick` and full runs.

Outputs in `--out-root`: `ws_global_vs_localized.csv`, `ws_global_vs_localized_agg.csv`, `ws_global_vs_localized.png`, `ws_error_by_distance.csv/.png`.

## 5. `swc_design_study`

```bash
python study_swc_design.py --dry-run                  # write + validate scenarios, print commands
python study_swc_design.py --preset smoke --fast      # smoke test (8 runs)
python study_swc_design.py --preset oat               # 15 runs
python study_swc_design.py --analyze-only             # redo the study's own summary + figures

python auto_compare.py --preset swc --list
python auto_compare.py --preset swc --smc hetero --ws hetero --reg noreg --dist spread --n 5 --k 1 2 3 4
python auto_compare.py --preset swc --smc homo --ws homo --reg noreg --dist spread --k 2 --n 1 2 3 5 10
```

The preset reads `mfc2_outputs/run_log_swc_study.csv` and adds `--position 8 0` (`--no-position` to skip). Filters: `--smc --ws --reg --dist --n --k`. Output: `comparison_output/swc_design_study/auto_<filters>/`.
The study's own figures (`study_summary.csv`, `study_report.md`, `fig_rmse_vs_nswc.png`, `fig_rmse_vs_k.png`, `fig_heatmap_n_vs_k.png`, ...) are written to `comparison_output/swc_design_study/`.

## Caveats

- **Real and mirror runs can mix.** Both use the same hypothesis scenario names and the same shared log, so the tool cannot tell them apart (the latest wins). Check `--list`, use a specific `--match`, or keep them in separate logs with `--log-file`.
- **`--stage both` logs two runs under one scenario name** (calibrate, then map). The latest, usually the map run, is the one compared.
- Scenario names for the real and mirror hypotheses are taken from the README, not from an actual log: confirm with `--list`.
