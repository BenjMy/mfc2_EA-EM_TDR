# bench_ws_global_vs_localized.py

Compares **global** (one Waxman-Smits set per horizon) and **localized**
(Gaspari-Cohn kernel, one set per EM point) calibration/mapping on synthetic mfc2
scenarios with known truth.

One pipeline run already produces both maps: `theta_calc.npy` (localized) and
`theta_calc_global.npy`, next to the truth `synthetic_time_lapse_theta.npy`.
Each scenario is therefore run **once** and both maps are scored against the same truth.
Nothing in the project is modified; only `<run>/output_model/*.npy` is read.

## Requirements

- Run from the repository root, in the environment that runs `run_pipeline.py`
  (needs `emagpy`, `numpy`, `pandas`, `matplotlib`).
- Branch with `--ws-map-calibration` **and** `--k-selection` (the runner passes both).
  Apply the patch if needed: `git am 0001-k-selection-localized-loo.patch`.

## Quick start

```bash
python bench_ws_global_vs_localized.py generate               # write + validate scenario JSON
python bench_ws_global_vs_localized.py run --quick --dry-run  # print the commands, run nothing
python bench_ws_global_vs_localized.py all --only xy_random_L4_P4 --quick   # one scenario first
python bench_ws_global_vs_localized.py all --design oat --quick             # full OAT design
python bench_ws_global_vs_localized.py score                  # re-score existing runs
```

| Command    | What it does |
|------------|--------------|
| `generate` | Writes `<out-root>/ws_bench_scenarios.json` (the repo scenarios plus the generated ones, via `extends`) and `design.json`; validates every scenario with `scenario_loader`. |
| `run`      | Calls `run_pipeline.py --data-source synthetic --mode 2d --ws-map-calibration localized --k-selection <mode>` per scenario. Skips scenarios that already have a run unless `--rerun`. |
| `score`    | Scores the latest run of each scenario; writes the CSV and figure. |
| `all`      | `generate`, `run`, `score`. |

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--design {oat,full}` | `oat` | `oat`: one factor at a time around the baseline (8 runs). `full`: cartesian product (36 runs). |
| `--out-root PATH` | `mfc2_outputs/ws_bench` | Where the JSON, runs, CSV and figure go. |
| `--corr-m FLOAT` | `4.0` | Correlation length [m] of the `xy_random` truth. |
| `--k-selection {global,localized-loo}` | `global` | Passed to `run_pipeline.py`. |
| `--only STR ...` | all | Run only scenarios whose name contains one of these strings. |
| `--quick` | off | Passed to `run_pipeline.py`. |
| `--rerun` | off | Re-run scenarios that already have a run. |
| `--dry-run` | off | Print the run commands only. |
| `--realization INT` | `0` | Truth ensemble member used for scoring. |

## Design

Factors (edit `FACTORS` and `BASELINE` at the top of the script):

- **`ws_truth`**: `homogeneous` (extends `2d`), `per_horizon` (extends `2d_horizon_ws`, depth trend:
  m rises, sigma_s falls), `xy_linear` and `xy_random` (extend `2d_xy_waxman`; m and sigma_s vary
  in x/y, smoothly or in random patches).
- **`n_layers`**: 2, 3, 4 (`depths0`, `cond_ranges` and the per-layer prior are adapted).
- **`n_probes`**: TDR positions per layer (2, 4, 8).

Baseline: `xy_random`, 4 layers, 4 probes. Scenario names: `wsb_<truth>_L<layers>_P<probes>`.
The localized bandwidth is picked per run by leave-one-probe-out over `BANDWIDTH_SWEEP`
(2, 5, 10, 20, 40).

What to expect: on `homogeneous` / `per_horizon` truth, localized should be about equal to global;
on `xy_*` truth it should win away from the probes up to roughly the correlation length, and
fall back to the global fit where the nearest probe is beyond 2c.

## Outputs (in `--out-root`)

- `ws_global_vs_localized.csv`: one row per run (columns below).
- `ws_global_vs_localized.png`: left, RMSE global vs localized (below the diagonal = localized
  better); right, gain per run for all points and for points far from probes.
- `design.json`, `ws_bench_scenarios.json`, and one `<scenario>/<seq>_<scenario>/` folder per run.

CSV columns:

- `rmse_*`, `bias_*` for `global` / `local`: over all points; `_near_`: within c of the nearest
  probe; `_far_`: beyond 2c (where localized used the global fit); `_L<i>_`: per horizon.
- `gain = rmse_global - rmse_local` (> 0: localized better); `gain_far` likewise on far points.
- `bandwidth`, `fallback_frac`, `K`, `K_global`, `n_probes_loc`.
- `param_<a|m|sigma_w|sigma_s>_rmse_{global,local}`: WS parameter error, only when
  `synthetic_ws_truth_*.npy` exist (the `xy_*` truths).

## Notes and caveats

- A run without `theta_calc_global.npy` (no bandwidth, or the localized fit failed) is skipped and logged.
- With `--k-selection localized-loo`, whether the ensemble inversion is
  re-done depends on the Step 1 cache, which I have not verified. Use a separate `--out-root` for it.
- Truth is the noise-free generated field; scores are over all EM points, not only the probes.
- The runner always uses `--mode 2d` (the scenarios it extends are 2-D).
- Score at `--realization 0` unless you generate several truth realizations.
