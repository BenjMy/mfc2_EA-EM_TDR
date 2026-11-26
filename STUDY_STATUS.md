# Study status

Logs: `mfc2_outputs/run_log.csv`, `mfc2_outputs/run_log_swc_study.csv`  (322 runs). HEAD: `5e05ea34`.

| study | expected | done | failed | missing | extra |
|---|---|---|---|---|---|
| surface_vs_full | 8 | 0 | 0 | 8 | 0 |
| surface_vs_full_hetero | 8 | 0 | 0 | 8 | 0 |
| hyp1_layers_ws_freedom | 12 | 7 | 0 | 5 | 0 |
| hyp2_train_val_split | 5 | 5 | 0 | 0 | 0 |
| hyp3_localized_bandwidth | 1 | 1 | 0 | 0 | 0 |
| real_scenarios | 16 | 1 | 0 | 15 | 0 |
| swc_design_study | 15 | 15 | 0 | 0 | 288 |

## surface_vs_full

Matrix: run_1d_surface_vs_full.sh (uniform truth)

**To run** (missing, then failed):

```bash
./run_one_scenario.sh -s 1d_surface_free1 --mode 1d --num-swc-positions 5
./run_one_scenario.sh -s 1d_full_free1 --mode 1d --num-swc-positions 5
./run_one_scenario.sh -s 1d_surface_free2 --mode 1d --num-swc-positions 5
./run_one_scenario.sh -s 1d_full_free2 --mode 1d --num-swc-positions 5
./run_one_scenario.sh -s 1d_surface_free3 --mode 1d --num-swc-positions 5
./run_one_scenario.sh -s 1d_full_free3 --mode 1d --num-swc-positions 5
./run_one_scenario.sh -s 1d_surface_free4 --mode 1d --num-swc-positions 5
./run_one_scenario.sh -s 1d_full_free4 --mode 1d --num-swc-positions 5
```

## surface_vs_full_hetero

Matrix: VARIANT=hetero ./run_1d_surface_vs_full.sh

**To run** (missing, then failed):

```bash
./run_one_scenario.sh -s 1d_surface_hetero_free1 --mode 1d --num-swc-positions 5
./run_one_scenario.sh -s 1d_full_hetero_free1 --mode 1d --num-swc-positions 5
./run_one_scenario.sh -s 1d_surface_hetero_free2 --mode 1d --num-swc-positions 5
./run_one_scenario.sh -s 1d_full_hetero_free2 --mode 1d --num-swc-positions 5
./run_one_scenario.sh -s 1d_surface_hetero_free3 --mode 1d --num-swc-positions 5
./run_one_scenario.sh -s 1d_full_hetero_free3 --mode 1d --num-swc-positions 5
./run_one_scenario.sh -s 1d_surface_hetero_free4 --mode 1d --num-swc-positions 5
./run_one_scenario.sh -s 1d_full_hetero_free4 --mode 1d --num-swc-positions 5
```

## hyp1_layers_ws_freedom

Matrix: python hyp1_layers_ws_freedom.py  (default K = 1 2 3 4; with --fast only K = 1 and 4, so K = 2, 3 reported missing after a --fast run is expected)

**To run** (missing, then failed):

```bash
python hyp1_layers_ws_freedom.py --designs L3 --free-k 3
python hyp1_layers_ws_freedom.py --designs L4 --free-k 1
python hyp1_layers_ws_freedom.py --designs L4 --free-k 2
python hyp1_layers_ws_freedom.py --designs L4 --free-k 3
python hyp1_layers_ws_freedom.py --designs L4 --free-k 4
```

**Caveats**

- 3 scenario(s) finished with quality WARN: real_study_L3_free1, real_study_L3_free2, real_study_L3_free4
- RMSE rating no skill / unsatisfactory (RMSE not clearly below the std of the SWC): real_study_L2_free1, real_study_L2_free2, real_study_L2_free3, real_study_L2_free4, real_study_L3_free1, real_study_L3_free2 ...
- Runs come from 2 different code versions (2d165a48, 7da186a2): results may not be comparable.
- Some runs were made with uncommitted changes (+dirty): not reproducible from git.
- No run was made with the current HEAD: re-run before drawing conclusions on the new code.
- 7 scenario(s) use a partial stage (calibrate only): no maps: real_study_L2_free1, real_study_L2_free2, real_study_L2_free3, real_study_L2_free4, real_study_L3_free1 ...

## hyp2_train_val_split

Matrix: python hyp2_train_val_split.py

**Caveats**

- 5 scenario(s) finished with quality WARN: real_tv_alternate_sorted, real_tv_by_probe, real_tv_by_time_block, real_tv_off, real_tv_random
- RMSE rating no skill / unsatisfactory (RMSE not clearly below the std of the SWC): real_tv_alternate_sorted, real_tv_by_probe, real_tv_by_time_block, real_tv_off, real_tv_random
- Runs come from 2 different code versions (5e05ea34, 7da186a2): results may not be comparable.
- Some runs were made with uncommitted changes (+dirty): not reproducible from git.
- 5 scenario(s) use a partial stage (calibrate only): no maps: real_tv_alternate_sorted, real_tv_by_probe, real_tv_by_time_block, real_tv_off, real_tv_random

## hyp3_localized_bandwidth

Matrix: python hyp3_localized_bandwidth.py

**Caveats**

- 1 scenario(s) finished with quality WARN: real_bw_sweep
- RMSE rating no skill / unsatisfactory (RMSE not clearly below the std of the SWC): real_bw_sweep
- Runs come from 2 different code versions (5e05ea34, 7da186a2): results may not be comparable.
- Some runs were made with uncommitted changes (+dirty): not reproducible from git.
- 1 scenario(s) use a partial stage (calibrate only): no maps: real_bw_sweep

## real_scenarios

Matrix: run_real_all_scenarios.sh (all scenarios of ensemble_scenarios_real.json)

**To run** (missing, then failed):

```bash
./run_one_scenario.sh -s real_single_base --mode 2d
./run_one_scenario.sh -s real_single_fslin --mode 2d
./run_one_scenario.sh -s real_single_cs --mode 2d
./run_one_scenario.sh -s real_single_depth_free --mode 2d
./run_one_scenario.sh -s real_single_ws_all_fixed --mode 2d
./run_one_scenario.sh -s real_single_ws_free_sigmaw --mode 2d
./run_one_scenario.sh -s real_single_ws_free_sigmas --mode 2d
./run_one_scenario.sh -s real_single_ws_free_all --mode 2d
./run_one_scenario.sh -s real_multi_base --mode 2d
./run_one_scenario.sh -s real_multi_fslin --mode 2d
./run_one_scenario.sh -s real_multi_cs --mode 2d
./run_one_scenario.sh -s real_multi_depth_free --mode 2d
./run_one_scenario.sh -s real_multi_ws_all_fixed --mode 2d
./run_one_scenario.sh -s real_multi_ws_free_sigmaw --mode 2d
./run_one_scenario.sh -s real_multi_ws_free_sigmas --mode 2d
```

**Caveats**

- One run per scenario, one noise realisation: no repeats, so differences smaller than the run-to-run spread cannot be interpreted.
- 1 scenario(s) finished with quality WARN: real_L3_base
- RMSE rating no skill / unsatisfactory (RMSE not clearly below the std of the SWC): real_L3_base
- Some runs were made with uncommitted changes (+dirty): not reproducible from git.
- No run was made with the current HEAD: re-run before drawing conclusions on the new code.
- 1 scenario(s) use a partial stage (calibrate only): no maps: real_L3_base

## swc_design_study

Matrix: python study_swc_design.py --preset oat  (own log run_log_swc_study.csv)

Extra scenarios in the log (not in the matrix): swcstudy_smchetero_wshetero_beta_clustered_n2_free1, swcstudy_smchetero_wshetero_beta_clustered_n2_free2, swcstudy_smchetero_wshetero_beta_clustered_n2_free3, swcstudy_smchetero_wshetero_beta_clustered_n2_free4, swcstudy_smchetero_wshetero_beta_clustered_n3_free1, swcstudy_smchetero_wshetero_beta_clustered_n3_free2, swcstudy_smchetero_wshetero_beta_clustered_n3_free3, swcstudy_smchetero_wshetero_beta_clustered_n3_free4, swcstudy_smchetero_wshetero_beta_clustered_n5_free1, swcstudy_smchetero_wshetero_beta_clustered_n5_free2 ...

**Caveats**

- One run per scenario, one noise realisation: no repeats, so differences smaller than the run-to-run spread cannot be interpreted.
- 252 scenario(s) finished with quality WARN: swcstudy_smchetero_wshetero_beta_clustered_n2_free2, swcstudy_smchetero_wshetero_beta_clustered_n2_free3, swcstudy_smchetero_wshetero_beta_clustered_n2_free4, swcstudy_smchetero_wshetero_beta_clustered_n3_free2, swcstudy_smchetero_wshetero_beta_clustered_n3_free3, swcstudy_smchetero_wshetero_beta_clustered_n3_free4 ...
- RMSE rating no skill / unsatisfactory (RMSE not clearly below the std of the SWC): swcstudy_smchetero_wshetero_beta_clustered_n2_free1, swcstudy_smchetero_wshetero_beta_clustered_n3_free1, swcstudy_smchetero_wshetero_beta_clustered_n5_free1, swcstudy_smchetero_wshetero_beta_random_n2_free1, swcstudy_smchetero_wshetero_beta_random_n3_free1, swcstudy_smchetero_wshetero_beta_random_n5_free1 ...
- Runs come from 3 different code versions (2d165a48, 5e05ea34, 7da186a2): results may not be comparable.
- Some runs were made with uncommitted changes (+dirty): not reproducible from git.
- 162 scenario(s) use WS start values outside the bounds / on a bound (ws_start_warnings): swcstudy_smchetero_wshetero_beta_clustered_n2_free1, swcstudy_smchetero_wshetero_beta_clustered_n2_free2, swcstudy_smchetero_wshetero_beta_clustered_n2_free3, swcstudy_smchetero_wshetero_beta_clustered_n3_free1, swcstudy_smchetero_wshetero_beta_clustered_n3_free2 ...
- 303 scenario(s) use a partial stage (calibrate only): no maps: swcstudy_smchetero_wshetero_beta_clustered_n2_free1, swcstudy_smchetero_wshetero_beta_clustered_n2_free2, swcstudy_smchetero_wshetero_beta_clustered_n2_free3, swcstudy_smchetero_wshetero_beta_clustered_n2_free4, swcstudy_smchetero_wshetero_beta_clustered_n3_free1 ...
