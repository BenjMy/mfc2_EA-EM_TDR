# Running the pipeline: real vs synthetic (cheat sheet)

Run everything from the repository root. Two datasets, same pipeline:

| | **real** | **synthetic** (mirror of the real deployment, known truth) |
|---|---|---|
| input | `./data/` (EMI_MFC2, daily_MFC, SOILNET_MFC2_coordinates_sensors.csv) | `mfc2_synthetic/` (made by `generate_synthetic_real_mirror.py`) |
| scenarios file | `ensemble_scenarios_real.json` | `mfc2_synthetic/mfc2_real_synthetic_mirror.json` |
| base scenario | `real_single_base` | `mfc2_real_synthetic_mirror` |
| hypothesis outputs | `mfc2_outputs/hypotheses/` | `mfc2_outputs/hypotheses_synthetic/` |

> The synthetic files are read through the real-data loader, so the pipeline banner
> says `REAL DATA` for both. The hypothesis scripts print a `DATA:` line that says
> which one it really is. In the plain launcher, look at the `EM=` path.

## 0. One-time: create the synthetic dataset

```bash
python generate_synthetic_real_mirror.py --standin-stations 20 --depths0 0.225 \
    --max-points 120 --outdir mfc2_synthetic
```
(Already committed in `mfc2_synthetic/`; re-run only to change it.)

## 1. One scenario, directly (no wrapper, no generated files)

```bash
# real
./run_one_scenario.sh -s real_single_base      -c ensemble_scenarios_real.json                   --mode 2d --stage calibrate
# synthetic
./run_one_scenario.sh -s mfc2_real_synthetic_mirror -c mfc2_synthetic/mfc2_real_synthetic_mirror.json --mode 2d --stage calibrate
```
Add `--quick --quick-epochs 3` for a smoke test (not a result; cannot be combined with `--stage map`).

## 2. The hypothesis tests: ONE switch, `--data real|synthetic`

```bash
python run_all_hypotheses.py --data real      --fast --dry-run     # plan only (shows DATA: ...)
python run_all_hypotheses.py --data synthetic --fast --dry-run

python run_all_hypotheses.py --data synthetic --fast               # smoke test
python run_all_hypotheses.py --data synthetic                      # full ensemble (long)
python run_all_hypotheses.py --data real                           # same on the field data

# a single hypothesis
python hyp1_layers_ws_freedom.py --data synthetic --designs L2 --free-k 0 1 2 3 4
python hyp3_localized_bandwidth.py --data synthetic
```
Extra flags for `run_one_scenario.sh` go after a bare `--`, e.g. `-- --em-njobs 8`.

`--data` sets `--config` and `--base` for you and keeps the synthetic outputs in
their own folder, so real and synthetic results never overwrite each other.
`--config <mirror json>` without `--data` is detected as synthetic automatically.

Note: the mirror only has TDR at 15 and 30 cm, so use `--designs L2` (L3 / L4 have
horizons without a sensor).

## 3. Calibrate, then map (final runs)

```bash
./run_one_scenario.sh -s real_L3_base -c ensemble_scenarios_real.json --mode 2d --stage calibrate
tail -n 3 mfc2_outputs/run_log.csv | cut -d, -f1,3,6,7          # find the calibrate folder
./run_one_scenario.sh -s real_L3_base -c ensemble_scenarios_real.json --mode 2d --stage map \
    --calibration mfc2_outputs/real_L3_base/0001_real_L3_base_calibrate --em-njobs 8
```
Or let a hypothesis script chain both: `--stage both`.

## 4. Score against truth (synthetic only)

```bash
./compare_scenarios.sh --list
./compare_scenarios.sh --ids <scenario>:1 --position 5 0
```
