# Test plan for patch 0007 (branch `todo/train-val-synthetic`)

Status: **steps 0-5 run on Ben's machine on 2026-11-18** (results below). Open: the two excluded test files, the `full` rows and repeats with other seeds (section *Still open*).

Run from the repo root, with the conda env that has `emagpy`. Everything is copy-paste. Paste text, not screenshots.

```bash
cd ~/Nextcloud/BenCSIC/Codes/Tech4agro_org/mfc2_EA-EM_TDR
OUT=$(python EA_EM_config.py --output-root)      # where runs are written (mfc2_outputs by default)
mkdir -p /tmp/mfc2_check
git log --oneline -8 | tee /tmp/mfc2_check/00_git.txt
git status -sb | tee -a /tmp/mfc2_check/00_git.txt
```

`OUT` is a shell variable: define it again in every new terminal, and run all commands from the repo root (`mfc2_outputs` is a relative path).

## Results at a glance

| Step | What | Result |
|---|---|---|
| 1 | unit tests | 260 passed (without `test_scenarios_golden.py`, `test_sweeps.py`) |
| 2 | scenarios load | OK: `by_probe` / 6 positions on the `_tv_probe` scenarios, `[True, False, False, False] truth` on the `_pin0` ones |
| 3 | regression `1d_surface_tdr` | exit 0, 0 tracebacks, final RMSE **0.031212** (equal to the recorded baseline), new `ws_params.npy` and `tdr_train_*.npy` present |
| 4 | train/val `1d_surface_free2_tv_probe` | 9 train / 9 val of 18; train RMSE 0.0105, val RMSE 0.0766 (K=8); divergence warning fires |
| 5 | TDR-pinned, uniform truth | pinTDR beats free at H1-H3; level with the oracle (table below) |
| 5 | TDR-pinned, heterogeneous truth | pinTDR beats free at H0-H2; level with the oracle (table below) |

---

## 1. Unit tests (~20 s)

```bash
python -m pytest tests -q --ignore=tests/test_scenarios_golden.py --ignore=tests/test_sweeps.py 2>&1 | tee /tmp/mfc2_check/01_pytest.txt | tail -5
```

Expected: about 260 passed, no failures. Skips are only normal if `emagpy` is missing.
Once `tests/data/scenarios_golden.json` is committed (see `.gitignore` note in `to-do.md`), run without the two `--ignore` flags. **Not yet run on Ben's machine.**

## 2. Scenarios load (~5 s)

```bash
python - <<'PY' 2>&1 | tee /tmp/mfc2_check/02_scenarios.txt
import scenario_loader as sl
for n in ("1d_surface_free2_tv_probe", "1d_full_free2_tv_probe",
          "1d_surface_free2_pin0", "1d_surface_hetero_free2_pin0"):
    c = sl.load_scenario(n)
    print(n, "| split:", c.train_val_split, "| pos:", c.num_swc_positions,
          "| pin:", c.pin_layers, c.pin_source)
PY
```

Expected: `by_probe` / `6` on the two `_tv_probe` scenarios, `[True, False, False, False] truth` on the two `_pin0` scenarios.

## 3. Regression: existing results must not change (~12 min)

```bash
python run_pipeline.py --scenario 1d_surface_tdr --mode 1d --num-swc-positions 5 > /tmp/mfc2_check/03_baseline.log 2>&1
echo "exit code: $?" | tee /tmp/mfc2_check/03_exit.txt
RUN=$(ls -d "$OUT"/field_survey_1d_surface_tdr/[0-9]*_* | tail -1); echo "$RUN"
grep -a "Final RMSE (Step 5)" "$RUN"/pipeline_report_*.md | tee /tmp/mfc2_check/03_rmse.txt
ls "$RUN/output_model" | tee /tmp/mfc2_check/03_files.txt
grep -a -c -i "traceback" /tmp/mfc2_check/03_baseline.log | tee -a /tmp/mfc2_check/03_exit.txt
```

Expected: exit code 0, traceback count 0, final RMSE **0.031212** (compare only within one environment), and the new files `ws_params.npy`, `tdr_train_theta.npy`, `tdr_train_positions.npy`, `tdr_train_times.npy`, `tdr_train_layers.npy` in `output_model/`.
If the RMSE differs, the Step 5 change had a side effect: diff the log against a run from `main`.

## 4. Train/validation example (~12 min)

```bash
python run_pipeline.py --scenario 1d_surface_free2_tv_probe --mode 1d > /tmp/mfc2_check/04_tv.log 2>&1
echo "exit code: $?" | tee /tmp/mfc2_check/04_exit.txt
tr '\r' '\n' < /tmp/mfc2_check/04_tv.log | grep -a "\[SPLIT\]\|\[STEP 5b\]" | cut -c1-200 | tee /tmp/mfc2_check/04_split.txt
```

Expected: `by_probe (independent=True): 9 train / 9 val observations (of 18 total)`, a train RMSE of about 0.01-0.02, a val RMSE of about 0.08 and the warning that val diverges (>30 %) from train.
Reference values: Ben's machine 0.0105 / 0.0766; sandbox 0.0155 / 0.085 (values vary a little by environment).
Caveat: 3 probes per side, so the K selected (K=8) is not robust; read the warning as working, not as a result about K.
Optional in-sample reference: same scenario with `--train-val-split off`.

## 5. TDR-pinned experiment (~12 + 5 min per scenario)

Pass 1 (free) and pass 2 (layer 0 pinned to the EC derived from TDR + calibrated WS) are run by `run_pinned_tdr.py`; the oracle is a separate run pinned to the true EC.

```bash
# uniform truth, K = 2 free WS parameters, surface-only TDR
python run_pinned_tdr.py --scenario 1d_surface_free2 --mode 1d --num-swc-positions 5 > /tmp/mfc2_check/05_pinned.log 2>&1
echo "exit code: $?" | tee /tmp/mfc2_check/05_exit.txt
tr '\r' '\n' < /tmp/mfc2_check/05_pinned.log | grep -a "\[pinned-tdr\]\|\[PIN\]\|Layer pinning" | cut -c1-220 | tee /tmp/mfc2_check/05_pin.txt
python run_pipeline.py --scenario 1d_surface_free2_pin0 --mode 1d --num-swc-positions 5 > /tmp/mfc2_check/05_oracle.log 2>&1

# heterogeneous truth (needs K >= 2)
python run_pinned_tdr.py --scenario 1d_surface_hetero_free2 --mode 1d --num-swc-positions 5 > /tmp/mfc2_check/05h_pinned.log 2>&1
python run_pipeline.py --scenario 1d_surface_hetero_free2_pin0 --mode 1d --num-swc-positions 5 > /tmp/mfc2_check/05h_oracle.log 2>&1
```

Optional `full` rows (about 12 min each, separate runs): `1d_full_free2` and `1d_full_hetero_free2`, both with `--mode 1d --num-swc-positions 5`.

### Compare

**Important:**
- The pinned scenario `<name>_pintdr` is NOT in `ensemble_scenarios.json` (it lives in `<pass 1 run>/pass2_scenarios.json`), so `compare_pinned.py` cannot resolve it by name. Pass the **run folders** instead (or `--config <pass2_scenarios.json>` together with the scenario name).
- Folders accumulate (`0001_`, `0002_`, ...). Use the latest of each, and check that the `free` run is the pass 1 that wrote `pass2_scenarios.json`:
  `ls -t "$OUT"/field_survey_1d_surface_hetero_free2/*/pass2_scenarios.json | head -1`
- Always redirect stderr (`2>&1`), otherwise a failure leaves the `.md` empty and the traceback only on screen.

```bash
latest() { ls -d "$OUT"/field_survey_"$1"/[0-9]*_* | tail -1; }

python compare_pinned.py \
  free="$(latest 1d_surface_free2)" \
  pinTDR="$(latest 1d_surface_free2_pintdr)" \
  oracle="$(latest 1d_surface_free2_pin0)" \
  --csv /tmp/mfc2_check/05_uniform.csv --plot /tmp/mfc2_check/05_uniform.png \
  2>&1 | tee /tmp/mfc2_check/05_uniform.md

python compare_pinned.py \
  free="$(latest 1d_surface_hetero_free2)" \
  pinTDR="$(latest 1d_surface_hetero_free2_pintdr)" \
  oracle="$(latest 1d_surface_hetero_free2_pin0)" \
  --csv /tmp/mfc2_check/05_hetero.csv --plot /tmp/mfc2_check/05_hetero.png \
  2>&1 | tee /tmp/mfc2_check/05_hetero.md
```

Add `full="$(latest 1d_full_free2)"` (or `1d_full_hetero_free2`) if you ran it.

### Results (Ben, 2026-11-18, one realisation, 5 positions)

RMSE (bias) vs truth, all horizons:

| truth | run | K | theta RMSE [m3/m3] | EC RMSE [mS/m] |
|---|---|---|---|---|
| uniform | free | 3 | 0.154 (+0.114) | 51.0 (+35.5) |
| uniform | pinTDR | 48 | 0.121 (+0.092) | 29.4 (+21.6) |
| uniform | oracle | 26 | 0.127 (+0.100) | 29.4 (+20.3) |
| hetero | free | 7 | 0.180 (+0.140) | 54.6 (+34.7) |
| hetero | pinTDR | 47 | 0.154 (+0.119) | 35.9 (+22.0) |
| hetero | oracle | 26 | 0.162 (+0.131) | 38.0 (+22.2) |

Per horizon at H1 (the layer just below the pinned one): theta 0.199 -> 0.153 (uniform) and 0.230 -> 0.178 (hetero); EC 57.2 -> 31.6 and 86.1 -> 54.7 mS/m (free -> pinTDR).

### Success criterion

- **Met:** pinTDR beats free in the deep horizons (H1-H3 uniform, H0-H2 hetero for theta; every deep horizon for EC).
- **Reworded:** the original "sits between free and oracle" is not met strictly. pinTDR is level with the oracle (differences 0.001-0.008 theta, within the noise of one realisation); read it as "pinTDR is not worse than the oracle by more than about 0.01 theta".
- **Not solved by pinning:** the wet bias in H1-H3 stays at +0.06 to +0.22 m3/m3 in all three runs (deep layers use the lab prior).
- **To check:** the oracle is worse than pinTDR for theta in H0 (0.069 vs 0.012) although its H0 EC is exact (look at `pin_from_tdr`); K differs a lot between rows (3 / 48 / 26, 7 / 47 / 26), so part of the gap may be member selection.

---

## Still open

| What | How | Why |
|---|---|---|
| `tests/test_scenarios_golden.py`, `tests/test_sweeps.py` | commit `tests/data/scenarios_golden.json` (fix `.gitignore`), then `python -m pytest tests -q` | only part of step 1 not yet run |
| `full` rows (uniform and hetero) | see step 5 | reference row for the 4-row table |
| Repeats with other seeds | loop `run_pinned_tdr.py` + `compare_pinned.py --csv` per seed, average the CSVs | one realisation only; K differs between rows |
| Level 3 (per-member EC draw for the pinned layer) | `to-do.md`, pinned-horizon item, step 4 | keeps the uncertainty honest |

## Bundle and send

```bash
cd /tmp/mfc2_check && tar czf mfc2_check.tgz *.txt *.md *.csv *.png 2>/dev/null; ls -la mfc2_check.tgz
```

Logs (`*.log`) are large; send them only if something failed, then the last 80 lines of the relevant one:

```bash
tr '\r' '\n' < /tmp/mfc2_check/0X_name.log | tail -80
```

| Step | Send | Why |
|---|---|---|
| 0 | `00_git.txt` | confirm the patches are applied |
| 1 | last 5 lines of pytest (+ failures) | code works on your machine |
| 2 | `02_scenarios.txt` | scenarios load |
| 3 | exit code, final RMSE, file list | no regression |
| 4 | `[SPLIT]` / `[STEP 5b]` lines | train/val example works |
| 5 | `[PIN]` lines, `05_uniform.md`, `05_hetero.md` (+ CSV, PNG) | the pinned result reproduces |
