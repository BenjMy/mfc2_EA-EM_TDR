# Test plan after applying patch 0007 (branch `todo/train-val-synthetic`)

Run from the repo root, with the conda env that has `emagpy`.
Everything is copy-paste. Each step says **what to send back**; paste the text, not screenshots.
Steps 1-3 are quick (under 15 min total). Steps 4-5 are optional experiments (about 12 + 5 min each).

```bash
cd ~/Nextcloud/BenCSIC/Codes/Tech4agro_org/mfc2_EA-EM_TDR
OUT=$(python EA_EM_config.py --output-root)      # where runs are written (mfc2_outputs by default)
mkdir -p /tmp/mfc2_check
git log --oneline -8 | tee /tmp/mfc2_check/00_git.txt
git status -sb | tee -a /tmp/mfc2_check/00_git.txt
git log -1 --format='%an <%ae>' | tee -a /tmp/mfc2_check/00_git.txt
```

**Send back:** `/tmp/mfc2_check/00_git.txt` (confirms what is applied and the author).

---

## 1. Unit tests (~20 s)

```bash
python -m pytest tests -q --ignore=tests/test_scenarios_golden.py --ignore=tests/test_sweeps.py 2>&1 | tee /tmp/mfc2_check/01_pytest.txt | tail -5
```

Expected (my sandbox): about 258 passed (256 after patch 6, plus 2 in `test_compare_pinned.py`). Skips are only normal if `emagpy` is missing.

If you committed `tests/data/scenarios_golden.json`, also run without the two `--ignore` flags.

**Send back:** the last 5 lines, and the full text of any failure.

---

## 2. Scenarios load (~5 s)

```bash
python - <<'EOF' 2>&1 | tee /tmp/mfc2_check/02_scenarios.txt
import scenario_loader as sl
for n in ("1d_surface_free2_tv_probe", "1d_full_free2_tv_probe",
          "1d_surface_free2_pin0", "1d_surface_hetero_free2_pin0"):
    c = sl.load_scenario(n)
    print(n, "| split:", c.train_val_split, "| pos:", c.num_swc_positions,
          "| pin:", c.pin_layers, c.pin_source)
EOF
```

Expected: the four names with `by_probe` / `6` on the two `_tv_probe` scenarios and `[True, False, False, False] truth` on the two `_pin0` scenarios.

**Send back:** the printed lines, or the traceback.

---

## 3. Regression: existing results must not change (~12 min)

The Step 5 change only adds files, so the baseline run must match the number in `refactoring.md`.

```bash
python run_pipeline.py --scenario 1d_surface_tdr --mode 1d --num-swc-positions 5 > /tmp/mfc2_check/03_baseline.log 2>&1
echo "exit code: $?" | tee /tmp/mfc2_check/03_exit.txt
RUN=$(ls -d "$OUT"/field_survey_1d_surface_tdr/[0-9]*_* | tail -1); echo "$RUN"
grep -a "Final RMSE (Step 5)" "$RUN"/pipeline_report_*.md | tee /tmp/mfc2_check/03_rmse.txt
ls "$RUN/output_model" | tee /tmp/mfc2_check/03_files.txt
grep -a -c -i "traceback" /tmp/mfc2_check/03_baseline.log | tee -a /tmp/mfc2_check/03_exit.txt
```

Expected:
- exit code 0, traceback count 0.
- `final_rmse` about **0.031212** (baseline recorded for your machine; compare only within one environment).
- `output_model/` contains the new files `ws_params.npy`, `tdr_train_theta.npy`, `tdr_train_positions.npy`, `tdr_train_times.npy`, `tdr_train_layers.npy`.

**Send back:** `03_exit.txt`, `03_rmse.txt`, `03_files.txt`.
If `final_rmse` differs, the Step 5 change had a side effect, so send also the first 40 lines of the `03_baseline.log` diff against a run from `main` before the patches.

---

## 4. Train/validation example (optional, ~12 min)

```bash
python run_pipeline.py --scenario 1d_surface_free2_tv_probe --mode 1d > /tmp/mfc2_check/04_tv.log 2>&1
echo "exit code: $?" | tee /tmp/mfc2_check/04_exit.txt
tr '\r' '\n' < /tmp/mfc2_check/04_tv.log | grep -a "\[SPLIT\]\|\[STEP 5b\]" | cut -c1-200 | tee /tmp/mfc2_check/04_split.txt
```

Expected (my sandbox): `by_probe (independent=True): 9 train / 9 val observations (of 18 total)`; Train RMSE about 0.0155, Val RMSE about 0.085, plus a warning that val diverges from train. Your numbers may differ a little by environment.

Optional in-sample reference (same scenario, no split):

```bash
python run_pipeline.py --scenario 1d_surface_free2_tv_probe --mode 1d --train-val-split off > /tmp/mfc2_check/04b_off.log 2>&1
```

**Send back:** `04_exit.txt`, `04_split.txt`.

---

## 5. TDR-pinned experiment (optional, ~12 + 5 min per scenario)

Uniform truth, K = 2 free WS parameters, surface-only TDR:

```bash
python run_pinned_tdr.py --scenario 1d_surface_free2 --mode 1d --num-swc-positions 5 > /tmp/mfc2_check/05_pinned.log 2>&1
echo "exit code: $?" | tee /tmp/mfc2_check/05_exit.txt
tr '\r' '\n' < /tmp/mfc2_check/05_pinned.log | grep -a "\[pinned-tdr\]\|\[PIN\]\|Layer pinning" | cut -c1-220 | tee /tmp/mfc2_check/05_pin.txt
```

Oracle and full-depth rows (for the 4-row table; each about 12 min for `full`, 5 min for `pin0` reusing nothing, since they are separate runs):

```bash
python run_pipeline.py --scenario 1d_surface_free2_pin0 --mode 1d --num-swc-positions 5 > /tmp/mfc2_check/05_oracle.log 2>&1
python run_pipeline.py --scenario 1d_full_free2         --mode 1d --num-swc-positions 5 > /tmp/mfc2_check/05_full.log 2>&1
```

Comparison table:

```bash
python compare_pinned.py free=1d_surface_free2 pinTDR=1d_surface_free2_pintdr \
    oracle=1d_surface_free2_pin0 full=1d_full_free2 --csv /tmp/mfc2_check/05_uniform.csv \
    | tee /tmp/mfc2_check/05_uniform.md
```

(Drop `oracle=` / `full=` if you did not run them.)

Same for the heterogeneous truth (needs K >= 2, hence free2):

```bash
python run_pinned_tdr.py --scenario 1d_surface_hetero_free2 --mode 1d --num-swc-positions 5 > /tmp/mfc2_check/05h_pinned.log 2>&1
python run_pipeline.py --scenario 1d_surface_hetero_free2_pin0 --mode 1d --num-swc-positions 5 > /tmp/mfc2_check/05h_oracle.log 2>&1
python run_pipeline.py --scenario 1d_full_hetero_free2         --mode 1d --num-swc-positions 5 > /tmp/mfc2_check/05h_full.log 2>&1
python compare_pinned.py free=1d_surface_hetero_free2 pinTDR=1d_surface_hetero_free2_pintdr \
    oracle=1d_surface_hetero_free2_pin0 full=1d_full_hetero_free2 --csv /tmp/mfc2_check/05_hetero.csv \
    | tee /tmp/mfc2_check/05_hetero.md
```

Expected for the uniform pair (my sandbox, one realisation): all-horizon theta RMSE 0.152 (free) vs 0.119 (pinTDR); EC RMSE at H1 57.8 vs 30.9 mS/m. Success criterion from `to-do.md`: pinTDR beats free at H1-H3 and sits between free and oracle.

**Send back:** `05_exit.txt`, `05_pin.txt`, `05_uniform.md`, `05_hetero.md` (and the two CSVs if you like).

---

## Bundle everything

```bash
cd /tmp/mfc2_check && tar czf mfc2_check.tgz *.txt *.md *.csv 2>/dev/null; ls -la mfc2_check.tgz
```

The logs (`*.log`) are large; send them only if something failed, then send the last 80 lines of the relevant one:

```bash
tr '\r' '\n' < /tmp/mfc2_check/0X_name.log | tail -80
```

## What I need, in short

| Step | Needed | Why |
|---|---|---|
| 0 | `00_git.txt` | confirm patches applied, author |
| 1 | last 5 lines of pytest (+ failures) | code works on your machine |
| 2 | `02_scenarios.txt` | scenarios load |
| 3 | exit code, `final_rmse`, file list | proves no regression |
| 4 | `[SPLIT]` / `[STEP 5b]` lines | train/val example works for you |
| 5 | `[PIN]` lines + comparison tables | TDR-pinned result reproduces |
