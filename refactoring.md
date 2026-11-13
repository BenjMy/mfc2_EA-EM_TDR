# Refactoring plan — mfc2_EA-EM_TDR (readability first)

Goal: make it obvious **what is synthetic, what is real, how scenarios are built, where each plot comes from**, without changing any scientific result.

Everything below is based on reading the repo at commit `10a4c9a`. Line numbers refer to that commit. Nothing here has been executed except where stated, so verify with the regression check in section 0 before and after each phase.

---

## Status (2026-11-11, branch `refactor/readability`, tag `pre-refactor` = `10a4c9a`)

| Phase | Status | Evidence |
|---|---|---|
| 0 Reproducible runs | **done, verified** | seeding was already in the branch; two identical `1d_surface_tdr` runs give bit-identical `theta_calc`, `ec_optimal`, `rmse`, `cv_map`, `K` (final_rmse 0.036118). New baseline recorded. |
| 1 Hygiene | **done** | `__pycache__`, `comparison_output/`, `scoring.csv` untracked + ignored; `IA_requests/`, `backup/`, legacy `tests/*` moved to `archive/`; README exists. |
| 2 Delete obsolete | **partly done** | removed `capture_synthetic_theta.py` (pipeline saves the arrays itself) and `apply_sensor_ids_fix.py` (already applied). `run_and_score.py` / `score_against_truth.py` / `generate_synthetic_real_mirror.py` **kept** (decision 2026-11-12: still used for the paper; they are described in `EM_TDR_strategy_progress_report.md` and `compare_scenarios.py` reads their scoring CSVs). |
| 3 Importable modules | **done** | `0_generate_synthetic_data`->`data_synthetic`, `0_real_data_loader`->`data_real`, `1_..`->`step1_inversion`, `2_..`->`step2_petro`, `3_..`->`step3_average`, `4_..`->`step4_select`; importlib / fallback chains replaced by plain imports. `0_realEM_data_loader.py`, `0_realTDR_data_loader.py` were unreferenced (grep): moved to `archive/`. |
| 4 Plot helpers | **done** | `plots/_common.py`; the 6 `_ensure_dir` + 2 `_save` copies are gone. |
| 5 Split plotting out of core | **partly done** | the 5 plotting functions of `em_methodology.py` (`plot_step1_ensemble_*`, `plot_raw/filtered_apparent_conductivity`, `_flag_dropped_axes`) moved byte-identical to `plots/pre_inversion.py`; `em_methodology.py` no longer imports matplotlib or `os`; `run_pipeline.py` calls `pre_inversion.*` (a re-export from `em_methodology` would be circular). Checked: pyflakes (no undefined names), ensemble plots run on random data, `tests/test_plots_pre_inversion.py`. **Verified on Ben's machine 2026-11-12** (`1d_surface_tdr`, worktrees at `06b042a` = before vs branch HEAD = after, plus a 2nd run of HEAD): all 20 saved arrays bit-identical before vs after and after vs after2; `t00/raw_eca_transect.png` and `t00/filtered_eca_transect.png` (the two plots not executed in the cloud) were produced. The `grep -ci 'traceback|error'` on `run.log` matched 2 lines, not yet inspected. `plot_truth_tdr` moved byte-identical to `plots/truth_tdr.py` (2026-11-12; `run_pipeline` re-imports it under the same name, `run_and_score.py` still calls `run_pipeline.plot_truth_tdr`; `tests/test_plots_truth_tdr.py`). **Open:** splitting `em_methodology.py` by responsibility; the raw/filtered ECa plots were not executed in the cloud (need real EM data). |
| 6 Split `main()` | **started** (2 of ~17 blocks) | `steps/` package created. **Steps 6d and 6e** (solution SWC / EC_bulk plots; forward-simulated ECa check) moved to `steps/step6_plots.py::run_step6d_solution_plots` / `run_step6e_forward_eca`; `_resolve_npy_path` moved to `steps/_common.py` (re-imported under the old name). Deliberate deviation from the plan: explicit keyword arguments instead of `run(ctx)`; introduce `PipelineContext` once 3-4 blocks are out and the shared arguments are visible. Regression check (`1d_surface_tdr`, before/after in the same environment): 20/20 arrays bit-identical, same 107 figures, same `final_rmse` 0.032468 (re-checked after 6e: also all manifest metrics equal). `tests/test_step6_plots.py`. Remaining blocks, in order of how self-contained they are: 6b/6c/6c-bis, 1b-1d (real-data diagnostics), 6, 5c, 5b, then S0-S5. `main()` is ~4,350 lines. |
| 7 Data-source interface | open | |
| 8 Scenario building | **done** | `"extends"` implemented + tested; unknown-key validation (typo -> `ValueError` with "did you mean", `_`-prefixed keys = comments) and `python scenario_loader.py --list-scenarios [--config FILE]` (name + description) done and tested; all shipped scenarios (synthetic + real) still validate. **Sweeps (2026-11-12):** `sweeps.py` generates `1d_{surface,full}_free{K}` and `1d_{surface,full}_hetero_free{K}` from the two base scenarios `1d_surface_tdr` / `1d_full_depth_ref` (adding a K = one line, `FREE_KS`); the 16 hand-pasted entries (808 lines) are gone from `ensemble_scenarios.json`; an entry of the same name in the JSON still overrides the generated one; `load_scenario`, `list_scenarios`, `validate_all`, `describe_scenarios` all see them. **Old-vs-new equality:** `tests/data/scenarios_golden.json` (resolved config of all 42 scenarios, recorded before the removal) + `tests/test_scenarios_golden.py`; regenerate with `python tests/make_scenario_golden.py` only after an intentional scenario change. `_resolve_extends` now keeps the input order. |
| 9 Run manifest | **done** | `run_log.py`: `RunLog` (append / rows / by_id / by_seq / latest; `python run_log.py latest <scenario>` replaces the column-position `awk` in `run_1d_surface_vs_full.sh`), `write_run_manifest` / `read_run_manifest`, `git_commit`. Every run writes `<output_dir>/run_manifest.json` (schema_version 1: run_id, scenario, mode, data source, status, quality gate + reasons, runtime, git commit (`+dirty` if uncommitted), python/host, seeds, resolved scenario config, `metrics`, `step_timing`; also on failure). `compare_scenarios.parse_report` reads the manifest (given the report, the manifest or the run folder) and falls back to the markdown only for runs without one; `tests/test_run_log.py` checks that both routes give the same record. `DEFAULT_INVERSION_SEED = 1234` is now a named constant (same value). Not done: `compare_scenarios.resolve_ids_to_reports` still reads the CSV with pandas (works, own seq logic). |
| 10 Tests | **started** | `pytest tests` -> 159 passed (~8 s, real `emagpy` installed; on 2026-11-11 it was 50 passed / 1 skipped without it). Since then also `test_scenarios_golden.py`, `test_sweeps.py`, `test_run_log.py`, `test_plots_truth_tdr.py`. `tests/test_scenarios.py` (loader, `extends`, `ws_layer_params`, unknown keys, listing) and `tests/test_spatial_reference_grid.py` (moved from root, now real pytest asserts; `tests/conftest.py` stubs `emagpy` when absent, the one test needing the real forward model is skipped then). Not yet: WS round trip, end-to-end seeded 1-D run. |

Regression check 2026-11-12 (phases 5 `plot_truth_tdr`, 8 sweeps, 9 manifest; cloud environment, Python 3.13): `1d_surface_tdr --mode 1d --num-swc-positions 5` run at `3d065d9` (before) and at the branch HEAD (after): all 20 arrays of `output_model/` bit-identical, `final_rmse` 0.032468 both (runtime ~4.4 min on 2 cores), no traceback, `run_manifest.json` written and read by `compare_scenarios`. Different environment from your machine (0.031212 there), so compare only within one environment.

Regression check after phases 0-4: `1d_surface_tdr` re-run after Phase 4, all 20 saved arrays bit-identical to the pre-refactor reference. Phases 1-4 touched no scientific code.

Not refactoring (behaviour changes, separate commits): 1d_* EC bound 60 -> 300 mS/m; `ws_layer_params` + `hetero` scenarios + `extends`. See `to-do.md`.

Environment note: `emagpy` is not in the system Python of the cloud session; it was installed in a venv for these runs. Python 3.13 / numpy 2.5 / pandas 3.0 (your results above were made with 3.12).

Baselines of `1d_surface_tdr` (`--mode 1d --num-swc-positions 5`), final_rmse / cv_mean / cv_max, run 2026-11-12 on Ben's machine (conda base): branch HEAD (EC bound 300 mS/m) **0.031212 / 0.3360 / 1.9646** (reproduced bit-identically 2x); commit `0d43365` (EC bound 60 mS/m) **0.032023 / 0.0000 / 0.0000**. The cloud reference for `0d43365` was 0.036118, so the same code gives a different value on the two machines (different Python/library versions, not investigated): compare runs only within one environment. With the 60 mS/m bound the ensemble spread is exactly 0 (consistent with every member sitting on the bound); with 300 it is not.

---

## 0. Ground rules (read first)

### Git rules (mandatory)
0. **Always start a work session by getting the latest upstream** (anyone, including you from another machine, may have pushed). Whoever works on the code, human or assistant, starts from a fresh clone of `https://github.com/BenjMy/mfc2_EA-EM_TDR`, and reports the HEAD commit and whether it differs from the commit the last analysis or patch was based on (currently `10a4c9a`). In your local copy the equivalent is:
   ```bash
   git fetch origin && git status -sb      # "behind N" means upstream has new commits
   git log --oneline HEAD..origin/main     # what is new
   git pull --ff-only                      # only on a clean tree
   ```
   Patches and analysis written against an older commit must be re-checked with `git apply --check <patch>` before use.
1. **All refactoring happens on a separate branch, never on `main`.** Before starting, on `main`:
   ```bash
   git tag pre-refactor            # safety anchor, lets you diff or restore at any time
   git switch -c refactor/readability
   ```
   `main` stays untouched and runnable (it is what produced your current results). Merge only through a pull request after the regression check (below) passes, and delete nothing from `main` until then.
2. **Ben is the author of every refactoring commit.** Commits are made from your own machine with your own identity; no commit is authored by, or carries a `Co-authored-by:` trailer for, an assistant or any other tool. Check before the first commit:
   ```bash
   git config user.name         # should be your name
   git config user.email        # the repo history uses: 36472081+BenjMy@users.noreply.github.com
   git log -1 --format='%an <%ae>'   # after the first commit: must show you
   ```
   If someone else or a tool prepares changes for you (patches, scripts), apply them yourself (`git apply`, review the diff, `git commit`) so the commit is yours. To fix a wrong author on an unpushed commit: `git commit --amend --reset-author`.
3. **One commit per phase step** (or finer), with a message saying what moved/changed and that behaviour is unchanged, e.g. `refactor: rename numbered step modules (no behaviour change)`. Never mix a refactor and a bug fix in one commit.

### Working rules
4. **Regression check** (before touching anything, and after every phase): run `1d_surface_tdr` and `1d_full_depth_ref` and compare `final_rmse` in `run_log.csv`. Baseline from your current log: `0.034585` (surface) and `0.035992` (full) m³/m³ — but see Phase 0: these numbers are **probably not reproducible yet**, so Phase 0 comes first.
5. **Move first, rewrite later.** Prefer `git mv` + import fixes over rewriting logic. Rewrites only where section 4 says so.
6. The `run_*_PAPER.sh` scripts do **not** need to keep working with the old file names: rewrite them to the new names as part of Phase 3 (`run_synthetic_PAPER.sh`, `run_real_PAPER.sh`, `run_1d_surface_vs_full.sh`).

---

## 1. Current map (what each file is today)

| Category | Files | Notes |
|---|---|---|
| **Entry points / launchers** | `run_pipeline.sh` (1087 l), `run_pipeline.py` (5862 l), `compare_scenarios.sh/.py`, `run_1d_surface_vs_full.sh`, `run_synthetic_PAPER.sh`, `run_real_PAPER.sh` | `run_pipeline.py:main()` alone is ~4,400 lines (from l.1467), about 75% of the file |
| **Scenario definition** | `ensemble_scenarios.json`, `ensemble_scenarios_real.json`, `scenario_loader.py`, `backup/ensemble_scenarios.json` | JSON entries are near-copies of each other; docs live in a `_comment` list of strings |
| **Synthetic data** | `0_generate_synthetic_data.py`, `generate_synthetic_real_mirror.py`, `capture_synthetic_theta.py`, `score_against_truth.py`, `run_and_score.py` | "mirror" = synthetic twin of the real deployment |
| **Real data** | `0_real_data_loader.py`, `0_realEM_data_loader.py`, `0_realTDR_data_loader.py`, `train_val_split.py` | three loaders with overlapping roles; the real branch is also inlined in `run_pipeline.py` (`_prepare_real_em_csv`, `_load_real_data`, l.559–1166) |
| **Pipeline steps** | `1_run_model_generation.py` (S1), `2_run_petro.py` (S2), `3_run_average.py` (S3), `4_run_select_assess.py` (S4/5) | thin driver layers over `em_methodology.py` |
| **Core library** | `em_methodology.py` (4216 l, 45 defs), `em_sensitivity.py`, `geometry_utils.py` | mixes inversion, petrophysics (Waxman-Smits, localized), averaging, selection, quality gates **and plotting** (`plot_step1_*`, `plot_raw_*`, l.3397+) |
| **Plots — pre-inversion / real** | `plot_pre_inversion_diagnostics.py`, `plot_em_diagnostics_real.py`, `plot_tdr_diagnostics.py`, `plot_ini_model.py`, `plot_probe_matching.py`, `plot_em_footprint.py` | |
| **Plots — inversion / solution** | `plot_emagpy_diagnostics.py`, `plot_solution_diagnostics.py`, `plot_water_content.py` (3627 l, 35 defs), `plot_calibration_density.py`, `plot_train_val_diagnostics.py` | |
| **Plots — cross-run** | `compare_scenarios.py` (1108 l, 26 defs) | also does parsing of `pipeline_report_*.md` and truth comparison |
| **One-off / legacy** | `apply_sensor_ids_fix.py`, `capture_synthetic_theta.py`, `run_and_score.py`, `IA_requests/`, `backup/`, `tests/nb_smoothing.*`, `tests/tmp` | see section 2 |
| **Tests** | `test_spatial_reference_grid.py` (root, no pytest), `tests/` | not a test suite yet |
| **Tracked artefacts** | 48 `__pycache__/*.pyc`, `comparison_output/`, `scoring.csv` | should not be in git |

---

## 2. Findings (evidence, ordered by impact on readability)

1. **`main()` is a 4,400-line function** whose steps are only comments (`# STEP 0 … STEP 6e`, l.1866–5329). Hard to read, test or reuse a single step.
2. **Synthetic vs real is decided by ~32 inline `data_source == "synthetic" | "real"` branches** inside `main()`. There is no single place that says "this is what a synthetic run does / what a real run does".
3. **Numbered module names (`0_…`, `1_…`) cannot be imported normally.** `run_pipeline.py` uses `importlib.import_module("0_generate_synthetic_data")` (l.75–78) and `spec_from_file_location` for the real loader (l.1004–1012). `run_and_score.py` monkey-patches modules to capture results.
4. **Stale / contradictory docs.** `capture_synthetic_theta.py` says `theta_calc` is never saved, but `run_pipeline.py` now saves it (l.~4770–4795, "PERSIST STEP 4/5 RESULT"). That script, and the monkey-patch approach in `run_and_score.py`, are probably obsolete — confirm before deleting.
5. **Duplicated helpers.** `_ensure_dir` is defined 6 times and `_save` twice (`plot_calibration_density`, `plot_em_footprint`, `plot_emagpy_diagnostics`, `plot_pre_inversion_diagnostics`, `plot_probe_matching`, `plot_solution_diagnostics`).
6. **Plotting is spread over 12+ modules and also inside the core library** (`em_methodology.py`, `compare_scenarios.py`, `run_pipeline.py:plot_truth_tdr`).
7. **Scenario JSON is copy-paste.** `1d_surface_tdr` and `1d_full_depth_ref` differ by 2 fields (`swc_depths`, `allow_uncovered_horizons`) plus name/output/cache tag; the 8 `1d_*_free{K}` entries differ by `ws_fixed_params`. Documentation is a hand-maintained `_comment` list that already drifts from the entries.
8. **One-off patch scripts live next to the code** (`apply_sensor_ids_fix.py` rewrites `2_run_petro.py` and `plot_water_content.py` in place).
9. **Run bookkeeping is by convention.** `run_1d_surface_vs_full.sh` finds runs with `awk` on column 3 of `run_log.csv`; the output folder name encodes the run number. Fragile if the log format changes.
10. **Many modules have no docstring or a stale one** (`0_generate_synthetic_data.py` has none).

---

## 3. Target layout

```
mfc2/                           # python package (import mfc2.xxx)
  config/
    scenario.py                 # ScenarioConfig dataclass + validation   (from scenario_loader.py)
    loader.py                   # load_scenario(name, path)
  scenarios/
    base.json                   # shared defaults
    synthetic.json              # was ensemble_scenarios.json
    real.json                   # was ensemble_scenarios_real.json
    sweeps.py                   # generates families (e.g. free1..4 x surface/full)
  data/
    base.py                     # SurveyData dataclass + DataSource protocol
    synthetic/                  # generate.py (was 0_generate_synthetic_data), mirror.py
    real/                       # em.py, tdr.py, coords.py, train_val.py
  core/
    inversion.py                # ensemble inversion (S1)
    petrophysics.py             # Waxman-Smits, localized WS (S2, S4)
    averaging.py                # S3
    selection.py                # S4/5 + quality gates
    forward.py                  # emagpy forward helpers
    geometry.py  sensitivity.py
  pipeline/
    run.py                      # orchestrator: calls step functions in order
    steps/                      # step0_data.py ... step6_plots.py, one function each
    reporting.py                # PipelineReporter, run log
  plots/
    _common.py                  # ensure_dir, save_fig, style (single copy)
    pre_inversion.py  solution.py  water_content.py  calibration.py
    validation.py               # 1:1 plots (ECa, EC, theta) — see section 6
    compare.py                  # cross-run comparison
  scoring/                      # score_against_truth, metrics
cli/                            # thin scripts: run_pipeline.sh, compare_scenarios.sh, studies/
studies/
  surface_vs_full_1d/           # run_1d_surface_vs_full.sh + README.md
tests/                          # pytest
docs/                           # this file, pipeline overview, scenario guide
```

You do not have to reach this in one go. Phases in section 4 get there incrementally.

---

## 4. Step-by-step plan

Each phase: **what**, **how**, **done when**.

### Phase 0 — Make runs reproducible (~2 h, do before anything else)
**Status found in the code (not yet confirmed by running it):**
- Synthetic data generation **is seeded**: global `np.random.seed(42)` (`0_generate_synthetic_data.py` l.381), per-member `default_rng(seed + ens_idx)` for the random xy fields (l.406, 427) and `ws_xy_seed`, noise `default_rng(seed=0)` (l.760). TDR observation noise is seeded too (`np.random.seed(42)`, run_pipeline.py l.4065).
- The **inversion (Step 1) is not seeded.** The scenarios use `method: ROPE`, an emagpy sampling-based (MCMC-family) optimiser, and run in `ProcessPoolExecutor` workers started with `spawn` (run_pipeline.py l.2652). A spawned worker starts with a fresh OS-entropy RNG, and `_invert_single_worker` (l.1210) sets no seed. So two identical runs most likely give different ensembles and a different `final_rmse`.
- Consistent with this, your `run_log.csv` shows the same scenario name with spread RMSEs: `2d` 0.0307 / 0.0365 / 0.0360 / 0.0348; `2d_horizon_ws` 0.0363 / 0.0294 / 0.0242. (The log does not record CLI options, so some of that spread may come from different settings; it is evidence, not proof.)

**Fix:** `seed_inversion_workers.patch` (also contained in `mfc2_all_changes.patch`; seeds `random` and `numpy` per task with `inversion_seed + task_idx`, default 1234, so results do not depend on which worker picks which task). Then:
1. Run `1d_surface_tdr` twice. `final_rmse` must be **identical** (all digits).
2. If it still differs, emagpy's ROPE draws from its own RNG or its own process pool: look for `np.random`/`Generator` use inside the installed `emagpy` (not available in the review sandbox) and seed that, or pass its seed argument if the method exposes one.
3. Expose the seed as a scenario field (`"inversion_seed": 1234`) and print it in the pipeline report.
4. Record the new baseline RMSE values (they will differ from the old ones; that is expected, not a regression).
**Done when:** two identical runs give bit-identical `final_rmse` and `theta_calc.npy`.

### Phase 1 — Hygiene (no code change; ~1 h)
- `git rm -r --cached __pycache__ comparison_output scoring.csv` and add to `.gitignore`: `__pycache__/`, `*.pyc`, `comparison_output/`, `scoring.csv`, `mfc2_outputs/`.
- Move `IA_requests/`, `backup/`, `tests/tmp`, `tests/nb_smoothing.*`, `tests/beta_comparison.png`, `tests/coverCrop.csv` to `archive/` (or delete after checking nobody needs them).
- Add a top-level `README.md`: 15 lines — what the project is, how to run one scenario, where outputs go, link to this file.
- **Done when:** `git status` is clean after a run; the regression check passes.

### Phase 2 — Delete or archive what is obsolete (~1 h)
- Confirm `capture_synthetic_theta.py` is obsolete (the pipeline already writes `theta_calc.npy`, `K.npy`, `rmse.npy`, `cv_map.npy`, `is_calibrated.npy` into `output_model/`). If so, remove it and fix any doc that points to it.
- `apply_sensor_ids_fix.py`: check the fix is already in `2_run_petro.py` / `plot_water_content.py`; then delete.
- `run_and_score.py` + `score_against_truth.py`: decide whether the real-mirror scoring is still used. If yes, keep as `scoring/`; if no, archive.
- **Done when:** no remaining reference (`grep -rn <name>`) to the removed files.

### Phase 3 — Make modules importable (~2 h, mechanical)
- `git mv` the numbered files: `0_generate_synthetic_data.py → data_synthetic.py`, `0_real_data_loader.py → data_real.py`, `1_run_model_generation.py → step1_inversion.py`, `2_run_petro.py → step2_petro.py`, `3_run_average.py → step3_average.py`, `4_run_select_assess.py → step4_select.py`.
- Replace `importlib.import_module("0_…")` (run_pipeline.py l.75–78) and the `spec_from_file_location` call (l.1004–1012) with plain `import`.
- Update `run_and_score.py`/shell scripts that reference the old names (`grep -rn "run_petro\|run_average\|generate_synthetic"`).
- Rewrite `run_synthetic_PAPER.sh`, `run_real_PAPER.sh` and `run_1d_surface_vs_full.sh` for the new names (no backward compatibility needed).
- **Done when:** `python -c "import step2_petro"` works and the regression check passes.

### Phase 4 — One copy of the plotting helpers (~1 h)
- Create `plots/_common.py` with `ensure_dir`, `save_fig` (use the `plot_solution_diagnostics` versions as the reference, `dpi=180`).
- Replace the 6 local `_ensure_dir` and 2 `_save` definitions with imports.
- **Done when:** `grep -n "def _ensure_dir\|def _save" *.py` returns nothing outside `_common.py`.

### Phase 5 — Separate plotting from the core library (~half a day)
- Move `plot_step1_ensemble_single_file`, `plot_step1_ensemble_by_point`, `plot_raw_apparent_conductivity`, `plot_filtered_apparent_conductivity`, `_flag_dropped_axes` (em_methodology.py l.3397–3980) into `plots/pre_inversion.py`. Keep a re-export in `em_methodology.py` for one release if other code imports them.
- Move `plot_truth_tdr` (run_pipeline.py l.1322) to `plots/`.
- Split `em_methodology.py` (4216 l) by responsibility, pure `git mv` of functions:
  `inversion` (build/generate ensemble, drop coils, temperature) / `petrophysics` (WS + localized + bandwidth sweep) / `averaging` / `selection` (+ quality gates) / `forward`.
- **Done when:** `em_methodology.py` no longer imports matplotlib; no file in `core/` imports `plots/`.

### Phase 6 — Split `main()` into step functions (the big one; 1–2 days)
Do it in the order the steps already appear. For each block between two `# STEP …` comment banners:
1. Identify its inputs (variables read) and outputs (variables later used).
2. Move it to `steps/stepN_*.py` as `def run(ctx) -> ctx` where `ctx` is a dataclass (`PipelineContext`) holding `cfg`, `output_dirs`, `survey`, `ensemble`, `averaged_models`, `selected_output`, `reporter`.
3. Replace the block in `main()` by the call. Run the regression check after **each** step, not at the end.
- Target order: S0 data → S1a pre-diagnostics → S1 inversion → S1b-d real diagnostics → S2 → S3 → S4 → S5/5b/5c → S6 plots (6, 6b, 6c, 6c-bis, 6d, 6e as separate functions inside `steps/step6_plots.py`).
- **Done when:** `main()` is < 150 lines: parse config → build context → loop over steps → write report.

### Phase 7 — Synthetic vs real as an explicit interface (~1 day, after Phase 6)
- Define in `data/base.py`:
  ```python
  @dataclass
  class SurveyData:
      em_csv_paths: list[str]
      x: np.ndarray; y: np.ndarray
      probes: ProbeTable | None
      truth: TruthArrays | None          # None for real data
      matched_indices: list | None       # real time-lapse only
  class DataSource(Protocol):
      def load(self, cfg) -> SurveyData: ...
  ```
- `SyntheticSource` wraps `data_synthetic`; `RealSource` wraps `data_real` (+ the code now inlined at run_pipeline.py l.559–1166).
- Replace the ~32 `if data_source == …` branches by: (a) `survey.truth is not None` for "plot against truth", (b) `survey.matched_indices is not None` for the reference-grid remap. Anything left is a real behavioural difference and goes into the source class.
- **Done when:** `grep -c 'data_source ==' pipeline/` is 0 or 1.

### Phase 8 — Scenario building (~half a day)
- Add an `"extends": "<name>"` key to `scenario_loader.py` (deep-merge parent then child). Then `1d_full_depth_ref` becomes ~5 lines instead of ~80.
- Generate sweep families in code instead of pasting JSON. Example for the surface-vs-full study: `scenarios/sweeps.py` builds `1d_{surface,full}_free{K}` from `1d_surface_tdr`/`1d_full_depth_ref` + the freeing order `[sigma_s, m, sigma_w, a]` (the generator I used for the 8 entries added earlier is a starting point). `load_scenario` falls back to the generator when the name is not in the JSON.
- Replace the `_comment` string list by a `description` per scenario (already present) and a generated `--list-scenarios` command.
- Validate unknown keys (typos such as the `ws_localized_bandwidth_sweep_NULL` mentioned in `run_pipeline.sh` should fail loudly).
- **Done when:** adding a sweep value is a one-line change, and `load_scenario` of every existing name returns the same config as before (write a test that compares old vs new for all names).

### Phase 9 — Run bookkeeping (~2 h)
- Write a small `RunLog` class (read/append/`latest(scenario)`/`by_id`) used by `run_pipeline.py`, `compare_scenarios.py` and the study scripts, instead of `awk` on column positions in shell.
- Write a `run_manifest.json` inside every output dir (scenario name, resolved config, git commit, seeds, runtime). Comparison scripts read this instead of parsing `pipeline_report_*.md`.
- **Done when:** `compare_scenarios` works without parsing markdown reports.

### Phase 10 — Tests (~1 day, can start after Phase 3)
- Move `test_spatial_reference_grid.py` to `tests/`, convert to pytest.
- Add: scenario loader round-trip (every scenario loads), Waxman-Smits forward/inverse round trip, `match_epochs_to_reference_grid` cases, a tiny 1-D synthetic end-to-end run with a fixed seed (this is your regression check automated).
- **Done when:** `pytest` passes in CI or locally in < 5 min.

---

## 5. Naming and code conventions
- Files: `snake_case`, no leading digits. Steps: `stepN_<verb>.py`.
- Functions that draw a figure: `plot_<what>(…, out_dir) -> str` returning the saved path. No plotting inside functions that compute.
- Functions that compute return data, never write PNG; writers are `save_*`.
- Every module: a docstring with *what it does, inputs, outputs, who calls it*. Keep the existing good ones (`geometry_utils.py`, `plot_solution_diagnostics.py`).
- Units in names or docstrings (`ec_ms_m`, `theta_m3m3`). The current mixture of `mS/m` and `S/m` is the usual source of silent bugs.
- Shapes in docstrings: `(n_time, n_pts, n_layers)`. Already done well in `plot_solution_diagnostics.py`; do the same everywhere.

---

## 6. Plotting plan (what exists, what to add)

**Already exists (per run, in `<output_dir>/solution/`):**
- `eca_forward_optimal/eca_forward_1to1_t??.png`: ECa **observed (noisy survey)** vs ECa **forward-simulated from the optimal EC model**, 1:1 over all coils/points, RMSE annotated (`plot_solution_diagnostics.forward_simulate_eca_optimal`, Step 6e).
- `solution_ec_bulk_profiles.png`: EC_bulk profiles, **spatial mean** only, truth overlaid dashed.
- `compare_scenarios.py:plot_scatter_and_residual`: theta (calc vs truth) 1:1, multi-run.

**Gaps:**
- No ECa 1:1 against the **noise-free truth** (`synthetic_time_lapse_eca.npy`), so you cannot see the noise floor.
- No EC_bulk 1:1 **per horizon** against truth, and `ec_optimal` was never saved to disk.
- Nothing puts surface-only and full-depth side by side per horizon.

**Added alongside this file:**
- `save_ec_optimal.patch` (also contained in `mfc2_all_changes.patch`): 6 lines in `run_pipeline.py` saving `output_model/ec_optimal.npy`.
- `plot_validation_1to1.py`: ECa 1:1 (truth vs optimal, plus noisy observed as hollow markers), EC_bulk 1:1 per horizon and theta 1:1 per horizon, one row per run, with RMSE/bias/R² and `validation_metrics.csv`. Horizons are coloured by calibrated / not calibrated when `is_calibrated.npy` exists.

**Which comparison to use:** keep both. ECa 1:1 is a *data-fit* check (the inversion is fitted to ECa, so it will look good for any configuration). EC per horizon (and theta) is the *truth-recovery* check and is the one that discriminates surface-only from full-depth. Report EC/theta per horizon in the paper, keep ECa as a sanity check.

**Target:** move `plot_validation_1to1.py` into `plots/validation.py` and have `compare_scenarios` call it, so the study script produces these figures in the same `comparison_output/` folder.

---

## 7. Suggested order and effort

| Phase | Effort | Risk | Do before |
|---|---|---|---|
| 0 Reproducible runs | 2 h | low | **everything** |
| 1 Hygiene | 1 h | none | 2 |
| 2 Delete obsolete | 1 h | low | 3 |
| 3 Importable modules | 2 h | low | 5, 6 |
| 4 Plot helpers | 1 h | none | 5 |
| 5 Split plots / core | ½ day | low | 6 |
| 6 Split `main()` | 1–2 days | **medium** | 7 |
| 7 Data source interface | 1 day | medium | — |
| 8 Scenario `extends` + sweeps | ½ day | low | independent |
| 9 Run manifest | 2 h | low | independent |
| 10 Tests | 1 day | none | start after 3 |

If you only have a day: do Phases 0–4 and 8. That removes most of the clutter and the JSON duplication with almost no risk.

---

## 8. Decisions and open questions

**Answered**
- *Seeded / deterministic runs?* Required (you want identical RMSE for identical runs). Data generation already is; the inversion is not yet. See Phase 0 and `seed_inversion_workers.patch`.
- *Keep `run_*_PAPER.sh` working with old names?* No. Rewrite them (Phase 3). No release tag needed for them; the `pre-refactor` tag in section 0 is enough.
- *Where does the work happen?* Separate branch `refactor/readability`, authored by you (section 0).

**Answered 2026-11-12**
- *Scoring scripts (`run_and_score.py`, `score_against_truth.py`, real-mirror generator) still used for the paper?* Yes: keep (Phase 2 closed).
- *One scenario JSON or two?* Two: `ensemble_scenarios.json` (synthetic) and `ensemble_scenarios_real.json` (real) stay separate (Phase 8: no merge).

**Still open**
_none at the moment._
