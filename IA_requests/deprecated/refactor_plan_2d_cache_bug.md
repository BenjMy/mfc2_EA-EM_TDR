# Refactor Plan — Stale Ensemble Cache & Silent Point-Trimming (v2, implemented)

**Status: Steps 1 and 2 below are implemented in `run_pipeline.py` (attached), `py_compile`
clean. Superseded plan** — the original fingerprint-fix approach (v1) was dropped per
your instruction; the cache is now simply disabled for synthetic runs instead of made
smarter.

---

## Step 1 — Cache is now real-data-only (implemented)

**File:** `run_pipeline.py`, Step 1 cache lookup (~L2181) and cache save (~L2582)

Rather than extending `_config_fingerprint()` to cover `mode`/`grid_num_x`/`grid_num_y`
(the original plan), the cache lookup/save block is now gated on `data_source == "real"`:

- `data_source != "real"` -> cache lookup is skipped entirely, every synthetic run always
  does a fresh inversion (`[CACHE] Skipped for synthetic data...`).
- The save block only writes `ensemble_cache_*.pkl` when `data_source == "real"`, so a
  synthetic run can never leave behind a cache file for a later run to stumble into.
- Real-data caching is untouched -- it was never the source of this bug and is genuinely
  worth keeping (real inversions are much more expensive).

This is strictly simpler than the fingerprint fix and closes the whole bug class (not
just the specific `mode`/`grid_num_x`/`grid_num_y` gap) for synthetic runs, at the cost
of re-inverting every synthetic run from scratch -- acceptable since synthetic ensembles
are cheap.

**No `trimSurveys()` needed.** Checked against the installed `emagpy` source
(`Problem.py`): `Problem.trimSurveys()` requires a multi-epoch `Problem` built via
`createTimeLapseSurvey()`, but your production Step 1 path (`_invert_single_worker` +
`ProcessPoolExecutor`) inverts one file per epoch per parallel task -- it never builds
that multi-epoch `Problem`, so `trimSurveys()` doesn't fit this architecture without a
larger rewrite (dropping per-task parallelism). See Step 2 for what actually replaces the
silent-drop behavior you were worried about.

---

## Step 2 — Synthetic epoch alignment now coordinate-based, not row-position (implemented)

**File:** `run_pipeline.py`, Step 1 "SPATIAL ASSEMBLY OF THE ENSEMBLE ACROSS EPOCHS"
(~L2415-2543)

### What was there before

Two separate code paths after inversion:
- **Real data:** `geometry_utils.match_epochs_to_reference_grid(epoch_xy, tolerance_m=...)`
  -- matches each epoch's points to a union reference grid by actual (x, y) coordinate,
  logs how many points per epoch matched, NaN-fills anything unmatched. Correct and
  already well-tested.
- **Synthetic data:** `arr[:min_pts]` -- trimmed every epoch's array to the smallest
  epoch's row **count**, pairing "row i of epoch A" with "row i of epoch B" purely by
  array **position**. Only logged when it actually reduced the count. Safe *only* because
  Step 0 currently writes byte-identical, identically-ordered coordinates to every
  epoch's CSV -- an assumption that isn't verified anywhere near this code, and would
  silently mispair points if it ever stopped holding (e.g. a future per-epoch filtering
  or reordering change).

### What's there now

The `if data_source == "real": ... else: ...` split is gone. **Every timelapse run** --
real or synthetic -- now goes through `geometry_utils.match_epochs_to_reference_grid`:

```python
_reference_grid_tolerance = (
    SPATIAL_MATCH_TOLERANCE_M if data_source == "real"
    else 1e-6   # synthetic coords are bit-identical across epochs by construction
)
```

- Real data keeps its existing (typically ~2 m) GPS-noise tolerance.
- Synthetic data uses a near-zero tolerance (`1e-6` m) -- appropriate because coordinates
  are the exact same float array written to every epoch's CSV
  (`0_generate_synthetic_data.export_eca_surveys_to_csv`), so only float round-trip noise
  needs covering, not real positional uncertainty. `1e-6` m is far below your minimum
  point spacing (2.0 m), so there's no risk of two distinct grid points being merged.
- Every epoch now logs its own match count (`"N own point(s) -> M/n_ref matched"`) and the
  union point count / tolerance are recorded as report metrics -- for a clean synthetic
  run you should see every epoch matching 100% of the reference grid; anything less is
  now visible instead of silently trimmed away.

### Why this addresses "avoid the silent drop of points"

- Matching is now by actual coordinate identity (confirmed via `emagpy`'s own
  `_matchSurveys()`, which keys on combined x+y -- the same principle
  `match_epochs_to_reference_grid` implements), not row position -- no possibility of
  mispairing physically different locations under the same index.
- No points are ever discarded outright: an epoch that doesn't cover a reference point
  gets `NaN` there (already NaN-safe downstream, per the existing real-data comment
  block), rather than the whole array being truncated because of one under-covered epoch.
- One code path instead of two -- future changes to Step 0/1 only need reasoning about a
  single, already-hardened matching implementation.

### Validation performed

- `python -m py_compile run_pipeline.py` -- clean.
- Full diff reviewed -- behavior-only, no unrelated churn (see attached file).

### Validation still to do (needs your environment / other files)

- `geometry_utils.py` was not supplied, so I couldn't unit-test
  `match_epochs_to_reference_grid` directly -- I'm relying on its existing, already-proven
  real-data behavior extended to a tighter tolerance. Recommend a quick assertion after
  your next synthetic 2D run:
  ```python
  assert models_clean.shape[-2 if timelapse else -1] == grid_num_x * grid_num_y, (
      "synthetic ensemble spatial axis doesn't match the configured grid"
  )
  ```
- Re-run the `2d` scenario end-to-end and confirm Step 6 completes without the shape
  crash, and that the per-epoch match-count log shows 100% for every epoch.
- Delete any existing `mfc2_outputs/field_survey_2d/ensemble_cache_*.pkl` once, by hand --
  it'll never be written again for this scenario, but a stale one already on disk from
  before this fix is now simply ignored (not deleted) since the lookup is skipped
  entirely for synthetic; harmless to leave, but fine to remove for tidiness.

---

## Still open from the original plan

| Item | Status |
|---|---|
| Step 3 (route `[CACHE]` prints through `reporter.log()`) | Superseded -- synthetic path now prints one clear skip line; real-data cache logging unchanged. Can still do this for the real branch if you want cache hits/misses visible in reports -- say the word. |
| Step 4 (defensive shape guard in `plot_water_content.py`) | Still blocked on that file. With Steps 1-2 in place it should no longer be reachable for synthetic runs, but real-data time-lapse with partial epoch coverage could still hit a shape mismatch -- guard still worth adding as defense-in-depth. |
| Step 5 (`threed=True` for genuine 2D coupling) | Unchanged, deferred, your call. |
