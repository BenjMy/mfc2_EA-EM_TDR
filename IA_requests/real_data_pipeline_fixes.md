# Real-Data Pipeline — Three Open Issues: Diagnosis & Proposed Fixes

Status: **investigation only, no code changes yet**. This doc is meant to be
handed to a future Claude Code session as the spec for the actual patch.

Scope: `run_pipeline.py`, `0_real_data_loader.py`, `em_methodology.py`,
`scenario_loader.py`, `ensemble_scenarios.json`, `plot_em_diagnostics_real.py`.

---

## 1. `em_unit` defaults to `"ppt"` but real GEM‑2 data is `"ppm"`

### Where it lives
- `scenario_loader.ScenarioConfig.em_unit: str = "ppt"` — hard default.
- `ensemble_scenarios.json` → scenario `"2d_real"` has **no `em_unit` key**,
  so it silently resolves to `"ppt"`.
- `run_pipeline.main()`: `EM_UNIT = getattr(cfg, "em_unit", "ppt")`, threaded
  into `_load_real_data`, `_prepare_real_em_csv` is unaffected (it's a column
  renamer only), but `filter_repeated_em_points(..., unit=em_unit)`,
  every `Problem.createSurvey(..., unit=em_unit)` call, and all Step 1
  inversion workers use it to convert quadrature → LIN ECa.
- `em_methodology.py`'s own docstring for `em_unit` already states the
  correct rule: `'ppm'` for GEM‑2 / CMD‑Explorer, `'ppt'` for GF Instruments,
  and warns *"Leaving this at 'ppt' with ppm data inflates every ECa value by
  1000×."*

### Root cause
The real-data loader (`0_real_data_loader.py`, `EM_HEADER_MAP`,
`_COLUMN_VARIANTS` in `run_pipeline.py`) is written specifically for GEM‑2
multi-frequency quadrature exports (`Q_425Hz`, `Q_1525Hz`, …) — i.e. this
pipeline's real-data branch *only* ever ingests GEM‑2 data. GEM‑2 is a `ppm`
instrument. But nothing enforces `em_unit="ppm"` for real-data scenarios; the
dataclass default and the JSON scenario both leave it at the GF-Instruments
(`ppt`) default. Every real-data run to date has therefore been inverting
ECa values that are ~1000× too large unless `em_unit` was manually overridden
on the CLI (`--` flag exists? — check: there is currently **no CLI flag for
em_unit** in `run_pipeline.py`'s `argparse` section, so the *only* way to set
it correctly today is to add `"em_unit": "ppm"` inside the scenario JSON by
hand).

### Proposed fix
1. **Scenario JSON**: add `"em_unit": "ppm"` explicitly to every real-data
   scenario (`2d_real`, and any future `mfc2_real`-style scenario).
2. **Validation gate** (`scenario_loader._validate`): when
   `data_source == "real"`, *require* `em_unit` to be present in the raw
   scenario dict (not just fall back to a default) — raise `ValueError` if
   absent, with a message pointing at the GEM‑2 → `ppm` rule. This converts a
   silent 1000× data-corruption bug into a load-time error.
3. Optionally add a `--em-unit` CLI flag to `run_pipeline.py` (mirroring
   `--drop-coils` / `--window-m` override pattern) for ad-hoc real-data runs
   with a custom config file, defaulting to `None` → scenario value.
4. **Sanity check at load time** (defense in depth): after
   `Problem.createSurvey`, log the median absolute LIN ECa value; GEM‑2 field
   soils typically fall in the 1–300 mS/m range. If the median is >3 orders
   of magnitude outside that band, emit a `[WARN]` suggesting an `em_unit`
   mismatch. This catches the *next* instrument variant too, not just this
   one instance.

### Non-goals
- Auto-detecting unit from column names/magnitude is explicitly discouraged
  here — GEM‑2's `Q_425Hz` naming already unambiguously identifies the
  instrument family, so it should be a hard requirement, not a heuristic.

---

## 2. Time-lapse point trimming collapses all epochs to `min_pts` **by row index**

### Where it lives
`run_pipeline.py`, Step 1, inside the "no cache hit" branch, right after the
parallel `ProcessPoolExecutor` pool returns:

```python
if timelapse:
    min_pts = min(arr.shape[0] for combo in packed_models for arr in combo)
    ...
    packed_models = [[arr[:min_pts] for arr in combo] for combo in packed_models]
    packed_depths = [[arr[:min_pts] for arr in combo] for combo in packed_depths]
models_clean = np.array(packed_models)   # (n_combo, n_time, min_pts, n_layers)
```

### Root cause
Each survey epoch is inverted **independently** (`_invert_single_worker`,
one `createSurvey` per file, no `trimSurveys`), so different epochs can and
do have different point counts (the surveyor walks a slightly different path
each campaign — acknowledged directly in the surrounding code comment). To
get a single dense `(ensemble, n_time, n_pts, n_layers)` NumPy array, the
code takes the **minimum point count across all epochs and combos**, then
slices every epoch to `arr[:min_pts]` — i.e. it keeps whatever the first
`min_pts` **rows in file order** happen to be for that particular epoch.

This is a row-count trim, not a spatial trim. Row *i* in epoch A's CSV and
row *i* in epoch B's CSV are only the same physical location if the surveyor
walked an identical path in an identical order on both dates — which the
same code block's own comment says is *not* generally true for real GEM‑2
campaigns. Two consequences:

- **Data loss**: any epoch with more points than `min_pts` throws away all
  points beyond the shortest survey's count, even though those points were
  successfully inverted and could contribute to the ensemble.
- **Silent spatial misalignment**: for every epoch *except* the one that
  happens to define `min_pts` (or happens to share exact path order), the
  points kept at a given index are not the same physical location as the
  points kept at that same index in another epoch. This corrupts every
  downstream consumer that assumes `models_clean[:, t, i, :]` is "the same
  place" for all `t` — in particular:
  - `sample_positions` (built once from `EM_FNAMES[0]`'s `survey_df`, see
    Issue 3) is reused unchanged across all `t`.
  - Step 6's XY plan-view plots reuse a single `utm_x`/`utm_y` pair per
    spatial index across all time steps.
  - Temporal comparisons (Step 6d ground-truth overlay, Fig. 3 RMSE-vs-K,
    forward-ECa check) implicitly treat index `i` as a fixed physical point.

### Proposed fix — spatial reference-grid matching, not row trimming

The pipeline already contains the *right* algorithm for this, just scoped to
a diagnostic-only plot:
`plot_em_diagnostics_real._match_epochs_to_reference()` builds a reference
grid from the densest epoch and nearest-neighbour matches every other
epoch's points onto it (within `match_tolerance_m`), leaving `NaN` where no
match exists — **no epoch's own point set is ever subsetted or reordered**.

Recommendation: promote this pattern from "diagnostics only" to the actual
Step 1 ensemble-assembly step:

1. After all per-epoch inversions complete (still done independently, full
   point count, exactly as today — no change needed there), build one
   reference spatial grid. Two reasonable choices, pick one and document it:
   - **Densest epoch** (mirrors `_match_epochs_to_reference`'s existing
     choice) — simplest, reuses proven code.
   - **Union of all epochs' points**, deduplicated within a small tolerance
     — keeps strictly more spatial coverage than "densest epoch alone" but
     requires building the union grid rather than reusing one epoch's own
     coordinates.
   Given the user's stated goal ("keep all the points"), the **union**
   option is the better fit even though it's more work; the densest-epoch
   option should be treated as a fallback if the union route proves too
   invasive for a first pass.
2. For each epoch, nearest-neighbour match its own inverted EC array onto
   the reference grid (same distance-matrix approach as
   `_match_epochs_to_reference`, or reuse it directly via a shared helper),
   filling `NaN` for reference points with no match within a configurable
   tolerance (new scenario field, e.g. `spatial_match_tolerance_m`, distinct
   from `window_m` which governs TDR↔EM matching, not EM↔EM epoch matching).
3. Stack the matched arrays into `models_clean` as today — shape unchanged,
   `(n_combo, n_time, n_ref_pts, n_layers)` — except cells are `NaN` instead
   of silently-wrong data when a reference point wasn't actually surveyed at
   that epoch.
4. Every downstream consumer that already tolerates `NaN`/non-finite EC
   (`compute_coverage_map` already treats non-finite values as "invalid" and
   folds them into the coverage map; `average_inverted_models` would need a
   `nan`-aware weighted mean instead of `np.sum(w_view * models[...])`) needs
   an audit pass. This is the main implementation cost of this fix.
5. `sample_positions`/probe↔EM-index matching (Issue 3) should be run
   against the **reference grid**, not against `EM_FNAMES[0]` alone, so probe
   routing is consistent with how the ensemble array itself is now indexed.

### Lighter-weight alternative (if the NaN-propagation audit is too costly for v1)
Keep the dense-array requirement but stop throwing away points: use the
**union reference grid** as above, but instead of `NaN`-filling missing
epoch/point pairs, carry them forward from the nearest **temporally**
adjacent epoch's own value at that reference point (documented as an
explicit interpolation/hold-last-value choice, not silent). This avoids
having to make every downstream reducer NaN-aware, at the cost of injecting
mildly stale data for gap-filled cells. Should be flagged clearly as a
stop-gap, not the long-term fix, since it reintroduces a different kind of
"pretend this cell is something it isn't."

### Trade-offs
- The NaN-aware route (main proposal) is more correct but touches more
  files: `average_inverted_models`, `_compute_rmse_weights` interactions,
  `select_and_assess_model`, `compute_coverage_map` (partially NaN-aware
  already), and all plotting code that currently assumes finite arrays.
- The reference-grid tolerance is a new tunable that interacts with
  `window_m` (Issue 3) and should probably default to something tied to GPS
  precision (a few meters) rather than the TDR-probe window (~1 m).

---

## 3. TDR ↔ EM spatial "1 m window" — verify it is actually matching in the same CRS

### Current implementation (this part looks correct in isolation)
`em_methodology.extract_sample_positions_from_survey` → `_probe_distances`
computes Euclidean (or `|Δx|`-only) distance between each probe and every EM
measurement, keeps everything within `window_m`, and falls back to
nearest-point with a `[WARN]` + `used_fallback=True` flag when nothing is
within the window. This is then aggregated by
`summarize_probe_matching(...)` into a `ProbeMatchingSummary` with an
explicit `level` ("OK"/"WARN"/"ERROR") and `n_fallback_matches`, and
rendered visually by `plot_probe_matching.py`. **The window-matching
algorithm itself is sound and already instrumented for diagnosis** — the
open question is whether its *inputs* are in a consistent coordinate system.

### Suspected root cause — coordinate system mismatch
- TDR/SWC probe coordinates (`probe_table['utm_x'/'utm_y']`) come from
  `0_real_data_loader.load_tdr_dataframe` → `SOILNET_MFC2_coordinates_sensors.csv`,
  which are genuine UTM eastings/northings in metres (confirmed by the
  `ALENTO_UTM_ORIGIN_X/Y` constants used elsewhere in the codebase, ~515000 E
  / ~4468000 N, UTM Zone 33N).
- EM survey `x`/`y` columns come from `_prepare_real_em_csv`'s
  `_COLUMN_VARIANTS` mapping, which accepts **either**
  `Easting`/`Northing` (already metric/UTM) **or** `Longitude`/`Latitude`
  (decimal degrees) as valid aliases for the same canonical `x`/`y` columns
  — the function's own docstring documents both "Variant A" (semicolon,
  `X`/`Y`) and "Variant B" (comma, `Longitude`/`Latitude`) GEM‑2 export
  formats as things it actually sees in practice.
- Every `Problem.createSurvey(..., targetProjection=None, ...)` call in the
  real-data path (`_load_real_data`'s position load, the Step 1 inversion
  workers, `extract_sample_positions_from_survey`'s caller in
  `run_pipeline.main`) passes `targetProjection=None` — i.e. **no
  reprojection is ever requested**, regardless of which column variant fed
  the survey.
- If any given raw EM CSV happens to be "Variant B" (`Longitude`/`Latitude`),
  `survey_df['x']`/`['y']` end up in **decimal degrees**, while
  `probe_table['utm_x']`/`['utm_y']` are in **metres**. Subtracting the two
  in `_probe_distances` produces a "distance" that is off by ~5 orders of
  magnitude from a true metre distance. With `window_m = 1.0`, this would
  mean **literally every probe falls back to nearest-point matching**
  (`used_fallback=True` for 100% of probes), and the "nearest point" found
  this way is meaningless — it will just be whatever EM point happens to
  have the numerically closest (degrees vs metres, dominated by whichever
  axis has larger raw magnitude) coordinate value, not the geographically
  nearest one.

### How to confirm before fixing
This is fully diagnosable with data already in hand, without writing new
code:
1. Check `matching_summary.n_fallback_matches` / `.level` in a recent real
   run's Markdown report (`Step 1 — TDR<->EM spatial matching summary`
   table) — if `n_fallback_matches` ≈ `n_probes` (100% fallback), that is
   strong direct evidence of exactly this bug.
2. Open `probe_em_matching.png` / `probe_em_matching_real_stations.png`
   (from `plot_probe_matching.py`) — if EM points and TDR probes are visibly
   plotted in two unrelated coordinate ranges (or the connector lines all
   converge on one or two EM points regardless of probe location), that
   confirms it visually.
3. Inspect the raw GEM‑2 CSV header actually used for a given site/campaign
   to see which `_COLUMN_VARIANTS` alias matched (`Easting`/`Northing` vs
   `Longitude`/`Latitude`).

### Proposed fix
1. **Track which alias matched.** In `_prepare_real_em_csv`, record whether
   the resolved `x`/`y` columns came from a metric alias (`X`, `Easting`,
   `Northing`) or a geographic alias (`Longitude`, `Latitude`, `lon`, `lat`)
   and propagate that flag alongside the processed CSV (e.g. a JSON sidecar
   or a return-value tuple change) instead of silently normalizing both into
   generically-named `x`/`y` columns with no CRS metadata.
2. **Reproject geographic coordinates to the site's UTM zone** before they
   ever reach `createSurvey`/`extract_sample_positions_from_survey`, using
   `targetProjection='EPSG:326XX'` (UTM Zone 33N per the Alento constants —
   confirm the correct EPSG for the actual real-data site) whenever the
   geographic alias was used. `Problem.createSurvey` already accepts
   `targetProjection`; today it's just never passed a non-`None` value on
   the real-data path.
3. **Add a load-time sanity check** independent of the reprojection fix
   itself: immediately after building `survey_df` (both in
   `_load_real_data`'s position load and inside each Step 1 worker), compare
   the numeric range of `x`/`y` against the known site bounding box (reuse
   the `ALENTO_UTM_ORIGIN_X/Y` + extent constants, or read them from
   `probe_table['utm_x'/'utm_y']`'s own min/max as a self-consistent
   reference). If the ranges differ by more than ~10×, raise a `[WARN]` (or
   hard error) naming the suspected CRS mismatch, instead of only surfacing
   it indirectly via a 100%-fallback matching summary several steps later.
4. Once coordinates are confirmed consistent, `window_m` itself does not
   need to change — the existing window/fallback logic and its diagnostics
   (`ProbeMatchingSummary`, `plot_probe_matching.py`) are sufficient to
   verify correctness after the CRS fix; no changes needed there.

### Interaction with Issue 2
Real-data `sample_positions` (the probe→EM-spatial-index map) is currently
computed **once**, against `EM_FNAMES[0]`'s `survey_df` only
(`station_position_results = EMmethod.extract_sample_positions_from_survey(...)`
in `run_pipeline.main`, real-data branch), and the resulting
`station_to_spatial_idx` is reused for every time step via
`sample_positions = [station_to_spatial_idx[obs['probe_index']] for obs in observations_list]`.
This is only valid if "spatial index `i`" means the same physical location
at every epoch — which is exactly what Issue 2's row-count trim currently
does **not** guarantee. If Issue 2 is fixed via a shared reference grid
(Section 2), probe matching should be run against that same reference grid
once, and it will then be self-consistent by construction. If Issue 2 is
*not* fixed first, fixing Issue 3's CRS bug alone will make the window
matching geographically correct **for epoch 0 only**, while other epochs'
"same index" points may still silently refer to different physical
locations. **Recommend fixing Issue 2 before or together with Issue 3**,
since Issue 3's correctness assumes Issue 2's indexing assumption already
holds.

---

## Suggested implementation order

1. **Issue 1 (units)** — trivial, isolated, no interaction with the other
   two. Fix first; unblocks getting physically plausible ECa magnitudes at
   all, which is a prerequisite for usefully debugging Issues 2–3 (garbage
   EC magnitudes make it hard to tell whether a bad window-match result is
   due to CRS or unit scaling).
2. **Issue 3 diagnosis step only** — confirm via existing
   `matching_summary`/`plot_probe_matching.py` outputs (no code change)
   whether the CRS-mismatch hypothesis holds for the actual data in hand,
   before investing in the reprojection fix.
3. **Issue 2 (reference-grid spatial matching for time-lapse epochs)** —
   the largest change; do this before finalizing Issue 3's fix so probe
   matching lands on a stable, already-correct spatial index.
4. **Issue 3 (CRS fix + load-time sanity check)** — implement against the
   post-Issue-2 reference grid.

## Suggested new validations / tests
- `scenario_loader._validate`: hard-require `em_unit` for `data_source="real"`.
- Load-time coordinate-range sanity check (Issue 3, item 3) — cheap, catches
  both the CRS bug and any future instrument/column-alias regression.
- A regression test asserting `matching_summary.level != "ERROR"` (or a
  configurable fallback-fraction ceiling) as part of any real-data CI run,
  so a reintroduced CRS/window bug fails loudly instead of silently
  degrading calibration quality.
- After the Issue 2 rewrite: a test that a synthetic two-epoch fixture with
  deliberately different point counts and orders (i) loses zero points that
  are within tolerance of the reference grid, and (ii) never mixes two
  different physical locations under the same spatial index.
