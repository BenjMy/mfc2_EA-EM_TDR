# Kernel-weighted localization — 2-D map + bandwidth sweep added

**New in this pass** (on top of the earlier gap-fix — see below):

## 1. Proper 2-D map plot

`2_run_petro.py` gains `plot_localized_sigma_s_map(...)`: one panel per
horizon, a true scatter map over `(target_x, target_y)` colored by any
fitted parameter (default `sigma_s`), shared color scale across horizons,
uncalibrated targets drawn as gray x's, TDR probes overlaid as black
triangles. Raises a clear error if called on a 1-D fit (`target_y is None`)
rather than drawing a meaningless map.

`run_petro_localized(...)` now calls this automatically whenever
`point_y_coords` was given, producing
`petro_calibration_localized_map.png` alongside the existing x-only
profile plot — no extra wiring needed on your end.

## 2. Bandwidth selection via leave-one-probe-out (LOO) CV

Added because the in-sample `LocalizedHorizonWaxmanSmitsParams.rmse`
can't be used to pick `bandwidth` honestly: as `bandwidth -> 0`, a target
at a probe's own location just memorizes that probe's reading, and its
"RMSE" goes to ~0 for the wrong reason.

- `em_methodology.py`: `sweep_localization_bandwidth(...)` — for each
  candidate `c` and each observation, refits with that observation
  EXCLUDED (weighted by distance to its own location), predicts at its
  own location, and pools the held-out residuals into one LOO-CV RMSE per
  bandwidth. Returns `BandwidthSweepResult` per candidate (`loo_rmse`,
  per-horizon breakdown, `n_folds`, `n_fallback`).
- `2_run_petro.py`: `run_bandwidth_sweep(...)` driver — prints a per-
  candidate summary, calls `plot_bandwidth_sweep(...)` (RMSE-vs-bandwidth
  curve with the recommended minimum marked, plus a fallback-count panel),
  and returns `(sweep_results, recommended_bandwidth)`.
- Verified on a synthetic smoke test: too-small bandwidths showed
  100%-fallback and high LOO-CV RMSE, RMSE bottomed out at a mid-range
  bandwidth, then crept back up toward the global-fit RMSE as
  `bandwidth -> large` — the expected U/L-shaped curve.
- Intended as a one-off tuning step (run once against a reference model,
  e.g. Step 2's rank-1 member) to choose `ws_localized_bandwidth` for the
  scenario config, not something to run inside every pipeline execution.

Both additions compile clean and were smoke-tested end-to-end (including
the plots, attached as reference renders in this conversation).

---

# Kernel-weighted localization — gap analysis + fixes applied

**Status: both gaps below are now fixed** in the attached files
(`em_methodology.py`, `2_run_petro.py`, `run_pipeline.py`). The original
gap analysis is kept further down for reference; this section describes
what changed.

## Fixes applied

1. **2-D Euclidean localization.** `calibrate_petrophysical_relationship_localized`
   now takes an optional `point_y_coords` argument. When given, distances
   fed into `gaspari_cohn` become `sqrt(dx**2 + dy**2)` instead of the
   signed 1-D `dx`; when omitted, behaviour is unchanged (default `None`).
   `gaspari_cohn` itself needed no change — it already only ever uses
   `|dist|`, so a nonnegative Euclidean distance is a drop-in replacement
   for the signed 1-D difference. `run_petro_localized` in `2_run_petro.py`
   forwards this straight through and logs which distance mode was used.
   `LocalizedHorizonWaxmanSmitsParams` gained a `target_y` field (`None` in
   the 1-D case) so a genuine 2-D map can be reconstructed later if needed,
   alongside the existing `target_x`-only diagnostic plot (which now notes
   in its docstring that it projects onto x only, even in 2-D mode).

2. **`run_pipeline.py` wiring.** Added the diagnostic-sibling call described
   below, right after `plot_ws_parameter_histograms` and before
   `reporter.end_step("Step 2")`. It's gated on
   `getattr(cfg, "ws_localized_bandwidth", None)`, wrapped in try/except so
   a failure can't take down an otherwise-successful run, and picks 1-D vs
   2-D distance automatically: a new `y_transect` (added alongside the
   existing `x_transect`, from the same `survey_df`/reference-grid `"y"`
   column, in both the real-data-timelapse and single-epoch branches) is
   only passed as `point_y_coords` when `mode == "2d"` and a UTM grid was
   actually loaded (`utm_y is not None`); otherwise it stays `None` and the
   original 1-D-transect behaviour is preserved exactly.

Everything below this line is the original gap analysis, unchanged.

---

# Kernel-weighted localization — what's missing to complete it

**Short answer: the localization method itself is fully implemented. The only
missing piece is wiring it into `run_pipeline.py` — it's built but never
called.**

## What already exists (confirmed in the uploaded files)

`em_methodology.py`:
- `gaspari_cohn(dist, c)` — the compact-support kernel, implemented exactly
  as in your snippet (lines ~1802–1838).
- `calibrate_petrophysical_relationship_localized(...)` — refits
  `[a, m, sigma_w, sigma_s]` at every target EM point, multiplying
  `gaspari_cohn(obs_x - target_x, bandwidth)` into the existing
  `theta_obs_std`-derived `obs_weights` before calling the same
  `_log_ratio_objective` (lines ~2030–2280+). `bandwidth → ∞` reproduces the
  global fit; `bandwidth → 0` collapses to nearest-neighbour — exactly the
  generalization you described.
- `LocalizedPetroFit` / per-horizon result dataclass carrying `target_x`,
  `sigma_s` (array), `is_calibrated` (array), `bandwidth`.

`2_run_petro.py`:
- `run_petro_localized(...)` — the driver function, sibling to `run_petro`,
  calls the methodology function and prints a per-horizon summary.
- `plot_localized_sigma_s_profile(...)` — the diagnostic plot that actually
  shows σ_s(x) varying continuously per horizon (solid where calibrated,
  dotted where no observation falls within `2×bandwidth`), with probe
  positions overlaid as vertical reference lines.

`scenario_loader.py`:
- `ws_localized_bandwidth: float | None = None` field on the scenario
  config, fully validated (must be `None` or a positive float), with a
  docstring that **explicitly states**:
  > "When set, `run_pipeline.py` additionally runs
  > `step2.run_petro_localized` as a DIAGNOSTIC sibling to the normal Step 2
  > calibration."

## What's missing

**`run_pipeline.py` never reads `ws_localized_bandwidth` and never calls
`step2.run_petro_localized`.** Grepping the pipeline file for
`localized`/`bandwidth`/`gaspari` returns nothing — the scenario field is
parsed and validated by `scenario_loader.py` but silently ignored by the
pipeline that's supposed to consume it. Right now, setting
`ws_localized_bandwidth` in a scenario JSON has zero effect on any run.

Concretely, in `run_pipeline.py`:

1. No `WS_LOCALIZED_BANDWIDTH = getattr(cfg, "ws_localized_bandwidth", None)`
   pulled out alongside the other WS config values (`WS_FIXED_PARAMS`,
   `LAB_P0` are pulled this way around line 1231/1239 — bandwidth has no
   equivalent line).
2. No call to `step2.run_petro_localized(...)` anywhere in the Step 2 block
   (the only `step2.` call is `step2.run_petro(...)` at line ~2926).
3. No `point_x_coords` argument assembled for it — the pipeline already has
   the right array sitting in `x_transect` (built at line ~2512/2549, one
   x-coordinate per measured point, already used for exactly this purpose in
   Step 6's `pwc.plot_from_pipeline(x_transect=...)`), it's just never
   forwarded to the localized calibration.
4. No log line reporting whether the localized diagnostic ran or was
   skipped, so a user has no way to tell from the report whether their
   `ws_localized_bandwidth` setting actually did anything.

### The fix

Insert this right after the existing Step 2 block (after the
`plot_ws_parameter_histograms` call, before `STEP 3` begins, i.e. after line
~3010 in the current file), reusing values already computed earlier in the
same Step 2 section (`x_transect`, `theta_obs`, `sample_positions`,
`sample_times`, `probe_layer_indices`, `EFFECTIVE_DEPTHS0`, `WS_FIXED_PARAMS`,
`LAB_P0`, `_theta_obs_std_step2`):

```python
# ------------------------------------------------------------------
# Step 2 (diagnostic sibling) — kernel-localized Waxman-Smits calibration
# (spatial_calibration_alternatives.md §2). Does NOT feed Steps 3-5;
# only produces petro_calibration_localized.png for inspection.
# ------------------------------------------------------------------
WS_LOCALIZED_BANDWIDTH = getattr(cfg, "ws_localized_bandwidth", None) if cfg is not None else None
if WS_LOCALIZED_BANDWIDTH is not None:
    reporter.log(
        f"[STEP 2 LOCALIZED] ws_localized_bandwidth={WS_LOCALIZED_BANDWIDTH:g} set -- "
        f"running kernel-localized calibration as a diagnostic sibling to Step 2."
    )
    try:
        fits_step2_localized = step2.run_petro_localized(
            models=models_clean,
            sample_positions=sample_positions,
            sample_times=sample_times,
            theta_obs=theta_obs,
            point_x_coords=x_transect,
            bandwidth=WS_LOCALIZED_BANDWIDTH,
            depths0=DEPTHS0,
            output_dir=OUTPUT_DIR,
            probe_layer_indices=probe_layer_indices,
            fixed_params=WS_FIXED_PARAMS,
            lab_p0=LAB_P0,
            theta_obs_std=_theta_obs_std_step2,
        )
        reporter.log(
            f"[STEP 2 LOCALIZED] Diagnostic complete -> "
            f"{OUTPUT_DIR}/petro_calibration_localized.png"
        )
    except Exception as e:
        # Diagnostic sibling only -- never let it fail the real run.
        reporter.log(f"[STEP 2 LOCALIZED][WARN] Localized diagnostic failed: {e}")
else:
    reporter.log(
        "[STEP 2 LOCALIZED] ws_localized_bandwidth not set -- skipping "
        "kernel-localized diagnostic (unchanged default behaviour)."
    )
```

Notes on this insertion:

- **Wrapped in try/except and placed after the real Step 2 outputs are
  already produced**, matching how the other optional diagnostics in Step 6
  (`plot_truth_from_pipeline`, `plot_pct_diff_from_pipeline`) are guarded —
  a failure here must not take down a run that would otherwise succeed,
  since `run_petro_localized`'s docstring itself says it's a *sibling*, not
  a replacement.
- **Does not touch `fits_step2`** or anything Steps 3–5 consume — exactly as
  the scenario field's docstring promises ("does not replace fits_step2 and
  does not change Steps 3-5").
- **`x_transect` as `point_x_coords`** is the correct array: it's already
  one x-coordinate per measured point (same length/order `sample_positions`
  indexes into), and it's the same array Step 6 passes as `x_coords` for
  analogous per-point plotting.

## One real limitation to flag, not fix silently

`calibrate_petrophysical_relationship_localized` / `gaspari_cohn` only take
a single `point_x_coords` (1-D distance). In **2-D mode** (`mode == "2d"`,
`utm_x`/`utm_y` both populated), the pipeline has genuine XY probe geometry
(`PROBE_X_COORDS`, `PROBE_Y_COORDS`, `UNIQUE_XY_COORDS`), but the localized
calibration as written can only localize along `x_transect`, ignoring `y`
entirely. That's fine for 1-D transects (the common case here) but would
silently under- or over-weight observations that are close in `y` but far
in `x` (or vice versa) in a true 2-D survey. Worth either:
- explicitly restricting `ws_localized_bandwidth` to 1-D-mode scenarios
  (validate + warn in `scenario_loader.py` if `mode == "2d"`), or
- extending `gaspari_cohn`'s distance argument to Euclidean XY distance
  before this is used on a real 2-D dataset.

This isn't blocking for the fix above (it reproduces exactly what
`run_petro_localized` already assumes), but it should be documented so a 2-D
run doesn't get a false sense of precision from the localized plot.

## Everything else your snippet touched on

- The `c`-is-empirical-same-as-`shrinkage_k` framing, `bandwidth → ∞ / → 0`
  limiting behaviour, and the "why localization" ensemble-DA framing are all
  already captured almost verbatim in the code's own docstrings/comments —
  no gap there, just confirming your reasoning matches what's implemented.
- No `shrinkage_k` / `min_obs_per_position` implementation was found
  anywhere in these files, so there's nothing to "replace" — the localized
  method is the only spatial-pooling scheme built for `calibrate_
  petrophysical_relationship`, not an alternative competing with a shrinkage
  version already in the pipeline.
