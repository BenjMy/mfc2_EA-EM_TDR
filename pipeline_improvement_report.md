# EA-EM Pipeline — Improvement Recommendations

**Scope:** general recommendations based on a review of `run_pipeline.py`, `em_methodology.py`, `scenario_loader.py`, the Step 0–6 modules, and one real-data run (`2d_real`, run `20260709_213341`). Findings from that specific run are used as illustrations, not as the sole basis for the recommendations.

---

## 1. Executive summary

The pipeline is architecturally sound — clear step separation, a real run reporter, config validation, caching with fingerprinting, and defensive coding around several previously-fixed bugs (bounds, layer routing, coordinate matching). The main gaps are **not in engineering robustness but in scientific validity and observability**:

1. The petrophysical calibration (Waxman-Smits fit) has no built-in sanity checks, so it can silently converge to physically implausible or bound-saturated parameters and nothing flags it.
2. Spatial matching between TDR probes and EM measurement points can silently collapse many probes onto very few EM indices, and this is only visible by manually reading a log line.
3. Uncertainty and calibration-quality information (CV, per-horizon coverage, bound-hits) is computed but not surfaced as pass/fail gates — everything proceeds even when results are unusable.
4. ~96% of wall-clock time is a single monolithic Step 0, with no incremental/partial re-run path other than an all-or-nothing pickle cache.
5. There's real duplication (three separate `_layer_centers` implementations, two full copies of `calibrate_petrophysical_relationship`, parallel `_build_inversion_bnds` in two files) that increases the chance of exactly the kind of "one path fixed, one path not" bug already seen.

The recommendations below are grouped by theme and roughly ordered by expected impact within each group.

---

## 2. Petrophysical calibration (Waxman-Smits) — highest priority

### 2.1 Add automated sanity checks on fitted parameters
Right now a fit can converge with `σ_w` pinned exactly at its upper bound (1000 mS/m) and nothing distinguishes that from a well-converged fit — both get reported identically. Add a post-fit check per horizon:
- Flag (log + report table) any parameter that lands within, say, 1% of its `WS_BOUNDS` edge.
- Flag any `theta_pred` / `theta_calc` values outside a configurable physical plausibility range (e.g. `[0.02, 0.55]` m³/m³ by default).
- Surface these flags in the `PipelineReporter` output, not just as a `print()` — they currently don't appear in the markdown report at all, even though the underlying numbers (σ_w = 1000.00 mS/m exactly) are sitting right there in the tables.

### 2.2 Reconsider defaulting to fixed `a`/`m`
`ws_fixed_params` lets users pin `a` and `m` to lab values, which is reasonable when lab data exists and TDR coverage per horizon is thin (the documented rationale). But there's no guard against pinning values that are simply wrong for the field being surveyed (e.g. `m=3.0`, more typical of cemented/consolidated media than field soil). Consider:
- A one-time "unconstrained calibration" diagnostic pass (fit `a, m, σ_w, σ_s` all free on a subset of the data) that's compared against the fixed-parameter result, so users can see whether the fixed values are actually compatible with the data before committing to them for the full ensemble.
- Documenting in `ensemble_scenarios.json`'s schema comments what physically reasonable ranges for `a`/`m` look like for common soil textures, since the JSON is the main interface users touch.

### 2.3 Make the log-ratio objective's degenerate cases visible
`_log_ratio_objective` divides by `log_ert` (`data_misfit = np.sum(np.abs((log_ert - log_ws) / log_ert))`), which is unstable as `log_ert → 0` (i.e. σ_b near 1 mS/m). There's no check for this, and it would silently distort the objective landscape rather than raise anything. Consider switching to a symmetric relative-error form (e.g. `abs(log_ert - log_ws)` without the division, or `/max(abs(log_ert), eps)`), and/or logging when `sigma_b_layer` values are close to 1 mS/m.

### 2.4 Consolidate the two calibration implementations
`OLD_calibrate_petrophysical_relationship` still exists in full alongside the current `calibrate_petrophysical_relationship`, ~300 lines of near-duplicate code. If it's kept for reference, move it to a `legacy/` module or delete it — as-is, it's a maintenance hazard (a fix applied to one and not the other, as already happened once with `probe_depths_reflect_fitted_interfaces`).

---

## 3. Spatial matching (TDR ↔ EM) — high priority

### 3.1 Make many-to-one matches visible and gate-able
`extract_sample_positions_from_survey` can map many probes onto the same EM `representative_index` (11 of 20 stations onto one point index in the run reviewed). The function already knows this happened (it's implicit in the `indices`/`representative_index` fields) but doesn't report it. Add:
- A summary count in the reporter: "N distinct EM points used out of M probes" and "largest group size."
- A configurable warning/error threshold (e.g. "> 50% of probes share fewer than X distinct EM points") that fails fast rather than silently degrading calibration quality.

### 3.2 Distinguish genuine nearest-neighbor matches from window fallbacks
When `window_m` fails to find any point, the code falls back to nearest-neighbor with a `print()`-only warning. These fallback cases should be counted and reported in the run report (not just stdout), since they indicate the EM survey geometry doesn't actually cover a probe — a very different situation from a well-matched probe.

### 3.3 Visualize the matching itself
`plot_ini_model.py` / diagnostics already draw ECa and SWC as separate XY maps. Add one more diagnostic: EM survey points colored by "number of probes matched to this point," with probe locations overlaid and connected by a line to their matched EM point. This would have made the 219-collapse issue visible in seconds rather than requiring log inspection.

---

## 4. Uncertainty & quality as gates, not just numbers

Currently CV maps, coverage maps, and per-horizon calibration status are computed and plotted, but nothing in the pipeline *acts* on them — a run with 30% of cells above CV=0.5, or two of four horizons entirely uncalibrated, completes with the same "SUCCESS" status as a clean run.

Suggestions:
- Add a configurable quality-gate step (post Step 5) that downgrades the run's final status to a warning state (distinct from ✅ SUCCESS) when e.g. `mean(cv_map) > threshold`, `fraction(cv_map > 1.0) > threshold`, or `n_calibrated_horizons < n_layers`.
- Surface a compact "data coverage" summary early (right after Step 0 real-data loading): which horizons have TDR coverage, which don't, given `depths0` vs actual probe depths. This is knowable before the 30-minute inversion even runs, and would let a user fix `depths0` (e.g. match sensor depths) before spending the compute.

---

## 5. Performance & iteration speed

### 5.1 Step 0 dominates runtime with no partial-progress option
In the reviewed run, Step 0 (432 dispatched inversion tasks) took 1940 s — 96% of total wall time — while everything downstream (calibration through final plots) took under 90 s combined. The cache is all-or-nothing: any config drift invalidates the entire ensemble.

- Consider checkpointing inversion results incrementally (e.g. one file per completed task, merged at the end) so a crash or interrupt during Step 0 doesn't lose all progress, and so `--resume` becomes possible.
- Consider a `--dry-run` / `--sample-grid` mode that runs Step 0 on a small subset of the `param_grid × cond_ranges` combinations (e.g. 5% of the 432 tasks) end-to-end through Step 6, so users can catch downstream plotting/shape bugs (like the one that triggered this review) in seconds instead of 33 minutes.

### 5.2 Parallelism granularity
Tasks are dispatched per (combo × time-step), each spinning up its own `emagpy.Problem` and re-reading the CSV. For real-data runs with many time steps this means re-parsing the same survey file across many worker processes. Pre-parsing once and passing lighter-weight arrays (or using a shared read-only mapped structure) could reduce Step-0 overhead further, though the current per-process isolation is deliberate (documented as protecting against EMagPy cross-process state issues) — any change here should preserve that isolation.

---

## 6. Code duplication & consistency risk

Several pieces of logic are implemented more than once, which is exactly how the `_layer_centers` shape-mismatch bug happened (correct in the synthetic generator, incomplete in `run_pipeline.py`'s copy):

| Duplicated logic | Locations |
|---|---|
| Layer-center depth calculation | `0_generate_synthetic_data.py` (inline), `run_pipeline.py::_layer_centers`, `run_pipeline_short.py::_layer_centers` |
| Inversion bounds builder | `run_pipeline.py::_build_inversion_bnds`, `1_run_model_generation.py::_build_inversion_bnds` |
| Waxman-Smits calibration | `em_methodology.py::calibrate_petrophysical_relationship` and `OLD_calibrate_petrophysical_relationship` |
| EC-at-probe extraction/routing | `2_run_petro.py::_extract_ec_for_plots`, `4_run_select_assess.py::_extract_ec_at_probes` (similar horizon-routing logic, separately implemented) |

**Recommendation:** move these into `em_methodology.py` (or a small shared `geometry_utils.py`) as the single source of truth, and have every other module import from there. This is the single highest-leverage code-health change — it directly prevents the class of bug that caused the Step 6d crash.

---

## 7. Real vs. synthetic pipeline parity

The real-data branch of `run_pipeline.py::main()` is built as a set of parallel code paths alongside the synthetic branch (separate `synthetic_geometry` construction, separate observation-array assembly, separate probe routing), rather than both branches funneling into one shared representation early. This is understandable given how differently the two data sources arrive, but it means every downstream consumer needs to independently support both shapes/keys, and it's easy for one branch to lag the other (again, the `layer_centers` bug is a direct instance of this pattern — the synthetic path got a feature "for free" that the real-data path had to reimplement and got wrong).

**Recommendation:** define a single `RunGeometry` (or similar) dataclass produced by *both* branches through the same constructor function, with required fields (`layer_bottoms`, `layer_centers`, `num_layers`, `mode`, `utm_x`/`utm_y` optional) validated at construction time (e.g. `assert len(layer_centers) == num_layers`). This turns the class of bug that just occurred into an immediate `AssertionError` at Step 0 instead of a crash 33 minutes later in Step 6d.

---

## 8. Configuration & scenario management

- `ensemble_scenarios.json`'s `_comment` block is a good idea (self-documenting schema) but is easy to let drift from the actual `ScenarioConfig` dataclass fields — consider generating this documentation from the dataclass docstrings/field metadata instead of maintaining it by hand in JSON.
- `scenario_loader.validate_all()` checks structural validity (types, ranges, required fields) but not cross-field physical plausibility (e.g. `swc_depths` all shallower than `depths0`'s deepest interface, meaning some horizons can never be calibrated — exactly what happened in the reviewed run). Adding this check to `_validate()` would catch the horizon-2/3-uncalibrated situation at scenario-load time, before any inversion runs.

---

## 9. Testing

No test suite is visible among the provided files. Given the amount of defensive/fix commentary already embedded in the code (many `# FIX:` comments referencing specific past bugs), a regression suite would pay for itself quickly:
- Unit tests for the shape-sensitive helpers (`_layer_centers`, `_build_inversion_bnds`, `extract_sample_positions_from_survey`) with both 1-D/2-D and fixed/free-depth cases.
- A small synthetic end-to-end smoke test (few points, 1 time step, 2 ensemble members) that runs Steps 0–6 and asserts output file existence + basic shape consistency — this would have caught the `layer_centers` mismatch immediately without needing 33 minutes of real inversion.

---

## 10. Prioritized action list

| # | Action | Effort | Impact |
|---|---|---|---|
| 1 | Add physical-plausibility / bound-saturation checks on WS fit parameters, surfaced in the report | Low | High |
| 2 | Add a `RunGeometry`-style shared construction path for synthetic/real branches, with assertions | Medium | High |
| 3 | Consolidate duplicated layer-center / bounds / calibration logic into one module | Medium | High |
| 4 | Report many-to-one / fallback TDR↔EM matches, add a configurable failure threshold | Low | Medium-High |
| 5 | Add a `--dry-run` fast path through the full pipeline for catching downstream bugs cheaply | Medium | Medium-High |
| 6 | Add cross-field scenario validation (sensor depth vs. `depths0` coverage) | Low | Medium |
| 7 | Add a regression/smoke test suite | Medium | Medium (long-term high) |
| 8 | Checkpoint/resume support for Step 1 inversion tasks | Medium-High | Medium |
| 9 | Remove/relocate `OLD_calibrate_petrophysical_relationship` | Low | Low-Medium (hygiene) |

Items 1–4 directly address both the crash that triggered this review and the calibration-quality concerns raised about the ECa→θ conversion; items 5–9 are general robustness/velocity improvements.

---

## 11. Synthetically comparing EM footprint vs. TDR footprint

### 11.1 Why this matters

EM (EMI) and TDR measure soil moisture at fundamentally different **support volumes** (the physical volume of soil each reading actually represents), and the mismatch between them is a common source of poor calibration that's easy to misdiagnose as "bad parameters" when it's actually a scale mismatch:

- **TDR** is a near-point measurement: its sensing volume is a thin cylinder around the rods (radius of a few cm), at one precise, fixed depth. It's spatially sparse but essentially exact where it is.
- **EM (EMI)** is a laterally- and vertically-*integrated* measurement: each reading is a depth-weighted average of conductivity over a volume whose size and shape depend on coil spacing, orientation (HCP/VCP), and frequency — the classic McNeill-type cumulative response function. A single ECa reading can represent tens of cm to a few meters both horizontally and vertically, and different coils in the same ensemble (e.g. `COILS_LIST` in `0_generate_synthetic_data.py`) have different depths of exploration.

When calibrating EC↔θ (Step 2/4), the pipeline implicitly assumes an EM-derived "layer EC" and a TDR "point θ" from the same horizon are comparable. In reality the EM value is smoothed over a footprint the TDR reading knows nothing about, and vice versa — the TDR reading may be dominated by local micro-heterogeneity (root channels, a stone, a wet spot right under the rod) that the EM footprint averages away. This is a strong candidate explanation for calibration RMSE that won't go away no matter how the WS parameters are tuned.

### 11.2 Interpreting each measurement type

| Property | TDR | EM (EMI) |
|---|---|---|
| Support volume | ~cm³ (near-point) | ~m³, footprint grows with coil spacing/depth |
| Depth control | Exact, fixed by installation | Approximate, weighted/blended across the profile (varies by coil) |
| Spatial density achievable | Sparse (installation cost) | Dense (walked survey) |
| What a single reading represents | The soil at that exact spot | A spatially- and depth-weighted average — effectively a **low-pass-filtered** version of the true field |
| Best interpreted as | Ground truth at a point | An indirect, smoothed proxy that needs a petrophysical/inversion model to become θ |

The practical implication: EM should be interpreted as recovering the **large-scale, smoothed structure** of soil moisture, while TDR anchors the **absolute value and fine-scale detail** at specific points. Calibration is fundamentally trying to bridge a smoothed measurement to point ground truth — some irreducible mismatch is expected, and the goal is to *quantify* it rather than treat it purely as fitting error.

### 11.3 How to test this synthetically with the existing pipeline

This pipeline is well suited to this because in **synthetic mode** the true, unsmoothed θ field (`theta_true_realization` / `synthetic_time_lapse_theta.npy`) is known at *every* measurement point and layer, before any noise or EM forward-modelling is applied. That true field is effectively "a TDR probe installed everywhere" — so instead of only comparing EM against the handful of `num_swc_positions` probes as the pipeline does today, you can compare the point-by-point true field against the EM-recovered field at every single grid node. Concretely:

**Step A — Generate a controlled synthetic ground truth with known spatial structure.**
Use `0_generate_synthetic_data.generate_time_lapse_dataset(mode="2d", ...)` with a fine grid (`grid_num_x`/`grid_num_y` pushed higher than the default 10×5) so there are many virtual "TDR" locations. The existing `homogeneous` flag and the `xy_gradient`/`xz_gradient` terms already give you smooth heterogeneity; for a footprint study you specifically want *sharp* structure, so this is the one place worth a small code addition:

- Add a `pattern` option to `generate_time_lapse_dataset` (e.g. `"linear"` = current behaviour, `"patch"` = a step change in θ across a boundary in the XY plane, `"point_anomaly"` = a single wet/dry cell in an otherwise uniform field). A patch/step pattern is the standard way to probe lateral resolution; a single point anomaly turns the experiment into a literal **impulse response** measurement of the EM footprint.
- For vertical footprint, do the analogous thing across `depths0` — e.g. a sharp θ step between two adjacent layers — and see how much the recovered layer-EC "leaks" between layers relative to the true step.

**Step B — Run the two "sensors" through their own physics on the same truth.**
- *TDR-equivalent:* just read `theta_true_realization` directly at each grid point/layer — this is what a TDR would report if placed there (optionally add TDR's own small, local Gaussian noise via `tdr_noise_std`, since real TDR isn't perfectly exact either).
- *EM-equivalent:* let the existing pipeline do what it already does — `forward_waxman_smits` → `Problem.forward()` → ECa (Step 0), then the full Step 1–5 ensemble inversion → `selected_output.theta_calc`. This is the EM instrument's actual footprint-integrated, model-mediated estimate of θ at the same nominal locations.

**Step C — Compare the two fields directly, point by point and in aggregate.**
This is new analysis, not something the pipeline currently produces, but it composes cleanly from existing outputs:
1. **Residual map**: `theta_calc - theta_true_realization` at every (position, layer, time), plotted the same way `plot_solution_diagnostics.py` already plots θ (XZ/XY), but colored by residual instead of θ itself. Large, spatially coherent residual patches indicate the EM footprint blending across a real gradient the TDR would have caught exactly.
2. **Boundary/edge response** (for the "patch" pattern): plot θ_true vs. θ_calc as a function of distance from the true boundary. The EM-recovered curve will be a smoothed sigmoid instead of the true step; the horizontal distance over which it transitions (e.g. 10–90% of the step) is a direct, empirical estimate of the **effective lateral footprint** for that coil/grid configuration.
3. **Point-response / impulse footprint** (for the "point_anomaly" pattern): the recovered θ around the anomaly approximates the instrument's 2-D point-spread function. Its half-width at half-maximum is a practical footprint-radius estimate, comparable across different coil sets (`COILS_LIST` in `0_generate_synthetic_data.py`) or different `window_m` matching radii.
4. **Spatial spectrum / variogram comparison**: compute a simple empirical variogram (or even a 1-D FFT along a transect) of `theta_true_realization` vs. `theta_calc`. The EM-recovered field should show suppressed variance at short lag distances relative to truth — quantifying "how much fine-scale detail EM cannot see" independent of any particular boundary shape.
5. **Sensitivity to `window_m`**: re-run the TDR-matching step (`extract_sample_positions_from_survey`) with several `window_m` values on the *same* synthetic truth and compare resulting calibration RMSE and residual maps. This directly shows how the *TDR-side* averaging window trades off against the EM's own intrinsic footprint — useful for choosing `window_m` on real data (e.g. the `2d_real` run's window_m=1.0 with its clustered EM-index matches, discussed above).

**Step D — Package it as a reusable diagnostic.**
Given the shape of the existing codebase, the cleanest way to add this is a new `plot_footprint_comparison.py` module (mirroring the style of `plot_solution_diagnostics.py`), with a single entry point like:

```python
plot_footprint_comparison(
    theta_true,       # synthetic_time_lapse_theta.npy — the "everywhere-TDR" truth
    theta_calc,       # selected_output.theta_calc — the EM-recovered field
    x_coords, y_coords, depths0,
    output_dir=...,
)
```
producing the residual map, boundary-response, and variogram figures described above. Because it only needs arrays the pipeline already produces (plus the new `pattern` option in Step 0 for the boundary/impulse cases), it slots in as an optional Step 6-adjacent diagnostic without touching the core inversion/calibration logic — and it directly gives an empirical, data-driven answer to "how comparable are my EM and TDR measurements here," rather than relying on generic literature footprint numbers that don't account for this survey's specific coil set, grid spacing, and inversion settings.
