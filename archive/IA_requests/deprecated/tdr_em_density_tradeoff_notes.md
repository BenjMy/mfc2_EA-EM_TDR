# TDR ↔ EM Density Trade-off: What the Pipeline Already Answers, What It's Missing, and How to Write It Up

This note is aimed at the "how many TDR sensors do we need, and at what
depths, to reliably convert EM to θ" question, and at what the codebase
would need to say/produce so that question becomes a defensible, citable
figure in a paper rather than an anecdote in the discussion section.

---

## 1. Restating the trade-off precisely

EM (GEM-2) and TDR are complementary along orthogonal axes:

| | Spatial density | Temporal density | What it measures |
|---|---|---|---|
| **EM survey** | Extensive (thousands of points/campaign) | Sparse (2–7 campaigns over ~1.5 yr in the MFC2 dataset) | Bulk EC, an *indirect* proxy for θ via Waxman–Smits |
| **TDR/SWC probe** | Sparse (tens of fixed stations) | Dense (sub-daily, continuous) | θ directly, but only at the probe's own depth and footprint |

The pipeline's actual scientific product is not "the EM map" or "the TDR
series" — it's the **petrophysical bridge** between them
(`em_methodology.calibrate_petrophysical_relationship`, Waxman–Smits
per-horizon fit). The paper's central methodological question is therefore:
*how few, how sparse, how shallow can the TDR network be before that
bridge stops being trustworthy at the horizons/locations EM alone has to
extrapolate to?* That is a **sample-size / experimental-design** question,
and the codebase already contains most of the machinery to answer it
quantitatively — it's just not yet organized as a sweep with an aggregate
output figure.

---

## 2. What the pipeline already computes that's directly usable

These aren't hypothetical — they're implemented, and each maps to a
specific figure/table for the paper:

1. **Per-horizon calibration status**
   `HorizonWaxmanSmitsParams.is_calibrated`, surfaced pipeline-wide via
   `PetroFit.n_calibrated_horizons` and
   `em_methodology.check_early_horizon_coverage` (Step 0b, runs *before*
   the expensive inversion). This is the direct answer to "at which depths
   do we currently have zero TDR coverage" — it already fails loud rather
   than silently falling back to prior parameters.

2. **Probe→EM-point collapse diagnostics**
   `em_methodology.summarize_probe_matching` /
   `ProbeMatchingSummary.largest_group_fraction` and the
   `probe_em_matching*.png` figures (`plot_probe_matching.py`,
   `plot_pre_inversion_diagnostics.py`). This answers "of N nominal TDR
   stations, how many are *effectively* independent EC observations after
   spatial windowing" — directly relevant to "how many sensors do we need"
   because the honest denominator is *distinct EM points matched*, not
   *stations installed*.

3. **Calibrated-vs-uncalibrated skill contrast against known ground truth**
   `generate_synthetic_real_mirror.py` + `run_and_score.py` +
   `score_against_truth.py` is *exactly* the depth-extrapolation
   experiment: it builds a synthetic twin with the real deployment's
   station count/depths (15/30 cm) and known truth at *every* depth
   including ones no TDR reaches, then reports RMSE/bias/R² split by
   `calibrated` vs `uncalibrated` horizon. **This is the headline number**:
   "uncalibrated horizons cost you X m³/m³ RMSE relative to calibrated
   ones, at this sensor density." Right now it only runs at one density
   (whatever `mfc2_real_synthetic_mirror` encodes) — see §3 for extending
   it to a sweep.

4. **RMSE-vs-K stacking curve and CV/coverage maps**
   `4_run_select_assess.py`'s Figure 3 (`figure_3_rmse_curve.png`) and
   `EMmethod.compute_coverage_map` / `evaluate_quality_gates` quantify how
   much the *ensemble* uncertainty (not calibration uncertainty) shrinks
   with more inversion realizations — a different axis than TDR density,
   but worth explicitly separating in the paper's uncertainty budget
   (ensemble spread vs. petrophysical-calibration sparsity are two
   different uncertainty sources that are easy to conflate in the
   discussion section).

5. **Existing density scenarios**
   `ensemble_scenarios.json` already contrasts `2d` (2 SWC positions) vs.
   `2d_dense_tdr` (10 positions) and `moisture_2d_free` (6 positions) vs.
   `moisture_2d_dense_tdr` (12 positions) — a ready-made ablation pair, but
   currently there's no script that runs both and differences the output
   automatically.

---

## 3. Concrete experiment to answer "how many sensors, at what depths"

**Sweep `num_swc_positions` (and separately `swc_depths`) through the
synthetic-real-mirror pipeline, scoring each run against the same known
ground truth.**

```
for N in [2, 4, 6, 10, 16, 24]:               # station counts
    run_pipeline.main(cfg, ..., num_swc_positions=N)
    run_and_score.py --out capture_N{N}.npz
    score_against_truth.py --capture capture_N{N}.npz --out scoring_N{N}.csv
```

This is already parameter-complete in `run_pipeline.main()`
(`num_swc_positions` is a first-class override), so the missing piece is
purely a driver script + an aggregator — not new pipeline logic. Concretely:

- **New script `sweep_sensor_density.py`** (doesn't exist yet): loops
  `num_swc_positions` × `swc_depths` combinations, calls
  `run_and_score.py` + `score_against_truth.py` for each, and concatenates
  the per-combination `scoring_summary.csv` into one long-format table with
  an added `n_probes` / `probe_depths` column.
- **Output figure**: RMSE (pooled over calibrated horizons) vs.
  `n_probes`, one line per depth configuration — this *is* the "how many
  sensors" figure. The point where the curve flattens is your answer, and
  it's defensible because it's benchmarked against known ground truth, not
  against the TDR data used to fit it (which would be circular).
- **Depth question specifically**: hold `n_probes` fixed, vary
  `swc_depths` (e.g. all-shallow vs. one-per-horizon vs.
  deliberately-skip-one-horizon) and read off
  `check_early_horizon_coverage` + the calibrated/uncalibrated RMSE split.
  This directly demonstrates *why* depth allocation matters independently
  of raw count — a station at 15 cm and 45 cm is not equivalent to two
  stations at 15 cm for horizon-4 (>1 m) calibration.

This reuses 100% of existing scoring machinery; nothing about
`em_methodology.py`, `4_run_select_assess.py`, or `score_against_truth.py`
needs to change for the sweep itself.

---

## 4. Documentation improvements that would help the manuscript directly

- **`em_methodology.py` module docstring** doesn't currently state the
  central trade-off (spatial-extensive/temporally-sparse EM vs.
  temporally-dense/spatially-sparse TDR) anywhere — it's implicit in the
  code (`is_calibrated`, the horizon-routing warnings) but never named.
  Adding one paragraph there, and cross-referencing it from
  `check_early_horizon_coverage`'s docstring, means a reader (or a
  co-author skimming the repo before the paper is written) gets the
  framing for free.
- **`score_against_truth.py`** already says in its own module docstring
  *"this is the number that answers 'is the spatial map still consistent'
  even at epochs/horizons with NO direct TDR coverage"* — this is
  paper-ready language almost verbatim; it should be pulled into the
  Methods section directly rather than re-derived.
- **`ensemble_scenarios.json`**'s `_comment` block documents individual
  fields well but never states *why* `2d_dense_tdr` / `dense_tdr` variants
  exist (i.e., as one endpoint of the density ablation) — a one-line
  addition there turns the JSON into self-documenting evidence of the
  experimental design when a reviewer reads the supplementary code.
- **`generate_synthetic_real_mirror.py`**'s docstring is already
  excellent about *why* the synthetic twin exists ("real data can never
  tell you this") — worth lifting near-verbatim into the paper's synthetic
  validation subsection, since it's already been through several rounds of
  self-correction (see the NOTE about the unverified WS forward-model
  match, which a reviewer will ask about — resolve that TODO before
  submission, see §5).

## 5. One correctness note worth resolving before using this for a paper

`generate_synthetic_real_mirror.py::forward_waxman_smits` carries a
docstring NOTE flagging that its functional form
(`sigma = (theta**m/a)*sigma_w + sigma_s`) was never verified against
`em_methodology.py`'s actual forward form. Comparing against
`em_methodology.waxman_smits_theta`'s *inverse* relation
(`theta = (a·sigma_b/(sigma_w+sigma_s))^(1/m)`) — solving that for
sigma_b gives exactly `sigma_b = (theta**m/a)·(sigma_w+sigma_s)`, i.e.
**additive-sigma_s is NOT what the inverse implies**; the correct forward
form from `em_methodology.py`'s own convention is
`sigma_b = (theta**m/a)·(sigma_w+sigma_s)`, not
`(theta**m/a)·sigma_w + sigma_s`. `0_generate_synthetic_data.py`'s own
`forward_waxman_smits` (used for the *other* synthetic generator,
`0_generate_synthetic_data.py`) already uses the correct
`(theta**m/a)*(sigma_w+sigma_s)` form. **The synthetic-real-mirror
generator's WS forward model disagrees with the rest of the codebase and
should be fixed before the ground-truth-scoring experiment in §3 is run**
— otherwise the "known ground truth" the sweep is scored against was
built with a different constitutive relationship than the one the pipeline
calibrates, which would bias every calibrated/uncalibrated RMSE number in
a way that's hard to disentangle from a genuine sensor-density effect.

---

## 6. `PipelineReporter` additions that would make each run self-documenting for the paper

The reporter already logs a lot (matching summaries, coverage checks,
quality gates) but nothing ties a single run's report back to *where it
sits in the density ablation*. Suggested additions, all incremental
(no architecture change):

1. **A "Sensor Network Design" section**, written once near the top of the
   report (next to "Scenario Configuration"), pulling together in one
   place what's currently scattered across Steps 0b/1/2: `num_swc_positions`,
   resolved `swc_depths`, per-horizon probe counts
   (`check_early_horizon_coverage`'s `horizon_probe_counts`), and the
   post-matching `n_distinct_em_points` — i.e. "nominal N stations →
   effective M independent EC observations, covering horizons
   {0,1,2,3}". This single table is the thing a reviewer will ask for
   first, and right now assembling it means reading three different log
   sections.
2. **`reporter.metric("uncalibrated horizon count / fraction", ...)`**
   already exists implicitly via `evaluate_quality_gates`, but isn't
   labeled as a *design* metric distinct from a *quality* metric — worth a
   second, explicitly-named entry (`"Design — horizons with zero TDR
   coverage"`) so it survives being grepped out of a batch of run reports
   for a supplementary table, independent of whatever quality-gate
   thresholds happen to be configured.
3. **A machine-readable summary sidecar** (`run_summary.json` next to the
   Markdown report) containing exactly the fields needed for the §3 sweep
   aggregator: `num_swc_positions`, `swc_depths`, `n_calibrated_horizons`,
   `n_total_horizons`, `matching_summary.n_distinct_em_points`, `K`,
   final RMSE, CV mean. `PipelineReporter` already collects every one of
   these as `self._metrics`; the only gap is that they're currently only
   serialized as Markdown prose (`| Metric | Value |` tables), which is
   fine for a human reading one report but requires re-parsing Markdown to
   aggregate across N runs. Emitting the same `self._metrics` dict as JSON
   alongside costs almost nothing and makes §3's aggregator trivial
   instead of a Markdown-scraping script.
4. **Cross-reference the depth-extrapolation framing directly in
   `finish()`**: when `quality_gate_result.reasons` includes an
   uncalibrated-horizon warning, the report could point at
   `score_against_truth.py`'s calibrated/uncalibrated contrast by name —
   turning a bare warning into "this run has an evidentiary gap; here is
   the tool that quantifies its cost" for whoever reads the report months
   later during paper revisions.

None of these require touching the inversion, calibration, or averaging
logic — they're additive logging/aggregation, safe to add without risking
a result change to anything already validated.

---

## 7. Skeleton language for the manuscript's Methods/Discussion

> *"EM surveys provide spatially exhaustive but temporally sparse coverage
> (n = {N_EPOCHS} campaigns over {SPAN}), while TDR/SWC stations provide
> continuous but spatially limited coverage (n = {N_STATIONS} stations at
> {DEPTHS} cm). We quantify the calibration cost of this asymmetry using a
> synthetic twin of the field deployment with known ground truth at every
> depth and location (Methods §X), scoring the Waxman–Smits-calibrated EM
> inversion separately at TDR-calibrated vs. uncalibrated horizons
> (Fig. X). [At the deployed density,] uncalibrated horizons show
> {ΔRMSE} higher error than calibrated ones. A sensitivity sweep over
> station count ({N sweep}) and depth allocation (Fig. Y) indicates
> [RMSE plateaus beyond N≈__ stations / one station per horizon is
> necessary and sufficient / etc. — fill in once §3's sweep is run]."*

Fill in the bracketed values from the §3 sweep output once the forward-
model fix in §5 is applied.
