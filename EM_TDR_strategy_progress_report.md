- Can we improve soil mositure estimates at depth just using only superficial TDR / simulate very conductive first layer/very resistive first layer??
- Use soil moisture from EO or cosmic-ray??

# Ensemble-Averaged Electromagnetic Induction Inversion Calibrated with Sparse TDR Data for Soil Water Content Mapping: Methodological Progress Report

*Working manuscript, structured as a peer-review article. Status: methods implemented, verification partially complete, quantitative results pending. Statements about performance are explicitly marked as not yet supported.*

---

## Abstract

We describe a software pipeline that maps volumetric soil water content (θ) from multi-frequency frequency-domain electromagnetic induction (FDEM, GEM-2) surveys, using sparse in-situ time-domain reflectometry (TDR, SoilNet) probes as the only source of petrophysical calibration. The strategy adapts the ensemble-averaging approach of Loiseau et al. (2026): (i) a Cartesian-product ensemble of EMagPy inversions is generated; (ii) each member is converted to θ through a per-horizon Waxman–Smits (WS) relation calibrated against TDR; (iii) members are ranked by calibration RMSE and cumulatively averaged with RMSE-normalised weights; (iv) the stack depth K is chosen by recalibrating each averaged model; and (v) a coefficient-of-variation (CV) map and coverage map quantify reliability. Beyond the baseline method, the present work adds a kernel-localised (Gaspari–Cohn) spatially varying WS calibration with leave-one-probe-out bandwidth selection, a McNeill depth-sensitivity diagnostic for probe-to-layer routing, a train/validation split, a spatial reference-grid for time-lapse epochs with differing footprints, CRS and unit sanity checks, and a synthetic "twin" of the real deployment with known truth. The implementation is functionally complete; **no quantitative accuracy results are reported here**, and several assumptions (listed in §6) must be tested before the method can be claimed to generalise.

---

## 1. Introduction

### 1.1 Problem
FDEM surveys cover large areas rapidly but measure apparent conductivity (ECa), not water content. TDR probes measure θ accurately but at a handful of points and depths. The scientific problem is to transfer the point-scale TDR calibration to the survey scale, with defensible uncertainty, when:

1. the inversion of ECa to layered bulk conductivity (EC_b) is non-unique and sensitive to regularisation and starting models;
2. TDR coverage per horizon is very small (often two to a few tens of observations), so a four-parameter petrophysical model is weakly identifiable;
3. real time-lapse surveys have different footprints and point counts at each epoch;
4. EM footprint is depth- and geometry-dependent, whereas TDR is a point measurement.

### 1.2 Contribution of this work
The pipeline treats inversion non-uniqueness as an ensemble to be exploited rather than a single model to be tuned, uses TDR twice (to rank members and to select the stack depth), and surfaces data-quality and coverage problems as explicit gates rather than silent degradations.

### 1.3 Study configuration
- **Site:** MFC2 catchment (real); UTM coordinates; seven EM epochs (Aug 2020 – Dec 2021) with strongly unequal point counts, including one degraded epoch (15/12/2020, ~0.5% of the points of the largest survey).
- **EM data:** GEM-2, six frequencies (425 – 92 775 Hz), HCP geometry at 1.66 m, quadrature in ppm.
- **TDR data:** SoilNet loggers, 15 cm and 30 cm SWC, 24 stations (units parsed from `MTX##` file names), time-matched to EM dates within a configurable tolerance (default ±1 d).
- **Layer model:** either one interface at 0.225 m (2 layers; matches TDR depths) or three interfaces at 0.3/0.6/1.0 m (4 layers).

---

## 2. Methods

### 2.1 Data ingestion and pre-processing (Step 0)
- EM files are discovered by `DDMMYY` date patterns, delimiter-sniffed, and column aliases are mapped to EMagPy canonical names.
- **Coordinate handling:** each alias is classified as metric or geographic. Geographic coordinates are reprojected to UTM (zone auto-derived) so that EM and TDR share a CRS. A defence-in-depth span/centroid check compares EM and probe extents and warns on an order-of-magnitude mismatch.
- **Unit handling:** `em_unit` is mandatory for real-data scenarios (GEM-2 = `ppm`); a median-|ECa| plausibility check warns on a ~1000× discrepancy.
- **Raw-ECa cleaning, once per file at load time:** drop designated coils → `filterRange` → `filterRepeated` → `filterPercentile` → `rollingMean`. Coil dropping precedes range filtering so that a channel that will be excluded cannot veto otherwise valid points.
- **TDR:** station timestamps are assembled from six columns; values >1 are interpreted as percent and divided by 100; observations are averaged within the tolerance window around each EM date.
- **Temperature correction:** σ_corr = σ·[1+α(T−T_ref)], α = 0.023 °C⁻¹, T_ref = 20 °C. Temperature comes from a scenario override or a TDR "temp" column; if neither exists the correction is explicitly skipped (previously it was a silent no-op).

### 2.2 Ensemble inversion (Step 1)
- Members form a Cartesian product over inversion hyper-parameters (α, β, γ, optimiser, forward model: FSlin, CS) and per-layer starting conductivities (min/max bounds per layer).
- Solver bounds are built explicitly from `cond_ranges`; without this, EMagPy silently clamps conductivities to 2–100 mS/m when any depth is free.
- Interfaces may be fixed or free. With free interfaces, TDR probe-to-horizon routing uses the ensemble-median **fitted** depths.
- ROPE is the optimiser used in the real-data scenarios because it supports smoothing parameters; MCMC-family posterior standard deviations (`k.pstds`) are captured, cached and summarised.
- Inversions run in a spawn-context process pool (avoids fork-inherited state hangs).
- **Time-lapse assembly:** epochs are inverted independently and placed on a **union spatial reference grid** built by coordinate matching within a tolerance (`spatial_match_tolerance_m`), replacing the earlier row-count truncation that paired unrelated locations. Unmatched epoch/point cells are NaN and all downstream reductions are NaN-aware.
- The ensemble cache is used for real data only and is keyed to a configuration fingerprint (depths, fixed flags, grid, bounds, γ, dropped coils, tolerance).

### 2.3 Petrophysical calibration (Step 2)
Waxman–Smits in the form used throughout the code base:

θ = (a · σ_b / (σ_w + σ_s))^(1/m)

- Calibrated **separately per horizon and per ensemble member**. Observations are routed to horizons by probe depth; a horizon without probes is flagged uncalibrated and excluded from ranking (no borrowing of unrelated observations).
- Objective: sum of squared log₁₀ differences between observed EC_b and WS-predicted EC (replacing a relative-L1 form that was singular near 1 mS/m), plus a Gaussian prior penalty (λ = 0.1) centred on `lab_p0`. Optional inverse-variance weighting by TDR noise, renormalised to mean 1.
- Parameters may be pinned via `ws_fixed_params`. The real-data scenarios fix *a* and *m* and let σ_w and σ_s float, converting an underdetermined 4-parameter fit into a 2-parameter one.
- NaN observations from the reference-grid remap are dropped per horizon.

### 2.4 Kernel-localised calibration (diagnostic sibling of Step 2)
- A separate WS parameter set is fitted at every EM target location, with observations weighted by a Gaspari–Cohn kernel of 1-D or 2-D Euclidean distance (support 2c). c → ∞ recovers the global fit; c → 0 approaches nearest-neighbour.
- Targets without any observation in support revert to the prior and are flagged.
- **Bandwidth selection:** leave-one-probe-out cross-validation over candidate c values, pooled RMSE, run once on the rank-1 member. In-sample RMSE is deliberately not used because it rewards memorisation at small c.
- Outputs: σ(x) profiles, XZ and XY parameter maps, calibration-density maps (hard window and kernel-weighted), and a global-vs-local overlay on the EC–θ cross-plot.
- This branch does **not** feed Steps 3–5.

### 2.5 Ranking and cumulative averaging (Step 3)
Members are ranked by RMSE. For each r = 1…N, normalised weights (Loiseau Eq. 6–7) are recomputed on the first r members and the weighted mean ρ_r is formed. NaN cells propagate only where every member is NaN.

### 2.6 Recalibration, K selection and uncertainty (Steps 4–5)
- Each ρ_r is recalibrated with warm-start continuation along K to suppress local-minimum jitter in an underdetermined fit.
- K is chosen as the argmin of RMSE(θ_pred at probes vs TDR). Prediction uses the EC of **the probe's own horizon** only, routed through that horizon's parameters (a previous version averaged across layers and used horizon-0 parameters for all probes).
- CV map: weighted standard deviation / weighted mean across the K constituent members (Eq. 8); NaN is preserved rather than reported as zero uncertainty. A coverage map in [−1, 1] marks cells where fewer than half of the K members give a physically plausible EC.

### 2.7 Held-out validation (Step 5b)
Observations may be split into training (drives Steps 2–5) and validation subsets (touched once after K is chosen) by `alternate_sorted`, `by_probe`, `by_time_block` or `random`. Validation RMSE is computed at every K with parameters pinned at their training values, giving train-vs-validation RMSE–K curves and parity plots. `by_probe` and `by_time_block` are independent; `alternate_sorted` and `random` are not.

### 2.8 EM footprint diagnostic (Step 5c, optional)
McNeill (1980) cumulative-response kernels give closed-form layer weights per coil geometry (exact for the single real HCP/1.66 m geometry; an explicitly labelled equal-weighted average over six geometries for synthetic scenarios). The "effective" depth-weighted θ is compared with the hard-routed layer θ against TDR. This is a diagnostic only and does not alter calibration.

### 2.9 Quality gates
- *Pre-inversion:* TDR depths must reach every horizon implied by `depths0` (validated at scenario load and again after data loading); EM/TDR CRS consistency; em_unit plausibility; probe-to-EM matching collapse (largest-group fraction, window-fallback count) with warn/error thresholds.
- *Post-selection:* mean CV, fraction of cells above CV = 1.0, and calibrated-horizon count are converted into an OK/WARN verdict propagated to the run report and run log. A WARN never aborts a run.

### 2.10 Synthetic twin and scoring
A twin of the real deployment (same epoch dates and relative point-count asymmetry, same 24 station coordinates, 15/30 cm depths) is generated with a smooth deterministic spatial field × seasonal cosine × depth trend and constant WS truth parameters, forward-modelled with EMagPy and converted back to quadrature ppm through `eca2Q`. A separate synthetic generator supports linear, full-range, random-patch, infiltration and spatially varying WS-parameter truths. `run_and_score.py` and `score_against_truth.py` compute RMSE, bias, R² and mean CV per epoch × horizon, and contrast calibrated against uncalibrated horizons.

### 2.11 Reproducibility infrastructure
Each run writes to a numbered folder, a Markdown report (configuration, per-step tables, metrics, timings, failure traceback), a `run_id.json` marker, and one row in a shared run-log CSV. `compare_scenarios.py` assembles multi-run tables and truth-comparison figures; shell drivers encode the planned scenario matrix (14 real-data scenarios = 2 layer geometries × 7 variants: both/FSlin/CS forward models, free depths, all-fixed / free-σ_w / free-σ_s WS parameters; β, γ and train/validation split swept from the CLI).

---

## 3. Results

**No quantitative results are available in the material reviewed.** The only results file supplied (`scoring.csv`) contains the schema (epoch, date, horizon, depth range, calibrated flag, RMSE, bias, R², n, mean CV) for 28 rows, but its numerical content was not available to this summary and nothing below should be read as a finding. The interpretation notes provided so far identify the diagnostics to be examined: reference-grid coverage per epoch; kernel-weighted calibration density; the RMSE-vs-K curve and selected K; the bandwidth sweep; the WS-parameter and RMSE maps; and per-sensor parity of predicted vs observed TDR.

### 3.1 Evidence that exists (verification, not accuracy)
- A standalone test script covers the reference-grid union, NaN-fill remap, NaN-aware RMSE/CV, NaN-observation dropping in calibration, NaN propagation in averaging, and scenario-schema validation of the new tolerance field.
- Numerous defects were identified and fixed during development (see §5). Each corresponds to a failure mode that would have biased results silently.

### 3.2 Results that must be produced before submission
| Question | Required output |
|---|---|
| Does ensemble averaging beat the single best member? | RMSE vs K, with train and validation curves |
| Does K generalise? | Validation RMSE at selected K, independent split only |
| Value of localisation? | LOO-CV RMSE vs c; global vs localised validation RMSE |
| Cost of uncalibrated horizons? | Calibrated vs uncalibrated horizon RMSE/bias on the twin |
| Sensitivity to design | Scenario matrix (forward model, free depths, WS-fixed sets, β, γ) |
| Does the CV map predict error? | Rank correlation of CV vs |θ_calc − θ_true| on the twin |
| Does depth weighting help? | Hard-routed vs McNeill-weighted RMSE (Step 5c) |

---

## 4. Discussion

### 4.1 Strengths
1. **Inversion non-uniqueness is used, not hidden.** The ensemble plus TDR-based ranking provides a data-driven rather than subjective regularisation choice.
2. **Identifiability is managed explicitly:** parameter pinning, priors, noise weighting, warm-start continuation, and a LOO-CV bandwidth criterion.
3. **Failure modes are loud.** Unit, CRS, coverage, matching-collapse, unmatched-epoch and calibration-gap conditions each produce explicit warnings or gates.
4. **A truth-bearing twin** allows separating sampling/coverage error from methodological error, which real data cannot.

### 4.2 Limitations and assumptions requiring scrutiny
1. **Single static petrophysical model per horizon** (or per target with kernel localisation); no hysteresis, texture change, or seasonal pore-water conductivity change. σ_w is treated as constant in time, which is questionable over two winters and two summers.
2. **TDR as a point sample at 15/30 cm** assigned to horizons 0 and 1. Real probes often integrate over the rod length. The twin's own source code flags this as unverified.
3. **Hard routing of each probe to one layer** versus an EM response spanning several layers. Step 5c is the test; its caveat is that the McNeill kernel assumes low induction number, which degrades at the highest GEM-2 frequencies and for the CS forward model.
4. **In-sample selection.** With the split off, the same observations fit the WS parameters, rank members and select K. Reported RMSE is then optimistic. Validation using `alternate_sorted` is not independent.
5. **Very small effective n.** Many stations may match one EM point; the matching-collapse gate warns but does not fix this. Kernel localisation with few probes risks fitting noise despite the prior.
6. **Spatial-match tolerance** (GPS-scale) and **window_m** (EM–TDR association) answer different questions and interact with the localisation bandwidth; their joint sensitivity has not been quantified.
7. **Temperature** is a first-order correction with a sign that has been derived algebraically, not validated against instrument behaviour.
8. **Synthetic truth shares the model family with the calibration** (same WS functional form), so recovery on the twin is a best-case test of the pipeline, not of WS validity.
9. **CV measures ensemble spread, not error.** It is not a calibrated uncertainty and must be validated against truth before being interpreted as such.
10. **Time-lapse treatment** is independent per epoch; temporal regularisation (γ) is only meaningful in the joint-inversion mode, which the real-data scenarios do not use.

### 4.3 Open code-level items noticed
- `plot_em_diagnostics_real._plot_eca_emagpy_show` still calls `filterRepeated(index=0)`; the same invalid keyword previously caused every inversion task to fail elsewhere. It is caught by try/except, but the diagnostic will not render.
- `plot_emagpy_diagnostics` re-inverts the full survey for each sampled realisation and time step; it has been capped at one realisation because of runtime.
- ~~`run_pipeline.py` had an import fallback chain (`ldata_real`, `_data_real`) suggesting former file naming.~~ Fixed on `refactor/readability`: modules are renamed to importable names (`data_real`, `step2_petro`, ...) and imported plainly.
- `generate_synthetic_real_mirror.py` defaults to 1% point counts and a 150-point cap; conclusions about density effects require running at larger scale.
- `train_val_split(by_probe)` falls back to a non-independent split in synthetic runs; the scenario loader now rejects that combination at load time.
- The WS truth in the twin follows the multiplicative form used by the calibration; an earlier additive-σ_s form was corrected and should be recorded as a past inconsistency.

---

## 5. Record of methodological corrections (audit trail)

| Area | Defect | Consequence if uncorrected | Correction |
|---|---|---|---|
| Inversion bounds | EMagPy default 2–100 mS/m with free depths | Silent EC cap | Explicit bounds from `cond_ranges` |
| Preprocessing | `filterRepeated(index=0)` invalid keyword | Every inversion task failed | Filtering moved to load time with correct call |
| Time-lapse alignment | Row-count truncation (`arr[:min_pts]`) | Unrelated locations paired | Coordinate-based union reference grid + NaN fill |
| CRS | Geographic EM coords vs metric TDR | All probes fell back to nearest point, meaninglessly | Alias classification, reprojection, consistency check |
| Units | `em_unit` defaulted to `ppt` for GEM-2 data | ~1000× inflated ECa | Mandatory for real data; plausibility check |
| Temperature | Hard-coded T = T_ref | Correction always 1 | Real source or explicit skip |
| Misfit | Relative-L1 log misfit divided by log₁₀ σ | Singular near 1 mS/m | Squared log₁₀ residuals |
| K selection | Layer-averaged EC with horizon-0 parameters | RMSE–K curve decorrelated from per-horizon fit | Per-probe own-horizon EC and parameters |
| Averaging | Plain mean over members | Ignored fit quality | RMSE-normalised weights, recomputed per r |
| CV | NaN mean reported as CV = 0 | False "zero uncertainty" | NaN preserved |
| Plot routing | Independent re-implementations of horizon routing, NaN drop missing in one | Misaligned scatter/curves | Shared `geometry_utils` routing with `kept_indices` |
| Probe frames | Local and UTM probe coordinates conflated | Mislocated markers, collapsed grid | Separate local and UTM probe arrays |
| TDR noise | Forced to 0 for real data | Weighting silently disabled | Scenario value passes through |
| Caching | Synthetic cache reused across modes | Stale ensemble | Cache restricted to real data; fingerprint check |
| Pool shutdown | Fork-inherited matplotlib/BLAS state | Hang after 100% | Headless backend and spawn context |

---

## 6. Recommended validation programme

1. **Baseline on the twin** at increasing point density (scale 0.01 → 1.0) with the real probe layout; report calibrated vs uncalibrated horizon RMSE and bias.
2. **Probe-density experiment:** vary the number of stations, hold depth coverage fixed, and plot RMSE against effective probes per horizon.
3. **Independent validation only:** use `by_probe` (real) and `by_time_block`; treat `alternate_sorted` results as indicative.
4. **Misspecification test on the twin:** generate truth with spatially varying WS parameters (`ws_xy_variation`) and compare global and localised calibrations against the true parameter maps.
5. **Depth-integration test:** regenerate TDR as a 0–30 cm average rather than a point value and quantify the bias introduced under the current routing.
6. **Uncertainty calibration:** relate CV and MCMC posterior standard deviation to actual error in binned form.
7. **Ablations** across the scenario matrix: forward model, free vs fixed depths, WS fixed sets, β, γ, dropped low-frequency channels.
8. **Sensitivity of the bandwidth** to `window_m` and `spatial_match_tolerance_m`.
9. **Temperature sign check** using epoch pairs at contrasting soil temperature.

---

## 7. Conclusions (provisional)

The pipeline now implements an end-to-end, testable workflow for converting sparse-TDR-calibrated ensemble EM inversions into θ maps with explicit coverage and uncertainty fields, together with the tooling (synthetic twin, scoring, scenario comparison, run logging) required to evaluate it. The principal methodological risks are the weak identifiability of per-horizon WS parameters, the point-sample and single-layer treatment of TDR, and the potential for optimistic, in-sample selection. Until the validation programme in §6 is completed, the method should be described as *implemented and verified for internal consistency*, not as *demonstrated to be accurate*.

---

## 8. Reviewer-style checklist

**Major concerns**
1. No out-of-sample accuracy is reported. Provide independent validation RMSE and bias for the real data, and truth-based scores on the twin.
2. The claim that the localised calibration improves over the global one is untested. Show LOO-CV and held-out improvement with confidence intervals.
3. The CV map is presented as a reliability measure; demonstrate its relationship to error.
4. The point-sample TDR assumption and single-layer routing need an explicit sensitivity analysis.

**Minor concerns**
1. Justify the prior widths (0.5, 0.5, 150, 30) and λ = 0.1, which were chosen when the misfit changed form.
2. State the number of effective independent probes per horizon in the real data.
3. Report ensemble size per scenario and the fraction of failed inversion tasks.
4. Clarify why the localisation diagnostic is excluded from K selection and the final map.
5. Document the numerical effect of dropping individual GEM-2 frequencies.

**Reproducibility**
Configuration, tolerances, and scenario files are version-controllable and logged per run; the remaining gap is archiving the real-data inputs and the run-log CSV with the manuscript.

---

## References (as cited in the code base)
- Loiseau et al. (2026): ensemble-averaged inversion; Eq. 1 (temperature), 3 (misfit), 5–8 (weights, CV).
- Waxman & Smits (1968): petrophysical model (simplified multiplicative form used here).
- McNeill (1980): cumulative EM response of layered ground.
- Gaspari & Cohn (1999): compact-support localisation.
- Benoit et al. (2019); Dimech et al. (2023); Calamita et al. (2012): spatial integration window between EM and point sensors.
- Travelletti et al. (2012): temperature coefficient.
- McLachlan et al., EMagPy: inversion engine and forward models.
