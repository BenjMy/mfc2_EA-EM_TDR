# Design doc: folding the McNeill depth-weighted (EM-footprint) comparison into calibration / K-selection

**Status: design only — nothing in this doc has been implemented.**
Scope: how to move the Step 5c "diagnostic sibling" comparison from a
post-hoc report into something that actually changes Step 2/4 calibration
and/or Step 3/5 K-selection, and what breaks/needs a decision along the way.

---

## 0. The problem: EM's volumetric footprint vs. TDR's point measurement

This whole diagnostic exists because **EM and TDR don't measure the same
thing, physically.**

- A **TDR/SWC probe is a point sensor**: it reports volumetric water
  content at one specific depth, over a small sensing volume around its
  rods/waveguide (centimeters). It has no ambiguity about which horizon it
  "belongs to" — it's installed at a known depth.
- An **EM coil reading is not a point, and not a single depth.** The
  secondary field the coil measures is a **continuous, depth-integrated
  response** — every depth beneath the coil contributes to the reading,
  weighted by a sensitivity function that decays with depth (McNeill,
  1980), and that function's shape depends on coil geometry (spacing,
  orientation) and, in the LIN limit, is independent of the true
  conductivity profile. In other words: one EM reading is *simultaneously*
  informed by all four horizons at once, in different proportions — never
  purely "the horizon the probe happens to sit in."

So every time this pipeline compares a TDR observation to "the" EC/θ at an
EM point, it is implicitly answering the question *"which part of the
coil's blended, depth-integrated response should I compare this point
observation against?"* — and that question has a real, physically
motivated answer (the McNeill weighting), which the pipeline currently
does **not** use anywhere except in this one opt-in diagnostic.

### What the pipeline actually does about this mismatch today: `window_m` (horizontal only)

The only footprint-matching mechanism wired into calibration/K-selection is
`window_m` — and it only resolves the **horizontal (x/y)** side of the
mismatch, not the **vertical (depth)** side:

> `window_m` : half-width of the spatial averaging window used to match TDR
> probes to EM measurement indices (`run_pipeline.py` docstring).

Concretely: `window_m` decides *which lateral EM survey point(s)* a TDR
probe gets matched/averaged against (nearest EM index within `±window_m`,
with a documented nearest-point fallback — see the Step 1a "Window-fallback
(nearest-point) matches" diagnostic). It says nothing about *depth*. Once a
probe is matched to an EM point, **every downstream step treats that EM
point's response as if it belonged entirely to the probe's own installation
horizon** — `probe_layer_indices` performs a hard, all-or-nothing depth
routing: 100% of the weight to the probe's horizon, 0% to every other
horizon, even though the actual EM measurement at that point was never
"about" only that horizon.

That hard depth-routing is exactly the McNeill-weighted diagnostic
addresses (§1 below) — except only as a report-time comparison. It has no
counterpart to `window_m`'s role: there is currently no equivalent
"depth-window" concept that calibration or K-selection actually uses.

### Why this specifically shows up as a discrepancy — corrected

**Correction:** an earlier version of this doc claimed calibration fits
each horizon against "the coil's raw blended reading," implying `models[...,
layer]` is itself a McNeill-weighted mix that gets wrongly treated as pure.
That's not accurate, and it's worth being precise about why.

`models[ensemble, position, layer]` is **not** the raw apparent
conductivity (ECa) the coil measured — it is the output of Step 1's
inversion (EMagPy/ROPE, `forwardModel ∈ {FSlin, CS}`), which already takes
the coil's depth-integrated ECa response and **unmixes it into a per-layer
EC estimate**, at the actual, correct depth of each horizon. So the claim
"calibration compares a blended reading against a single horizon" is wrong
by construction: the inversion's entire job is to remove that blending
*before* calibration ever sees the data. Calibration genuinely does operate
on per-horizon, per-depth EC values — the depth is the actual depth.

**The real issue is one level upstream: not what depth is being compared,
but how well-determined each layer's inverted EC value is.** The inversion
that produces `models[..., layer]` has to solve for `n_layers` unknowns
(4 here) from a handful of coil readings whose *sensitivity to each layer*
is exactly `EM_LAYER_WEIGHTS` — the same McNeill kernel. A layer the coil
is highly sensitive to (this run: horizon 0 at 35.3%, horizon 3 at 33.0%)
is well-constrained by the raw ECa data itself. A layer the coil is barely
sensitive to (horizon 2 at 13.1%) is inverted mostly by the regularization/
prior structure of the inversion (α, β in `param_grid`), not by independent
signal in the ECa — the same non-uniqueness problem every layered EM/ERT
inversion has for its least-sensitive layers. **That's** the physical
mechanism behind the discrepancy: low-`EM_LAYER_WEIGHTS` layers get a
*depth-correct but poorly-constrained* EC estimate, not a *blended* one.

This pipeline already tracks a version of this — the MCMC posterior std
per point/layer (`k.pstds`, logged in Step 1a: *"MCMC posterior std
(conductivity, mS/m, cells with std>0 only, n=11520): mean=0.820
median=0.135 p90=2.375 max=15.929"*) and the coverage map (Step 5,
`EMmethod.compute_coverage_map`) are the pipeline's existing signals of
per-cell inversion confidence. Neither is currently cross-checked against
`EM_LAYER_WEIGHTS`, but they're measuring a closely related thing:
how much the ensemble members agree on a cell's EC (posterior
std/coverage) should broadly correlate with how sensitive the coil actually
is to that layer (`EM_LAYER_WEIGHTS`) — a poorly-constrained, low-weight
layer is exactly where you'd expect ensemble disagreement to be highest.

Two consequences follow, restated correctly:

1. **Calibration risk (revised).** Fitting `(a, m, σ_w, σ_s)` per horizon
   treats every horizon's inverted EC as equally trustworthy input, even
   though horizons the coil is less sensitive to (`EM_LAYER_WEIGHTS`
   small) have systematically noisier/less-identified EC estimates behind
   them. The WS fit doesn't currently down-weight or flag horizons on that
   basis — it has no visibility into `EM_LAYER_WEIGHTS` at all.
2. **K-selection risk (revised).** `4_run_select_assess.py`'s `rmse_by_k`
   scores every stack depth `K` using each probe's own horizon's inverted
   EC, uniformly, regardless of how well- or poorly-constrained that
   horizon's inversion was for this coil array. `K` could look "optimal"
   partly because it happens to fit well on the well-constrained horizons,
   while the poorly-constrained ones (which the McNeill weights would flag
   in advance) contribute noise to the same average.

This reframes what the "possible improvement" below should actually be:
not "replace hard-routing with a McNeill-weighted blend of already-inverted
θ" (Option A in §3, which — per this correction — blends values that are
already at the right depth, just of uneven reliability), but rather
**using `EM_LAYER_WEIGHTS` as a per-horizon confidence/precision weight in
calibration and K-selection**, analogous to how `theta_obs_std` already
down-weights noisy TDR readings (§ discussion below) — i.e. weighting each
horizon's contribution to the misfit by how sensitive the coil actually was
to it, not blending its value with its neighbors'.

### The possible improvement — and the chosen direction

Extend the existing McNeill kernel (already implemented and used for the
diagnostic in `em_sensitivity.py`) from a **report-time comparison** into
an actual **depth-weighting scheme used by calibration and/or
K-selection** — i.e. give the pipeline a genuine "depth footprint" concept
to sit alongside `window_m`'s existing "horizontal footprint" concept,
instead of hard-routing every observation to a single horizon once its
lateral match is resolved.

**Decision: we will implement Option B — blend in EC-space** (§3 below
describes both options in full; Option A is kept in this document as the
rejected alternative, not removed, since it's still useful context for why
B was chosen). Blending the layers' EC *before* inversion, weighted by
`EM_LAYER_WEIGHTS`, is the option that's actually closer to what the coil
physically does (§3), and — per the correction above — it also happens to
double as a natural precision weighting: a layer the coil is barely
sensitive to (small `EM_LAYER_WEIGHTS`, poorly-constrained inversion)
automatically contributes proportionally less to `EC_eff`, rather than
being blended in at face value or ignored outright. That lines up with the
corrected mechanism in the previous section — it's not a coincidence, since
both the blending weight and the inversion's confidence in that layer trace
back to the same underlying coil sensitivity.

Choosing B does **not** remove the open design question flagged in §3 —
whose `(a, m, σ_w, σ_s)` to invert `EC_eff` with — that still needs to be
resolved explicitly (§3, §4 below) before any code is written.

## 1. How the current diagnostic table is built (baseline, for reference)

`run_pipeline.py`, Step 5c (`if em_footprint_diagnostic:`), using
`selected_output` already returned by Step 5:

```python
_footprint_theta_calc = selected_output.theta_calc          # (n_time, n_points, n_layers)
_eff_theta = emsens.effective_theta(_footprint_theta_calc, EM_LAYER_WEIGHTS)

_fp_hard_routed = _footprint_theta_calc[t_idx, pos, probe_layer_idx]   # single layer
_fp_weighted    = _eff_theta[t_idx, pos]                               # blend across layers
```

- **`theta_calc`** comes from `4_run_select_assess.py::run_select_assess`,
  specifically `_ec_grid_to_theta(avg_model_opt, best_fit_opt)`: the
  RMSE-weighted, K-stacked EC grid (`avg_model_opt`), inverted to θ **per
  layer independently** using that layer's own fitted `(a, m, σ_w, σ_s)`
  (`best_fit_opt.horizon_fits[l]`):
  ```python
  ratio = hf.a * ec_col / (hf.sigma_w + hf.sigma_s)
  theta = ratio ** (1 / hf.m)
  ```
- **`EM_LAYER_WEIGHTS`** comes from `em_sensitivity.py`
  (`layer_weights` / `layer_weights_multi_coil`), a closed-form McNeill
  (1980) cumulative-response kernel — physically grounded in coil geometry,
  valid in the low-induction-number (LIN) limit.
- **Hard-routed** = one layer's independently-inverted θ, picked by
  `probe_layer_indices` (which horizon the probe was actually installed
  in).
- **Weighted** = a **theta-space post-hoc blend** of all four layers'
  independently-inverted θ values at that position/time, using
  `EM_LAYER_WEIGHTS`. It never touches EC or re-inverts anything.

This is why it's cheap and safe to compute after the fact: it reuses
`theta_calc` and adds one `tensordot`. It changes nothing upstream.

## 2. Where "hard-routing" actually lives today (the parts that would need to change)

There are **three separate places** in the pipeline that independently
hard-route each TDR observation to a single horizon and would all need to
agree on a new routing/weighting scheme for feedback to be consistent:

| Location | File | What it does | Feeds |
|---|---|---|---|
| Step 2 calibration | `em_methodology.calibrate_petrophysical_relationship` | Fits `[a, m, σ_w, σ_s]` per horizon, per ensemble member, using only the TDR obs whose `probe_layer_indices` match that horizon | `fits_step2` — ranks ensemble members (Step 3) |
| Step 4 recalibration | `em_methodology.recalibrate_petrophysical_relationship` | Same fit, re-run on each cumulative stack-depth `K`'s averaged EC | `fits_step4` |
| Step 4/5 K-selection RMSE | `4_run_select_assess.run_select_assess`, main loop (lines ~107–177) | For each `K`, pulls `ec_predicted[i] = avg_model_k[t, pos, probe_layer_idx]` (**one layer only**), converts via `EMmethod.ec_to_water_content(..., probe_layer_indices=...)`, computes `rmse_by_k[k] = RMSE(theta_obs, theta_pred)` | `optimal_K = argmin(rmse_by_k)` — this **is** the K-selection objective |

**Key point:** Step 4/5's `rmse_by_k` — the thing that literally picks `K`
— is built from the exact same "one probe → one horizon → one EC value →
`ec_to_water_content`" hard-routing as the calibration step. The Step 5c
diagnostic's "weighted" comparison currently has **no equivalent** at
either the calibration objective or the K-selection objective — it's
computed once, after `K` is already fixed and `theta_calc` already exists.

To make the diagnostic "feed back," at least one, and for consistency
probably both, of these RMSE computations needs to be replaced with a
version that uses `EM_LAYER_WEIGHTS`.

## 3. The physical decision that has to be made first (this is the actual hard part)

Per `spatial_calibration_alternatives.md` and `em_sensitivity.py`'s own
caveats, there are two structurally different ways to "use" the McNeill
weights, and they are **not interchangeable**. Both are described in full
below — Option A is kept for context even though **Option B is the chosen
direction** (see decision box after both descriptions).

### Option A — Blend in θ-space (what the diagnostic already does) — *not chosen*
Invert each layer's EC independently (current behavior), then blend the
resulting θ values:
```
θ_eff = Σ_layer  w_layer · θ_layer(EC_layer; a_layer, m_layer, σw_layer, σs_layer)
```
- Cheap, reuses the existing per-horizon calibration unchanged.
- **Physically questionable**: the coil doesn't measure a blend of four
  independently-true θ values: it measures a *single* apparent
  conductivity that is itself a depth-weighted integral of the *true*
  bulk EC profile. Blending already-inverted θ is a reporting-time
  approximation, not a re-derivation of what the instrument actually saw.
- Per §0's correction: since each layer's inverted EC is already at the
  correct depth, Option A blends values that don't need to be blended for
  depth-correctness — it just averages out inversion noise/uncertainty
  across horizons, which is a much weaker justification than it first
  appears.

### Option B — Blend in EC-space (physically closer to what the coil does) — **chosen**
Blend the layers' EC first, then invert once:
```
EC_eff = Σ_layer  w_layer · EC_layer
θ_eff  = WS⁻¹(EC_eff; a_eff, m_eff, σw_eff, σs_eff)
```
- Closer to the actual measurement physics (the coil's secondary field
  response genuinely is a weighted integral over depth of the true EC(z)).
- Per §0's correction: this also naturally implements confidence weighting
  — a poorly-constrained layer (low `EM_LAYER_WEIGHTS`) contributes
  proportionally less EC to the blend, rather than being averaged in at
  full strength as in Option A's theta-space blend.
- **Introduces a new question, still open**: whose `(a, m, σ_w, σ_s)` do
  you use to invert the blended `EC_eff`? A single set of parameters can't
  be "the blend of four horizons' parameters" in any principled way unless
  the horizons share the same rock-physics behavior — which is exactly the
  cross-horizon heterogeneity the pipeline is trying to resolve in the
  first place. Candidate resolutions (none implemented yet, see §4):
  1. **Weighted-blend the parameters too**, using the same
     `EM_LAYER_WEIGHTS`: `a_eff = Σ w_layer · a_layer`, and likewise for
     `m, σ_w, σ_s`. Symmetric with the EC blend, cheap, but assumes the WS
     relationship is "linear enough" in its own parameters for a weighted
     average to be meaningful — untested assumption, needs validation
     against synthetic ground truth before trusting it on real data.
  2. **One shared-across-horizons parameter set**, fit jointly rather than
     blended post-hoc — more principled but a bigger structural change (see
     §4b) and loses some of the point of per-horizon calibration.
  3. Some other combination rule — not attempted here; flagged as a
     fallback if (1) doesn't validate well.
  This document proceeds with candidate (1) as the working assumption for
  §4's wiring sketch, explicitly flagged as unvalidated.

**Decision recorded:** proceed with **Option B**, using parameter-blend
candidate (1) above as the initial implementation target, validated against
synthetic ground truth (where the true `a, m, σ_w, σ_s` are known — see the
Step 0 report table) before ever trusting it on real MFC2 data.

`em_sensitivity.py`'s own module docstring adds a second wrinkle specific
to synthetic scenarios: the six-coil `layer_weights_multi_coil` average is
explicitly labeled "not a claim about the array's true joint sensitivity"
— the real inversion already blends six coil geometries' information
nonlinearly, and averaging their footprints post-hoc is a labeled
approximation. Feeding that approximation into calibration would make an
already-approximate quantity load-bearing rather than diagnostic.

## 4. Where Option B would actually need to be wired in

### 4a. K-selection (`4_run_select_assess.py`)
Inside the `for k in range(num_models):` loop (lines ~68–175), replacing
the existing hard-routed `ec_predicted` extraction:
1. Extract **every layer's** EC at each probe's position/time (not just its
   own horizon) — `avg_model_k[t_idx, pos, :]`.
2. Blend across layers with `EM_LAYER_WEIGHTS` **in EC-space** to get
   `EC_eff[i] = Σ_layer w_layer · avg_model_k[t_idx, pos, layer]` — a new
   function alongside `emsens.effective_theta`, e.g.
   `emsens.effective_ec(ec_layers, weights)`, same `tensordot` shape, just
   applied before inversion instead of after.
3. Blend `current_fit`'s per-horizon parameters the same way (candidate
   (1) from §3): `a_eff, m_eff, σw_eff, σs_eff = Σ_layer w_layer · (that
   horizon's parameter)`.
4. Invert once: `theta_pred_weighted[i] = WS⁻¹(EC_eff[i]; a_eff, m_eff,
   σw_eff, σs_eff)` — a single call to `waxman_smits_theta`, not
   `ec_to_water_content`'s per-horizon routing.
5. Decide what `rmse_by_k[k]` now means: replace the hard-routed RMSE
   entirely, or compute both and pick `K` by a combined objective?
   **This directly changes which `K` gets selected** — the RMSE-vs-K curve
   in the report would no longer describe the same objective it does today,
   and any downstream comparison to "hard-routed RMSE" (`min_rmse`,
   `train_val` diagnostics) needs to be redefined consistently.

### 4b. Calibration (`em_methodology.calibrate_petrophysical_relationship` / `recalibrate_petrophysical_relationship`)
This is the structurally bigger change. Currently the per-horizon objective
(`_log_ratio_objective` / `_fit_ws_horizon_weighted`) fits each horizon's
`(a, m, σ_w, σ_s)` **independently**, one at a time
(`for layer in range(num_layers): ...`), using only the observations
routed to that horizon. Under Option B, the forward prediction for a given
TDR observation would depend on *all four* horizons' EC **and** all four
horizons' in-progress parameters simultaneously (via the EC-space blend and
the parameter blend from §3 candidate (1)):
```
θ_pred(obs_i) = WS⁻¹( Σ_layer w_layer · EC_layer(obs_i's position) ;
                       Σ_layer w_layer · a_layer, ..., Σ_layer w_layer · σs_layer )
```
That couples all four horizons' optimizations into **one joint 16-parameter
problem** instead of four independent 4-parameter problems. This is a real
restructuring of the optimizer loop, not a change to its inputs — and it
compounds the identifiability risk flagged in §5 (each observation's
gradient now touches every horizon's parameters, not just its own).
**Recommendation (see §6): implement 4a first and validate before
attempting 4b**, since 4b's added complexity is only worth taking on if 4a
already shows Option B outperforming hard-routing on held-out validation.

### 4c. Kernel-localized calibration (`run_petro_localized`, the bandwidth diagnostic from earlier in this conversation)
Orthogonal to this — that mechanism localizes calibration **spatially**
(nearby TDR probes vs. far ones); this diagnostic is about blending
**across depth/horizons** at a fixed spatial location. They could in
principle compose (spatially-localized AND depth-weighted), but that's a
further increase in scope, not a prerequisite.

## 5. Consistency and identifiability risks (apply to Option B; most also applied to A)

- **Objective mismatch across steps.** If only K-selection (4a) is changed
  and calibration (4b) is left hard-routed (or vice versa), the pipeline
  optimizes two different, disagreeing things at different stages — worse
  than the current diagnostic-only state, where at least the mismatch is
  documented and visible rather than silently baked into two different
  numbers.
- **Reduced identifiability.** `fixed_params`'s own docstring already flags
  that some horizons are underdetermined with ~2 obs per horizon (4
  unknowns). A cross-horizon weighted objective (4b) spreads each
  observation's influence across *multiple* horizons' parameters
  simultaneously — this could make already-marginal horizons even less
  identifiable, independent of whether the resulting aggregate RMSE looks
  better.
- **Circularity via the LIN approximation.** `em_sensitivity.py`'s own
  caveat: the McNeill kernel is only valid in the low-induction-number
  limit over a layered/half-space assumption, and is explicitly flagged as
  approximate for `forwardModel="CS"` / higher GEM-2 frequencies. Right now
  that approximation only affects a diagnostic plot. Wiring it into
  calibration or K-selection would make the *actual solution* depend on an
  approximation that the module's own docstring says shouldn't be trusted
  as a re-derivation of the inversion in those regimes.
- **Real-vs-synthetic asymmetry.** The real-data path has one exact coil
  geometry (`layer_weights`, single HCP config) — a well-posed weight
  vector. The synthetic path's six-coil average (`layer_weights_multi_coil`)
  is explicitly labeled a non-physical approximation. Any feedback wiring
  would behave on genuinely different epistemic footing in the two data
  sources; this asymmetry doesn't exist today because the diagnostic is
  opt-in and clearly labeled.
- **The parameter-blend assumption (§3 candidate (1)) is untested.** Unlike
  Option A (which reuses an already-implemented, already-run diagnostic),
  Option B's EC-space blend plus parameter-blend has never actually been
  computed anywhere in this codebase — there is no existing number to point
  to (the 0.02720 vs. 0.04999 comparison in the attached report is Option
  A's θ-space result, not Option B's). Whether Option B's EC-space blend
  outperforms hard-routing on this or any scenario is genuinely unknown
  until §6 step 2–3's validation is actually run — the risk isn't "we know
  it's worse," it's "we don't yet know either way, and shouldn't assume B
  is better just because it's more physically motivated."

## 6. Recommended path, given Option B is the chosen direction

1. **Don't wire it into calibration yet.** Start with 4a only
   (K-selection), since it's the smaller, more contained change and
   doesn't require restructuring the per-horizon optimizer into the joint
   16-parameter fit described in 4b.
2. **Add it as a second, parallel RMSE-vs-K curve first**, not a
   replacement — compute both hard-routed and EC-space-weighted `rmse_by_k`
   trajectories, plot them together, and check on more scenarios (not just
   this one, where the θ-space weighted comparison was worse — Option B's
   EC-space blend hasn't been tested yet and may behave differently) whether
   the weighted objective picks a meaningfully different `K` and whether
   that `K` generalizes better on held-out validation
   (`--train-val-split`) before ever making it the thing `argmin` uses.
3. **Validate the parameter-blend assumption (§3 candidate (1)) against
   synthetic ground truth first.** The Step 0 report table gives the true
   `(a, m, σ_w, σ_s)` per layer for synthetic scenarios — use a scenario
   with deliberately *different* per-horizon truth parameters (unlike this
   run's report, where truth is identical across all layers by
   construction, which can't actually test whether parameter-blending
   is a good approximation) to see whether `Σ w_layer · param_layer`
   recovers something usable, before trusting it on real MFC2 data where
   there's no ground truth to check against.
4. **Only attempt 4b (calibration) if 4a's results hold up** — 4b requires
   solving the joint 16-parameter optimization and compounds the
   identifiability risk (§5); it's not worth taking on until Option B is
   shown to help at the K-selection stage first.
5. **Leave `run_petro_localized`'s spatial localization and this
   depth-weighting as separate, independently-toggleable diagnostics**
   until each is independently validated — composing them before either is
   trusted alone makes failures hard to attribute.
6. **Gate any of this behind an opt-in flag**, defaulting off, exactly like
   the localized-bandwidth calibration and `--em-footprint-diagnostic` —
   never make it the default K-selection/calibration objective without
   explicit opt-in, given the caveats above.

## 7. Open questions before writing any code

- Is 4a (K-selection only) an acceptable initial scope, given 4b
  (calibration) is the structurally bigger change (joint 16-parameter
  optimization instead of four independent 4-parameter fits)?
- For the parameter-blend question in §3 (candidate (1), weighted average
  of `a, m, σ_w, σ_s` across horizons): is a synthetic scenario with
  deliberately heterogeneous per-horizon truth parameters available to
  validate this against, or does one need to be constructed first? (This
  run's synthetic truth has identical parameters across all layers by
  design, so it can't distinguish a good blend from a bad one.)
- Should this go through `--train-val-split` validation before being
  trusted as anything other than a second diagnostic curve?
- Should the new `emsens.effective_ec` function (§4a step 2) live in
  `em_sensitivity.py` alongside `effective_theta`, keeping the module as
  the single source of truth for both blending conventions?
