# Propagating EC-ensemble uncertainty to θ (volumetric water content)

## Question

`cv_ec_vs_cv_theta.md` explains why `CV_EC` (computed on the top-K
inverted-EC ensemble at Step 5) is the right *early* diagnostic, and notes
that `CV_θ ≈ CV_EC / m` if you wanted a θ-space number. This document works
out how to actually **compute** `CV_θ` — not just approximate it — for both
the flat per-horizon Waxman-Smits (WS) calibration and the kernel-localized
variant, and audits whether the localized fits currently feed into
Steps 3–5 at all.

---

## 1. Two ways to get θ-uncertainty

### 1a. Delta method (analytic, already in `cv_ec_vs_cv_theta.md`)

Since `θ = (a·σ_b / (σ_w+σ_s))^(1/m)`, first-order error propagation gives:

```
CV_θ ≈ CV_EC / m
```

Cheap, no extra computation — but it's a first-order approximation. It
assumes fixed WS parameters and breaks down for large `CV_EC` (the
power-law curvature isn't captured).

### 1b. Full nonlinear ensemble propagation (Monte Carlo through WS) — preferred

`calibrate_petrophysical_relationship` (Step 2, called from `run_petro` in
`2_run_petro.py`) already returns **one `PetroFit` per individual ensemble
member** (`len(fits) == ensemble_size`), not just per averaged model. So
`fits_step2[i]` holds `[a, m, σ_w, σ_s]` per horizon calibrated
specifically against `models[i]`.

That means for the top-K EC models already selected in
`select_and_assess_model` (Step 5):

1. Take `top_k_indices = ranking[:K]` (already computed there).
2. For each `k`, convert `models[top_k_indices[k]]` to a θ grid using
   **that member's own** `fits_step2[top_k_indices[k]]` — applying
   `waxman_smits_theta` per horizon layer across the *full spatial grid*
   (unlike `ec_to_water_content`, which only handles probe locations).
3. Reuse the same RMSE-derived weights `w_i` already computed in
   `select_and_assess_model` to get a weighted mean/std/CV, now in
   θ-space instead of EC-space.

This isn't an approximation — it's the actual propagated ensemble, and it
naturally folds in two sources of spread at once: EC-inversion
disagreement *and* per-member petrophysical fit variability (each top-K
model gets its own WS params rather than one fixed set).

```python
def theta_cv_map(top_k_models: np.ndarray,
                  top_k_fits: Sequence[PetroFit],
                  weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Propagate the top-K EC ensemble through each member's own Waxman-Smits
    fit to get a weighted mean/std/CV map in theta-space.

    top_k_models : (K, *spatial, num_layers)  EC ensemble, mS/m
    top_k_fits   : length-K sequence of PetroFit, aligned with top_k_models
    weights      : (K,) RMSE-derived weights (same as used for CV_EC)
    """
    K = len(top_k_fits)
    theta_stack = np.empty_like(top_k_models, dtype=float)

    for k, fit in enumerate(top_k_fits):
        for hz in fit.horizon_fits:
            layer = hz.horizon
            theta_stack[k, ..., layer] = waxman_smits_theta(
                top_k_models[k, ..., layer], hz.a, hz.m, hz.sigma_w, hz.sigma_s
            )

    n_extra = theta_stack.ndim - 1
    w_view = weights.reshape((K,) + (1,) * n_extra)

    cell_mean = np.sum(w_view * theta_stack, axis=0)
    if K == 1:
        cell_std = np.zeros_like(cell_mean)
    else:
        sq_dev = w_view * (theta_stack - cell_mean) ** 2
        cell_std = np.sqrt((K / (K - 1)) * np.sum(sq_dev, axis=0))

    with np.errstate(divide="ignore", invalid="ignore"):
        cv_theta = np.where(np.abs(cell_mean) > 1e-8,
                             cell_std / cell_mean,
                             np.where(np.isfinite(cell_mean), 0.0, np.nan))

    return cell_mean, cell_std, cv_theta
```

Called from `4_run_select_assess.py` right after `select_and_assess_model`,
passing `[fits_step2[i] for i in top_k_indices]` and the weights array
(currently computed inline inside `select_and_assess_model` but not
returned — would need exposing via `SelectedModel` or recomputed the same
way, since it's cheap).

**Open design question:** compute θ using each member's **own** WS fit
(above — captures petrophysical variability too), or apply the single
**selected/recalibrated** fit (`recalibration_fits[best_r_index]`)
uniformly to all K members (isolates EC-inversion spread only, closer in
spirit to `CV_EC` as a "before petrophysics" diagnostic)? Both are
legitimate; the former is the more complete end-to-end uncertainty, the
latter the cleaner "is the EM inversion struggling" number.

---

## 2. The kernel-localized variant

### 2a. What's different about the localized fit

`calibrate_petrophysical_relationship_localized` also returns one
`LocalizedPetroFit` per ensemble member (`len(fits) == ensemble_size`,
same as the global case). But each `LocalizedHorizonWaxmanSmitsParams`
carries **per-target arrays** — `a`, `m`, `sigma_w`, `sigma_s`,
`is_calibrated`, all shape `(n_targets,)` — instead of one scalar per
horizon, and `target_indices` may not cover every point in the grid if a
restricted subset was requested.

```python
def theta_cv_map_localized(top_k_models: np.ndarray,
                            top_k_localized_fits: Sequence[LocalizedPetroFit],
                            weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Localized analogue of theta_cv_map: each member's own per-target
    Waxman-Smits params are used instead of one scalar per horizon.

    Returns cell_mean, cell_std, cv_theta, calibrated_fraction
    (calibrated_fraction: weighted share of top-K members whose value at
    that cell came from a real local fit rather than the fallback prior).
    """
    K = len(top_k_localized_fits)
    theta_stack = np.full_like(top_k_models, np.nan, dtype=float)
    calib_stack = np.zeros_like(top_k_models, dtype=float)

    for k, fit in enumerate(top_k_localized_fits):
        for hz in fit.horizon_fits:
            layer = hz.horizon
            idx = hz.target_indices                      # (n_targets,)
            ec_slice = top_k_models[k, ..., layer]        # (*spatial_no_layer,)
            theta_stack[k, ..., idx, layer] = waxman_smits_theta(
                ec_slice[..., idx], hz.a, hz.m, hz.sigma_w, hz.sigma_s
            )
            calib_stack[k, ..., idx, layer] = hz.is_calibrated.astype(float)

    n_extra = theta_stack.ndim - 1
    w_view = weights.reshape((K,) + (1,) * n_extra)

    # nan-aware: a cell can be NaN for a given member if it fell outside
    # that member's target_indices
    valid = np.isfinite(theta_stack)
    w_eff = w_view * valid
    w_sum = np.sum(w_eff, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cell_mean = np.sum(w_eff * np.nan_to_num(theta_stack), axis=0) / w_sum
        calibrated_fraction = np.sum(w_eff * calib_stack, axis=0) / w_sum

    if K == 1:
        cell_std = np.zeros_like(cell_mean)
    else:
        sq_dev = w_eff * (np.nan_to_num(theta_stack) - cell_mean) ** 2
        cell_std = np.sqrt(
            w_sum / np.clip(w_sum - (w_eff ** 2).sum(axis=0) / np.clip(w_sum, 1e-15, None), 1e-15, None)
            * np.sum(sq_dev, axis=0)
        )  # weighted-std with partial coverage

    with np.errstate(divide="ignore", invalid="ignore"):
        cv_theta = np.where(np.abs(cell_mean) > 1e-8,
                             cell_std / cell_mean,
                             np.where(np.isfinite(cell_mean), 0.0, np.nan))

    return cell_mean, cell_std, cv_theta, calibrated_fraction
```

With the default `target_indices=None` (every point), `w_eff` reduces to
`w_view` everywhere and this degenerates back to the same Bessel-corrected
formula as the global version — the partial-coverage handling only
matters if `target_indices` was restricted.

### 2b. `calibrated_fraction` is a new failure mode worth surfacing

In the localized case, a cell can look well-constrained in EC
(`coverage_map` from Step 5) but have its θ conversion coming entirely
from `is_calibrated=False` targets — every one of the top-K members fell
back to the WS prior there because no probe was within `2×bandwidth`.
That's distinct from EC ambiguity, and it's silently absorbed into
`cv_theta` otherwise: a *low* CV there would look reassuring but actually
means "K members agree on an uncalibrated guess," not "K members agree on
a calibrated θ." Recommend reporting `calibrated_fraction` as a companion
map/threshold in `evaluate_quality_gates`-style logic, the same way
`coverage` sits alongside `cv_map` today.

---

## 3. Are the localized fits actually fed into Steps 3–5? — No

Confirmed by grep across `2_run_petro.py`, `3_run_average.py`,
`4_run_select_assess.py`, and `run_pipeline.py`:

- `fits_step2_localized` is computed exactly once, at
  `run_pipeline.py:3850`, and **never referenced again**.
- `step3.run_average` (Step 3), `EMmethod.recalibrate_petrophysical_relationship`
  (Step 4), and `step4_5.run_select_assess` (Step 5) all run exclusively on
  `fits_step2` → `fits_step4`, both plain `PetroFit` objects.

### Why, by design

1. **The call is wrapped in a try/except that swallows failures on
   purpose.** The comment reads: *"Diagnostic sibling only — never let it
   fail a run that would otherwise succeed."* Strong signal it was built
   as an optional, non-critical exploratory output, not a pipeline
   dependency.
2. **It's opt-in.** The call only fires when `ws_localized_bandwidth` is
   set in the scenario config, whereas Steps 3–5 always run. If it were
   load-bearing for CV/selection, it couldn't be conditionally skipped.
3. **Type mismatch all the way down.** `recalibrate_petrophysical_relationship`,
   `select_and_assess_model`, and `ec_to_water_content` all hardcode
   `PetroFit` / `HorizonWaxmanSmitsParams` (scalar-per-horizon) in their
   signatures. None accept a `LocalizedPetroFit` (per-target array). A
   parallel `recalibrate_petrophysical_relationship_localized` for Step 4
   doesn't exist yet — only the Step-2-equivalent
   (`calibrate_petrophysical_relationship_localized`) does.
4. **Its stated purpose is narrower than "improve the pipeline's θ
   estimate."** Per `run_petro_localized`'s docstring, it exists to let
   `σ_s` (etc.) vary continuously along the transect for diagnosing sparse
   TDR coverage, visualized via `plot_localized_sigma_s_profile` / `_map` /
   the overlay plot — and only for the single rank-1 member
   (`global_fit=fits_step2[0]`), not the full ensemble's top-K. Even
   though `calibrate_petrophysical_relationship_localized` *can* produce
   fits for every member, the orchestration in `run_pipeline.py` never
   asks it to do more than diagnose one member's spatial calibration
   behavior.

**Net effect:** as currently wired, the localized fits are a dead end
after Step 2 — useful for eyeballing whether a flat per-horizon `σ_s` is
hiding real spatial structure, but structurally disconnected from ranking,
recalibration, or CV mapping.

---

## 4. Recommended integration path

Minimal-diff option (recommended): leave Steps 3–5 untouched (ranking/K
still selected on global RMSE), and after Step 5 finishes:

1. Either call `run_petro_localized` restricted to
   `models[top_k_indices]`, or reuse a full-ensemble
   `fits_step2_localized` you already computed and index into it with
   `top_k_indices`.
2. Feed those K localized fits into `theta_cv_map_localized` above.

This avoids touching the validated Step 3/4/5 code path and keeps the
localized variant exactly what its docstrings say it is — a sibling, not a
replacement — just extended to feed one more diagnostic (θ-CV) instead of
only plots.

Larger-diff alternative: also re-rank using localized RMSE
(`LocalizedPetroFit.rmse`, already a simple per-member mean over targets)
so `K` itself reflects local calibration quality — bigger change, only
worth it if global-RMSE ranking is suspected of picking a different K than
localized calibration would prefer.

---

## 5. Practical takeaway

| Approach | What it captures | Cost | Status in pipeline |
|---|---|---|---|
| `CV_θ ≈ CV_EC / m` | First-order approximation, fixed WS params | Free | Not implemented, just noted in `cv_ec_vs_cv_theta.md` |
| `theta_cv_map` (global, per-member WS) | True nonlinear propagation; EC-inversion spread + per-member petrophysical variability | One extra pass over top-K models | Not wired in; drop-in after Step 5 |
| `theta_cv_map_localized` (per-target WS) | Same, plus per-target spatial variation in σ_s etc.; needs `calibrated_fraction` companion | Slightly more bookkeeping (per-target indexing, NaN handling) | Localized fits exist per-ensemble-member but are currently computed only for the diagnostic overlay, never for the top-K subset or fed past Step 2 |

---

## 6. Coverage: what it means, and how it's computed

`SelectedModel.coverage` is produced by `compute_coverage_map` at the end
of `select_and_assess_model` (Step 5). It answers a narrower question than
`cv_map`: not "how much do the top-K members *disagree*", but "how many of
the top-K members even *returned a usable number* at this cell in the
first place."

### The formula

```python
subset = models[top_k_indices].astype(float)   # (K, [n_time,] n_pts, n_layers)

valid = (
    np.isfinite(subset) &        # not NaN / not inf
    (subset >= ec_min) &         # not below the instrument noise floor (default 0.1 mS/m)
    (subset <= ec_max)           # not a non-physical saturation/blow-up (default 5000 mS/m)
)

n_valid = valid.sum(axis=0)               # how many of the K members were valid, per cell
coverage = 2.0 * (n_valid / K) - 1.0      # rescaled from [0, K] to [-1, +1]
```

Per cell, independently of every other cell:

| `n_valid` (out of K) | `coverage` | Meaning |
|---|---|---|
| K (all) | **+1.0** | Fully constrained — every top-K member produced a plausible EC here |
| > K/2 | between 0 and +1 | More than half agree on a usable value |
| = K/2 | **0.0** | Marginal — exactly half failed |
| < K/2 | between −1 and 0 | Poorly constrained — most members failed here |
| 0 (none) | **−1.0** | Fully unconstrained / masked — no member returned anything usable |

The plotting layer hatches any cell with `coverage ≤ 0` (fewer than half
the ensemble valid) as a visual "don't trust this cell" flag, independent
of whatever `cv_map`/`cv_theta` says about it.

A member counts as **invalid** at a cell for any of three reasons:
- **Numerical failure** — NaN or ±inf. This is also how a real-data
  time-lapse cell with no coverage at that epoch (`geometry_utils.match_epochs_to_reference_grid`)
  gets folded into "unconstrained" automatically, with no special-casing needed.
- **Below `ec_min`** (default 0.1 mS/m) — non-physical, below the
  instrument noise floor.
- **Above `ec_max`** (default 5000 mS/m) — non-physical saturation or a
  forward-model blow-up.

### Worked example

The figure below is a synthetic, illustrative example (not real pipeline
output) with `K = 6` members and 10 toy cells, engineered to show each
outcome:

![Coverage map explainer](coverage_map_explainer.png)

- **Cell 2**: only member 2 is NaN → `5/6` valid → `coverage = 2·(5/6)−1 = +0.67`.
- **Cell 4**: 3 members fail (two NaN, one saturated at 8000 mS/m) →
  exactly half valid → `coverage = 0.0` (the marginal boundary — already hatched).
- **Cell 6**: only 1 member valid → `coverage = 2·(1/6)−1 ≈ −0.67` — heavily hatched.
- **Cell 8**: every member NaN → `coverage = −1.0` — fully masked.
- The remaining cells have all 6 members valid → `coverage = +1.0`.

### Why it's a separate signal from `cv_map` / `cv_theta`

`cv_map` (and the `theta_cv_map`/`theta_cv_map_localized` proposed above)
are only meaningful **among the members that returned a value** — they
measure *disagreement*, not *existence*. A cell could in principle have a
deceptively low CV built from just one or two surviving members (`n_valid`
small) that happen to agree with each other by chance, while `coverage`
correctly flags that cell as unreliable. Always read `coverage` and
`cv_map`/`cv_theta` together, not `cv_map` alone — this is exactly the
same reasoning behind adding `calibrated_fraction` in §2b: a cell can look
fine on one axis (agreement) while failing badly on the other (how many
members/targets actually contributed a real value at all).

### What it looks like in your actual production plots

Now that you've shared `plot_water_content.py`, here's the same synthetic
scenario rendered with your **real** `plot_water_content_section()`
function — same `Blues` θ colormap, same `_add_hatching` styling, same
legend wording:

![Coverage in production-style plot](coverage_production_style.png)

Two independent hatch types, both driven by different arrays:
- **Light-grey `///`** — `coverage ≤ 0` (the synthetic "bad patch" around
  x ≈ 8–13 m, deepest two layers, where 6 of 8 members returned NaN).
- **White `xxx`** — `cv_map > 100 %` (high disagreement among whichever
  members *did* return a value — shown here as a second, independent
  patch around x ≈ 16–19 m, layers 0–1, deliberately placed away from the
  coverage patch to make clear these are two separate signals).

Notice `coverage` is never given its own colorbar/colormap in your
codebase — it only ever appears as a hatch mask on top of whatever the
main panel is showing (θ here, or CV in `plot_water_content_paired`'s
right-hand panel). The mechanism figure above (§ worked example) unpacks
the arithmetic behind that mask; this one shows how it actually renders.

*(Note: `depths0` in your convention lists the interior layer interfaces,
so `len(depths0) == n_layers − 1` — e.g. `depths0=[0.3, 0.6, 1.0]` for 4
layers, matching `DEPTHS0` in `1_run_model_generation.py`. Worth keeping
in mind if you ever pass a `depths0` built independently from the
inversion's `cond_ranges`/`fixed_depths` lists elsewhere in the pipeline —
an off-by-one there would silently mislabel the deepest layer's plotted
depth.)*
