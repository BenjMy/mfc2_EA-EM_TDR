# What `shrinkage_k`, `spatial_calibration`, and `min_obs_per_position` actually do

## TL;DR

- **`shrinkage_k`** is a tuning number **I invented** for the patch — it does not exist anywhere in Loiseau et al. (2026). It controls how much a per-position `σ_s` estimate gets pulled back toward the single horizon-wide `σ_s`, to protect against overfitting noise when a position only has a handful of observations.
- **`spatial_calibration`** is just an on/off switch. `False` (default) = today's code, unchanged, which is also exactly what the paper does. `True` = the new per-position logic runs.
- **`min_obs_per_position`** is a hard floor: below this many observations, don't even attempt a local estimate — just use the global one.
- The paper itself pools **every** TDR/water-content observation for a horizon into **one** regression, profile-wide (that's the behavior your pipeline currently replicates). The closest thing it does to "spatial calibration" is manually splitting Larzac into meadow vs. forest zones and fitting each separately — a hand-done, coarse version of what `spatial_calibration=True` would do automatically.

---

## 1. The shrinkage formula, in plain terms

```python
weight = n_local / (n_local + shrinkage_k)
sigma_s_position = weight * sigma_s_local_estimate + (1 - weight) * sigma_s_global
```

Think of `shrinkage_k` as **"how many local observations would it take before I start trusting the local estimate as much as the global one?"** It's a pseudo-count, not a physical quantity — there's no formula from soil physics that tells you what it should be. It's the standard empirical-Bayes / James-Stein trick: shrink small-sample local estimates toward a well-supported pooled estimate, in proportion to how little local data you have.

Worked example, with your dense scenario's actual sample size (4 depths × 3 time steps = 12 obs per position, but only θ observations feeding one horizon's worth at a time — so really more like 3 obs/position/horizon since each of the 4 depths belongs to a different horizon):

| n_local (obs at this position, this horizon) | shrinkage_k=1 | shrinkage_k=4 | shrinkage_k=10 | shrinkage_k=20 |
|---|---|---|---|---|
| 2 | 0.67 | 0.33 | 0.17 | 0.09 |
| 3 | 0.75 | 0.43 | 0.23 | 0.13 |
| 6 | 0.86 | 0.60 | 0.38 | 0.23 |
| 12 | 0.92 | 0.75 | 0.55 | 0.38 |

Reading a row: with `n_local=3` (typical for one horizon at one position in your setup) and `shrinkage_k=4`, the local estimate only gets 43% weight — mostly still deferring to the global fit. That's appropriate given how little data any single position has. With `shrinkage_k=1`, the same 3 observations would get 75% weight — probably too aggressive given how noisy a 3-point fit is.

The two extremes:
- **`shrinkage_k → 0`**: weight → 1 regardless of `n_local`. Pure local fit, no pooling at all. Every position calibrates independently — fine if you have plenty of obs per position, dangerous with only 2–3.
- **`shrinkage_k → ∞`**: weight → 0. Pure global fit — i.e. **exactly today's code**, no behavior change. This is why `spatial_calibration=False` and "shrinkage_k effectively infinite" are the same thing.

**It's empirical — there's no principled formula to compute it from soil physics.** The honest way to pick it is the same way the paper picks its own hyperparameters (like the horizontal integration window, which they explicitly tuned by sweeping 1–5 m and picking whatever minimized validation RMSE — see Section 3.3.2): sweep `shrinkage_k` over a few values (e.g. 1, 4, 10, 20), refit, and pick whichever minimizes RMSE on a held-out validation subset. You already have the machinery for this since your pipeline computes RMSE-vs-K curves; the same idea extends to RMSE-vs-shrinkage_k.

## 2. `spatial_calibration: bool`

Just a feature flag. `False` skips the whole per-position block and preserves current behavior exactly (so this patch is safe to merge without changing any existing report). `True` runs the per-position estimate + shrinkage step described above, then populates `sigma_s_by_position` on the fit object for Step 5 to consume.

## 3. `min_obs_per_position: int`

A hard cutoff *before* shrinkage even runs. If a position has fewer than this many matched observations for a given horizon, it's not just heavily shrunk — it's skipped entirely and set to the global value. This exists because shrinkage handles "somewhat sparse" gracefully but a position with, say, exactly 1 observation shouldn't be fitting anything (a single point can't even estimate a residual). Given your ~3 obs/position/horizon, `min_obs_per_position=2` is a reasonable floor — it excludes only the most extreme cases (e.g. a probe that failed at 2 of 3 time steps) while still letting shrinkage do the real work everywhere else.

## 4. How does Loiseau et al. actually handle this?

Short answer: **they don't spatially partition the petrophysical fit at all** — this is the paper's explicit, acknowledged simplifying assumption, not an oversight unique to your pipeline.

**What the paper actually does (Section 2.3):** for each of the *N* inverted models, resistivity is extracted at every water-content probe location and time step, producing pairs `[ρ(xᵢ, tⱼ), θ(xᵢ, tⱼ)]` pooled across **all** positions along the profile. One Waxman-Smits relationship (`m, n, σ_w, σ_s`) is then fit **per soil horizon**, using every position's data together — this is precisely the pooling behavior your `calibrate_petrophysical_relationship` replicates. The paper is explicit that porosity, when unknown, and all four WS parameters are optimized this way per horizon, not per position.

The one thing that does vary spatially is the **horizontal integration window** (4.2 m at Avignon, 2.8 m at Larzac) used to average resistivity around each probe before extraction — but that's smoothing the *EC side* to bridge the ERT/point-sensor resolution mismatch, not letting the *θ-EC relationship itself* vary by location.

**The closest thing to `spatial_calibration=True` in the paper** is a manual, coarse version: at Larzac, they split the site into two land-use zones (meadow vs. forest) and fit each horizon's WS relationship separately within each zone, rather than pooling meadow-and-forest observations into one fit per horizon. That's zonal calibration done by hand, using zone boundaries chosen from prior site knowledge — exactly the idea behind `spatial_calibration`, just not automated or shrinkage-regularized.

**The paper names this exact limitation directly**, in the Discussion:

> the approach raises fundamental questions about environmental homogeneity in terms of the measured property (water content) and its sampling... we assume the environment to be homogeneous in terms of texture or soil type. This assumption is a known limitation since natural soils are inherently heterogeneous at various scales.

So this isn't a bug you introduced by running a denser probe network — it's the paper's own stated caveat, and your dense-TDR scenario is the situation where that caveat actually bites (enough spatial coverage to reveal the heterogeneity the pooled fit can't represent).

**On the CV/optimal-K interaction from last time:** the paper's Eq. 8 CV is computed directly on **resistivity** dispersion across the top-K models — the same raw-EC-domain computation your pipeline implements. The paper itself flags this as a partial indicator, not a full uncertainty quantification:

> this coefficient of variation reflects the dispersion associated with inversion parameter choices but does not constitute a full probabilistic quantification of total model uncertainty.

That confirms what I flagged as a "don't touch without checking the paper" item last time: the CV metric behaving the way it does (rising with deeper K-stacks) is consistent with how the paper defines and caveats it, not a defect in your implementation.

## 5. Practical suggestion

Before building the full per-position + shrinkage machinery, consider trying the paper's own approach first: hand-split `2d_dense_tdr` into 2–3 spatial zones (the way Larzac was split into meadow/forest) based on whatever you know about the underlying synthetic field's heterogeneity, and fit each zone separately with the *existing* code (just call `calibrate_petrophysical_relationship` once per zone with that zone's subset of `sample_positions`). That's zero new code, mirrors exactly what's validated in the paper, and will tell you quickly whether zonal splitting alone recovers a passing quality gate — before deciding whether the fancier automated shrinkage version in the earlier patch is worth building.
