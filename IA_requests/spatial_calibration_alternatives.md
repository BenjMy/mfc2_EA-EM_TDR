# Beyond fixed shrinkage: geostatistics, nearest-neighbour, adaptive shrinkage, and localization

Short answer: yes, all four of those are real, better-established alternatives, and each fixes a specific weakness of the fixed-`shrinkage_k` patch. Here's what each one is, and which one actually fits your data (1D transect, 2–10 spatial positions, ~3 obs/position/horizon).

## The weakness in plain fixed shrinkage, first

The patch from last time only has two states per position: "has local data → shrink it toward global" or "has no local data → use global, full stop." That second case is exactly your sparse scenario's problem: positions 1–8 (out of 10) have zero TDR probes, so they just silently inherit the pure global value — no spatial information from the two positions that *do* have probes gets transferred to their neighbors. Fixed shrinkage never extrapolates; it only regularizes where you already have direct data.

## The four alternatives

### 1. Nearest-neighbour / IDW (simplest fix for the extrapolation gap)

Assign each EM point's `σ_s` from its nearest TDR-calibrated position, or a distance-weighted blend of several:

```python
sigma_s_at_x = sum(sigma_s_i / dist(x, x_i)**p for i in nearby_positions) \
             / sum(1 / dist(x, x_i)**p for i in nearby_positions)
```

**Pros:** trivial to implement, actually extrapolates to unsampled points (unlike fixed shrinkage).
**Cons:** no denoising — if the "nearest" position's local estimate is itself noisy (which it will be, with ~3 obs), you just hand that noise straight to its neighbors. Also produces discontinuous, blocky step patterns at the boundary between two probes' zones of influence.

### 2. Kernel-weighted / localization (this is what I'd actually recommend)

Instead of hard-slicing observations into "belongs to this position" buckets, weight **every** observation in the horizon by a smooth function of its distance from the target EM point, and refit at every target location:

```python
def gaspari_cohn(d, c):
    """Compact-support localization kernel (0 outside 2c), standard in
    ensemble data assimilation (Gaspari & Cohn, 1999) for exactly this
    problem: tapering the influence of observations by distance."""
    r = np.abs(d) / c
    w = np.zeros_like(r)
    m1 = r <= 1
    w[m1] = -0.25*r[m1]**5 + 0.5*r[m1]**4 + 0.625*r[m1]**3 - 5/3*r[m1]**2 + 1
    m2 = (r > 1) & (r <= 2)
    w[m2] = (r[m2]**5)/12 - 0.5*r[m2]**4 + 0.625*r[m2]**3 + 5/3*r[m2]**2 - 5*r[m2] + 4 - 2/(3*r[m2])
    return np.clip(w, 0, None)

# In calibrate_petrophysical_relationship, for each target EM point x_target:
spatial_w = gaspari_cohn(x_all_obs - x_target, c=bandwidth)
combined_w = obs_weights * spatial_w   # multiply with your existing theta_obs_std weights
# feed combined_w into the same _log_ratio_objective as the per-obs weight,
# and re-fit sigma_s once per target EM point instead of once per horizon globally.
```

This is a strict generalization of both extremes: `bandwidth → ∞` reproduces today's single global fit exactly; `bandwidth → 0` collapses to nearest-neighbour. One tunable parameter (`c`, the localization radius) instead of shrinkage's two (`shrinkage_k`, `min_obs_per_position`), and it extrapolates continuously to every EM point rather than only the ones with direct probes.

`c` itself is exactly as "empirical" as `shrinkage_k` was — pick it the same way, sweeping a few values and checking validation RMSE.

**Why "localization"?** This is standard terminology from ensemble data assimilation (ensemble Kalman filter/EnKF): when you have too few ensemble members or too few observations to trust a global covariance/parameter estimate, you taper its influence by distance instead of using it everywhere uniformly. Your Step 3/4 (RMSE-weighted ensemble averaging) is already structurally an ensemble-DA-style procedure, so this fits naturally — see item 5 below for the fuller version of this idea.

### 3. Adaptive / empirical-Bayes shrinkage (removes the need to hand-pick `shrinkage_k`)

Instead of you guessing `shrinkage_k`, estimate it from the data via a random-effects (hierarchical) model:

```
sigma_s_i  ~  Normal(sigma_s_global, tau^2)      # between-position variance
local estimate at i has known/estimated noise variance v_i
```

The statistically optimal shrinkage weight per position then falls out automatically:

```python
weight_i = tau**2 / (tau**2 + v_i)   # no hand-picked k anywhere
```

`tau²` (how much `σ_s` genuinely varies position-to-position) and `v_i` (how noisy each position's own local estimate is) are both estimated from your own 10 (or 2) position-level fits — e.g. via a one-way random-effects ANOVA / method-of-moments, or `statsmodels.MixedLM`. If you want to go further, this is the same family as Stephens (2016)'s "adaptive shrinkage" (`ashr`), which lets the *shape* of the prior on position-level deviations be estimated non-parametrically instead of assuming Gaussian — more flexible, more machinery.

**Pros:** no manually-tuned constant.
**Cons:** with only ~10 positions (or 2!), estimating `tau²` itself is a small-sample problem — you've just moved the "not enough data" issue up one level rather than eliminating it. Still doesn't extrapolate to unsampled positions (same binary local/global split as fixed shrinkage).

### 4. Geostatistics / kriging (most rigorous, but the most data-hungry)

Model `σ_s(x)` as a spatially correlated random field with a covariance function `C(h)` (e.g. exponential, `C(h) = σ² · exp(-h/range)`). Ordinary or simple kriging then gives the best linear unbiased predictor **at any location**, including every unsampled EM point, plus an explicit **kriging variance** — a real, principled per-location uncertainty you could use instead of (or alongside) the ensemble CV map.

**The catch:** estimating an empirical variogram needs many observation pairs across a range of lag distances. With 10 positions you get 45 pairs — thin but arguably workable with a robust/parametric variogram fit. With 2 positions (your sparse scenario) you get exactly **one** pair — a variogram is not estimable at all. In that case you'd have to *impose* a variogram (fixed range/sill from prior/expert knowledge) rather than fit one — this is "simple kriging with known covariance parameters" or Bayesian kriging with an informative prior, not empirical variogram-fitting.

**A trick specific to your setup:** you actually *do* have 10 spatially dense points regardless of TDR probe count — the raw EM/ECa inversion itself is always sampled at all 10 measured points. You could estimate the spatial correlation length from the **EC field's** own spatial autocorrelation (which has full coverage) and import that correlation length as the kriging range for `σ_s`, rather than trying to estimate it from the sparse TDR positions themselves. That sidesteps the "not enough TDR points for a variogram" problem entirely.

### 5. The complete version: localization applied to the ensemble weights themselves, not just σ_s

Everything above only fixes the Step 2/4 calibration pooling. But recall from before: Step 3/4/5's RMSE-based ensemble weighting (`w_i`, Eq. 6–7 in Loiseau et al.) is currently computed **once, globally**, and applied identically to every spatial cell — this is what drives the optimal-K-depth/CV inflation effect too. A fully localized version would compute a **separate weight vector per spatial cell**, using only nearby TDR observations to rank/weight ensemble members for that cell, the same way a localized EnKF update works. That would address the calibration-pooling issue *and* the CV/optimal-K coupling in one consistent design, rather than patching calibration and leaving the ensemble-weighting/CV machinery untouched. It's the most invasive option (touches Steps 3, 4, and 5, not just 2), but it's the architecturally "correct" generalization if you want to actually fix both symptoms at their common root.

## Comparison

| Method | Extrapolates to unsampled points | Works with only 2 TDR positions | New tunables | Code footprint | Gives its own spatial uncertainty |
|---|---|---|---|---|---|
| Nearest-neighbour / IDW | yes | yes, but crude | 0–1 | trivial | no |
| Fixed shrinkage (last patch) | no | degrades to pure-global | 2 | small | no |
| Kernel weighting / localization | yes, continuously | yes, gracefully | 1 | small–moderate | no |
| Adaptive (empirical-Bayes) shrinkage | no | poorly (too few positions for `τ²`) | 0 (data-driven) | moderate | approximate |
| Kriging / geostatistics | yes, best in principle | only with an *imposed*, not fitted, variogram | 1–2 | higher | yes |
| Full localized ensemble weighting (Steps 3–5) | yes | yes | 1 | highest | yes |

## My recommendation for your specific case

1. **Start with kernel weighting / localization (§2)** on the σ_s calibration. It's a strict, one-parameter generalization of the shrinkage patch, it actually extrapolates to unsampled EM points (which fixed shrinkage never did), and it's a small edit to the same `calibrate_petrophysical_relationship` function.
2. **If you want real per-location uncertainty**, not just a bias fix, the honest answer is §5 — extend the same localization idea to the ensemble-weighting/CV machinery in Steps 3–5, not just the calibration in Step 2/4. That's the piece that actually reconciles the "denser TDR → deeper optimal K → higher CV" behavior from before, rather than leaving it as a documented paper limitation.
3. **Kriging is worth it only if you can borrow a correlation length from your existing dense EC data** rather than trying to fit a variogram on 10 (or 2) TDR points directly — otherwise you're imposing a somewhat arbitrary range parameter, which isn't obviously better than picking a kernel bandwidth in §2.
