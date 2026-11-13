# Why interpret CV in EC-space rather than θ-space?

## Question

> Why is it useful to interpret CV in terms of EC? Should CV be computed on θ
> instead? What does the EC-based CV allow us to conclude about the quality of
> the ECa → SMC (soil moisture content) translation?

## Two stacked uncertainty sources

The pipeline has two distinct steps, each with its own uncertainty:

1. **EM inversion** — turning apparent conductivity (ECa) into bulk EC per
   layer/cell. This step is *non-unique*: different `alpha` / `method` /
   `forwardModel` combinations in `param_grid` converge to different EC
   values for the same ECa data. The ensemble of "top-K models" captures
   exactly this disagreement.
2. **Petrophysical translation** — turning EC into θ via Waxman-Smits,

   ```
   θ = (a · σ_b / (σ_w + σ_s)) ^ (1/m)
   ```

   calibrated against the TDR probes.

## Why CV is computed on EC, not θ

Computing CV directly on the EC ensemble

```python
cell_mean = np.sum(w_view * top_k_models, axis=0)
cell_std  = np.sqrt((K/(K-1)) * np.sum(w_view * (top_k_models - cell_mean)**2, axis=0))
cv_map    = cell_std / cell_mean
```

(`top_k_models` in mS/m) measures **only source #1** — how much the
geophysical inversion disagrees with itself, *before* petrophysics enters
the picture. That's useful because:

- It's a diagnostic that doesn't depend on TDR calibration quality at all —
  it could be computed even in a survey with zero probes.
- It answers "is the EM signal at this cell well-constrained by the
  inversion," independent of whether the soil-moisture calibration is any
  good.

## What happens if CV is computed on θ instead

Because `θ = EC^(1/m)` with `m = 3.0` (fixed in these runs), first-order
error propagation compresses relative uncertainty by roughly a factor of
`m`:

```
CV_θ ≈ CV_EC / m ≈ CV_EC / 3
```

A cell with `CV_EC = 0.37` (enough to trip the 0.30 quality-gate threshold)
would show up as roughly `CV_θ ≈ 0.12` — comfortably under threshold.

**Implication:** if the quality gate were built on θ-space CV instead of
EC-space CV, it would be far less sensitive — it would rarely catch
inversion ambiguity, because the cube-root transform mathematically shrinks
it. Measuring CV upstream, in EC space, is the more conservative and more
sensitive early-warning signal, catching trouble in the geophysics before
it gets diluted by the nonlinear transform.

## What CV(EC) does *not* tell you

CV(EC) is computed *before* the Waxman-Smits step runs. It says nothing
about:

- whether the fixed `a`, `m` values are appropriate,
- how good the fitted `σ_w` / `σ_s` are against the actual TDR readings,
- whether the translation itself is biased or noisy.

## What actually answers the "quality of ECa → SMC translation" question

That requires metrics computed *after* the petrophysical step:

| Metric | What it shows |
|---|---|
| Step 4/5 RMSE (e.g. 0.037 → 0.031 → 0.029 m³/m³ across the three scenarios) | Direct measure of how well calculated θ matches true θ — the whole ECa → SMC chain, end to end |
| Scatter / residual plot (calc θ vs. truth θ) | Bias and spread of the translation directly in water-content space |
| % difference map | Where the translation over- or under-predicts, spatially |

## Practical takeaway

- **CV(EC)** — "how ambiguous is the geophysical inversion here." Use as an
  early, conservative quality gate, independent of calibration quality.
- **RMSE / scatter vs. truth** — "how good is the full ECa → SMC
  translation." This is the direct answer to the translation-quality
  question, and it's already produced by the pipeline reports.
- **CV(θ)**, if computed additionally (propagating the same top-K ensemble
  through the Waxman-Smits fit instead of stopping at EC), would give a more
  intuitive "how uncertain is my final soil-moisture map" number for
  stakeholders — but expect it to be structurally quieter (roughly ÷3) than
  CV(EC), and it will silently fold in any petrophysical miscalibration
  along with the geophysical ambiguity. That makes it a less clean
  diagnostic if the specific goal is isolating where the EM inversion is
  struggling.
