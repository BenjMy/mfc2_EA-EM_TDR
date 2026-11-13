# 1-D surface-only vs full-depth TDR: observations and interpretation

*Based on the 8 synthetic runs of `run_1d_surface_vs_full.sh` (K = 1..4 free Waxman-Smits parameters, 5 probe positions), run on 2026-11-11. Inputs: `comparison_table.csv/.md` and `truth_error_summary.csv`. One run per configuration.*

## 1. Bottom line

Full-depth TDR beats surface-only TDR against the truth in all four K settings, but the gap is **modest** (about 11-13% lower point RMSE for K ≥ 2, 26% for K = 1), and surface-only is far from failing. I think this is less dramatic than you expected, and I do not think the experiment as built can show the strong effect you were probably looking for. The main reason is the design, not the method (section 5).

The overall "Final RMSE (Step 5)" column is **not** a fair way to compare the two arms, and it hides the real difference (section 4b).

## 2. What the two arms are

- **Surface-only:** 5 probes at 0.2 m, i.e. horizon 0 only: 15 observations. Horizons 1-3 are not calibrated and use the lab prior (`lab_p0`: a = 0.5, m = 3.0, sigma\_w = 500, sigma\_s = 100).
- **Full-depth:** 20 probes (4 horizons x 5 positions): 60 observations.
- **Truth:** m and sigma\_s vary smoothly along x (m in 2-4, sigma\_s in 50-200) and are **the same at every depth**. a and sigma\_w are constant, at 0.50 and 500, which are exactly the `lab_p0` values.
- **K** = number of free WS parameters, freed in the order sigma\_s, m, sigma\_w, a.

## 3. Observations

Against the truth (`truth_error_summary.csv`, n = 40 values per run, units assumed m3/m3). The "spread" column is the standard deviation of the error, computed as sqrt(RMSE² - bias²).

| K | Surface RMSE | Full RMSE | Surface bias | Full bias | Surface spread | Full spread |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0562 | 0.0416 | +0.008 | +0.010 | 0.056 | 0.040 |
| 2 | 0.0437 | 0.0390 | -0.009 | -0.025 | 0.043 | 0.030 |
| 3 | 0.0436 | 0.0380 | -0.002 | -0.024 | 0.044 | 0.030 |
| 4 | 0.0442 | 0.0393 | -0.003 | -0.021 | 0.044 | 0.033 |

From the comparison table:

| K | Water content RMSE vs truth, XZ section \[%VWC\], surface / full | Final RMSE (Step 5) \[m3/m3\], surface / full | Optimal stack depth, surface / full | CV map max, surface / full |
| --- | --- | --- | --- | --- |
| 1 | 6.62 / 6.09 | 0.068 / 0.059 | 4 / 16 | 2.13 / 0.97 |
| 2 | 5.55 / 3.74 | 0.033 / 0.036 | 6 / 23 | 1.13 / 0.88 |
| 3 | 5.39 / 3.69 | 0.036 / 0.035 | 7 / 27 | 1.23 / 0.97 |
| 4 | 6.18 / 3.90 | 0.033 / 0.036 | 4 / 14 | 1.26 / 0.96 |

Other points:

- The EM-inversion posterior standard deviation is essentially identical in all runs (mean about 1.0 mS/m, median about 0.2 mS/m).
- All 8 runs finished with quality-gate **WARN** and 0 failed inversion tasks. The reasons are not in these tables.
- The mean calculated water content is lower for full-depth at K ≥ 2 (0.229 vs 0.248-0.250), consistent with its negative bias.

## 4. Interpretation

**a. Full-depth wins on truth, consistently.** Point RMSE falls by 26% (K = 1) and 11-13% (K = 2-4). The XZ-section RMSE falls by only 8% at K = 1 but by 32-37% at K ≥ 2. The two truth metrics sample the field differently, so compare their ranking, not their magnitudes.

**b. The Step 5 RMSE misleads here.** It measures the fit to the TDR observations the run was given: 15 values at one depth for surface-only, 60 values over four depths for full-depth. Fitting one horizon is easier than fitting four, so surface-only can show a lower number (K = 2 and K = 4) while being worse against the truth. The apparent "surface-only is better" in the earlier log is, in my reading, a selection artefact. Use the truth-based columns for the comparison.

**c. Different kinds of error.** Surface-only is nearly unbiased but noisy (spread 0.043-0.044). Full-depth has a 25-32% smaller spread but a systematic dry bias of about -0.021 to -0.025. I cannot say from these tables which horizon the bias comes from. The per-horizon 1:1 plots will show it.

**d. K = 1 is the worst case for both.** Only sigma\_s is free, while the truth also varies m between 2 and 4 and m is fixed at 3.0. That is a deliberate misspecification, and it degrades surface-only more (0.056 vs 0.042), plausibly because four times the data constrains the single free parameter better.

**e. K = 3 and K = 4 bring nothing.** Fixing a and sigma\_w at their prior values is already exactly right, because the truth uses those same values. Freeing them can only add freedom, and the results show no gain (and a slight loss for full-depth at K = 4). The well-specified model is K = 2.

**f. Fewer observations lead to a smaller, noisier ensemble stack.** With 15 observations the selection keeps 4-7 members, against 14-27 with 60 observations, and surface-only has a higher maximum CV (1.1-2.1 vs about 0.9-1.0). That points to a few poorly constrained locations when only horizon 0 is calibrated.

**g. EM uncertainty is independent of the probe design.** The posterior standard deviation does not change, as expected, since the EM inversion does not use the TDR data.

## 5. Why this is probably not what you expected

The experiment favours surface-only by construction:

1. **Truth does not vary with depth.** m and sigma\_s are the same at every depth, so what is learned at 0.2 m transfers to every horizon. Real soils will not do that.
2. **The prior sits at the truth.** The uncalibrated horizons use `lab_p0`: m = 3.0 is the centre of the true range (2-4), sigma\_s = 100 is within the range (50-200) and below its midpoint, and a and sigma\_w are exactly the true values. Not calibrating a horizon therefore costs little.
3. **The arms differ in probe count and depth together.** 5 vs 20 probes (15 vs 60 observations). The benefit of depth coverage cannot be separated from the benefit of more data.

So the honest reading is: *in this setting*, five surface probes recover most of the benefit, and the depth coverage adds roughly 11-13% in point RMSE. It is not evidence that surface-only calibration is sufficient in general.

## 6. Caveats

- One run per configuration, one truth realization and one `ws_xy_seed`. Differences of 5-10% are within what I would expect from run-to-run variation.
- Whether two identical runs give the same result was not yet confirmed (the inversion seeding patch is untested). Until it is, the numbers above could change on rerun.
- n = 40 error values come from spatially correlated points, so they are not 40 independent samples.
- I have not seen the per-horizon plots or the quality-gate warnings.

## 7. Suggested next steps (in order)

1. Run `plot_validation_1to1.py` (needs `ec_optimal.npy`) to see per-horizon EC and theta errors, which show where full-depth wins and where the bias comes from.
2. Confirm reproducibility (two runs of `1d_surface_tdr`, identical `final_rmse`), then repeat the study with 5 or more truth seeds / realizations and report mean and spread.
3. Detune the prior: for example `lab_p0` with m = 2.2 and sigma\_s = 60 (or 3.8 and 180), so the uncalibrated horizons are genuinely wrong.
4. Make the truth depth-dependent (different m and sigma\_s per horizon). This is the case surface-only calibration should fail at.
5. Separate depth from data volume: compare surface-only with more probe positions against full-depth with fewer positions, so observation counts match.
6. Read the quality-gate messages in one `pipeline_report_*.md`, to check that WARN is not hiding a real problem.
