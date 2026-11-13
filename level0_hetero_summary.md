# Level 0: pinned-layer ceiling (0002_1d_surface_hetero_free2)
## Hypothesis being checked

Surface-only TDR calibrates only horizon 0, yet the question is whether this still improves the
DEEP layers (1-3) of the EC / SWC model. The EM inversion never sees TDR data, so the ECa data
can be explained by trading error between layer 0 and layers 1-3 (non-uniqueness). If layer 0
were known (pinned), that trade-off would disappear and the deep layers should improve.

Level 0 tests this WITHOUT any new inversion, on the ensemble already computed:
(a) if members that fit layer 0 better ALSO fit layers 1-3 better, then picking members by a
layer-0 criterion (which is all surface TDR can offer) helps the deep layers; (b) if layer-0
error anti-correlates with layer-1..3 error across members, that is the trade-off itself.

## The four stacks (all of the same size K = optimal stack depth chosen by the pipeline)

| stack | what it is | what it tells us |
|---|---|---|
| `all` | equal-weight mean of ALL ensemble members, no selection | reference: no TDR information used |
| `tdr` | the pipeline's own result: top-K members ranked by misfit to the surface TDR (Step 2/3), RMSE-weighted | what surface-only TDR actually delivers |
| `oracle` | top-K members ranked by their TRUE layer-0 EC error vs the synthetic truth, equal weights | ceiling of what selection on a perfectly known layer 0 can reach ("pinned layer 0" by selection) |
| `random` | K members drawn at random (mean over many draws; 5-95 % band in the table) | null model: any gain over `random` is due to the selection criterion, not to stacking |

Reading the table: `tdr` < `random` at layers 1-3 means surface TDR helps the deep layers;
`oracle` < `tdr` there means a known layer 0 would help more (pinning is worth building);
`oracle` ~ `tdr` ~ `random` or `oracle` worse means selection cannot help and only conditioning
INSIDE the inversion (a pinned layer) could. Errors are bulk EC in mS/m, bias = stack - truth.

## How to reproduce (synthetic data only, one realisation per run)

Needs the code with `ensemble_members.npz`, `member_ranking.npy` and `member_rmse.npy` saved
(commits "Save raw ensemble members ..." and "Level 0 analysis ..."), and `emagpy` 1.4.6.

```bash
# 1. run the scenario (about 7 min on one CPU; 5 surface probes at 0.2 m, K=2 free WS parameters)
./run_pipeline.sh -s 1d_surface_hetero_free2 --mode 1d --num-swc-positions 5   # two-zone vertical truth
./run_pipeline.sh -s 1d_surface_free2        --mode 1d --num-swc-positions 5   # uniform-in-depth truth
# 2. analyse the run folder (the latest numbered folder under mfc2_outputs/field_survey_<scenario>/)
python analysis_pinned_ceiling.py mfc2_outputs/field_survey_1d_surface_hetero_free2/<seq>_1d_surface_hetero_free2
python analysis_pinned_ceiling.py mfc2_outputs/field_survey_1d_surface_free2/<seq>_1d_surface_free2
```
Outputs land in `<run>/analysis_pinned_ceiling/` (`summary.md`, `stack_errors.csv`,
`correlations.csv`, `fig_pinned_ceiling.png`). `--n-random N` sets the number of random draws
(default 2000, seed fixed). Tests: `pytest tests/test_analysis_pinned_ceiling.py`.
Run-to-run reproducibility of the inversion itself has not been confirmed (seeding patch
untested), so numbers can change slightly on a rerun.

## Limits of this test

One truth realisation per run; K can be small (a few members), which makes `oracle` and `random`
noisy; members never saw a pinned layer, so selection is a weaker test than pinning inside the
inversion (a null result here does NOT rule pinning out); a bias shared by all members (e.g. from
the 300 mS/m EC upper bound) cannot be removed by any selection.

## Results

K = 4 members; overlap of TDR-selected and oracle-selected sets: 0%

| stack | L0 RMSE (bias) | L1 RMSE (bias) | L2 RMSE (bias) | L3 RMSE (bias) |
|---|---|---|---|---|
| all | 20.3 (+9.2) | 78.0 (+72.3) | 54.5 (+39.3) | 24.5 (+5.1) |
| tdr | 23.5 (+14.5) | 71.0 (+61.8) | 61.3 (+44.9) | 36.3 (+1.9) |
| oracle | 16.1 (+4.6) | 86.5 (+78.9) | 72.8 (+51.4) | 32.6 (+10.6) |
| random | 22.2 (+9.2) | 81.1 (+72.2) | 61.1 (+39.4) | 28.1 (+5.1) |

Random 5-95 % RMSE band per layer: L0 19.2-25.2, L1 70.5-91.8, L2 52.6-70.3, L3 24.2-33.2

Correlation of layer-0 error with deeper-layer error across members (negative = trade-off):

- layer 1: cellwise -0.19, member-mean -0.31
- layer 2: cellwise -0.07, member-mean -0.18
- layer 3: cellwise -0.06, member-mean -0.08

## Perspectives: more complex procedures to test the hypothesis

1. **Oracle pin inside the inversion (Level 1).** Fix layer 0 to its TRUE EC in the Step 1
   inversion (EMagPy `fixedConds`); scenarios `1d_surface_free4_pin0` and
   `1d_surface_hetero_free4_pin0` already exist. Upper bound of what pinning can give.
2. **Pin to a TDR-derived EC (Level 2).** TDR theta at the probes -> calibrated Waxman-Smits ->
   EC of horizon 0, interpolated along x (per epoch for time-lapse); pin layer 0 to it
   (new `pin_source: "tdr"`, horizon 0 first, then any calibrated horizon).
3. **Honest uncertainty (Level 3).** Give each member its own EC draw for the pinned layer
   (spread = calibration + interpolation error) so pinning does not make the deep layers look
   more certain than they are.
4. **Full comparison (Level 4).** Per horizon, EC and theta RMSE/bias vs truth for: free (now),
   pinned-to-TDR (realistic), pinned-to-truth (oracle), full-depth TDR (reference). Success =
   pinned-to-TDR beats free at H1-H3 and sits between free and oracle.
5. **Robustness.** Repeat over several truth seeds / realisations and report mean and spread;
   vary K, the 300 mS/m EC upper bound (try ~150), the number of probes, and the uniform vs
   hetero truth; confirm run-to-run reproducibility first.
6. **Beyond 1-D synthetic.** 2-D grids and time-lapse (one pinned profile per epoch), then the
   real data, where horizon 0 may be thicker than the ~0.2 m TDR footprint (representativeness
   error) and horizons without sensors still need WS parameters (prior or borrowed from H0).
