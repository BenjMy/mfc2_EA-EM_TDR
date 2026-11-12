# To-do (status checked 2026-11-11, branch `refactor/readability`)

Update 2026-11-11 (evening): refactoring phases 8 (unknown-key validation, `--list-scenarios`) and 10 (pytest: 48 passed, 1 skipped without `emagpy`) advanced, loaders archived; see `refactoring.md`. The two open items below still need a decision (item 4) or a ~40 min run on a machine with `emagpy` (hetero study).

- [x] do refactoring.md — phases 0-4 done, 5-10 open; see the status table at the top of `refactoring.md`
- [x] what is free1 2 3? — answered: `K` in `1d_*_free{K}` = number of FREE Waxman-Smits parameters in the calibration; freed in the order sigma_s (K=1), +m (K=2), +sigma_w (K=3), +a (K=4); the others stay fixed at `lab_p0`. So yes, free1 = one free WS parameter (sigma_s).
- [x] fig_ec_1to1_by_horizon: initial EC range too small — upper bound of `cond_ranges` raised 60 -> 300 mS/m in the ten `1d_*` scenarios (commit `fee56d0`). Saturation at ~55 mS/m is gone, but see the caveat below: errors went UP, not down.
- [~] parked 2026-11-12 ("forget for the moment"): calibration of each horizon in the surface-only study. Details kept under *Parked* below.
- [~] repeat run_1d_surface_vs_full.sh with heterogeneous truth (two layers, H0-1 / H2-3 WS parameters change) — implemented, runs NOT done: new scenarios `1d_{surface,full}_hetero_free{1..4}` (truth a 0.5 -> 0.35 and sigma_w 500 -> 700 from horizon 2 down; m and sigma_s still vary along x) and `VARIANT=hetero ./run_1d_surface_vs_full.sh`. Only `1d_full_hetero_free3` was run (as a check; RMSE 0.0526, truth verified). The 8-run study takes ~40 min on this machine.


## New (added 2026-11-12)
- [ ] Plot the WS **truth** spatial distribution (a, m, sigma_w, sigma_s per horizon along x) for the uniform-in-depth case (`1d_*_free{K}`) and the heterogeneous case (`1d_*_hetero_free{K}`). Source: `output_model/synthetic_ws_truth_{a,m,sigma_w,sigma_s}.npy` (check shapes first). Suggested: `plots/ws_distribution.py`, one figure per run, called from the study script.
- [ ] Plot the WS **initial** spatial distribution (starting values used in the calibration: `lab_p0` / per-horizon values, with free/fixed flag per parameter), same layout as the truth plot so both can be overlaid.
- [ ] Demonstrate that **surface-only TDR calibration still improves the deep SWC and/or EC model**. Hypothesis: without a known first layer the ECa inversion trades layer-0 error against layers 1-3 (non-uniqueness); pinning the calibrated horizon removes that. Design (feasible with EMagPy 1.4.6, checked in its source: `Problem.setInit(depths0, conds0, fixedDepths, fixedConds)` takes a per-layer `fixedConds` flag and `conds0` as an (n_points x n_layers) matrix):
  1. Pass 1 = current pipeline (all layers free): Step 1 -> Step 2/4 calibration with surface TDR.
  2. Derive EC of each calibrated horizon at every EM point: TDR theta at the probes -> calibrated WS -> EC_bulk at the probes, interpolated along x (1-D) / over the grid (2-D) (time-lapse: one profile per epoch).
  3. Pass 2 = Step 1 again with `fixedConds=[True, False, False, False]` and `conds0[:, 0]` = that EC. Needs: `step1_inversion._invert_single_worker` to accept `fixed_conds` and a per-point `conds0`; `geometry_utils.build_inversion_bnds` to drop the bounds of fixed layers (EMagPy wants `len(bnds)` = number of variable parameters); ensemble corner combinations only over the free layers. New scenario field, e.g. `constrain_calibrated_horizons: "off" | "tdr" | "truth"`; generalises to any calibrated horizon via `is_calibrated`.
  4. Keep the uncertainty honest: instead of one hard value, give each ensemble member its own EC draw for the pinned layer (spread = WS calibration + interpolation error), so the constraint does not make the deep layers look more certain than they are.
  5. Compare per horizon (EC and theta RMSE/bias vs truth, `plot_validation_1to1.py`): free (now) vs pinned-to-TDR (realistic) vs pinned-to-truth (oracle upper bound) vs full-depth TDR (reference). Success = pinned-to-TDR beats free at H1-H3 and sits between free and oracle. Run on the uniform and the hetero truth.
  Open points: the deeper horizons still need WS parameters (default `lab_p0`, or the parked "borrowing" option); TDR sees ~0.2 m while horizon 0 may be thicker (representativeness error).

## Parked
- [ ] repeat run_1d_surface_vs_full.sh with calibration of each horizon (not only the first one) — NOT DONE, needs a decision: the surface-only runs have TDR only at 0.2 m, so horizons 1-3 have no data to calibrate with. Options: (a) the existing `full` runs already calibrate all 4 horizons, nothing to add; (b) share horizon 0's fitted WS parameters with the deeper horizons (a new "borrowing" option; the code deliberately does not do this today); (c) add TDR at more depths to the surface study.

## Caveat on the EC range change (item 3)
Same pair (`surface_free1` / `full_free1`), per-horizon EC RMSE, old range -> new range:
H0 15.4 -> 25.1 (surface), 13.6 -> 25.0 (full); H1 31.8 -> 61.2 / 31.5 -> 54.0; H2 47.7 -> 61.7 / 45.3 -> 66.0; H3 57.6 -> 41.3 / 57.6 -> 21.7.
Only the deepest horizon improved. The old cap hid part of the non-uniqueness; with room to move, the optimal EC now overshoots (bias +50 mS/m at H1-H2). theta RMSE at free1 also rose (H1 0.046 -> 0.187 surface). Not reverted since it is what the to-do asked for; a value between 60 and 300 (e.g. 150) is worth a try.
