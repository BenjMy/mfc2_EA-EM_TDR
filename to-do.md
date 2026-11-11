# To-do (status checked 2026-11-11, branch `refactor/readability`)

- [x] do refactoring.md — phases 0-4 done, 5-10 open; see the status table at the top of `refactoring.md`
- [x] what is free1 2 3? — answered: `K` in `1d_*_free{K}` = number of FREE Waxman-Smits parameters in the calibration; freed in the order sigma_s (K=1), +m (K=2), +sigma_w (K=3), +a (K=4); the others stay fixed at `lab_p0`. So yes, free1 = one free WS parameter (sigma_s).
- [x] fig_ec_1to1_by_horizon: initial EC range too small — upper bound of `cond_ranges` raised 60 -> 300 mS/m in the ten `1d_*` scenarios (commit `fee56d0`). Saturation at ~55 mS/m is gone, but see the caveat below: errors went UP, not down.
- [ ] repeat run_1d_surface_vs_full.sh with calibration of each horizon (not only the first one) — NOT DONE, needs a decision: the surface-only runs have TDR only at 0.2 m, so horizons 1-3 have no data to calibrate with. Options: (a) the existing `full` runs already calibrate all 4 horizons, nothing to add; (b) share horizon 0's fitted WS parameters with the deeper horizons (a new "borrowing" option; the code deliberately does not do this today); (c) add TDR at more depths to the surface study.
- [~] repeat run_1d_surface_vs_full.sh with heterogeneous truth (two layers, H0-1 / H2-3 WS parameters change) — implemented, runs NOT done: new scenarios `1d_{surface,full}_hetero_free{1..4}` (truth a 0.5 -> 0.35 and sigma_w 500 -> 700 from horizon 2 down; m and sigma_s still vary along x) and `VARIANT=hetero ./run_1d_surface_vs_full.sh`. Only `1d_full_hetero_free3` was run (as a check; RMSE 0.0526, truth verified). The 8-run study takes ~40 min on this machine.

## Caveat on the EC range change (item 3)
Same pair (`surface_free1` / `full_free1`), per-horizon EC RMSE, old range -> new range:
H0 15.4 -> 25.1 (surface), 13.6 -> 25.0 (full); H1 31.8 -> 61.2 / 31.5 -> 54.0; H2 47.7 -> 61.7 / 45.3 -> 66.0; H3 57.6 -> 41.3 / 57.6 -> 21.7.
Only the deepest horizon improved. The old cap hid part of the non-uniqueness; with room to move, the optimal EC now overshoots (bias +50 mS/m at H1-H2). theta RMSE at free1 also rose (H1 0.046 -> 0.187 surface). Not reverted since it is what the to-do asked for; a value between 60 and 300 (e.g. 150) is worth a try.
