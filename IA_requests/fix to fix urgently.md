

3. Preprocessing is dropping a lot of data

filterRange alone removed 6276 + 4494 = 10,770 points before any other filter ran, and total preprocessing removed 14,506 of 34,329 raw points (~42%). Worth double-checking the range bounds aren't over-aggressive for this survey — that's a big chunk of the raw EM signal being discarded before it ever reaches the inversion.

4. TDR noise weighting appears silently disabled despite being configured

The scenario sets tdr_noise_std=0.01, but Step 2 logs: "TDR observation weighting: disabled (no positive tdr_noise_std available for this run)". That's a config value not making it through to where it's needed — likely a plumbing bug between the scenario loader and Step 2's weighting logic, worth tracing.



Still open, need your input:

forward_simulate_eca_optimal (real-data + time-lapse only) — needs matched_indices threaded through, a signature change I didn't want to make without your OK.
station_to_spatial_idx[obs['probe_index']] — needs 0_real_data_loader.py to actually verify, still don't have that one.

Everything's in the updated report and code. Want me to go ahead and fix #5 (forward_simulate_eca_optimal) now, or send over 0_real_data_loader.py for #6?




# FOR the real dataset Localized (kernel-weighted) WS calibration failed

STEP 2 LOCALIZED bandwidth sweep completed fine (c=20 recommended, LOO-CV RMSE 0.112, 0/39 folds fell back to prior at c=20 — good). But the follow-up localized diagnostic fit crashed:

boolean index did not match indexed array along axis 0; size of axis is 39 but size of corresponding boolean axis is 71

Looks like a horizon-filtered array (39 = Hz0 sensor-epoch pairs) being indexed with a mask built against the full 71-observation array. This burned 195.7 s before failing and produced no localized output — worth a quick look at the localized-calibration code path (probably in whatever step builds the boolean mask for valid folds vs. the full observation count).


The error fires inside step2.run_petro_localized(...), called at run_pipeline.py:4232. That function lives in module 2_run_petro (step2 = importlib.import_module("2_run_petro") at line 75) — that file wasn't uploaded, only em_methodology.py (EMmethod).

I checked the only candidate in em_methodology.py, calibrate_petrophysical_relationship_localized (line 2163) — it can't be the source of the 39 vs 71 mismatch: every boolean mask it builds (_finite, etc.) is sized against indices (the per-horizon subset, e.g. Hz0's 39 obs), never against the full 71-observation array. Same for sweep_localization_bandwidth, which is why the sweep step ran fine and only the follow-up run_petro_localized call failed.

The one suspicious clue I can see from run_pipeline.py is this extra keyword at the call site:

python
fits_step2_localized = step2.run_petro_localized(
    ...
    theta_obs_std=_theta_obs_std_step2,
    global_fit=fits_step2[0],   # <-- not a param of calibrate_petrophysical_relationship_localized
)

global_fit isn't accepted by the em_methodology function's signature at all, so 2_run_petro.run_petro_localized must be a wrapper that does its own per-horizon looping and, somewhere in there, builds a boolean mask against the full 71-length theta_obs/sample_positions array (matching fits_step2[0], i.e. global_fit, which does cover all 71 obs) and then applies it to an already-horizon-filtered 39-length array — that's exactly the size of axis is 39 but size of corresponding boolean axis is 71 shape mismatch.





# performamnce with localisation activated


how can I show the code performance when I use localisation? when there is no localisation its easy I plot the petrophysical R2 but with location if I put all on the same plot logically its spreads out. Maybe add a new plot and plotting result of the petro for each sensors would make the trick? 


# check all indexing 

for instance: 

Your codebase explicitly notes (in _extract_ec_for_plots's docstring, 2_run_petro.py line ~897) that best_fit.observed_theta is not in the same order as the sample_positions you pass into run_petro. It gets re-routed/reordered by horizon via geometry_utils.route_ec_to_horizons(...). So if I naively zip your original sample_positions array against best_fit.observed_theta to color/facet by sensor, the sensor labels would silently line up with the wrong observations — a bug that wouldn't throw an error, just quietly mislabel points.


# check all indexing as for files if needed
- check indexing after localisation etc ... 
- check indexing after gridding 
- check indexing in the plots ...
- write a report




can you generate a scenario where waxman parms are varying in xy direction?


# Create a new scenario file to test algoritm validity in "stupid" not realistic case such as:
- no noise, homogeneous soiyesl in SMC and WS parms
- scenario with fixed WS parms
- ... (suggest other test cases)

for each write should I oberved i.e. for instance (check if the estimated WS parms and p0 aligned etc ..

by the way in which case (scenario) my estimated and p0 values of waxman parms should be aligned? also why sometimes the picked estimated parm is not the central value of the hist ditribution? 


# quid case where I want to invert in 3d the WS parms? up to now its only 1D per horizon. How should I do? implement?



# remove noise in EC and TDR? in real data mode + temperature correction


# varying spatial Waxman parameters

# would be nice to have also scenario with only VCP or HCP coil orientations


# Propagating EC-ensemble uncertainty to θ (volumetric water content)


# does ws-localized-bandwidth-sweep and window_m somehow overlap? ws-localized-bandwidth-sweep activated deactived somehow window_m?

This is conceptually closer to your --ws-localized-bandwidth-sweep (LOO-CV bandwidth selection) than to window_m itself — both are "sweep candidate widths, pick the one that minimizes validation error" solutions to a horizontal-footprint problem. 


# for the real dataset

| `window_m` | `1.0` |
| `spatial_match_tolerance_m (real time-lapse)` | `2.0` |

what are the differences between both?

# How to compensate for the lack of sensors at depth?

# remove CV thershold and contour lines in water_content_paired_2d.png



# The bottleneck

The idea is: don't run a full 4-parameter optimization at every single EM measurement point — fit on a coarser subset of targets, then fill in the rest by interpolating the fitted parameters rather than re-optimizing.

**Why it's defensible here specifically**

The localized fit is Gaspari-Cohn weighted with a `bandwidth` (`c`) that controls how far each target "sees." Two targets that are close together relative to that bandwidth are looking at almost the same weighted subset of observations — their weight vectors `combined_w` are nearly identical, so their optimal `(a, m, sigma_w, sigma_s)` are nearly identical too. The Waxman-Smits parameters are physical soil properties (tortuosity, cementation exponent, pore/surface conductivity) — they vary smoothly in space by construction, not point-to-point. So the *true* underlying `a(x), m(x), sigma_w(x), sigma_s(x)` curves are smooth, and solving for every single grid point is spending a full L-BFGS-B run to resolve variation that mostly isn't there.

**Concretely, in this code:**

- `target_x_all` (and `target_y_all` in 2D) is already the ordered array of candidate target locations, and `target_indices` in the docstring is explicitly exposed for subsampling — it's already built for this.
- Instead of `for t_i, target in enumerate(target_x_all)` calling `_fit_ws_horizon_weighted` every time, you'd pick a stride — say every 3rd or 5th target, or an adaptive spacing tied to `bandwidth` (e.g. one fit per `bandwidth/2`) — and only call the optimizer at those "anchor" targets.
- For the skipped targets in between, you'd get `a, m, sigma_w, sigma_s` by interpolating (linear or cubic spline in 1D; something like `scipy.interpolate.griddata` in 2D) between the nearest anchor fits, indexed by `target_x_all[t_i]` position.
- RMSE for interpolated targets would still be computed directly (cheap — it's just evaluating `waxman_smits_theta`, no optimization), so `rmse_arr` stays meaningful and comparable to today's convention.

**The tradeoff:** you're trading exactness for speed. Anywhere the parameter field is smooth (typical case, given the physical interpretation), interpolation is nearly indistinguishable from a full fit. Anywhere there's a sharp discontinuity in soil properties over a short spatial scale (e.g. a lithology boundary crossing the transect), interpolation would smear it — the anchors on either side would still capture it, but points near the boundary would look like a gradual transition instead of a step.

Two ways to control that risk:
1. Tie anchor spacing to `bandwidth` itself (e.g. `bandwidth / 2` or `bandwidth / 3`) so you're never interpolating over a distance larger than the kernel's own support — that keeps the approximation error bounded by roughly the same smoothness assumption the Gaspari-Cohn weighting already relies on.
2. Keep it opt-in via a stride/interpolation-factor argument, defaulting to fitting every target (today's exact behavior) so existing calls are unaffected unless explicitly requested.

Want me to implement this as an additional optional parameter (e.g. `target_stride` or `interp_factor`) on `calibrate_petrophysical_relationship_localized`, off by default?


# The paper's footprint issue

Section 2.3 states it directly: "ERT and water-content measurements differ significantly in spatial resolution and integration volume. ERT integrates over significantly larger volumes (≈m3) than water content sensors (≈cm3)." This is the exact ERT analogue of your EM-vs-TDR mismatch — same physical problem, different instrument.

Their fix — and its limits

They address only the horizontal side, the same way window_m does in your pipeline:

Before extracting resistivity at a probe location, they average ρ within a horizontal window centered on the sensor, to smooth lateral noise/artifacts.
The window width isn't fixed a priori — they swept it (1–5 m in 0.2 m steps) and picked whichever width minimized RMSE between ERT-derived and measured θ. Avignon ended up at 4.2 m, Larzac at 2.8 m — site-specific, empirically tuned, not closed-form.

This is conceptually closer to your --ws-localized-bandwidth-sweep (LOO-CV bandwidth selection) than to window_m itself — both are "sweep candidate widths, pick the one that minimizes validation error" solutions to a horizontal-footprint problem.

Critically, they explicitly did not solve the vertical (depth) side, and they say why in the Discussion:

the integration volume does not account for vertical resistivity variability. Adjusting the integration height around each sensor may be relevant. Yet, vertical averaging would also combine data with different sensitivities, which may not be desirable, as ERT sensitivity decreases and uncertainty increases with depth.

And they spell out why they couldn't just do it, unlike the horizontal case:

resistivity values could be integrated over a volume that varies with depth. However... the integration volume is affected not only by electrode spacing and array geometry, but also by the inversion mesh, inversion settings, and the applied regularization... and may therefore vary from one inversion model to another.

In other words: for ERT, there's no clean closed-form depth-sensitivity kernel to weight by — it depends on the mesh and regularization of that particular inversion, which changes across their 240-model ensemble. So they left it as an open limitation rather than a solved problem, and their eventual convergence criterion (RMSE at probe depth) is effectively hard-routed by depth too — same as your probe_layer_indices.

Does McNeill help in your case? Yes — and for a reason the paper couldn't use

This is actually the key asymmetry between ERT and EM here: EM has exactly the closed-form depth-sensitivity function the ERT paper says it lacks. em_sensitivity.py's own docstring makes this precise:

physically grounded per coil geometry instead of empirically tuned by RMSE sweep

McNeill's cumulative-response kernel (phi_hcp/phi_vcp) depends only on coil spacing/orientation and depth — it's a fixed analytic function of the instrument, not of the inversion mesh or regularization settings. That's structurally different from ERT's depth sensitivity, which (per the paper) is inversion-parameter-dependent and would need re-deriving per model. So where the ERT authors had to punt on depth-weighting because "no single physically-correct way" existed without re-deriving it per inversion, your EM case genuinely has that closed form already implemented and already computed once per run (EM_LAYER_WEIGHTS, logged in Step 1a).

That said, two caveats from the design doc still apply and are worth keeping in view given this paper:

McNeill/LIN validity has its own approximation regime (em_sensitivity.py's own caveat, degrading for CS/high-frequency), the EM equivalent of the ERT paper's "sensitivity decreases with depth" concern — it's a better-grounded approximation, not a perfect one.
The paper's CV/coverage-based reliability mapping (Step 5, their §2.5) is the same instinct as your MCMC posterior std / coverage map — worth noting they do eventually connect low-sensitivity zones to high uncertainty (their CV>100% hatching), which is exactly the "precision weighting" mechanism I flagged in the design doc's correction. Confirms that reframing was on the right track: low-sensitivity regions being unreliable, not "wrongly blended," is the paper's own framework too.

One more thing worth flagging: this paper is literally the "Loiseau et al., 2026" your run_pipeline.py docstring says the whole EA-EM method is "adapted from" — so Steps 2–5 in your pipeline are a direct port of their Steps 2–5 (petrophysical calibration → inverted model averaging → recalibration → selection/CV), and their explicit non-solution to the depth-footprint problem is effectively the gap your Step 5c diagnostic (and the Option B design) is trying to close that the source method never did. Want me to fold this paper's context (§2.3's horizontal-window precedent, the explicit "didn't solve depth" limitation, and their CV/uncertainty framing) into the design doc as grounding for why Option B is worth pursuing?
