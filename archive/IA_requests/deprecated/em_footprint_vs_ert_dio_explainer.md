# Why ERT's coverage/DOI tools don't transfer directly to EM, and why depth-varying integration is hard

## 1. EM cumulative sensitivity functions (McNeill-type) vs. ERT coverage/DOI

### What ERT's "coverage" and "DOI" actually measure

In ERT, the paper's coverage/DOI indicators come **out of the inversion itself**:

- **Coverage** is the column-sum of the sensitivity (Jacobian) matrix, accumulated iteration by iteration during the Gauss-Newton inversion. It tells you how much each model cell influenced the measured data, *given the current model and regularization*.
- **DOI (Depth of Investigation index)** compares two inversions run from different reference models and looks at where they diverge, again *after* inversion.

Both are therefore:
- **Inversion-dependent** — they change with regularization, mesh, starting model, and data noise, not just with survey geometry.
- **Not a fixed physical property of the sensor** — you cannot precompute a coverage map before you invert; it's an output of the algorithm, specific to that run.

### What EM sensitivity functions are instead

Electromagnetic induction (EMI) instruments like the GEM-2 don't require this kind of inversion-derived indicator at all, because for the low-induction-number approximation, the depth response is **known analytically before any inversion happens**. McNeill (1980) derived the cumulative response function directly from Maxwell's equations for a given coil configuration:

- **Horizontal coplanar (HCP) coils**: 
  φ(z) = 1 / (4z² + 1)^0.5

- **Vertical coplanar (VCP) coils**:
  φ(z) = (4z² + 1)^0.5 − 2z

where z is depth normalized by the intercoil spacing. These give you the **cumulative fraction of the total secondary field response** that comes from all material above depth z — a closed-form curve, not something you extract from an inversion.

### Why this matters practically for your pipeline

1. **You can compute it once per coil geometry/frequency, offline.** Your `param_grid` already sweeps `forwardModel: [FSlin, CS]` — each forward model implies its own depth sensitivity shape. You can attach a McNeill-type (or full electromagnetic, frequency-dependent — FSlin/CS are more general than the simple LIN approximation McNeill used) cumulative sensitivity curve to each forward model choice and get a depth-weighting function *before* Step 1 even runs, rather than deriving it from ensemble spread after Step 5.

2. **It's the natural EM analogue of the paper's "integration volume" problem.** Where the ERT paper struggles because coverage/DOI are inversion-dependent and don't cleanly define a fixed "footprint," EM gives you something closer to a fixed footprint by construction (for a homogeneous or layered half-space assumption) — this is actually an advantage you should exploit rather than reproduce their coverage-based workaround.

3. **It plugs directly into your objective (ii).** Instead of comparing your discretized layer-θ values against TDR at a nominal depth, you can compute a depth-weighted "effective EM θ" using φ(z) as a kernel over your layer profile, and compare *that* to the TDR point value. The mismatch between discretized-layer θ and sensitivity-weighted θ **is** your footprint discrepancy metric — grounded in a real physical response function rather than an empirical average.

4. **Caveat**: McNeill's original curves assume the low-induction-number (LIN) limit and a purely conductive half-space — they degrade at higher conductivities/frequencies where your CS ("cumulative sensitivity"?) or full nonlinear forward model departs from LIN behavior. For a rigorous footprint you'd want the sensitivity kernel consistent with whichever forward model (FSlin vs CS) generated the ensemble member, not a generic textbook McNeill curve bolted on afterward.

---

## 2. Why the paper calls depth-varying integration volumes "too complex" — and what that complexity actually consists of

The relevant passage:

> "Because ERT resolution decreases with depth, the effective investigation volume increases. To counter this fact, resistivity values could be integrated over a volume that varies with depth. However, this integration introduces additional complexity. The integration volume is affected not only by electrode spacing and array geometry, but also by the inversion mesh, inversion settings, and the applied regularization, and may therefore vary from one inversion model to another."

Unpacking each source of complexity they're pointing to:

### a. The footprint isn't fixed — it grows with depth
In ERT, near-surface resolution is sharp (small support volume near the electrodes) and gets progressively blurrier with depth, because current spreads out and sensitivity to any single cell decreases. So a "correct" depth-varying integration window would need to be **narrow near the surface and wide at depth** — the opposite of the fixed horizontal window they actually used (2.1–2.8 m, constant at all depths).

### b. The footprint shape depends on things that change between ensemble members
This is the real killer for their ensemble approach specifically. Unlike the EM case above, ERT's effective resolution volume at a given depth is not a closed-form geometric function of electrode spacing alone — it also depends on:
- **the mesh** (cell size/shape near that location),
- **inversion settings** (which norm — L1/L2, damping),
- **the regularization strength (λ)** and **anisotropy weighting (zweight)**.

Their whole method runs **240 different inversions with different λ, zweight, norm, and reference-model choices** specifically to build the ensemble. That means the "correct" depth-varying integration volume would be **different for every one of the 240 models** — you'd need to recompute a custom resolution kernel per model, per depth, per cell, before you could even extract comparable ρ–θ pairs. That's not a fixed preprocessing step anymore; it becomes part of the inversion loop itself, multiplying the computational and implementation burden by ~240×.

### c. It would entangle two things they deliberately kept separate
Their whole design keeps **Step 1 (inversion/geometry)** and **Step 2 (petrophysical extraction)** as separable stages — you invert once per parameter set, then extract at fixed points afterward. A depth-varying, inversion-dependent integration volume would collapse that separation: extraction could no longer be a simple "read ρ at this x,y,z" operation identical across all 240 models — it would need model-specific logic. That undermines the simplicity/automatability they explicitly value (the whole point of EA-ERT is to be "automatable").

### d. Why they chose the simpler fallback instead
Given (a)–(c), they picked a **compromise they could keep constant across all models**: a fixed horizontal averaging window, empirically tuned once (by sweeping 1–5 m and picking the best RMSE), applied identically regardless of depth or inversion parameters. It's not physically correct at depth, but it's tractable, uniform across the ensemble, and doesn't reintroduce per-model complexity into what's supposed to be a lightweight post-processing step.

### Why your EM case is comparatively easier here
This connects back to point 1 above: because EM sensitivity (McNeill-type or forward-model-consistent) is a **closed-form function of geometry/frequency**, not an emergent property of an inversion mesh and regularization choice, you *can* afford a depth-varying integration/weighting scheme without the combinatorial blow-up the ERT authors were avoiding. Your Step 1 ensemble (α, forwardModel, ROPE) doesn't reshape the sensitivity kernel the way ERT's mesh/λ/norm choices do — so a depth-weighted footprint correction is a much more tractable addition to your pipeline than it would have been to theirs.

---

## 3. Implementing this in your pipeline

### What your pipeline does today

Your GEM-2 coil channels are named like `HCP1.66f425h1_quad` … `HCP1.66f92775h1_quad`
(`run_pipeline.py`, coil-alias table ~line 441): **horizontal coplanar (HCP)**
geometry, **1.66 m** intercoil spacing, six frequencies (425 Hz–92,775 Hz) at
that *same fixed spacing*. In the McNeill/LIN approximation, depth response
depends only on geometry (spacing + orientation), not frequency — so all six
channels share one φ(z) curve; the frequency sweep buys you conductivity
info, not depth info, unless you're relying on `forwardModel="CS"` to go
beyond LIN.

TDR/EM matching currently happens in `EMmethod.extract_sample_positions_from_survey`
(called ~line 2916), which:
- matches each probe **horizontally** using `WINDOW_M` (unaffected by
  anything below — that axis is already handled),
- routes each probe **vertically** to a single `layer_index` via `depths0`
  (nearest/containing layer — a hard assignment, the EM analogue of "pick a
  fixed horizontal window" rather than a depth-weighted one).

That `layer_index` (`probe_layer_indices`) is what Steps 2/4 use to compare
one layer's θ_calc against one TDR reading. Nothing currently asks "how much
of what the coil actually measured came from other layers too?"

*(Verified against `em_methodology.py`: `layer_index_from_depth` confirms
`depths0` is exactly the boundary-depths convention assumed below — "horizon
0 spans `[0, depths0[0])` ... the last horizon spans `[depths0[-1], +inf)`"
— so `layer_weights()`'s `len(depths0)+1` output (finite layers + an
implicit half-space) lines up with `num_layers` exactly as designed, no
off-by-one to fix.)*

### Step A — add `em_sensitivity.py` (closed-form kernel, no dependency on Steps 1-5 internals)

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
McNeill (1980) cumulative EM response kernels + depth-weighted layer->point
comparison. Closed-form (computed once, offline) -- the EM analogue of the
ERT paper's fixed horizontal integration window (Sec. 2.3), but physically
grounded per coil geometry instead of empirically tuned by RMSE sweep.

Caveat (see explainer Sec. 1, point 4): valid in the low-induction-number
(LIN) limit over a layered/half-space assumption. Treat results from
forwardModel="CS" (or the higher GEM-2 frequencies, where LIN degrades) as
approximate -- a diagnostic, not a re-derivation of the inversion.
"""
from __future__ import annotations

import re
import numpy as np


def parse_coil_spacing_m(coil_name: str) -> tuple[float | None, str | None]:
    """
    'HCP1.66f425h1_quad' -> (1.66, 'HCP').
    Returns (None, None) with a warning if the name doesn't match this
    pipeline's coil-naming convention (see the coil alias table in
    run_pipeline.py ~line 441) -- update the regex if that changes.
    """
    m = re.match(r"([A-Za-z]+)([\d.]+)f", coil_name)
    if not m:
        print(f"[EM_SENS][WARN] Could not parse coil spacing from '{coil_name}'")
        return None, None
    orientation, spacing = m.group(1).upper(), float(m.group(2))
    return spacing, orientation


def phi_hcp(z_norm: np.ndarray) -> np.ndarray:
    """McNeill (1980) cumulative response fraction ABOVE depth z, HCP coils."""
    return 1.0 / np.sqrt(4.0 * z_norm**2 + 1.0)


def phi_vcp(z_norm: np.ndarray) -> np.ndarray:
    """McNeill (1980) cumulative response fraction ABOVE depth z, VCP coils."""
    return np.sqrt(4.0 * z_norm**2 + 1.0) - 2.0 * z_norm


def layer_weights(
    depths0: list[float], coil_spacing_m: float, orientation: str = "HCP",
) -> np.ndarray:
    """
    depths0 : layer BOTTOM depths [m] -- same convention as this pipeline's
              DEPTHS0 -- plus an implicit half-space below depths0[-1].

    Returns an array of length len(depths0)+1 (finite layers + half-space)
    summing to 1: weight_i = fraction of the coil's total secondary-field
    response attributable to layer i, given phi(0)=1 (all response from
    the surface down) and phi decreasing monotonically with depth.
    """
    phi = phi_hcp if orientation == "HCP" else phi_vcp
    edges = np.array([0.0] + list(depths0)) / coil_spacing_m
    phi_edges = phi(edges)
    w_layers = -np.diff(phi_edges)        # response contributed by each finite layer
    w_half_space = phi_edges[-1]          # remainder, from below depths0[-1]
    weights = np.append(w_layers, w_half_space)
    return weights / weights.sum()        # renormalize (guards float error only)


def effective_theta(theta_layers: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    theta_layers : (..., n_layers) array of per-layer theta_calc (last axis
                   = layer, matching `weights`'s length).
    weights      : (n_layers,), from layer_weights(), summing to 1.

    Returns the depth-weighted "effective EM theta" seen by the coil at
    each remaining index (e.g. per position, per time step) -- the number
    to compare against a point TDR reading instead of a single hard-routed
    layer's theta_calc.
    """
    return np.tensordot(theta_layers, weights, axes=([-1], [0]))
```

### Step B — compute the weights once, right after `DEPTHS0` is resolved

Around line 1339 in `run_pipeline.py`, immediately after `DEPTHS0 = cfg.depths0`
(or its synthetic-mode fallback ~line 1349):

```python
import em_sensitivity as emsens

# All six GEM-2 channels share one spacing (1.66 m, HCP) in this pipeline's
# coil set, so one weight vector covers every frequency in PARAM_GRID.
# If you ever mix coil geometries/spacings, compute one weight vector per
# distinct (spacing, orientation) pair instead of a single EM_LAYER_WEIGHTS.
_COIL_SPACING_M, _COIL_ORIENTATION = 1.66, "HCP"   # from the HCP1.66f... coil names
EM_LAYER_WEIGHTS = emsens.layer_weights(
    DEPTHS0, coil_spacing_m=_COIL_SPACING_M, orientation=_COIL_ORIENTATION,
)
reporter_pending_log = (
    f"[EM_SENS] McNeill {_COIL_ORIENTATION} weights (spacing={_COIL_SPACING_M} m) "
    f"over layers {DEPTHS0} + half-space: "
    f"{np.round(EM_LAYER_WEIGHTS, 4).tolist()}"
)
# (log this once `reporter` exists, a few lines below where DEPTHS0's
# resolution is already logged)
```

Add a scenario/CLI switch alongside the others (near `--window-m`), default
off so this is purely additive until you opt in:

```python
parser.add_argument(
    "--em-footprint-diagnostic", action="store_true",
    help="Compute a McNeill-type depth-weighted 'effective EM theta' per "
         "probe (explainer Sec. 1/3) and report its discrepancy against "
         "theta_calc at the hard-routed layer_index and against theta_obs. "
         "Diagnostic only -- does not change Step 2/4 calibration or Step "
         "3/5 K-selection.",
)
```

### Step B (addendum) — real vs. synthetic coil sets aren't the same geometry

`em_methodology.py`/`0_generate_synthetic_data.py` show your **synthetic**
scenarios use `COILS_LIST = ['VCP0.5', 'HCP0.5', 'VCP1.0', 'HCP1.0', 'VCP1.5',
'HCP1.5']` — three spacings (0.5/1.0/1.5 m) × both orientations, six distinct
geometries feeding one inversion. That's different from the real GEM-2 data
path (`run_pipeline.py`'s coil-alias table), which is one geometry (HCP,
1.66 m) at six frequencies. So a single `EM_LAYER_WEIGHTS` vector is only
correct for the real-data branch; for synthetic scenarios you have six
candidate φ(z) curves feeding the same inverted layer stack, and there's no
single physically-correct way to collapse them into one "the" footprint
kernel — the inversion itself already blends their information nonlinearly.

For the diagnostic below, two honest options, pick based on what you're
trying to check:
- **Representative-coil mode**: compute `EM_LAYER_WEIGHTS` from whichever
  coil in the survey has spacing closest to your shallowest layer thickness
  (the one you most care about resolving) — a deliberately narrow,
  single-purpose diagnostic, not a claim about the whole array's footprint.
- **Multi-coil average mode**: compute one weight vector per `(spacing,
  orientation)` pair actually present in `param_grid`/`COILS_LIST`, then
  average them (equal-weighted, or weighted by however you weight coils in
  the inversion's misfit if that's exposed) — an approximation, and worth
  labelling as one in the report, since it isn't the array's true joint
  sensitivity.

Don't silently default to "just pick HCP1.66" for synthetic runs; that's the
real-data geometry, not the synthetic one, and the two will disagree.

### Step C — report the footprint-discrepancy metric (diagnostic, non-invasive)

`4_run_select_assess.py::run_select_assess` already contains the exact
"hard-routed" computation this diagnostic compares against — its own probe
loop (search `resolved_layer_indices` / `ec_predicted` in that file) pulls
each probe's EC from its own horizon only, then calls
`EMmethod.ec_to_water_content(ec_predicted, current_fit,
probe_layer_indices=resolved_layer_indices)`. Rather than re-deriving that
logic, the diagnostic below reuses `selected_output.theta_calc` (the
already-computed output of exactly that loop, evaluated at `K` for the
*whole grid*, not just probe points — see `SelectionOutput.theta_calc` in
`4_run_select_assess.py`) and just adds the McNeill-weighted comparison
alongside it.

One shape correction from the previous draft: `SelectionOutput.theta_calc`
is **always 3-D**, `(n_time, num_measured_points, num_layers)` — even for a
single-time survey, `4_run_select_assess.py` explicitly does
`theta_calc = theta_calc[np.newaxis, ...]` when `theta_calc.ndim == 2`. So
there's no `timelapse`-conditional 2-D branch to handle; `t_idx = 0` covers
the single-time case directly.

Insert after the existing `selected_output = step4_5.run_select_assess(...)`
block (~line 3590):

```python
if getattr(args, "em_footprint_diagnostic", False):
    reporter.step("Step 5c", "EM footprint discrepancy (McNeill depth weighting)")

    # theta_calc is ALWAYS (n_time, num_measured_points, num_layers) --
    # SelectionOutput guarantees this (see 4_run_select_assess.py, the
    # `if theta_calc.ndim == 2: theta_calc = theta_calc[np.newaxis, ...]`
    # normalisation right before it returns SelectionOutput).
    theta_calc = selected_output.theta_calc
    eff_theta = emsens.effective_theta(theta_calc, EM_LAYER_WEIGHTS)
    # eff_theta shape: (n_time, num_measured_points) -- one depth-weighted
    # "what the coil actually saw" value per position per time step.

    discrepancies = []
    for i, (pos, li, obs) in enumerate(zip(sample_positions, probe_layer_indices, theta_obs)):
        t_idx = sample_times[i] if timelapse else 0
        hard_routed_theta = theta_calc[t_idx, pos, li]
        weighted_theta    = eff_theta[t_idx, pos]
        discrepancies.append({
            "probe": i, "obs": obs,
            "hard_routed": hard_routed_theta, "weighted": weighted_theta,
            "hard_routed_err": hard_routed_theta - obs,
            "weighted_err": weighted_theta - obs,
        })

    _hr_rmse = np.sqrt(np.nanmean([d["hard_routed_err"]**2 for d in discrepancies]))
    _w_rmse  = np.sqrt(np.nanmean([d["weighted_err"]**2     for d in discrepancies]))
    reporter.log(
        f"[STEP 5c] RMSE vs TDR — hard-routed (nearest layer): {_hr_rmse:.5f} m³/m³  "
        f"|  depth-weighted (McNeill {_COIL_ORIENTATION}): {_w_rmse:.5f} m³/m³"
    )
    reporter.metric("Footprint RMSE — hard-routed layer", f"{_hr_rmse:.5f} m³/m³")
    reporter.metric("Footprint RMSE — McNeill-weighted",  f"{_w_rmse:.5f} m³/m³")
    if _w_rmse < _hr_rmse:
        reporter.log(
            "[STEP 5c] Depth-weighted comparison fits TDR better than the "
            "hard-routed layer -- the current nearest-layer probe routing "
            "may be discarding real cross-layer signal contribution. "
            "See Step D for what it would take to fold this into "
            "calibration itself, and note the caveats there before doing so."
        )
    else:
        reporter.log(
            "[STEP 5c] Hard-routed layer already matches TDR at least as "
            "well as the McNeill-weighted blend -- either your layers are "
            "thick relative to the coil spacing used (weight concentrates in "
            "one layer anyway) or CS/high-frequency LIN departure (Sec. 1, "
            "point 4) is degrading the kernel here. Check EM_LAYER_WEIGHTS "
            "-- if one layer already carries >90% of the weight, this "
            "result is expected and the current routing is fine."
        )
```

A note on what "weighted" is actually weighting: `theta_calc`'s per-layer
values already went through `_ec_grid_to_theta`, i.e. each layer's own EC
converted via *that layer's own horizon's* fitted `[a, m, σw, σs]`
(`4_run_select_assess.py`). So `effective_theta()` here blends **already
horizon-converted θ values**, not raw EC — which sidesteps a real ambiguity
(mixing EC across horizons with different WS parameters has no single
correct conversion) at the cost of blending in θ-space rather than the
physically primary EC-space McNeill's φ(z) was derived for. That's a
defensible simplification for a diagnostic; it's not a substitute for doing
this properly inside the EC→θ conversion, which is what Step D would
require.

### Step D — optional, deeper integration (only after Step C shows it's worth it, and only with eyes open on the cost)

Having now seen `em_methodology.py`'s actual fitting loop
(`calibrate_petrophysical_relationship`, the `_log_ratio_objective` closure
around line 1655), folding the McNeill kernel into calibration itself is a
bigger change than "pass `EM_LAYER_WEIGHTS` into the extraction step" — the
architecture is more constrained than that:

- Each horizon is fit **independently**: the objective closure captures
  `sigma_b_layer` — the EC values already routed to *that one horizon only*
  — and fits `[a, m, σw, σs]` against `theta_layer` for that horizon alone
  (`data_misfit = np.sum(obs_weights * (log_ert - log_ws) ** 2)` per
  horizon, in a `for layer in range(num_layers):` loop). There is currently
  no code path where one observation's misfit term involves more than one
  horizon's parameters at once.
- A McNeill-blended target (one probe's θ_obs compared against a
  depth-weighted combination of *multiple* horizons' EC) doesn't fit this
  shape directly — whose `[a, m, σw, σs]` would the blended prediction use?
  There's no single horizon to attribute it to.

Two ways to reconcile that, in increasing order of invasiveness:

1. **Weight-aware routing, still single-horizon** (small change): instead of
   `layer_index_from_depth`'s hard threshold (`np.searchsorted`), route each
   probe to its **dominant** layer under `EM_LAYER_WEIGHTS` — i.e.
   `np.argmax(weights)` for that probe's local layer stack — rather than the
   geometric containing layer. This changes *which* horizon gets which
   probe's observation in edge cases (a probe just above a boundary where
   the coil's response is still mostly from the layer below), but keeps the
   existing one-observation-one-horizon fitting loop completely unchanged.
   Cheapest option; only helps probes near layer boundaries, does nothing
   for probes where several layers genuinely each contribute non-trivially.
2. **Multi-horizon blended objective** (real redesign): extend
   `_log_ratio_objective` so a probe's predicted θ is
   `Σ_layer weight[layer] * waxman_smits_theta(sigma_b[layer], *params[layer])`
   summed over the layers the probe's kernel actually touches, and let the
   optimizer fit **all horizons jointly** for that probe's contribution
   instead of one at a time. This is a genuine change to the optimization
   problem (multi-horizon coupling, more parameters entangled with fewer
   observations — the opposite direction from what `fixed_params` was added
   to help with, per that parameter's own docstring: "with only ~2
   observations per horizon, fitting all 4 parameters is underdetermined").
   Given how sparse your TDR network already is, option 2 likely makes the
   per-horizon underdetermination *worse*, not better, unless you also
   densify TDR coverage or lean harder on `fixed_params`/`lab_p0` priors to
   compensate.

Given that trade-off, option 1 is the more realistic next step if Step C's
diagnostic shows a persistent gap — option 2 is worth doing only once your
TDR network is dense enough that per-horizon fits are already
well-constrained on their own (the same densification threshold that shows
up in the train/val split doc's independent-split discussion).

### Rollout order

1. Add `em_sensitivity.py`, compute `EM_LAYER_WEIGHTS` once (mind the
   real-vs-synthetic coil-geometry distinction above), log it — zero
   pipeline behaviour change.
2. Run with `--em-footprint-diagnostic` on a real-data scenario and look at
   the Step 5c RMSE comparison. If your layers are much thicker than the
   coil spacing, expect the two numbers to be close (weight concentrates
   near the top layer regardless) — the diagnostic mainly earns its keep
   when a TDR probe sits near a layer boundary or your shallow layers are
   thin relative to coil spacing.
3. If Step C shows a persistent gap, try Step D option 1 (weight-aware
   dominant-layer routing) first — it's a small, low-risk change. Only
   consider option 2 (joint multi-horizon fitting) once TDR density can
   support it; it's a real redesign of the calibration objective, not a
   parameter tweak.
