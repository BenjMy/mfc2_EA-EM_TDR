# Adding a train/validation split to the EA-EM pipeline

## Why

Loiseau et al. select the ensemble stack depth `K` (Step 3) and report final
skill (Step 5/2.6) using **two different subsets** of the θ–ρ pairs:

- **Training subset** → fits the Waxman–Smits relationship (Step 2/4) and
  ranks/selects `K` by minimum RMSE (Step 3/5).
- **Validation subset** → held out entirely from fitting/selection, used only
  once at the end to report RMSE and r.

They get away with claiming "no obvious overfitting" because training and
validation RMSE end up close (Avignon: 3.24 vs 3.11 % vol.; Larzac: 2.25 vs
2.23 % vol.) — but they're explicit that at Larzac the split wasn't fully
independent (alternating sorted values, not separate probes/time periods),
because the network was too sparse to do better.

Right now `run_pipeline.py` passes the **same** `theta_obs` /
`sample_positions` / `sample_times` into `step2.run_petro` (fit + RMSE),
`step3.run_average` (ranking), `EMmethod.recalibrate_petrophysical_relationship`
(Step 4 fit + RMSE), and `step4_5.run_select_assess` (picks `K` from that
same RMSE). So `K` is selected and reported on the same data it was fit on —
exactly the thing the paper's split is designed to catch. With a TDR network
sparser than Larzac's, this is worth flagging clearly in the report even
before you fix it, and fixing it once you have more sensors.

## Design

1. Split once, right after the TDR/SWC observation arrays are assembled
   (before Step 2), into `train` and `val` index sets over the flattened
   `(position, time)` observation pairs.
2. Feed **train** into everything that currently drives fitting/selection:
   `step2.run_petro`, `step3.run_average`, `EMmethod.recalibrate_petrophysical_relationship`,
   `step4_5.run_select_assess`. This changes which `K` gets picked.
3. After `K` is selected, **re-run the forward Waxman–Smits conversion**
   (train-fitted parameters, applied to the selected `K`-averaged model) at
   the **val** positions only, compute RMSE/r there, and log both numbers in
   the report next to each other — mirroring their Fig. 5 layout.
4. Because your network is sparser than Larzac's (44/43 split there), don't
   pretend to full independence yet: use the same "sort by θ, take every
   other point" strategy they used, and say so in the report, exactly as
   they did. Switch to a genuinely independent split (by probe, or by
   time block) once you densify.

## New module: `train_val_split.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train/validation split for TDR-vs-ERT/EM calibration pairs.

Mirrors Loiseau et al. (2026), Sections 2.3 and 2.6: training data drives
petrophysical fitting AND ensemble-K selection; validation data is held out
and touched only once, at the end, to report an unbiased RMSE/r.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SplitIndices:
    train: np.ndarray   # int indices into the flattened obs arrays
    val: np.ndarray
    method: str
    independent: bool   # False if train/val share probes or time steps


def split_train_val(
    theta_obs: np.ndarray,
    sample_positions: np.ndarray,
    sample_times: np.ndarray,
    probe_layer_indices: np.ndarray | None = None,
    probe_id: np.ndarray | None = None,
    method: str = "alternate_sorted",
    seed: int = 0,
) -> SplitIndices:
    """
    Parameters
    ----------
    theta_obs, sample_positions, sample_times, probe_layer_indices
        Same flattened, same-length arrays already passed into
        ``step2.run_petro`` today (one entry per sensor-epoch pair).
    probe_id
        Optional per-observation probe/pit identifier. Required for
        ``method="by_probe"``. Without it, a fully independent split
        isn't possible and the function falls back to ``alternate_sorted``
        with a loud warning.
    method
        - "alternate_sorted"  : sort by theta_obs, take every other value
                                 (what the paper used at data-poor Larzac;
                                 NOT fully independent -- both subsets can
                                 contain the same probe/time step).
        - "by_probe"          : hold out whole probes for validation
                                 (needs probe_id). Fully independent.
        - "by_time_block"     : hold out whole time steps for validation
                                 (what the paper used at Avignon: "using
                                 data from every other time step").
                                 Fully independent w.r.t. time, not probes.
        - "random"            : uniform random 50/50 split, for comparison
                                 only -- not recommended as your primary
                                 split (doesn't guard against pseudo-
                                 replication the way the other methods do).
    """
    n = len(theta_obs)
    rng = np.random.default_rng(seed)

    if method == "by_probe":
        if probe_id is None:
            print(
                "[SPLIT][WARN] method='by_probe' requested but no probe_id "
                "given -- falling back to 'alternate_sorted' (not fully "
                "independent). Pass probe_id once you can identify which "
                "observation came from which physical probe/pit."
            )
            return split_train_val(
                theta_obs, sample_positions, sample_times,
                probe_layer_indices, probe_id, method="alternate_sorted",
                seed=seed,
            )
        uniq_probes = np.unique(probe_id)
        rng.shuffle(uniq_probes)
        half = max(1, len(uniq_probes) // 2)
        train_probes, val_probes = uniq_probes[:half], uniq_probes[half:]
        train = np.where(np.isin(probe_id, train_probes))[0]
        val = np.where(np.isin(probe_id, val_probes))[0]
        return SplitIndices(train, val, method, independent=True)

    if method == "by_time_block":
        uniq_times = np.unique(sample_times)
        train_times = uniq_times[0::2]
        val_times = uniq_times[1::2]
        train = np.where(np.isin(sample_times, train_times))[0]
        val = np.where(np.isin(sample_times, val_times))[0]
        return SplitIndices(train, val, method, independent=True)

    if method == "random":
        idx = rng.permutation(n)
        half = n // 2
        return SplitIndices(idx[:half], idx[half:], method, independent=False)

    # default: "alternate_sorted" -- matches the paper's Larzac fallback
    order = np.argsort(theta_obs)
    train = order[0::2]
    val = order[1::2]
    return SplitIndices(np.sort(train), np.sort(val), method, independent=False)


def subset(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Small helper so call sites read as ``subset(theta_obs, split.train)``."""
    return arr[idx] if arr is not None else None
```

## Wiring it into `run_pipeline.py`

### 1. Add a scenario/CLI switch

In the `argparse` block (near the other `--*` flags, e.g. next to
`--window-m`):

```python
parser.add_argument(
    "--train-val-split", default=None,
    choices=["alternate_sorted", "by_probe", "by_time_block", "random", "off"],
    help="Split TDR/SWC observations into train (fits WS relationship + "
         "selects K) and validation (reported once, untouched by "
         "selection) subsets, as in Loiseau et al. (2026) Sec. 2.3/2.6. "
         "'off' disables the split (legacy behaviour: same data fits and "
         "reports). Overrides 'train_val_split' in the scenario config.",
)
```

And resolve it the same way other overrides are resolved:

```python
TRAIN_VAL_SPLIT = (args.train_val_split
                    or getattr(cfg, "train_val_split", "off"))
```

### 2. Split right before Step 2

Insert this immediately before the existing `fits_step2 = step2.run_petro(...)`
call (around line 3249):

```python
import train_val_split as tvs

if TRAIN_VAL_SPLIT != "off":
    _probe_id = getattr(cfg, "probe_id", None)  # wire up if/when you track it
    split = tvs.split_train_val(
        theta_obs=theta_obs, sample_positions=sample_positions,
        sample_times=sample_times, probe_layer_indices=probe_layer_indices,
        probe_id=_probe_id, method=TRAIN_VAL_SPLIT,
    )
    reporter.log(
        f"[SPLIT] method={split.method}  train={len(split.train)}  "
        f"val={len(split.val)}  independent={split.independent}"
    )
    if not split.independent:
        reporter.log(
            "[SPLIT][CAVEAT] Train/val subsets are NOT fully independent "
            "(may share a probe or time step) -- current TDR network is too "
            "sparse for a cleaner split. Same caveat the paper makes for "
            "Larzac (Sec. 3.3.2/4). Densify TDR coverage, then switch to "
            "'by_probe' or 'by_time_block'."
        )

    theta_obs_fit          = tvs.subset(theta_obs, split.train)
    sample_positions_fit   = tvs.subset(sample_positions, split.train)
    sample_times_fit       = tvs.subset(sample_times, split.train)
    probe_layer_fit        = tvs.subset(probe_layer_indices, split.train) \
                              if probe_layer_indices is not None else None

    theta_obs_val          = tvs.subset(theta_obs, split.val)
    sample_positions_val   = tvs.subset(sample_positions, split.val)
    sample_times_val       = tvs.subset(sample_times, split.val)
    probe_layer_val        = tvs.subset(probe_layer_indices, split.val) \
                              if probe_layer_indices is not None else None
else:
    theta_obs_fit, sample_positions_fit, sample_times_fit, probe_layer_fit = (
        theta_obs, sample_positions, sample_times, probe_layer_indices
    )
    theta_obs_val = sample_positions_val = sample_times_val = probe_layer_val = None
```

### 3. Fit/select on `*_fit` only

Replace the arguments in the three existing calls (Steps 2, 4, and the
`step4_5.run_select_assess` call already ranks by whatever RMSE Step 4 used,
so fixing Steps 2 and 4 is enough — no signature change needed there):

```python
fits_step2 = step2.run_petro(
    models=models_clean, sample_positions=sample_positions_fit,
    sample_times=sample_times_fit, theta_obs=theta_obs_fit,
    depths0=DEPTHS0, output_dir=OUTPUT_DIR,
    probe_layer_indices=probe_layer_fit,
    depths_are_fixed=FIXED_DEPTHS if isinstance(FIXED_DEPTHS, bool) else all(FIXED_DEPTHS),
    fixed_params=WS_FIXED_PARAMS,
    probe_depths_reflect_fitted_interfaces=PROBE_ROUTING_USES_FITTED_DEPTHS,
    lab_p0=LAB_P0,
    theta_obs_std=_theta_obs_std_step2,
)
```

```python
fits_step4 = EMmethod.recalibrate_petrophysical_relationship(
    averaged_models=averaged_models, sample_positions=sample_positions_fit,
    sample_times=sample_times_fit, theta_obs=theta_obs_fit,
    depths0=DEPTHS0, probe_layer_indices=probe_layer_fit,
    depths_are_fixed=FIXED_DEPTHS if isinstance(FIXED_DEPTHS, bool) else all(FIXED_DEPTHS),
    fixed_params=WS_FIXED_PARAMS,
    probe_depths_reflect_fitted_interfaces=PROBE_ROUTING_USES_FITTED_DEPTHS,
    lab_p0=LAB_P0,
    theta_obs_std=_theta_obs_std_step2,
)
```

`step3.run_average` and `step4_5.run_select_assess` don't need signature
changes — they already just consume `fits_step2`/`fits_step4`, and those now
carry train-only RMSE, so `K` selection (Step 3/5) is automatically
train-only too.

### 4. One-shot validation pass after `K` is selected

Add this right after the existing `selected_output = step4_5.run_select_assess(...)`
block (around line 3590), gated on the split being active. This version is
verified against the actual `em_methodology.py`/`2_run_petro.py` you shared
(not the placeholder from the first draft of this doc — see the note at the
end of this section):

```python
if TRAIN_VAL_SPLIT != "off" and theta_obs_val is not None and len(theta_obs_val):
    reporter.step("Step 5b", "Held-out validation (Sec. 2.6)")

    # Forward-only evaluation, no re-fitting: EMmethod.calibrate_petro-
    # physical_relationship's `fixed_params` mechanism pins each named
    # parameter's L-BFGS-B bound to (lab_p0_value, lab_p0_value) -- see
    # em_methodology.py's docstring for calibrate_petrophysical_relationship
    # ("Implemented by pinning that parameter's L-BFGS-B bound to (p0, p0)").
    # Passing lab_p0=<train-fitted params> with ALL FOUR keys fixed makes
    # the optimizer a no-op and the returned PetroFit.rmse is simply the
    # RMSE of those exact train-fitted parameters evaluated against
    # theta_obs_val -- exactly the held-out check we want, using the
    # existing API rather than a new code path.
    #
    # PetroFit.params (a @property already defined on the dataclass) gives
    # exactly the (num_layers, 4) = [a, m, sigma_w, sigma_s] array lab_p0
    # expects -- no new plumbing needed.
    train_fitted_params = fits_step4[selected_output.K - 1].params  # (num_layers, 4)

    val_fit = step2.run_petro(
        # calibrate_petrophysical_relationship expects `models` with an
        # ensemble-size leading axis; averaged_models[K-1] has that axis
        # already collapsed (Step 3), so wrap it back into a length-1
        # ensemble with np.stack -- a bare list won't work since
        # run_petro/calibrate_petrophysical_relationship call `.ndim` on
        # `models` directly.
        models=np.stack([averaged_models[selected_output.K - 1]]),
        sample_positions=sample_positions_val, sample_times=sample_times_val,
        theta_obs=theta_obs_val, depths0=DEPTHS0, output_dir=OUTPUT_DIR,
        generate_plots=False,   # diagnostic-only pass, skip the Step 2 plot set
        probe_layer_indices=probe_layer_val,
        depths_are_fixed=FIXED_DEPTHS if isinstance(FIXED_DEPTHS, bool) else all(FIXED_DEPTHS),
        fixed_params={"a": True, "m": True, "sigma_w": True, "sigma_s": True},
        probe_depths_reflect_fitted_interfaces=PROBE_ROUTING_USES_FITTED_DEPTHS,
        lab_p0=train_fitted_params,   # pin to TRAIN-fitted values, not LAB_P0
        theta_obs_std=_theta_obs_std_step2,
    )[0]

    train_rmse = selected_output.rmse
    val_rmse   = val_fit.rmse
    reporter.log(
        f"[STEP 5b] Train RMSE (K={selected_output.K}, n={len(split.train)}): "
        f"{train_rmse:.5f} m³/m³"
    )
    reporter.log(
        f"[STEP 5b] Val   RMSE (K={selected_output.K}, n={len(split.val)}): "
        f"{val_rmse:.5f} m³/m³"
    )
    _gap = val_rmse - train_rmse
    reporter.metric("Train RMSE", f"{train_rmse:.5f} m³/m³")
    reporter.metric("Validation RMSE", f"{val_rmse:.5f} m³/m³")
    reporter.metric("Train↔Val RMSE gap", f"{_gap:+.5f} m³/m³")
    if abs(_gap) > 0.3 * train_rmse:
        reporter.log(
            "[STEP 5b][WARN] Val RMSE diverges >30% from train RMSE -- "
            "possible overfitting in K selection, or val subset too small "
            "to be representative. Treat the selected K with caution."
        )
```

One subtlety worth flagging: `train_fitted_params` (from `fits_step4`, i.e.
Step 4's *recalibration* of the K-averaged model) is what actually
determined K-selection, so it's the right thing to freeze for the held-out
check — not `fits_step2[...]` (Step 2's per-raw-member fit, a different
object indexed differently and not what Step 5 picked K on).

**Note on this section vs. the first draft:** the previous version of this
doc used a placeholder (`fits_step4[K-1].as_fixed_params()`, a method that
doesn't exist) because only `run_pipeline.py` had been shared at the time.
Having now seen `em_methodology.py`, the real mechanism is the
`lab_p0` + `fixed_params={"a": True, "m": True, "sigma_w": True, "sigma_s":
True}` combination shown above — `calibrate_petrophysical_relationship`
already supports exactly this "pin to a given value, don't refit" case
natively (it's how `WS_FIXED_PARAMS` partial-fixing works today), so no new
`apply_only` code path is needed after all.

### 5. Report header

In `PipelineReporter.__init__`, add the split method to the scenario table so
every report is self-documenting:

```python
self._write(f"| Train/val split | `{TRAIN_VAL_SPLIT}` |")
```

(pass `train_val_split=TRAIN_VAL_SPLIT` into the `PipelineReporter(...)`
constructor call alongside the existing `scenario_name`, `mode`, `cfg` args).

## Rollout order

1. Add `train_val_split.py`, wire the CLI flag, default `"off"` — zero
   behaviour change until someone opts in.
2. Run once with `--train-val-split alternate_sorted` on your current sparse
   network, purely to see the train/val RMSE gap and log the caveat — this
   alone lets you report an honest number even before the split is
   independent.
3. Once TDR coverage is denser, add `probe_id` tracking through the
   pipeline and switch to `--train-val-split by_probe` (or `by_time_block`
   if probes stay sparse but epochs pile up) for a genuinely independent
   split, matching what the paper did at Avignon.

## Verified against your actual source

The original version of this doc was written from `run_pipeline.py` alone
(`step2.py`, `step3.py`, `4_run_select_assess.py`, and `em_methodology.py`
weren't uploaded yet). Having now seen all of them, everything above checks
out against the real signatures with one correction (Step 4's validation
call, fixed above) and a few confirmations worth noting:

- `step2.run_petro` → `EMmethod.calibrate_petrophysical_relationship`'s
  actual signature matches what was assumed: `models, sample_positions,
  sample_times, theta_obs, depths0, probe_layer_indices, depths_are_fixed,
  fixed_params, probe_depths_reflect_fitted_interfaces, lab_p0,
  theta_obs_std` are all real keyword arguments.
- `PetroFit.rmse` is a `@property` = `np.nanmean` of per-horizon RMSE
  (horizons with no TDR coverage are `NaN` and excluded) — so `split.train`
  producing zero observations in some horizon just leaves that horizon's
  RMSE `NaN` and it's excluded from the mean automatically; it won't crash,
  but do check the Step 2/4 per-horizon log output after switching the
  split on, in case a horizon that used to be calibrated on train+val
  combined becomes uncalibrated (`is_calibrated=False`) on train alone —
  more likely at your sparser TDR sites than it was in the paper's Larzac
  case.
- `EMmethod.recalibrate_petrophysical_relationship` (Step 4) is a thin
  wrapper: it just stacks `averaged_models` and calls
  `calibrate_petrophysical_relationship` with the same keyword set — so
  everything above about Step 2 applies identically to Step 4, confirmed.
- `average_inverted_models` (Step 3) and `select_and_assess_model` /
  `run_select_assess` (Step 4/5) only ever consume `fit.rmse` from the fits
  you hand them — they have no independent access to `theta_obs`, so
  routing `*_fit` into Steps 2/4 really is sufficient to make `K` selection
  train-only; there's no second place downstream that silently touches the
  full dataset.
- `geometry_utils.py` (imported by `2_run_petro.py`, `1_run_model_generation.py`
  and others for `build_inversion_bnds` / `route_ec_to_horizons`) still
  wasn't uploaded, but nothing in this doc depends on it.
