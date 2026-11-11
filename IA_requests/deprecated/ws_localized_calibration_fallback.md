# Do unsupported cells fall back to the prior in the kernel-localized WS calibration?

**Short answer: yes.** In the bandwidth (`c`) localized Waxman-Smits calibration
(`em_methodology.run_petro_localized`, called from `2_run_petro.py` /
`run_pipeline.py` Step 2 LOCALIZED), any target location whose kernel weights
are all effectively zero is **not** interpolated, extrapolated, or given some
minimum-weight blend — it is set exactly to the fixed prior `p0`, and flagged
as *uncalibrated*. This happens at two levels: per-horizon (whole layer) and
per-target (single point within an otherwise-calibrated layer).

## 1. How "support" is defined

Spatial weighting uses the Gaspari-Cohn compact-support kernel
(`gaspari_cohn(dist, c)` in `em_methodology.py`, lines ~1802–1859):

- Weight is `1` at zero distance, tapers smoothly, and is **exactly `0`** once
  `|dist| > 2c` — unlike a Gaussian kernel, there is no infinite tail.
- `c → ∞` reproduces the single global fit (every observation gets weight 1).
- `c → 0` collapses to nearest-neighbour.

So "support" for a given target = the set of same-horizon TDR observations
whose distance to that target is `≤ 2·bandwidth`. Anything farther away
contributes **zero**, not a small number.

## 2. Per-horizon fallback (no probes at all in that layer)

Before any spatial weighting, horizon routing is a hard depth partition
(`probe_layer_indices` or the strided heuristic). If a horizon ends up with
**zero** TDR probes routed to it at all:

```python
if not indices:
    print(
        f"[STEP 2 LOCALIZED][WARN] No TDR probes mapped to layer "
        f"{layer}. Every target in this horizon falls back to "
        f"the prior (uncalibrated)."
    )
    horizon_fits.append(LocalizedHorizonWaxmanSmitsParams(
        ...
        a=np.full(n_targets, p0[0]),
        m=np.full(n_targets, p0[1]),
        sigma_w=np.full(n_targets, p0[2]),
        sigma_s=np.full(n_targets, p0[3]),
        rmse=np.full(n_targets, np.nan),
        is_calibrated=np.zeros(n_targets, dtype=bool),
        bandwidth=bandwidth,
    ))
    continue
```

Every target in that horizon gets `p0` (the fixed prior — `lab_p0` if given,
else the default WS prior), `rmse = NaN`, and `is_calibrated = False`. This
is independent of bandwidth; it's about probe **coverage**, not distance.

## 3. Per-target fallback (probes exist in the horizon, but none near this point)

This is the actual bandwidth-driven case. For each target location, the code
combines the (optional) inverse-variance base weight with the spatial kernel
weight and checks whether *anything* survives:

```python
spatial_w = gaspari_cohn(dist, bandwidth)
combined_w_raw = base_weights * spatial_w

if np.sum(combined_w_raw) < 1e-8:
    # No observation of this horizon falls within the
    # kernel's support (2*bandwidth) of this target -- same
    # spirit as "no TDR probes mapped to this horizon", just
    # per-target instead of per-horizon. Fall back to prior.
    a_arr[t_i], m_arr[t_i], sigma_w_arr[t_i], sigma_s_arr[t_i] = p0
    continue
```

If `sum(combined_w_raw)` is (numerically) zero — i.e. every same-horizon
observation is farther than `2c` from this target — the target's `[a, m,
sigma_w, sigma_s]` are set **directly and exactly to `p0`**, `is_calibrated`
stays `False`, and no weighted fit is attempted. There is no partial
weighting, smoothing, or graceful degradation as support thins out right up
to the cutoff — it's a hard binary: *any* nonzero combined weight → full
weighted fit; *zero* combined weight → prior, verbatim.

When at least one observation *is* within support, weights are renormalized
to mean 1 (`combined_w = combined_w_raw / np.mean(combined_w_raw)`) before
being passed to `_fit_ws_horizon_weighted`, so bandwidth choice doesn't
require re-tuning the regularization strength.

## 4. Same rule inside the LOO-CV bandwidth sweep

`sweep_localization_bandwidth` (used by `run_bandwidth_sweep` / the
`--ws-localized-bandwidth-sweep` diagnostic) applies the identical rule per
leave-one-out fold, and explicitly counts how often it happens:

```python
if np.sum(combined_w_raw) < 1e-8:
    a, m, sigma_w, sigma_s = p0
    n_fallback += 1
else:
    combined_w = combined_w_raw / np.mean(combined_w_raw)
    a, m, sigma_w, sigma_s, _ = _fit_ws_horizon_weighted(...)
```

This `n_fallback` count is surfaced in the pipeline report table as **"Folds
fell back to prior"** next to each candidate bandwidth `c`, so you can see
directly how many folds had no support at a given `c` — a large fraction of
fallback folds at your chosen bandwidth is a signal that `c` is too small
relative to probe spacing, and the "recommended" bandwidth from the sweep is
chosen partly by penalizing this via the pooled LOO-CV RMSE (fallback
predictions using `p0` typically fit their held-out point worse, which drives
RMSE up for too-small `c`).

## 5. Practical implications

- **This diagnostic never fails silently** for unsupported cells: `rmse =
  NaN` and `is_calibrated = False` mark exactly which targets/horizons used
  the prior rather than a genuine fit, so downstream plots (e.g.
  `plot_ws_params_xz`, `plot_calibration_density_kernel`) can distinguish
  calibrated vs. prior-only regions.
- **Bandwidth choice controls how much of the domain falls back.** Too small
  a `c` (much less than the spacing between adjacent TDR/SWC probes) means
  most of the domain reverts to the prior; the docstring's rule of thumb is
  to pick `c` at least ~half the spacing between your closest adjacent
  probes. Too large a `c` washes out spatial variation and approaches the
  single global fit.
- **The fallback prior is fixed, not adaptive** — it's `lab_p0` (or the
  default WS prior) for that horizon, the same value used to seed the
  un-localized global fit, not some spatially-weighted extrapolation from
  the nearest (but out-of-support) probe.
- This behavior only affects the **diagnostic sibling** output
  (`petro_calibration_localized.png`, the localized `LocalizedPetroFit`
  objects, and their plots) — it never changes `fits_step2` (the real Step 2
  ensemble calibration) or anything in Steps 3–5.
