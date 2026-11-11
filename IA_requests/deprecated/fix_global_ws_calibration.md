# Fix: dense-TDR WARN is not (mainly) about noise — it's the pooled, non-spatial Waxman-Smits calibration

## 0. Correction on my earlier noise theory

Your 1%-noise re-run of `2d_dense_tdr` proves noise wasn't the driver:

| | 3% noise | 1% noise |
|---|---|---|
| mean CV | 0.3899 | 0.3897 |

That's a rounding error, not a trend. I went back into the actual code (`em_methodology.py`, `4_run_select_assess.py`) instead of guessing again. Here's what's actually happening, with file/line references.

## 1. Confirmed root cause

**`calibrate_petrophysical_relationship`** (`em_methodology.py:1278-1777`) fits **one** `(a, m, σ_w, σ_s)` per horizon per ensemble member, pooling every TDR observation routed to that horizon — regardless of which `sample_positions` (i.e. which EM point / x-location) it came from:

```python
# em_methodology.py ~1518-1577
if probe_layer_indices is not None:
    indices = [i for i, li in enumerate(probe_layer_indices) if li == layer]
    ...
    theta_layer = theta_obs[indices]
    sigma_b_layer = sigma_b_full[indices]
```

`indices` only filters by **horizon**, never by **position**. All 10 spatial positions' observations for horizon *L* go into one fit for horizon *L*.

**`recalibrate_petrophysical_relationship`** (Step 4, line 1957) delegates straight into the same function:

```python
# em_methodology.py:2004-2018
stacked_averages = np.stack(averaged_models, axis=0)
return calibrate_petrophysical_relationship(models=stacked_averages, ...)
```

So Step 4 inherits the identical pooling — there's no spatial awareness anywhere in the calibration chain.

That single global fit is then applied uniformly to **every** measured point when building `theta_calc`:

```python
# 4_run_select_assess.py:294-312, _ec_grid_to_theta
for l_idx in range(n_layers):
    hf = fit.horizon_fits[l_idx]          # ONE param set, same for all 10 points
    ratio = hf.a * ec_col / (hf.sigma_w + hf.sigma_s)
    theta_flat[:, l_idx] = np.power(np.clip(ratio, 0.0, None), 1.0 / hf.m)
```

Your `ws_fixed_params = {'a': True, 'm': True, 'sigma_w': True, 'sigma_s': False}` means `a`, `m`, `σ_w` are already pinned to the same value everywhere — **`σ_s` is the only parameter that ever varies**, and it's exactly the one being forced into a single compromise value across all 10 positions.

## 2. Why this produces the specific WARN pattern (secondary, also code-confirmed)

Step 5's CV (`4_run_select_assess.py:194-248`) is the weighted std / weighted mean of the **raw top-K EC ensemble members themselves** — not a standard error of the mean, and not computed in θ-space (correcting what I said last time). The optimal `K` (`np.nanargmin(rmse_by_k)`, line 190) is whatever cumulative stack, pushed through the one pooled `σ_s`, best matches **all** TDR probes at once.

- Sparse run: 24 obs from 2 positions → best global `σ_s` found at a shallow stack, `K=4`.
- Dense run: 120 obs from 10 positions → reconciling one `σ_s` against 10 different true locations needs a much deeper stack, `K=32–35`.

A deeper top-K subset necessarily pulls in more, weaker-ranked, more mutually divergent EC members — which mechanically inflates the measured spread (CV), independent of `eca_noise_level`. That's why 3%→1% noise barely moved anything: **CV here tracks optimal-K depth, and optimal-K depth tracks how badly the pooled fit has to compromise spatially — not the noise level.**

*(I'm not proposing to change the CV formula itself — it may be exactly what Loiseau et al. Eq. 8 intends. Flagging it only so you know it's a second, related effect, not the thing to patch.)*

## 3. The fix: make σ_s position-aware, with shrinkage

Use the `sample_positions` value you already have (it's the EM-point/x-location index) as the natural zone key — no new spatial concept needed. Fit a local `σ_s` per position, shrunk toward the horizon-global `σ_s` in proportion to how much local data that position has (empirical-Bayes style), so sparsely-sampled positions don't overfit noise.

### 3a. Extend the result container

```python
# em_methodology.py — HorizonWaxmanSmitsParams
@dataclass
class HorizonWaxmanSmitsParams:
    horizon: int
    a: float
    m: float
    sigma_w: float
    sigma_s: float
    rmse: float
    predicted_theta: np.ndarray
    observed_theta: np.ndarray
    is_calibrated: bool = True
    sigma_s_by_position: dict[int, float] | None = None   # NEW — optional, None = old behaviour
```

### 3b. New opt-in args on `calibrate_petrophysical_relationship`

```python
def calibrate_petrophysical_relationship(
    models: np.ndarray,
    sample_positions: Sequence[int],
    sample_times: Sequence[int],
    theta_obs: np.ndarray,
    depths0: Sequence[float] | None = None,
    lab_p0: np.ndarray | None = None,
    probe_layer_indices: Sequence[int] | None = None,
    depths_are_fixed: bool = True,
    fixed_params: dict[str, bool] | None = None,
    warm_start_continuation: bool = False,
    probe_depths_reflect_fitted_interfaces: bool = False,
    theta_obs_std: float | Sequence[float] | None = None,
    spatial_calibration: bool = False,     # NEW, default False = fully backward compatible
    shrinkage_k: float = 4.0,              # NEW — higher = more pooling toward the global fit
    min_obs_per_position: int = 2,         # NEW — below this, position falls back to global sigma_s
) -> list[PetroFit]:
```

### 3c. After the existing global fit converges, add a per-position shrinkage pass

Insert this right after `a, m, sigma_w, sigma_s = popt` (around line 1757), still inside the `for layer in range(num_layers):` loop, using the **same** `indices` (or slice) already computed for `theta_layer`/`sigma_b_layer` so no new routing logic is needed:

```python
sigma_s_by_position = None
if spatial_calibration and is_calibrated_layer:
    # sample_positions restricted to this horizon's observations, same
    # order as theta_layer / sigma_b_layer.
    positions_layer = np.asarray(sample_positions)[indices] if probe_layer_indices is not None \
        else np.asarray(sample_positions)[start:end] if n_samples >= num_layers \
        else np.asarray(sample_positions)[indices]

    sigma_s_by_position = {}
    for pos in np.unique(positions_layer):
        mask = positions_layer == pos
        n_local = int(mask.sum())
        if n_local < min_obs_per_position:
            sigma_s_by_position[int(pos)] = sigma_s   # not enough data — use global
            continue

        theta_local = theta_layer[mask]
        sigma_b_local = sigma_b_layer[mask]

        # 1-D closed-form inversion of the WS model for sigma_s, since a/m/sigma_w
        # are already fixed at this point (matches your ws_fixed_params config,
        # where sigma_s is the only floating parameter). This needs no optimizer:
        #   theta = (a * sigma_b / (sigma_w + sigma_s)) ** (1/m)
        #   =>  sigma_s = a * sigma_b / theta**m  -  sigma_w
        theta_clip = np.clip(theta_local, 1e-6, 1.0)
        sigma_s_local_est = np.median(
            a * sigma_b_local / (theta_clip ** m) - sigma_w
        )
        sigma_s_local_est = float(np.clip(sigma_s_local_est, 0.0, 200.0))  # keep in WS_BOUNDS range

        # Empirical-Bayes shrinkage toward the horizon-global sigma_s.
        weight = n_local / (n_local + shrinkage_k)
        sigma_s_by_position[int(pos)] = weight * sigma_s_local_est + (1 - weight) * sigma_s
```

> Note: if `a`/`m`/`σ_w` are ever floated (not your current config), replace the closed-form step with a 1-D bounded `minimize` over `σ_s` only, holding `a, m, σ_w` at their already-converged global values — same shrinkage step afterward.

Then pass it through when building the result:

```python
horizon_fits.append(
    HorizonWaxmanSmitsParams(
        horizon=layer, a=float(a), m=float(m), sigma_w=float(sigma_w), sigma_s=float(sigma_s),
        rmse=rmse, predicted_theta=theta_pred, observed_theta=theta_layer,
        is_calibrated=is_calibrated_layer,
        sigma_s_by_position=sigma_s_by_position,   # NEW
    )
)
```

### 3d. Consume it in Step 5's EC→θ conversion

`_ec_grid_to_theta` (`4_run_select_assess.py:294-312`) currently collapses the points axis away before applying the fit. It needs the EM-point index preserved so it can look up the right local `σ_s`:

```python
def _ec_grid_to_theta(ec_grid: np.ndarray, fit) -> np.ndarray:
    # ec_grid shape: (..., n_points, n_layers) — keep n_points visible,
    # only flatten any leading time axis.
    *lead, n_points, n_layers = ec_grid.shape
    ec_flat = ec_grid.reshape(-1, n_points, n_layers)
    theta_flat = np.empty_like(ec_flat)

    for l_idx in range(n_layers):
        if l_idx >= len(fit.horizon_fits):
            theta_flat[:, :, l_idx] = 0.0
            continue
        hf = fit.horizon_fits[l_idx]
        for p_idx in range(n_points):
            sigma_s_p = (
                hf.sigma_s_by_position.get(p_idx, hf.sigma_s)
                if hf.sigma_s_by_position else hf.sigma_s
            )
            ec_col = np.clip(ec_flat[:, p_idx, l_idx], 1e-6, None)
            ratio = hf.a * ec_col / (hf.sigma_w + sigma_s_p)
            theta_flat[:, p_idx, l_idx] = np.power(np.clip(ratio, 0.0, None), 1.0 / hf.m)

    return theta_flat.reshape(ec_grid.shape)
```

EM points with **no** nearby TDR probe (e.g. positions 1–8 in your sparse run) simply fall back to `hf.sigma_s` (the global fit) via `.get(p_idx, hf.sigma_s)` — no behaviour change there.

### 3e. Wire the flag through `run_pipeline.py`

Wherever Step 2 and Step 4 currently call `EMmethod.calibrate_petrophysical_relationship(...)` / `EMmethod.recalibrate_petrophysical_relationship(...)`, add:

```python
spatial_calibration=True,
shrinkage_k=4.0,          # tune: lower = trust local data more, higher = trust global more
min_obs_per_position=2,
```

(I don't have `run_pipeline.py`'s exact call site content in front of me — if you paste that section I'll give you the exact diff instead of a description.)

## 4. What to check after applying this

1. Re-run `2d_dense_tdr` and look at Step 2's per-position `σ_s` spread — if the field really is heterogeneous, you should see `σ_s` trend or cluster meaningfully with x-position rather than being forced to one number.
2. Check whether optimal `K` in Step 5 comes back down closer to the sparse run's — it should, since each position no longer needs deep stacking to reconcile a compromise value.
3. Check whether mean CV drops below 0.30. If it does, the WARN should clear. If it drops but stays elevated, that's likely the secondary K-depth effect from §2, which is a separate (and possibly intentional, per the paper) question.
4. Try `shrinkage_k` at a couple of values (e.g. 1, 4, 10) — with only ~12 obs per position (4 depths × 3 times), shrinkage strength will matter a lot; too low risks overfitting per-position noise, too high just reproduces the current global behavior.

## 5. Files touched

- `em_methodology.py` — `HorizonWaxmanSmitsParams` dataclass, `calibrate_petrophysical_relationship` (Step 2; Step 4 inherits automatically via delegation)
- `4_run_select_assess.py` — `_ec_grid_to_theta`
- `run_pipeline.py` — pass the three new kwargs at both call sites (share the file if you want the exact diff)

Happy to turn this into an actual working diff against your files once I can see `run_pipeline.py`'s Step 2/4 call sites — I skipped that file this round since it's 200K and I wanted to confirm the mechanism first.
