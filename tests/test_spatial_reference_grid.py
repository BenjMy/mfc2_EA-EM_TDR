#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_spatial_reference_grid.py
===============================
pytest verification (no real `emagpy` install required: tests/conftest.py
stubs it when absent) for
the "spatial reference-grid matching, not row trimming" fix:

  1. geometry_utils.match_epochs_to_reference_grid — the core algorithm.
  2. The remap-with-NaN-fill logic run_pipeline.py Step 1 now uses instead
     of `arr[:min_pts]` (reproduced here rather than imported, since it is
     defined as a closure inside run_pipeline.main() and not separately
     importable -- this test recreates it verbatim from the patched source
     to check its behaviour in isolation).
  3. The NaN-safety follow-on fixes this made necessary:
       - em_methodology.calibrate_petrophysical_relationship drops NaN
         observations instead of poisoning a whole horizon's fit.
       - em_methodology.average_inverted_models / plot_step1_ensemble_
         single_file tolerate NaN cells.
       - step4_select.py's rmse_by_k / cv_map computations use
         NaN-aware reductions and preserve NaN instead of miscoding it as
         a false "CV = 0.0 / zero uncertainty".

Run with:  python3 -m pytest tests/test_spatial_reference_grid.py -v
"""

from __future__ import annotations

import sys
import numpy as np

import geometry_utils


def check(name: str, condition: bool, detail: str = "") -> None:
    """Assertion with a readable message (pytest reports the failing check)."""
    assert condition, f"{name}  {detail}"


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ===========================================================================
# 1. geometry_utils.match_epochs_to_reference_grid — core algorithm
# ===========================================================================

def test_reference_grid_basic_union():
    section("match_epochs_to_reference_grid: basic union, no overlap")

    # Epoch 0: 3 points along x=0,10,20 (y=0)
    # Epoch 1: 2 DIFFERENT points along x=5,15 (y=0), none within tolerance
    #          of epoch 0's points -> every point should survive in the
    #          union reference grid (this is the "keep all the points"
    #          requirement -- the old row-count trim would have thrown
    #          away everything beyond min(3, 2) = 2 points per epoch).
    epoch0_x = np.array([0.0, 10.0, 20.0])
    epoch0_y = np.zeros(3)
    epoch1_x = np.array([5.0, 15.0])
    epoch1_y = np.zeros(2)

    ref_x, ref_y, matched = geometry_utils.match_epochs_to_reference_grid(
        [(epoch0_x, epoch0_y), (epoch1_x, epoch1_y)], tolerance_m=1.0,
    )

    check("union grid keeps ALL points (3 + 2 = 5), not min(3,2) = 2",
          len(ref_x) == 5, f"got {len(ref_x)}")

    check("epoch 0 matches exactly its own 3 points (distance 0)",
          int(np.sum(matched[0] >= 0)) == 3, f"got {int(np.sum(matched[0] >= 0))}")
    check("epoch 1 matches exactly its own 2 points (distance 0)",
          int(np.sum(matched[1] >= 0)) == 2, f"got {int(np.sum(matched[1] >= 0))}")

    # epoch 0 seeds the grid first & in order -> reference indices 0,1,2
    # should be epoch 0's own points 0,1,2 respectively (distance-0 match).
    check("epoch 0's own points map 1:1, in order, to the first 3 ref points",
          list(matched[0][:3]) == [0, 1, 2],
          f"got {matched[0][:3]}")
    check("epoch 0 has NaN/-1 for the 2 reference points it doesn't cover",
          list(matched[0][3:]) == [-1, -1], f"got {matched[0][3:]}")


def test_reference_grid_overlap_dedup():
    section("match_epochs_to_reference_grid: overlapping points deduplicate")

    # Epoch 0 and epoch 1 share two locations exactly (0.0 and 10.0), and
    # each also has one point unique to itself.
    epoch0_x = np.array([0.0, 10.0, 99.0])   # 99.0 unique to epoch 0
    epoch0_y = np.zeros(3)
    epoch1_x = np.array([0.05, 10.02, -50.0])  # near-duplicates of 0.0/10.0
    epoch1_y = np.zeros(3)                      # (within 1 m tol) + unique -50.0

    ref_x, ref_y, matched = geometry_utils.match_epochs_to_reference_grid(
        [(epoch0_x, epoch0_y), (epoch1_x, epoch1_y)], tolerance_m=1.0,
    )

    # Union should be: {0.0, 10.0, 99.0} from epoch 0 (unchanged) plus
    # ONLY -50.0 newly added from epoch 1 (0.05/10.02 fold into the
    # existing epoch-0 reference points within tolerance) -> 4 total.
    check("overlapping (within-tolerance) points collapse to ONE reference point",
          len(ref_x) == 4, f"got {len(ref_x)}: {ref_x}")

    check("epoch 0 still matches all 3 of its own points",
          int(np.sum(matched[0] >= 0)) == 3)
    check("epoch 1 matches all 3 of its own points too (2 folded + 1 new)",
          int(np.sum(matched[1] >= 0)) == 3)

    # The two near-duplicate epoch-1 points should map to the SAME
    # reference indices as epoch 0's corresponding points.
    idx_0_at_0 = matched[0][0]   # epoch0's point at x=0.0 -> ref index
    idx_1_at_0 = matched[1][0]   # epoch1's point at x=0.05 -> ref index
    check("near-duplicate points across epochs share one reference index",
          idx_0_at_0 == idx_1_at_0,
          f"epoch0->{idx_0_at_0}  epoch1->{idx_1_at_0}")


def test_reference_grid_no_data_loss_matches_union_cardinality():
    section("match_epochs_to_reference_grid: never fewer ref points than the "
            "old min_pts trim would have kept, and never loses a genuinely "
            "unique point")

    rng = np.random.default_rng(42)
    # 4 epochs with DIFFERENT point counts (mimics real GEM-2 surveys
    # walking different paths on different days) and no accidental overlap
    # (spacing 100 m apart, well outside a realistic tolerance).
    epoch_sizes = [37, 51, 19, 44]
    epochs = []
    cursor = 0.0
    for n in epoch_sizes:
        xs = cursor + np.arange(n) * 100.0
        ys = np.zeros(n)
        epochs.append((xs, ys))
        cursor += n * 100.0 + 10_000.0  # big gap so epochs never overlap

    ref_x, ref_y, matched = geometry_utils.match_epochs_to_reference_grid(
        epochs, tolerance_m=2.0,
    )

    old_min_pts_trim_size = min(epoch_sizes)          # what the OLD code kept
    union_size = sum(epoch_sizes)                     # every point unique here

    check(f"reference grid has {union_size} points (== sum of all epochs, "
          f"since none overlap), not the old min_pts={old_min_pts_trim_size}",
          len(ref_x) == union_size, f"got {len(ref_x)}")
    check("reference grid is strictly larger than the old row-count trim would be",
          len(ref_x) > old_min_pts_trim_size)

    for i, n in enumerate(epoch_sizes):
        n_matched = int(np.sum(matched[i] >= 0))
        check(f"epoch {i} ({n} own points) matches ALL {n} of its own points",
              n_matched == n, f"got {n_matched}")


# ===========================================================================
# 2. Remap-with-NaN-fill (reproduces run_pipeline.py's
#    `_remap_combo_to_reference` closure so it can be unit-tested standalone)
# ===========================================================================

def _remap_combo_to_reference(arr_per_time, matched_indices, n_ref):
    """Verbatim reproduction of the closure defined inside
    run_pipeline.main()'s Step 1 (real + time-lapse branch)."""
    remapped = []
    for t_idx, arr in enumerate(arr_per_time):
        out = np.full((n_ref, arr.shape[-1]), np.nan, dtype=float)
        idx_map = matched_indices[t_idx]
        valid = idx_map >= 0
        if np.any(valid):
            out[valid] = arr[idx_map[valid]]
        remapped.append(out)
    return remapped


def test_remap_preserves_values_and_nan_fills_gaps():
    section("Step 1 remap: values preserved at matched cells, NaN at "
            "unmatched cells, for BOTH EC models and depths arrays")

    epoch0_x = np.array([0.0, 10.0, 20.0])
    epoch0_y = np.zeros(3)
    epoch1_x = np.array([10.02, 30.0])   # point 0 ~matches epoch0's x=10.0;
    epoch1_y = np.zeros(2)               # point 1 (x=30.0) is new

    ref_x, ref_y, matched = geometry_utils.match_epochs_to_reference_grid(
        [(epoch0_x, epoch0_y), (epoch1_x, epoch1_y)], tolerance_m=1.0,
    )
    n_ref = len(ref_x)   # expect 4: {0.0, 10.0, 20.0, 30.0}
    check("sanity: reference grid has 4 points", n_ref == 4, f"got {n_ref}")

    n_layers = 3
    # Epoch 0's inverted EC model: 3 points x 3 layers, values = point_idx*10 + layer
    ec_epoch0 = np.array([[p * 10 + l for l in range(n_layers)] for p in range(3)], dtype=float)
    # Epoch 1's inverted EC model: 2 points x 3 layers, values = 100 + point_idx*10 + layer
    ec_epoch1 = np.array([[100 + p * 10 + l for l in range(n_layers)] for p in range(2)], dtype=float)

    combo = [ec_epoch0, ec_epoch1]   # one "combo" across 2 epochs
    remapped = _remap_combo_to_reference(combo, matched, n_ref)

    remapped_t0, remapped_t1 = remapped

    # --- Epoch 0's remapped slice ---------------------------------------
    check("epoch-0 remap: point 0 (x=0.0) keeps its original EC row",
          np.array_equal(remapped_t0[0], ec_epoch0[0]))
    check("epoch-0 remap: point 1 (x=10.0) keeps its original EC row",
          np.array_equal(remapped_t0[1], ec_epoch0[1]))
    check("epoch-0 remap: point 2 (x=20.0) keeps its original EC row",
          np.array_equal(remapped_t0[2], ec_epoch0[2]))
    check("epoch-0 remap: reference point x=30.0 (not covered by epoch 0) is all-NaN",
          bool(np.all(np.isnan(remapped_t0[3]))))

    # --- Epoch 1's remapped slice ---------------------------------------
    # epoch1's point 0 (x=10.02) should land on the SAME reference index as
    # epoch0's point at x=10.0 (folded within tolerance).
    ref_idx_of_10 = matched[0][1]
    check("epoch-1 remap: near-duplicate point (x=10.02) lands on ref index for x=10.0",
          np.array_equal(remapped_t1[ref_idx_of_10], ec_epoch1[0]))
    check("epoch-1 remap: x=0.0 (not covered by epoch 1) is all-NaN",
          bool(np.all(np.isnan(remapped_t1[0]))))
    check("epoch-1 remap: x=20.0 (not covered by epoch 1) is all-NaN",
          bool(np.all(np.isnan(remapped_t1[2]))))
    check("epoch-1 remap: x=30.0 (epoch 1's own unique point) keeps its EC row",
          np.array_equal(remapped_t1[3], ec_epoch1[1]))

    # --- No silent misalignment: verify a NaN pattern is NOT accidentally
    #     identical across epochs when it genuinely shouldn't be (i.e. this
    #     isn't just returning zeros / a constant array by accident).
    check("epoch-0 and epoch-1 NaN masks differ (they cover different points)",
          not np.array_equal(np.isnan(remapped_t0), np.isnan(remapped_t1)))


# ===========================================================================
# 3. NaN-safety follow-on fixes: RMSE and CV-map reductions
#    (reproduces the exact expressions used in step4_select.py)
# ===========================================================================

def test_rmse_nanmean_vs_old_plain_mean():
    section("step4_select.py fix: nanmean RMSE vs. the old plain-mean bug")

    theta_obs  = np.array([0.20, 0.25, 0.30, 0.22])
    theta_pred = np.array([0.21, np.nan, 0.29, 0.20])   # probe 1 unmatched at its epoch

    # OLD (buggy) behaviour: a single NaN poisons the ENTIRE rmse_by_k[k].
    old_sq_err = (theta_obs - theta_pred) ** 2
    old_rmse = np.sqrt(np.mean(old_sq_err))
    check("OLD np.mean-based RMSE is NaN when ANY observation is unmatched "
          "(demonstrates the bug this patch fixes)",
          bool(np.isnan(old_rmse)))

    # NEW behaviour: nanmean scores the stack against only the observations
    # that actually have data at their own survey epoch.
    _sq_err = (theta_obs - theta_pred) ** 2
    if np.all(np.isnan(_sq_err)):
        new_rmse = float("nan")
    else:
        new_rmse = float(np.sqrt(np.nanmean(_sq_err)))

    expected = float(np.sqrt(np.mean([
        (0.20 - 0.21) ** 2, (0.30 - 0.29) ** 2, (0.22 - 0.20) ** 2,
    ])))
    check("NEW nanmean-based RMSE is finite and ignores only the NaN probe",
          np.isfinite(new_rmse))
    check("NEW RMSE numerically matches manual nanmean over the 3 valid probes",
          abs(new_rmse - expected) < 1e-12, f"got {new_rmse}, expected {expected}")


def test_rmse_all_nan_stack_is_reported_not_silently_wrong():
    section("step4_select.py fix: an all-NaN stack depth is flagged, "
            "not silently coerced to a fake number")

    theta_obs  = np.array([0.20, 0.25])
    theta_pred = np.array([np.nan, np.nan])
    _sq_err = (theta_obs - theta_pred) ** 2
    if np.all(np.isnan(_sq_err)):
        rmse_k = float("nan")
    else:
        rmse_k = float(np.sqrt(np.nanmean(_sq_err)))
    check("an entirely-unmatched stack depth K correctly reports NaN "
          "(caller then raises rather than np.argmin silently picking a "
          "bogus 'best' K)",
          bool(np.isnan(rmse_k)))


def test_cv_map_preserves_nan_instead_of_false_zero():
    section("step4_select.py / em_methodology.py fix: CV map NaN "
            "handling (no more false 'CV=0.0' for genuinely empty cells)")

    # subset_mean has: a normal cell, a near-zero (but finite) cell, and a
    # NaN cell (e.g. every top-K member unmatched at this reference-grid
    # point / epoch combination).
    subset_mean = np.array([10.0, 1e-7, np.nan])
    subset_std  = np.array([2.0, 1e-8, np.nan])

    # OLD (buggy) formula: comparisons with NaN are always False, so the
    # NaN cell silently fell into the "near-zero -> 0.0" branch.
    with np.errstate(invalid="ignore"):
        old_cv = np.where(np.abs(subset_mean) > 1e-5, subset_std / subset_mean, 0.0)
    check("OLD formula mis-reports the NaN cell as CV=0.0 ('zero uncertainty') "
          "-- this is the bug being fixed",
          old_cv[2] == 0.0, f"got {old_cv[2]}")

    # NEW formula: NaN mean -> NaN cv; near-zero-but-finite mean -> 0.0
    # (unchanged); normal mean -> std/mean (unchanged).
    with np.errstate(divide="ignore", invalid="ignore"):
        new_cv = np.where(
            np.abs(subset_mean) > 1e-5,
            subset_std / subset_mean,
            np.where(np.isfinite(subset_mean), 0.0, np.nan),
        )
    check("NEW formula: normal cell unchanged (std/mean = 0.2)",
          abs(new_cv[0] - 0.2) < 1e-12, f"got {new_cv[0]}")
    check("NEW formula: near-zero finite cell still reports 0.0 (unchanged behaviour)",
          new_cv[1] == 0.0, f"got {new_cv[1]}")
    check("NEW formula: NaN-mean cell correctly reports NaN, not a false 0.0",
          bool(np.isnan(new_cv[2])))


# ===========================================================================
# 4. em_methodology.calibrate_petrophysical_relationship — NaN-observation
#    filtering inside the actual patched function (imported live, with a
#    stub `emagpy` module so the import succeeds without the real package).
# ===========================================================================

def test_calibrate_drops_nan_observations_not_whole_horizon():
    section("em_methodology.calibrate_petrophysical_relationship: NaN "
            "observations are dropped per-horizon, not left to poison the "
            "whole fit or wrongly mark the horizon uncalibrated")

    import em_methodology as EMmethod

    # Single-horizon-of-interest test: 1 ensemble member, 4 spatial points,
    # 1 layer, 2 time steps. Build a (ensemble=1, n_time=2, n_pts=4, n_layers=1)
    # array where one (time, pos) combination is NaN -- exactly what the
    # spatial reference-grid remap now produces for an unmatched epoch/point.
    models = np.zeros((1, 2, 4, 1), dtype=float)
    # sigma_b as a function of theta via forward Waxman-Smits with a=1.5,
    # m=1.8, sigma_w=50, sigma_s=5 (matching DEFAULT_WS_P0) so a fit close
    # to the prior is expected and the optimizer has an easy target.
    a0, m0, sw0, ss0 = 1.5, 1.8, 50.0, 5.0
    thetas = np.array([0.15, 0.20, 0.25, 0.30])

    def forward(theta):
        return (theta ** m0 / a0) * (sw0 + ss0)

    sample_positions = [0, 1, 2, 3]
    sample_times     = [0, 0, 0, 0]
    probe_layer_indices = [0, 0, 0, 0]

    for i, (pos, t) in enumerate(zip(sample_positions, sample_times)):
        models[0, t, pos, 0] = forward(thetas[i])

    # Now poison ONE observation with NaN, mimicking an epoch-unmatched
    # reference-grid cell from geometry_utils.match_epochs_to_reference_grid.
    models[0, sample_times[1], sample_positions[1], 0] = np.nan

    theta_obs = thetas.copy()

    fits = EMmethod.calibrate_petrophysical_relationship(
        models=models,
        sample_positions=sample_positions,
        sample_times=sample_times,
        theta_obs=theta_obs,
        probe_layer_indices=probe_layer_indices,
        depths_are_fixed=True,
    )

    check("exactly one PetroFit is returned (ensemble_size=1)", len(fits) == 1)
    hz0 = fits[0].horizon_fits[0]

    check("horizon stays CALIBRATED (3 valid obs remain after dropping the 1 NaN)",
          hz0.is_calibrated is True)
    check("exactly 3 observations survive the NaN filter (started with 4, dropped 1)",
          len(hz0.observed_theta) == 3, f"got {len(hz0.observed_theta)}")
    check("no NaN leaked into observed_theta",
          not np.any(np.isnan(hz0.observed_theta)))
    check("no NaN leaked into predicted_theta",
          not np.any(np.isnan(hz0.predicted_theta)))
    check("rmse is finite (a NaN observation no longer poisons the whole horizon fit)",
          np.isfinite(hz0.rmse), f"got {hz0.rmse}")


def test_calibrate_all_nan_horizon_becomes_uncalibrated_not_crash():
    section("em_methodology.calibrate_petrophysical_relationship: a horizon "
            "where EVERY observation is NaN is marked uncalibrated (not a crash, "
            "not a fabricated fit)")

    import em_methodology as EMmethod

    models = np.full((1, 1, 2, 1), np.nan, dtype=float)  # every EC value NaN
    theta_obs = np.array([0.2, 0.25])
    sample_positions = [0, 1]
    sample_times = [0, 0]
    probe_layer_indices = [0, 0]

    fits = EMmethod.calibrate_petrophysical_relationship(
        models=models,
        sample_positions=sample_positions,
        sample_times=sample_times,
        theta_obs=theta_obs,
        probe_layer_indices=probe_layer_indices,
        depths_are_fixed=True,
    )
    hz0 = fits[0].horizon_fits[0]
    check("horizon with 100% NaN observations is marked UNCALIBRATED",
          hz0.is_calibrated is False)
    check("uncalibrated horizon's rmse is NaN (excluded from ranking, per existing convention)",
          bool(np.isnan(hz0.rmse)))


# ===========================================================================
# 5. em_methodology.average_inverted_models — NaN propagation across the
#    ensemble axis (same NaN pattern for every member at a fixed (t, point))
# ===========================================================================

def test_average_inverted_models_propagates_nan_correctly():
    section("em_methodology.average_inverted_models: NaN cells stay NaN "
            "(never silently averaged into a fake number) when every member "
            "shares the same NaN pattern")

    import em_methodology as EMmethod

    # 3 ensemble members, each shape (2 points, 1 layer). Point 1 is NaN
    # for EVERY member (mimicking an epoch-unmatched reference-grid point --
    # the NaN pattern is identical across the ensemble axis by construction,
    # see geometry_utils.match_epochs_to_reference_grid).
    models = np.array([
        [[10.0], [np.nan]],
        [[12.0], [np.nan]],
        [[11.0], [np.nan]],
    ])

    class _FakeHorizonFit:
        def __init__(self, rmse):
            self.rmse = rmse

    class _FakeFit:
        def __init__(self, rmse):
            self.horizon_fits = [_FakeHorizonFit(rmse)]
            self.rmse = rmse

    fits = [_FakeFit(0.01), _FakeFit(0.02), _FakeFit(0.03)]

    averaged_models, ranking = EMmethod.average_inverted_models(models, fits)

    check("3 cumulative averaged models produced (one per K=1..3)",
          len(averaged_models) == 3)
    for k, avg in enumerate(averaged_models):
        check(f"K={k+1}: point 0 (fully covered) is finite",
              np.isfinite(avg[0, 0]))
        check(f"K={k+1}: point 1 (NaN for every member) stays NaN, not a fabricated value",
              bool(np.isnan(avg[1, 0])))


# ===========================================================================
# 6. scenario_loader — new spatial_match_tolerance_m field
# ===========================================================================

def test_scenario_loader_spatial_match_tolerance_field():
    section("scenario_loader: spatial_match_tolerance_m field validates and defaults correctly")

    import scenario_loader as sl

    base_scenario = {
        "depths0": [0.3, 0.6],
        "fixed_depths": True,
        "param_grid": {"alpha": [0.1]},
        "cond_ranges": [[1.0, 10.0], [1.0, 10.0], [1.0, 10.0]],
    }

    # Default (field omitted) must not raise.
    try:
        sl._validate("test_default", dict(base_scenario))
        default_ok = True
    except Exception as e:
        default_ok = False
        print(f"    unexpected exception: {e}")
    check("scenario without spatial_match_tolerance_m validates fine (uses default)",
          default_ok)

    # Explicit valid value.
    good = dict(base_scenario)
    good["spatial_match_tolerance_m"] = 3.5
    try:
        sl._validate("test_good", good)
        good_ok = True
    except Exception as e:
        good_ok = False
        print(f"    unexpected exception: {e}")
    check("explicit positive spatial_match_tolerance_m validates fine", good_ok)

    # Invalid (negative) value must raise.
    bad = dict(base_scenario)
    bad["spatial_match_tolerance_m"] = -1.0
    raised = False
    try:
        sl._validate("test_bad", bad)
    except ValueError:
        raised = True
    check("negative spatial_match_tolerance_m is rejected by validation", raised)

    # Default resolves to 2.0 in a loaded ScenarioConfig.
    import dataclasses
    field_default = None
    for f in dataclasses.fields(sl.ScenarioConfig):
        if f.name == "spatial_match_tolerance_m":
            field_default = f.default
    check("ScenarioConfig.spatial_match_tolerance_m default is 2.0",
          field_default == 2.0, f"got {field_default}")
