"""bandwidth_sweep_check: nearest-neighbour spacing, default sweep and the warning."""
import numpy as np
import pytest

import bandwidth_sweep_check as bs

# 4 probes on a line, 25 m apart; each probe repeated for 3 horizons (as sample_positions is)
X = np.repeat([0.0, 25.0, 50.0, 75.0], 3)


def test_unique_locations_collapses_repeats():
    assert len(bs.unique_probe_locations(X)) == 4


def test_nn_distances_1d():
    assert np.allclose(bs.nearest_neighbour_distances(X), 25.0)


def test_nn_distances_2d_uses_euclidean():
    nn = bs.nearest_neighbour_distances([0, 3, 100], [0, 4, 0])
    assert nn[0] == pytest.approx(5.0)
    assert nn[1] == pytest.approx(5.0)
    assert nn[2] == pytest.approx(np.hypot(97.0, 4.0))


def test_single_location_gives_no_distances():
    assert bs.nearest_neighbour_distances([1.0, 1.0]).size == 0
    assert bs.sweep_spacing_warning([1, 2, 3], [1.0, 1.0]) is None


def test_default_sweep_scales_with_median_spacing():
    assert bs.default_bandwidth_sweep(X) == [12.5, 25.0, 37.5, 50.0, 75.0, 125.0]


def test_default_sweep_needs_two_locations():
    with pytest.raises(ValueError):
        bs.default_bandwidth_sweep([1.0, 1.0])


def test_warning_when_support_never_reaches_a_neighbour():
    # c <= 12 -> reach 24 < 25 m spacing: every fold falls back to the pooled fit
    msg = bs.sweep_spacing_warning([0.5, 5, 12], X)
    assert msg and "no probe has a neighbour" in msg


def test_boundary_reach_equal_to_spacing_still_warns():
    assert bs.sweep_spacing_warning([12.5], X) is not None


def test_weak_warning_when_few_probes_reach():
    x = [0.0, 10.0, 100.0, 200.0, 300.0]   # one close pair, three isolated probes
    msg = bs.sweep_spacing_warning([6.0], x)
    assert msg and "only 2 of 5" in msg


def test_no_warning_for_adequate_sweep():
    assert bs.sweep_spacing_warning([15, 20, 25, 30, 40], X) is None
