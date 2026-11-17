"""Synthetic train/validation example: whole probes held out (by_probe).

The synthetic transect is time-lapse (3 epochs): by_time_block holds out an
epoch but keeps the same probes in train and validation, so the spatially
independent split is by_probe; the probe id of a synthetic observation is its
survey position.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scenario_loader as sl  # noqa: E402
import train_val_split as tvs  # noqa: E402


def _obs(n_pos=6, n_layers=4, n_time=1):
    """Flattened (epoch, position, layer) observations, as in run_pipeline."""
    t, pos, layer = (a.ravel() for a in np.meshgrid(
        np.arange(n_time), np.arange(n_pos), np.arange(n_layers), indexing="ij"))
    theta = 0.2 + 0.01 * pos + 0.005 * layer + 0.002 * t
    return theta, pos, t, layer


def test_probe_id_synthetic_is_position():
    _, pos, _, _ = _obs()
    assert np.array_equal(tvs.probe_id_for_split("synthetic", pos), pos)


def test_probe_id_real_uses_station_index():
    pos = np.array([0, 0, 1])
    stations = np.array([7, 7, 9])
    assert np.array_equal(tvs.probe_id_for_split("real", pos, stations), stations)
    assert tvs.probe_id_for_split("real", pos, None) is None


@pytest.mark.parametrize("n_layers", [1, 4])
def test_by_probe_holds_out_whole_positions(n_layers):
    theta, pos, t, layer = _obs(6, n_layers)
    sp = tvs.split_train_val(theta, pos, t, layer,
                             probe_id=tvs.probe_id_for_split("synthetic", pos),
                             method="by_probe", seed=0)
    assert sp.independent
    assert not set(pos[sp.train]) & set(pos[sp.val])          # no shared probe
    assert len(sp.train) + len(sp.val) == len(theta)           # nothing lost
    assert len(set(pos[sp.train])) == len(set(pos[sp.val])) == 3
    # every train probe keeps all its horizons
    assert len(sp.train) == 3 * n_layers


def test_by_probe_is_reproducible():
    theta, pos, t, layer = _obs()
    kw = dict(probe_id=pos, method="by_probe", seed=0)
    a = tvs.split_train_val(theta, pos, t, layer, **kw)
    b = tvs.split_train_val(theta, pos, t, layer, **kw)
    assert np.array_equal(a.train, b.train) and np.array_equal(a.val, b.val)


def test_by_time_block_shares_probes_but_by_probe_does_not():
    """Why by_probe matters on the 3-epoch synthetic transect."""
    theta, pos, t, layer = _obs(6, 1, n_time=3)
    tb = tvs.split_train_val(theta, pos, t, layer, method="by_time_block")
    bp = tvs.split_train_val(theta, pos, t, layer, probe_id=pos, method="by_probe")
    assert set(pos[tb.train]) == set(pos[tb.val])             # same probes both sides
    assert not set(pos[bp.train]) & set(pos[bp.val])          # disjoint probes
    assert len(bp.train) == len(bp.val) == 9                  # 18 obs, as in the e2e run


def test_by_time_block_has_no_validation_with_a_single_epoch():
    theta, pos, t, layer = _obs(6, 4, n_time=1)
    assert len(tvs.split_train_val(theta, pos, t, layer, method="by_time_block").val) == 0


@pytest.mark.parametrize("name,n_obs", [("1d_surface_free2_tv_probe", 1),
                                        ("1d_full_free2_tv_probe", 4)])
def test_tv_scenarios_load_and_validate(name, n_obs):
    cfg = sl.load_scenario(name)
    assert cfg.data_source == "synthetic"
    assert cfg.train_val_split == "by_probe"
    assert cfg.num_swc_positions == 6
    assert cfg.num_swc_positions % 2 == 0          # equal train/val halves


def test_tv_scenarios_differ_from_base_only_in_split_and_positions():
    for tv, base in (("1d_surface_free2_tv_probe", "1d_surface_free2"),
                     ("1d_full_free2_tv_probe", "1d_full_free2")):
        a, b = sl.load_scenario(tv), sl.load_scenario(base)
        diff = {k for k in vars(a) if getattr(a, k) != getattr(b, k)}
        assert diff <= {"name", "description", "output_dir", "cache_tag",
                        "train_val_split", "num_swc_positions"}, diff
