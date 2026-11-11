"""Scenario loader tests: every shipped scenario loads, ``extends`` merges as
documented, ``ws_layer_params`` is validated and reaches the synthetic truth."""
import copy
import json
import os

import numpy as np
import pytest

import scenario_loader as sl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH_JSON = os.path.join(ROOT, "ensemble_scenarios.json")


def _names(path):
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [k for k in raw if k not in sl._SKIP_KEYS]


@pytest.mark.parametrize("name", _names(SYNTH_JSON))
def test_every_synthetic_scenario_loads(name):
    cfg = sl.load_scenario(name, config_path=SYNTH_JSON)
    assert cfg.name == name
    assert "extends" not in cfg.__dict__


def _write(tmp_path, scenarios):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(scenarios), encoding="utf-8")
    return str(p)


def _base():
    with open(SYNTH_JSON, encoding="utf-8") as fh:
        return json.load(fh)["1d_full_depth_ref"]


def test_extends_merges_nested_objects_and_replaces_lists(tmp_path):
    base = _base()
    child = {"extends": "base", "ws_fixed_params": {"m": True}, "ws_xy_params": ["sigma_s"]}
    path = _write(tmp_path, {"base": base, "child": child})
    cfg = sl.load_scenario("child", config_path=path)
    parent = sl.load_scenario("base", config_path=path)
    assert cfg.ws_xy_params == ["sigma_s"]                 # list replaced, not merged
    assert cfg.ws_fixed_params["m"] is True                # nested key overridden...
    assert cfg.ws_fixed_params["sigma_s"] == parent.ws_fixed_params["sigma_s"]  # ...others kept


def test_extends_chain_and_errors(tmp_path):
    base = _base()
    path = _write(tmp_path, {"a": base, "b": {"extends": "a", "tdr_noise_std": 0.02},
                             "c": {"extends": "b", "realization_index": 3}})
    cfg = sl.load_scenario("c", config_path=path)
    assert cfg.tdr_noise_std == 0.02 and cfg.realization_index == 3

    with pytest.raises(ValueError, match="unknown scenario"):
        sl.load_scenario("x", config_path=_write(tmp_path, {"x": {"extends": "nope"}}))
    with pytest.raises(ValueError, match="cycle"):
        sl.load_scenario("p", config_path=_write(tmp_path, {"p": {"extends": "q"}, "q": {"extends": "p"}}))


def test_hetero_scenarios_differ_from_base_only_by_design():
    het = sl.load_scenario("1d_full_hetero_free3", config_path=SYNTH_JSON)
    base = sl.load_scenario("1d_full_free3", config_path=SYNTH_JSON)
    assert het.ws_layer_params == {2: {"a": 0.35, "sigma_w": 700.0}, 3: {"a": 0.35, "sigma_w": 700.0}}
    assert base.ws_layer_params is None
    for field in ("ws_fixed_params", "cond_ranges", "depths0", "swc_depths", "ws_xy_params", "ws_xy_ranges"):
        assert getattr(het, field) == getattr(base, field), field


@pytest.mark.parametrize("bad, msg", [
    ({"9": {"a": 0.4}}, "outside"),
    ({"x": {"a": 0.4}}, "not a layer index"),
    ({"2": {"foo": 1.0}}, "unknown"),
    ({"2": {"a": -1.0}}, "positive"),
    ({"2": {}}, "non-empty"),
    ({}, "non-empty"),
])
def test_ws_layer_params_validation(tmp_path, bad, msg):
    s = _base(); s["ws_layer_params"] = bad
    with pytest.raises(ValueError, match=msg):
        sl.load_scenario("s", config_path=_write(tmp_path, {"s": s}))


def _emagpy_is_stub():
    import emagpy
    return getattr(emagpy, "__is_stub__", False)


@pytest.mark.skipif(_emagpy_is_stub(), reason="needs the real emagpy forward model")
def test_generator_applies_ws_layer_params(tmp_path):
    import data_synthetic as gen
    kw = dict(output_dir=str(tmp_path), mode="1d", depths0=[0.3, 0.6, 1.0])
    over = {2: {"a": 0.35, "sigma_w": 700.0}, 3: {"a": 0.35, "sigma_w": 700.0}}
    gen.generate_time_lapse_dataset(ws_xy_variation=True, ws_xy_params=["m", "sigma_s"],
                                    ws_layer_params=over, **kw)
    a = np.load(tmp_path / "synthetic_ws_truth_a.npy")
    sw = np.load(tmp_path / "synthetic_ws_truth_sigma_w.npy")
    assert np.allclose(a[..., :2], 0.5) and np.allclose(a[..., 2:], 0.35)
    assert np.allclose(sw[..., :2], 500.0) and np.allclose(sw[..., 2:], 700.0)


def test_unknown_key_is_rejected_with_hint(tmp_path):
    s = _base(); s["ws_fixed_paramz"] = {"a": 1.0}
    with pytest.raises(ValueError, match="unknown key.*ws_fixed_paramz"):
        sl.load_scenario("s", config_path=_write(tmp_path, {"s": s}))


def test_underscore_keys_are_comments(tmp_path):
    s = _base(); s["_note"] = "free text"
    sl.load_scenario("s", config_path=_write(tmp_path, {"s": s}))


@pytest.mark.parametrize("path", [SYNTH_JSON, os.path.join(ROOT, "ensemble_scenarios_real.json")])
def test_describe_scenarios_matches_names(path):
    assert [n for n, _ in sl.describe_scenarios(path)] == _names(path)
