"""synthetic_ws_truth: WS truth modes (pure numpy, fast tier)."""
import numpy as np
import pytest

import synthetic_ws_truth as wst

BBOX = (0.0, 100.0, 0.0, 50.0)
NL = 4


def _p(truth, x, layer=0):
    return wst.ws_params(truth, np.asarray(x, float), np.zeros(len(x)), layer, BBOX, NL)


def test_constant_is_the_base_value_everywhere():
    p = _p(wst.WSTruth(), [0, 50, 100], layer=2)
    for k, v in zip(wst.PARAM_NAMES, (0.5, 3.0, 500.0, 100.0)):
        assert np.all(p[k] == v)


def test_per_horizon_changes_with_depth_not_with_x():
    t = wst.WSTruth(mode="per_horizon")
    p0, p3 = _p(t, [0, 100], 0), _p(t, [0, 100], 3)
    assert p0["a"][0] == p0["a"][1] == 0.5 and p3["a"][0] == p3["a"][1] == 0.35
    assert p3["sigma_w"][0] == 700.0 and p0["sigma_w"][0] == 500.0


def test_per_horizon_wrong_length_is_an_error():
    t = wst.WSTruth(mode="per_horizon", per_horizon={"a": [0.5, 0.5]})
    with pytest.raises(ValueError, match="need 4"):
        _p(t, [10], 0)


def test_gradient_is_linear_in_x_and_leaves_other_params_alone():
    t = wst.WSTruth(mode="gradient", sigma_s_lo=0.0, sigma_s_hi=200.0)
    p = _p(t, [0, 25, 50, 100, 150])               # 150 is outside the box -> clipped
    assert np.allclose(p["sigma_s"], [0, 50, 100, 200, 200])
    assert np.all(p["sigma_w"] == 500.0) and np.all(p["a"] == 0.5)


def test_two_zones_jump_at_the_split():
    t = wst.WSTruth(mode="two_zones", split=0.4)
    p = _p(t, [0, 39.9, 40.0, 100], layer=1)
    assert list(p["sigma_s"]) == [0.0, 0.0, 200.0, 200.0]


def test_bulk_ec_matches_the_legacy_formula():
    th = np.array([0.1, 0.3])
    p = _p(wst.WSTruth(), [0, 1])
    assert np.allclose(wst.bulk_ec(th, p), (th ** 3 / 0.5) * 600.0)
    assert np.isfinite(wst.bulk_ec(np.array([0.0, -1.0]), p)).all()   # clipped, no NaN


def test_params_array_shape_and_order():
    arr = wst.params_array(wst.WSTruth(mode="two_zones"), np.array([10.0, 90.0]),
                           np.zeros(2), BBOX, NL)
    assert arr.shape == (2, NL, 4)
    assert list(arr[:, 0, 3]) == [0.0, 200.0]          # [a, m, sigma_w, sigma_s]


def test_as_truth_accepts_name_dict_and_instance_and_rejects_junk():
    assert wst.as_truth("gradient").mode == "gradient"
    assert wst.as_truth({"mode": "two_zones", "split": 0.3}).split == 0.3
    t = wst.WSTruth()
    assert wst.as_truth(t) is t
    with pytest.raises(ValueError):
        wst.as_truth("nope")
    with pytest.raises(TypeError):
        wst.as_truth(3)
    with pytest.raises(ValueError):
        wst.WSTruth(split=1.5)
