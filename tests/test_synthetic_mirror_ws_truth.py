"""generate_synthetic_real_mirror: spatially / per-horizon varying sigma_s truth (to-do J2)."""
import json

import numpy as np
import pytest

pytest.importorskip("emagpy.invertHelper")  # generator imports eca2Q at module level
import generate_synthetic_real_mirror as g  # noqa: E402


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    monkeypatch.setattr(g, "WS_TRUTH_MODE", "constant")
    monkeypatch.setattr(g, "DEPTHS0", list(g.DEPTHS0))
    monkeypatch.setattr(g, "N_LAYERS", g.N_LAYERS)


XW, XE = g._X0 + 0.1 * (g._X1 - g._X0), g._X0 + 0.9 * (g._X1 - g._X0)


def test_constant_is_the_previous_formula_and_ignores_xy():
    theta = np.array([[0.2, 0.3], [0.25, 0.35]])
    old = (np.clip(theta, 0.01, 1.0) ** g.WS_M / g.WS_A) * (g.WS_SIGMA_W + g.WS_SIGMA_S)
    assert np.array_equal(g.forward_waxman_smits(theta), old)
    assert np.array_equal(g.forward_waxman_smits(theta, [XW, XE], [0.0, 0.0]), old)


def test_per_horizon_varies_with_layer_not_x():
    g.configure_ws_truth("per_horizon")
    s = g.ws_sigma_s_truth(np.array([XW, XE])[:, None], 0.0, np.arange(3)[None, :])
    assert np.allclose(s[0], s[1])
    assert np.allclose(s[0], g.WS_SIGMA_S * np.array([1.0, 1.5, 2.0]))


def test_gradient_increases_west_to_east_and_is_layer_independent():
    g.configure_ws_truth("gradient")
    xs = np.linspace(g._X0, g._X1, 6)
    s = g.ws_sigma_s_truth(xs, 0.0, 0)
    assert np.all(np.diff(s) > 0)
    assert np.isclose(s[0], 0.5 * g.WS_SIGMA_S) and np.isclose(s[-1], 1.5 * g.WS_SIGMA_S)
    assert np.allclose(g.ws_sigma_s_truth(xs, 0.0, 0), g.ws_sigma_s_truth(xs, 0.0, 1))


def test_zones_two_values_split_at_mid_footprint():
    g.configure_ws_truth("zones")
    xs = np.linspace(g._X0, g._X1, 11)
    s = g.ws_sigma_s_truth(xs, 0.0, 0)
    assert set(np.unique(s)) == {0.5 * g.WS_SIGMA_S, 1.5 * g.WS_SIGMA_S}
    assert s[0] < s[-1]


def test_forward_ec_reflects_the_truth_mode():
    theta = np.full((2, 2), 0.25)
    xy = ([XW, XE], [0.0, 0.0])
    g.configure_ws_truth("zones")
    ec = g.forward_waxman_smits(theta, *xy)
    assert ec.shape == theta.shape and ec[1, 0] > ec[0, 0]
    factor = (g.WS_SIGMA_W + 1.5 * g.WS_SIGMA_S) / (g.WS_SIGMA_W + 0.5 * g.WS_SIGMA_S)
    assert np.isclose(ec[1, 0] / ec[0, 0], factor)


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        g.configure_ws_truth("nope")


def test_scenario_description_names_mode(tmp_path):
    g.configure_ws_truth("zones")
    path = g.write_scenario_json(str(tmp_path), "em", "tdr", "c.csv")
    assert "zones" in json.load(open(path))["mfc2_real_synthetic_mirror"]["description"]
