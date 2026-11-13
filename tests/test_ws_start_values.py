"""scenario_loader.check_ws_start_values: lab_p0 / true ws_layer_params vs the WS bounds."""
import dataclasses
import os

import pytest

import scenario_loader as sl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.join(ROOT, "ensemble_scenarios.json")
REAL = os.path.join(ROOT, "ensemble_scenarios_real.json")


def _cfg(**over):
    base = sl.load_scenario("1d_full_free2", config_path=SYN)
    return dataclasses.replace(base, **over)


def test_all_real_scenarios_start_inside_the_bounds():
    for name in sl.list_scenarios(REAL):
        assert sl.check_ws_start_values(sl.load_scenario(name, config_path=REAL)) == [], name


def test_fixed_parameter_on_the_bound_is_not_reported():
    # real_*_base: a = 0.5 sits on the lower bound but a is fixed -> harmless
    cfg = sl.load_scenario("real_single_base", config_path=REAL)
    assert cfg.lab_p0[0] == 0.5 and cfg.ws_fixed_params["a"] is True
    assert sl.check_ws_start_values(cfg) == []


def test_free_parameter_on_the_bound_is_reported():
    cfg = _cfg(lab_p0=[0.5, 3.0, 500.0, 100.0],
               ws_fixed_params={"a": False, "m": True, "sigma_w": True, "sigma_s": True})
    (msg,) = sl.check_ws_start_values(cfg)
    assert "free a=0.5 starts ON the lower bound" in msg and "pile up" in msg


def test_ws_bounds_override_moves_the_bound():
    cfg = _cfg(lab_p0=[0.5, 3.0, 500.0, 100.0], ws_bounds={"a": [0.2, 5.0]},
               ws_fixed_params={"a": False, "m": True, "sigma_w": True, "sigma_s": True})
    assert sl.check_ws_start_values(cfg) == []  # 0.5 is now inside, off the bound


@pytest.mark.parametrize("fixed", [True, False])
def test_start_value_outside_bounds_is_reported_fixed_or_free(fixed):
    cfg = _cfg(lab_p0=[0.3, 3.0, 500.0, 100.0],
               ws_fixed_params={"a": fixed, "m": True, "sigma_w": True, "sigma_s": True})
    (msg,) = sl.check_ws_start_values(cfg)
    assert "a=0.3 is OUTSIDE" in msg and ("(fixed)" if fixed else "(free)") in msg


def test_per_horizon_lab_p0_names_the_horizon():
    cfg = _cfg(lab_p0=[[1.5, 1.8, 50.0, 5.0], [1.5, 1.8, 2000.0, 5.0]])
    (msg,) = sl.check_ws_start_values(cfg)
    assert "horizon 1" in msg and "sigma_w=2000" in msg


def test_unset_lab_p0_uses_the_library_default_and_is_inside():
    assert sl.check_ws_start_values(_cfg(lab_p0=None)) == []


def test_hetero_truth_below_the_default_bound_is_flagged_only_when_bounds_not_widened():
    narrow = sl.load_scenario("1d_full_hetero_free3", config_path=SYN)
    wide = sl.load_scenario("1d_full_hetero_free4", config_path=SYN)   # ws_bounds a [0.2, 5]
    assert [m for m in sl.check_ws_start_values(narrow) if "ws_layer_params" in m]
    assert not [m for m in sl.check_ws_start_values(wide) if "ws_layer_params" in m]
