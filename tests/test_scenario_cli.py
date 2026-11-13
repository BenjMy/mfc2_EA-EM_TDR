"""scenario_loader CLI (--field), the 'wrong config' hint, the forward-model banner and
the early stop of run_pipeline.sh for a scenario that is in another config file."""
import os
import shutil
import subprocess
import sys

import pytest

import scenario_loader as sl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.join(ROOT, "ensemble_scenarios.json")
REAL = os.path.join(ROOT, "ensemble_scenarios_real.json")


def _cli(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scenario_loader.py"), *args],
                          capture_output=True, text=True)


def test_field_prints_resolved_value_incl_generated_scenarios():
    assert _cli("real_single_base", "--config", REAL, "--field", "data_source").stdout.strip() == "real"
    r = _cli("1d_full_hetero_free2", "--field", "data_source")  # generated + extends
    assert r.returncode == 0 and r.stdout.strip() == "synthetic"
    assert _cli("1d_full_free2", "--field", "coord_file").stdout.strip() != "None"


def test_unknown_scenario_exits_3_with_reason_on_stderr():
    r = _cli("nonsense", "--field", "name")
    assert r.returncode == 3 and "not found" in r.stderr and r.stdout == ""


def test_real_scenario_with_default_config_names_the_right_file():
    with pytest.raises(KeyError, match="defined in ensemble_scenarios_real.json"):
        sl.load_scenario("real_single_base", config_path=SYN)
    r = _cli("real_single_base", "--field", "name")
    assert r.returncode == 3 and "ensemble_scenarios_real.json" in r.stderr


def test_no_hint_when_nothing_else_defines_the_scenario():
    with pytest.raises(KeyError) as e:
        sl.load_scenario("nonsense", config_path=SYN)
    assert "defined in" not in str(e.value)


@pytest.mark.parametrize("name,expect", [
    ("real_single_fslin", "FSlin  (param_grid sweep"),
    ("real_single_cs", "CS  (param_grid sweep"),
    ("real_single_base", "FSlin, CS  (param_grid sweep"),
])
def test_banner_shows_the_forward_model_that_runs(name, expect):
    line = [l for l in sl.load_scenario(name, config_path=REAL).summary().splitlines()
            if l.strip().startswith("Forward model")][0]
    assert expect in line and "not used" in line


def test_banner_falls_back_to_field_without_a_grid_entry():
    cfg = sl.load_scenario("real_single_base", config_path=REAL)
    cfg.param_grid = {}
    cfg.forward_model = "CS"
    assert cfg._forward_model_label() == "CS"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_launcher_stops_early_instead_of_defaulting_to_synthetic():
    r = subprocess.run(["bash", os.path.join(ROOT, "run_pipeline.sh"), "-s", "real_single_base"],
                       capture_output=True, text=True, cwd=ROOT, timeout=60)
    out = r.stdout + r.stderr
    assert r.returncode != 0
    assert "defined in ensemble_scenarios_real.json" in out
    assert "defaulting" not in out and "SYNTHETIC" not in out
