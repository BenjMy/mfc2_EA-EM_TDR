import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import run_real_study as rs
import scenario_loader as sl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "ensemble_scenarios_real.json")


def test_parse_design():
    assert rs.parse_design("L3b=0.2,0.5") == ("L3b", [0.2, 0.5])
    for bad in ("L3", "L=0.5,0.2", "L=0,1"):
        with pytest.raises(ValueError):
            rs.parse_design(bad)


def test_coverage_flags_empty_horizons():
    assert rs.horizon_coverage([0.225]) == {0: 1, 1: 1}
    assert rs.horizon_coverage([0.225, 0.6]) == {0: 1, 1: 1, 2: 0}
    assert "NO SENSOR" in rs.coverage_text([0.3, 0.6, 1.0])


def test_ws_freedom():
    f = rs.ws_freedom(2, [0.5, 3.0, 500.0, 100.0])
    assert f["ws_fixed_params"] == {"a": True, "m": False, "sigma_w": True, "sigma_s": False}
    assert "lab_p0" not in f
    f4 = rs.ws_freedom(4, [0.5, 3.0, 500.0, 100.0])
    assert f4["lab_p0"][0] == 1.0 and not any(f4["ws_fixed_params"].values())
    with pytest.raises(ValueError):
        rs.ws_freedom(5, [0.5, 3.0, 500.0, 100.0])


def test_study_config_loads_and_shares_cache(tmp_path):
    with open(CFG, encoding="utf-8") as f:
        raw = {k: v for k, v in json.load(f).items() if k not in sl._SKIP_KEYS}
    base = sl._resolve_extends(raw)["real_single_base"]
    cfg = rs.build_study_config(base, {"L2": [0.225], "L3": [0.225, 0.6]}, [1, 4], fast=True)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    sc = sl.load_scenario("real_study_L3_free4", config_path=str(p))
    assert len(sc.depths0) == 2
    assert cfg["real_study_L3_free1"]["cache_tag"] == cfg["real_study_L3_free4"]["cache_tag"]
    assert cfg["real_study_L2_free1"]["cache_tag"] != cfg["real_study_L3_free1"]["cache_tag"]
    assert cfg["real_study_parent"]["em_subsample"]["mode"] == "probes"


def test_dry_run_cli(tmp_path, capsys):
    assert rs.main(["--fast", "--dry-run", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "6 scenarios" in out  # 3 designs x K in (1, 4)
