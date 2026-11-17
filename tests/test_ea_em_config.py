"""EA_EM_config: dataset registry and path-resolution precedence."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import EA_EM_config as cfg  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("EA_EM_DATASET", raising=False)
    saved = dict(cfg.DATASETS)
    yield
    cfg.DATASETS.clear()
    cfg.DATASETS.update(saved)


def test_default_is_mfc2_with_legacy_paths():
    ds = cfg.get_dataset()
    assert ds.name == "mfc2"
    assert ds.em_dir == "./data/EMI_MFC2/"
    assert ds.tdr_dir == "./data/daily_MFC/"
    assert ds.coord_file == "./data/SOILNET_MFC2_coordinates_sensors.csv"


def test_unknown_dataset_lists_known():
    with pytest.raises(KeyError, match="mfc2"):
        cfg.get_dataset("nope")


def test_register_and_env_selection(monkeypatch):
    cfg.register_dataset("site_x", "a/em/", "a/tdr/", "a/c.csv")
    monkeypatch.setenv("EA_EM_DATASET", "site_x")
    assert cfg.get_dataset().em_dir == "a/em/"


def test_precedence():
    cfg.register_dataset("site_x", "x/em/", "x/tdr/", "x/c.csv")
    scen = dict(scenario_em_dir="s/em", scenario_tdr_dir="s/tdr", scenario_coord_file="s/c.csv")
    # nothing else: scenario JSON beats the default dataset
    assert cfg.resolve_real_paths(**scen).em_dir == "s/em"
    # no scenario paths: default dataset
    assert cfg.resolve_real_paths().em_dir == "./data/EMI_MFC2/"
    # --dataset beats scenario JSON
    r = cfg.resolve_real_paths(cli_dataset="site_x", **scen)
    assert (r.em_dir, r.tdr_dir, r.coord_file, r.dataset) == ("x/em/", "x/tdr/", "x/c.csv", "site_x")
    # explicit path beats --dataset, field by field
    r = cfg.resolve_real_paths(cli_em_dir="cli/em", cli_dataset="site_x", **scen)
    assert (r.em_dir, r.tdr_dir) == ("cli/em", "x/tdr/")


def test_problems_and_user_json(tmp_path):
    em, tdr = tmp_path / "em", tmp_path / "tdr"
    em.mkdir(); tdr.mkdir()
    coord = tmp_path / "c.csv"
    ds = cfg.register_dataset("t", str(em), str(tdr), str(coord))
    assert len(ds.problems()) == 1            # coord file missing
    coord.write_text("x")
    assert ds.problems() == []

    f = tmp_path / "extra.json"
    f.write_text(json.dumps({"_comment": "x", "j": {"em_dir": "e", "tdr_dir": "t", "coord_file": "c"}}))
    cfg._load_user_datasets(str(f))
    assert cfg.get_dataset("j").coord_file == "c"


def test_cli_field_and_unknown():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "EA_EM_config.py"), "--field", "tdr_dir"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.stdout.strip() == "./data/daily_MFC/"
    bad = subprocess.run([sys.executable, os.path.join(ROOT, "EA_EM_config.py"), "--dataset", "zzz", "--field", "em_dir"],
                         capture_output=True, text=True, cwd=ROOT)
    assert bad.returncode == 3


def test_scenario_may_omit_paths(tmp_path):
    import scenario_loader as sl
    raw = json.load(open(os.path.join(ROOT, "ensemble_scenarios_real.json")))
    raw["no_paths"] = {k: v for k, v in raw["real_single_base"].items()
                       if k not in ("em_data_dir", "tdr_data_dir", "coord_file")}
    f = tmp_path / "s.json"
    f.write_text(json.dumps(raw))
    c = sl.load_scenario("no_paths", config_path=str(f))
    assert c.data_source == "real"
    assert c.em_data_dir is None and c.tdr_data_dir is None and c.coord_file is None
    # an explicitly empty path is still rejected
    raw["no_paths"]["em_data_dir"] = ""
    f.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="em_data_dir"):
        sl.load_scenario("no_paths", config_path=str(f))
