"""generate_synthetic_real_mirror: built-in station layout and the callable build_dataset."""
import os

import numpy as np
import pandas as pd
import pytest

emagpy = pytest.importorskip("emagpy")
if getattr(emagpy, "__is_stub__", False):
    pytest.skip("needs the real emagpy (forward model)", allow_module_level=True)

import generate_synthetic_real_mirror as gen  # noqa: E402


def test_station_layout_is_deterministic_and_in_the_box(tmp_path):
    a = gen.write_station_layout(str(tmp_path / "a.csv"), n_stations=20, seed=1)
    b = gen.write_station_layout(str(tmp_path / "b.csv"), n_stations=20, seed=1)
    c = gen.write_station_layout(str(tmp_path / "c.csv"), n_stations=20, seed=2)
    da, db, dc = (pd.read_csv(p, sep=";") for p in (a, b, c))
    assert list(da.columns) == ["Pos", "x", "y"] and len(da) == 20
    assert da.equals(db) and not da.equals(dc)
    assert list(da["Pos"][:2]) == ["Unit 1", "Unit 2"]
    x0, x1, y0, y1 = gen.STATION_BOX
    assert da["x"].between(x0, x1).all() and da["y"].between(y0, y1).all()
    pts = da[["x", "y"]].to_numpy()
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1) + np.eye(20) * 1e9
    assert d.min() > 10.0                                   # no stacked stations


def test_build_dataset_writes_everything_the_real_loader_needs(tmp_path):
    out = gen.build_dataset(str(tmp_path), scale=0.001, max_points=20, n_stations=4, tdr_freq_days=30)
    assert out["n_em_epochs"] == 7
    em = sorted(os.listdir(out["em_dir"]))
    assert len(em) == 7 and all(f.endswith(".csv") for f in em)
    tdr = sorted(os.listdir(out["tdr_dir"]))
    assert tdr == [f"MTX0{i}_synthetic.xlsx" for i in range(1, 5)]
    assert os.path.isfile(out["coord_file"]) and os.path.isfile(out["scenario_json"])
    assert os.path.isdir(out["ground_truth_dir"]) and os.listdir(out["ground_truth_dir"])
    first = pd.read_csv(os.path.join(out["em_dir"], em[0]), sep=";")
    assert {"X", "Y", "GPSalt", "Q_425Hz", "Q_92775Hz"} <= set(first.columns) and len(first) == 20


def test_scenario_json_output_dir_is_configurable(tmp_path):
    import json
    out = gen.build_dataset(str(tmp_path / "d"), scale=0.001, max_points=20, n_stations=3,
                            tdr_freq_days=60, output_dir=str(tmp_path / "runs"))
    sc = json.load(open(out["scenario_json"]))[gen.SCENARIO_NAME]
    assert sc["output_dir"] == str(tmp_path / "runs")
    default = json.load(open(gen.write_scenario_json(
        str(tmp_path), out["em_dir"], out["tdr_dir"], out["coord_file"])))[gen.SCENARIO_NAME]
    assert default["output_dir"].endswith("/" + gen.SCENARIO_NAME)


def test_session_fixture_registers_dataset_in_process_and_for_subprocess(mfc2_synthetic):
    import json
    import subprocess
    import sys

    import EA_EM_config as cfg
    ds = cfg.get_dataset("mfc2_synthetic")
    assert ds.coord_file == mfc2_synthetic["coord_file"]
    assert os.path.isdir(ds.em_dir) and os.path.isdir(ds.tdr_dir)
    assert mfc2_synthetic["run_dir"].startswith(mfc2_synthetic["root"])
    assert json.load(open(mfc2_synthetic["datasets_file"]))["mfc2_synthetic"]["em_dir"] == ds.em_dir
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ("import EA_EM_config as c; d=c.get_dataset('mfc2_synthetic'); print(d.coord_file)")
    res = subprocess.run([sys.executable, "-c", code], cwd=root, env=mfc2_synthetic["env"],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == ds.coord_file


def test_session_fixture_scenario_loads_with_the_real_loader(mfc2_synthetic):
    import scenario_loader
    path = mfc2_synthetic["scenario_json"]
    assert scenario_loader.list_scenarios(path) == [gen.SCENARIO_NAME]
    cfg = scenario_loader.load_scenario(gen.SCENARIO_NAME, config_path=path)
    assert cfg.data_source == "real" and cfg.output_dir == mfc2_synthetic["run_dir"]
    assert list(cfg.depths0) == list(gen.DEPTHS0)


def _em_q(out):
    d = out["em_dir"]
    return {f: pd.read_csv(os.path.join(d, f), sep=";") for f in sorted(os.listdir(d))}


def _small(tmp_path, name, **kw):
    return gen.build_dataset(str(tmp_path / name), scale=0.001, max_points=20, n_stations=3,
                             tdr_freq_days=90, **kw)


def test_constant_ws_truth_reproduces_the_legacy_generator_exactly(tmp_path):
    legacy = _em_q(_small(tmp_path, "legacy"))
    const = _em_q(_small(tmp_path, "const", ws_truth="constant"))
    assert legacy.keys() == const.keys()
    for f in legacy:
        pd.testing.assert_frame_equal(legacy[f], const[f])


def test_two_zones_truth_changes_em_only_where_sigma_s_differs(tmp_path):
    base = _em_q(_small(tmp_path, "b", ws_truth="constant"))
    zones = _em_q(_small(tmp_path, "z", ws_truth={"mode": "two_zones", "sigma_s_lo": 100.0,
                                                  "sigma_s_hi": 300.0}))
    x0, x1 = gen._X0, gen._X1
    f = sorted(base)[0]
    west = (base[f]["X"] - x0) / (x1 - x0) < 0.5
    q_b, q_z = base[f]["Q_92775Hz"], zones[f]["Q_92775Hz"]
    assert np.allclose(q_b[west], q_z[west])                 # sigma_s_lo == base value: unchanged
    assert (q_z[~west].abs() > q_b[~west].abs()).all()       # more surface conduction: stronger signal


def test_ws_truth_is_saved_as_ground_truth(tmp_path):
    out = _small(tmp_path, "g", ws_truth="gradient")
    gt = out["ground_truth_dir"]
    arr = np.load(os.path.join(gt, "em_epoch1_2020-08-04_ws.npy"))
    xy = np.load(os.path.join(gt, "em_epoch1_2020-08-04_xy.npy"))
    assert arr.shape == (len(xy), gen.N_LAYERS, 4)
    order = np.argsort(xy[:, 0])
    s = arr[order, 0, 3]
    assert np.all(np.diff(s) >= -1e-9) and s[-1] > s[0]      # sigma_s grows with x
    import json
    assert json.load(open(os.path.join(gt, "ws_truth.json")))["mode"] == "gradient"


def test_ec_bulk_truth_follows_the_saved_truth_and_defaults_to_legacy(tmp_path):
    out = _small(tmp_path, "e", ws_truth={"mode": "two_zones", "sigma_s_lo": 0.0, "sigma_s_hi": 200.0})
    truth = gen.load_ws_truth(out["ground_truth_dir"])
    assert truth.mode == "two_zones"
    x = np.array([gen._X0 + 1.0, gen._X1 - 1.0])
    y = np.array([gen._Y0, gen._Y0])
    theta = np.full((1, 2, gen.N_LAYERS), 0.3)
    ec = gen.ec_bulk_truth(theta, x, y, truth)
    assert np.allclose(ec[0, 0], 0.3 ** 3 / 0.5 * (500.0 + 0.0))
    assert np.allclose(ec[0, 1], 0.3 ** 3 / 0.5 * (500.0 + 200.0))
    assert np.allclose(gen.ec_bulk_truth(theta, x, y), gen.forward_waxman_smits(theta))
    legacy = _small(tmp_path, "l")
    assert gen.load_ws_truth(legacy["ground_truth_dir"]) is None


# ---------------------------------------------------------------------------
# near_station_points (thinned survey: guarantee a point next to every station)
# ---------------------------------------------------------------------------

def _em_files(out):
    return [pd.read_csv(os.path.join(out["em_dir"], f), sep=";")
            for f in sorted(os.listdir(out["em_dir"]))]


def _near(tmp_path, name, **kw):
    return gen.build_dataset(str(tmp_path / name), scale=0.001, max_points=20, n_stations=6,
                             tdr_freq_days=60, **kw)


def test_near_station_points_default_off_is_unchanged(tmp_path):
    a, b = _near(tmp_path, "a"), _near(tmp_path, "b", near_station_points=0)
    for fa, fb in zip(_em_files(a), _em_files(b)):
        pd.testing.assert_frame_equal(fa, fb)


def test_near_station_points_cover_every_station_and_keep_the_original_points(tmp_path):
    base = _near(tmp_path, "base")
    near = _near(tmp_path, "near", near_station_points=2)
    st = pd.read_csv(near["coord_file"], sep=";")
    key = ["X", "Y", "GPSalt"]
    for fb, fn in zip(_em_files(base), _em_files(near)):
        assert len(fn) == len(fb) + 2 * len(st)
        for r in st.itertuples():                      # >= 2 points within 1 m of each station
            assert (np.hypot(fn["X"] - r.x, fn["Y"] - r.y) <= gen.NEAR_STATION_RADIUS_M).sum() >= 2
        # the original points are all still there (separate RNG stream), only the order differs
        merged = fb[key].merge(fn[key], on=key, how="left", indicator=True)
        assert (merged["_merge"] == "both").all()


def test_near_station_points_are_ordered_along_a_serpentine_path(tmp_path):
    fn = _em_files(_near(tmp_path, "near", near_station_points=1))[0]
    # consecutive points are much closer than the survey is wide (a random order is not)
    step = np.hypot(np.diff(fn["X"]), np.diff(fn["Y"]))
    assert np.median(step) < 0.35 * (fn["X"].max() - fn["X"].min())


def test_near_station_points_need_station_coordinates():
    with pytest.raises(ValueError, match="station_xy"):
        gen.generate_em_surveys("/tmp/never", 0.001, 20, 0.0, 1, near_station_points=1)


# ---------------------------------------------------------------------------
# truth_forward_model (no inverse crime)
# ---------------------------------------------------------------------------

def test_truth_forward_model_changes_the_eca_but_not_the_geometry(tmp_path):
    cs = _near(tmp_path, "cs")
    fs = _near(tmp_path, "fs", truth_forward_model="FSlin")
    for a, b in zip(_em_files(cs), _em_files(fs)):
        pd.testing.assert_frame_equal(a[["X", "Y", "GPSalt"]], b[["X", "Y", "GPSalt"]])
        q = [c for c in a.columns if c.startswith("Q_")]
        assert not np.allclose(a[q].to_numpy(), b[q].to_numpy())
        assert np.isfinite(b[q].to_numpy()).all()
