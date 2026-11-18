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
