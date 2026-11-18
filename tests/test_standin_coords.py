"""generate_synthetic_real_mirror.make_standin_coords: layout inside the footprint."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("emagpy.invertHelper")  # the generator imports eca2Q at module level
import generate_synthetic_real_mirror as g  # noqa: E402


def _read(path):
    df = pd.read_csv(path, delimiter=";", comment="[")
    df.columns = df.columns.str.strip()
    df["Pos"] = df["Pos"].str.strip()
    return df


def test_format_matches_real_reader(tmp_path):
    df = _read(g.make_standin_coords(str(tmp_path), 20))
    assert list(df.columns) == ["Pos", "x", "y"] and len(df) == 20
    assert df["Pos"].tolist() == [f"Unit {i}" for i in range(1, 21)]
    assert int(df["Pos"][0].replace("Unit", "").strip()) == 1  # same parse as generate_tdr_stations


def test_inside_footprint_of_every_dense_epoch(tmp_path):
    df = _read(g.make_standin_coords(str(tmp_path), 24))
    for e in g.REAL_EM_EPOCHS:
        if e[0] == 3:  # degraded epoch, covers a small patch only
            continue
        assert df["x"].between(e[4], e[5]).all() and df["y"].between(e[6], e[7]).all()


def test_deterministic_and_seed_dependent(tmp_path):
    a = _read(g.make_standin_coords(str(tmp_path / "a"), 20, seed=1))
    b = _read(g.make_standin_coords(str(tmp_path / "b"), 20, seed=1))
    c = _read(g.make_standin_coords(str(tmp_path / "c"), 20, seed=2))
    assert a.equals(b) and not a.equals(c)


def test_stations_do_not_collapse(tmp_path):
    df = _read(g.make_standin_coords(str(tmp_path), 20))
    xy = df[["x", "y"]].to_numpy()
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    assert d.min() > 8.0  # metres
