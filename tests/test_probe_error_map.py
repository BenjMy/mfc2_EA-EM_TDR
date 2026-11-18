"""probe_error_map: error table, summaries, CSV/figure output from a fake run folder (fast tier)."""
import csv
import os

import numpy as np
import pytest

from plots import probe_error_map as pe

T, P, L = 3, 4, 2
X = np.array([0.0, 10.0, 20.0, 30.0])
CALC = np.arange(T * P * L, dtype=float).reshape(T, P, L) / 100.0


def _obs():
    pos = np.array([0, 1, 1, 3])
    tim = np.array([0, 0, 2, 1])
    lay = np.array([0, 1, 0, 1])
    theta = CALC[tim, pos, lay] - np.array([0.01, -0.02, 0.03, 0.0])   # known errors
    return theta, pos, tim, lay


def test_error_is_calc_minus_tdr_and_indexing():
    theta, pos, tim, lay = _obs()
    tab = pe.build_probe_error_table(theta, CALC, pos, tim, lay, X)
    assert np.allclose(tab["error"], [0.01, -0.02, 0.03, 0.0])
    assert np.allclose(tab["x"], X[pos]) and tab["y"].sum() == 0 and not tab["held_out_flag"].any()


def test_2d_theta_calc_is_one_epoch():
    tab = pe.build_probe_error_table([0.1], CALC[0], [2], [0], [1], X)
    assert tab["theta_calc"][0] == CALC[0, 2, 1]


@pytest.mark.parametrize("bad", [dict(times=[5, 0, 0, 0]), dict(positions=[9, 0, 0, 0]),
                                 dict(layers=[0, 0, 0, 7])])
def test_out_of_range_indices_raise(bad):
    theta, pos, tim, lay = _obs()
    kw = dict(positions=pos, times=tim, layers=lay) | bad
    with pytest.raises(ValueError):
        pe.build_probe_error_table(theta, CALC, kw["positions"], kw["times"], kw["layers"], X)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        pe.build_probe_error_table([0.1, 0.2], CALC, [0], [0], [0], X)


def test_summaries_rmse_and_bias():
    theta, pos, tim, lay = _obs()
    tab = pe.build_probe_error_table(theta, CALC, pos, tim, lay, X)
    st = {(r["station"], r["horizon"]): r for r in pe.summarise(tab, "station")}
    assert st[(1, 1)]["bias"] == pytest.approx(-0.02) and st[(1, 1)]["RMSE"] == pytest.approx(0.02)
    ep = {(r["epoch"], r["horizon"]): r for r in pe.summarise(tab, "epoch")}
    assert ep[(0, 0)]["n"] == 1 and ep[(2, 0)]["bias"] == pytest.approx(0.03)


def _fake_run(tmp_path, with_xy=True, held=None):
    om = tmp_path / "run" / "output_model"
    om.mkdir(parents=True)
    theta, pos, tim, lay = _obs()
    for n, a in dict(tdr_train_theta=theta, tdr_train_positions=pos, tdr_train_times=tim,
                     tdr_train_layers=lay, theta_calc=CALC).items():
        np.save(om / f"{n}.npy", a)
    if with_xy:
        np.save(om / "synthetic_x_coords.npy", X)
        np.save(om / "synthetic_y_coords.npy", np.zeros(P))
    if held is not None:
        np.save(om / "held_out.npy", np.asarray(held))
    return str(tmp_path / "run")


def test_run_writes_csvs_and_map(tmp_path):
    paths = pe.run_from_output_model(_fake_run(tmp_path))
    assert os.path.getsize(paths["map"]) > 1000 and "map_held_out" not in paths
    rows = list(csv.DictReader(open(paths["probe_errors"])))
    assert len(rows) == 4 and list(rows[0]) == pe.COLUMNS
    assert float(rows[2]["error"]) == pytest.approx(0.03)
    assert rows[0]["held_out_flag"] == "False"


def test_held_out_map_and_missing_coords_fall_back_to_index(tmp_path):
    paths = pe.run_from_output_model(_fake_run(tmp_path, with_xy=False, held=[False, True, True, False]))
    assert "map_held_out" in paths
    rows = list(csv.DictReader(open(paths["probe_errors"])))
    assert float(rows[3]["x"]) == 3.0          # index used when no coordinates are saved


def test_held_out_only_without_flags_raises(tmp_path):
    theta, pos, tim, lay = _obs()
    tab = pe.build_probe_error_table(theta, CALC, pos, tim, lay, X)
    with pytest.raises(ValueError):
        pe.plot_probe_error_map(tab, str(tmp_path / "f.png"), held_out_only=True)


def test_missing_arrays_give_clear_error(tmp_path):
    (tmp_path / "r" / "output_model").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="lacks"):
        pe.run_from_output_model(str(tmp_path / "r"))
