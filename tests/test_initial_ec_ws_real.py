"""plots/initial_ec_ws_real.py: horizon routing, WS forward, both figures."""
import os

import numpy as np
import pandas as pd

from plots import initial_ec_ws_real as iew


def test_horizon_index_and_labels():
    d = [0.3, 0.6, 1.0]
    assert [iew.horizon_index(z, d) for z in (0.15, 0.3, 0.45, 0.8, 1.2)] == [0, 1, 1, 2, 3]
    assert iew.horizon_labels(d)[0].startswith("H0") and iew.horizon_labels(d)[-1].endswith("bottom m")


def test_ws_forward_matches_formula():
    assert np.isclose(iew.ws_ec_from_theta(0.2, 1.0, 2.0, 500.0, 100.0), 600.0 * 0.04)


def _data():
    pt = pd.DataFrame({"probe_index": [0, 1, 2], "station_name": list("abc"),
                       "utm_x": [10.0, 20.0, 30.0], "utm_y": [5.0, 6.0, 7.0]})
    obs = [{"probe_index": p, "time_index": t, "depth": z, "swc": 0.15 + 0.05 * p + 0.02 * t}
           for p in range(3) for t in range(2) for z in (0.15, 0.30)]
    return pt, obs


def test_initial_ec_from_tdr_single_interface():
    pt, obs = _data()
    init = iew.resolve_initial_ws([0.5, 3.0, 500.0, 100.0], 2)
    d = iew.initial_ec_from_tdr(obs, pt, [0.225], init)
    assert len(d["ec_init"]) == len(obs)
    assert set(d["horizon"]) == {0, 1}              # 15 cm -> H0, 30 cm -> H1
    assert np.all(d["ec_init"] > 0)


def test_both_figures_written(tmp_path):
    pt, obs = _data()
    out = iew.plot_initial_ec_ws_real(
        str(tmp_path), cond_ranges=[[0.01, 300.0], [0.01, 300.0]], depths0=[0.225],
        lab_p0=[0.5, 3.0, 500.0, 100.0],
        ws_fixed_params={"a": True, "m": True, "sigma_w": False, "sigma_s": False},
        observations=obs, probe_table=pt)
    for k in ("ec", "ws"):
        assert out[k] and os.path.getsize(out[k]) > 0


def test_no_tdr_and_horizon_without_probe(tmp_path):
    # 4 horizons, TDR only in the first two: empty panels must not crash
    pt, obs = _data()
    out = iew.plot_initial_ec_ws_real(
        str(tmp_path), cond_ranges=[[1, 300]] * 4, depths0=[0.225, 0.6, 1.0],
        observations=obs, probe_table=pt)
    assert os.path.getsize(out["ec"]) > 0
    out = iew.plot_initial_ec_ws_real(str(tmp_path / "b"), cond_ranges=[[1, 300]] * 2, depths0=[0.225])
    assert os.path.getsize(out["ws"]) > 0
