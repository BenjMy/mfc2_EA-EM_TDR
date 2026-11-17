import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compare_pinned as cp  # noqa: E402


def _run(tmp, name, theta_off, ec_off, k=3):
    d = tmp / name / "output_model"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    th = 0.2 + 0.05 * rng.random((4, 3, 5, 2))
    ec = 20 + 5 * rng.random((4, 3, 5, 2))
    np.save(d / "synthetic_time_lapse_theta.npy", th)
    np.save(d / "synthetic_time_lapse_ec.npy", ec)
    np.save(d / "theta_calc.npy", th[0] + theta_off)
    np.save(d / "ec_optimal.npy", ec[0] + ec_off)
    np.save(d / "K.npy", np.array(k))
    return str(tmp / name)


def test_constant_offset_gives_rmse_equal_bias(tmp_path):
    a = _run(tmp_path, "a", 0.05, 10.0)
    b = _run(tmp_path, "b", -0.02, 0.0, k=7)
    res = cp.compare({"free": a, "pin": b})
    for rm, bias in res["free"]["theta"]:
        assert np.isclose(rm, 0.05) and np.isclose(bias, 0.05)
    assert np.isclose(res["pin"]["theta"][-1][1], -0.02)
    assert res["pin"]["K"] == 7 and len(res["free"]["ec"]) == 3   # 2 horizons + overall
    assert "| free | 3 |" in cp.format_table(res)


def test_csv(tmp_path):
    a = _run(tmp_path, "a", 0.0, 0.0)
    out = tmp_path / "o.csv"
    cp.main([f"x={a}", "--csv", str(out)])
    assert out.read_text().splitlines()[0] == "run,quantity,horizon,rmse,bias"
