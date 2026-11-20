import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("xarray"); pytest.importorskip("netCDF4")
import maps_export as mx
import plot_run_log as prl
from run_log import RunLog


def _make(d, th, x, y, **kw):
    return mx.export_maps(str(d / "output_model" / mx.MAPS_NAME), th, x, y,
                          layer_depth=[0.1, 0.3], attrs={"scenario": "s", "rsr": 0.4}, **kw)


def test_roundtrip_and_shapes(tmp_path):
    th = np.random.default_rng(0).uniform(0.1, 0.4, (2, 5, 2))
    p = _make(tmp_path, th, np.arange(5.0), np.arange(5.0) * 2, extra={"theta_std_total": th * 0.1, "bad": np.zeros(3)})
    ds = mx.load_maps(p)
    assert ds["theta"].shape == (2, 5, 2) and "theta_std_total" in ds and "bad" not in ds
    assert ds.attrs["scenario"] == "s" and float(ds["layer_depth"][1]) == 0.3
    np.testing.assert_allclose(ds["theta"].values, th)


def test_2d_input_and_bad_coords(tmp_path, capsys):
    assert _make(tmp_path, np.ones((5, 2)), np.arange(5.0), None) is not None
    assert mx.export_maps(str(tmp_path / "b.nc"), np.ones((5, 2)), np.arange(3.0)) is None
    assert "WARNING" in capsys.readouterr().out
    assert mx.load_maps(str(tmp_path / "missing.nc")) is None


def test_collect_extra(tmp_path):
    np.save(tmp_path / "theta_std_ec.npy", np.ones((1, 4, 2)))
    np.save(tmp_path / "theta_std_ws.npy", np.ones((3,)))
    assert set(mx.collect_extra(str(tmp_path), (1, 4, 2))) == {"theta_std_ec"}


def test_shared_limits():
    lo, hi = prl.shared_limits([np.array([0.1, 0.2]), np.array([0.3, np.nan])])
    assert 0.1 <= lo < hi <= 0.3 and prl.shared_limits([]) == (0.0, 1.0)


def test_side_by_side_2d_and_1d(tmp_path):
    log = RunLog(str(tmp_path / "l.csv"))
    rng = np.random.default_rng(1)
    for sc in ("1d_surface_free1", "1d_surface_free2"):
        d = tmp_path / "runs" / sc
        _make(d, rng.uniform(0.1, 0.4, (1, 6, 2)), np.arange(6.0), np.arange(6.0) % 3)
        log.append({"run_id": sc, "scenario": sc, "status": "SUCCESS", "final_rmse": "0.05", "output_dir": str(d)})
    rows = prl.group_by_study(prl.load_rows(str(tmp_path / "l.csv")))["surface_vs_full"]
    out = tmp_path / "o"
    files = prl.plot_study("surface_vs_full", rows, str(out), maps=True)
    assert any(f.endswith("maps_theta_epoch-1.png") for f in files)
    # 1-D: all y equal -> lines along x
    d = tmp_path / "runs" / "1d_surface_free1"
    _make(d, rng.uniform(0.1, 0.4, (1, 6, 2)), np.arange(6.0), np.zeros(6))
    f = prl.plot_maps_side_by_side(prl.load_run_maps(prl.latest_per_scenario(rows)), str(out), "theta", 0, "t")
    assert f and os.path.isfile(f)
    assert prl.plot_maps_side_by_side({}, str(out)) is None
