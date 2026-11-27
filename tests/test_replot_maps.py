"""replot_maps: finds the newest map run and redraws XY maps from maps.nc (no inversion)."""
import os

import numpy as np
import pytest

pytest.importorskip("xarray")
import maps_export as mx
import replot_maps as rm


def _fake_run(root, n, scenario, nt=3, npt=40, nl=3):
    d = os.path.join(root, f"{n:04d}_{scenario}_map", "output_model")
    os.makedirs(d)
    rng = np.random.default_rng(n)
    th = rng.uniform(0.1, 0.4, (nt, npt, nl))
    th[:, ::2, :] = np.nan                                   # unmatched reference-grid cells
    mx.export_maps(os.path.join(d, "maps.nc"), th, rng.uniform(0, 100, npt), rng.uniform(0, 80, npt))
    return os.path.dirname(d)


def test_latest_map_run_picks_highest_number(tmp_path):
    _fake_run(str(tmp_path), 1, "scn")
    r2 = _fake_run(str(tmp_path), 2, "scn")
    os.makedirs(tmp_path / "0003_scn_map")                    # no maps.nc -> ignored
    assert rm.latest_map_run(str(tmp_path), "scn") == r2
    assert rm.latest_map_run(str(tmp_path), "other") is None


def test_replot_writes_pngs(tmp_path):
    run = _fake_run(str(tmp_path), 1, "scn")
    out = rm.replot_map_run(run, "scn", window_m=3.0)
    assert len(out) == 2 and all(os.path.getsize(p) > 0 for p in out)
    assert all(os.path.dirname(p) == os.path.join(run, "replot") for p in out)
