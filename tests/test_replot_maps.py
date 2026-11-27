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


def test_load_probe_xy_real_style_file(tmp_path):
    # real SOILNET file: '[' header lines, 'x;y;z;Pos;Notes', decimal commas, a station without coords
    f = tmp_path / "coords.csv"
    f.write_text("[Header]\n[more]\nx;y;z;Pos;Notes\n515542,64;4468241,59;1;Unit 1;a\n"
                 "515573.54;4468245.77;1;Unit 2;b\n;;1;Unit 3;no coords\n")
    x, y = rm.load_probe_xy(str(f))
    assert np.allclose(x, [515542.64, 515573.54]) and np.allclose(y, [4468241.59, 4468245.77])
    g = tmp_path / "syn.csv"
    g.write_text("Pos;x;y\nUnit 1;1.5;2.5\n")
    assert np.allclose(rm.load_probe_xy(str(g))[0], [1.5])
    assert rm.load_probe_xy(str(tmp_path / "missing.csv")) is None


def test_epoch_labels_sorted_by_date_and_count_checked(tmp_path):
    for n, d in [(2, "291020"), (1, "040820"), (3, "151220")]:
        (tmp_path / f"{n}_ALENTO_{d}.csv").write_text("X;Y\n")
    assert rm.epoch_labels(str(tmp_path), 3) == ["2020-08-04", "2020-10-29", "2020-12-15"]
    assert rm.epoch_labels(str(tmp_path), 7) is None            # count mismatch -> default labels
    assert rm.epoch_labels(None, 3) is None


def test_replot_with_probes_and_labels_and_frame_guard(tmp_path, capsys):
    run = _fake_run(str(tmp_path), 1, "scn", nt=2)
    em = tmp_path / "em"; em.mkdir()
    for n, d in [(1, "040820"), (2, "291020")]:
        (em / f"{n}_X_{d}.csv").write_text("X;Y\n")
    co = tmp_path / "co.csv"; co.write_text("Pos;x;y\nU1;50;40\nU2;60;30\n")
    out = rm.replot_map_run(run, "scn", window_m=3.0, em_dir=str(em), coord_file=str(co))
    assert all(os.path.getsize(p) > 0 for p in out)
    far = tmp_path / "far.csv"; far.write_text("Pos;x;y\nU1;515000;4468000\n")   # other frame
    rm.replot_map_run(run, "scn", coord_file=str(far))
    assert "probe markers skipped" in capsys.readouterr().out
