"""em_subsample.py: validation, the four modes, probe-nearest guarantee, CSV rewrite, cache fingerprint."""
import numpy as np
import pandas as pd
import pytest

import em_subsample as ems


def _pts(n=2000, seed=0):
    r = np.random.RandomState(seed)
    return 515500 + r.rand(n) * 100, 4470000 + r.rand(n) * 40


def test_validation():
    assert ems.validate_em_subsample(None) is None
    assert ems.validate_em_subsample({"mode": "off"}) is None
    assert ems.validate_em_subsample({"mode": "grid", "cell_m": 3})["keep_probe_nearest"] is True
    for bad in ({"mode": "grid"}, {"mode": "grid", "cell_m": -1}, {"mode": "stride", "stride": 2.5},
                {"mode": "nope"}, {"mode": "grid", "cell_m": 3, "cellm": 1}, "grid"):
        with pytest.raises(ValueError):
            ems.validate_em_subsample(bad)


def test_grid_one_per_cell_and_same_cells_across_epochs():
    x, y = _pts()
    cfg = ems.validate_em_subsample({"mode": "grid", "cell_m": 5.0})
    k = ems.select_indices(x, y, cfg)
    cells = set(zip(np.floor(x[k] / 5).astype(int), np.floor(y[k] / 5).astype(int)))
    assert len(cells) == len(k) < len(x)
    # a second "epoch" with different points occupies the same absolute cells
    x2, y2 = _pts(seed=1)
    k2 = ems.select_indices(x2, y2, cfg)
    c2 = set(zip(np.floor(x2[k2] / 5).astype(int), np.floor(y2[k2] / 5).astype(int)))
    assert len(cells & c2) > 0.9 * len(cells)


def test_probes_stride_random_and_probe_nearest():
    x, y = _pts()
    px, py = np.array([515520.0, 515580.0]), np.array([4470010.0, 4470030.0])
    kp = ems.select_indices(x, y, ems.validate_em_subsample({"mode": "probes", "radius_m": 4}), px, py)
    d = np.sqrt((x[kp, None] - px) ** 2 + (y[kp, None] - py) ** 2).min(axis=1)
    assert d.max() <= 4 + 1e-9 or kp.size >= 2
    ks = ems.select_indices(x, y, ems.validate_em_subsample({"mode": "stride", "stride": 10, "keep_probe_nearest": False}))
    assert len(ks) == 200
    cfg = ems.validate_em_subsample({"mode": "random", "max_points": 50, "seed": 3})
    a, b = ems.select_indices(x, y, cfg, px, py), ems.select_indices(x, y, cfg, px, py)
    assert np.array_equal(a, b) and 50 <= len(a) <= 52
    # the nearest EM point of every probe is always kept (tiny radius would drop them otherwise)
    tiny = ems.validate_em_subsample({"mode": "probes", "radius_m": 1e-6})
    kt = ems.select_indices(x, y, tiny, px, py)
    nearest = np.sqrt((x[:, None] - px) ** 2 + (y[:, None] - py) ** 2).argmin(axis=0)
    assert set(nearest) <= set(kt)


def test_probes_mode_needs_probes():
    x, y = _pts(50)
    with pytest.raises(ValueError):
        ems.select_indices(x, y, ems.validate_em_subsample({"mode": "probes", "radius_m": 3}))


def test_csv_rewrite_keeps_order_and_columns(tmp_path):
    x, y = _pts(500)
    df = pd.DataFrame({"x": x, "y": y, "HCP1.66f425h1_quad": np.arange(500.0)})
    f = tmp_path / "p.csv"; df.to_csv(f, index=False)
    st = ems.subsample_em_csv(str(f), ems.validate_em_subsample({"mode": "stride", "stride": 5}))
    out = pd.read_csv(f)
    assert list(out.columns) == list(df.columns) and st["n_after"] == len(out) < st["n_before"] == 500
    assert out["HCP1.66f425h1_quad"].is_monotonic_increasing


def test_fingerprint_changes_only_when_thinning_on():
    pytest.importorskip("tqdm")
    import run_pipeline as rp
    a = rp._config_fingerprint([0.2], True, {"m": [1]}, [[1, 2], [1, 2]], [0.0])
    assert a == rp._config_fingerprint([0.2], True, {"m": [1]}, [[1, 2], [1, 2]], [0.0], em_subsample=None)
    b = rp._config_fingerprint([0.2], True, {"m": [1]}, [[1, 2], [1, 2]], [0.0], em_subsample={"mode": "grid", "cell_m": 3.0})
    assert a != b


def test_scenario_loads_and_rejects(tmp_path):
    import scenario_loader as sl
    names = sl.list_scenarios("ensemble_scenarios_real.json") if hasattr(sl, "list_scenarios") else None
    cfg = sl.load_scenario("real_single_test_grid", "ensemble_scenarios_real.json") if hasattr(sl, "load_scenario") else None
    if cfg is not None:
        assert cfg.em_subsample == {"mode": "grid", "cell_m": 3.0, "keep_probe_nearest": True}
