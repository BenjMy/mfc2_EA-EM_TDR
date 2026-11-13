"""plots/ws_distribution.py: initial-value resolution, truth description, figure."""
import os

import numpy as np
import pytest

from plots import ws_distribution as wsd


def _emagpy_is_stub():
    import emagpy
    return getattr(emagpy, "__is_stub__", False)


def test_resolve_initial_default_flat_and_matrix():
    d = wsd.resolve_initial_ws(None, 4)
    assert d.shape == (4, 4) and np.allclose(d[0], wsd.DEFAULT_WS_P0)
    f = wsd.resolve_initial_ws([1.0, 2.0, 3.0, 4.0], 3)
    assert f.shape == (3, 4) and np.allclose(f, [1.0, 2.0, 3.0, 4.0])
    m = np.arange(8.0).reshape(2, 4)
    assert np.allclose(wsd.resolve_initial_ws(m, 2), m)


@pytest.mark.parametrize("bad,n", [([1, 2, 3], 2), (np.zeros((3, 4)), 2)])
def test_resolve_initial_rejects_bad_shape(bad, n):
    with pytest.raises(ValueError):
        wsd.resolve_initial_ws(bad, n)


def test_uniform_truth_from_params_and_description(tmp_path):
    params = {0: dict(a=1, m=2, sigma_w=500, sigma_s=5),
              1: dict(a=1, m=2, sigma_w=500, sigma_s=5),
              2: dict(a=0.5, m=2, sigma_w=700, sigma_s=5)}
    t = wsd.load_ws_truth(str(tmp_path), ws_truth_params=params, n_points=6)
    assert t.shape == (6, 3, 4)
    v = wsd.describe_truth(t)
    assert v["a"] == (False, True) and v["m"] == (False, False)
    np.save(tmp_path / "synthetic_x_coords.npy", np.arange(6.0))
    out = wsd.plot_ws_truth_vs_initial(str(tmp_path), ws_truth_params=params)
    assert out and os.path.getsize(out) > 0


def test_no_truth_returns_none(tmp_path):
    assert wsd.plot_ws_truth_vs_initial(str(tmp_path)) is None


@pytest.mark.skipif(_emagpy_is_stub(), reason="needs the real emagpy forward model")
@pytest.mark.parametrize("hetero", [False, True])
def test_figure_for_generated_truth(tmp_path, hetero):
    import data_synthetic as gen
    over = {2: {"a": 0.35, "sigma_w": 700.0}, 3: {"a": 0.35, "sigma_w": 700.0}} if hetero else None
    gen.generate_time_lapse_dataset(
        output_dir=str(tmp_path), mode="1d", depths0=[0.3, 0.6, 1.0],
        ws_xy_variation=True, ws_xy_params=["m", "sigma_s"], ws_layer_params=over)
    truth = wsd.load_ws_truth(str(tmp_path))
    v = wsd.describe_truth(truth)
    assert v["m"][0] and v["sigma_s"][0]                  # vary along x
    assert v["a"][1] is hetero and v["sigma_w"][1] is hetero   # vary with depth only in hetero
    out = wsd.plot_ws_truth_vs_initial(str(tmp_path), ws_fixed_params={"a": True})
    assert out and os.path.getsize(out) > 0
