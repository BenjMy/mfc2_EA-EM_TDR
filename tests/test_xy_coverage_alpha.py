"""XY rasters: sparse pixels are faded (alpha) by default; hatch is still available;
the pipeline writes a linear-interpolated *_imshow and a binned *_imshow_binned figure."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import plot_water_content as pwc


def _grid_points(spacing=2.0, nx=12, ny=6):
    xs, ys = np.meshgrid(np.arange(nx) * spacing, np.arange(ny) * spacing)
    x, y = xs.ravel(), ys.ravel()
    return x, y, 10.0 + 0.5 * x + 0.2 * y


def test_coverage_alpha_ramp():
    a = pwc._coverage_alpha(np.array([0, 1, 3, 4, 50]), threshold=3, alpha_min=0.4)
    assert a[0] == pytest.approx(0.4)
    assert np.all(np.diff(a) >= 0)
    assert a[3] == pytest.approx(1.0) and a[4] == pytest.approx(1.0)   # > threshold: opaque


def test_panel_default_fades_sparse_pixels():
    x, y, v = _grid_points()
    fig, ax = plt.subplots()
    im = pwc._plot_xy_panel(ax, v, x, y, render_mode="imshow", interp_method="linear",
                            pixel_size_m=1.0)
    alpha = im.get_alpha()
    assert alpha is not None and np.ndim(alpha) == 2 and alpha.shape == im.get_array().shape
    assert alpha.min() < 1.0                      # 1 pt per ~2 m: most pixels are sparse here
    plt.close(fig)


def test_panel_hatch_style_keeps_uniform_alpha():
    x, y, v = _grid_points()
    fig, ax = plt.subplots()
    im = pwc._plot_xy_panel(ax, v, x, y, render_mode="imshow", interp_method="linear",
                            pixel_size_m=1.0, coverage_style="hatch")
    assert im.get_alpha() is None
    plt.close(fig)


def test_panel_rejects_unknown_style():
    x, y, v = _grid_points()
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        pwc._plot_xy_panel(ax, v, x, y, render_mode="imshow", coverage_style="blur")
    plt.close(fig)


def test_linear_has_no_blank_pixels_binned_does():
    x, y, v = _grid_points()
    fig, ax = plt.subplots()
    lin = pwc._plot_xy_panel(ax, v, x, y, render_mode="imshow", interp_method="linear", pixel_size_m=1.0)
    binned = pwc._plot_xy_panel(ax, v, x, y, render_mode="imshow", interp_method="mean", pixel_size_m=1.0)
    assert not np.ma.getmaskarray(np.ma.masked_invalid(lin.get_array())).any()
    assert np.ma.getmaskarray(np.ma.masked_invalid(binned.get_array())).any()
    plt.close(fig)


def test_plot_water_content_xy_writes_file(tmp_path):
    x, y, v = _grid_points()
    out = tmp_path / "xy.png"
    pwc.plot_water_content_xy(v, x, y, render_mode="imshow", interp_method="linear",
                              bandwidth_m=5.0, output_path=str(out))
    assert out.is_file() and out.stat().st_size > 0
