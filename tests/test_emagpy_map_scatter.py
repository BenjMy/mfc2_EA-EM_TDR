"""ECa plan-view panels must be point scatters (no tricontourf interpolation)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
emagpy = pytest.importorskip("emagpy")
import plot_emagpy_diagnostics as ped  # noqa: E402


@pytest.fixture
def problem(tmp_path):
    rng = np.random.default_rng(0)
    x, y = np.meshgrid(np.arange(10.0), np.arange(6.0))
    x, y = x.ravel(), y.ravel()
    df = pd.DataFrame({"x": x, "y": y,
                       "HCP0.32": 20 + x + rng.normal(0, 1, x.size),
                       "HCP0.71": 30 + y + rng.normal(0, 1, x.size)})
    f = tmp_path / "s.csv"
    df.to_csv(f, index=False)
    k = emagpy.Problem()
    k.createSurvey(str(f))
    return k, x.size


def test_map_is_scatter_one_dot_per_point(problem):
    k, n = problem
    fig, ax = plt.subplots()
    ped._panel_map(k, ax)
    assert len(ax.collections) == 1 and isinstance(ax.collections[0], PathCollection)
    assert len(ax.collections[0].get_offsets()) == n
    assert ax.get_xlabel() == "x [m]" and ax.get_ylabel() == "y [m]"
    plt.close(fig)


def test_scatter_skips_nan_points():
    fig, ax = plt.subplots()
    ped._scatter_xy(ax, [0, 1, 2], [0, 1, 2], [1.0, np.nan, 3.0], "z")
    assert len(ax.collections[0].get_offsets()) == 2
    plt.close(fig)
