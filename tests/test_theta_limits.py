"""theta colour limits: historical 0..p98 default, optional percentile mode, shared truth/inverted scale."""
import inspect

import numpy as np
import pytest

import plot_water_content as pwc


def test_zero_p98_matches_historical_formula():
    rng = np.random.default_rng(0)
    th = rng.uniform(12, 33, size=(2, 50, 3))
    old = max(5.0, np.ceil(float(np.nanpercentile(th, 98)) / 5) * 5)
    assert pwc._theta_color_limits([th]) == (0.0, old)


def test_percentile_follows_the_data():
    th = np.linspace(14.2, 33.7, 1000)
    vmin, vmax = pwc._theta_color_limits([th], "percentile")
    assert 14.0 <= vmin <= 16.0 and 32.0 <= vmax <= 34.0
    assert vmin == np.floor(vmin) and vmax == np.ceil(vmax)       # whole %VWC


def test_percentile_pools_arrays_and_skips_none_and_nan():
    a = np.array([20.0, 21.0, np.nan])
    b = np.array([30.0, 31.0])
    lo, hi = pwc._theta_color_limits([a, None, b], "percentile", (0, 100))
    assert (lo, hi) == (20.0, 31.0)


def test_degenerate_range_keeps_vmax_above_vmin():
    vmin, vmax = pwc._theta_color_limits([np.full(10, 25.0)], "percentile")
    assert vmax > vmin


def test_empty_input_and_bad_mode():
    assert pwc._theta_color_limits([np.array([np.nan])]) == (0.0, 45.0)
    with pytest.raises(ValueError):
        pwc._theta_color_limits([np.ones(3)], "log")


def test_pipeline_plotters_accept_limits():
    for fn in (pwc.plot_from_pipeline, pwc.plot_truth_from_pipeline):
        params = inspect.signature(fn).parameters
        assert {"theta_limits", "theta_vmin", "theta_vmax"} <= set(params)
        assert params["theta_limits"].default == "zero_p98"       # default unchanged
