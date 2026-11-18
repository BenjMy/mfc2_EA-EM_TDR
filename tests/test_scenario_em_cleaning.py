"""Scenario fields of the raw-ECa cleaning chain (were getattr-only in run_pipeline.py)."""
import json

import pytest

import scenario_loader as sl


def _cfg(tmp_path, **over):
    base = {"description": "t", "output_dir": "o", "depths0": [0.3], "fixed_depths": True,
            "param_grid": {"method": ["ROPE"], "forwardModel": ["CS"]},
            "cond_ranges": [[0.01, 300.0]] * 2}
    base.update(over)
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"s": base}))
    return sl.load_scenario("s", config_path=str(p))


def test_defaults_are_the_values_run_pipeline_always_used(tmp_path):
    c = _cfg(tmp_path)
    assert (c.filter_repeated_tolerance, c.filter_range_vmin, c.filter_range_vmax) == (1e-10, 0.0, None)
    assert (c.filter_percentile_qmin, c.filter_percentile_qmax, c.rolling_mean_window) == (5.0, 95.0, 5)


def test_can_switch_the_chain_off(tmp_path):
    c = _cfg(tmp_path, rolling_mean_window=None, filter_percentile_qmin=None,
             filter_percentile_qmax=None)
    assert c.rolling_mean_window is None
    assert c.filter_percentile_qmin is None and c.filter_percentile_qmax is None


def test_range_filter_can_be_opened(tmp_path):
    c = _cfg(tmp_path, filter_range_vmin=None, filter_range_vmax=500)
    assert c.filter_range_vmin is None and c.filter_range_vmax == 500.0


@pytest.mark.parametrize("over, msg", [
    ({"filter_percentile_qmin": None}, "both be set or both null"),
    ({"filter_percentile_qmin": 60, "filter_percentile_qmax": 40}, "qmin < qmax"),
    ({"rolling_mean_window": 0}, "rolling_mean_window"),
    ({"rolling_mean_window": 2.5}, "rolling_mean_window"),
    ({"filter_range_vmin": "x"}, "filter_range_vmin"),
])
def test_bad_values_are_rejected(tmp_path, over, msg):
    with pytest.raises(ValueError, match=msg):
        _cfg(tmp_path, **over)
