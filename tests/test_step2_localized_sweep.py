"""steps/step2_localized_sweep.run_step2_localized_sweep with stubs (no emagpy, no inversion)."""
from types import SimpleNamespace

import numpy as np
import pytest

from steps import step2_localized_sweep as s2


class Reporter:
    def __init__(self):
        self.calls = []

    def step(self, *a, **k):
        self.calls.append(("step", a))

    def end_step(self, *a, **k):
        self.calls.append(("end_step", a))

    def log(self, msg):
        self.calls.append(("log", msg))

    def table(self, *a, **k):
        self.calls.append(("table", a, k))

    def metric(self, name, value):
        self.calls.append(("metric", name, value))

    def logs(self):
        return [c[1] for c in self.calls if c[0] == "log"]


ROWS = [SimpleNamespace(bandwidth=b, loo_rmse=r, n_folds=18, n_fallback=f)
        for b, r, f in [(15.0, 0.1126, 0), (25.0, 0.1100, 0), (50.0, float("nan"), 3)]]


def _call(rep, bandwidths=(15.0, 25.0, 50.0), bandwidth=None, x=None, **over):
    kw = dict(bandwidths=list(bandwidths), bandwidth=bandwidth,
              fits_step2=[SimpleNamespace(model_index=2, rmse=0.02)],
              models_clean=[f"model{i}" for i in range(4)],
              sample_positions=[0, 1, 1, 2], sample_times=[0, 0, 1, 0], theta_obs=np.zeros(4),
              x_transect=np.array([0.0, 25.0, 50.0]) if x is None else x, localized_y=None,
              probe_layer_indices=[0, 0, 1, 0], lab_p0="p0", fixed_params={"m": True},
              theta_obs_std=0.02, ws_bounds={"a": [0.2, 5]}, output_dir="/out")
    kw.update(over)
    return s2.run_step2_localized_sweep(rep, **kw)


def test_success_returns_recommended_and_reports(monkeypatch):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return ROWS, 25.0

    monkeypatch.setattr(s2.step2, "run_bandwidth_sweep", fake)
    rep = Reporter()
    bw, from_sweep, rows = _call(rep, bandwidth=7.0)
    assert (bw, from_sweep, rows) == (25.0, True, ROWS)
    # arguments reach the sweep unchanged; the reference model is the rank-1 member's
    assert seen["sigma_b_reference"] == "model2" and seen["bandwidths"] == [15.0, 25.0, 50.0]
    assert seen["output_dir"] == "/out" and seen["ws_bounds"] == {"a": [0.2, 5]}
    assert seen["point_y_coords"] is None and seen["theta_obs_std"] == 0.02
    kinds = [c[0] for c in rep.calls]
    assert kinds[0] == "step" and kinds[-1] == "end_step" and "table" in kinds
    table = next(c for c in rep.calls if c[0] == "table")
    assert table[1][2][2][:2] == ["50", "N/A"]          # table rows: (title, headers, rows)
    metrics = {c[1]: c[2] for c in rep.calls if c[0] == "metric"}
    assert metrics["Step 2 LOCALIZED — recommended bandwidth (c)"] == "25"


def test_failure_keeps_fixed_bandwidth_and_still_closes_step(monkeypatch):
    def boom(**kw):
        raise RuntimeError("no valid folds")

    monkeypatch.setattr(s2.step2, "run_bandwidth_sweep", boom)
    rep = Reporter()
    bw, from_sweep, rows = _call(rep, bandwidth=7.0)
    assert (bw, from_sweep, rows) == (7.0, False, None)
    assert any("Bandwidth sweep failed: no valid folds" in m and "7.0" in m for m in rep.logs())
    assert rep.calls[-1][0] == "end_step"


def test_spacing_warning_logged_when_sweep_cannot_reach_neighbours(monkeypatch):
    monkeypatch.setattr(s2.step2, "run_bandwidth_sweep", lambda **kw: (ROWS, 15.0))
    rep = Reporter()
    _call(rep, bandwidths=(0.5, 5.0, 10.0))               # reach 20 < 25 m spacing
    assert any("[WARN]" in m and "no probe has a neighbour" in m for m in rep.logs())


def test_no_warning_for_adequate_sweep(monkeypatch):
    monkeypatch.setattr(s2.step2, "run_bandwidth_sweep", lambda **kw: (ROWS, 15.0))
    rep = Reporter()
    _call(rep, bandwidths=(15.0, 25.0, 50.0))
    assert not any("no probe has a neighbour" in m or "only " in m and "probe locations" in m
                   for m in rep.logs())


def test_broken_spacing_check_does_not_stop_the_sweep(monkeypatch):
    monkeypatch.setattr(s2.step2, "run_bandwidth_sweep", lambda **kw: (ROWS, 25.0))
    rep = Reporter()
    bw, from_sweep, _ = _call(rep, x=np.array(["a", "b", "c"], dtype=object))
    assert (bw, from_sweep) == (25.0, True)
    assert any("spacing check skipped" in m for m in rep.logs())


def test_2d_distance_is_announced_and_y_is_passed(monkeypatch):
    seen = {}
    monkeypatch.setattr(s2.step2, "run_bandwidth_sweep", lambda **kw: (seen.update(kw) or (ROWS, 25.0)))
    rep = Reporter()
    y = np.array([0.0, 10.0, 20.0])
    _call(rep, localized_y=y)
    assert seen["point_y_coords"] is y
    assert any("2-D Euclidean" in m for m in rep.logs())
