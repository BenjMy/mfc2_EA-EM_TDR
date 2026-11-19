"""steps/step0b_coverage: probe-depth source selection, reporting, WARN message."""
from types import SimpleNamespace

from steps import step0b_coverage as s0b


class Rep:
    def __init__(self): self.lines, self.metrics = [], {}
    def log(self, m): self.lines.append(m)
    def metric(self, n, v): self.metrics[n] = v


def _stub(monkeypatch, level="OK", uncovered=()):
    seen = {}
    res = SimpleNamespace(message="cov msg", level=level, uncovered_horizons=list(uncovered))
    monkeypatch.setattr(s0b.EMmethod, "check_early_horizon_coverage",
                        lambda **kw: seen.update(kw) or res)
    return seen, res


def test_real_uses_sorted_unique_observation_depths(monkeypatch):
    seen, res = _stub(monkeypatch)
    rep = Rep()
    obs = [{"depth": 0.3}, {"depth": 0.15}, {"depth": 0.3}]
    out = s0b.run_step0b_coverage(rep, data_source="real", observations_list=obs, cfg=None,
                                  synthetic_geometry=None, depths0=[0.225])
    assert out is res and seen == {"probe_depths": [0.15, 0.3], "depths0": [0.225]}
    assert rep.metrics["Early coverage — level"] == "OK" and rep.lines == ["cov msg"]


def test_synthetic_prefers_cfg_swc_depths(monkeypatch):
    seen, _ = _stub(monkeypatch)
    s0b.run_step0b_coverage(Rep(), data_source="synthetic", observations_list=None,
                            cfg=SimpleNamespace(swc_depths=[0.1, 0.5]),
                            synthetic_geometry={"layer_centers": [9.0]}, depths0=[0.3])
    assert seen["probe_depths"] == [0.1, 0.5]


def test_synthetic_falls_back_to_layer_centers(monkeypatch):
    seen, _ = _stub(monkeypatch)
    for cfg in (None, SimpleNamespace(swc_depths=None), SimpleNamespace()):
        s0b.run_step0b_coverage(Rep(), data_source="synthetic", observations_list=None, cfg=cfg,
                                synthetic_geometry={"layer_centers": [0.15, 0.45]}, depths0=[0.3])
        assert seen["probe_depths"] == [0.15, 0.45]


def test_warn_logs_proceeding_message_and_counts_uncovered(monkeypatch):
    _stub(monkeypatch, level="WARN", uncovered=[2, 3])
    rep = Rep()
    s0b.run_step0b_coverage(rep, data_source="synthetic", observations_list=None, cfg=None,
                            synthetic_geometry={"layer_centers": [0.1]}, depths0=[0.3])
    assert rep.metrics["Early coverage — uncovered horizons"] == 2
    assert any("[PIPELINE][WARN] Proceeding despite" in l for l in rep.lines)
