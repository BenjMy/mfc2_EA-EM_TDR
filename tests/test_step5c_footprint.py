"""steps/step5c_footprint: discrepancy records, RMSEs, plot failure tolerated."""
from types import SimpleNamespace

import numpy as np

from steps import step5c_footprint as s5


class Rep:
    def __init__(self):
        self.lines, self.started, self.ended, self.metrics, self.tables = [], [], [], {}, []

    def step(self, n, t=""): self.started.append(n)
    def end_step(self, n): self.ended.append(n)
    def log(self, m): self.lines.append(m)
    def metric(self, n, v): self.metrics[n] = v
    def table(self, *a, **k): self.tables.append(a)


def _run(monkeypatch, plot_exc=None, timelapse=False):
    theta = np.full((1, 3, 2), 0.2)
    theta[0, 1, 0] = 0.3
    calls = []

    def fake(name):
        def f(**kw):
            if plot_exc:
                raise plot_exc
            calls.append(name)
        return f
    for n in ("plot_footprint_probe_comparison", "plot_footprint_spatial_discrepancy",
              "plot_footprint_spatial_discrepancy_xz"):
        monkeypatch.setattr(s5.pef, n, fake(n))
    rep = Rep()
    cp = [SimpleNamespace(probe_x=0.0, probe_y=0.0, probe_depth=0.1)] * 2
    out = s5.run_step5c_footprint(
        rep, selected_output=SimpleNamespace(theta_calc=theta),
        em_layer_weights=np.array([1.0, 0.0]), sample_positions=[0, 1],
        probe_layer_indices=[0, 0], theta_obs=[0.2, 0.2], sample_times=[0, 0],
        timelapse=timelapse, data_source="synthetic", coil_orientation="HCP",
        survey_df=None, calib_position_results=cp, effective_depths0=[0.2],
        output_dir="o")
    return out, rep, calls


def test_records_rmse_and_plots(monkeypatch):
    (disc, hr, w), rep, calls = _run(monkeypatch, timelapse=True)
    assert [d["pos"] for d in disc] == [0, 1] and [d["horizon"] for d in disc] == [0, 0]
    assert np.isclose(hr, np.sqrt(0.5 * 0.1 ** 2)) and np.isclose(hr, w)  # weight all on layer 0
    assert len(calls) == 3 and rep.started == rep.ended == ["Step 5c"]
    assert set(rep.metrics) == {"Footprint RMSE — hard-routed layer", "Footprint RMSE — McNeill-weighted"}
    assert len(rep.tables[0][2]) == 2
    assert any("at least as well" in l for l in rep.lines)


def test_plot_failure_warns_and_still_returns(monkeypatch):
    (disc, hr, w), rep, _ = _run(monkeypatch, plot_exc=RuntimeError("x"))
    assert len(disc) == 2 and rep.ended == ["Step 5c"]
    assert any("[STEP 5c][WARN]" in l and "RuntimeError: x" in l for l in rep.lines)
