"""steps/step1_diagnostics: wiring, metrics, and failures never abort."""
import numpy as np

from steps import step1_diagnostics as s1


class Rep:
    def __init__(self):
        self.lines, self.started, self.ended, self.metrics = [], [], [], {}

    def step(self, name, title=""):
        self.started.append(name)

    def end_step(self, name):
        self.ended.append(name)

    def log(self, msg):
        self.lines.append(msg)

    def metric(self, name, value):
        self.metrics[name] = value


def test_1b_counts_figures_and_forwards_args(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(s1.pediag_real, "plot_real_em_diagnostics",
                        lambda **kw: (seen.update(kw), ["a", "b"])[1])
    rep = Rep()
    n = s1.run_step1b_em_diagnostics(rep, raw_em_files=["x.csv"], min_pts=7,
                                     probe_table="PT", output_dir=str(tmp_path), em_unit="mS/m")
    assert n == 2 and seen["min_pts"] == 7 and seen["probe_table"] == "PT"
    assert rep.metrics == {"Real-data EM diagnostic figures": 2}
    assert rep.started == rep.ended == ["Step 1b"]


def test_1b_and_1c_failures_warn_and_close_step(monkeypatch, tmp_path):
    def boom(**kw):
        raise RuntimeError("bad")
    monkeypatch.setattr(s1.pediag_real, "plot_real_em_diagnostics", boom)
    monkeypatch.setattr(s1.ptdrdiag, "plot_tdr_diagnostics", boom)
    rep = Rep()
    assert s1.run_step1b_em_diagnostics(rep, raw_em_files=[], min_pts=0, probe_table=None,
                                        output_dir=str(tmp_path), em_unit="mS/m") == 0
    assert s1.run_step1c_tdr_diagnostics(rep, tdr_dir="d", coord_file="c", em_dates=[],
                                         time_tolerance_days=1.0, output_dir=str(tmp_path)) == 0
    assert sum("[WARN]" in l and "RuntimeError: bad" in l for l in rep.lines) == 2
    assert rep.ended == ["Step 1b", "Step 1c"] and rep.metrics == {}


def test_1d_runs_both_plots_and_survives_first_failure(monkeypatch, tmp_path):
    seen = {}

    def boom(**kw):
        raise ValueError("no data")
    monkeypatch.setattr(s1.pim, "plot_initial_model", boom)
    monkeypatch.setattr(s1.initial_ec_ws_real, "plot_initial_ec_ws_real",
                        lambda out, **kw: (seen.update(out=out, **kw), {"ec": "e.png", "ws": "w.png"})[1])
    rep = Rep()
    paths = s1.run_step1d_initial_model(
        rep, em_dir="e", tdr_dir="t", coord_file="c", output_dir=str(tmp_path), em_unit="mS/m",
        cond_ranges="CR", depths0=np.array([0.2]), lab_p0=[1], ws_fixed_params={"a": True},
        ws_bounds=("lo", "hi"), observations=[1], probe_table="PT")
    assert paths == {} and seen["out"] == str(tmp_path) and seen["ws_bounds"] == ("lo", "hi")
    assert any("initial EC/WS" not in l and "Initial model plots failed" in l for l in rep.lines)
    assert any("e.png" in l for l in rep.lines) and rep.ended == ["Step 1d"]
