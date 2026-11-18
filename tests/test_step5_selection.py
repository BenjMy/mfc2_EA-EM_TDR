"""steps/step5_selection: select call, reporting, persisted arrays, quality gate."""
import os
from types import SimpleNamespace

import numpy as np

from steps import step5_selection as s5


class Rep:
    def __init__(self):
        self.lines, self.started, self.metrics = [], [], {}

    def step(self, n, t=""): self.started.append((n, t))
    def log(self, m): self.lines.append(m)
    def metric(self, n, v): self.metrics[n] = v


def _selected(theta=True, with_fit=True, coverage=True):
    horizon_fits = [SimpleNamespace(is_calibrated=True), SimpleNamespace(is_calibrated=False)]
    return SimpleNamespace(
        K=2, rmse=0.0321, cv_map=np.full((1, 3, 2), 0.2),
        theta_calc=np.full((1, 3, 2), 0.25) if theta else None,
        coverage=np.ones((1, 3, 2)) if coverage else None,
        best_fit=SimpleNamespace(params=np.arange(4.0), horizon_fits=horizon_fits) if with_fit else None)


def test_select_opens_step_and_forwards_arguments(monkeypatch):
    seen = {}
    monkeypatch.setattr(s5.step4_5, "run_select_assess", lambda **kw: seen.update(kw) or "SEL")
    rep = Rep()
    out = s5.run_step5_select(
        rep, models_clean="m", averaged_models="a", ranking="r", sample_positions="p",
        sample_times="t", theta_obs="o", fits_step4="f4", depths0="d", output_dir="out",
        probe_layer_indices="li")
    assert out == "SEL" and rep.started[0][0] == "Step 5"
    # Step 4 fits go in as fits_step2= (name expected by step4_select); plots on.
    assert seen["fits_step2"] == "f4" and seen["generate_plots"] is True
    assert seen["models"] == "m" and seen["output_dir"] == "out" and seen["probe_layer_indices"] == "li"


def test_report_logs_k_cv_and_theta():
    rep = Rep()
    s5.report_step5_selection(rep, selected_output=_selected(), n_candidates=7)
    txt = "\n".join(rep.lines)
    assert "K = 2  (out of 7 candidates)" in txt and "0.032100" in txt
    assert "CV map shape" in txt and "n_time=1" in txt and "theta_calc stats" in txt


def test_report_handles_2d_cv_and_missing_theta():
    sel = _selected(theta=False)
    sel.cv_map = np.full((3, 2), 0.1)
    rep = Rep()
    s5.report_step5_selection(rep, selected_output=sel, n_candidates=1)
    txt = "\n".join(rep.lines)
    assert "(n_measured_points=3, n_layers=2)" in txt and "theta_calc" not in txt


def test_persist_writes_expected_arrays(tmp_path):
    rep = Rep()
    keys = s5.persist_step5_result(
        rep, selected_output=_selected(), averaged_models=[np.zeros((1, 3, 2)), np.ones((1, 3, 2))],
        theta_obs=[0.2, 0.3], sample_positions=[0, 1], sample_times=[0, 0],
        probe_layer_indices=[0, 1], output_model_dir=str(tmp_path))
    assert keys == ["K", "rmse", "cv_map", "theta_calc", "ws_params", "tdr_train_theta",
                    "tdr_train_positions", "tdr_train_times", "tdr_train_layers",
                    "coverage", "is_calibrated", "ec_optimal"]
    for k in keys:
        assert os.path.exists(tmp_path / f"{k}.npy")
    assert int(np.load(tmp_path / "K.npy")) == 2
    assert np.load(tmp_path / "is_calibrated.npy").tolist() == [True, False]
    assert np.all(np.load(tmp_path / "ec_optimal.npy") == 1.0)  # averaged_models[K - 1]


def test_persist_tolerates_missing_pieces(tmp_path):
    rep = Rep()
    keys = s5.persist_step5_result(
        rep, selected_output=_selected(theta=False, with_fit=False, coverage=False),
        averaged_models=[], theta_obs=[0.2], sample_positions=[0], sample_times=[0],
        probe_layer_indices=None, output_model_dir=str(tmp_path))
    assert keys == ["K", "rmse", "cv_map"]
    assert sum("[STEP 5][WARN]" in l for l in rep.lines) == 2  # WS params + ec_optimal


def test_quality_gate_uses_cfg_defaults_and_reports(monkeypatch):
    seen = {}
    res = SimpleNamespace(message="gate msg", level="WARN", cv_mean=0.4, cv_high_fraction=0.1,
                          n_calibrated_horizons=1, n_total_horizons=2, reasons=["r"])
    monkeypatch.setattr(s5.EMmethod, "evaluate_quality_gates", lambda **kw: seen.update(kw) or res)
    rep = Rep()
    out = s5.run_step5_quality_gate(rep, selected_output=_selected(), cfg=None)
    assert out is res
    assert seen["cv_mean_warn_threshold"] == 0.3 and seen["cv_high_threshold"] == 1.0
    assert seen["cv_high_fraction_warn_threshold"] == 0.3
    assert seen["require_all_horizons_calibrated"] is True
    assert rep.metrics["Quality gate level"] == "WARN"
    assert rep.metrics["Quality gate — calibrated horizons"] == "1/2"
    assert any("[PIPELINE][WARN]" in l for l in rep.lines)


def test_quality_gate_reads_cfg_overrides(monkeypatch):
    seen = {}
    res = SimpleNamespace(message="ok", level="OK", cv_mean=0.0, cv_high_fraction=0.0,
                          n_calibrated_horizons=2, n_total_horizons=2, reasons=[])
    monkeypatch.setattr(s5.EMmethod, "evaluate_quality_gates", lambda **kw: seen.update(kw) or res)
    cfg = SimpleNamespace(quality_gate_cv_mean_warn=0.5,
                          quality_gate_require_all_horizons_calibrated=False)
    rep = Rep()
    s5.run_step5_quality_gate(rep, selected_output=_selected(), cfg=cfg)
    assert seen["cv_mean_warn_threshold"] == 0.5
    assert seen["require_all_horizons_calibrated"] is False
    assert not any("[PIPELINE][WARN]" in l for l in rep.lines)
