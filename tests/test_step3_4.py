"""steps/step3_4_average_recalibrate: wiring, logs, and reporting of Steps 3 and 4."""
from types import SimpleNamespace

import numpy as np

from steps import step3_4_average_recalibrate as s34


class Rep:
    def __init__(self):
        self.lines, self.started, self.tables = [], [], []

    def step(self, n, t=""): self.started.append(n)
    def log(self, m): self.lines.append(m)
    def args_table(self, title, rows): self.tables.append((title, rows))


def test_step3_returns_ranking_and_logs_best_worst_and_shape(monkeypatch):
    seen = {}
    avg = [np.zeros((2, 3, 4)), np.zeros((2, 3, 4))]
    monkeypatch.setattr(s34.step3, "run_average",
                        lambda **kw: (seen.update(kw), (avg, [2, 0, 1]))[1])
    fits = [SimpleNamespace(rmse=r) for r in (0.05, 0.07, 0.03)]
    rep = Rep()
    out = s34.run_step3_average(rep, models_clean="M", fits_step2=fits, depths0="D",
                                output_dir="o", timelapse=True)
    assert out == (avg, [2, 0, 1]) and seen["models"] == "M" and rep.started == ["Step 3"]
    assert any("Best  member  : index 2,  RMSE = 0.03000" in l for l in rep.lines)
    assert any("Worst member  : index 1, RMSE = 0.07000" in l for l in rep.lines)
    assert any("n_time=2, n_measured_points=3, n_layers=4" in l for l in rep.lines)


def test_step3_single_time_shape_message(monkeypatch):
    monkeypatch.setattr(s34.step3, "run_average", lambda **kw: ([np.zeros((3, 4))], [0]))
    rep = Rep()
    s34.run_step3_average(rep, models_clean=None, fits_step2=[SimpleNamespace(rmse=0.1)],
                          depths0=None, output_dir="o", timelapse=False)
    assert any("(n_measured_points=3, n_layers=4)" in l for l in rep.lines)


def _hz(h, cal):
    return SimpleNamespace(horizon=h, is_calibrated=cal, a=1.0, m=2.0, sigma_w=3.0,
                           sigma_s=4.0, rmse=0.02)


def test_step4_forwards_args_and_tabulates_best_k(monkeypatch):
    seen = {}
    fits = [SimpleNamespace(rmse=0.05, horizon_fits=[_hz(0, True)]),
            SimpleNamespace(rmse=0.02, horizon_fits=[_hz(0, True), _hz(1, False)])]
    monkeypatch.setattr(s34.EMmethod, "recalibrate_petrophysical_relationship",
                        lambda **kw: (seen.update(kw), fits)[1])
    rep = Rep()
    out, rmse = s34.run_step4_recalibrate(
        rep, averaged_models=["a"], sample_positions=[0], sample_times=[0], theta_obs=[0.2],
        depths0=[0.2], probe_layer_indices=[0], fixed_depths=[True, False],
        ws_fixed_params={"a": True}, probe_depths_reflect_fitted_interfaces=True,
        lab_p0=[1], theta_obs_std=None, ws_bounds_override={"a": [0.2, 5]})
    assert out is fits and rmse == [0.05, 0.02] and rep.started == ["Step 4"]
    assert seen["depths_are_fixed"] is False and seen["ws_bounds"] == {"a": [0.2, 5]}
    title, rows = rep.tables[0]
    assert "K=2" in title and "RMSE=0.02000" in title
    assert any(v.startswith("UNCALIBRATED") for v in rows.values()) and len(rows) == 2
