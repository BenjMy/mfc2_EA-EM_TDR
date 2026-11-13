"""steps/step5b_validation: per-K validation loop, metrics, warnings, plot tolerance."""
from types import SimpleNamespace

import numpy as np

from steps import step5b_validation as s5


class Rep:
    def __init__(self):
        self.lines, self.started, self.ended, self.metrics = [], [], [], {}

    def step(self, n, t=""): self.started.append(n)
    def end_step(self, n): self.ended.append(n)
    def log(self, m): self.lines.append(m)
    def metric(self, n, v): self.metrics[n] = v


def _setup(monkeypatch, val_rmses, plot_exc=None):
    calls = []

    def fake_run_petro(**kw):
        i = len(calls)
        calls.append(kw)
        return [SimpleNamespace(rmse=val_rmses[i], observed_theta="vo", predicted_theta="vp")]

    monkeypatch.setattr(s5.step2, "run_petro", fake_run_petro)
    plots = []

    def pl(name):
        def f(**kw):
            if plot_exc:
                raise plot_exc
            plots.append((name, kw))
        return f
    monkeypatch.setattr(s5.ptv, "plot_rmse_vs_k", pl("k"))
    monkeypatch.setattr(s5.ptv, "plot_parity_train_val", pl("parity"))
    return calls, plots


def _run(tmp_path, fixed_depths=True, independent=True, K=2):
    fits = [SimpleNamespace(params=np.full((2, 4), float(i)), observed_theta="to",
                            predicted_theta="tp") for i in range(3)]
    rep = Rep()
    out = s5.run_step5b_validation(
        rep, split=SimpleNamespace(train=[0, 1], val=[2], independent=independent, method="m"),
        selected_output=SimpleNamespace(K=K, rmse=0.04), fits_step4=fits,
        train_rmse_by_k=[0.05, 0.04, 0.03], averaged_models=[np.zeros((1, 2)), np.ones((1, 2)), np.ones((1, 2))],
        sample_positions_val=[0], sample_times_val=[0], theta_obs_val=[0.2],
        probe_layer_val=[0], depths0=[0.2], output_dir=str(tmp_path), fixed_depths=fixed_depths,
        probe_depths_reflect_fitted_interfaces=False, theta_obs_std=None)
    return out, rep


def test_loop_pins_train_params_per_k_and_returns_rmses(monkeypatch, tmp_path):
    calls, plots = _setup(monkeypatch, [0.05, 0.045, 0.06])
    (tr, va, by_k), rep = _run(tmp_path)
    assert len(calls) == 3 and tr == 0.04 and va == 0.045
    np.testing.assert_array_equal(by_k, [0.05, 0.045, 0.06])
    assert all(c["fixed_params"] == {"a": True, "m": True, "sigma_w": True, "sigma_s": True} for c in calls)
    assert [float(c["lab_p0"][0, 0]) for c in calls] == [0.0, 1.0, 2.0]
    assert calls[1]["models"].shape == (1, 1, 2) and calls[0]["generate_plots"] is False
    assert [p[0] for p in plots] == ["k", "parity"]
    assert rep.metrics["Validation RMSE"] == "0.04500 m³/m³"
    assert rep.metrics["Train↔Val RMSE gap"] == "+0.00500 m³/m³"
    assert not any("[WARN]" in l for l in rep.lines) and rep.ended == ["Step 5b"]


def test_per_horizon_fixed_depths_reduce_with_all(monkeypatch, tmp_path):
    calls, _ = _setup(monkeypatch, [0.04] * 3)
    _run(tmp_path, fixed_depths=[True, False])
    assert calls[0]["depths_are_fixed"] is False


def test_large_gap_and_non_independent_split_are_flagged(monkeypatch, tmp_path):
    _setup(monkeypatch, [0.05, 0.08, 0.06])
    _, rep = _run(tmp_path, independent=False)
    assert any("diverges >30%" in l for l in rep.lines)
    assert any("[NOTE]" in l and "not fully independent" in l for l in rep.lines)


def test_plot_failure_warns_but_returns(monkeypatch, tmp_path):
    _setup(monkeypatch, [0.04] * 3, plot_exc=RuntimeError("png"))
    (tr, va, _), rep = _run(tmp_path)
    assert va == 0.04 and rep.ended == ["Step 5b"]
    assert any("[STEP 5b][WARN]" in l and "RuntimeError: png" in l for l in rep.lines)
