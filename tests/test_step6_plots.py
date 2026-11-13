"""steps/step6_plots.run_step6d_solution_plots: wiring, synthetic truth overlay, warning."""
from types import SimpleNamespace

import numpy as np
import pytest

from steps import step6_plots


class FakeReporter:
    def __init__(self):
        self.lines, self.started, self.ended = [], [], []

    def step(self, name, title=""):
        self.started.append(name)

    def end_step(self, name):
        self.ended.append(name)

    def log(self, msg):
        self.lines.append(msg)

    def metric(self, name, value):
        self.metrics = getattr(self, "metrics", {})
        self.metrics[name] = value


@pytest.fixture
def calls(monkeypatch):
    rec = {}

    def fake_swc(**kw):
        rec["swc"] = kw
        return "swc.png"

    def fake_ec(**kw):
        rec["ec"] = kw
        return "ec.png"

    monkeypatch.setattr(step6_plots.psol, "plot_solution_swc", fake_swc)
    monkeypatch.setattr(step6_plots.psol, "plot_solution_ec_bulk", fake_ec)
    return rec


def _run(tmp_path, data_source, rep, **over):
    sel = SimpleNamespace(theta_calc=np.zeros((2, 3)), K=2, rmse=0.03)
    kw = dict(data_source=data_source, theta_true_realization="THETA",
              output_model_dir=str(tmp_path / "output_model"), output_dir=str(tmp_path),
              realization_index=1, selected_output=sel,
              averaged_models=[np.zeros(1), np.ones(1)], depths0=np.array([0.2, 0.5]),
              layer_centers=np.array([0.1, 0.35, 0.8]), cfg=SimpleNamespace(name="sc"),
              timelapse=True)
    kw.update(over)
    return step6_plots.run_step6d_solution_plots(rep, **kw)


def test_synthetic_overlays_truth_and_returns_paths(tmp_path, calls):
    (tmp_path / "output_model").mkdir()
    ec = np.arange(24, dtype=float).reshape(3, 2, 4)
    np.save(tmp_path / "output_model" / "synthetic_time_lapse_ec.npy", ec)
    rep = FakeReporter()
    assert _run(tmp_path, "synthetic", rep) == ("swc.png", "ec.png")
    assert calls["swc"]["theta_true"] == "THETA" and calls["swc"]["K"] == 2
    np.testing.assert_array_equal(calls["ec"]["ec_bulk_true"], ec[1])
    np.testing.assert_array_equal(calls["ec"]["ec_optimal"], np.ones(1))  # K - 1 = 1
    assert calls["swc"]["scenario_name"] == "sc"
    assert rep.started == ["Step 6d"] and rep.ended == ["Step 6d"]


def test_synthetic_without_ec_file_warns_and_skips_overlay(tmp_path, calls):
    rep = FakeReporter()
    _run(tmp_path, "synthetic", rep)
    assert calls["ec"]["ec_bulk_true"] is None
    assert any("[STEP 6d][WARN]" in line for line in rep.lines)


def test_real_data_has_no_truth_and_default_scenario_name(tmp_path, calls):
    _run(tmp_path, "real", FakeReporter(), cfg=None)
    assert calls["swc"]["theta_true"] is None and calls["ec"]["ec_bulk_true"] is None
    assert calls["swc"]["scenario_name"] == "default"


def test_step6e_forwards_arguments_and_records_one_metric_per_time(monkeypatch, tmp_path):
    seen = {}

    def fake_forward(**kw):
        seen.update(kw)
        return [(0, 1.25), (1, 0.5)]

    monkeypatch.setattr(step6_plots.psol, "forward_simulate_eca_optimal", fake_forward)
    rep = FakeReporter()
    out = step6_plots.run_step6e_forward_eca(
        rep, em_fnames=["a.csv"], averaged_models=[1], selected_output="SEL",
        depths0=[0.2], output_dir=str(tmp_path), x_transect=[0.0], forward_model="FSlin",
        cfg=None, coils_list=["HCP05"], matched_indices=None)
    assert out == [(0, 1.25), (1, 0.5)]
    assert seen["csv_paths"] == ["a.csv"] and seen["forward_model"] == "FSlin"
    assert seen["coils_list"] == ["HCP05"] and seen["matched_indices"] is None
    assert seen["scenario_name"] == "default"
    assert rep.metrics == {"Forward ECa RMSE (optimal, t=0)": "1.2500 mS/m",
                           "Forward ECa RMSE (optimal, t=1)": "0.5000 mS/m"}
    assert rep.started == ["Step 6e"] and rep.ended == ["Step 6e"]


def test_step6c_runs_stage_plots_then_one_to_one_and_closes_step(monkeypatch, tmp_path):
    order, seen = [], {}
    monkeypatch.setattr(step6_plots.pwc, "plot_swc_ensemble_stages",
                        lambda **kw: (order.append("stages"), seen.setdefault("stages", kw)))
    monkeypatch.setattr(step6_plots.pwc, "plot_one_to_one_stages",
                        lambda **kw: (order.append("1to1"), seen.setdefault("1to1", kw)))
    fits = ["fit1", "fit2", "fit3"]
    rep = FakeReporter()
    step6_plots.run_step6c_swc_stage_plots(
        rep, models_clean="M", averaged_models=["a1", "a2", "a3"],
        selected_output=SimpleNamespace(K=2), fits_step4=fits, x_transect="X",
        depths0="D", output_dir=str(tmp_path), cfg=SimpleNamespace(name="sc"),
        timelapse=True, utm_x="ux", utm_y="uy", probe_x_coords="px", probe_y_coords="py",
        probe_depths="pd", theta_obs="obs", sample_positions="sp", sample_times="st",
        probe_layer_indices="pli")
    assert order == ["stages", "1to1"]
    assert seen["stages"]["best_fit"] == "fit2" and seen["1to1"]["best_fit"] == "fit2"
    assert seen["stages"]["max_realisations"] == 4 and seen["stages"]["survey_labels"] is None
    assert seen["1to1"]["theta_obs"] == "obs" and seen["1to1"]["probe_layer_indices"] == "pli"
    assert seen["stages"]["probe_x_coords"] == "px"
    assert rep.started == ["Step 6c"] and rep.ended == ["Step 6c"]


def test_step6b_calls_per_realisation_then_optimal(monkeypatch, tmp_path):
    order, seen = [], {}
    monkeypatch.setattr(step6_plots.pediag, "run_emagpy_plots_per_realisation",
                        lambda **kw: (order.append("per"), seen.setdefault("per", kw)))
    monkeypatch.setattr(step6_plots.pediag, "run_emagpy_plots_optimal",
                        lambda **kw: (order.append("opt"), seen.setdefault("opt", kw)))
    rep = FakeReporter()
    step6_plots.run_step6b_emagpy_plots(
        rep, em_fnames=["a.csv", "b.csv"], mode="1d",
        selected_output=SimpleNamespace(K=3), models_clean=np.zeros((5, 2, 4)),
        averaged_models=[1, 2, 3], tasks=["t"], depths_clean="dc", depths0="d0",
        output_dir=str(tmp_path), fixed_depths=True, invert_kwargs={"alpha": 0.03},
        em_unit="mS/m")
    assert order == ["per", "opt"]
    assert seen["per"]["max_realisations"] == 1 and seen["per"]["tasks"] == ["t"]
    assert seen["opt"]["invert_kwargs"] == {"alpha": 0.03} and seen["opt"]["conds0"] is None
    assert seen["opt"]["fixed_depths"] is True and seen["opt"]["em_unit"] == "mS/m"
    assert any("surveys        : 2" in line for line in rep.lines)
    assert any("ensemble size  : 5" in line for line in rep.lines)
    assert rep.started == ["Step 6b"] and rep.ended == ["Step 6b"]


def _kw(tmp_path, data_source, **over):
    kw = dict(mode="1d", data_source=data_source, selected_output="SEL", x_transect="X",
              depths0="D", probe_x_coords="px", probe_y_coords="py", probe_depths="pd",
              probe_x_coords_utm="pux", probe_y_coords_utm="puy", probe_labels="pl",
              utm_x="ux", utm_y="uy", output_dir=str(tmp_path),
              output_model_dir=str(tmp_path / "om"), scenario_name="sc", timelapse=True,
              xy_render_mode="scatter", window_m=3.0, theta_true_realization="TRUE",
              realization_index=1, cond_ranges="CR", coils_list=["HCP05"],
              forward_model="FSlin")
    kw.update(over)
    return kw


def _stub_pwc(monkeypatch, fail=None):
    seen = {}

    def mk(name, ret=None):
        def f(**kw):
            if name == fail:
                raise RuntimeError("boom")
            seen[name] = kw
            return ret
        return f
    em = {"xz_rmse_path": "r.png", "xz_mae_path": "m.png",
          "xz_mae_overall": 1.5, "xz_rmse_overall": 2.5}
    monkeypatch.setattr(step6_plots.pwc, "plot_from_pipeline", mk("main"))
    monkeypatch.setattr(step6_plots.pwc, "plot_truth_from_pipeline", mk("truth"))
    monkeypatch.setattr(step6_plots.pwc, "plot_error_metrics_from_pipeline", mk("err", em))
    monkeypatch.setattr(step6_plots.pwc, "plot_initial_model_vs_true_ec", mk("iniec", "iec.png"))
    monkeypatch.setattr(step6_plots.pwc, "plot_initial_eca_vs_true_eca", mk("inieca", "ieca.png"))
    return seen


def test_step6_real_only_main_plot_and_opens_step_without_closing(monkeypatch, tmp_path):
    seen = _stub_pwc(monkeypatch)
    rep = FakeReporter()
    step6_plots.run_step6_water_content(rep, **_kw(tmp_path, "real"))
    assert list(seen) == ["main"] and rep.started == ["Step 6"] and rep.ended == []
    m = seen["main"]
    assert m["scenario_name"] == "sc" and m["probe_utm_x"] == "pux" and m["probe_ids"] == "pl"
    assert m["probe_x_coords"] == "px" and m["window_m"] == 3.0 and m["survey_labels"] is None
    assert any("NO (1-D mode)" in l for l in rep.lines)


def test_step6_synthetic_adds_truth_error_metrics_and_initial_model(monkeypatch, tmp_path):
    seen = _stub_pwc(monkeypatch)
    (tmp_path / "om").mkdir()
    np.save(tmp_path / "om" / "synthetic_time_lapse_ec.npy", np.zeros((3, 2, 4)))
    np.save(tmp_path / "om" / "synthetic_time_lapse_eca.npy", np.zeros((3, 2, 6)))
    rep = FakeReporter()
    step6_plots.run_step6_water_content(rep, **_kw(tmp_path, "synthetic"))
    assert list(seen) == ["main", "truth", "err", "iniec", "inieca"]
    assert seen["err"]["xy_render_mode"] == "imshow" and seen["truth"]["theta_true"] == "TRUE"
    assert seen["inieca"]["coils"] == ["HCP05"] and seen["inieca"]["forward_model"] == "FSlin"
    assert rep.metrics == {"Water Content MAE vs Truth (XZ)": "1.500 %VWC",
                           "Water Content RMSE vs Truth (XZ)": "2.500 %VWC"}


def test_step6_synthetic_failures_warn_and_continue(monkeypatch, tmp_path):
    seen = _stub_pwc(monkeypatch, fail="truth")
    rep = FakeReporter()
    step6_plots.run_step6_water_content(rep, **_kw(tmp_path, "synthetic"))
    assert any("[STEP 6][WARN] plot_truth_from_pipeline failed: boom" in l for l in rep.lines)
    assert "err" in seen and any("synthetic_time_lapse_ec.npy' not found" in l for l in rep.lines)
