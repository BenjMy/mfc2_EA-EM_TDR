"""pipeline_plan: what --dry-run reports, and which option combinations it rejects (fast tier)."""
from types import SimpleNamespace

import numpy as np
import pytest

import pipeline_plan as pp


def _cfg(**over):
    kw = dict(name="s", param_grid={"method": ["MCMC"], "forwardModel": ["CS", "FSlin"]},
              alpha_range_log=[0.001, 5.0, 9], gamma=0.0, cond_ranges=[[0.01, 300.0]] * 3,
              train_val_split=None, ws_localized_bandwidth=None, ws_localized_bandwidth_sweep=None,
              output_dir="out", n_time=3)
    kw.update(over)
    return SimpleNamespace(**kw)


def _rows(rows):
    return dict(rows)


def test_full_grid_counts():
    rows, errs = pp.build_plan(_cfg(), mode="1d", data_source="synthetic")
    assert not errs
    r = _rows(rows)
    assert "9 combination" not in r["ensemble grid"] and "18 combination" in r["ensemble grid"]   # 9 alphas x 2 models
    assert r["starting-conductivity combinations (estimate)"] == 8                                 # 2**3 corners
    assert r["Step-1 inversion tasks (estimate)"] == 18 * 8 * 3


def test_quick_keeps_middle_alpha_and_first_model():
    grid, info = pp.effective_param_grid(_cfg(), quick=True)
    assert grid["alpha"] == [pytest.approx(np.logspace(np.log10(0.001), np.log10(5.0), 9)[4])]
    assert grid["forwardModel"] == ["CS"] and set(info) == {"alpha", "forwardModel"}
    rows, _ = pp.build_plan(_cfg(), mode="1d", data_source="synthetic", quick=True)
    r = _rows(rows)
    assert r["Step-1 inversion tasks (estimate)"] == 1 * 8 * 3
    assert r["quick preset"].startswith("ON") and r["output folder (not created)"].endswith("_quick")


def test_beta_gamma_overrides_win():
    grid, _ = pp.effective_param_grid(_cfg(), beta=[1, 2], gamma=[0.1, 0.2, 0.3])
    assert grid["beta"] == [1, 2] and grid["gamma"] == [0.1, 0.2, 0.3]


@pytest.mark.parametrize("kw, text", [
    (dict(quick=True, quick_epochs=2), "real-data runs only"),            # synthetic + quick-epochs
    (dict(quick_epochs=2), "requires --quick"),
    (dict(stage="map", quick=True, calibration="x"), "cannot be combined with --quick"),
    (dict(calibration="x"), "requires --stage map"),
    (dict(stage="map"), "needs --calibration"),
    (dict(train_val_split="bogus"), "unknown train_val_split"),
])
def test_inconsistent_options_are_errors(kw, text):
    _, errs = pp.build_plan(_cfg(), mode="1d", data_source="synthetic", **kw)
    assert any(text in e for e in errs), errs


def test_real_epochs_from_em_dir_and_quick_epochs(tmp_path):
    for i in range(7):
        (tmp_path / f"{i}_x.csv").write_text("a\n")
    (tmp_path / "notes.txt").write_text("not a survey")
    rows, errs = pp.build_plan(_cfg(), mode="2d", data_source="real", em_dir=str(tmp_path),
                               quick=True, quick_epochs=3)
    assert not errs and _rows(rows)["epochs"] == 3
    assert pp.n_real_epochs(str(tmp_path)) == 7
    _, errs = pp.build_plan(_cfg(), mode="2d", data_source="real", em_dir=str(tmp_path / "missing"))
    assert any("no EM *.csv" in e for e in errs)


def test_format_plan_marks_ok_and_errors():
    rows, errs = pp.build_plan(_cfg(), mode="1d", data_source="synthetic")
    assert "DRY RUN] OK" in pp.format_plan(rows, errs) and "nothing is written" in pp.format_plan(rows, errs)
    assert "[ERROR] boom" in pp.format_plan(rows, ["boom"])
