"""ws_recovery.py: report parsing (text in the exact format the pipeline writes; no run, no emagpy)."""
import os

import pytest

import ws_recovery as wr


def _table(title, rows):                       # same layout as run_pipeline.Reporter.args_table
    out = ["", f"**{title}**", "", "| Key | Value |", "|-----|-------|"]
    out += [f"| `{k}` | `{v}` |" for k, v in rows.items()]
    return out + [""]


def _p(a, m, sw, ss):
    return f"{a:.4f} / {m:.4f} / {sw:.2f} mS/m / {ss:.2f} mS/m"


def make_report(*, fixed, bounds=None, truth=None, start=(0.5, 3.0, 500.0, 100.0), fit=None, step4=True,
                baseline=False, uncalibrated=(), prior_line=True):
    truth = truth or {0: (0.5, 3.0, 500.0, 100.0), 1: (0.5, 3.0, 500.0, 100.0)}
    fit = fit or {0: (0.52, 1.6, 480.0, 122.0, 0.037), 1: (0.49, 1.35, 510.0, 123.0, 0.05)}
    L = ["# Pipeline report", ""]
    L += _table("Run arguments", {"ws_fixed_params": str(fixed),
                                  "ws_bounds": str(bounds) if bounds else "(defaults — EMmethod.WS_BOUNDS)"})
    L += _table("Step 0 — Synthetic ground-truth Waxman-Smits parameters (EC_bulk = theta^m / a * (sigma_w + sigma_s))"
                + ("  [layer baseline -- see note + XY map below]" if baseline else ""),
                {f"Layer {l} — a / m / σ_w / σ_s": _p(*p) for l, p in truth.items()})
    if prior_line:
        L.append("[STEP 0]   Calibration prior/start (a, m, sigma_w, sigma_s) = "
                 f"({start[0]:.2f}, {start[1]:.2f}, {start[2]:.1f} mS/m, {start[3]:.1f} mS/m) vs. TRUTH (0.50, 3.00, 500.0 mS/m, 100.0 mS/m) -- prior matches truth")
    rows = {}
    for hz, (a, m, sw, ss, rm) in fit.items():
        if hz in uncalibrated:
            rows[f"Hz {hz} — a / m / σ_w / σ_s / RMSE"] = ("UNCALIBRATED — no TDR probe in this horizon; showing prior "
                                                          f"p0 only: {_p(a, m, sw, ss)} / RMSE=N/A")
        else:
            rows[f"Hz {hz} — a / m / σ_w / σ_s / RMSE"] = f"{_p(a, m, sw, ss)} / {rm:.5f} m³/m³"
    L += _table("Step 2 — Best-ranked Waxman-Smits fit (rank-1 ensemble member)",
                {k: v.replace("0.5200", "0.9999") for k, v in rows.items()} | {"n_obs supporting best-fit": 20})
    if step4:
        L += _table("Step 4 — Best recalibrated fit (K=3, RMSE=0.03700 m³/m³)", rows)
    return "\n".join(L)


ALL_FIXED = {"a": True, "m": True, "sigma_w": True, "sigma_s": True}
K2 = {"a": True, "m": False, "sigma_w": True, "sigma_s": False}


def test_basic_row():
    rows = wr.parse_report(make_report(fixed=K2), "scn", "r.md")
    assert [r["horizon"] for r in rows] == [0, 1] and rows[0]["fit_source"] == "step4"
    r = rows[0]
    assert (r["a_fit"], r["m_fit"], r["sigma_w_fit"], r["sigma_s_fit"]) == (0.52, 1.6, 480.0, 122.0)
    assert (r["a_truth"], r["m_truth"], r["sigma_w_truth"], r["sigma_s_truth"]) == (0.5, 3.0, 500.0, 100.0)
    assert (r["a_start"], r["m_start"], r["sigma_w_start"], r["sigma_s_start"]) == (0.5, 3.0, 500.0, 100.0)
    assert [r[f"{p}_fixed"] for p in wr.PARAMS] == [True, False, True, False]
    assert r["calibrated"] is True and r["truth_baseline_only"] is False and r["scenario"] == "scn"


def test_derived_C():
    r = wr.parse_report(make_report(fixed=K2), "s")[0]
    assert r["C_truth"] == pytest.approx(0.5 / 600.0)
    assert r["C_start"] == pytest.approx(0.5 / 600.0)
    assert r["C_fit"] == pytest.approx(0.52 / (480.0 + 122.0))


def test_step4_preferred_and_step2_fallback():
    assert wr.parse_report(make_report(fixed=K2, step4=True), "s")[0]["a_fit"] == 0.52
    r = wr.parse_report(make_report(fixed=K2, step4=False), "s")[0]     # Step 2 table of the helper differs on purpose
    assert r["fit_source"] == "step2" and r["a_fit"] == 0.9999


def test_bounds_default_and_override():
    r = wr.parse_report(make_report(fixed=K2), "s")[0]
    assert (r["a_lo"], r["a_hi"], r["m_lo"], r["m_hi"]) == (0.5, 5.0, 1.0, 4.0)
    r = wr.parse_report(make_report(fixed=K2, bounds={"a": [0.2, 5.0]}), "s")[0]
    assert (r["a_lo"], r["a_hi"]) == (0.2, 5.0) and (r["sigma_s_lo"], r["sigma_s_hi"]) == (0.0, 200.0)


def test_uncalibrated_horizon_uses_prior():
    r = wr.parse_report(make_report(fixed=ALL_FIXED, uncalibrated=(0,)), "s")[0]
    assert r["calibrated"] is False and r["sigma_w_fit"] == 480.0 and r["m_fit"] == 1.6


def test_layer_truth_and_baseline_flag():
    truth = {0: (0.5, 3.0, 500.0, 100.0), 1: (0.35, 3.0, 700.0, 100.0)}
    rows = wr.parse_report(make_report(fixed=K2, truth=truth, baseline=True), "s")
    assert rows[1]["a_truth"] == 0.35 and rows[1]["sigma_w_truth"] == 700.0 and rows[0]["a_truth"] == 0.5
    assert all(r["truth_baseline_only"] for r in rows)


def test_missing_pieces_give_none_not_errors():
    rep = make_report(fixed=K2, prior_line=False).replace("Step 0 — Synthetic ground-truth", "Step 0 — Other")
    r = wr.parse_report(rep, "s")[0]
    assert r["a_start"] is None and r["a_truth"] is None and r["C_truth"] is None and r["C_fit"] is not None


def test_no_fit_table_gives_no_rows():
    assert wr.parse_report("# nothing here\n", "s") == []


def test_collect_and_csv(tmp_path):
    run = tmp_path / "0001_scn"
    run.mkdir()
    (run / "pipeline_report_20260101_000000.md").write_text(make_report(fixed=ALL_FIXED, step4=False), encoding="utf-8")
    (run / "pipeline_report_20260102_000000.md").write_text(make_report(fixed=K2), encoding="utf-8")   # newest wins
    (run / "pipeline_report_shell_20260103.md").write_text("shell companion", encoding="utf-8")
    rows = wr.collect({"scn": str(run), "gone": str(tmp_path / "nope")})
    assert len(rows) == 2 and rows[0]["m_fixed"] is False and rows[0]["fit_source"] == "step4"
    out = wr.write_csv(rows, str(tmp_path / "o" / "rec.csv"))
    lines = open(out, encoding="utf-8").read().splitlines()
    assert lines[0].split(",") == wr.COLUMNS and len(lines) == 3


def test_cli(tmp_path):
    run = tmp_path / "0001_scn"
    run.mkdir()
    (run / "pipeline_report_2026.md").write_text(make_report(fixed=K2), encoding="utf-8")
    assert wr.main([str(run), "-o", str(tmp_path / "x.csv")]) == 0 and os.path.isfile(tmp_path / "x.csv")
    assert wr.main([str(tmp_path / "empty_dir_that_does_not_exist"), "-o", str(tmp_path / "y.csv")]) == 1


# ------------------------------------------------------------------ figure
def _rows():
    a = wr.parse_report(make_report(fixed=K2), "swcstudy_smchomo_wshomo_noreg_spread_n5_free2")
    b = wr.parse_report(make_report(fixed=ALL_FIXED, uncalibrated=(1,), baseline=True),
                        "swcstudy_smchomo_wshomo_noreg_spread_n5_free0")
    return a + b


def test_figure_panels_and_content(tmp_path):
    from plots import ws_recovery_fig as fr
    fig = fr.build_figure(_rows())
    assert fig is not None and len(fig.axes) == 5
    m_ax = fig.axes[1]
    # 2 scenarios x 2 horizons: one truth segment each (solid lines); bounds only on the 4 parameter panels
    assert sum(len(l.get_xdata()) == 2 for l in m_ax.lines) == 4
    assert len(m_ax.patches) == 2 and len(fig.axes[4].patches) == 0
    ticks = [t.get_text() for t in fig.axes[-1].get_xticklabels()]
    assert ticks[0].startswith("smchomo_wshomo") and not any(t.startswith("swcstudy_") for t in ticks)
    # truth line sits at the truth value, free fit is filled, pinned fit hollow
    assert any(l.get_ydata()[0] == 3.0 for l in m_ax.lines if len(l.get_xdata()) == 2)
    fits = [l for l in m_ax.lines if l.get_marker() == "o"]
    assert {l.get_markerfacecolor() == "none" for l in fits} == {True, False}
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_figure_file_and_empty(tmp_path):
    from plots import ws_recovery_fig as fr
    assert fr.plot_ws_recovery([], str(tmp_path)) is None
    path = fr.plot_ws_recovery(_rows(), str(tmp_path / "o"))
    assert path and os.path.getsize(path) > 5000
