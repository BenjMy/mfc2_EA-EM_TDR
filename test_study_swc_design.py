"""Tests of study_swc_design.py (pure helpers, scenario generation, metrics, plots).

No pipeline run and no emagpy needed: scenarios are validated with scenario_loader,
metrics and plots use small fake arrays / a fake summary.
"""
import json
import os

import numpy as np
import pytest

import scenario_loader as sl
import study_swc_design as st
import sweeps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = {f: tuple(v) for f, v in st.LEVELS.items()}
SURVEY = st.survey_x()


@pytest.fixture(scope="module")
def parent():
    with open(os.path.join(ROOT, "ensemble_scenarios.json"), encoding="utf-8") as f:
        raw = sweeps.with_generated({k: v for k, v in json.load(f).items() if k not in sl._SKIP_KEYS})
    return sl._resolve_extends(raw)["1d_full_depth_ref"]


# ---------------------------------------------------------------- positions
@pytest.mark.parametrize("dist", st.LEVELS["dist"])
@pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
def test_positions_valid(dist, n):
    xs = st.probe_positions(dist, n, SURVEY)
    assert len(xs) == n and xs[0] == 0.0                    # re-anchored at the survey origin
    assert xs == sorted(set(xs)) and set(xs) <= set(SURVEY)


def test_positions_patterns():
    assert st.probe_positions("clustered", 3, SURVEY) == [0.0, 2.0, 4.0]
    spread = st.probe_positions("spread", 5, SURVEY)
    assert spread[0] == 0.0 and spread[-1] == SURVEY[-1]
    assert st.probe_positions("random", 4, SURVEY) == st.probe_positions("random", 4, SURVEY)   # seeded
    assert st.probe_positions("random", 4, SURVEY, seed=1) != st.probe_positions("random", 4, SURVEY, seed=2)
    assert st.probe_positions("spread", 10, SURVEY) == SURVEY.tolist()
    assert st.probe_positions("spread", 99, SURVEY) == SURVEY.tolist()        # capped at the survey size


def test_unknown_distribution():
    with pytest.raises(ValueError):
        st.probe_positions("nope", 3, SURVEY)


# ------------------------------------------------------------------ designs
@pytest.mark.parametrize("preset,count", [("smoke", 8), ("oat", 15), ("main", 160), ("full", 704)])
def test_design_counts(preset, count):
    assert len(st.build_designs(preset, LEVELS)) == count


def test_degenerate_positions_are_merged():
    designs = st.build_designs("full", LEVELS)
    keys = [(d.smc, d.ws, d.reg, d.k, st.position_signature(st.probe_positions(d.dist, d.n, SURVEY)))
            for d in designs]
    assert len(keys) == len(set(keys))
    # n = 1 and n = 10 exist once per (smc, ws, reg, k), not once per distribution
    assert sum(d.n == 1 for d in designs) == 2 * 2 * 4 * 4
    assert sum(d.n == 10 for d in designs) == 2 * 2 * 4 * 4


def test_restricted_levels():
    levels = {**LEVELS, "n": (3, 5), "dist": ("clustered",), "smc": ("hetero",), "ws": ("hetero",),
              "k": (2,), "reg": ("noreg", "beta")}
    names = [d.name for d in st.build_designs("oat", levels)]
    assert len(names) == len(set(names)) == 3                # n=3 and n=5 (ref) + beta at the ref n


# ------------------------------------------------------------ K = 0 (floor)
def test_k0_is_opt_in():
    assert 0 not in st.LEVELS["k"] and 0 in st.K_CHOICES          # presets keep their designs
    for preset in ("smoke", "oat", "main", "full"):
        assert all(d.k != 0 for d in st.build_designs(preset, LEVELS))


def test_k0_one_design_per_truth():
    levels = {**LEVELS, "k": (0, 1)}
    k0 = [d for d in st.build_designs("full", levels) if d.k == 0]
    assert len(k0) == 2 * 2 * 4                                    # smc x ws x reg, whatever n / dist
    assert {d.n for d in k0} == {1} and {d.dist for d in k0} == {"spread"}   # smallest n, first distribution
    assert len({(d.smc, d.ws, d.reg) for d in k0}) == len(k0)
    assert sum(d.k == 1 for d in st.build_designs("full", levels)) == sum(d.k == 1 for d in st.build_designs("full", LEVELS))   # K >= 1 untouched


def test_k0_floor_runs_of_the_plan():
    levels = {**LEVELS, "k": (0,), "n": (5,), "dist": ("spread",), "reg": ("noreg",)}
    names = {d.name for d in st.build_designs("full", levels)}
    assert len(names) == 4 and all(n.endswith("_spread_n5_free0") for n in names)


def test_k0_scenario_pins_everything(parent, tmp_path):
    d = st.Design("hetero", "homo", "noreg", "spread", 5, 0)
    cfg = st.build_study_config(parent, [d], p0_bias=0.2)
    assert "lab_p0" not in cfg[d.name]                             # nothing is free: no biased start
    path = tmp_path / "k0.json"
    path.write_text(json.dumps(cfg))
    sc = sl.load_scenario(d.name, config_path=str(path))
    assert all(sc.ws_fixed_params.values()) and len(sc.ws_fixed_params) == 4
    assert st.parse_name(d.name) == d


def test_name_roundtrip():
    for d in st.build_designs("main", LEVELS):
        assert st.parse_name(d.name) == d
    assert st.parse_name("1d_surface_free1") is None


# ---------------------------------------------------------------- scenarios
def test_scenarios_load_and_fields(parent, tmp_path):
    designs = st.build_designs("main", LEVELS)
    cfg = st.build_study_config(parent, designs)
    path = tmp_path / "study.json"
    path.write_text(json.dumps(cfg))
    for d in designs:
        sc = sl.load_scenario(d.name, config_path=str(path))
        assert sc.homogeneous is (d.smc == "homo")
        assert sc.ws_xy_variation is (d.ws == "hetero")
        assert bool(sc.ws_layer_params) is (d.ws == "hetero")
        free = [p for p, fixed in sc.ws_fixed_params.items() if not fixed]
        assert free == sorted(sweeps.FREE_ORDER[:d.k], key=("a", "m", "sigma_w", "sigma_s").index)
        assert sc.gamma == (st.GAMMA_VALUE if d.reg in ("gamma", "both") else 0.0)
        assert (sc.param_grid.get("beta") == [st.BETA_VALUE]) is (d.reg in ("beta", "both"))
        assert "alpha" in sc.param_grid and "forwardModel" in sc.param_grid      # merged, not replaced
        assert sc.num_swc_positions == d.n
    assert len({cfg[d.name]["output_dir"] for d in designs}) == len(designs)


def test_start_bias(parent):
    d = st.Design("hetero", "homo", "noreg", "spread", 5, 3)       # K = 3 frees sigma_s, m, sigma_w ('a' stays fixed)
    p0 = st.build_study_config(parent, [d], p0_bias=0.2)[d.name]["lab_p0"]
    assert p0 == pytest.approx([0.5, 3.6, 600.0, 120.0])
    assert "lab_p0" not in st.build_study_config(parent, [d], p0_bias=0.0)[d.name]    # inherited from the parent
    d4 = st.Design("hetero", "homo", "noreg", "spread", 5, 4)      # K = 4 also frees 'a': starts at 1.0 (sweeps convention)
    assert st.build_study_config(parent, [d4], p0_bias=0.2)[d4.name]["lab_p0"][0] == 1.0


def test_probe_csv(parent, tmp_path):
    import pandas as pd
    depths = st.probe_depths(parent)
    assert len(depths) == 4
    path = st.write_probe_csv(str(tmp_path / "p.csv"), [0.0, 6.0], depths)
    df = pd.read_csv(path)
    assert list(df.columns) == ["x", "y", "z", "label"] and len(df) == 8
    assert set(df["x"]) == {0.0, 6.0} and (df["y"] == 0).all()


def test_command_and_reg():
    d = st.Design("homo", "homo", "both", "random", 3, 2)
    cmd = st.pipeline_command(d, "c.json", "p.csv", "1d", True, ["--em-footprint-diagnostic"])
    assert cmd[:4] == ["./run_one_scenario.sh", "-s", d.name, "-c"] and "--quick" in cmd
    assert cmd[-1] == "--em-footprint-diagnostic" and "--probe-xyz" in cmd
    with pytest.raises(ValueError):
        st.reg_fields("sigma")


def test_study_has_its_own_run_log():
    """The study must not write into the shared real-data log mfc2_outputs/run_log.csv."""
    assert os.path.basename(st.RUN_LOG) == "run_log_swc_study.csv"
    d = st.Design("homo", "homo", "both", "random", 3, 2)
    cmd = st.pipeline_command(d, "c.json", "p.csv", "1d", False, [])
    assert cmd[cmd.index("--log-file") + 1] == st.RUN_LOG
    cmd = st.pipeline_command(d, "c.json", "p.csv", "1d", False, [], "other.csv")
    assert cmd[cmd.index("--log-file") + 1] == "other.csv"


# ------------------------------------------------------------------ metrics
def _fake_run(tmp_path, offset=0.02, calibrated=(True, False, False, False)):
    om = tmp_path / "run" / "output_model"
    om.mkdir(parents=True)
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.1, 0.4, size=(4, 3, 10, 4))
    calc = truth[0] + offset
    np.save(om / "synthetic_time_lapse_theta.npy", truth)
    np.save(om / "theta_calc.npy", calc)
    np.save(om / "is_calibrated.npy", np.array(calibrated))
    np.save(om / "synthetic_x_coords.npy", SURVEY)
    return str(tmp_path / "run")


def test_compute_metrics(tmp_path):
    run = _fake_run(tmp_path)
    m = st.compute_metrics(run, 0, probe_x=[0.0, 4.0], residual_path=str(tmp_path / "res" / "a.npz"))
    assert m["rmse_truth"] == pytest.approx(0.02) and m["bias_truth"] == pytest.approx(0.02)
    assert m["rmse_calibrated_h"] == pytest.approx(0.02) and m["rmse_near_probe"] == pytest.approx(0.02)
    assert len(json.loads(m["rmse_by_layer"])) == 4
    z = np.load(tmp_path / "res" / "a.npz")
    assert z["resid"].shape == (3, 10, 4) and z["dist"].tolist()[:3] == [0.0, 2.0, 0.0]
    assert st.compute_metrics(str(tmp_path / "nothing")) == {}


# -------------------------------------------------------------------- plots
def _fake_summary(tmp_path):
    designs = st.build_designs("main", LEVELS)
    rng = np.random.default_rng(1)
    rows = []
    for d in designs:
        r = st.summary_row(d, None, str(tmp_path), SURVEY)
        r.update(status="OK", rmse_truth=float(0.02 + 0.03 / d.n + 0.005 * rng.random()), rmse_near_probe=0.01,
                 rmse_far_probe=0.03, scenario=d.name)
        np.savez_compressed(tmp_path / "residuals" / f"{d.name}.npz", resid=rng.normal(0, 0.02, (3, 10, 4)),
                            dist=np.zeros(10), is_calibrated=np.array([]))
        rows.append(r)
    st.write_summary(rows, str(tmp_path / "study_summary.csv"))
    return st.load_summary(str(tmp_path / "study_summary.csv"))


def test_expand_degenerate(tmp_path):
    (tmp_path / "residuals").mkdir()
    df = _fake_summary(tmp_path)
    ex = st.expand_degenerate(df)
    for dist in st.LEVELS["dist"]:
        assert (ex[(ex["dist"] == dist) & (ex["n"] == 1)].shape[0]
                == df[(df["n"] == 1)].shape[0])


def test_plots_and_report(tmp_path):
    (tmp_path / "residuals").mkdir()
    df = _fake_summary(tmp_path)
    paths = st.make_plots(df, str(tmp_path))
    names = {os.path.basename(p) for p in paths}
    assert {"fig_rmse_vs_nswc.png", "fig_rmse_vs_k.png", "fig_heatmap_n_vs_k.png",
            "fig_distribution_effect.png", "fig_regularisation_effect.png",
            "fig_truth_heterogeneity_effect.png", "fig_error_at_vs_away_from_probes.png",
            "fig_residual_distribution.png"} <= names
    assert all(os.path.getsize(p) > 2000 for p in paths)
    st.write_report(df, str(tmp_path / "r.md"))
    assert "Best design per truth" in (tmp_path / "r.md").read_text()


def test_plots_without_results(tmp_path):
    (tmp_path / "residuals").mkdir()
    df = _fake_summary(tmp_path)
    df["rmse_truth"] = np.nan
    assert st.make_plots(df, str(tmp_path)) == []


# ---------------------------------------------------------------- design + truth overview
def _fake_truth_run(path, hetero_ws: bool, n_pts=10, n_lay=3):
    path.mkdir(parents=True)
    x = np.arange(n_pts) * 2.0
    np.save(path / "synthetic_x_coords.npy", x)
    np.save(path / "synthetic_layer_centers.npy", np.array([0.1, 0.4, 0.9])[:n_lay])
    np.save(path / "synthetic_time_lapse_theta.npy", np.random.default_rng(0).uniform(0.1, 0.4, (1, 2, n_pts, n_lay)))
    params = {"a": 1.0, "m": 1.5, "sigma_w": 500.0, "sigma_s": 100.0}
    (path / "synthetic_ws_truth_params.json").write_text(json.dumps({str(l): params for l in range(n_lay)}))
    if hetero_ws:                                   # per-point truth, as written when ws_xy_variation is on
        for i, p in enumerate(("a", "m", "sigma_w", "sigma_s")):
            np.save(path / f"synthetic_ws_truth_{p}.npy",
                    (params[p] * (1 + 0.2 * np.linspace(0, 1, n_pts)))[None, :, None].repeat(n_lay, 2))
    return str(path)


def test_design_and_truth_figure(tmp_path):
    from plots import swc_design_overview as ov
    rows = [{"smc": s, "ws": w, "run_dir": _fake_truth_run(tmp_path / f"{s}_{w}", w == "hetero")}
            for s in ("homo", "hetero") for w in ("homo", "hetero")]
    path = ov.plot_design_and_truth(rows, str(tmp_path / "out"), SURVEY, st.probe_positions)
    assert os.path.basename(path) == "fig_design_and_truth.png" and os.path.getsize(path) > 2000


def test_design_and_truth_figure_finds_output_model_layout(tmp_path):
    """Real runs keep synthetic_*.npy in <run>/output_model/, not in <run> itself."""
    import shutil
    from plots import swc_design_overview as ov
    rows = []
    for s in ("homo", "hetero"):
        for w in ("homo", "hetero"):
            flat = _fake_truth_run(tmp_path / "flat" / f"{s}_{w}", w == "hetero")
            run = tmp_path / "runs" / f"{s}_{w}"
            shutil.copytree(flat, run / "output_model")
            rows.append({"smc": s, "ws": w, "run_dir": str(run)})
    assert all(ov.pick_run_dir(rows, "smc", v) is not None for v in ("homo", "hetero"))
    assert ov.pick_run_dir(rows, "ws", "hetero").endswith("output_model")
    path = ov.plot_design_and_truth(rows, str(tmp_path / "out"), SURVEY, st.probe_positions)
    assert os.path.getsize(path) > 2000


def test_design_and_truth_homo_ws_json_stays_in_run_folder(tmp_path):
    """As in the pipeline: *.npy moved to output_model/, synthetic_ws_truth_params.json left in the run folder."""
    import shutil
    from plots import swc_design_overview as ov
    rows = []
    for s in ("homo", "hetero"):
        for w in ("homo", "hetero"):
            flat = _fake_truth_run(tmp_path / "flat" / f"{s}_{w}", w == "hetero")
            run = tmp_path / "runs" / f"{s}_{w}"
            (run / "output_model").mkdir(parents=True)
            for f in os.listdir(flat):
                dst = run / "output_model" if f.endswith(".npy") else run
                shutil.copy(os.path.join(flat, f), dst / f)
            rows.append({"smc": s, "ws": w, "run_dir": str(run)})
    d = ov.pick_run_dir(rows, "ws", "homo")
    t = ov._load_ws(d, 0, 10)
    assert t is not None and t.shape == (10, 3, 4) and float(np.ptp(t, axis=0).max()) == 0.0
    assert ov._load_ws(ov.pick_run_dir(rows, "ws", "hetero"), 0, 10).shape == (10, 3, 4)
    path = ov.plot_design_and_truth(rows, str(tmp_path / "out"), SURVEY, st.probe_positions)
    assert os.path.getsize(path) > 2000


def test_design_and_truth_homo_ws_from_pipeline_report(tmp_path):
    """Runs from before synthetic_ws_truth_params.json: truth comes from the report's Step 0 table."""
    import shutil
    from plots import swc_design_overview as ov
    flat = _fake_truth_run(tmp_path / "flat", hetero_ws=False)
    run = tmp_path / "run"
    (run / "output_model").mkdir(parents=True)
    for f in os.listdir(flat):
        if f.endswith(".npy"):
            shutil.copy(os.path.join(flat, f), run / "output_model" / f)
    lines = ["**Step 0 - Synthetic ground-truth Waxman-Smits parameters**", "", "| Key | Value |", "|-----|-------|"]
    lines += [f"| `Layer {l} \u2014 a / m / \u03c3_w / \u03c3_s` | `0.5000 / 3.0000 / 500.00 mS/m / 100.00 mS/m` |"
              for l in range(3)]
    (run / "pipeline_report_20261125_111933.md").write_text("\n".join(lines), encoding="utf-8")
    (run / "pipeline_report_shell_20261125_112028.md").write_text("no table here", encoding="utf-8")
    t = ov._load_ws(str(run / "output_model"), 0, 10)
    assert t.shape == (10, 3, 4) and t[0, 0, 2] == 500.0 and t[0, 0, 1] == 3.0 and float(np.ptp(t, axis=0).max()) == 0.0
    assert ov._load_ws(str(tmp_path / "nothing"), 0, 10) is None


def test_design_and_truth_figure_without_runs(tmp_path):
    from plots import swc_design_overview as ov          # the layout row needs no run
    path = ov.plot_design_and_truth([], str(tmp_path), SURVEY, st.probe_positions)
    assert os.path.getsize(path) > 2000


def test_homogeneous_ws_truth_rebuilt_from_json(tmp_path):
    from plots.ws_distribution import load_ws_truth
    d = _fake_truth_run(tmp_path / "r", hetero_ws=False)
    t = load_ws_truth(d, 0, None, 10)
    assert t.shape == (10, 3, 4) and np.ptp(t, axis=0).max() == 0 and t[0, 0, 2] == 500.0


def test_pick_run_dir_prefers_reference(tmp_path):
    from plots import swc_design_overview as ov
    a, b = _fake_truth_run(tmp_path / "a", False), _fake_truth_run(tmp_path / "b", False)
    rows = [{"smc": "homo", "ws": "homo", "n": 1, "run_dir": a}, {"smc": "homo", "ws": "homo", "n": 5, "run_dir": b},
            {"smc": "homo", "ws": "homo", "n": 5, "run_dir": str(tmp_path / "missing")}]
    assert ov.pick_run_dir(rows, "smc", "homo", {"n": 5}) == b
    assert ov.pick_run_dir(rows, "smc", "hetero") is None


# ---------------------------------------------------------------- heat map variants
def test_heatmap_variants_main_preset(tmp_path):
    (tmp_path / "residuals").mkdir()
    df = _fake_summary(tmp_path)
    names = {os.path.basename(p) for p in st.plot_heatmap_variants(df, str(tmp_path))}
    # main: other distributions at the reference regularisation; reg is only run at n = 5 (no heat map)
    assert names == {"fig_heatmap_n_vs_k_clustered_noreg.png", "fig_heatmap_n_vs_k_random_noreg.png"}
    assert "fig_heatmap_n_vs_k.png" not in names                       # the reference one is written separately
    assert "fig_heatmap_n_vs_k_clustered_noreg.png" in {os.path.basename(p) for p in st.make_plots(df, str(tmp_path))}


def test_heatmap_variants_other_regularisation(tmp_path):
    import pandas as pd
    (tmp_path / "residuals").mkdir()
    df = _fake_summary(tmp_path)
    beta = df[(df["reg"] == "noreg") & (df["dist"] == "spread")].copy()
    beta["reg"] = "beta"                                                # as if reg had been run at every n
    names = {os.path.basename(p) for p in st.plot_heatmap_variants(pd.concat([df, beta], ignore_index=True), str(tmp_path))}
    assert "fig_heatmap_n_vs_k_spread_beta.png" in names


def test_heatmap_variants_no_results(tmp_path):
    (tmp_path / "residuals").mkdir()
    df = _fake_summary(tmp_path)
    df["rmse_truth"] = np.nan
    assert st.plot_heatmap_variants(df, str(tmp_path)) == []


def test_k0_only_defaults_to_full_preset(tmp_path):
    out = tmp_path / "o"
    argv = ["--smc", "homo", "hetero", "--ws", "homo", "hetero", "--reg", "noreg", "--free-k", "0",
            "--n-swc", "5", "--dist", "spread", "--dry-run", "--out-dir", str(out), "--runs-dir", str(tmp_path / "r")]
    assert st.main(argv) == 0
    cfg = json.loads((out / "study_scenarios.json").read_text())
    assert sum(st.parse_name(k) is not None for k in cfg) == 4
