import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_run_log as prl
import plot_swc_study_log as psl
from run_log import RunLog


def _fill(path):
    log = RunLog(str(path))
    for sc, rm, std in [("1d_surface_free1", .12, .08), ("1d_surface_free2", .10, .08), ("1d_surface_free2", .09, .08),
                        ("1d_full_free1", .08, .08), ("1d_full_free2", .04, .08), ("real_study_L2_free1", .092, .09),
                        ("swcstudy_smchetero_wshetero_noreg_spread_n5_free1", .06, .08),
                        ("swcstudy_smchetero_wshetero_noreg_spread_n5_free2", .05, .08),
                        ("swcstudy_smchetero_wshetero_noreg_spread_n3_free1", .07, .08),
                        ("swcstudy_smchetero_wshetero_noreg_spread_n3_free2", .065, .08)]:
        log.append({"run_id": f"{sc}-{rm}", "scenario": sc, "status": "SUCCESS", "final_rmse": f"{rm} m³/m³",
                    "runtime_s": "60", "timestamp": "2026-11-20", "rsr": str(rm / std), "swc_obs_std": str(std)})
    log.append({"run_id": "bad", "scenario": "1d_full_free3", "status": "FAILED"})
    return log


def test_split_and_groups(tmp_path):
    assert prl.split_free_k("1d_surface_hetero_free3") == ("1d_surface_hetero", 3)
    assert prl.split_free_k("foo") == ("foo", None)
    _fill(tmp_path / "l.csv")
    g = prl.group_by_study(prl.load_rows(str(tmp_path / "l.csv")))
    assert {"surface_vs_full", "hyp1_layers_ws_freedom", "swc_design_study"} <= set(g)
    lat = prl.latest_per_scenario(g["surface_vs_full"])
    assert lat["1d_surface_free2"]["rmse_pct"] == 9.0 and "1d_full_free3" not in lat


def test_plots_written(tmp_path):
    _fill(tmp_path / "l.csv")
    out = tmp_path / "o"
    assert prl.main(["--log-file", str(tmp_path / "l.csv"), "--out-dir", str(out)]) == 0
    for f in ("rmse_by_scenario.png", "rmse_vs_k.png", "rmse_history.png", "runs.csv"):
        assert (out / "surface_vs_full" / f).is_file(), f
    assert prl.main(["--log-file", str(tmp_path / "l.csv"), "--study-type", "nope"]) == 1
    out2 = tmp_path / "o2"
    assert psl.main(["--log-file", str(tmp_path / "l.csv"), "--out-dir", str(out2)]) == 0
    assert (out2 / "swc_runs.csv").is_file() and (out2 / "swc_rmse_vs_k.png").is_file()


def _png(path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(1, 1)); fig.savefig(path); plt.close(fig)


def test_gather_and_tile_figures(tmp_path):
    log = RunLog(str(tmp_path / "l.csv"))
    for sc in ("1d_surface_free1", "1d_surface_free2"):
        d = tmp_path / "runs" / sc
        _png(d / "water_content_xy_x_imshow.png"); _png(d / "water_content_xy_x_pctdiff.png")
        _png(d / "solution" / "solution_swc_uncertainty.png")
        log.append({"run_id": sc, "scenario": sc, "status": "SUCCESS", "final_rmse": "0.05", "output_dir": str(d)})
    # a run without figures must be skipped silently
    log.append({"run_id": "n", "scenario": "1d_surface_free3", "status": "SUCCESS", "final_rmse": "0.05",
                "output_dir": str(tmp_path / "nothing")})
    rows = prl.group_by_study(prl.load_rows(str(tmp_path / "l.csv")))["surface_vs_full"]
    lat = prl.latest_per_scenario(rows)
    assert prl.find_figure(lat["1d_surface_free1"]["output_dir"], prl.KEY_FIGURES["swc_map_xy"]).endswith("_imshow.png")
    out = tmp_path / "o"
    files = prl.plot_study("surface_vs_full", rows, str(out), figures=True, tile=True)
    assert (out / "surface_vs_full/figures/swc_map_xy/1d_surface_free2.png").is_file()
    assert not (out / "surface_vs_full/figures/swc_map_xy/1d_surface_free3.png").exists()
    assert (out / "surface_vs_full/figures/swc_uncertainty_tiled.png").is_file()
    assert any(f.endswith("swc_map_xy_tiled.png") for f in files)


def test_stale_other_study_type_is_reinferred(tmp_path):
    import csv
    from run_log import RUN_LOG_COLUMNS
    p = tmp_path / "l.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_LOG_COLUMNS); w.writeheader()
        w.writerow({"run_id": "a", "scenario": "wsb_xy_random_L4_P4_C10_bw5", "status": "SUCCESS",
                    "final_rmse": "0.05", "study_type": "other"})
    assert prl.load_rows(str(p))[0]["study_type"] == "ws_global_vs_localized"


def _manifest(d, run_id, config, commit="abc12345def+dirty", mode="1d", stage="all"):
    import json
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_manifest.json").write_text(json.dumps(
        {"run_id": run_id, "mode": mode, "stage": stage, "git_commit": commit, "config": config}))


def test_truth_hint_and_config_differences():
    assert "heterogeneous" in prl.truth_hint("1d_full_hetero_free2")
    assert "homogeneous-in-depth" in prl.truth_hint("1d_full_free2")
    assert prl.truth_hint("swcstudy_smchomo_wshetero_noreg_spread_n5_free1") == "SMC homogeneous; WS heterogeneous"
    differ, common = prl.config_differences({"a": {"x": 1, "y": {"z": [1, 2]}, "k": 3},
                                             "b": {"x": 2, "y": {"z": [1, 2]}, "k": 3}})
    assert differ == {"x": {"1": ["a"], "2": ["b"]}} and common == {"k": "3", "y.z": "[1, 2]"}


def test_provenance_written(tmp_path):
    log = RunLog(str(tmp_path / "l.csv"))
    for sc, cfg in (("1d_surface_free1", {"n_zones": 1, "k": 1}), ("1d_surface_free2", {"n_zones": 1, "k": 2})):
        d = tmp_path / "runs" / sc
        _manifest(d, sc, cfg)
        log.append({"run_id": sc, "scenario": sc, "status": "SUCCESS", "final_rmse": "0.05", "output_dir": str(d)})
    # third run: manifest of a LATER run, fourth: no manifest
    d3 = tmp_path / "runs" / "1d_full_free1"; _manifest(d3, "LATER", {"n_zones": 1})
    log.append({"run_id": "r3", "scenario": "1d_full_free1", "status": "SUCCESS", "final_rmse": "0.05", "output_dir": str(d3)})
    log.append({"run_id": "r4", "scenario": "1d_full_free2", "status": "SUCCESS", "final_rmse": "0.05",
                "output_dir": str(tmp_path / "runs" / "none")})
    out = tmp_path / "o"
    assert prl.main(["--log-file", str(tmp_path / "l.csv"), "--out-dir", str(out)]) == 0
    text = (out / "surface_vs_full" / "provenance.md").read_text()
    assert "homogeneous-in-depth" in text                                            # truth hint from the name
    assert "`k`:" in text and "**1** -> 1d_surface_free1" in text and "**2** -> 1d_surface_free2" in text
    assert "abc12345+dirty" in text and "LATER run" in text and "no run_manifest.json" in text
    assert (out / "surface_vs_full" / "manifests" / "1d_surface_free1.json").is_file()
