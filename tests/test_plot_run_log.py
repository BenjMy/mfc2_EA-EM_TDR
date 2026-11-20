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
