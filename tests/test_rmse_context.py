import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rmse_context as rc
import run_log
from run_log import RunLog, RUN_LOG_COLUMNS, infer_study_type


def test_parse_rmse():
    assert rc.parse_rmse("0.031212 m³/m³") == 0.031212
    assert np.isnan(rc.parse_rmse("n/a")) and np.isnan(rc.parse_rmse(None))


def test_rmse_context_values():
    obs = np.array([0.1, 0.2, 0.3, 0.4])          # std (ddof=0) = 0.1118
    c = rc.rmse_context(0.05, obs)
    assert abs(c["rmse_pct"] - 5.0) < 1e-9
    assert abs(c["rsr"] - 0.05 / obs.std()) < 1e-12
    assert abs(c["skill"] - (1 - c["rsr"] ** 2)) < 1e-12
    assert c["rating"] == "very good" and abs(c["nrmse_mean_pct"] - 20.0) < 1e-9


def test_ratings_and_degenerate():
    assert rc.rating_from_rsr(0.55) == "good" and rc.rating_from_rsr(0.65) == "satisfactory"
    assert rc.rating_from_rsr(0.8) == "unsatisfactory" and rc.rating_from_rsr(1.2) == "no skill"
    assert rc.rmse_context(0.05, [0.2, 0.2, 0.2])["rating"] == "std too small"
    assert rc.rmse_context(0.05, [0.2])["rating"] == "n/a"
    assert rc.rmse_context(0.05, [np.nan, 0.1, 0.4])["n_obs"] == 2


def test_metric_and_log_fields():
    c = rc.rmse_context(0.05, [0.1, 0.2, 0.3, 0.4])
    names = [n for n, _ in rc.metric_lines(c)]
    assert "Final RMSE (% vol)" in names and "Normalised RMSE (RMSE/std, RSR)" in names
    assert set(rc.log_fields(c)) == set(rc.CONTEXT_COLUMNS)


def test_infer_study_type():
    cases = {"swcstudy_smchetero_wshetero_noreg_spread_n5_free4": "swc_design_study",
             "real_study_L2_free1": "hyp1_layers_ws_freedom", "real_tv_by_probe": "hyp2_train_val_split",
             "real_bw_sweep": "hyp3_localized_bandwidth", "1d_surface_free2": "surface_vs_full",
             "1d_full_hetero_free3": "surface_vs_full_hetero", "1d_surface_free4_pin0": "pinned_horizon",
             "1d_full_free2_tv_probe": "train_val_synthetic", "real_single_base": "real_scenarios",
             "mfc2_real_synthetic_mirror": "synthetic_mirror", "ws_bench_nfree": "ws_global_vs_localized",
             "wsb_xy_random_L4_P4_C10_bw5_s22": "ws_global_vs_localized",
             "foo": "other"}
    for name, want in cases.items():
        assert infer_study_type(name) == want, name


def test_old_log_is_upgraded_and_appended(tmp_path):
    p = tmp_path / "log.csv"
    old = RUN_LOG_COLUMNS[:15]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=old); w.writeheader()
        w.writerow({"run_id": "a", "scenario": "real_study_L2_free1", "final_rmse": "0.0920 m³/m³", "status": "SUCCESS"})
    log = RunLog(str(p))
    assert log.append({"run_id": "b", "scenario": "1d_surface_free1", "final_rmse": "0.05 m³/m³", "status": "SUCCESS"})
    rows = log.rows()
    assert list(rows[0].keys()) == RUN_LOG_COLUMNS
    assert rows[0]["study_type"] == "hyp1_layers_ws_freedom" and rows[0]["rmse_pct"] == "9.200"
    assert rows[1]["study_type"] == "surface_vs_full" and rows[1]["rmse_pct"] == "5.000"
    assert os.path.isfile(str(p) + ".bak")
    assert log.ensure_schema() is False            # idempotent


def test_prefixed_metric_lines():
    c = rc.rmse_context(0.1, [0.1, 0.2, 0.3, 0.4])
    d = dict(rc.prefixed_metric_lines("Validation", c))
    assert d["Validation RMSE (% vol)"].startswith("10.00")
    assert d["Validation RMSE rating"] == "unsatisfactory"      # RSR 0.1 / 0.1118 = 0.89


def test_parse_leading_number():
    assert rc.parse_leading_number("0.51  (obs std 0.1, n=3)") == 0.51
    assert np.isnan(rc.parse_leading_number("n/a  (obs std n/a, n=1)"))
    assert np.isnan(rc.parse_leading_number(None))


def test_per_horizon_context():
    from types import SimpleNamespace as NS
    rng = np.random.default_rng(0)
    obs0 = np.array([0.1, 0.2, 0.3, 0.4])
    hf = [NS(observed_theta=obs0, predicted_theta=obs0 + 0.05, is_calibrated=True),
          NS(observed_theta=np.array([]), predicted_theta=np.array([]), is_calibrated=False),
          NS(observed_theta=np.array([0.2, np.nan]), predicted_theta=np.array([0.2, 0.3]), is_calibrated=True)]
    c = rc.per_horizon_context(hf)
    assert abs(c[0]["rmse_pct"] - 5.0) < 1e-9 and c[0]["rating"] == "very good"
    assert c[1]["rating"] == "no TDR" and c[2]["n_obs"] == 1
    lines = dict(rc.horizon_metric_lines(c))
    assert lines["RMSE horizon H0"].startswith("5.00 % vol, RSR 0.447") and "no TDR" in lines["RMSE horizon H1"]
