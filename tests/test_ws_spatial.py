"""ws_spatial.py: bandwidth sweep (A), per-station fit (B), spreading + LOSO (C),
station mapping (D), probe errors (F), station count (G).

Synthetic truth: exact Waxman-Smits data with sigma_s = SS_WEST for x < 100 and SS_EAST
beyond, so the expected per-station values and maps are known.
"""
import csv
from types import SimpleNamespace

import os
import numpy as np
import pytest

import localized_mapping as lmap
import ws_spatial as wss

A, M, SW = 1.0, 2.0, 100.0
SS_WEST, SS_EAST = 20.0, 80.0
NX, NT = 10, 4
X = np.arange(NX) * 20.0
SS = np.where(X < 100.0, SS_WEST, SS_EAST)


def _ec():
    return np.random.default_rng(0).uniform(20.0, 120.0, (NT, NX, 1))


def _obs(ec=None, ss=SS):
    ec = _ec() if ec is None else ec
    pos = np.tile(np.arange(NX), NT)
    tim = np.repeat(np.arange(NT), NX)
    theta = (A * ec[:, :, 0] / (SW + ss[None])) ** (1.0 / M)
    return ec, wss.build_probe_obs(
        ec, sample_positions=pos, sample_times=tim, theta_obs=theta[tim, pos],
        probe_layer_indices=[0] * len(pos), sensor_ids=None, point_x=X, point_y=None)


def _pooled(ss=50.0):
    return np.array([[A, M, SW, ss]])


# ------------------------------------------------------------------ A
def test_nearest_neighbour_spacing_and_default_sweep():
    x = [0.0, 10.0, 30.0]
    assert np.allclose(wss.nearest_neighbour_spacing(x), [10.0, 10.0, 20.0])
    sweep = wss.default_bandwidth_sweep(x)
    assert sweep == sorted(sweep) and sweep[0] == 5.0 and sweep[-1] == 50.0
    assert wss.default_bandwidth_sweep([1.0]) == []


def test_sweep_below_spacing_flags_old_sweep():
    x = np.arange(5) * 27.0
    assert wss.sweep_below_spacing([0.5, 5.0, 20.0], x)[0] is True      # the flat 0.5-20 m sweep
    assert wss.sweep_below_spacing([15.0, 50.0, 100.0], x)[0] is False  # 15-100 m


def test_duplicate_locations_do_not_give_zero_spacing():
    assert wss.nearest_neighbour_spacing([0.0, 0.0, 10.0]).min() == 10.0


# ------------------------------------------------------------------ G
def test_count_stations_prints_real_number_not_config():
    ids = ["U1", "U1", "U2", "U3"]
    assert wss.count_stations(ids, fallback=4) == 3
    assert wss.count_stations(None, fallback=4) == 4


# ------------------------------------------------------------------ B
def test_fit_stations_recovers_west_east_contrast():
    _, obs = _obs()
    sf = wss.fit_stations(obs, _pooled(), shrinkage=0.0)
    ss = sf.params[3, 0]
    assert np.allclose(ss[:5], SS_WEST, atol=0.5) and np.allclose(ss[5:], SS_EAST, atol=0.5)
    assert sf.valid.all() and not sf.at_bound.any()
    assert np.allclose(sf.params[:3, 0, :], np.array([A, M, SW])[:, None])   # others held


def test_shrinkage_pulls_towards_pooled():
    _, obs = _obs()
    free = wss.fit_stations(obs, _pooled(), shrinkage=0.0).params[3, 0]
    shr = wss.fit_stations(obs, _pooled(), shrinkage=1.0).params[3, 0]
    assert np.all(np.abs(shr - 50.0) < np.abs(free - 50.0))


def test_station_with_too_few_obs_takes_pooled_value():
    _, obs = _obs()
    keep = ~((obs.station == "0") & (obs.time >= 1))            # station 0 keeps 1 observation
    sf = wss.fit_stations(obs.subset(keep), _pooled(), shrinkage=0.0)
    i = sf.stations.index("0")
    assert not sf.valid[0, i] and sf.params[3, 0, i] == 50.0


def test_fixed_sigma_s_gives_pooled_and_a_note():
    _, obs = _obs()
    sf = wss.fit_stations(obs, _pooled(), fixed_params={"sigma_s": True})
    assert np.allclose(sf.params[3], 50.0) and sf.notes and not sf.valid.any()


def test_at_bound_flag():
    _, obs = _obs(ss=np.full(NX, 400.0))                        # truth far above the [0, 200] bound
    sf = wss.fit_stations(obs, _pooled(), shrinkage=0.0, ws_bounds={"sigma_s": [0.0, 200.0]})
    assert sf.at_bound.all()


def test_ws_by_station_csv_and_dict(tmp_path):
    _, obs = _obs()
    sf = wss.fit_stations(obs, _pooled())
    path = wss.write_ws_by_station_csv(sf, str(tmp_path / "sub" / "ws_by_station.csv"))
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == NX and set(rows[0]) >= {"station", "x", "y", "horizon", "sigma_s", "n_obs",
                                                "rmse", "fitted", "at_bound"}
    d = wss.station_fits_to_dict(sf)
    assert d["free"] == ["sigma_s"] and len(d["params"]["sigma_s"][0]) == NX


# ------------------------------------------------------------------ C
def _sf_two():
    return SimpleNamespace(
        x=np.array([0.0, 100.0]), y=np.zeros(2), valid=np.array([[True, True]]),
        params=np.array([[[1.0, 1.0]], [[2.0, 2.0]], [[100.0, 100.0]], [[20.0, 80.0]]]))


def test_voronoi_takes_nearest_station():
    p, fb = wss.spread_station_params(_sf_two(), [10.0, 90.0], None, "voronoi", _pooled())
    assert np.allclose(p[3, 0], [20.0, 80.0]) and not fb.any()


def test_idw_honours_station_and_interpolates():
    p, _ = wss.spread_station_params(_sf_two(), [0.0, 50.0, 100.0], None, "idw", _pooled())
    assert np.allclose(p[3, 0], [20.0, 50.0, 80.0])


def test_horizon_without_fitted_station_falls_back_to_pooled():
    sf = _sf_two()
    sf.valid = np.array([[False, False]])
    p, fb = wss.spread_station_params(sf, [10.0], None, "idw", _pooled())
    assert fb.all() and p[3, 0, 0] == 50.0


def test_spread_rejects_unknown_method():
    with pytest.raises(ValueError):
        wss.spread_station_params(_sf_two(), [0.0], None, "kriging", _pooled())


def test_loso_station_methods_beat_pooled_with_spatial_structure():
    _, obs = _obs()
    preds = wss.loso_predictions(obs, 1, methods=("global", "voronoi", "idw"),
                                 lab_p0=np.array([A, M, SW, 50.0]), shrinkage=0.0)
    s = wss.loso_summary(obs, preds)
    assert s["voronoi"]["rmse"] < s["global"]["rmse"]
    assert s["idw"]["rmse"] < s["global"]["rmse"]
    assert s["global"]["n"] == s["voronoi"]["n"] == len(obs)    # same observations scored


def test_choose_method_prefers_simplest_within_one_se():
    summ = {"global": {"rmse": 0.115, "se": 0.0}, "voronoi": {"rmse": 0.100, "se": 0.01},
            "localized": {"rmse": 0.099, "se": 0.01}}
    assert wss.choose_method(summ)[0] == "voronoi"      # pooled 0.115 > 0.099 + 0.01; voronoi is simpler than kernel
    summ["global"]["rmse"] = 0.105                      # now within the limit 0.109
    assert wss.choose_method(summ)[0] == "global"
    summ["global"]["rmse"] = 0.2
    summ["voronoi"]["rmse"] = 0.2
    assert wss.choose_method(summ)[0] == "localized"       # only the best qualifies
    assert wss.choose_method({})[0] is None


# ------------------------------------------------------------------ D
def test_station_mapping_recovers_truth_map():
    ec, obs = _obs()
    gfit = SimpleNamespace(horizon_fits=[SimpleNamespace(a=A, m=M, sigma_w=SW, sigma_s=50.0,
                                                         is_calibrated=True)])
    mp = wss.build_station_mapping(
        ec, gfit, method="voronoi", point_x=X, point_y=None, sample_positions=obs.pos,
        sample_times=obs.time, theta_obs=obs.theta, probe_layer_indices=[0] * len(obs),
        shrinkage=0.0)
    assert isinstance(mp, lmap.LocalizedMapping) and np.isnan(mp.bandwidth)
    truth = (A * ec[:, :, 0] / (SW + SS[None])) ** (1.0 / M)
    assert np.allclose(mp.theta[:, :, 0], truth, atol=2e-3)
    assert mp.params.shape == (4, 1, NX) and not mp.fallback.any()


def test_station_mapping_equals_scalar_conversion_when_all_stations_share_a_value():
    ec, obs = _obs(ss=np.full(NX, 50.0))
    gfit = SimpleNamespace(horizon_fits=[SimpleNamespace(a=A, m=M, sigma_w=SW, sigma_s=50.0,
                                                         is_calibrated=True)])
    mp = wss.build_station_mapping(
        ec, gfit, method="idw", point_x=X, point_y=None, sample_positions=obs.pos,
        sample_times=obs.time, theta_obs=obs.theta, probe_layer_indices=[0] * len(obs),
        shrinkage=1.0)
    scalar = (A * np.clip(ec, 1e-6, None) / (SW + 50.0)) ** (1.0 / M)
    assert np.allclose(mp.theta, scalar, atol=1e-3)


def test_station_mapping_validates_shapes():
    ec, obs = _obs()
    gfit = SimpleNamespace(horizon_fits=[SimpleNamespace(a=A, m=M, sigma_w=SW, sigma_s=50.0,
                                                         is_calibrated=True)])
    with pytest.raises(ValueError):
        wss.build_station_mapping(ec, gfit, method="idw", point_x=X[:3], point_y=None,
                                  sample_positions=obs.pos, sample_times=obs.time,
                                  theta_obs=obs.theta)


# ------------------------------------------------------------------ F
def test_probe_errors_in_sample_metrics_and_csv(tmp_path):
    ec, obs = _obs()
    params = np.repeat(np.array([A, M, SW, 50.0])[:, None, None], NX, axis=2)       # (4, 1, NX)
    calc = wss.in_sample_theta(obs, params)
    recs = wss.probe_error_records(obs, calc, held_out=False)
    assert len(recs) == len(obs) and all(not r["held_out_flag"] for r in recs)
    assert recs[0]["error"] == pytest.approx(recs[0]["theta_calc"] - recs[0]["theta_TDR"])
    met = wss.error_metrics(recs)
    assert met["n"] == len(obs) and set(met["by_epoch"]) == set(range(NT))
    assert len(met["by_station"]) == NX
    assert wss.error_metrics(recs, held_out=True)["n"] == 0
    path = wss.write_probe_errors_csv(recs, str(tmp_path / "probe_errors.csv"))
    assert next(csv.reader(open(path))) == list(wss.PROBE_ERROR_COLUMNS)


def test_nan_ec_is_skipped_in_records():
    ec, obs = _obs()
    obs.ec[0] = np.nan
    params = np.repeat(np.array([A, M, SW, 50.0])[:, None, None], NX, axis=2)
    recs = wss.probe_error_records(obs, wss.in_sample_theta(obs, params), held_out=False)
    assert len(recs) == len(obs) - 1


# ------------------------------------------------------------------ E
def test_station_plot_is_written(tmp_path):
    from plots import ws_spatial_plots
    ec, obs = _obs()
    gfit = SimpleNamespace(horizon_fits=[SimpleNamespace(a=A, m=M, sigma_w=SW, sigma_s=50.0,
                                                         is_calibrated=True)])
    mp = wss.build_station_mapping(
        ec, gfit, method="idw", point_x=X, point_y=None, sample_positions=obs.pos,
        sample_times=obs.time, theta_obs=obs.theta, probe_layer_indices=[0] * len(obs))
    out = ws_spatial_plots.plot_station_params(mp, X, None, str(tmp_path / "m.png"))
    assert (tmp_path / "m.png").stat().st_size > 5000 and out.endswith("m.png")
    with pytest.raises(ValueError):
        ws_spatial_plots.plot_station_params(SimpleNamespace(station_fits=None), X, None, "x.png")


# ------------------------------------------------------------------ D: Step 5 wiring
def test_step5_station_branch_switches_the_maps_and_writes_outputs(tmp_path):
    from step4_select import SelectionOutput  # noqa: F401  (same container as test_localized_mapping)
    from steps import step5_selection as s5

    class Rep:
        def __init__(self):
            self.logs, self.metrics = [], {}
        def log(self, m):
            self.logs.append(m)
        def metric(self, k, v):
            self.metrics[k] = v

    ec, obs = _obs()
    gfit = SimpleNamespace(horizon_fits=[SimpleNamespace(a=A, m=M, sigma_w=SW, sigma_s=50.0,
                                                         is_calibrated=True)])
    so = SimpleNamespace(K=1, best_fit=gfit, theta_calc=np.zeros((NT, NX, 1)))
    rep = Rep()
    mp = s5.run_step5_localized_mapping(
        rep, selected_output=so, averaged_models=[ec], requested_mode=None, bandwidth=None,
        bandwidth_source="", point_x=X, point_y=None, sample_positions=obs.pos,
        sample_times=obs.time, theta_obs=obs.theta, probe_layer_indices=[0] * len(obs), lab_p0=None,
        fixed_params=None, theta_obs_std=None, ws_bounds=None, ws_spatial="voronoi",
        sensor_ids=None, output_dir=str(tmp_path), spatial_shrinkage=0.0)
    assert mp is not None and so.ws_map_mode == "voronoi" and so.localized_mapping is mp
    assert so.theta_calc_global.shape == so.theta_calc.shape
    assert (tmp_path / "ws_by_station.csv").exists() and (tmp_path / "ws_spatial_sigma_s_map.png").exists()
    assert rep.metrics["Step 5 — stations used"] == str(NX)
    truth = (A * ec[:, :, 0] / (SW + SS[None])) ** (1.0 / M)
    assert np.allclose(so.theta_calc[:, :, 0], truth, atol=2e-3)


def test_step5_pooled_and_unknown_spatial_choice():
    from steps import step5_selection as s5
    so = SimpleNamespace(K=1, best_fit=None, theta_calc=None)
    rep = SimpleNamespace(log=lambda m: None, metric=lambda k, v: None)
    kw = dict(selected_output=so, averaged_models=[], requested_mode=None, bandwidth=30.0,
              bandwidth_source="", point_x=X, point_y=None, sample_positions=[], sample_times=[],
              theta_obs=[], probe_layer_indices=None, lab_p0=None, fixed_params=None,
              theta_obs_std=None, ws_bounds=None)
    assert s5.run_step5_localized_mapping(rep, ws_spatial="global", **kw) is None
    assert so.ws_map_mode == "global"
    with pytest.raises(ValueError):
        s5.run_step5_localized_mapping(rep, ws_spatial="kriging", **kw)


def test_renamed_choices_and_old_aliases(capsys):
    assert wss.SPATIAL_CHOICES == ("global", "localized")            # selectable on the command line
    assert wss.STATION_METHODS == ("voronoi", "idw")                  # comparison only (RD0.1)
    assert wss.ALL_METHODS == ("global", "localized", "voronoi", "idw")
    assert wss.SIMPLICITY_ORDER == ("global", "voronoi", "idw", "localized")
    assert wss.normalize_spatial("pooled") == "global" and wss.normalize_spatial("kernel") == "localized"
    assert "renamed" in capsys.readouterr().out
    assert wss.normalize_spatial("idw") == "idw" and wss.normalize_spatial(None) is None
    assert wss.normalize_spatial("kriging") == "kriging"          # unknown: left for the validator


def test_old_names_still_accepted_downstream():
    _, obs = _obs()
    p = wss.loso_predictions(obs, 1, methods=("pooled", "voronoi"), lab_p0=np.array([A, M, SW, 50.0]),
                             shrinkage=0.0)
    assert set(p) == {"global", "voronoi"}                          # old name mapped to the new one


def test_cli_accepts_old_and_new_names():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws-spatial", type=wss.normalize_spatial, choices=list(wss.SPATIAL_CHOICES))
    assert ap.parse_args(["--ws-spatial", "localized"]).ws_spatial == "localized"
    assert ap.parse_args(["--ws-spatial", "kernel"]).ws_spatial == "localized"
    assert ap.parse_args(["--ws-spatial", "pooled"]).ws_spatial == "global"


def test_station_methods_are_not_cli_choices():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws-spatial", type=wss.normalize_spatial, choices=list(wss.SPATIAL_CHOICES))
    for name in ("voronoi", "idw"):
        with pytest.raises(SystemExit):
            ap.parse_args(["--ws-spatial", name])


def test_run_pipeline_cli_has_no_station_options():
    import re
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_pipeline.py")).read()
    assert "--ws-spatial-power" not in src and "--ws-spatial-shrinkage" not in src
    sh = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_one_scenario.sh")).read()
    assert "(global|localized)" in sh and "voronoi|idw" not in sh
