"""stage_split.py: calibrate/map hand-over, member selection, compatibility, drift."""
import itertools
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

import stage_split as ss

ALPHAS = list(np.logspace(np.log10(0.001), np.log10(5.0), 5))
FMS = ["FSlin", "CS"]
CONDS = [[0.01, 0.01], [0.01, 300.0], [300.0, 0.01], [300.0, 300.0]]


def _combo_list():
    return [(a, 0.0, 0.0, "ROPE", f, list(c))
            for a, f in itertools.product(ALPHAS, FMS) for c in CONDS]


def _meta(combos):
    return [dict(alpha=a, beta=b, gamma=g, method=m, forwardModel=f, conds0=c)
            for a, b, g, m, f, c in combos]


def _record(selected_K=3, **over):
    combos = _combo_list()
    meta = _meta(combos)
    rng = np.random.default_rng(0)
    rmse = rng.uniform(0.02, 0.1, len(meta))
    ranking = np.argsort(rmse)
    ws = [SimpleNamespace(horizon=0, is_calibrated=True, a=0.6, m=1.5, sigma_w=300.0, sigma_s=10.0, rmse=0.03),
          SimpleNamespace(horizon=1, is_calibrated=True, a=0.7, m=1.6, sigma_w=250.0, sigma_s=20.0, rmse=0.04)]
    kw = dict(scenario="real_single_base", run_id="rid", output_dir="out/0001_x_calibrate",
              ranking=ranking, param_metadata=meta, rmse_by_member=rmse,
              selected_K=selected_K, selected_rmse=0.03,
              ws_by_horizon=ss.ws_rows_from_fit(ws), localized_bandwidth=12.5,
              bandwidth_from_sweep=True, depths0=[0.225], cond_ranges=[[0.01, 300.0]] * 2,
              em_files=["/x/processed_1.csv", "/x/processed_2.csv"],
              em_subsample={"mode": "probes", "radius_m": 5.0}, quick=False, n_probe_obs=80)
    kw.update(over)
    return ss.build_calibration_record(**kw), ranking, combos


def test_stage_label_and_cache_path_never_equal_full():
    full = "out/s/ensemble_cache_s.pkl"
    assert ss.stage_label("full") == "" and ss.stage_label(None) == ""
    assert ss.stage_label("calibrate") == "_calibrate" and ss.stage_label("map") == "_map"
    assert ss.stage_cache_path(full, "full") == full
    assert ss.stage_cache_path(full, "calibrate") == "out/s/ensemble_cache_s_calibrate.pkl"
    assert len({ss.stage_cache_path(full, s) for s in ss.STAGES}) == 3
    # composes with the --quick suffix
    assert ss.stage_cache_path("out/s/c_quick_e3.pkl", "map").endswith("c_quick_e3_map.pkl")


def test_calibrate_em_subsample_forces_probes_but_keeps_own_radius():
    cfg, replaced = ss.calibrate_em_subsample(None)
    assert cfg["mode"] == "probes" and cfg["radius_m"] == ss.DEFAULT_PROBE_RADIUS_M and replaced
    cfg, replaced = ss.calibrate_em_subsample({"mode": "grid", "cell_m": 3.0})
    assert cfg["mode"] == "probes" and replaced
    cfg, replaced = ss.calibrate_em_subsample({"mode": "probes", "radius_m": 2.0})
    assert cfg["radius_m"] == 2.0 and not replaced


def test_member_key_is_stable_across_json_and_numpy_types():
    a = np.float64(0.07071067811865475)
    k1 = ss.member_key(a, 0.0, 0.0, "ROPE", "FSlin", [np.float64(0.01), 300.0])
    k2 = ss.member_key(json.loads(json.dumps(float(a))), 0, 0, "ROPE", "FSlin", [0.01, 300])
    assert k1 == k2 and hash(k1) == hash(k2)


def test_record_has_top_K_members_in_rank_order_and_survives_json(tmp_path):
    rec, ranking, combos = _record(selected_K=3)
    assert rec["selected_K"] == 3 and [m["rank"] for m in rec["members"]] == [1, 2, 3]
    assert [m["ensemble_index"] for m in rec["members"]] == [int(i) for i in ranking[:3]]
    assert rec["members"][0]["rmse"] <= rec["members"][1]["rmse"] <= rec["members"][2]["rmse"]
    assert rec["em_files"] == ["processed_1.csv", "processed_2.csv"]
    path = ss.write_calibration(str(tmp_path / "run"), rec)
    assert os.path.basename(path) == ss.CALIBRATION_FILENAME
    assert ss.read_calibration(path) == rec


def test_select_members_keeps_exactly_the_selected_ones_in_grid_order():
    rec, ranking, combos = _record(selected_K=5)
    kept, missing = ss.select_members(combos, rec)
    assert missing == [] and len(kept) == 5
    assert {ss.key_from_combo(c) for c in kept} == {ss.key_from_meta(m) for m in rec["members"]}
    assert [combos.index(c) for c in kept] == sorted(combos.index(c) for c in kept)


def test_select_members_reports_missing_when_grid_changed():
    rec, ranking, combos = _record(selected_K=4)
    smaller = [c for c in combos if c[4] == "CS"]          # forward-model list changed
    kept, missing = ss.select_members(smaller, rec)
    n_cs = sum(1 for m in rec["members"] if m["forwardModel"] == "CS")
    assert len(kept) == n_cs and len(missing) == 4 - n_cs


def test_build_record_validates_inputs():
    with pytest.raises(ValueError):
        _record(selected_K=0)
    with pytest.raises(ValueError):
        _record(selected_K=999)
    with pytest.raises(ValueError):
        _record(rmse_by_member=[0.1])


def test_read_calibration_rejects_bad_files(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ValueError):
        ss.read_calibration(str(p))
    rec, *_ = _record()
    rec["schema_version"] = 99
    p.write_text(json.dumps(rec))
    with pytest.raises(ValueError):
        ss.read_calibration(str(p))
    rec["schema_version"] = 1
    rec["selected_K"] = 2                      # != len(members)
    p.write_text(json.dumps(rec))
    with pytest.raises(ValueError):
        ss.read_calibration(str(p))


def test_resolve_calibration_path_file_dir_latest(tmp_path):
    rec, *_ = _record()
    old = ss.write_calibration(str(tmp_path / "0001_s_calibrate"), rec)
    time.sleep(0.02)
    new = ss.write_calibration(str(tmp_path / "0002_s_calibrate"), rec)
    ss.write_calibration(str(tmp_path / "0003_s_map"), rec)     # not a calibrate run
    assert ss.resolve_calibration_path(None, str(tmp_path)) == new
    assert ss.resolve_calibration_path("latest", str(tmp_path)) == new
    assert ss.resolve_calibration_path(os.path.dirname(old), None) == old
    assert ss.resolve_calibration_path(old, None) == old
    with pytest.raises(FileNotFoundError):
        ss.resolve_calibration_path(None, str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError):
        ss.resolve_calibration_path(str(tmp_path / "nope.json"), None)


def test_check_compatibility():
    rec, *_ = _record(quick=True)
    ok = dict(depths0=[0.225], cond_ranges=[[0.01, 300.0]] * 2,
              em_files=["/a/processed_1.csv", "/b/processed_2.csv"], scenario="other")
    errors, warnings = ss.check_compatibility(rec, **ok)
    assert errors == [] and len(warnings) == 2          # other scenario + quick calibration
    assert ss.check_compatibility(rec, **{**ok, "depths0": [0.3]})[0]
    assert ss.check_compatibility(rec, **{**ok, "cond_ranges": [[0.01, 100.0]] * 2})[0]
    assert ss.check_compatibility(rec, **{**ok, "em_files": ["processed_1.csv"]})[0]


def test_drift_report():
    rec, *_ = _record(selected_K=3)
    same = [dict(r) for r in rec["ws_by_horizon"]]
    d = ss.drift_report(rec, 3, same)
    assert not d["warn"] and d["max_rel"] == 0 and len(d["rows"]) == 8
    moved = [dict(r) for r in same]
    moved[0]["a"] *= 1.5                                  # +50 %
    d = ss.drift_report(rec, 3, moved)
    assert d["warn"] and [r["param"] for r in d["rows"] if r["drift"]] == ["a"]
    d = ss.drift_report(rec, 2, same)
    assert d["warn"] and d["k_changed"]
    # an uncalibrated horizon is skipped, not compared
    unc = [dict(r) for r in same]
    unc[1]["is_calibrated"] = False
    assert len(ss.drift_report(rec, 3, unc)["rows"]) == 4
