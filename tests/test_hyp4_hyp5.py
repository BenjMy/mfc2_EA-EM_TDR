"""hyp4_solver_type.py and hyp5_layer_depths.py: scenarios differ ONLY by the factor under test."""
import argparse
import json
import os

import pytest

import hyp4_solver_type as hyp4
import hyp5_layer_depths as hyp5
import run_all_hypotheses as rah
import scenario_loader as sl


def _dry(mod, tmp_path, *args):
    out = str(tmp_path / mod.TAG)
    assert mod.main(["--dry-run", "--out-dir", out, *args]) == 0
    path = next(os.path.join(out, f) for f in os.listdir(out) if f.endswith("_scenarios.json"))
    with open(path, encoding="utf-8") as f:
        return json.load(f), path


def test_hyp4_children_differ_only_by_solver(tmp_path):
    cfg, path = _dry(hyp4, tmp_path)
    rope = sl.load_scenario("real_solver_rope", config_path=path)
    lb = sl.load_scenario("real_solver_lbfgsb", config_path=path)
    assert rope.param_grid["method"] == ["ROPE"] and lb.param_grid["method"] == ["L-BFGS-B"]
    assert rope.param_grid["forwardModel"] == lb.param_grid["forwardModel"]    # rest of param_grid inherited
    for field in ("depths0", "fixed_depths", "ws_fixed_params", "lab_p0", "cond_ranges"):
        assert getattr(rope, field) == getattr(lb, field)
    assert rope.cache_tag != lb.cache_tag and rope.output_dir != lb.output_dir   # Step 1 depends on the solver


def test_hyp4_has_no_em_method_option_and_k0_shares_the_solver_cache(tmp_path):
    with pytest.raises(SystemExit):
        hyp4.main(["--dry-run", "--em-method", "lbfgsb", "--out-dir", str(tmp_path)])
    cfg, path = _dry(hyp4, tmp_path)
    for s in hyp4.SOLVERS:
        k0 = sl.load_scenario(hyp4.k0_name(s), config_path=path)
        sc = sl.load_scenario(hyp4.scenario_name(s), config_path=path)
        assert all(k0.ws_fixed_params.values()) and k0.cache_tag == sc.cache_tag and k0.output_dir == sc.output_dir
        assert k0.param_grid["method"] == sc.param_grid["method"]


def test_hyp4_solver_subset_and_no_k0(tmp_path):
    cfg, _ = _dry(hyp4, tmp_path, "--solvers", "rope", "--no-k0")
    assert "real_solver_rope" in cfg and "real_solver_lbfgsb" not in cfg
    assert not any(k.startswith("real_solver_k0_") for k in cfg)


def test_hyp5_children_differ_only_by_fixed_depths(tmp_path):
    cfg, path = _dry(hyp5, tmp_path)
    fx = sl.load_scenario("real_depth_fixed", config_path=path)
    fr = sl.load_scenario("real_depth_free", config_path=path)
    assert fx.fixed_depths is True and fr.fixed_depths is False
    assert fx.param_grid == fr.param_grid and fx.depths0 == fr.depths0 and fx.ws_fixed_params == fr.ws_fixed_params
    assert fx.cache_tag != fr.cache_tag
    for m, sc in (("fixed", fx), ("free", fr)):
        k0 = sl.load_scenario(hyp5.k0_name(m), config_path=path)
        assert k0.fixed_depths == sc.fixed_depths and k0.cache_tag == sc.cache_tag and all(k0.ws_fixed_params.values())


def test_hyp5_rejects_lbfgsbfix_because_it_would_cancel_the_factor(tmp_path):
    with pytest.raises(SystemExit):
        hyp5.main(["--dry-run", "--em-method", "lbfgsbfix", "--out-dir", str(tmp_path)])
    cfg, path = _dry(hyp5, tmp_path, "--em-method", "rope")
    assert sl.load_scenario("real_depth_free", config_path=path).fixed_depths is False   # em did not force it
    assert sl.load_scenario("real_depth_free", config_path=path).param_grid["method"] == ["ROPE"]


def test_no_scenario_name_is_a_prefix_of_another():
    """run_log rows are matched with startswith(name)."""
    for names in (["real_solver_rope", "real_solver_lbfgsb", *hyp4.K0_NAMES],
                  ["real_depth_fixed", "real_depth_free", *hyp5.K0_NAMES]):
        assert not any(a != b and b.startswith(a) for a in names for b in names)


def test_run_all_forwards_em_method_only_where_accepted(tmp_path):
    ns = argparse.Namespace(config="c.json", base="b", mode="2d", stage="calibrate", em_method="lbfgsbfix",
                            fast=False, dry_run=False, keep_going=False, no_k0=True)
    assert "--em-method" not in rah.forwarded_args(ns, hyp4, str(tmp_path))     # solver is hyp4's factor
    assert "--em-method" not in rah.forwarded_args(ns, hyp5, str(tmp_path))     # lbfgsbfix not accepted
    ns.em_method = "rope"
    assert "--em-method" in rah.forwarded_args(ns, hyp5, str(tmp_path))
    assert "--no-k0" in rah.forwarded_args(ns, hyp4, str(tmp_path))


def test_run_all_default_set_excludes_the_slow_hypotheses():
    assert set(rah.DEFAULT_SET) == {"1", "2", "3"} and {"4", "5"} <= set(rah.HYPOTHESES)


def test_run_log_classifies_hyp4_and_hyp5_scenarios(tmp_path):
    from run_log import RunLog
    log = RunLog(str(tmp_path / "l.csv"))
    names = [*(hyp4.scenario_name(s) for s in hyp4.SOLVERS), *hyp4.K0_NAMES,
             *(hyp5.scenario_name(m) for m in hyp5.DEPTH_MODES), *hyp5.K0_NAMES]
    for i, n in enumerate(names):
        log.append({"run_id": str(i), "scenario": n})
    types = {r["scenario"]: r["study_type"] for r in log.rows()}
    assert {types[n] for n in names if n.startswith("real_solver_")} == {"hyp4_solver_type"}
    assert {types[n] for n in names if n.startswith("real_depth_")} == {"hyp5_layer_depths"}
