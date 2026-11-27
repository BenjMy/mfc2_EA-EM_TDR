"""Tests of em_options.py (EM inversion method switch) and its use by the study / hypothesis / real runs.

No pipeline run and no emagpy needed: scenarios are only generated and validated with scenario_loader.
"""
import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import em_options as eo  # noqa: E402
import hyp3_localized_bandwidth as hyp3  # noqa: E402
import run_real_study as rs  # noqa: E402
import scenario_loader as sl  # noqa: E402
import study_swc_design as st  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = {f: tuple(v) for f, v in st.LEVELS.items()}


# ----------------------------------------------------------------------------- em_options
def test_base_is_a_noop():
    sc = {"param_grid": {"method": ["ROPE"]}, "cache_tag": "x", "output_dir": "out/x"}
    before = copy.deepcopy(sc)
    assert eo.apply_em_method(sc, "base") == before
    assert eo.em_param_grid("base") == {}
    assert eo.method_name("base") is None


def test_lbfgsb_sets_method_and_isolates_paths():
    sc = {"param_grid": {"alpha": [0.1], "method": ["ROPE"]}, "cache_tag": "x", "output_dir": "out/x"}
    eo.apply_em_method(sc, "lbfgsb")
    assert sc["param_grid"] == {"alpha": [0.1], "method": ["L-BFGS-B"]}      # other grid entries kept
    assert sc["cache_tag"] == "x_lbfgsb" and sc["output_dir"] == "out/x_lbfgsb"
    eo.apply_em_method(sc, "lbfgsb")                                          # idempotent
    assert sc["cache_tag"] == "x_lbfgsb"


def test_tag_paths_false_and_inherited_method_untouched():
    child = {"extends": "parent", "cache_tag": "c"}                          # no own param_grid.method
    eo.apply_em_method(child, "lbfgsb")
    assert "param_grid" not in child
    own = {"param_grid": {"method": ["ROPE"]}, "cache_tag": "c"}
    eo.apply_em_method(own, "lbfgsb", tag_paths=False)
    assert own["cache_tag"] == "c" and own["param_grid"]["method"] == ["L-BFGS-B"]


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        eo.check_em("rope_fast")
    assert "nm" not in eo.EM_CHOICES        # Nelder-Mead ignores the bounds in EMagPy: not offered


def test_apply_to_config_does_not_mutate_and_keeps_comments():
    cfg = {"_comment": ["c"], "_schema": {"x": 1},
           "a": {"param_grid": {"method": ["ROPE"]}, "cache_tag": "a", "output_dir": "o/a"}}
    before = copy.deepcopy(cfg)
    out = eo.apply_to_config(cfg, "lbfgsb")
    assert cfg == before                                                       # input untouched
    assert out["_schema"] == {"x": 1} and out["a"]["param_grid"]["method"] == ["L-BFGS-B"]
    assert "lbfgsb" in out["_comment"][-1].lower() or "L-BFGS-B" in out["_comment"][-1]
    assert eo.apply_to_config(cfg, "base") == before


@pytest.mark.parametrize("cfg_name", ["ensemble_scenarios_real.json", "ensemble_scenarios.json"])
def test_derived_config_loads_and_every_method_is_switched(cfg_name, tmp_path):
    src = os.path.join(ROOT, cfg_name)
    out = str(tmp_path / "derived.json")
    assert eo.main(["--config", src, "--em-method", "lbfgsb", "--out", out]) == 0
    with open(src, encoding="utf-8") as f:
        a = json.load(f)
    with open(out, encoding="utf-8") as f:
        b = json.load(f)
    n = 0
    for k, v in a.items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        if isinstance(v.get("param_grid"), dict) and "method" in v["param_grid"]:
            assert b[k]["param_grid"]["method"] == ["L-BFGS-B"]
            n += 1
        if "cache_tag" in v:
            assert b[k]["cache_tag"] == v["cache_tag"] + "_lbfgsb"
    assert n >= 5
    sl.load_scenario(next(k for k in b if not k.startswith("_") and isinstance(b[k], dict)), config_path=out)


def test_derived_path():
    assert eo.derived_path("a/b/ensemble_scenarios_real.json", "lbfgsb") == "a/b/ensemble_scenarios_real_lbfgsb.json"


# ----------------------------------------------------------------------------- study: em factor
def test_default_designs_and_names_are_unchanged_by_em():
    for preset in ("smoke", "oat", "main", "known"):
        for d in st.build_designs(preset, LEVELS):
            assert d.em == "base" and "_em" not in d.name


def test_em_name_roundtrip_and_order():
    d = st.Design("homo", "homo", "noreg", "spread", 5, 0, "k", 0, "lbfgsb")
    assert d.name == "swcstudy_smchomo_wshomo_noreg_spread_n5_free0_emlbfgsb"
    assert st.parse_name(d.name) == d
    d2 = st.Design("homo", "soilhomo", "noreg", "spread", 5, 3, "sw", 10, "lbfgsb")
    assert d2.name.endswith("_pinsw_errp10_emlbfgsb") and st.parse_name(d2.name) == d2
    assert st.parse_name("swcstudy_smchomo_wshomo_noreg_spread_n5_free0_emnope") is None
    assert st.parse_name("swcstudy_smchomo_wshomo_noreg_spread_n5_free0_embase") is None   # base never in a name


def test_em_levels_multiply_every_block_and_k0_is_deduplicated_per_em():
    lv = {**LEVELS, "em": ("base", "lbfgsb"), "k": (0, 4), "n": (5,), "dist": ("spread",), "smc": ("homo",),
          "ws": ("homo",), "reg": ("noreg",)}
    ds = st.build_designs("known", lv)
    assert {d.em for d in ds} == {"base", "lbfgsb"}
    by_em = {e: [d for d in ds if d.em == e] for e in ("base", "lbfgsb")}
    assert len(by_em["base"]) == len(by_em["lbfgsb"]) > 0
    assert sum(d.k == 0 and d.pin == "k" for d in by_em["lbfgsb"]) == 1            # one K=0 floor per em


def _study_cfg(tmp_path, *extra):
    """Write the study config through the CLI (dry run) and return the parsed scenarios."""
    out = tmp_path / "study"
    argv = ["--preset", "known", "--smc", "homo", "--reg", "noreg", "--free-k", "0", "--n-swc", "5",
            "--dist", "spread", "--out-dir", str(out), "--dry-run", *extra]
    assert st.main(argv) in (0, None)
    with open(out / "study_scenarios.json", encoding="utf-8") as f:
        return json.load(f), str(out / "study_scenarios.json")


def test_cli_em_method_generates_loadable_lbfgsb_scenarios(tmp_path):
    cfg, path = _study_cfg(tmp_path, "--ws", "homo", "--em-method", "base", "lbfgsb")
    names = [k for k in cfg if k.startswith("swcstudy_") and k != "swcstudy_parent"]
    lb = [n for n in names if n.endswith("_emlbfgsb")]
    base = [n for n in names if not n.endswith("_emlbfgsb")]
    assert lb and len(lb) == len(base)
    for n in lb:
        sc = sl.load_scenario(n, config_path=path)
        assert sc.param_grid["method"] == ["L-BFGS-B"]
        assert sc.param_grid["forwardModel"] == ["FSlin", "CS"] and sc.param_grid["alpha"]   # ensemble kept
    for n in base:
        assert "method" not in (cfg[n].get("param_grid") or {})                     # inherits the base scenario
        assert sl.load_scenario(n, config_path=path).param_grid["method"] == ["L-BFGS-B"]   # now L-BFGS-B


# ----------------------------------------------------------------------------- study: soil ws cases
SOIL = ("soilhomo", "soilheterox", "soilheteroz")


def test_soil_baseline_values_and_legacy_baseline_untouched():
    assert st.SOIL_WS_BASELINE == (0.8, 2.0, 100.0, 100.0)
    for ws in ("homo", "hetero", "heterox", "heteroz"):
        assert st.WS_BASELINE[ws] == st.GENERATOR_WS_DEFAULTS
        assert list(st.ws_truth_baseline(ws)[0])[:1] == [0.5] or ws in ("hetero",)     # legacy layer-0 truth: a = 0.5
    legacy_homo = st.ws_truth_baseline("homo")
    assert all(list(row) == list(st.GENERATOR_WS_DEFAULTS) for row in legacy_homo)     # (0.5, 3, 500, 100) in every layer
    soil = st.ws_truth_baseline("soilhomo")
    assert all(list(row) == list(st.SOIL_WS_BASELINE) for row in soil)
    assert list(st.ws_truth_baseline("soilheteroz")[0]) == pytest.approx([0.8, 2.0 * 2 / 3, 100.0, 50.0], abs=1e-3)


def test_soil_xy_ranges_scale_with_the_baseline():
    f = st.WS_FIELDS["soilheterox"]
    assert f["ws_xy_ranges"]["m"] == pytest.approx([2.0 * 2 / 3, 2.0 * 4 / 3], abs=1e-3)
    assert f["ws_xy_ranges"]["sigma_s"] == [50.0, 200.0]
    legacy = st.WS_FIELDS["heterox"]["ws_xy_ranges"]
    assert legacy["m"] == [2.0, 4.0] and legacy["sigma_s"] == [50.0, 200.0]          # legacy = same rule at m = 3


def test_soil_heteroz_ramps_inside_default_bounds():
    lp = st.WS_FIELDS["soilheteroz"]["ws_layer_params"]
    ms = [lp[str(i)]["m"] for i in range(st.N_LAYERS)]
    ss = [lp[str(i)]["sigma_s"] for i in range(st.N_LAYERS)]
    assert ms == sorted(ms) and min(ms) >= 1.0 and max(ms) <= 4.0                 # default m bounds [1, 4]
    assert ss == sorted(ss) and min(ss) >= 0.0 and max(ss) <= 200.0               # default sigma_s bounds [0, 200]


@pytest.mark.parametrize("ws", SOIL)
def test_soil_scenarios_load(ws, tmp_path):
    cfg, path = _study_cfg(tmp_path, "--ws", ws, "--em-method", "lbfgsb")
    names = [k for k in cfg if k.startswith("swcstudy_") and k != "swcstudy_parent"]
    assert names and all(f"_ws{ws}_" in n for n in names)
    for n in names:
        sl.load_scenario(n, config_path=path)            # also runs the WS start-value / bounds checks


def test_soil_k0_prior_is_the_soil_baseline_not_the_legacy_one(tmp_path):
    cfg, path = _study_cfg(tmp_path, "--ws", "soilhomo", "homo", "--em-method", "base")    # base = no name suffix
    soil = sl.load_scenario("swcstudy_smchomo_wssoilhomo_noreg_spread_n5_free0", config_path=path)
    legacy = sl.load_scenario("swcstudy_smchomo_wshomo_noreg_spread_n5_free0", config_path=path)
    assert list(soil.lab_p0) == list(st.SOIL_WS_BASELINE)
    assert list(legacy.lab_p0) == list(st.GENERATOR_WS_DEFAULTS)       # legacy scenario content unchanged


def test_soil_pinned_values_are_the_soil_truth(tmp_path):
    cfg, path = _study_cfg(tmp_path, "--ws", "soilhomo", "--em-method", "lbfgsb")
    n = next(n for n in cfg if n.endswith("_pinsws_emlbfgsb"))                      # sigma_w and sigma_s known
    sc = sl.load_scenario(n, config_path=path)
    assert sc.lab_p0[2] == 100.0 and sc.lab_p0[3] == 100.0                           # pinned at the soil truth
    assert sc.lab_p0[0] != 0.8 and sc.lab_p0[1] != 2.0                               # the free a, m start off the truth


# ----------------------------------------------------------------------------- hypothesis / real study hooks
def test_hyp_dry_run_with_em_method_writes_separate_config(tmp_path, capsys):
    out = str(tmp_path / "hyp3")
    assert hyp3.main(["--dry-run", "--out-dir", out, "--em-method", "lbfgsb"]) == 0
    assert os.path.exists(os.path.join(out, f"{hyp3.TAG}_lbfgsb_scenarios.json"))
    assert not os.path.exists(os.path.join(out, f"{hyp3.TAG}_scenarios.json"))
    with open(os.path.join(out, f"{hyp3.TAG}_lbfgsb_scenarios.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    entries = [v for k, v in cfg.items() if not k.startswith("_") and isinstance(v, dict)]
    assert all(v["param_grid"]["method"] == ["L-BFGS-B"] for v in entries if "method" in (v.get("param_grid") or {}))
    assert all(v["cache_tag"].endswith("_lbfgsb") for v in entries if "cache_tag" in v)
    assert "L-BFGS-B" in capsys.readouterr().out


def test_hyp_dry_run_default_is_lbfgsb_and_base_is_the_legacy_untagged_run(tmp_path):
    out = str(tmp_path / "hyp3")
    assert hyp3.main(["--dry-run", "--out-dir", out]) == 0                       # default = lbfgsb
    assert os.path.exists(os.path.join(out, f"{hyp3.TAG}_lbfgsb_scenarios.json"))
    out_b = str(tmp_path / "hyp3_base")
    assert hyp3.main(["--dry-run", "--out-dir", out_b, "--em-method", "base"]) == 0   # no override, no suffix
    with open(os.path.join(out_b, f"{hyp3.TAG}_scenarios.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert not any("lbfgsb" in json.dumps(v) for v in cfg.values())


def test_real_study_dry_run_with_em_method(tmp_path):
    out = str(tmp_path / "real")
    assert rs.main(["--dry-run", "--fast", "--free-k", "1", "--out-dir", out, "--em-method", "lbfgsb"]) in (0, None)
    path = os.path.join(out, "study_lbfgsb_scenarios.json")
    assert os.path.exists(path) and not os.path.exists(os.path.join(out, "study_scenarios.json"))
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    names = [k for k in cfg if k.startswith("real_study_") and k != rs.parent_name()]
    assert names
    for n in names:
        assert sl.load_scenario(n, config_path=path).param_grid["method"] == ["L-BFGS-B"]


# ----------------------------------------------------------------------------- lbfgsbfix preset
def test_lbfgsbfix_sets_method_and_fixed_depths_only_on_complete_entries():
    sc = {"param_grid": {"method": ["ROPE"]}, "fixed_depths": False, "cache_tag": "x"}
    eo.apply_em_method(sc, "lbfgsbfix")
    assert sc["param_grid"]["method"] == ["L-BFGS-B"] and sc["fixed_depths"] is True
    assert sc["cache_tag"] == "x_lbfgsbfix"
    child = {"extends": "p", "cache_tag": "c"}                        # inherits both from its (changed) parent
    eo.apply_em_method(child, "lbfgsbfix")
    assert "fixed_depths" not in child and "param_grid" not in child
    plain = {"param_grid": {"method": ["ROPE"]}, "fixed_depths": False}
    eo.apply_em_method(plain, "lbfgsb")                               # the plain preset leaves the depths alone
    assert plain["fixed_depths"] is False
    assert eo.em_fixed_depths("lbfgsb") is None and eo.em_fixed_depths("lbfgsbfix") is True


def test_lbfgsbfix_derived_real_config_keeps_fixed_depths_scenarios_fixed(tmp_path):
    out = str(tmp_path / "d.json")
    eo.main(["--config", os.path.join(ROOT, "ensemble_scenarios_real.json"), "--em-method", "lbfgsbfix", "--out", out])
    with open(out, encoding="utf-8") as f:
        b = json.load(f)
    assert b["real_single_depth_free"]["fixed_depths"] is True          # the only free-depth real scenario is fixed now
    assert b["real_single_base"]["fixed_depths"] is True


def test_study_lbfgsbfix_scenarios_have_fixed_depths_and_roundtrip(tmp_path):
    d = st.Design("homo", "homo", "noreg", "spread", 5, 0, "k", 0, "lbfgsbfix")
    assert d.name.endswith("_emlbfgsbfix") and st.parse_name(d.name) == d
    cfg, path = _study_cfg(tmp_path, "--ws", "homo", "--em-method", "lbfgsb", "lbfgsbfix")
    fix = [n for n in cfg if n.endswith("_emlbfgsbfix")]
    plain = [n for n in cfg if n.endswith("_emlbfgsb")]
    assert fix and plain
    def all_fixed(name):
        fd = sl.load_scenario(name, config_path=path).fixed_depths
        return all(fd) if isinstance(fd, (list, tuple)) else bool(fd)

    for n in fix:
        assert sl.load_scenario(n, config_path=path).param_grid["method"] == ["L-BFGS-B"] and all_fixed(n)
    for n in plain:
        assert not all_fixed(n)                                          # plain lbfgsb keeps the scenario's free depths


# ----------------------------------------------------------------------------- njobs rule
@pytest.mark.parametrize("tasks,cpu,expect", [(4, 16, 4), (3, 8, 2), (8, 8, 1), (40, 8, 1), (1, 1, 1)])
def test_auto_njobs_fills_idle_cores_only(tasks, cpu, expect):
    n, why = eo.resolve_em_njobs("auto", n_tasks=tasks, methods=["L-BFGS-B"], betas=[0.0], cpu_count=cpu)
    assert n == expect and isinstance(why, str) and why


def test_njobs_always_one_for_random_samplers_and_lateral_smoothing():
    assert eo.resolve_em_njobs("auto", n_tasks=2, methods=["ROPE"], betas=[0.0], cpu_count=16)[0] == 1
    assert eo.resolve_em_njobs(8, n_tasks=2, methods=["L-BFGS-B", "ROPE"], betas=[0.0], cpu_count=16)[0] == 1
    assert eo.resolve_em_njobs("auto", n_tasks=2, methods=["L-BFGS-B"], betas=[0.0, 0.5], cpu_count=16)[0] == 1


def test_explicit_njobs_and_parser():
    assert eo.resolve_em_njobs(3, n_tasks=100, methods=["L-BFGS-B"], betas=[0], cpu_count=8)[0] == 3
    assert eo.parse_em_njobs("auto") == "auto" and eo.parse_em_njobs("4") == 4
    import argparse
    for bad in ("0", "-2", "many"):
        with pytest.raises(argparse.ArgumentTypeError):
            eo.parse_em_njobs(bad)
    with pytest.raises(ValueError):
        eo.resolve_em_njobs(0, n_tasks=1, methods=["L-BFGS-B"], betas=[0], cpu_count=4)


def test_run_pipeline_exposes_em_njobs_flag():
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(ROOT, "run_pipeline.py"), "--help"],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0 and "--em-njobs" in r.stdout


def _real_emagpy():
    try:
        import emagpy
        return hasattr(emagpy.Problem, "invert")      # tests/conftest.py stubs emagpy when it is not installed
    except ImportError:
        return False


@pytest.mark.skipif(not _real_emagpy(), reason="needs the real emagpy")
def test_njobs_does_not_change_a_deterministic_inversion(tmp_path):
    import io
    import contextlib
    import numpy as np
    import data_synthetic as ds
    from emagpy import Problem
    gen = str(tmp_path / "gen")
    with contextlib.redirect_stdout(io.StringIO()):
        np.random.seed(0)
        ds.generate_time_lapse_dataset(output_dir=gen, mode="1d", depths0=[0.3, 0.6, 1.0], homogeneous=False)
        csvs, _, _ = ds.export_eca_surveys_to_csv(output_dir=gen, eca_noise_level=0.01)
    out = {}
    for nj in (1, 2):
        k = Problem()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            k.createSurvey(csvs[0], unit="ppt")
            k.setInit(depths0=[0.3, 0.6, 1.0], conds0=[20] * 4, fixedDepths=[True] * 3)
            k.invert(forwardModel="CS", method="L-BFGS-B", alpha=0.1, bnds=[(0.01, 300.0)] * 4, njobs=nj)
        out[nj] = np.asarray(k.models[0])
    np.testing.assert_allclose(out[1], out[2], rtol=1e-9, atol=1e-9)


# ----------------------------------------------------------------------------- default solvers, ROPE optional
def test_defaults_are_lbfgsb_and_rope_is_an_optional_preset():
    assert eo.RUN_DEFAULT_EM == "lbfgsb" and eo.STUDY_DEFAULT_EM == "lbfgsbfix"
    assert eo.method_name("rope") == "ROPE" and eo.em_fixed_depths("rope") is None
    assert "rope" in eo.EM_CHOICES


@pytest.mark.parametrize("fname", ["ensemble_scenarios.json", "ensemble_scenarios_real.json"])
def test_scenario_files_no_longer_carry_rope(fname):
    with open(os.path.join(ROOT, fname), encoding="utf-8") as f:
        assert '"ROPE"' not in f.read()


def test_study_default_is_lbfgsbfix_and_rope_is_opt_in(tmp_path):
    cfg, _ = _study_cfg(tmp_path, "--ws", "homo")
    names = [k for k in cfg if k.startswith("swcstudy_") and k != "swcstudy_parent"]
    assert names and all(n.endswith("_emlbfgsbfix") for n in names)
    cfg, path = _study_cfg(tmp_path, "--ws", "homo", "--em-method", "rope")
    n = next(k for k in cfg if k.endswith("_emrope"))
    assert sl.load_scenario(n, config_path=path).param_grid["method"] == ["ROPE"]


# ----------------------------------------------------------------------------- K = 0 reference of the hypotheses
def test_hyp3_and_hyp2_carry_a_k0_reference_that_shares_the_step1_cache(tmp_path):
    import hyp2_train_val_split as hyp2
    for mod, name in ((hyp3, hyp3.K0_NAME), (hyp2, hyp2.K0_NAME)):
        out = str(tmp_path / mod.TAG)
        assert mod.main(["--dry-run", "--out-dir", out]) == 0
        with open(os.path.join(out, f"{mod.TAG}_lbfgsb_scenarios.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        sc = sl.load_scenario(name, config_path=os.path.join(out, f"{mod.TAG}_lbfgsb_scenarios.json"))
        assert all(sc.ws_fixed_params.values()) and len(sc.ws_fixed_params) == 4
        parent = cfg[name]["extends"]
        assert cfg[name]["cache_tag"] == next(v["cache_tag"] for k, v in cfg.items()
                                              if isinstance(v, dict) and v.get("extends") == parent and k != name)
        out2 = str(tmp_path / (mod.TAG + "_nok0"))
        assert mod.main(["--dry-run", "--no-k0", "--out-dir", out2]) == 0
        with open(os.path.join(out2, f"{mod.TAG}_lbfgsb_scenarios.json"), encoding="utf-8") as f:
            assert name not in json.load(f)


def test_run_all_hypotheses_forwards_em_method_and_no_k0(tmp_path):
    import run_all_hypotheses as rah
    import hyp2_train_val_split as hyp2
    import argparse
    ns = argparse.Namespace(config="c.json", base="b", mode="2d", stage="calibrate", em_method="rope",
                            fast=False, dry_run=False, keep_going=False, no_k0=True)
    argv = rah.forwarded_args(ns, hyp3, str(tmp_path))
    assert argv[argv.index("--em-method") + 1] == "rope" and "--no-k0" in argv
    ns.no_k0 = False
    assert "--no-k0" not in rah.forwarded_args(ns, hyp2, str(tmp_path))
