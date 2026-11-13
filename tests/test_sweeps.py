"""sweeps.py: generated 1d_*_free{K} / 1d_*_hetero_free{K} scenario families."""
import json
import os

import pytest

import sweeps
from scenario_loader import describe_scenarios, list_scenarios, load_scenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.join(ROOT, "ensemble_scenarios.json")
REAL = os.path.join(ROOT, "ensemble_scenarios_real.json")
GOLDEN = json.load(open(os.path.join(ROOT, "tests", "data", "scenarios_golden.json"),
                        encoding="utf-8"))["ensemble_scenarios.json"]

NAMES = [f"1d_{kind}_{fam}{k}" for kind in ("surface", "full")
         for fam in ("free", "hetero_free") for k in sweeps.FREE_KS]


def _raw():
    d = json.load(open(SYN, encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}


def test_sixteen_names_generated_from_the_two_bases():
    gen = sweeps.generate_study_scenarios(_raw())
    assert sorted(gen) == sorted(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_generated_scenario_loads_and_equals_golden(name):
    """load_scenario (JSON or generated) returns the config recorded before Phase 8."""
    import dataclasses
    cfg = json.loads(json.dumps(dataclasses.asdict(load_scenario(name, SYN)), default=str))
    assert cfg == GOLDEN[name]


def test_free_k_frees_parameters_in_documented_order():
    gen = sweeps.generate_study_scenarios(_raw())
    for k in sweeps.FREE_KS:
        fixed = gen[f"1d_full_free{k}"]["ws_fixed_params"]
        assert [p for p in sweeps.FREE_ORDER if not fixed[p]] == list(sweeps.FREE_ORDER[:k])
    assert gen["1d_full_free4"]["ws_bounds"] == {"a": [0.2, 5.0]}
    assert gen["1d_full_free4"]["lab_p0"][0] == 1.0
    assert "ws_bounds" not in gen["1d_full_free3"]


def test_surface_and_full_differ_only_by_probe_coverage():
    raw = _raw()
    gen = sweeps.generate_study_scenarios(raw)
    s, f = gen["1d_surface_free2"], gen["1d_full_free2"]
    diff = {k for k in set(s) | set(f) if s.get(k) != f.get(k)}
    assert "swc_depths" in diff
    assert diff <= {"swc_depths", "description", "output_dir", "cache_tag",
                    "allow_uncovered_horizons"}


def test_json_entry_wins_over_generated():
    raw = _raw()
    raw["1d_full_free3"] = {"marker": 1}
    merged = sweeps.with_generated(raw)
    assert merged["1d_full_free3"] == {"marker": 1}


def test_nothing_generated_without_bases():
    real = json.load(open(REAL, encoding="utf-8"))
    assert sweeps.generate_study_scenarios(
        {k: v for k, v in real.items() if not k.startswith("_")}) == {}
    assert not any(n.startswith("1d_") for n in list_scenarios(REAL))


def test_added_k_is_one_line(monkeypatch):
    monkeypatch.setattr(sweeps, "FREE_KS", (1, 2, 3, 4, 5))
    # K=5 frees nothing new (only 4 parameters) but must still produce a valid entry
    gen = sweeps.generate_study_scenarios(_raw())
    assert "1d_surface_free5" in gen and "1d_full_hetero_free5" in gen


def test_listing_includes_generated_names():
    names = [n for n, _ in describe_scenarios(SYN)]
    assert set(NAMES) <= set(names)
    assert set(NAMES) <= set(list_scenarios(SYN))
