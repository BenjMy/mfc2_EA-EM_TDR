"""quick_preset.py: grid reduction, epoch subset, observation remap, cache safety, fingerprint."""
import numpy as np
import pytest

import quick_preset as qp


def _real_single_base_grid():
    return {
        "method": ["ROPE"],
        "forwardModel": ["FSlin", "CS"],
        "alpha": list(np.logspace(np.log10(0.001), np.log10(5.0), 5)),
        "beta": [0.0],
        "gamma": [0.0],
    }


def test_task_counts_match_readme_280_28_12():
    grid, n_start = _real_single_base_grid(), 4
    assert qp.count_tasks(grid, n_start, 7) == 280
    reduced, _ = qp.reduce_param_grid(grid)
    assert qp.count_tasks(reduced, n_start, 7) == 28
    assert qp.count_tasks(reduced, n_start, len(qp.select_epochs(7, 3))) == 12


def test_reduce_keeps_middle_alpha_first_model_and_does_not_mutate():
    grid = _real_single_base_grid()
    before = {k: list(v) for k, v in grid.items()}
    reduced, info = qp.reduce_param_grid(grid)
    assert reduced["alpha"] == [grid["alpha"][2]]
    assert reduced["forwardModel"] == ["FSlin"]
    assert reduced["method"] == ["ROPE"] and reduced["beta"] == [0.0]
    assert {k: list(v) for k, v in grid.items()} == before
    assert info["alpha"]["n_full"] == 5 and info["forwardModel"]["n_full"] == 2


def test_reduce_leaves_single_valued_axes_alone():
    reduced, info = qp.reduce_param_grid({"alpha": [0.03], "forwardModel": ["CS"]})
    assert reduced == {"alpha": [0.03], "forwardModel": ["CS"]} and info == {}


def test_select_epochs():
    assert qp.select_epochs(7, 3) == [0, 3, 6]
    assert qp.select_epochs(7, 2) == [0, 6]
    assert qp.select_epochs(7, 1) == [0]
    assert qp.select_epochs(7, None) == list(range(7))
    assert qp.select_epochs(7, 99) == list(range(7))
    for n, k in [(7, 4), (10, 5), (3, 2), (50, 7)]:
        e = qp.select_epochs(n, k)
        assert e[0] == 0 and e[-1] == n - 1 and e == sorted(set(e)) and len(e) == k
    with pytest.raises(ValueError):
        qp.select_epochs(7, 0)


def test_subset_epochs_remaps_time_index_and_does_not_mutate():
    files = [f"f{i}.csv" for i in range(7)]
    dates = list(range(7))
    obs = [{"probe_index": p, "time_index": t, "depth": 0.15, "swc": 0.2}
           for p in range(2) for t in range(7)]
    snapshot = [dict(o) for o in obs]
    f, d, o = qp.subset_epochs(files, dates, obs, [0, 3, 6])
    assert f == ["f0.csv", "f3.csv", "f6.csv"] and d == [0, 3, 6]
    assert sorted({x["time_index"] for x in o}) == [0, 1, 2]
    assert len(o) == 2 * 3
    # the observation that was at original epoch 3 is now epoch 1
    assert all(x["time_index"] in (0, 1, 2) for x in o)
    assert obs == snapshot


def test_quick_cache_never_equals_full_cache_path():
    full = "mfc2_outputs/real_single_base/ensemble_cache_real_single_base.pkl"
    assert qp.quick_cache_path(full, None) != full
    assert qp.quick_cache_path(full, None).endswith("_quick.pkl")
    assert qp.quick_cache_path(full, 3).endswith("_quick_e3.pkl")
    assert qp.quick_cache_path(full, 3) != qp.quick_cache_path(full, None)


def test_fingerprint_changes_with_quick_and_epochs_but_not_otherwise():
    import run_pipeline as rp
    grid = _real_single_base_grid()
    args = dict(depths0=[0.225], fixed_depths=True, cond_ranges=[[0.01, 300.0]] * 2, gamma=[0.0])
    full = rp._config_fingerprint(param_grid=grid, **args)
    # existing caches keep their hash when epochs is not used
    assert full == rp._config_fingerprint(param_grid=grid, epochs=None, **args)
    reduced, _ = qp.reduce_param_grid(grid)
    q = rp._config_fingerprint(param_grid=reduced, **args)
    q3 = rp._config_fingerprint(param_grid=reduced, epochs=[0, 3, 6], **args)
    q3b = rp._config_fingerprint(param_grid=reduced, epochs=[0, 2, 6], **args)
    assert len({full, q, q3, q3b}) == 4
