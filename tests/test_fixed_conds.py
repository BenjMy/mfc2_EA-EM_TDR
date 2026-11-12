"""Layer pinning hooks for Step 1: bounds drop pinned layers, and the worker
forwards fixedConds / a per-point conds0 matrix to EMagPy only when asked."""
import numpy as np
import pytest

import geometry_utils as gu
import run_pipeline as rp

DEPTHS0 = [0.3, 0.6, 1.0]
RANGES = [[5.0, 250.0], [5.0, 250.0], [10.0, 300.0], [10.0, 300.0]]


def test_bnds_default_unchanged_and_pinned_layers_dropped():
    base = gu.build_inversion_bnds(DEPTHS0, [True] * 3, RANGES)
    assert base == [tuple(r) for r in RANGES]
    assert gu.build_inversion_bnds(DEPTHS0, [True] * 3, RANGES, fixed_conds=None) == base
    pinned = gu.build_inversion_bnds(DEPTHS0, [True] * 3, RANGES,
                                     fixed_conds=[True, False, False, False])
    assert pinned == base[1:]
    free_depths = gu.build_inversion_bnds(DEPTHS0, [False, True, True], RANGES,
                                          fixed_conds=[True, False, False, False])
    assert len(free_depths) == 1 + 3                      # 1 free depth + 3 free conds
    with pytest.raises(ValueError):
        gu.build_inversion_bnds(DEPTHS0, [True] * 3, RANGES, fixed_conds=[True])


class _FakeProblem:
    calls = {}

    def __init__(self):
        self.models, self.depths, self.pstds = [], [], []

    def createSurvey(self, *a, **k):
        pass

    def setInit(self, **kw):
        _FakeProblem.calls["setInit"] = kw

    def invert(self, **kw):
        _FakeProblem.calls["invert"] = kw
        self.models = [np.ones((5, 4))]
        self.depths = [np.ones((5, 3))]
        self.pstds = [np.zeros((5, 0))]


def _run(monkeypatch, **extra):
    import emagpy
    monkeypatch.setattr(emagpy, "Problem", _FakeProblem)
    kw = {"bnds": [(0.0, 1.0)], "alpha": 0.03, **extra}
    out = rp._invert_single_worker((0, "f.csv", DEPTHS0, [20.0] * 4, [True] * 3, kw))
    assert out[-1] is None, out[-1]
    return _FakeProblem.calls


def test_worker_default_does_not_pin(monkeypatch):
    calls = _run(monkeypatch)
    assert "fixedConds" not in calls["setInit"]
    assert list(calls["setInit"]["conds0"]) == [20.0] * 4


def test_worker_forwards_pinning_and_strips_private_kwargs(monkeypatch):
    mat = np.tile([7.0, 20.0, 20.0, 20.0], (5, 1))
    calls = _run(monkeypatch, fixed_conds=[True, False, False, False], conds0_matrix=mat)
    assert calls["setInit"]["fixedConds"] == [True, False, False, False]
    assert np.array_equal(calls["setInit"]["conds0"], mat)
    assert "fixed_conds" not in calls["invert"] and "conds0_matrix" not in calls["invert"]
