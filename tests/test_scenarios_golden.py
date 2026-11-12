"""Every shipped scenario still resolves to the same config as the golden snapshot.

Guards refactoring Phase 8 (``extends`` / generated sweeps): moving or generating
scenario entries must not change any resolved ScenarioConfig. After an
INTENTIONAL scenario change run ``python tests/make_scenario_golden.py``.
"""
import json
import os

import pytest

import make_scenario_golden as mk

GOLDEN = json.load(open(mk.OUT, encoding="utf-8"))


def test_same_scenario_names():
    now = mk.snapshot()
    for fname in mk.FILES:
        assert sorted(now[fname]) == sorted(GOLDEN[fname]), fname


@pytest.mark.parametrize("fname", mk.FILES)
def test_every_scenario_equals_golden(fname):
    now = mk.snapshot()[fname]
    bad = [n for n, cfg in now.items() if cfg != GOLDEN[fname].get(n)]
    assert not bad, f"{fname}: resolved config changed for {bad}"
