"""Regenerate tests/data/scenarios_golden.json (golden snapshot of every scenario).

The snapshot holds ``dataclasses.asdict(load_scenario(name))`` for every scenario
of ``ensemble_scenarios.json`` and ``ensemble_scenarios_real.json``. It is the
"old vs new" safety net of refactoring Phase 8: restructuring the scenario files
(``extends``, generated sweeps) must not change any resolved config.

Run it ONLY when a scenario is changed on purpose:

    python tests/make_scenario_golden.py
"""
import dataclasses
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scenario_loader import list_scenarios, load_scenario  # noqa: E402

FILES = ("ensemble_scenarios.json", "ensemble_scenarios_real.json")
OUT = os.path.join(ROOT, "tests", "data", "scenarios_golden.json")


def snapshot() -> dict:
    out = {}
    for fname in FILES:
        path = os.path.join(ROOT, fname)
        out[fname] = {
            name: json.loads(json.dumps(
                dataclasses.asdict(load_scenario(name, config_path=path)), default=str))
            for name in sorted(list_scenarios(path))
        }
    return out


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    snap = snapshot()
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {OUT}: " + ", ".join(f"{k}={len(v)}" for k, v in snap.items()))
