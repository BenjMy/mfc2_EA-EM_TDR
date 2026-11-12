"""Generated scenario families (Phase 8 of refactoring.md).

The surface-vs-full 1-D study needs 16 near-identical scenarios. Instead of
pasting them in ``ensemble_scenarios.json`` they are generated here from the two
base scenarios that stay in the JSON:

    1d_surface_tdr    (TDR only at 0.2 m, horizon 0 calibrated)
    1d_full_depth_ref (TDR in every horizon)

    1d_{surface,full}_free{K}         K = number of FREE Waxman-Smits parameters
    1d_{surface,full}_hetero_free{K}  same, with a two-zone vertical truth

Freeing order (K=1..4): sigma_s, m, sigma_w, a; the others stay fixed at ``lab_p0``.
Adding a K is a one-line change of ``FREE_KS``.

``generate_study_scenarios(scenarios)`` returns raw (unresolved) entries; the
hetero ones use ``"extends"``. ``scenario_loader`` adds them to the file's own
entries (an entry present in the JSON always wins, so a hand-written override
keeps working). Nothing is generated when the base scenarios are absent (e.g.
``ensemble_scenarios_real.json``).
"""
from __future__ import annotations

import copy

FREE_ORDER = ("sigma_s", "m", "sigma_w", "a")  # order in which parameters are freed
FREE_KS = (1, 2, 3, 4)

# kind -> (base scenario, how the TDR coverage is described)
BASES = {
    "surface": ("1d_surface_tdr",
                "TDR ONLY at 0.2 m (horizon 0 calibrated; horizons 1-3 uncalibrated)"),
    "full": ("1d_full_depth_ref",
             "TDR probes in all 4 horizons (reference)"),
}

# K = 4 frees 'a': start it off its lower bound and let it reach below 0.5.
A_START_WHEN_FREE = 1.0
A_BOUNDS_WHEN_FREE = [0.2, 5.0]

# Two-zone vertical truth of the *_hetero_* family (horizons 2-3).
HETERO_LAYER_PARAMS = {"2": {"a": 0.35, "sigma_w": 700.0},
                       "3": {"a": 0.35, "sigma_w": 700.0}}


def _free_params(k: int) -> tuple:
    return FREE_ORDER[:k]


def _free_entry(kind: str, k: int, base: dict) -> dict:
    base_name, coverage = BASES[kind]
    name = f"1d_{kind}_free{k}"
    free = _free_params(k)
    entry = copy.deepcopy(base)
    entry["description"] = (
        f"Surface-vs-full study, K={k} free Waxman-Smits parameter(s): "
        f"{', '.join(free)} (freed in order {', '.join(FREE_ORDER)}; the rest fixed at "
        f"lab_p0). {coverage}. Identical truth/noise/inversion to '{base_name}'. "
        f"Run with --mode 1d --num-swc-positions N; compare '1d_surface_free{k}' vs "
        f"'1d_full_free{k}' via run_1d_surface_vs_full.sh."
    )
    entry["output_dir"] = f"mfc2_outputs/field_survey_{name}"
    entry["cache_tag"] = name
    entry["ws_fixed_params"] = {p: (p not in free) for p in ("a", "m", "sigma_w", "sigma_s")}
    if "a" in free:
        lab_p0 = list(entry["lab_p0"])
        lab_p0[0] = A_START_WHEN_FREE
        entry["lab_p0"] = lab_p0
        entry["ws_bounds"] = {"a": list(A_BOUNDS_WHEN_FREE)}
    return entry


def _hetero_entry(kind: str, k: int) -> dict:
    coverage = ("TDR ONLY at 0.2 m" if kind == "surface" else "TDR in every horizon")
    parent = f"1d_{kind}_free{k}"
    return {
        "extends": parent,
        "description": (
            f"Surface-vs-full study with a TWO-ZONE vertical truth, K={k} free WS "
            f"parameter(s) (same calibration as '{parent}'; {coverage}). True a and "
            f"sigma_w are the defaults (0.5, 500) in horizons 0-1 and change to "
            f"(0.35, 700) in horizons 2-3; m and sigma_s still vary along x as in the "
            f"base study."
        ),
        "output_dir": f"mfc2_outputs/field_survey_1d_{kind}_hetero_free{k}",
        "ws_layer_params": copy.deepcopy(HETERO_LAYER_PARAMS),
    }


def generate_study_scenarios(scenarios: dict) -> dict:
    """Return the generated ``1d_*_free{K}`` / ``1d_*_hetero_free{K}`` entries.

    ``scenarios`` is the raw name -> entry mapping of one JSON file. A family is
    generated only when its base scenario is present in it.
    """
    out: dict = {}
    for kind, (base_name, _) in BASES.items():
        base = scenarios.get(base_name)
        if base is None:
            continue
        for k in FREE_KS:
            out[f"1d_{kind}_free{k}"] = _free_entry(kind, k, base)
    for kind, (base_name, _) in BASES.items():
        if base_name not in scenarios:
            continue
        for k in FREE_KS:
            out[f"1d_{kind}_hetero_free{k}"] = _hetero_entry(kind, k)
    return out


def with_generated(scenarios: dict) -> dict:
    """``scenarios`` plus the generated entries that are not already in it."""
    gen = generate_study_scenarios(scenarios)
    return {**scenarios, **{k: v for k, v in gen.items() if k not in scenarios}}
