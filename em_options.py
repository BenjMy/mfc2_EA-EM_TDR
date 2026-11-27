#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
em_options.py -- choose the EM inversion method (``param_grid.method``) of any run.

Why
---
``diagnose_ec_inversion.py`` showed that the scenarios' ``method = ["ROPE"]`` (EMagPy default
``rep = 100`` random samples, never set by the pipeline) does not fit the data: ~600 % ECa misfit,
inverted EC 2-8 x too high, K = 0 floor 0.13-0.15 m3/m3.  ``L-BFGS-B`` fits (misfit ~5 %) and
gives a K = 0 floor of ~0.02.  This module lets every entry point switch the method WITHOUT
editing the scenario files and without touching the pipeline (the method is an ordinary scenario
field):

    study_swc_design.py   --em-method lbfgsb            (a design factor ``em``, see that file)
    hyp1/2/3_*.py, run_real_study.py   --em-method lbfgsb   (generated parent + children)
    any scenarios file    python em_options.py --config ensemble_scenarios_real.json \\
                                               --em-method lbfgsb
                          -> ensemble_scenarios_real_lbfgsb.json, to be used with
                             ``REAL_CFG=ensemble_scenarios_real_lbfgsb.json ./run_real_all_scenarios.sh``
                             or ``./run_one_scenario.sh -c ensemble_scenarios_real_lbfgsb.json -s <name>``

``base`` (the default everywhere) leaves the scenario's own method untouched, so every old run
folder, log row and cache stays valid.

Isolation of the outputs
------------------------
With a non-base method the derived scenarios get ``_<em>`` appended to ``cache_tag`` and
``output_dir`` (same names otherwise), so a ROPE and an L-BFGS-B run never share a cache or a run
folder.  (The pipeline's cache fingerprint also contains ``param_grid``, so a stale cache is never
reused silently either way.)

Speed (measured, one core, 60 points, 4 layers; ms per point and member)
-----------------------------------------------------------------------
                         CS      FSlin
    ROPE   fixed depths  26      153      (the study / real scenarios today, rep = 100)
    L-BFGS-B fixed       6.6     106      <- faster than ROPE (CS x4, FSlin x1.4; 2 layers: x5 / x3)
    ROPE   free depths   39      165
    L-BFGS-B free        78      1149     <- SLOWER than ROPE (CS x2, FSlin x7)
So L-BFGS-B only saves time with FIXED depths (all ``real_*`` scenarios except ``*_depth_free``); the
synthetic study scenario has free depths: use ``lbfgsbfix`` there (same truth layers, 7 x fewer free
parameters, accurate).  The forward model is the other big lever: FSlin costs ~16 x CS.  ``--em-njobs``
(see ``resolve_em_njobs``) splits the points of one Step-1 task across cores when there are fewer tasks
than cores (typically the ``map`` stage: K members x epochs).

Notes
-----
* alpha / beta / gamma smoothing work with L-BFGS-B too (EMagPy puts them in the objective of every
  ``scipy.optimize.minimize`` method); the ``ensemble_scenarios*.json`` comment "ROPE is the only
  optimiser that supports alpha/beta/gamma" only applies to the ANN / MCMC / DREAM / SCEUA solvers.
* Nelder-Mead fitted the data too in the diagnosis, but EMagPy documents that it ignores ``bnds``
  (EC could leave ``cond_ranges``), so it is not offered.  TNC did not fit.  To add a method, add a
  line to ``EM_METHODS``.
* ``L-BFGS-B`` is deterministic for a given start, so an ensemble of members differs only by
  alpha / forward model / start model (ROPE also differs by its random sampling).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

# tag (used in names / folders / --em-method) -> scenario fields it sets.  'base' = keep the scenario's own.
#   method        EMagPy ``param_grid.method``
#   fixed_depths  scenario ``fixed_depths`` (interfaces fixed at depths0); omitted = keep the scenario's own
EM_PRESETS: dict[str, dict] = {
    "base": {},
    "lbfgsb": {"method": "L-BFGS-B"},
    "lbfgsbfix": {"method": "L-BFGS-B", "fixed_depths": True},
}
EM_METHODS: dict[str, str | None] = {tag: p.get("method") for tag, p in EM_PRESETS.items()}
EM_CHOICES = tuple(EM_PRESETS)
DEFAULT_EM = "base"
MCMC_FAMILY = frozenset({"MCMC", "ROPE", "DREAM", "SCEUA"})   # random samplers (same set as run_pipeline.py)

_PATH_KEYS = ("cache_tag", "output_dir")


def check_em(em: str) -> str:
    if em not in EM_METHODS:
        raise ValueError(f"unknown EM method {em!r}; choose from {EM_CHOICES}")
    return em


def method_name(em: str) -> str | None:
    """EMagPy method string of ``em`` (None for 'base')."""
    return EM_METHODS[check_em(em)]


def em_fixed_depths(em: str) -> bool | None:
    """``fixed_depths`` value forced by ``em`` (None = keep the scenario's own)."""
    return EM_PRESETS[check_em(em)].get("fixed_depths")


def em_param_grid(em: str) -> dict:
    """``param_grid`` fragment to merge into a scenario (empty for 'base')."""
    m = method_name(em)
    return {"method": [m]} if m else {}


def apply_em_method(scenario: dict, em: str, *, tag_paths: bool = True) -> dict:
    """Set ``param_grid.method`` of ONE scenario dict in place (and return it).

    Only entries that carry their own ``param_grid.method`` are changed; an entry that ``extends`` a
    parent and has none inherits the parent's (already changed) value.  ``tag_paths`` appends
    ``_<em>`` to ``cache_tag`` / ``output_dir`` when present (not needed when the scenario name
    already carries the method, as in study_swc_design.py).  ``base`` is a no-op.
    """
    method = method_name(em)
    if method is None:
        return scenario
    pg = scenario.get("param_grid")
    own_method = isinstance(pg, dict) and "method" in pg
    if own_method:
        pg["method"] = [method]
    fixed = em_fixed_depths(em)
    if fixed is not None and (own_method or "fixed_depths" in scenario):
        scenario["fixed_depths"] = fixed
    if tag_paths:
        for key in _PATH_KEYS:
            val = scenario.get(key)
            if isinstance(val, str) and not val.endswith(f"_{em}"):
                scenario[key] = f"{val}_{em}"
    return scenario


def apply_to_config(cfg: dict, em: str, *, tag_paths: bool = True) -> dict:
    """Deep copy of a scenarios file content with every scenario switched to ``em``.

    Keys starting with ``_`` (comments, schema) are kept untouched.
    """
    out = copy.deepcopy(cfg)
    if method_name(em) is None:
        return out
    for key, entry in out.items():
        if not key.startswith("_") and isinstance(entry, dict):
            apply_em_method(entry, em, tag_paths=tag_paths)
    if isinstance(out.get("_comment"), list):
        out["_comment"] = [*out["_comment"],
                           f"EM method switched to {method_name(em)} by em_options.py (--em-method {em})."]
    return out


def add_em_arg(ap: argparse.ArgumentParser, *, multiple: bool = False) -> None:
    """``--em-method``: one value, or several for the study's factor (``multiple``)."""
    help_ = ("EM inversion method of every generated scenario: base (default) = the scenario's own, "
             "currently ROPE | lbfgsb = L-BFGS-B (fits the data; faster than ROPE only with fixed depths) | "
             "lbfgsbfix = L-BFGS-B with fixed depths (see em_options.py)")
    if multiple:
        ap.add_argument("--em-method", nargs="+", choices=EM_CHOICES, default=None, help=help_)
    else:
        ap.add_argument("--em-method", choices=EM_CHOICES, default=DEFAULT_EM, help=help_)


def resolve_em_njobs(setting, *, n_tasks: int, methods, betas, cpu_count: int | None = None) -> tuple[int, str]:
    """``njobs`` for EMagPy's ``invert`` in every Step-1 task, and why.

    EMagPy can split the POINTS of one inversion across cores (joblib).  Step 1 already runs one task per
    (member x epoch) in a process pool, so more parallelism only helps when there are fewer tasks than
    cores (typically ``--stage map``: K members x epochs).  Rules:

    * random samplers (ROPE / MCMC / DREAM / SCEUA) -> 1: the pipeline seeds every task, a split would
      change the random streams (results no longer reproducible);
    * lateral smoothing (any beta != 0) -> 1: EMagPy cannot split the points then;
    * ``setting`` an int >= 1 -> that value; ``"auto"`` -> ``cpu_count // n_tasks`` when n_tasks < cpu_count,
      else 1 (the pool is already full).  For a deterministic method the results do not depend on it.
    """
    cpu = max(1, int(cpu_count if cpu_count is not None else (os.cpu_count() or 1)))
    meth = {str(m) for m in methods}
    if meth & MCMC_FAMILY:
        return 1, f"random sampler(s) {sorted(meth & MCMC_FAMILY)}: kept serial (reproducible seeding)"
    if any(float(b) != 0.0 for b in betas):
        return 1, "lateral smoothing (beta != 0): EMagPy cannot split the points"
    if setting in (None, "auto"):
        if n_tasks >= cpu:
            return 1, f"{n_tasks} tasks >= {cpu} cores: the process pool is already full"
        n = max(1, cpu // max(1, n_tasks))
        return n, f"auto: {n_tasks} task(s) on {cpu} cores -> {n} job(s) per task"
    n = int(setting)
    if n < 1:
        raise ValueError(f"--em-njobs must be 'auto' or an integer >= 1, got {setting!r}")
    return n, f"explicit --em-njobs {n}"


def parse_em_njobs(text) -> str | int:
    """CLI value: 'auto' or an integer >= 1."""
    if str(text).lower() == "auto":
        return "auto"
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--em-njobs must be 'auto' or an integer >= 1, got {text!r}")
    if n < 1:
        raise argparse.ArgumentTypeError(f"--em-njobs must be 'auto' or an integer >= 1, got {text!r}")
    return n


def derived_path(config_path: str, em: str) -> str:
    root, ext = os.path.splitext(config_path)
    return f"{root}_{em}{ext or '.json'}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Write a copy of a scenarios file with another EM inversion method.")
    ap.add_argument("--config", required=True, help="scenarios file (e.g. ensemble_scenarios_real.json)")
    ap.add_argument("--em-method", required=True, choices=[e for e in EM_CHOICES if e != DEFAULT_EM])
    ap.add_argument("--out", default=None, help="output file (default: <config>_<em>.json)")
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    out_cfg = apply_to_config(cfg, args.em_method)
    out_path = args.out or derived_path(args.config, args.em_method)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2)
    n = sum(1 for k, v in out_cfg.items() if not k.startswith("_") and isinstance(v, dict)
            and isinstance(v.get("param_grid"), dict) and "method" in v["param_grid"])
    fd = em_fixed_depths(args.em_method)
    print(f"wrote {out_path}: {n} scenario(s) with method = {method_name(args.em_method)}"
          + (f", fixed_depths = {fd}" if fd is not None else "")
          + f", cache_tag / output_dir suffixed '_{args.em_method}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
