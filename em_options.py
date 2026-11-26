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

# tag (used in names / folders / --em-method) -> EMagPy method string; None = keep the scenario's own
EM_METHODS: dict[str, str | None] = {"base": None, "lbfgsb": "L-BFGS-B"}
EM_CHOICES = tuple(EM_METHODS)
DEFAULT_EM = "base"

_PATH_KEYS = ("cache_tag", "output_dir")


def check_em(em: str) -> str:
    if em not in EM_METHODS:
        raise ValueError(f"unknown EM method {em!r}; choose from {EM_CHOICES}")
    return em


def method_name(em: str) -> str | None:
    """EMagPy method string of ``em`` (None for 'base')."""
    return EM_METHODS[check_em(em)]


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
    if isinstance(pg, dict) and "method" in pg:
        pg["method"] = [method]
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
             "currently ROPE | lbfgsb = L-BFGS-B (fits the data; see em_options.py)")
    if multiple:
        ap.add_argument("--em-method", nargs="+", choices=EM_CHOICES, default=None, help=help_)
    else:
        ap.add_argument("--em-method", choices=EM_CHOICES, default=DEFAULT_EM, help=help_)


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
    print(f"wrote {out_path}: {n} scenario(s) with method = {method_name(args.em_method)}, "
          f"cache_tag / output_dir suffixed '_{args.em_method}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
