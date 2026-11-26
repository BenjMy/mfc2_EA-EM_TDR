#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ws_param_check.py  --  are the synthetic Waxman-Smits truth parameters sensible?
================================================================================

Companion of ``diagnose_ec_inversion.py`` (Patch 2b of swc_study_known_ws_plan.md).
Pure numpy, no EMagPy, runs in a second.  It only uses the formula of the
generator, ``data_synthetic.forward_waxman_smits``::

    EC_b = theta**m / a * (sigma_w + sigma_s)          [mS/m]

and answers four questions for one or several parameter sets:

  Q1  EC range   : EC_b over theta = 0.05-0.45 (the generator's clip) against the
                   1 mS/m floor (``max(1.0, ...)`` in the generator) and the
                   EM inversion bounds (``cond_ranges``).
  Q2  Formation factor at saturation F = a / phi**m (phi = 0.44), the number
                   usually reported for soils, and the exponent m.
  Q3  Identifiability: only ``C = a / (sigma_w + sigma_s)`` and ``m`` enter EC_b
                   (a, sigma_w, sigma_s are EXACTLY degenerate in this form);
                   two different sets with the same C give the same EC_b.
  Q4  Sensitivity: change of theta [m3/m3] at theta = 0.30 when ONE parameter is
                   wrong by +-10 % and the EC is exact (this is what a wrong
                   'known' value does in the swc study, ``--err``).

Usage
-----
    python ws_param_check.py                         # baseline + alternatives
    python ws_param_check.py --set 0.5 3 500 100     # one set (a m sigma_w sigma_s)
    python ws_param_check.py --out ws_param_check.csv
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

THETA = np.linspace(0.05, 0.45, 9)
PHI_SAT = 0.44                 # porosity used for the formation factor (sandy loam, see README refs)
EC_FLOOR = 1.0                 # data_synthetic: ec_bulk = max(1.0, ...)
EC_BOUNDS = (0.01, 300.0)      # cond_ranges of the study base scenario

# name -> (a, m, sigma_w, sigma_s)   [-, -, mS/m, mS/m]
SETS = {
    "baseline (study)": (0.5, 3.0, 500.0, 100.0),
    "sigma_w 100 (non-saline)": (0.5, 3.0, 100.0, 100.0),
    "m = 2": (0.5, 2.0, 500.0, 100.0),
    "F~4.6, m = 2, sigma_w 100": (1.0, 2.0, 100.0, 100.0),
    "same C as baseline": (1.0, 3.0, 1100.0, 100.0),
}


def ec_bulk(theta, a, m, sw, ss):
    """Same expression as data_synthetic.forward_waxman_smits (no clipping of EC)."""
    theta = np.clip(theta, 0.01, 1.0)
    return theta ** m / a * (sw + ss)


def theta_from_ec(ec, a, m, sw, ss):
    """Exact inverse: theta = (ec * a / (sw + ss)) ** (1/m)."""
    return (np.asarray(ec) * a / (sw + ss)) ** (1.0 / m)


def theta_at_ec(ec_target, a, m, sw, ss):
    return float(theta_from_ec(ec_target, a, m, sw, ss))


def formation_factor(a, m, phi=PHI_SAT):
    return a / phi ** m


def param_effect(a, m, sw, ss, theta0=0.30, rel=0.10):
    """theta error [m3/m3] if one parameter is off by +-rel and the EC is exact."""
    ec0 = ec_bulk(theta0, a, m, sw, ss)
    base = dict(a=a, m=m, sw=sw, ss=ss)
    rows = {}
    for name in base:
        for sign in (+1, -1):
            p = dict(base)
            p[name] = base[name] * (1 + sign * rel)
            rows[f"{name}{'+' if sign > 0 else '-'}{int(rel * 100)}%"] = float(
                theta_from_ec(ec0, p["a"], p["m"], p["sw"], p["ss"]) - theta0)
    return rows


def summarize(name, a, m, sw, ss) -> dict:
    ec = ec_bulk(THETA, a, m, sw, ss)
    row = dict(
        set=name, a=a, m=m, sigma_w=sw, sigma_s=ss,
        C=a / (sw + ss),
        surface_share=ss / (sw + ss),
        F_sat=formation_factor(a, m),
        ec_at_0p05=ec[0], ec_at_0p15=ec_bulk(0.15, a, m, sw, ss),
        ec_at_0p30=ec_bulk(0.30, a, m, sw, ss), ec_at_0p45=ec[-1],
        theta_at_floor=theta_at_ec(EC_FLOOR, a, m, sw, ss),
        theta_at_upper_bound=theta_at_ec(EC_BOUNDS[1], a, m, sw, ss),
    )
    row.update(param_effect(a, m, sw, ss))
    return row


def degeneracy_check(set_a, set_b) -> float:
    """Max |EC_a - EC_b| [mS/m] over theta for two sets (0 when C and m agree)."""
    return float(np.max(np.abs(ec_bulk(THETA, *set_a) - ec_bulk(THETA, *set_b))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", nargs=4, type=float, metavar=("a", "m", "sigma_w", "sigma_s"),
                    help="check one extra set")
    ap.add_argument("--out", default=None, help="write the table to this CSV")
    args = ap.parse_args(argv)

    sets = dict(SETS)
    if args.set:
        sets["custom"] = tuple(args.set)
    df = pd.DataFrame([summarize(n, *p) for n, p in sets.items()])
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.precision", 3)

    print("Q1/Q2  EC range, floor, formation factor")
    print(df[["set", "a", "m", "sigma_w", "sigma_s", "C", "surface_share", "F_sat",
              "ec_at_0p05", "ec_at_0p15", "ec_at_0p30", "ec_at_0p45",
              "theta_at_floor", "theta_at_upper_bound"]].to_string(index=False))
    print(f"\n  theta_at_floor: below this theta the generator raises EC to {EC_FLOOR} mS/m "
          f"(information lost); theta_at_upper_bound: above it EC exceeds {EC_BOUNDS[1]} mS/m.")

    names = list(sets)
    d = degeneracy_check(sets[names[0]], sets["same C as baseline"]) if "same C as baseline" in sets else np.nan
    print(f"\nQ3  degeneracy: max |EC(baseline) - EC(same C)| = {d:.2e} mS/m "
          "(different a, sigma_w, sigma_s; same C = a/(sigma_w+sigma_s) and m)")

    cols = [c for c in df.columns if c.endswith("%")]
    print("\nQ4  theta error [m3/m3] at theta = 0.30 if ONE parameter is off by +-10 % (EC exact)")
    print(df[["set"] + cols].to_string(index=False))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
