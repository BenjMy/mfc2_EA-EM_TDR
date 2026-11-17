#!/usr/bin/env python3
"""Level 4 of the TDR-pinned experiment: per-horizon RMSE / bias vs truth.

    python compare_pinned.py free=<run> pinTDR=<run> oracle=<run> full=<run> [--csv out.csv]

Each ``<run>`` is a run folder (``.../0001_<scenario>``) or a scenario name
(its latest run under the scenario's output_dir). Truth = the synthetic
arrays saved in the first run's ``output_model`` (realisation 0, as the
pipeline uses). Reports theta (m3/m3) and EC (mS/m), per horizon and overall.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_run(spec: str) -> str:
    if os.path.isdir(spec):
        return spec
    import scenario_loader as sl
    out = sl.load_scenario(spec).output_dir
    runs = sorted(glob.glob(os.path.join(HERE, out, "[0-9]*_*")))
    if not runs:
        raise FileNotFoundError(f"no run for scenario {spec!r} under {out}")
    return runs[-1]


def errors(calc: np.ndarray, truth: np.ndarray) -> list[tuple[float, float]]:
    """[(rmse, bias)] for each layer, then overall; arrays (T, P, L)."""
    d = calc - truth
    rows = [(float(np.sqrt(np.nanmean(d[..., h] ** 2))), float(np.nanmean(d[..., h])))
            for h in range(d.shape[-1])]
    rows.append((float(np.sqrt(np.nanmean(d ** 2))), float(np.nanmean(d))))
    return rows


def compare(runs: dict[str, str]) -> dict:
    """{label: {'theta': [...], 'ec': [...], 'K': int}} -- truth from the first run."""
    first = os.path.join(next(iter(runs.values())), "output_model")
    th_t = np.load(os.path.join(first, "synthetic_time_lapse_theta.npy"))[0]
    ec_t = np.load(os.path.join(first, "synthetic_time_lapse_ec.npy"))[0]
    out = {}
    for label, run in runs.items():
        d = os.path.join(run, "output_model")
        out[label] = {"theta": errors(np.load(os.path.join(d, "theta_calc.npy")), th_t),
                      "ec": errors(np.load(os.path.join(d, "ec_optimal.npy")), ec_t),
                      "K": int(np.load(os.path.join(d, "K.npy")))}
    return out


def format_table(res: dict) -> str:
    n = len(next(iter(res.values()))["theta"]) - 1
    names = [f"H{h}" for h in range(n)] + ["all"]
    lines = []
    for key, unit, fmt in (("theta", "m3/m3", "{:.3f} ({:+.3f})"), ("ec", "mS/m", "{:.1f} ({:+.1f})")):
        lines += [f"\n{key} RMSE (bias) vs truth [{unit}]",
                  "| run | K | " + " | ".join(names) + " |",
                  "|---|---|" + "---|" * len(names)]
        for lab, r in res.items():
            lines.append(f"| {lab} | {r['K']} | " + " | ".join(fmt.format(*e) for e in r[key]) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=<run folder or scenario>")
    ap.add_argument("--csv")
    a = ap.parse_args(argv)
    runs = {}
    for item in a.runs:
        lab, _, spec = item.partition("=")
        runs[lab] = resolve_run(spec)
    res = compare(runs)
    print(format_table(res))
    if a.csv:
        with open(a.csv, "w") as f:
            f.write("run,quantity,horizon,rmse,bias\n")
            for lab, r in res.items():
                for q in ("theta", "ec"):
                    for h, (rm, b) in enumerate(r[q]):
                        f.write(f"{lab},{q},{'all' if h == len(r[q]) - 1 else h},{rm:.6g},{b:.6g}\n")


if __name__ == "__main__":
    main()
