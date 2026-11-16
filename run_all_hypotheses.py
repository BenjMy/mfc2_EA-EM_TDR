#!/usr/bin/env python3
"""
run_all_hypotheses.py -- ONE entry point for the real-data hypothesis tests.

    1  hyp1_layers_ws_freedom      How many layers can the real TDR sensors support, and
                                   how many Waxman-Smits parameters can be freed?
    2  hyp2_train_val_split        Is the calibration still good on held-out data?
    3  hyp3_localized_bandwidth    Does a spatially localized WS calibration help, and
                                   at what bandwidth?

Each hypothesis is its own script (run it alone when you only need that one, and use
its own options such as --designs or --bandwidths). This script just calls them in turn
with the options they share, and prints one status line per hypothesis.

Usage
-----
    python run_all_hypotheses.py --list                  # the hypotheses, nothing run
    python run_all_hypotheses.py --fast --dry-run        # the plan of all of them
    python run_all_hypotheses.py --fast                  # smoke test of all of them
    python run_all_hypotheses.py --only 1 3              # a subset
    python run_all_hypotheses.py --base real_L3_base --only 2 3    # on 3 layers
    python run_all_hypotheses.py --fast -- --em-footprint-diagnostic   # after '--':
                                                         # flags for run_one_scenario.sh

Shared options (--config --base --mode --stage --fast --dry-run --keep-going) are
forwarded to every selected script. Hypothesis 1 builds its own layer designs, so
--base only changes the data/forward-model settings it inherits there.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

import hyp_common as hc

HYPOTHESES = {
    "1": "hyp1_layers_ws_freedom",
    "2": "hyp2_train_val_split",
    "3": "hyp3_localized_bandwidth",
}


def forwarded_args(args: argparse.Namespace, module, out_root: str) -> list[str]:
    """argv for one hypothesis script, built from the shared options."""
    out = ["--config", args.config, "--base", args.base, "--mode", args.mode,
           "--stage", args.stage, "--out-dir", os.path.join(out_root, os.path.basename(module.OUT_DIR))]
    for flag in ("fast", "dry_run", "keep_going"):
        if getattr(args, flag):
            out.append("--" + flag.replace("_", "-"))
    return out


def main(argv: list[str] | None = None) -> int:
    argv, extra = hc.split_extra(list(sys.argv[1:] if argv is None else argv))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list the hypotheses and exit")
    ap.add_argument("--only", nargs="+", choices=list(HYPOTHESES), metavar="N",
                    help=f"hypotheses to run (default: all of {list(HYPOTHESES)})")
    ap.add_argument("--skip", nargs="+", choices=list(HYPOTHESES), default=[], metavar="N")
    ap.add_argument("--out-root", default=hc.OUT_ROOT,
                    help="parent folder of the per-hypothesis output folders")
    hc.add_common_args(ap, default_out_dir=hc.OUT_ROOT)
    ap.set_defaults(out_dir=None)
    args = ap.parse_args(argv)
    if args.out_dir:
        ap.error("--out-dir is per hypothesis; use --out-root here")

    chosen = [n for n in (args.only or list(HYPOTHESES)) if n not in args.skip]
    modules = {n: importlib.import_module(HYPOTHESES[n]) for n in chosen}

    if args.list:
        for n, name in HYPOTHESES.items():
            print(f"  {n}  {name:28s} {importlib.import_module(name).QUESTION}")
        return 0

    status: dict[str, str] = {}
    for n in chosen:
        name = HYPOTHESES[n]
        print(f"\n{'=' * 78}\n HYPOTHESIS {n}: {name}\n {modules[n].QUESTION}\n{'=' * 78}")
        argv_n = forwarded_args(args, modules[n], args.out_root) + (["--", *extra] if extra else [])
        try:
            rc = modules[n].main(argv_n)
            status[n] = "ok" if rc == 0 else f"FAILED (exit {rc})"
        except SystemExit as exc:                       # argparse error inside a script
            rc = int(exc.code or 0)
            status[n] = "ok" if rc == 0 else f"FAILED (exit {rc})"
        except Exception as exc:                        # noqa: BLE001 - report, maybe continue
            rc = 1
            status[n] = f"ERROR {type(exc).__name__}: {exc}"
        if rc != 0 and not args.keep_going:
            print(f"!!! hypothesis {n} stopped the run (use --keep-going to continue)")
            break

    print(f"\n{'=' * 78}\n SUMMARY{'  (dry run: nothing was executed)' if args.dry_run else ''}\n{'=' * 78}")
    for n in chosen:
        print(f"  {n}  {HYPOTHESES[n]:28s} {status.get(n, 'not run')}")
    print(f"  results: {args.out_root}/<hypothesis>/hyp<N>_summary.csv")
    return 0 if all(v == "ok" for v in status.values()) and len(status) == len(chosen) else 1


if __name__ == "__main__":
    sys.exit(main())
