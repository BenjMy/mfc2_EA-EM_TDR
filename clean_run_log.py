"""Clean the run-log CSV: drop rows whose run folder no longer exists.

When you delete a run folder (``mfc2_outputs/<scenario>/<NNNN>_<scenario>/``) by
hand, its row stays in ``run_log.csv``. This script removes those orphan rows.

Usage (from the repo root, where ``output_dir`` paths are resolved):

    python3 clean_run_log.py                       # dry run: show what would be removed
    python3 clean_run_log.py --apply               # rewrite the log (backup: run_log.csv.bak)
    python3 clean_run_log.py --log mfc2_outputs/run_log_swc_study.csv --apply
    python3 clean_run_log.py --all-logs --apply    # every mfc2_outputs/run_log*.csv

Options:
    --seq N [N ...]   also drop rows of these run numbers (folder prefix ``<N>_``),
                      even if the folder still exists (e.g. you are about to delete it)
    --scenario NAME   restrict ``--seq`` to one scenario
    --root DIR        resolve relative ``output_dir`` against DIR (default: cwd)
    --no-backup       do not write ``<log>.bak``

Rows with an empty ``output_dir`` (e.g. a run that failed before creating its
folder) are kept: there is nothing to check them against.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import sys
import tempfile

from run_log import DEFAULT_RUN_LOG, RUN_LOG_COLUMNS, RunLog
from EA_EM_config import OUTPUT_ROOT


def _folder_exists(row: dict, root: str) -> bool:
    out = (row.get("output_dir") or "").strip()
    if not out:
        return True  # nothing to check -> keep
    path = out if os.path.isabs(out) else os.path.join(root, out)
    return os.path.isdir(path)


def clean(log_path: str, root: str, seqs: set[int], scenario: str | None,
          apply: bool, backup: bool) -> int:
    """Prune one log. Returns the number of rows removed (or that would be)."""
    if not os.path.isfile(log_path):
        print(f"[clean] {log_path}: not found, skipped")
        return 0

    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or RUN_LOG_COLUMNS
        rows = list(reader)

    keep, drop = [], []
    for r in rows:
        missing = not _folder_exists(r, root)
        by_seq = (RunLog.seq_of(r) in seqs
                  and (scenario is None or r.get("scenario") == scenario)) if seqs else False
        (drop if (missing or by_seq) else keep).append(r)

    print(f"[clean] {log_path}: {len(rows)} rows, {len(drop)} to remove, {len(keep)} kept")
    for r in drop:
        why = "folder missing" if not _folder_exists(r, root) else "--seq"
        print(f"    - {r.get('run_id', '?')}  {r.get('output_dir', '')}  ({why})")

    if not drop or not apply:
        if drop:
            print("[clean] dry run, nothing written (use --apply)")
        return len(drop)

    if backup:
        shutil.copy2(log_path, log_path + ".bak")
    # atomic rewrite: write to a temp file in the same dir, then replace
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(log_path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(keep)
        os.replace(tmp, log_path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print(f"[clean] {log_path} rewritten" + (f" (backup: {log_path}.bak)" if backup else ""))
    return len(drop)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Remove run-log rows whose run folder was deleted.")
    ap.add_argument("--log", default=DEFAULT_RUN_LOG, help=f"run-log CSV (default: {DEFAULT_RUN_LOG})")
    ap.add_argument("--all-logs", action="store_true",
                    help=f"clean every {OUTPUT_ROOT}/run_log*.csv")
    ap.add_argument("--seq", type=int, nargs="+", default=[], help="also drop these run numbers")
    ap.add_argument("--scenario", default=None, help="restrict --seq to this scenario")
    ap.add_argument("--root", default=os.getcwd(), help="base for relative output_dir paths")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--no-backup", action="store_true", help="skip the .bak copy")
    a = ap.parse_args(argv)

    logs = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "run_log*.csv"))) if a.all_logs else [a.log]
    total = 0
    for lp in logs:
        total += clean(lp, a.root, set(a.seq), a.scenario, a.apply, not a.no_backup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
