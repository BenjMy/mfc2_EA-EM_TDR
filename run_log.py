"""Run bookkeeping: the shared run-log CSV and the per-run ``run_manifest.json``.

RunLog(path)                      the shared CSV, one row per pipeline run
    .append(row)                  add a row (never raises: housekeeping must not stop a run)
    .rows()                       all rows, oldest first, as dicts of str
    .by_id(run_id)                the row with this run_id (or None)
    .by_seq(seq, scenario=None)   rows whose output folder is ``<seq>_...``
    .latest(scenario=None, status=None)   most recent matching row (or None)

write_run_manifest(output_dir, manifest)   -> path | None   (never raises)
read_run_manifest(path_or_dir)             -> dict | None

``run_manifest.json`` (schema_version 1) sits in every run's output folder and
holds what the markdown report only shows as text: run_id, scenario, mode, data
source, status, quality gate, runtime, git commit, seeds, the resolved scenario
config, the headline ``metrics`` ({name: str}) and ``step_timing`` ({step: s}).
``compare_scenarios`` reads it instead of parsing ``pipeline_report_*.md``.

Used by run_pipeline.py (writer), compare_scenarios.py, run_real_study.py and
the study scripts (readers).
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess

DEFAULT_RUN_LOG = "mfc2_outputs/run_log.csv"

RUN_LOG_COLUMNS = [
    "run_id", "timestamp", "scenario", "mode", "data_source", "status",
    "output_dir", "report_path", "final_rmse", "cv_mean", "cv_max",
    "ensemble_size", "runtime_s", "quality_level", "error",
]

MANIFEST_NAME = "run_manifest.json"
MANIFEST_SCHEMA_VERSION = 1


class RunLog:
    """The shared run-log CSV (one row per run, all scenarios)."""

    def __init__(self, path: str = DEFAULT_RUN_LOG) -> None:
        self.path = path

    def append(self, row: dict) -> bool:
        """Append one row (header written on first use). Returns False on failure."""
        try:
            log_dir = os.path.dirname(self.path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            write_header = (not os.path.isfile(self.path)) or os.path.getsize(self.path) == 0
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=RUN_LOG_COLUMNS)
                if write_header:
                    writer.writeheader()
                writer.writerow({k: row.get(k, "") for k in RUN_LOG_COLUMNS})
            print(f"[RUN LOG] Appended run '{row.get('run_id')}' ({row.get('status')}) → {self.path}")
            return True
        except OSError as e:
            print(f"[RUN LOG] WARNING: could not write run log at {self.path!r}: {e}")
            return False

    def rows(self) -> list[dict]:
        """All rows, oldest first (empty list when the file does not exist)."""
        if not os.path.isfile(self.path):
            return []
        with open(self.path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def by_id(self, run_id: str) -> dict | None:
        for r in reversed(self.rows()):
            if r.get("run_id") == run_id:
                return r
        return None

    @staticmethod
    def seq_of(row: dict) -> int | None:
        """Leading number of the run's output folder name (``0007_...`` -> 7)."""
        base = os.path.basename(os.path.normpath(str(row.get("output_dir", "") or "")))
        m = re.match(r"^(\d+)", base)
        return int(m.group(1)) if m else None

    def by_seq(self, seq: int, scenario: str | None = None) -> list[dict]:
        out = [r for r in self.rows() if self.seq_of(r) == int(seq)]
        if scenario is not None:
            out = [r for r in out if r.get("scenario") == scenario]
        return out

    def latest(self, scenario: str | None = None, status: str | None = None) -> dict | None:
        for r in reversed(self.rows()):
            if scenario is not None and r.get("scenario") != scenario:
                continue
            if status is not None and r.get("status") != status:
                continue
            return r
        return None


def git_commit(repo_dir: str | None = None) -> str | None:
    """HEAD commit of the code (with ``+dirty`` when the tree has changes), or None."""
    repo_dir = repo_dir or os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo_dir, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def write_run_manifest(output_dir: str, manifest: dict) -> str | None:
    """Write ``<output_dir>/run_manifest.json``; returns the path (None on failure)."""
    try:
        path = os.path.join(output_dir, MANIFEST_NAME)
        data = {"schema_version": MANIFEST_SCHEMA_VERSION, **manifest}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return path
    except (OSError, TypeError, ValueError) as e:
        print(f"[RUN LOG] WARNING: could not write {MANIFEST_NAME} in {output_dir!r}: {e}")
        return None


def read_run_manifest(path_or_dir: str) -> dict | None:
    """Read a manifest given its path or the run folder; None if absent/unreadable."""
    path = (os.path.join(path_or_dir, MANIFEST_NAME)
            if os.path.isdir(path_or_dir) else path_or_dir)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


if __name__ == "__main__":
    # Shell helper (replaces awk on CSV column positions):
    #   python3 run_log.py latest <scenario> [--log FILE] [--field run_id] [--status S]
    # prints the field of the most recent matching run (empty line when none).
    import argparse

    ap = argparse.ArgumentParser(description="Query the shared run-log CSV.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lt = sub.add_parser("latest", help="field of the most recent run of a scenario")
    lt.add_argument("scenario")
    lt.add_argument("--log", default=DEFAULT_RUN_LOG)
    lt.add_argument("--field", default="run_id", choices=RUN_LOG_COLUMNS)
    lt.add_argument("--status", default=None)
    a = ap.parse_args()
    row = RunLog(a.log).latest(scenario=a.scenario, status=a.status)
    print(row.get(a.field, "") if row else "")
