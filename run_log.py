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

Used by run_pipeline.py (writer), compare_scenarios.py, hyp1_layers_ws_freedom.py and
the study scripts (readers).
"""
from __future__ import annotations

from EA_EM_config import OUTPUT_ROOT  # central output root (EA_EM_config.py)
import csv
import json
import os
import re
import subprocess

DEFAULT_RUN_LOG = OUTPUT_ROOT + "/run_log.csv"

RUN_LOG_COLUMNS = [
    "run_id", "timestamp", "scenario", "mode", "data_source", "status",
    "output_dir", "report_path", "final_rmse", "cv_mean", "cv_max",
    "ensemble_size", "runtime_s", "quality_level", "error",
    # added 2026-11: which study the run belongs to + the RMSE in context
    # (see rmse_context.py: % vol, RMSE / std of the measured SWC, rating)
    "study_type", "rmse_pct", "swc_obs_std", "rsr", "rmse_rating",
]

# (regex on the scenario name, study type), first match wins. Keep in sync with the
# scenario families: hyp1/2/3 scripts, run_1d_surface_vs_full.sh, study_swc_design.py,
# generate_synthetic_real_mirror.py, bench_ws_global_vs_localized.py.
STUDY_TYPE_RULES = [
    (r"^swcstudy_", "swc_design_study"),
    (r"^ws_bench", "ws_global_vs_localized"),
    (r"^real_study_", "hyp1_layers_ws_freedom"),
    (r"^real_tv_", "hyp2_train_val_split"),
    (r"^real_bw_sweep", "hyp3_localized_bandwidth"),
    (r"^mfc2_real_synthetic_mirror", "synthetic_mirror"),
    (r"_pin0$", "pinned_horizon"),
    (r"_tv_probe$", "train_val_synthetic"),
    (r"^1d_(surface|full).*hetero", "surface_vs_full_hetero"),
    (r"^1d_(surface|full)", "surface_vs_full"),
    (r"^real_(single|multi|L\d)", "real_scenarios"),
    (r"^2d", "synthetic_2d"),
]


def infer_study_type(scenario: str | None) -> str:
    """Study family of a scenario name (``"other"`` when no rule matches)."""
    name = str(scenario or "")
    for pattern, study in STUDY_TYPE_RULES:
        if re.search(pattern, name):
            return study
    return "other"

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
            self.ensure_schema()      # old log without the new columns -> upgrade first
            row = dict(row)
            if not row.get("study_type"):
                row["study_type"] = infer_study_type(row.get("scenario"))
            if not row.get("rmse_pct") and row.get("final_rmse"):
                from rmse_context import log_fields_from_rmse
                row.update({k: v for k, v in log_fields_from_rmse(row["final_rmse"]).items()
                            if not row.get(k)})
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

    def ensure_schema(self, backup: bool = True) -> bool:
        """Upgrade an existing log to ``RUN_LOG_COLUMNS``; True when it was rewritten.

        Rows keep their values; the new columns are filled from what the old row
        allows: ``study_type`` from the scenario name (``infer_study_type``) and
        ``rmse_pct`` from ``final_rmse``. ``swc_obs_std`` / ``rsr`` / ``rmse_rating``
        need the observations and stay empty for old runs. The old file is kept as
        ``<log>.bak`` (once). Never raises.
        """
        try:
            if not os.path.isfile(self.path) or os.path.getsize(self.path) == 0:
                return False
            with open(self.path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                header = list(reader.fieldnames or [])
                old_rows = list(reader)
            if header == RUN_LOG_COLUMNS:
                return False
            if backup and not os.path.isfile(self.path + ".bak"):
                import shutil
                shutil.copy2(self.path, self.path + ".bak")
            from rmse_context import log_fields_from_rmse
            for r in old_rows:
                if not r.get("study_type"):
                    r["study_type"] = infer_study_type(r.get("scenario"))
                if not r.get("rmse_pct"):
                    r["rmse_pct"] = log_fields_from_rmse(r.get("final_rmse"))["rmse_pct"]
            tmp = self.path + ".tmp"
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=RUN_LOG_COLUMNS, extrasaction="ignore")
                w.writeheader()
                for r in old_rows:
                    w.writerow({k: r.get(k, "") or "" for k in RUN_LOG_COLUMNS})
            os.replace(tmp, self.path)
            print(f"[RUN LOG] Upgraded {self.path} to {len(RUN_LOG_COLUMNS)} columns "
                  f"({len(old_rows)} rows; backup {self.path}.bak)")
            return True
        except (OSError, csv.Error) as e:
            print(f"[RUN LOG] WARNING: could not upgrade run log {self.path!r}: {e}")
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
    up = sub.add_parser("upgrade", help="add the study_type / rmse_* columns to an existing log")
    up.add_argument("--log", default=DEFAULT_RUN_LOG)
    lt = sub.add_parser("latest", help="field of the most recent run of a scenario")
    lt.add_argument("scenario")
    lt.add_argument("--log", default=DEFAULT_RUN_LOG)
    lt.add_argument("--field", default="run_id", choices=RUN_LOG_COLUMNS)
    lt.add_argument("--status", default=None)
    a = ap.parse_args()
    if a.cmd == "upgrade":
        print("upgraded" if RunLog(a.log).ensure_schema() else "already up to date (or no log)")
        raise SystemExit(0)
    row = RunLog(a.log).latest(scenario=a.scenario, status=a.status)
    print(row.get(a.field, "") if row else "")
