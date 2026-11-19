"""clean_run_log.py: drop run-log rows whose run folder was deleted."""
import os

import clean_run_log
from run_log import RunLog


def _setup(tmp_path):
    log = RunLog(str(tmp_path / "run_log.csv"))
    for seq in (1, 2, 3):
        d = tmp_path / "out" / f"{seq:04d}_s"
        d.mkdir(parents=True)
        log.append({"run_id": f"r{seq}", "scenario": "s", "output_dir": str(d)})
    log.append({"run_id": "nodir", "scenario": "s", "output_dir": ""})
    return log, tmp_path


def test_dry_run_changes_nothing(tmp_path):
    log, root = _setup(tmp_path)
    os.rmdir(root / "out" / "0002_s")
    n = clean_run_log.clean(log.path, str(root), set(), None, apply=False, backup=True)
    assert n == 1 and len(log.rows()) == 4


def test_apply_removes_orphans_keeps_empty_output_dir_and_backs_up(tmp_path):
    log, root = _setup(tmp_path)
    os.rmdir(root / "out" / "0002_s")
    clean_run_log.clean(log.path, str(root), set(), None, apply=True, backup=True)
    assert [r["run_id"] for r in log.rows()] == ["r1", "r3", "nodir"]
    assert os.path.isfile(log.path + ".bak")


def test_seq_option_drops_even_if_folder_exists(tmp_path):
    log, root = _setup(tmp_path)
    clean_run_log.clean(log.path, str(root), {3}, None, apply=True, backup=False)
    assert [r["run_id"] for r in log.rows()] == ["r1", "r2", "nodir"]
