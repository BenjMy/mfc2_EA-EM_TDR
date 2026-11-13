"""run_log.py (RunLog, run_manifest.json) and compare_scenarios reading the manifest."""
import json
import math
import os

import pytest

import compare_scenarios as cs
import run_log
from run_log import RunLog


def _row(run_id, scenario="s", seq=1, status="SUCCESS", rmse="0.03"):
    return {"run_id": run_id, "scenario": scenario, "status": status,
            "output_dir": f"mfc2_outputs/x/{seq:04d}_{scenario}", "final_rmse": rmse}


# ---------------------------------------------------------------- RunLog ----

def test_append_writes_header_once_and_rows_round_trip(tmp_path):
    log = RunLog(str(tmp_path / "sub" / "run_log.csv"))
    assert log.rows() == []
    assert log.append(_row("a")) and log.append(_row("b", seq=2))
    text = open(log.path, encoding="utf-8").read()
    assert text.count("run_id,timestamp") == 1
    rows = log.rows()
    assert [r["run_id"] for r in rows] == ["a", "b"]
    assert list(rows[0]) == run_log.RUN_LOG_COLUMNS
    assert rows[0]["timestamp"] == ""  # missing keys are written empty


def test_lookups(tmp_path):
    log = RunLog(str(tmp_path / "log.csv"))
    log.append(_row("a", "s1", 1))
    log.append(_row("b", "s2", 1, status="FAILED"))
    log.append(_row("c", "s1", 2))
    assert log.by_id("b")["scenario"] == "s2" and log.by_id("zzz") is None
    assert [r["run_id"] for r in log.by_seq(1)] == ["a", "b"]
    assert [r["run_id"] for r in log.by_seq(1, scenario="s1")] == ["a"]
    assert log.latest()["run_id"] == "c"
    assert log.latest(scenario="s2")["run_id"] == "b"
    assert log.latest(scenario="s1", status="FAILED") is None
    assert RunLog.seq_of({"output_dir": "x/0007_foo"}) == 7
    assert RunLog.seq_of({"output_dir": "x/foo"}) is None


def test_append_never_raises(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert RunLog(str(blocker / "log.csv")).append(_row("a")) is False
    assert "WARNING" in capsys.readouterr().out


def test_run_pipeline_keeps_old_names(tmp_path):
    import run_pipeline
    assert run_pipeline.DEFAULT_RUN_LOG == run_log.DEFAULT_RUN_LOG
    assert run_pipeline.RUN_LOG_COLUMNS == run_log.RUN_LOG_COLUMNS
    run_pipeline.append_run_log(str(tmp_path / "l.csv"), _row("z"))
    assert RunLog(str(tmp_path / "l.csv")).by_id("z")


# -------------------------------------------------------------- manifest ----

def test_manifest_round_trip(tmp_path):
    p = run_log.write_run_manifest(str(tmp_path), {"run_id": "r", "x": {1: 2}})
    assert os.path.basename(p) == "run_manifest.json"
    man = run_log.read_run_manifest(str(tmp_path))
    assert man["schema_version"] == 1 and man["run_id"] == "r"
    assert run_log.read_run_manifest(p) == man
    assert run_log.read_run_manifest(str(tmp_path / "nope")) is None


def test_git_commit_is_none_or_sha(tmp_path):
    assert run_log.git_commit(str(tmp_path)) is None  # not a repo
    sha = run_log.git_commit()
    assert sha is None or len(sha.split("+")[0]) == 40


# ------------------------------- reporter -> manifest -> compare_scenarios ----

def _make_reporter(out_dir, cfg=None):
    import run_pipeline
    return run_pipeline.PipelineReporter(
        output_dir=str(out_dir), scenario_name="sc", mode="1d", cfg=cfg,
        data_source="synthetic", train_val_split="off")


def _fill(rep):
    rep.set_manifest_extra(stage="full", seeds={"inversion_seed": 1234})
    rep.step("Step 1", "inversion")
    rep.end_step("Step 1")
    rep.step("Step 2")
    rep.end_step("Step 2")
    rep.metric("Final RMSE (Step 5)", "0.031212 m³/m³")
    rep.metric("CV map mean (relative uncertainty)", "0.3360")
    rep.metric("CV map max", "1.9646")
    rep.metric("Ensemble size (Step 1)", "96 realisations")
    rep.metric("Optimal stack depth (K)", "3  (out of 10 candidates)")
    rep.metric("Forward ECa RMSE (optimal, t=0)", "1.25 mS/m")
    rep.metric("Quality gate level", "OK")


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    return a == b


@pytest.mark.parametrize("level", ["OK", "WARN"])
def test_manifest_record_equals_markdown_record(tmp_path, level):
    from scenario_loader import load_scenario
    cfg = load_scenario("1d_surface_tdr")  # a real ScenarioConfig must serialise
    rep = _make_reporter(tmp_path, cfg)
    _fill(rep)
    rep.finish(exit_ok=True, quality_level=level,
               quality_reasons=["CV too high"] if level == "WARN" else None)

    man = run_log.read_run_manifest(str(tmp_path))
    assert man["status"] == ("WARN" if level == "WARN" else "SUCCESS")
    assert man["run_id"] == rep.run_id and man["report"] == os.path.basename(rep.report_path)
    assert man["config"]["name"] == "1d_surface_tdr"
    assert man["seeds"]["inversion_seed"] == 1234
    assert man["metrics"]["CV map max"] == "1.9646"
    assert set(man["step_timing"]) == {"Step 1", "Step 2"}

    from_man = cs.parse_report(rep.report_path)
    from_md = cs.parse_report(rep.report_path, use_manifest=False)
    for rec in (from_man, from_md):
        rec.pop("_path"), rec.pop("result_line")
    assert _same(from_man, from_md)
    assert from_man["result_ok"] is (level == "OK")
    assert from_man["final_rmse"] == pytest.approx(0.031212)


def test_failure_writes_manifest(tmp_path):
    rep = _make_reporter(tmp_path)
    rep.metric("Ensemble size (Step 1)", "12")
    rep.failure(RuntimeError("boom"))
    man = run_log.read_run_manifest(str(tmp_path))
    assert man["status"] == "FAILED" and man["error"] == "boom"
    rec = cs.parse_report(rep.report_path)
    assert rec["result_ok"] is False and rec["ensemble_size"] == 12.0


def test_parse_report_accepts_manifest_path_and_run_folder(tmp_path):
    rep = _make_reporter(tmp_path)
    _fill(rep)
    rep.finish()
    a = cs.parse_report(str(tmp_path / "run_manifest.json"))
    b = cs.parse_report(str(tmp_path))
    assert a["run_id"] == b["run_id"] == rep.run_id


def test_old_run_without_manifest_falls_back_to_markdown(tmp_path):
    rep = _make_reporter(tmp_path)
    _fill(rep)
    rep.finish()
    os.remove(tmp_path / "run_manifest.json")
    rec = cs.parse_report(rep.report_path)
    assert rec["final_rmse"] == pytest.approx(0.031212) and rec["run_id"] == rep.run_id


def test_manifest_of_another_report_is_ignored(tmp_path):
    rep = _make_reporter(tmp_path)
    _fill(rep)
    rep.finish()
    other = tmp_path / "pipeline_report_other.md"
    other.write_text(open(rep.report_path, encoding="utf-8").read().replace("0.031212", "0.5"),
                     encoding="utf-8")
    assert cs.parse_report(str(other))["final_rmse"] == pytest.approx(0.5)


def test_cli_latest_matches_old_awk_semantics(tmp_path):
    """`run_log.py latest` replaces the awk in run_1d_surface_vs_full.sh."""
    import subprocess
    import sys
    log = RunLog(str(tmp_path / "log.csv"))
    log.append({**_row("old", "1d_full_free1", 1), "error": "a, comma"})
    log.append(_row("new", "1d_full_free1", 2, status="FAILED"))
    log.append(_row("other", "1d_surface_free1", 1))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def run(*args):
        return subprocess.run([sys.executable, os.path.join(root, "run_log.py"), "latest", *args,
                               "--log", log.path], capture_output=True, text=True,
                              check=True).stdout.strip()

    assert run("1d_full_free1") == "new"
    assert run("1d_full_free1", "--status", "SUCCESS") == "old"
    assert run("1d_full_free1", "--field", "status") == "FAILED"
    assert run("nope") == ""
