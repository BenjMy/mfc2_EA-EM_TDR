import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import study_status as ss
from run_log import RunLog


def _row(sc, status="SUCCESS", **kw):
    return dict(run_id=f"{sc}-{status}", scenario=sc, status=status, final_rmse="0.05 m³/m³", **kw)


def test_classify():
    rows = [_row("a"), _row("b", "FAILED"), _row("zzz")]
    rows[1]["scenario"] = "b"
    c = ss.classify(["a", "b", "c"], rows)
    assert c["done"] == ["a"] and c["failed"] == ["b"] and c["missing"] == ["c"] and c["extra"] == ["zzz"]


def test_failed_then_good_counts_as_done():
    c = ss.classify(["a"], [_row("a", "FAILED"), {**_row("a"), "run_id": "a2"}])
    assert c["done"] == ["a"]


def test_expected_matrix_contents():
    m = ss.expected_matrix()
    assert len(m["surface_vs_full"]["scenarios"]) == 8
    assert "1d_full_hetero_free4" in m["surface_vs_full_hetero"]["scenarios"]
    assert "./run_one_scenario.sh -s 1d_full_free3 --mode 1d" in m["surface_vs_full"]["command"]("1d_full_free3")
    if "hyp1_layers_ws_freedom" in m:
        assert "real_study_L2_free1" in m["hyp1_layers_ws_freedom"]["scenarios"]
        assert "--designs L2 --free-k 1" in m["hyp1_layers_ws_freedom"]["command"]("real_study_L2_free1")


def test_caveats():
    rows = [_row("a", rsr="", rmse_rating="no skill", output_dir="x"), _row("b", "WARN", rsr="0.4")]
    man = {rows[0]["run_id"]: {"git_commit": "abc+dirty", "ws_start_warnings": ["a on bound"], "config": {"em_subsample": "grid"}}}
    txt = " | ".join(ss.caveats(rows, man, "def"))
    for key in ("One run per scenario", "WARN", "no RSR", "no skill", "+dirty", "em_subsample", "ws_start_warnings", "current HEAD"):
        assert key in txt, key


def test_report_end_to_end(tmp_path):
    log = tmp_path / "l.csv"
    l = RunLog(str(log))
    l.append({"run_id": "r1", "scenario": "1d_surface_free1", "status": "SUCCESS", "final_rmse": "0.1 m³/m³"})
    l.append({"run_id": "r2", "scenario": "1d_full_free1", "status": "FAILED"})
    out = tmp_path / "S.md"
    assert ss.main(["--log-file", str(log), "--out", str(out), "--study-type", "surface_vs_full"]) == 0
    t = out.read_text()
    assert "| surface_vs_full | 8 | 1 | 1 | 6 | 0 |" in t
    assert "1d_surface_free2 --mode 1d" in t and "No expected matrix" not in t
