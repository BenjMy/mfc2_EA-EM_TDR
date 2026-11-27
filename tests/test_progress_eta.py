"""Tests of progress_eta.py (readable progress / ETA of Step 1). No emagpy, fake clocks, threads instead of processes."""
import concurrent.futures as cf
import datetime
import io
import os
import sys
import threading
import time

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]       # works next to the module or in a tests/ subfolder

import progress_eta as pe  # noqa: E402


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


FIXED_WALL = lambda: datetime.datetime(2026, 11, 27, 12, 0, 0)    # noqa: E731


# ----------------------------------------------------------------------------- formatting
@pytest.mark.parametrize("sec,txt", [(0, "0 s"), (45, "45 s"), (59.6, "1 min 00 s"), (425, "7 min 05 s"),
                                     (8040, "2 h 14 min"), (None, "?"), (float("nan"), "?"), (float("inf"), "?")])
def test_format_duration(sec, txt):
    assert pe.format_duration(sec) == txt


def test_format_count():
    assert pe.format_count(950) == "950" and pe.format_count(1500) == "1.5k"
    assert pe.format_count(53200) == "53k" and pe.format_count(2.5e6) == "2.5M"


def test_count_csv_points(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("x,y\n1,2\n3,4\n5,6\n")
    assert pe.count_csv_points(str(p)) == 3
    assert pe.count_csv_points(str(tmp_path / "missing.csv")) == 0


# ----------------------------------------------------------------------------- estimate
def test_estimate_single_worker_is_the_sum_plus_startup():
    assert pe.estimate_seconds([10, 20], [1.0, 0.5], 1, startup_s=3.0) == pytest.approx(3 + 10 + 10)


def test_estimate_pool_makespan_and_njobs():
    pts, cost = [100, 100, 100, 100], [1.0] * 4
    assert pe.estimate_seconds(pts, cost, 2, startup_s=0) == pytest.approx(200)      # 2 tasks per worker
    assert pe.estimate_seconds(pts, cost, 4, startup_s=0) == pytest.approx(100)
    assert pe.estimate_seconds(pts, cost, 2, njobs=4, startup_s=0) == pytest.approx(50)   # cores per task divide the cost
    assert pe.estimate_seconds([], [], 4, startup_s=2.0) == 2.0


def _task(idx, path, method="L-BFGS-B", fm="CS"):
    return (idx, path, [0.3], [20, 20], [True], {"method": method, "forwardModel": fm, "njobs": 4})


def test_benchmark_costs_times_a_truncated_csv_once_per_kind(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("x,y\n" + "\n".join(f"{i},{i}" for i in range(50)) + "\n")
    seen = []

    def worker(task):
        with open(task[1]) as f:
            seen.append((task[5]["method"], task[5]["forwardModel"], sum(1 for _ in f) - 1, "njobs" in task[5]))

    tasks = [_task(0, str(p), fm="CS"), _task(1, str(p), fm="CS"), _task(2, str(p), fm="FSlin")]
    clock = iter(x * 0.1 for x in range(100))                    # each timed call "takes" 0.1 s
    costs = pe.benchmark_costs(worker, tasks, n_points=3, clock=lambda: next(clock))
    assert set(costs) == {("L-BFGS-B", "CS"), ("L-BFGS-B", "FSlin")}
    assert all(c == pytest.approx(0.1 / 3) for c in costs.values())
    assert sorted(n for *_, n, _ in seen) == [1, 1, 3, 3]        # warm-up of 1 row + timed 3 rows, per kind
    assert not any(has_njobs for *_, has_njobs in seen)          # njobs is removed: cost is per core


def test_benchmark_failure_leaves_the_kind_out(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("x\n1\n2\n")

    def worker(task):
        raise RuntimeError("boom")

    assert pe.benchmark_costs(worker, [_task(0, str(p))]) == {}
    assert pe.describe_estimate({}, [_task(0, str(p))], [2], 1) is None


def test_describe_estimate_mentions_time_workers_and_solvers(tmp_path):
    tasks = [_task(0, "a"), _task(1, "a", fm="FSlin")]
    txt = pe.describe_estimate({("L-BFGS-B", "CS"): 0.0066, ("L-BFGS-B", "FSlin"): 0.106}, tasks, [1000, 1000], 2)
    assert "[ETA]" in txt and "2 worker(s)" in txt and "L-BFGS-B/FSlin 106 ms" in txt and "L-BFGS-B/CS 7 ms" in txt
    assert "about" in txt and "finishing around" in txt


# ----------------------------------------------------------------------------- output filter
def test_filter_turns_emagpy_progress_into_a_counter_and_drops_noise():
    real, shared = io.StringIO(), [0.0] * 3
    f = pe.EmagpyOutputFilter(real, 1, shared)
    f.write("456/1926 inverted")
    f.write('Converting quadrature columns to LIN ECa. You can use "FSlin" or "Q" as forward model.\n')
    f.write("\r900/1926 inverted\r")
    assert shared == [0.0, 900.0, 0.0]
    assert real.getvalue() == ""                                  # nothing of it reaches the terminal
    f.write("[WARN] something real\n")
    assert real.getvalue() == "[WARN] something real\n"
    assert f.write("abc") == 3                                    # write() reports the length it was given


def test_run_filtered_restores_stdout_and_reports_progress():
    shared = [0.0, 0.0]
    pe.init_worker(shared)
    try:
        before = sys.stdout

        def fn(task):
            print("12/30 inverted", end="")
            print("kept line")
            return task[0]

        buf = io.StringIO()
        sys.stdout = buf
        try:
            out = pe.run_filtered(fn, (1,))
            assert sys.stdout is buf                              # restored
        finally:
            sys.stdout = before
        assert out == 1 and shared[1] == 12.0 and buf.getvalue() == "kept line\n"
    finally:
        pe.init_worker(None)


# ----------------------------------------------------------------------------- tracker
def make(points, workers=2, shared=None, tty=False, clock=None, expected_rate=None):
    clock = clock or Clock()
    buf = io.StringIO()
    tr = pe.Step1Progress("Step 1 | map", points, workers, shared=shared, stream=buf, tty=tty, clock=clock,
                          wall=FIXED_WALL, expected_rate=expected_rate)
    return tr, buf, clock


def test_progress_counts_points_of_done_and_running_tasks():
    shared = [0.0, 40.0, 0.0, 0.0]
    tr, _, _ = make([100, 100, 200, 200], shared=shared)
    tr.mark_done(0)
    assert tr.total_points == 600 and tr.done_points() == 140     # task 0 complete + 40 points into task 1
    shared[1] = 999.0                                             # a counter above the task size is capped
    assert tr.done_points() == 200


def test_eta_is_none_early_then_follows_points_not_tasks():
    tr, _, clock = make([100, 100, 800])
    clock.advance(2)
    tr.mark_done(0)
    assert tr.eta_seconds() is None                               # < 5 s: no ETA yet
    clock.advance(8)                                              # 10 s, 100 of 1000 points -> 10 pts/s
    assert tr.rate() == pytest.approx(10.0)
    assert tr.eta_seconds() == pytest.approx(90.0)               # 900 points left (a task-count ETA would say 20 s)
    tr.mark_done(1)
    clock.advance(10)                                             # 20 s, 200 points -> still 10 pts/s
    assert tr.eta_seconds() == pytest.approx(80.0)


def test_recent_rate_is_blended_in_after_30_seconds():
    tr, _, clock = make([1000, 1000])
    clock.advance(10)
    tr.mark_done(0)                                               # 1000 points in 10 s = 100 pts/s
    tr.rate()
    clock.advance(100)                                            # nothing for 100 s: recent rate = 0
    r = tr.rate()
    assert r < 1000 / 110 and r > 0                               # below the global rate: a stall shows in the ETA


def test_render_contains_percent_tasks_points_eta_and_clock_time():
    tr, _, clock = make([100, 100, 800], workers=3)
    clock.advance(10)
    tr.mark_done(0)
    line = tr.render()
    for token in ("[Step 1 | map]", "10.0%", "tasks 1/3", "points 100/1.0k", "elapsed 10 s", "ETA 1 min 30 s",
                  "(~12:01)", "3 worker(s)", "10 pts/s"):
        assert token in line, (token, line)
    assert "ETA estimating" in make([10])[0].render()


def test_log_mode_prints_one_line_per_interval_tty_mode_refreshes_in_place():
    tr, buf, clock = make([100, 100], tty=False)
    assert tr.tick() is True and tr.tick() is False               # throttled
    clock.advance(pe.LOG_INTERVAL_S + 1)
    assert tr.tick() is True
    assert buf.getvalue().count("\n") == 2 and "\r" not in buf.getvalue()
    tr, buf, clock = make([100, 100], tty=True)
    tr.tick()
    clock.advance(pe.TTY_INTERVAL_S + 0.1)
    tr.tick()
    assert buf.getvalue().count("\r") == 2 and "\n" not in buf.getvalue()
    summary = tr.close()
    assert "done: 0/2 tasks" in summary and buf.getvalue().endswith("\n")


# ----------------------------------------------------------------------------- runner
def test_run_ordered_progress_yields_every_result_and_tracks_completion():
    tasks = [(i, "f", [], [], [], {}) for i in range(6)]

    def fn(task):
        time.sleep(0.05 * (6 - task[0]))                         # the FIRST task is the slowest (map() would block on it)
        return task[0] * 10

    tr, buf, _ = make([10] * 6, workers=3, clock=time.time)
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        out = list(pe.run_ordered_progress(ex, fn, tasks, tr, poll_s=0.05))
    assert sorted(out) == [0, 10, 20, 30, 40, 50]
    assert out[0] != 0                                           # finished out of submission order
    assert len(tr.done) == 6 and "done: 6/6 tasks" in buf.getvalue()


def test_run_ordered_progress_cancels_pending_and_reraises():
    gate = threading.Event()
    tasks = [(i, "f", [], [], [], {}) for i in range(4)]

    def fn(task):
        if task[0] == 0:
            raise ValueError("worker crashed")
        gate.wait(5)
        return task[0]

    tr, buf, _ = make([1] * 4, workers=1, clock=time.time)
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        with pytest.raises(ValueError, match="worker crashed"):
            list(pe.run_ordered_progress(ex, fn, tasks, tr, poll_s=0.05))
        gate.set()
    assert "done:" in buf.getvalue()                             # the summary line is still written


# ----------------------------------------------------------------------------- thread limit and slow-run warning
def test_limit_worker_threads_sets_one_and_respects_the_user():
    env = {}
    set_, kept = pe.limit_worker_threads(24, env)
    assert set(set_) == set(pe.THREAD_VARS) and all(env[v] == "1" for v in pe.THREAD_VARS) and kept == []
    env = {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "1"}
    set_, kept = pe.limit_worker_threads(24, env)
    assert env["OMP_NUM_THREADS"] == "8" and kept == ["OMP_NUM_THREADS=8"]       # user's value is never overwritten
    assert "OMP_NUM_THREADS" not in set_ and "MKL_NUM_THREADS" not in set_
    assert pe.limit_worker_threads(1, {"OMP_NUM_THREADS": "8"})[1] == []         # one worker: threads are fine


def test_expected_rate_from_the_benchmark():
    tasks = [_task(0, "a"), _task(1, "a"), _task(2, "a", fm="FSlin")]
    costs = {("L-BFGS-B", "CS"): 0.01, ("L-BFGS-B", "FSlin"): 0.1}
    r = pe.expected_rate(costs, tasks, [100, 100, 100], 3)
    assert r == pytest.approx(300 / 10.0)                                         # slowest task: 100 pts x 0.1 s
    assert pe.expected_rate({}, tasks, [100, 100, 100], 3) is None


def test_slow_warning_fires_once_only_when_far_below_the_benchmark():
    tr, buf, clock = make([1000, 1000], expected_rate=100.0)
    clock.advance(30)
    tr.mark_done(0)                                                               # 33 pts/s but only 30 s: too early
    assert tr.slow_warning() is None
    clock.advance(60)                                                             # 90 s, 1000 pts = 11 pts/s < 25
    msg = tr.slow_warning()
    assert msg and "[WARN]" in msg and "OMP_NUM_THREADS=1" in msg and "9 x below" in msg
    assert tr.slow_warning() is None                                              # once
    tr2, _, c2 = make([1000, 1000], expected_rate=100.0)
    c2.advance(90)
    tr2.mark_done(0)
    tr2.shared = None
    tr2.done.add(1)                                                               # 2000 pts in 90 s = 22 pts/s...
    tr2.points = [1000, 100000]                                                   # ...but make it healthy: 101000 done
    assert tr2.slow_warning() is None
    assert make([10])[0].slow_warning() is None                                   # no expected rate: never warns


def test_tick_prints_the_warning_after_the_progress_line():
    tr, buf, clock = make([1000, 1000], tty=False, expected_rate=100.0)
    clock.advance(90)
    tr.mark_done(0)
    assert tr.tick(force=True) is True
    lines = buf.getvalue().splitlines()
    assert lines[0].startswith("[Step 1 | map]") and lines[1].startswith("[WARN]")
    clock.advance(40)
    tr.tick(force=True)
    assert buf.getvalue().count("[WARN]") == 1
