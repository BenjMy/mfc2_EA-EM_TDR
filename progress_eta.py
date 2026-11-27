# -*- coding: utf-8 -*-
"""
progress_eta.py -- readable progress and time estimate of the long Step-1 EM inversion
(calibrate, map and full runs all go through it).

Why
---
The old display was ``tqdm(executor.map(...))``: ``map`` yields in SUBMISSION order, so the bar waits for the
slowest early task (the ETA is meaningless), and every worker printed its own ``456/1926 inverted`` and
``Converting quadrature columns ...`` lines into the same terminal.

What it does
------------
* ``estimate_seconds`` / ``benchmark_costs``: BEFORE dispatch, time a few points of each (method, forward model)
  in the main process and simulate the pool (greedy list scheduling) -> "about 42 min on 8 workers".
* ``Step1Progress`` + ``run_ordered_progress``: tasks are submitted individually and polled; the line shows
  percent, tasks, POINTS (tasks differ a lot in size, so ETA follows points, not task counts), elapsed, ETA with
  the expected clock time, workers and points/s.  On a terminal it refreshes in place; in a log file (nohup,
  batch) it prints one line every ``LOG_INTERVAL_S`` seconds.
* ``EmagpyOutputFilter`` / ``run_filtered`` (worker side): EMagPy's per-point ``n/N inverted`` text is turned into
  a shared per-task counter (live progress inside a long task) and dropped from the output, as is the
  quadrature-conversion notice.  Everything else a worker prints is passed through.

Pure Python (numpy not needed); the pipeline imports it, tests exercise it with fake clocks and threads.
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import datetime
import heapq
import os
import re
import shutil
import sys
import tempfile
import time

TTY_INTERVAL_S = 1.0        # refresh of the in-place line on a terminal
LOG_INTERVAL_S = 30.0       # one line per interval when the output is a file / pipe
POLL_S = 0.5                # how often the main loop wakes up
ETA_MIN_ELAPSED_S = 5.0     # no ETA before this
RATE_WINDOW_S = 180.0       # recent-rate window blended with the global rate
BAR_WIDTH = 20
SLOW_FACTOR = 0.25          # warn when the live rate is below this fraction of the benchmark's expected rate
SLOW_MIN_ELAPSED_S = 60.0   # ... but not before this (workers need time to start)
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")

_SHARED = None              # worker-side: multiprocessing.Array('d') of "points done" per task


# --------------------------------------------------------------------------------------- formatting
def format_duration(seconds: float | None) -> str:
    """'45 s', '7 min 05 s', '2 h 14 min'; '?' for None / non-finite."""
    if seconds is None or seconds != seconds or seconds in (float("inf"), float("-inf")):
        return "?"
    s = int(round(max(0.0, seconds)))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


def format_count(n: float) -> str:
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e4:
        return f"{n / 1e3:.0f}k"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{n:.0f}"


def count_csv_points(path: str) -> int:
    """Data rows of an EM survey CSV (lines minus the header); 0 when unreadable."""
    try:
        with open(path, "rb") as f:
            n = sum(1 for _ in f)
        return max(0, n - 1)
    except OSError:
        return 0


# --------------------------------------------------------------------------------------- estimate
def estimate_seconds(task_points: list[float], task_cost_per_point: list[float], n_workers: int,
                     njobs: int = 1, startup_s: float = 3.0) -> float:
    """Makespan [s] of the pool: tasks are taken in order by the first free worker (as ProcessPoolExecutor does).

    ``task_cost_per_point`` is seconds per point on ONE core; ``njobs`` cores per task divide it (ideal scaling).
    ``startup_s`` is the spawn / import cost of a worker.
    """
    n_workers = max(1, int(n_workers))
    free = [0.0] * n_workers
    heapq.heapify(free)
    for pts, cost in zip(task_points, task_cost_per_point):
        t0 = heapq.heappop(free)
        heapq.heappush(free, t0 + pts * cost / max(1, int(njobs)))
    return startup_s + (max(free) if free else 0.0)


def _task_key(task) -> tuple:
    kw = task[5]
    return (str(kw.get("method")), str(kw.get("forwardModel")))


def benchmark_costs(worker_fn, tasks: list, *, n_points: int = 3, max_seconds: float = 90.0,
                    clock=time.perf_counter) -> dict[tuple, float]:
    """Seconds per point on one core, per (method, forwardModel), measured on the first task of each kind.

    The first ``n_points`` rows of the task's CSV are inverted in THIS process through ``worker_fn`` (after a
    1-row warm-up for imports).  A kind that fails or exceeds ``max_seconds`` is left out (the caller then
    prints no estimate for it).
    """
    costs: dict[tuple, float] = {}
    firsts: dict[tuple, tuple] = {}
    for t in tasks:
        firsts.setdefault(_task_key(t), t)
    tmp = tempfile.mkdtemp(prefix="eta_bench_")
    try:
        for key, t in firsts.items():
            try:
                with open(t[1], newline="", encoding="utf-8", errors="replace") as f:
                    rows = list(csv.reader(f))
                if len(rows) < 2:
                    continue
                kw = {k: v for k, v in t[5].items() if k != "njobs"}
                times = []
                for n, label in ((1, "warm"), (min(n_points, len(rows) - 1), "timed")):
                    path = os.path.join(tmp, f"bench_{abs(hash(key))}_{label}.csv")
                    with open(path, "w", newline="", encoding="utf-8") as g:
                        csv.writer(g).writerows(rows[: n + 1])
                    t0 = clock()
                    worker_fn((t[0], path, *t[2:5], kw))
                    times.append((clock() - t0, n))
                    if times[-1][0] > max_seconds:
                        break
                else:
                    dt, n = times[-1]
                    costs[key] = dt / max(1, n)
            except Exception:       # noqa: BLE001 - an estimate must never stop a run
                continue
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return costs


def describe_estimate(costs: dict[tuple, float], tasks: list, task_points: list[float], n_workers: int,
                      njobs: int = 1, label: str = "Step 1") -> str | None:
    """Multi-line text of the upfront estimate, None when no cost could be measured."""
    per_task = []
    for t in tasks:
        c = costs.get(_task_key(t))
        if c is None:
            return None
        per_task.append(c)
    total = estimate_seconds(task_points, per_task, n_workers, njobs)
    now = datetime.datetime.now()
    kinds = ", ".join(f"{m}/{fm} {1000 * c:.0f} ms" for (m, fm), c in sorted(costs.items()))
    return (f"[ETA] {label} estimate: about {format_duration(total)} on {n_workers} worker(s)"
            f"{f' x {njobs} job(s)' if njobs > 1 else ''}, finishing around "
            f"{(now + datetime.timedelta(seconds=total)).strftime('%H:%M')}\n"
            f"[ETA]   {len(tasks)} task(s), {format_count(sum(task_points))} points; per point and member: {kinds}\n"
            f"[ETA]   rough (3-point benchmark, +-50 %): the live line below replaces it once the run is under way")


def expected_rate(costs: dict[tuple, float], tasks: list, task_points: list[float], n_workers: int,
                  njobs: int = 1) -> float | None:
    """Points per second the benchmark predicts for the whole pool (no start-up), None without a full benchmark."""
    per_task = []
    for t in tasks:
        c = costs.get(_task_key(t))
        if c is None:
            return None
        per_task.append(c)
    span = estimate_seconds(task_points, per_task, n_workers, njobs, startup_s=0.0)
    return sum(task_points) / span if span > 0 else None


def limit_worker_threads(n_workers: int, env=None) -> tuple[list[str], list[str]]:
    """One BLAS / OpenMP thread per Step-1 worker, unless the user already chose a value.

    The pool already runs one process per core; if every worker also starts one BLAS thread per core, the machine
    is oversubscribed (measured: 5.5 x slower on 24 workers).  Call BEFORE the pool is created (spawned workers
    inherit the environment when they import numpy).  Returns (variables set to 1, user values > 1 left alone).
    """
    env = os.environ if env is None else env
    set_, kept = [], []
    for v in THREAD_VARS:
        cur = env.get(v)
        if cur is None:
            env[v] = "1"
            set_.append(v)
        elif n_workers > 1:
            try:
                if int(cur) > 1:
                    kept.append(f"{v}={cur}")
            except ValueError:
                pass
    return set_, kept


# --------------------------------------------------------------------------------------- worker side
_PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*inverted")
_NOISE_RE = re.compile(r"Converting quadrature columns to LIN ECa\..*?forward model\.", re.S)


class EmagpyOutputFilter:
    """``sys.stdout`` replacement: EMagPy's ``n/N inverted`` -> shared counter, quadrature notice dropped."""

    def __init__(self, real, idx: int | None = None, shared=None):
        self._real, self._idx, self._shared = real, idx, shared

    def write(self, text: str) -> int:
        n = len(text)
        if not text:
            return 0
        for m in _PROGRESS_RE.finditer(text):
            if self._shared is not None and self._idx is not None:
                try:
                    self._shared[self._idx] = float(m.group(1))
                except (IndexError, ValueError):
                    pass
        stripped = _NOISE_RE.sub("", _PROGRESS_RE.sub("", text))
        cleaned = stripped.replace("\r", "")
        # Drop a write only when the filtering emptied it.  A bare "\n" (print() writes its end separately)
        # must pass through, otherwise consecutive lines of other output run together.
        if cleaned.strip() or (cleaned and stripped == text):
            self._real.write(cleaned)
        return n

    def flush(self) -> None:
        self._real.flush()

    def __getattr__(self, name):            # isatty, fileno, encoding ...
        return getattr(self._real, name)


def init_worker(shared) -> None:
    """ProcessPoolExecutor initializer: remember the shared per-task counter."""
    global _SHARED
    _SHARED = shared


def run_filtered(fn, task):
    """Worker entry: run ``fn(task)`` with EMagPy's chatter filtered; ``task[0]`` is the shared-counter index."""
    real = sys.stdout
    sys.stdout = EmagpyOutputFilter(real, task[0], _SHARED)
    try:
        return fn(task)
    finally:
        sys.stdout = real


# --------------------------------------------------------------------------------------- main side
class Step1Progress:
    """Progress / ETA of a set of tasks, measured in POINTS.

    ``shared`` (optional, a sequence indexed by task index) holds the points a RUNNING task has done so far.
    ``clock`` / ``wall`` / ``stream`` / ``tty`` are injectable for tests.
    """

    def __init__(self, label: str, task_points: list[float], n_workers: int, shared=None, *, stream=None,
                 tty: bool | None = None, clock=time.time, wall=datetime.datetime.now,
                 expected_rate: float | None = None):
        self.label = label
        self.points = [float(p) for p in task_points]
        self.n_workers = max(1, int(n_workers))
        self.shared = shared
        self.stream = stream if stream is not None else sys.stdout
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)()) if tty is None else tty
        self.clock, self.wall = clock, wall
        self.start = clock()
        self.done: set[int] = set()
        self._samples: list[tuple[float, float]] = []
        self._last_print = -1e18
        self._last_len = 0
        self.expected_rate = expected_rate      # points/s predicted by the benchmark (None: no slow-run warning)
        self._warned = False

    # ---- numbers
    @property
    def total_points(self) -> float:
        return sum(self.points)

    def mark_done(self, idx: int) -> None:
        self.done.add(int(idx))

    def done_points(self) -> float:
        tot = 0.0
        for i, p in enumerate(self.points):
            if i in self.done:
                tot += p
            elif self.shared is not None:
                try:
                    tot += min(float(self.shared[i]), p)
                except (IndexError, TypeError, ValueError):
                    pass
        return tot

    def rate(self, now: float | None = None) -> float | None:
        """Points per second: global, blended with the last ``RATE_WINDOW_S`` seconds once available."""
        now = self.clock() if now is None else now
        elapsed = now - self.start
        done = self.done_points()
        if done <= 0 or elapsed < ETA_MIN_ELAPSED_S:
            return None
        rate = done / elapsed
        self._samples.append((now, done))
        self._samples = [s for s in self._samples if now - s[0] <= RATE_WINDOW_S]
        t0, d0 = self._samples[0]
        if now - t0 >= 30.0:
            rate = 0.5 * rate + 0.5 * (done - d0) / (now - t0)
        return rate

    def eta_seconds(self, now: float | None = None) -> float | None:
        r = self.rate(now)
        if not r:
            return None
        return max(0.0, (self.total_points - self.done_points()) / r)

    # ---- text
    def render(self, now: float | None = None) -> str:
        now = self.clock() if now is None else now
        done, total = self.done_points(), self.total_points
        frac = 0.0 if total <= 0 else min(1.0, done / total)
        bar = "#" * int(round(frac * BAR_WIDTH)) + "-" * (BAR_WIDTH - int(round(frac * BAR_WIDTH)))
        elapsed = now - self.start
        eta = self.eta_seconds(now)
        rate = self.rate(now)
        if eta is None:
            eta_txt = "ETA estimating..."
        else:
            at = self.wall() + datetime.timedelta(seconds=eta)
            eta_txt = f"ETA {format_duration(eta)} (~{at.strftime('%H:%M')})"
        parts = [f"[{self.label}] [{bar}] {100 * frac:5.1f}%",
                 f"tasks {len(self.done)}/{len(self.points)}",
                 f"points {format_count(done)}/{format_count(total)}",
                 f"elapsed {format_duration(elapsed)}", eta_txt,
                 f"{self.n_workers} worker(s)" + (f", {rate:.0f} pts/s" if rate else "")]
        return " | ".join(parts)

    def slow_warning(self, now: float | None = None) -> str | None:
        """One-time warning text when the live rate is far below the benchmark's, else None."""
        if self._warned or not self.expected_rate:
            return None
        now = self.clock() if now is None else now
        if now - self.start < SLOW_MIN_ELAPSED_S:
            return None
        r = self.rate(now)
        if not r or r >= SLOW_FACTOR * self.expected_rate:
            return None
        self._warned = True
        return (f"[WARN] [{self.label}] live rate {r:.0f} pts/s is {self.expected_rate / r:.0f} x below the benchmark "
                f"(~{self.expected_rate:.0f} pts/s). Check: BLAS thread oversubscription (set OMP_NUM_THREADS=1 "
                f"OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1), other jobs on the machine, or fewer physical cores "
                f"than workers. Early values also lag (progress is counted when EMagPy reports).")

    def tick(self, now: float | None = None, force: bool = False) -> bool:
        """Print the line when due (terminal: every second, in place; file: every LOG_INTERVAL_S). True if printed."""
        now = self.clock() if now is None else now
        interval = TTY_INTERVAL_S if self.tty else LOG_INTERVAL_S
        if not force and now - self._last_print < interval:
            return False
        self._last_print = now
        line = self.render(now)
        if self.tty:
            pad = " " * max(0, self._last_len - len(line))
            self.stream.write("\r" + line + pad)
            self._last_len = len(line)
        else:
            self.stream.write(line + "\n")
        warn = self.slow_warning(now)
        if warn:
            self.stream.write(("\n" if self.tty else "") + warn + "\n")
            self._last_len = 0
        self.stream.flush()
        return True

    def close(self) -> str:
        """Final line (always a newline-terminated line); returns the summary text."""
        now = self.clock()
        summary = (f"[{self.label}] done: {len(self.done)}/{len(self.points)} tasks, "
                   f"{format_count(self.total_points)} points in {format_duration(now - self.start)}")
        if self.tty and self._last_len:
            self.stream.write("\r" + " " * self._last_len + "\r")
        self.stream.write(summary + "\n")
        self.stream.flush()
        return summary


def run_ordered_progress(executor, fn, tasks: list, tracker: Step1Progress, poll_s: float = POLL_S):
    """Submit every task, poll, refresh ``tracker``; yield each result when its task finishes (any order).

    ``fn(task)`` must be picklable for a process pool.  On an exception the pending tasks are cancelled and it is
    re-raised (as ``executor.map`` did).
    """
    futures = {executor.submit(fn, t): t[0] for t in tasks}
    pending = set(futures)
    try:
        tracker.tick(force=True)
        while pending:
            done, pending = cf.wait(pending, timeout=poll_s, return_when=cf.FIRST_COMPLETED)
            for f in done:
                tracker.mark_done(futures[f])
                yield f.result()
            tracker.tick()
    except BaseException:
        for f in pending:
            f.cancel()
        raise
    finally:
        tracker.close()
