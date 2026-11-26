"""
plots/swc_design_overview.py
============================
One overview figure of the SWC design study: WHERE the probes are and WHAT they sample.

Rows
----
1. probe positions of the three distributions (spread / clustered / random) at the
   reference number of probes, on the survey transect (grey = survey points, red = probes);
2. truth soil moisture at t0 (the initial model), homogeneous vs heterogeneous truth;
3. truth Waxman-Smits parameters (a, m, sigma_w, sigma_s), homogeneous WS truth;
4. same, heterogeneous WS truth.
The probes of the reference distribution are overlaid on rows 2-4.

Inputs: one finished run folder per truth (the truth does not depend on n, k, dist or reg).
Files are looked up in ``<run_dir>/output_model/`` (where the pipeline writes them) or, failing
that, directly in ``<run_dir>/``.
    <run_dir>/synthetic_time_lapse_theta.npy   (ens, T, points, layers)
    <run_dir>/synthetic_x_coords.npy, synthetic_layer_centers.npy
    <run_dir>/synthetic_ws_truth_{a,m,sigma_w,sigma_s}.npy  (WS-hetero runs) or
    <run_dir>/synthetic_ws_truth_params.json                (per-layer truth, always written; stays in
                                                            the run folder, not in output_model/)
The probe layout row needs no run. A panel without data says "no run".

Output : ``<out_dir>/fig_design_and_truth.png`` (path returned).
Called by study_swc_design.make_plots and plot_swc_study_log.plot_all.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plots.ws_distribution import PARAMS, load_ws_truth, load_ws_truth_params  # noqa: E402

FIG_NAME = "fig_design_and_truth.png"
DISTS = ("spread", "clustered", "random")
_PLABEL = {"a": "a [-]", "m": "m [-]", "sigma_w": "σw [mS/m]", "sigma_s": "σs [mS/m]"}


def _data_dir(d: str) -> str | None:
    """Folder that holds ``synthetic_*.npy``: the run's ``output_model/`` (real runs) or the folder itself."""
    for cand in (os.path.join(d, "output_model"), d):
        if os.path.isfile(os.path.join(cand, "synthetic_time_lapse_theta.npy")):
            return cand
    return None


def pick_run_dir(rows: Iterable[dict], key: str, value: str, ref: dict | None = None) -> str | None:
    """Data folder of a finished run whose ``key`` (smc / ws) is ``value``; prefers the reference design.

    ``rows`` are dicts with ``key`` and ``run_dir`` (or ``output_dir``) and, optionally, the other
    study factors. A run counts when ``synthetic_time_lapse_theta.npy`` exists in the run folder
    or in its ``output_model/`` subfolder; the returned path is the folder that holds it.
    """
    ref = ref or {}
    best, best_score = None, -1
    for r in rows:
        d = r.get("run_dir") or r.get("output_dir") or ""
        if r.get(key) != value or not isinstance(d, str) or not d:
            continue
        dd = _data_dir(d)
        if dd is None:
            continue
        score = sum(1 for k, v in ref.items() if k in r and r[k] == v)
        if score > best_score:
            best, best_score = dd, score
    return best


def _imshow(ax, arr, x, depths, vmin, vmax, cmap):
    """``arr`` (points, layers) as an x-vs-layer image."""
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    im = ax.imshow(arr.T, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[x.min() - dx / 2, x.max() + dx / 2, arr.shape[1] - 0.5, -0.5])
    ax.set_yticks(range(arr.shape[1]))
    ax.set_yticklabels([f"{d:.2f}" for d in depths] if depths is not None and len(depths) == arr.shape[1]
                       else [f"H{i}" for i in range(arr.shape[1])], fontsize=6)
    return im


def _empty(ax, text="no run"):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="0.5")
    ax.set_xticks([])
    ax.set_yticks([])


def _load_run(run_dir: str | None, realization: int):
    """(theta0, x, depths, run_dir) of a run; Nones when missing."""
    if not run_dir:
        return None
    try:
        theta = np.load(os.path.join(run_dir, "synthetic_time_lapse_theta.npy"))
        x = np.load(os.path.join(run_dir, "synthetic_x_coords.npy"))
    except OSError:
        return None
    th = theta[realization if realization < theta.shape[0] else 0][0]     # (points, layers) at t0
    p_c = os.path.join(run_dir, "synthetic_layer_centers.npy")
    depths = np.load(p_c) if os.path.isfile(p_c) else None
    return th, np.asarray(x, dtype=float), depths


_REPORT_ROW = re.compile(
    r"Layer\s+(\d+)[^|\n]*\|\s*`?\s*([-+\d.eE]+)\s*/\s*([-+\d.eE]+)\s*/\s*([-+\d.eE]+)\s*mS/m\s*/\s*([-+\d.eE]+)\s*mS/m")


def _ws_params_from_report(run_dir: str) -> dict | None:
    """``{layer: {param: value}}`` read from the run's ``pipeline_report_*.md`` (None when absent).

    Runs made before ``synthetic_ws_truth_params.json`` existed only have the Step 0 table
    "Layer l - a / m / sigma_w / sigma_s" in the report. Values are rounded there (4 / 2 decimals).
    """
    for path in sorted(glob.glob(os.path.join(run_dir, "pipeline_report_*.md")), reverse=True):
        if "_shell_" in os.path.basename(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        out = {int(m.group(1)): dict(zip(PARAMS, (float(m.group(i)) for i in range(2, 6))))
               for m in _REPORT_ROW.finditer(text)}
        if out:
            return out
    return None


def _load_ws(data_dir: str, realization: int, n_points: int):
    """Truth WS ``(points, layers, 4)`` of a run.

    The pipeline moves only ``*.npy`` into ``output_model/``; ``synthetic_ws_truth_params.json``
    (the whole truth of a homogeneous-WS run) stays in the run folder, so look there too. Runs
    that predate that file fall back to the Step 0 table of their pipeline report.
    """
    run_dir = os.path.dirname(os.path.normpath(data_dir)) \
        if os.path.basename(os.path.normpath(data_dir)) == "output_model" else data_dir
    params = load_ws_truth_params(data_dir) or load_ws_truth_params(run_dir)
    t = load_ws_truth(data_dir, realization, params, n_points)
    if t is None and params is None:
        t = load_ws_truth(data_dir, realization, _ws_params_from_report(run_dir), n_points)
    return t


def _overlay_probes(ax, xs, x0, n_layers):
    ax.plot(np.asarray(xs) + x0, np.full(len(xs), -0.35), "v", color="white", mec="k", ms=5, clip_on=False)


def plot_design_and_truth(rows: Iterable[dict], out_dir: str, survey, position_fn: Callable,
                          n_ref: int = 5, dist_ref: str = "spread", realization: int = 0,
                          ref: dict | None = None) -> str:
    """Draw the overview figure and return its path.

    rows        study rows: dicts with ``smc``, ``ws`` and ``run_dir`` / ``output_dir`` (+ factors for ``ref``)
    position_fn ``study_swc_design.probe_positions`` (dist, n, survey) -> x [m]
    """
    rows = list(rows)
    survey = np.asarray(survey, dtype=float)
    smc_runs = {v: pick_run_dir(rows, "smc", v, ref) for v in ("homo", "hetero")}
    ws_runs = {v: pick_run_dir(rows, "ws", v, ref) for v in ("homo", "hetero")}
    smc_data = {v: _load_run(d, realization) for v, d in smc_runs.items()}
    ws_data = {v: _load_run(d, realization) for v, d in ws_runs.items()}
    depths = next((d[2] for d in [*smc_data.values(), *ws_data.values()] if d is not None and d[2] is not None), None)
    n_eff = int(min(n_ref, len(survey)))
    ref_xs = position_fn(dist_ref, n_eff, survey)

    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(4, 12, hspace=0.55, wspace=2.6)

    # ---- row 1: probe layouts ------------------------------------------------------------
    z = np.asarray(depths, dtype=float) if depths is not None else np.array([0.0])
    for i, dist in enumerate(DISTS):
        ax = fig.add_subplot(gs[0, 4 * i:4 * i + 4])
        xs = position_fn(dist, n_eff, survey)
        ax.plot(np.repeat(survey, len(z)), np.tile(z, len(survey)), ".", color="0.75", ms=4, label="survey point")
        ax.plot(np.repeat(xs, len(z)), np.tile(z, len(xs)), "o", color="tab:red", ms=6, label="SWC probe")
        ax.set_title(f"{dist} (n = {len(xs)})", fontsize=9, fontweight="bold")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("depth [m]" if depths is not None else "")
        if depths is not None:
            ax.invert_yaxis()
        else:
            ax.set_yticks([])
        ax.set_xticks(survey)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6, loc="lower left")

    # ---- row 2: truth SMC at t0 ----------------------------------------------------------
    vals = [d[0] for d in smc_data.values() if d is not None]
    vmin = min(v.min() for v in vals) if vals else 0.0
    vmax = max(v.max() for v in vals) if vals else 1.0
    if vmax - vmin < 1e-9:
        vmin, vmax = vmin - 0.01, vmax + 0.01
    im = None
    for i, v in enumerate(("homo", "hetero")):
        ax = fig.add_subplot(gs[1, 6 * i:6 * i + 6])
        d = smc_data[v]
        ax.set_title(f"truth SMC at t0, {v}", fontsize=9, fontweight="bold")
        if d is None:
            _empty(ax)
            continue
        th, x, dep = d
        im = _imshow(ax, th, x, dep, vmin, vmax, "viridis_r")
        _overlay_probes(ax, ref_xs, x.min(), th.shape[1])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("depth [m]", fontsize=8)
        ax.tick_params(labelsize=7)
    if im is not None:
        fig.colorbar(im, ax=[a for a in fig.axes[-2:]], label="θ [m³/m³]", shrink=0.85, pad=0.02)

    # ---- rows 3-4: truth WS --------------------------------------------------------------
    loaded = {}
    for v in ("homo", "hetero"):
        d = ws_data[v]
        try:
            loaded[v] = None if d is None else _load_ws(ws_runs[v], realization, len(d[1]))
        except (IndexError, ValueError):
            loaded[v] = None
    for r, v in enumerate(("homo", "hetero")):
        for c, p in enumerate(PARAMS):
            ax = fig.add_subplot(gs[2 + r, 3 * c:3 * c + 3])
            t, d = loaded[v], ws_data[v]
            ax.set_title(f"{p}  (WS {v})", fontsize=8, fontweight="bold")
            if t is None or d is None:
                _empty(ax)
                continue
            pool = [loaded[k][:, :, c] for k in ("homo", "hetero") if loaded[k] is not None]
            lo, hi = min(a.min() for a in pool), max(a.max() for a in pool)
            if hi - lo < 1e-9:
                lo, hi = lo - 0.05 * abs(lo) - 1e-6, hi + 0.05 * abs(hi) + 1e-6
            im = _imshow(ax, t[:, :, c], d[1], d[2], lo, hi, "magma_r")
            _overlay_probes(ax, ref_xs, d[1].min(), t.shape[1])
            cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.06)
            cb.set_label(_PLABEL[p], fontsize=6)
            cb.ax.tick_params(labelsize=6)
            ax.set_xlabel("x [m]", fontsize=7)
            if c == 0:
                ax.set_ylabel("depth [m]", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle(f"SWC design study: probe layouts, truth SMC (initial model) and truth WS "
                 f"(white triangles: {dist_ref} probes, n = {n_eff})", fontsize=11, y=0.94)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, FIG_NAME)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
