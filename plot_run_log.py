#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_run_log.py -- read the run-log CSV and generate ALL the summary plots, one folder
per study type, so a study hypothesis can be checked without copying run ids by hand.

The log has a ``study_type`` column (hyp1_layers_ws_freedom, hyp2_train_val_split,
surface_vs_full, swc_design_study, ...; see ``run_log.STUDY_TYPE_RULES``). Old logs
without the column are handled on the fly (the type is inferred from the scenario
name); ``--upgrade`` rewrites the file once with the new columns.

Usage
-----
    python plot_run_log.py --list                       # runs per study type, nothing written
    python plot_run_log.py                              # all study types of the shared log
    python plot_run_log.py --study-type hyp1_layers_ws_freedom surface_vs_full
    python plot_run_log.py --log-file mfc2_outputs/run_log_swc_study.csv   # (see also plot_swc_study_log.py)
    python plot_run_log.py --compare                    # + compare_scenarios.py on each study
    python plot_run_log.py --upgrade                    # add the new columns to the CSV first
    python plot_run_log.py --figures                    # + copy the key pipeline figures of each run
    python plot_run_log.py --figures --tile             # + one contact sheet per figure (scenarios side by side)
    python plot_run_log.py --maps                       # + re-plot the SWC maps of all runs of a study on ONE colour scale
    python plot_run_log.py --maps --map-var theta_std_total --epoch 0

Output: ``<out-dir>/<study_type>/`` (default ``comparison_output/run_log_plots``), with

    rmse_by_scenario.png   headline RMSE in % vol per scenario (latest finished run), bars
                           coloured by the RMSE rating when the run logged RMSE/std (RSR)
    rmse_vs_rsr.png        RMSE [% vol] vs RSR = RMSE/std(observed SWC): is the error large
                           compared with the natural spread of the data? (RSR 0.5 / 0.7 / 1 lines)
    rmse_vs_k.png          RMSE vs number of free Waxman-Smits parameters K (names ``..free{K}``),
                           one line per scenario family (e.g. surface vs full)
    rmse_history.png       RMSE of every run over time, one line per scenario (did a change help?)
    runtime_vs_rmse.png    cost / accuracy trade-off
    runs.csv               the table behind the plots
    figures/<figure>/<scenario>.png   (--figures) the key figure of every run, one folder per figure
    maps_<var>_epoch<t>.png  (--maps) the maps of all scenarios redrawn from output_model/maps.nc
                           (maps_export.py): rows = scenarios, columns = layers, ONE shared colour
                           scale (2-98 percentile of all panels); 1-D runs: theta along x, one line per layer
    figures/<figure>_tiled.png        (--tile) those images side by side (a contact sheet of the
                           existing PNGs, not a re-plot: colour scales are those of each run;
                           a real re-plot needs the maps as arrays / NetCDF, see to-do.md)

Only the latest finished run of each scenario is used (failed runs are skipped), except
in rmse_history.png.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from EA_EM_config import OUTPUT_ROOT  # noqa: E402
from rmse_context import RSR_BANDS, parse_rmse, rating_from_rsr  # noqa: E402
from run_log import RunLog, infer_study_type  # noqa: E402

SHARED_LOG = os.path.join(OUTPUT_ROOT, "run_log.csv")
DEFAULT_OUT = os.path.join("comparison_output", "run_log_plots")
FAILED = {"FAIL", "FAILED", "ERROR"}
MAX_COMPARE = 12                                  # compare_scenarios gets unreadable above this
RATING_COLORS = {"very good": "#2e7d32", "good": "#7cb342", "satisfactory": "#f9a825",
                 "unsatisfactory": "#ef6c00", "no skill": "#c62828", "std too small": "#9e9e9e",
                 "": "#607d8b", "n/a": "#607d8b"}


# --------------------------------------------------------------------------
# Reading the log (pure helpers, unit-tested)
# --------------------------------------------------------------------------
def _num(value) -> float:
    return parse_rmse(value)


def load_rows(log_path: str) -> list[dict]:
    """Rows of the log with numeric columns parsed and ``study_type`` / ``rmse_pct`` filled in
    when the log predates them (the file itself is not modified)."""
    out = []
    for r in RunLog(log_path).rows():
        d = dict(r)
        d["study_type"] = r.get("study_type") or infer_study_type(r.get("scenario"))
        rmse = _num(r.get("final_rmse"))
        pct = _num(r.get("rmse_pct"))
        d["rmse"] = rmse
        d["rmse_pct"] = pct if np.isfinite(pct) else rmse * 100.0
        d["rsr"] = _num(r.get("rsr"))
        d["swc_obs_std"] = _num(r.get("swc_obs_std"))
        d["runtime"] = _num(r.get("runtime_s"))
        d["rating"] = r.get("rmse_rating") or (
            rating_from_rsr(d["rsr"], d["swc_obs_std"]) if np.isfinite(d["rsr"]) else "")
        d["status_ok"] = (r.get("status", "").upper() not in FAILED) and np.isfinite(rmse)
        out.append(d)
    return out


def latest_per_scenario(rows: list[dict]) -> dict[str, dict]:
    """scenario -> latest finished run (rows are oldest first, later rows win)."""
    out: dict[str, dict] = {}
    for r in rows:
        if r["status_ok"]:
            out[r["scenario"]] = r
    return out


def group_by_study(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["study_type"], []).append(r)
    return dict(sorted(groups.items()))


def split_free_k(scenario: str) -> tuple[str, int | None]:
    """``'1d_surface_hetero_free3'`` -> (``'1d_surface_hetero'``, 3); (name, None) without ``free{K}``."""
    m = re.match(r"^(.*?)_?free(\d+)(?:_(.*))?$", scenario)
    if not m:
        return scenario, None
    family = m.group(1) + (("_" + m.group(3)) if m.group(3) else "")
    return family, int(m.group(2))


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rmse_by_scenario(latest: dict[str, dict], out_dir: str, title: str) -> str | None:
    items = sorted(latest.values(), key=lambda r: r["rmse_pct"])
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(items) + 1.2)))
    y = np.arange(len(items))
    ax.barh(y, [r["rmse_pct"] for r in items], color=[RATING_COLORS.get(r["rating"], "#607d8b") for r in items])
    ax.set_yticks(y)
    ax.set_yticklabels([r["scenario"] for r in items], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE [% vol]  (= m³/m³ × 100)")
    ax.set_title(f"{title}: RMSE by scenario (latest run)")
    for yi, r in zip(y, items):
        txt = f"{r['rmse_pct']:.2f}" + (f"  (RSR {r['rsr']:.2f})" if np.isfinite(r["rsr"]) else "")
        ax.text(r["rmse_pct"], yi, " " + txt, va="center", fontsize=6)
    shown = sorted({r["rating"] for r in items if r["rating"]})
    if shown:
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=RATING_COLORS.get(k, "#607d8b")) for k in shown],
                  labels=[f"{k}" for k in shown], title="RMSE/std rating", fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, out_dir, "rmse_by_scenario.png")


def plot_rmse_vs_rsr(latest: dict[str, dict], out_dir: str, title: str) -> str | None:
    items = [r for r in latest.values() if np.isfinite(r["rsr"])]
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter([r["rmse_pct"] for r in items], [r["rsr"] for r in items],
               c=[RATING_COLORS.get(r["rating"], "#607d8b") for r in items], s=28)
    for r in items:
        ax.annotate(r["scenario"], (r["rmse_pct"], r["rsr"]), fontsize=5, alpha=0.8)
    for upper, label in RSR_BANDS:
        ax.axhline(upper, color="grey", lw=0.6, ls=":")
        ax.text(ax.get_xlim()[1], upper, f" {label}", va="center", fontsize=6, color="grey")
    ax.axhline(1.0, color="#c62828", lw=0.8, ls="--")
    ax.set_xlabel("RMSE [% vol]")
    ax.set_ylabel("RSR = RMSE / std(observed SWC)")
    ax.set_title(f"{title}: error relative to the spread of the data")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "rmse_vs_rsr.png")


def plot_rmse_vs_k(latest: dict[str, dict], out_dir: str, title: str) -> str | None:
    fam: dict[str, list[tuple[int, float]]] = {}
    for name, r in latest.items():
        f, k = split_free_k(name)
        if k is not None:
            fam.setdefault(f, []).append((k, r["rmse_pct"]))
    fam = {f: sorted(v) for f, v in fam.items() if len(v) >= 2}
    if not fam:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for f, v in sorted(fam.items()):
        ax.plot([k for k, _ in v], [e for _, e in v], "o-", label=f, lw=1.2)
    ax.set_xlabel("free Waxman-Smits parameters K")
    ax.set_ylabel("RMSE [% vol]")
    ax.set_xticks(sorted({k for v in fam.values() for k, _ in v}))
    ax.set_title(f"{title}: RMSE vs K")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "rmse_vs_k.png")


def plot_rmse_history(rows: list[dict], out_dir: str, title: str) -> str | None:
    by: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        if r["status_ok"]:
            by.setdefault(r["scenario"], []).append((r.get("timestamp", ""), r["rmse_pct"]))
    by = {k: v for k, v in by.items() if len(v) >= 2}
    if not by:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, v in sorted(by.items()):
        ax.plot(range(1, len(v) + 1), [e for _, e in v], "o-", lw=1, ms=3, label=name)
    ax.set_xlabel("run number of the scenario (oldest = 1)")
    ax.set_ylabel("RMSE [% vol]")
    ax.set_title(f"{title}: RMSE of repeated runs")
    ax.legend(fontsize=5)
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "rmse_history.png")


def plot_runtime_vs_rmse(latest: dict[str, dict], out_dir: str, title: str) -> str | None:
    items = [r for r in latest.values() if np.isfinite(r["runtime"])]
    if len(items) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter([r["runtime"] / 60 for r in items], [r["rmse_pct"] for r in items], s=25)
    for r in items:
        ax.annotate(r["scenario"], (r["runtime"] / 60, r["rmse_pct"]), fontsize=5, alpha=0.8)
    ax.set_xlabel("runtime [min]")
    ax.set_ylabel("RMSE [% vol]")
    ax.set_title(f"{title}: cost vs accuracy")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "runtime_vs_rmse.png")


def write_table(latest: dict[str, dict], out_dir: str) -> str:
    import csv
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "runs.csv")
    cols = ["scenario", "study_type", "run_id", "status", "rmse_pct", "swc_obs_std", "rsr", "rating", "runtime"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(latest.values(), key=lambda r: r["scenario"]):
            w.writerow([r.get("scenario"), r["study_type"], r.get("run_id"), r.get("status"),
                        f"{r['rmse_pct']:.3f}", "" if not np.isfinite(r["swc_obs_std"]) else f"{r['swc_obs_std']:.5f}",
                        "" if not np.isfinite(r["rsr"]) else f"{r['rsr']:.4f}", r["rating"],
                        "" if not np.isfinite(r["runtime"]) else f"{r['runtime']:.1f}"])
    return path


# --------------------------------------------------------------------------
# Key pipeline figures: copy and tile
# --------------------------------------------------------------------------
# label -> glob patterns (first match wins) searched recursively in the run folder
KEY_FIGURES = {
    "swc_map_xy": ["water_content_xy_*_imshow.png", "water_content_xy_*.png"],
    "swc_section_xz": ["water_content_xz_*.png"],
    "swc_uncertainty": ["solution_swc_uncertainty.png"],
    "swc_profiles": ["solution_swc_profiles.png"],
    "parity_by_sensor": ["parity_validation_by_sensor.png", "parity_validation.png"],
    "theta_1to1_by_horizon": ["fig_theta_1to1_by_horizon.png"],
    "ws_histograms": ["ws_parameter_histograms.png"],
    "bandwidth_sweep": ["bandwidth_sweep.png"],
}
_SKIP = ("pctdiff", "truth", "binned", "_test")


def find_figure(run_dir: str, patterns: list[str]) -> str | None:
    """First figure of the run folder matching one of the glob patterns (None if absent)."""
    import glob
    if not run_dir or not os.path.isdir(run_dir):
        return None
    for pat in patterns:
        hits = sorted(h for h in glob.glob(os.path.join(run_dir, "**", pat), recursive=True)
                      if not any(k in os.path.basename(h) for k in _SKIP)
                      or any(k in pat for k in _SKIP))
        if hits:
            return hits[0]
    return None


def gather_figures(latest: dict[str, dict], out_dir: str, labels: list[str] | None = None) -> dict[str, dict[str, str]]:
    """Copy the key figure of each run to ``<out_dir>/figures/<label>/<scenario>.png``.

    Returns {label: {scenario: copied path}}; figures a run does not have are skipped.
    """
    import shutil
    found: dict[str, dict[str, str]] = {}
    for label in (labels or list(KEY_FIGURES)):
        for name, r in sorted(latest.items()):
            src = find_figure(r.get("output_dir", ""), KEY_FIGURES[label])
            if src:
                dst_dir = os.path.join(out_dir, "figures", label)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, f"{name}.png")
                shutil.copyfile(src, dst)
                found.setdefault(label, {})[name] = dst
    return found


def tile_figures(label: str, images: dict[str, str], out_dir: str, ncols: int = 3) -> str | None:
    """One contact sheet: the images of several scenarios side by side, titled by scenario."""
    if len(images) < 2:
        return None
    import matplotlib.image as mpimg
    names = sorted(images)
    ncols = min(ncols, len(names))
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, n in zip(axes.ravel(), names):
        ax.imshow(mpimg.imread(images[n]))
        ax.set_title(n, fontsize=8)
    fig.suptitle(label, fontsize=10)
    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "figures"), f"{label}_tiled.png")


def load_run_maps(latest: dict[str, dict]) -> dict:
    """scenario -> xarray.Dataset of its ``output_model/maps.nc`` (runs without one are skipped)."""
    import maps_export as mx
    out = {}
    for name, r in sorted(latest.items()):
        d = r.get("output_dir", "")
        for sub in ("output_model", ""):
            p = os.path.join(d, sub, mx.MAPS_NAME)
            if d and os.path.isfile(p):
                ds = mx.load_maps(p)
                if ds is not None:
                    out[name] = ds
                break
    return out


def shared_limits(arrays: list, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    """Colour limits shared by all panels: percentiles of the pooled finite values."""
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays]) if arrays else np.array([])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(vals, [lo, hi])
    return (float(vmin), float(vmax)) if vmax > vmin else (float(vmin), float(vmin) + 1e-9)


def plot_maps_side_by_side(maps: dict, out_dir: str, var: str = "theta", epoch: int = -1,
                           title: str = "") -> str | None:
    """Redraw ``var`` of several runs on ONE colour scale (rows = scenarios, columns = layers).

    2-D runs: scatter of the measured points (x, y); 1-D runs (all y equal): ``var`` along x,
    one line per layer. Runs without ``var`` are skipped. Returns the figure path or None.
    """
    maps = {k: ds for k, ds in maps.items() if var in ds}
    if len(maps) < 1:
        return None
    unit = "% vol" if var.startswith("theta") else ""
    scale = 100.0 if unit else 1.0
    get = lambda ds: np.asarray(ds[var].isel(time=epoch if abs(epoch) <= ds.sizes["time"] else -1), dtype=float) * scale
    is2d = all(np.unique(np.asarray(ds["y"])).size > 1 for ds in maps.values())
    names = list(maps)
    nl = max(ds.sizes["layer"] for ds in maps.values())
    lab = lambda ds, l: (f"{float(ds['layer_depth'][l]):.2f} m" if "layer_depth" in ds.coords else f"layer {l}")
    vmin, vmax = shared_limits([get(ds) for ds in maps.values()])
    if is2d:
        fig, axes = plt.subplots(len(names), nl, figsize=(3.2 * nl + 1, 2.8 * len(names)), squeeze=False)
        sc = None
        for i, n in enumerate(names):
            ds = maps[n]
            a = get(ds)
            for j in range(nl):
                ax = axes[i, j]
                if j >= ds.sizes["layer"]:
                    ax.axis("off")
                    continue
                sc = ax.scatter(ds["x"], ds["y"], c=a[:, j], s=10, vmin=vmin, vmax=vmax, cmap="viridis_r")
                ax.set_aspect("equal")
                ax.tick_params(labelsize=5)
                ax.set_title(f"{n}\n{lab(ds, j)}" if j == 0 else lab(ds, j), fontsize=6)
        fig.colorbar(sc, ax=axes.ravel().tolist(), label=f"{var} [{unit}]" if unit else var, shrink=0.8)
    else:
        fig, axes = plt.subplots(len(names), 1, figsize=(6, 2.4 * len(names)), squeeze=False, sharex=False)
        for i, n in enumerate(names):
            ds = maps[n]
            a = get(ds)
            order = np.argsort(np.asarray(ds["x"]))
            for j in range(ds.sizes["layer"]):
                axes[i, 0].plot(np.asarray(ds["x"])[order], a[order, j], "o-", ms=3, lw=1, label=lab(ds, j))
            axes[i, 0].set_ylim(vmin, vmax)
            axes[i, 0].set_ylabel(f"{var} [{unit}]" if unit else var, fontsize=7)
            axes[i, 0].set_title(n, fontsize=7)
            axes[i, 0].legend(fontsize=5, ncol=4)
            axes[i, 0].grid(alpha=0.3)
        axes[-1, 0].set_xlabel("x [m]")
    fig.suptitle(f"{title}: {var}, epoch {epoch}, shared colour scale {vmin:.1f}-{vmax:.1f} {unit}".strip(), fontsize=9)
    if not is2d:
        fig.tight_layout()
    return _save(fig, out_dir, f"maps_{var}_epoch{epoch}.png")


def plot_study(study: str, rows: list[dict], out_root: str, figures: bool = False, tile: bool = False,
               maps: bool = False, map_var: str = "theta", epoch: int = -1) -> list[str]:
    """All plots of one study type; returns the files written."""
    out_dir = os.path.join(out_root, study)
    latest = latest_per_scenario(rows)
    if not latest:
        return []
    files = [write_table(latest, out_dir)]
    for fn, arg in ((plot_rmse_by_scenario, latest), (plot_rmse_vs_rsr, latest),
                    (plot_rmse_vs_k, latest), (plot_rmse_history, rows), (plot_runtime_vs_rmse, latest)):
        p = fn(arg, out_dir, study)
        if p:
            files.append(p)
    if figures or tile:
        found = gather_figures(latest, out_dir)
        files += [p for d in found.values() for p in d.values()]
        if tile:
            files += [t for lab, d in found.items() if (t := tile_figures(lab, d, out_dir))]
    if maps:
        p = plot_maps_side_by_side(load_run_maps(latest), out_dir, map_var, epoch, study)
        if p:
            files.append(p)
        else:
            print(f"[plot_run_log] {study}: no run has {map_var} in output_model/maps.nc "
                  f"(written by runs made after maps_export.py was added)")
    return files


def compare_study(study: str, rows: list[dict], log_path: str, out_root: str, dry_run: bool = False) -> int:
    """Run compare_scenarios.py on the latest run of each scenario of the study."""
    latest = latest_per_scenario(rows)
    if len(latest) < 2:
        print(f"[plot_run_log] {study}: need >= 2 finished scenarios to compare ({len(latest)}), skipped")
        return 0
    if len(latest) > MAX_COMPARE:
        print(f"[plot_run_log] {study}: {len(latest)} scenarios > {MAX_COMPARE}, too many for one compare_scenarios "
              f"figure; use auto_compare.py with filters. Skipped.")
        return 0
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_scenarios.py"),
           "--log-file", log_path, "--ids", *[r["run_id"] for r in latest.values()],
           "--labels", *latest.keys(), "--out-dir", os.path.join(out_root, study, "compare_scenarios")]
    print("[plot_run_log] " + " ".join(cmd))
    return 0 if dry_run else subprocess.call(cmd)


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-file", default=SHARED_LOG, help="run-log CSV (default: %(default)s)")
    ap.add_argument("--study-type", nargs="+", help="only these study types (default: all in the log)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--list", action="store_true", help="print the runs per study type and exit")
    ap.add_argument("--upgrade", action="store_true", help="add the new columns to the CSV first (backup .bak)")
    ap.add_argument("--compare", action="store_true", help="also run compare_scenarios.py per study type")
    ap.add_argument("--figures", action="store_true", help="also copy the key pipeline figures of each run")
    ap.add_argument("--tile", action="store_true", help="also tile those figures side by side (implies --figures)")
    ap.add_argument("--maps", action="store_true", help="re-plot the SWC maps of all runs on one colour scale (maps.nc)")
    ap.add_argument("--map-var", default="theta", help="variable of maps.nc (theta, theta_global, theta_std_total, ...)")
    ap.add_argument("--epoch", type=int, default=-1, help="survey epoch index for --maps (default: last)")
    ap.add_argument("--dry-run", action="store_true", help="with --compare: print the commands only")
    a = ap.parse_args(argv)

    if not os.path.isfile(a.log_file):
        print(f"[plot_run_log] run log not found: {a.log_file}", file=sys.stderr)
        return 2
    if a.upgrade:
        RunLog(a.log_file).ensure_schema()
    groups = group_by_study(load_rows(a.log_file))
    if a.study_type:
        unknown = [s for s in a.study_type if s not in groups]
        if unknown:
            print(f"[plot_run_log] no runs of study type {unknown}; available: {list(groups)}", file=sys.stderr)
            return 1
        groups = {s: groups[s] for s in a.study_type}
    print(f"[plot_run_log] {a.log_file}: {len(groups)} study type(s)")
    for s, rows in groups.items():
        print(f"   {s:<28s} {len(rows):4d} runs, {len(latest_per_scenario(rows)):3d} finished scenarios")
    if a.list:
        return 0
    for s, rows in groups.items():
        for f in plot_study(s, rows, a.out_dir, a.figures, a.tile, a.maps, a.map_var, a.epoch):
            print(f"   wrote {f}")
        if a.compare:
            compare_study(s, rows, a.log_file, a.out_dir, a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
