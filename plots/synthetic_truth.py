"""plots/synthetic_truth.py -- TRUTH maps of the synthetic mirror (theta, Waxman-Smits parameters).

``plot_truth_maps(gt_dir, out_dir, ...)`` writes, from the ``ground_truth/`` folder made by
``generate_synthetic_real_mirror.py``:

* ``truth_xy.png``         plan view: theta (two surveys) and sigma_s, one row per layer
* ``truth_xz.png``         vertical section: theta (two surveys) and sigma_s
* ``truth_ws_params.png``  ALL four Waxman-Smits parameters (a, m, sigma_w, sigma_s), plan view

Truth of the mirror: theta = fixed smooth spatial pattern + depth trend + a uniform seasonal
offset (so the pattern is the same on every date); a, m, sigma_w are constant in space and
time; only sigma_s can vary (generator option ``--ws-truth``, detected here from the saved
arrays). The truth is evaluated on a regular grid with the generator's own functions and
checked against the saved ``*.npy`` at the EM points (max difference is logged, ~0 expected).

Called by ``run_pipeline.main`` (Step 0, real-data branch) when the scenario sets
``ground_truth_dir``, and by the stand-alone ``plot_synthetic_truth.py``. Imports the
generator lazily (it needs emagpy, which the pipeline has anyway).
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt  # noqa: E402

TDR_DEPTHS_M = (0.15, 0.30)     # the two TDR depths of the deployment
THETA_CMAP, SIGMA_CMAP = "YlGnBu", "plasma"


def _gen():
    import generate_synthetic_real_mirror as gen
    return gen


# ---------------------------------------------------------------------------
# Reading the saved truth
# ---------------------------------------------------------------------------
def load_epochs(gt_dir: str) -> list[dict]:
    """One dict per EM survey: raw index, ISO date, saved xy / theta / sigma_s, depths0."""
    out = []
    for f in sorted(glob.glob(os.path.join(gt_dir, "em_epoch*_xy.npy"))):
        m = re.match(r"em_epoch(\d+)_(\d{4}-\d{2}-\d{2})_xy\.npy$", os.path.basename(f))
        if not m:
            continue
        stem = f[: -len("_xy.npy")]
        out.append({"idx": int(m.group(1)), "date": m.group(2),
                    "xy": np.load(f), "theta": np.load(stem + "_theta.npy"),
                    "sigma_s": np.load(stem + "_ws_sigma_s.npy"),
                    "depths": np.load(stem + "_depths.npy")})
    if not out:
        raise SystemExit(f"no em_epoch*_xy.npy in {gt_dir}: generate the mirror first "
                         "(python generate_synthetic_real_mirror.py --help)")
    return sorted(out, key=lambda e: e["idx"])


def detect_ws_mode(epochs: list[dict]) -> str:
    """The --ws-truth mode that reproduces the saved sigma_s (so the grid matches the data)."""
    gen = _gen()
    best, best_err = gen.WS_TRUTH_MODE, np.inf
    for mode in gen.WS_TRUTH_MODES:
        gen.configure_ws_truth(mode)
        err = max(float(np.max(np.abs(
            gen.ws_sigma_s_truth(e["xy"][:, :1], e["xy"][:, 1:], np.arange(e["sigma_s"].shape[1])[None, :])
            - e["sigma_s"]))) for e in epochs)
        if err < best_err:
            best, best_err = mode, err
    gen.configure_ws_truth(best)
    if best_err > 1e-6:
        print(f"[warn] no --ws-truth mode reproduces the saved sigma_s (best '{best}', max diff {best_err:.3g})")
    return best


def check_against_saved(epochs: list[dict]) -> float:
    """Max |analytic theta - saved theta| over every epoch / EM point / layer."""
    gen = _gen()
    worst = 0.0
    for e in epochs:
        x, y = e["xy"][:, 0], e["xy"][:, 1]
        for l in range(e["theta"].shape[1]):
            worst = max(worst, float(np.max(np.abs(
                gen.theta_truth_deterministic(x, y, l, e["date"]) - e["theta"][:, l]))))
    return worst


def read_stations(path: str) -> np.ndarray | None:
    if not path or not os.path.isfile(path):
        return None
    d = np.genfromtxt(path, delimiter=";", skip_header=1, usecols=(1, 2))
    return np.atleast_2d(d)


def layer_labels(depths: np.ndarray) -> list[str]:
    edges = [0.0, *[float(d) for d in depths], np.inf]
    return [f"L{i}  {a:g}-{'inf' if np.isinf(b) else f'{b:g}'} m" for i, (a, b) in enumerate(zip(edges, edges[1:]))]


def pick_epochs(epochs: list[dict], wanted: list[int] | None) -> list[dict]:
    gen = _gen()
    if wanted:
        by_idx = {e["idx"]: e for e in epochs}
        miss = [i for i in wanted if i not in by_idx]
        if miss:
            raise SystemExit(f"epochs {miss} not found; available: {sorted(by_idx)}")
        return [by_idx[i] for i in wanted]
    off = [gen.seasonal_offset(e["date"]) for e in epochs]
    lo, hi = int(np.argmin(off)), int(np.argmax(off))
    return [epochs[lo], epochs[hi]] if lo != hi else [epochs[lo]]


def footprint(idx: int):
    gen = _gen()
    for rec in gen.REAL_EM_EPOCHS:
        if rec[0] == idx:
            return rec[4], rec[5], rec[6], rec[7]
    return None


def ws_text(mode: str) -> str:
    gen = _gen()
    return (f"WS truth: a = {gen.WS_A:g}, m = {gen.WS_M:g}, sigma_w = {gen.WS_SIGMA_W:g} mS/m "
            f"(constant everywhere);  sigma_s: '{mode}' (panel on the right)")


def sigma_range(smin: float, smax: float) -> tuple[float, float]:
    """Colour range for sigma_s. A constant field gets 0 .. 2x its value, so that it is
    drawn as flat and not as structure around a +-1 range."""
    if smax - smin < 1e-9:
        return 0.0, max(2.0 * smax, 1.0)
    return smin, smax


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_xy(epochs_sel, depths, ws_mode, stations, out, nx, ny):
    gen = _gen()
    x0, x1, y0, y1 = gen._X0, gen._X1, gen._Y0, gen._Y1
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys)
    nl = len(depths) + 1
    th = {(e["idx"], l): gen.theta_truth_deterministic(X, Y, l, e["date"])
          for e in epochs_sel for l in range(nl)}
    sg = {l: gen.ws_sigma_s_truth(X, Y, l) for l in range(nl)}
    tmin, tmax = min(v.min() for v in th.values()), max(v.max() for v in th.values())
    smin, smax = min(v.min() for v in sg.values()), max(v.max() for v in sg.values())
    smin, smax = sigma_range(smin, smax)

    ncol = len(epochs_sel) + 1
    fig, axes = plt.subplots(nl, ncol, figsize=(3.6 * ncol, 3.1 * nl), squeeze=False, sharex=True, sharey=True)
    labels = layer_labels(depths)
    ext = [0, x1 - x0, 0, y1 - y0]
    for l in range(nl):
        for c, e in enumerate(epochs_sel):
            ax = axes[l, c]
            im_t = ax.imshow(th[(e["idx"], l)], origin="lower", extent=ext, vmin=tmin, vmax=tmax,
                             cmap=THETA_CMAP, aspect="equal")
            fp = footprint(e["idx"])
            if fp:
                ax.plot([fp[0] - x0, fp[1] - x0, fp[1] - x0, fp[0] - x0, fp[0] - x0],
                        [fp[2] - y0, fp[2] - y0, fp[3] - y0, fp[3] - y0, fp[2] - y0], "k--", lw=0.8)
            if l == 0:
                ax.set_title(f"theta, survey {e['idx']} ({e['date']})", fontsize=10)
        ax = axes[l, -1]
        im_s = ax.imshow(sg[l], origin="lower", extent=ext, vmin=smin, vmax=smax, cmap=SIGMA_CMAP, aspect="equal")
        if l == 0:
            ax.set_title("sigma_s truth [mS/m]", fontsize=10)
        axes[l, 0].set_ylabel(f"{labels[l]}\ny - y0 [m]")
    for ax in axes.ravel():
        if stations is not None:
            ax.plot(stations[:, 0] - x0, stations[:, 1] - y0, "k^", ms=3.5, mfc="w", mew=0.7)
    for ax in axes[-1]:
        ax.set_xlabel("x - x0 [m]")
    fig.colorbar(im_t, ax=axes[:, :-1], shrink=0.8, pad=0.02, label="theta [m3/m3]")
    fig.colorbar(im_s, ax=axes[:, -1], shrink=0.8, pad=0.02, label="sigma_s [mS/m]")
    fig.suptitle("Synthetic mirror: TRUTH, plan view (dashed = EM footprint of the survey, "
                 "triangles = TDR stations)\n" + ws_text(ws_mode), fontsize=9)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_xz(epochs_sel, depths, ws_mode, stations, y_sec, band, zmax, out, nx, nz):
    gen = _gen()
    x0, x1 = gen._X0, gen._X1
    xs = np.linspace(x0, x1, nx)
    zs = np.linspace(0.0, zmax, nz)
    layer_of_z = np.searchsorted(np.asarray(depths, dtype=float), zs, side="right")   # 0 .. nl-1
    nl = len(depths) + 1
    Y = np.full_like(xs, y_sec)

    def section(fn):
        cols = [fn(xs, Y, l) for l in range(nl)]            # nl arrays of length nx
        return np.stack([cols[l] for l in layer_of_z], axis=0)   # (nz, nx)

    th = {e["idx"]: section(lambda x, y, l, d=e["date"]: gen.theta_truth_deterministic(x, y, l, d))
          for e in epochs_sel}
    sg = section(lambda x, y, l: gen.ws_sigma_s_truth(x, y, l))
    tmin, tmax = min(v.min() for v in th.values()), max(v.max() for v in th.values())
    smin, smax = sigma_range(float(sg.min()), float(sg.max()))

    ncol = len(epochs_sel) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.4 * ncol, 3.6), squeeze=False, sharex=True, sharey=True)
    axes = axes[0]
    ext = [0, x1 - x0, zmax, 0]
    for c, e in enumerate(epochs_sel):
        im_t = axes[c].imshow(th[e["idx"]], aspect="auto", extent=ext, vmin=tmin, vmax=tmax, cmap=THETA_CMAP)
        axes[c].set_title(f"theta, survey {e['idx']} ({e['date']})", fontsize=10)
    im_s = axes[-1].imshow(sg, aspect="auto", extent=ext, vmin=smin, vmax=smax, cmap=SIGMA_CMAP)
    axes[-1].set_title("sigma_s truth [mS/m]", fontsize=10)
    for ax in axes:
        for d in depths:
            ax.axhline(d, color="w", lw=1.0)
        for d in TDR_DEPTHS_M:
            if d <= zmax:
                ax.axhline(d, color="k", lw=0.7, ls=":")
        if stations is not None:
            near = np.abs(stations[:, 1] - y_sec) <= band
            ax.plot(stations[near, 0] - x0, np.zeros(int(near.sum())), "k^", ms=5, mfc="w", clip_on=False)
        ax.set_xlabel("x - x0 [m]")
    axes[0].set_ylabel("depth [m]")
    fig.colorbar(im_t, ax=axes[:-1].tolist(), shrink=0.9, pad=0.02, label="theta [m3/m3]")
    fig.colorbar(im_s, ax=axes[-1], shrink=0.9, pad=0.04, label="sigma_s [mS/m]")
    fig.suptitle(f"Synthetic mirror: TRUTH, vertical section at northing y = {y_sec:.0f} m "
                 f"(white = layer interfaces, dotted = TDR depths,\ntriangles = stations within +-{band:g} m)\n"
                 + ws_text(ws_mode), fontsize=9, y=1.17)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out



def fig_ws_params(ws_mode, depths, stations, out, nx, ny):
    """All four Waxman-Smits parameters (a, m, sigma_w, sigma_s), plan view, one row per layer.
    a, m, sigma_w are constant in the mirror (flat panels, value in the title); sigma_s may vary."""
    gen = _gen()
    x0, x1, y0, y1 = gen._X0, gen._X1, gen._Y0, gen._Y1
    X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    nl = len(depths) + 1
    names = [("a", "a [-]"), ("m", "m [-]"), ("sigma_w", "sigma_w [mS/m]"), ("sigma_s", "sigma_s [mS/m]")]

    def field(key, l):
        if key == "sigma_s":
            return gen.ws_sigma_s_truth(X, Y, l)
        return np.full_like(X, {"a": gen.WS_A, "m": gen.WS_M, "sigma_w": gen.WS_SIGMA_W}[key])

    fig, axes = plt.subplots(nl, 4, figsize=(3.4 * 4, 2.9 * nl), squeeze=False, sharex=True, sharey=True)
    labels = layer_labels(depths)
    ext = [0, x1 - x0, 0, y1 - y0]
    for c, (key, label) in enumerate(names):
        fields = [field(key, l) for l in range(nl)]
        vmin, vmax = sigma_range(min(f.min() for f in fields), max(f.max() for f in fields))
        const = max(f.max() for f in fields) - min(f.min() for f in fields) < 1e-9
        for l in range(nl):
            ax = axes[l, c]
            im = ax.imshow(fields[l], origin="lower", extent=ext, vmin=vmin, vmax=vmax,
                           cmap=SIGMA_CMAP, aspect="equal")
            if stations is not None:
                ax.plot(stations[:, 0] - x0, stations[:, 1] - y0, "k^", ms=3, mfc="w", mew=0.6)
            if l == 0:
                ax.set_title(f"{label}\n" + (f"constant = {fields[0].flat[0]:g}" if const else "varies in space"),
                             fontsize=10)
        fig.colorbar(im, ax=axes[:, c], shrink=0.8, pad=0.02)
    for l in range(nl):
        axes[l, 0].set_ylabel(f"{labels[l]}\ny - y0 [m]")
    for ax in axes[-1]:
        ax.set_xlabel("x - x0 [m]")
    fig.suptitle("Synthetic mirror: TRUTH of the Waxman-Smits parameters (constant panels are flat by "
                 f"construction;  sigma_s mode: '{ws_mode}')", fontsize=9)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_truth_maps(gt_dir: str, out_dir: str, coord_file: str | None = None,
                    epochs: list[int] | None = None, y_sec: float | None = None,
                    band: float = 15.0, zmax: float = 1.0, nx: int = 150, ny: int = 150,
                    log=print) -> list[str]:
    """Write truth_xy.png, truth_xz.png and truth_ws_params.png into ``out_dir``; return the paths.

    gt_dir     ``ground_truth/`` folder of the mirror
    coord_file TDR station coordinates (``Pos;x;y``) to overlay; None / missing = no overlay
    epochs     survey indices (1-7) to draw; default the driest and the wettest
    y_sec      northing (m) of the xz section; default the field centre
    """
    gen = _gen()
    keep = (list(gen.DEPTHS0), gen.WS_TRUTH_MODE)          # the generator's globals: restored below
    try:
        eps = load_epochs(gt_dir)
        depths = eps[0]["depths"]
        gen.configure_layers(depths)
        ws_mode = detect_ws_mode(eps)
        diff = check_against_saved(eps)
        flag = "OK" if diff < 1e-9 else "MISMATCH: the saved truth was made with other settings"
        log(f"[truth] {len(eps)} surveys, layers {layer_labels(depths)}, sigma_s truth mode '{ws_mode}'; "
            f"grid vs saved theta at the EM points: max |diff| = {diff:.2e} ({flag})")
        sel = pick_epochs(eps, epochs)
        stations = read_stations(coord_file) if coord_file else None
        y_use = y_sec if y_sec is not None else 0.5 * (gen._Y0 + gen._Y1)
        os.makedirs(out_dir, exist_ok=True)
        paths = [
            fig_xy(sel, depths, ws_mode, stations, os.path.join(out_dir, "truth_xy.png"), nx, ny),
            fig_xz(sel, depths, ws_mode, stations, y_use, band, zmax, os.path.join(out_dir, "truth_xz.png"), nx, 200),
            fig_ws_params(ws_mode, depths, stations, os.path.join(out_dir, "truth_ws_params.png"), nx, ny),
        ]
        log(f"[truth] surveys drawn: {[e['idx'] for e in sel]}")
        return paths
    finally:
        gen.configure_layers(keep[0])
        gen.configure_ws_truth(keep[1])
