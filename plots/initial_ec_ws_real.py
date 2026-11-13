"""
plots/initial_ec_ws_real.py
===========================
Initial EC range (ECmin/ECmax) and initial Waxman-Smits (WS) values of a REAL-data
run, one subplot per horizon. Real-data counterpart of
``plots/ws_distribution.py`` (synthetic runs have a truth to compare against; a
real run has none, so this module shows the initial values against what the
field data imply).

Real data are X, Y + depth. Both figures therefore use one column per horizon.

Figure 1  ``initial_ec_range_by_horizon.png``  (2 rows x n_horizons)
  row 0  histogram of the EC_bulk implied by the TDR water content of that
         horizon and the INITIAL WS values:
             EC_init = (sigma_w + sigma_s) * theta**m / a
         vertical lines = ECmin / ECmax (``cond_ranges``), grey = outside the range.
  row 1  XY map of the TDR probes, colour = EC_init (time mean), colour scale =
         [ECmin, ECmax] of that horizon, "x" = probe outside the range.
Figure 2  ``initial_ws_by_horizon.png``  (1 row x n_horizons)
  per horizon, the 4 initial WS values (a, m, sigma_w, sigma_s) as a position
  inside their calibration bounds (0 = lower bound, 1 = upper bound), value
  written next to the bar, colour = [fixed] / [free] (``ws_fixed_params``),
  red edge = value on a bound (a free parameter starting on a bound cannot move
  that way, see to-do.md, free4 'a' pile-up).

Inputs (all plain Python / numpy, no import of the core library):
  cond_ranges     [[min, max], ...] per horizon (``cfg.cond_ranges``)
  depths0         interface depths [m]; horizon l lies above depths0[l]
  lab_p0          None | 4-vector | (n_horizons, 4)   (``cfg.lab_p0``)
  ws_fixed_params {param: bool} or None
  ws_bounds       (lo[4], hi[4]) in the ``WS_BOUNDS`` layout, or None for defaults
  observations    list of {probe_index, time_index, depth [m], swc [m3/m3]}
  probe_table     DataFrame with probe_index, station_name, utm_x, utm_y
Output  : the two PNGs in ``<output_dir>/initial_model/``; ``plot_initial_ec_ws_real``
          returns {"ec": path, "ws": path} (an entry is None when it cannot be drawn).
Called by: run_pipeline.py, Step 1d (real runs).
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from plots._common import ensure_dir, save_fig
from plots.ws_distribution import PARAMS, resolve_initial_ws

# Same defaults as em_methodology.WS_BOUNDS (kept literal: no core import).
DEFAULT_WS_BOUNDS = ((0.5, 1.0, 1.0, 0.0), (5.0, 4.0, 1000.0, 200.0))


def horizon_index(depth_m: float, depths0) -> int:
    """Horizon of a depth: 0 above ``depths0[0]``, ..., ``len(depths0)`` below the last."""
    return int(np.searchsorted(np.asarray(depths0, dtype=float), float(depth_m), side="right"))


def horizon_labels(depths0) -> list[str]:
    edges = [0.0] + [float(d) for d in depths0] + [float("inf")]
    return [f"H{l}  {edges[l]:.3g}-" + ("bottom" if np.isinf(edges[l + 1]) else f"{edges[l + 1]:.3g}") + " m"
            for l in range(len(edges) - 1)]


def ws_ec_from_theta(theta, a: float, m: float, sigma_w: float, sigma_s: float):
    """WS bulk EC [mS/m] from theta: (sigma_w + sigma_s) * theta**m / a."""
    return (sigma_w + sigma_s) * np.clip(np.asarray(theta, dtype=float), 0.0, None) ** m / a


def initial_ec_from_tdr(observations, probe_table, depths0, init_ws: np.ndarray):
    """One row per matched TDR observation with the EC implied by the initial WS.

    Returns a dict of equal-length arrays: probe_index, x, y, time_index, depth,
    horizon, theta, ec_init.
    """
    px = dict(zip(probe_table["probe_index"], probe_table["utm_x"]))
    py = dict(zip(probe_table["probe_index"], probe_table["utm_y"]))
    rows = {k: [] for k in ("probe_index", "x", "y", "time_index", "depth", "horizon", "theta", "ec_init")}
    for o in observations:
        h = min(horizon_index(o["depth"], depths0), init_ws.shape[0] - 1)
        a, m, sw, ss = init_ws[h]
        rows["probe_index"].append(o["probe_index"])
        rows["x"].append(px[o["probe_index"]])
        rows["y"].append(py[o["probe_index"]])
        rows["time_index"].append(o["time_index"])
        rows["depth"].append(float(o["depth"]))
        rows["horizon"].append(h)
        rows["theta"].append(float(o["swc"]))
        rows["ec_init"].append(float(ws_ec_from_theta(o["swc"], a, m, sw, ss)))
    return {k: np.asarray(v) for k, v in rows.items()}


def plot_initial_ec_ws_real(output_dir: str, cond_ranges, depths0, lab_p0=None,
                            ws_fixed_params: dict | None = None, ws_bounds=None,
                            observations=None, probe_table=None,
                            subdir: str = "initial_model") -> dict:
    """Draw both figures (see module docstring). Never raises for missing TDR data."""
    out_dir = ensure_dir(os.path.join(output_dir, subdir))
    n_h = len(cond_ranges)
    init = resolve_initial_ws(lab_p0, n_h)
    fixed = ws_fixed_params or {}
    lo, hi = (np.asarray(b, dtype=float) for b in (ws_bounds or DEFAULT_WS_BOUNDS))
    labels = horizon_labels(depths0)
    result = {"ec": None, "ws": None}

    # ------------------------------------------------------------ Figure 1: EC
    have_tdr = bool(observations) and probe_table is not None and len(probe_table) > 0
    d = initial_ec_from_tdr(observations, probe_table, depths0, init) if have_tdr else None
    fig, axes = plt.subplots(2, n_h, figsize=(4.6 * n_h, 8.0), squeeze=False)
    for l in range(n_h):
        rng = cond_ranges[l]
        ecmin, ecmax = (float(rng[0]), float(rng[-1]))
        ax_h, ax_m = axes[0][l], axes[1][l]
        sel = None if d is None else d["horizon"] == l
        n_sel = 0 if sel is None else int(sel.sum())
        ax_h.axvline(ecmin, color="tab:blue", lw=1.6, label=f"ECmin {ecmin:g}")
        ax_h.axvline(ecmax, color="tab:red", lw=1.6, label=f"ECmax {ecmax:g}")
        ax_h.axvspan(ecmin, ecmax, color="0.92", zorder=0)
        title = labels[l] + f"\nstart range [{ecmin:g}, {ecmax:g}] mS/m"
        if n_sel:
            ec = d["ec_init"][sel]
            n_out = int(((ec < ecmin) | (ec > ecmax)).sum())
            ax_h.hist(ec, bins=25, color="tab:green", alpha=0.8, zorder=3,
                      label=f"EC_init from TDR (n={n_sel})")
            title += f"\n{n_out}/{n_sel} TDR-implied EC outside the range"
            probes = np.unique(d["probe_index"][sel])
            ec_p = np.array([d["ec_init"][sel & (d["probe_index"] == p)].mean() for p in probes])
            xs = np.array([d["x"][sel & (d["probe_index"] == p)][0] for p in probes])
            ys = np.array([d["y"][sel & (d["probe_index"] == p)][0] for p in probes])
            sc = ax_m.scatter(xs, ys, c=ec_p, cmap="viridis", vmin=ecmin, vmax=ecmax,
                              s=70, edgecolor="k", linewidth=0.4)
            out = (ec_p < ecmin) | (ec_p > ecmax)
            if out.any():
                ax_m.scatter(xs[out], ys[out], marker="x", c="r", s=70, label="outside range")
                ax_m.legend(fontsize=7, loc="best")
            ax_m.set_aspect("equal", adjustable="datalim")
            fig.colorbar(sc, ax=ax_m, fraction=0.046, pad=0.04, label="EC_init, time mean [mS/m]")
        else:
            ax_m.text(0.5, 0.5, "no TDR probe in this horizon", ha="center", va="center",
                      transform=ax_m.transAxes, fontsize=9)
        ax_h.set_title(title, fontsize=8.5, fontweight="bold")
        ax_h.set_xlabel("EC_bulk [mS/m]")
        ax_h.set_ylabel("count")
        ax_h.legend(fontsize=7, loc="best")
        ax_h.grid(True, ls=":", alpha=0.6)
        ax_m.ticklabel_format(useOffset=False, style="plain")
        ax_m.set_xlabel("UTM Easting [m]")
        ax_m.set_ylabel("UTM Northing [m]")
        ax_m.set_title(f"{labels[l]}  - probe map", fontsize=8.5)
    fig.suptitle("Initial EC range (ECmin/ECmax) vs EC implied by TDR theta and initial WS, per horizon",
                 fontsize=11)
    fig.tight_layout(h_pad=3.0)
    result["ec"] = os.path.join(out_dir, "initial_ec_range_by_horizon.png")
    save_fig(fig, result["ec"], dpi=170, tag="INI-EC")

    # ------------------------------------------------------------- Figure 2: WS
    fig, axes = plt.subplots(1, n_h, figsize=(4.0 * n_h, 3.8), squeeze=False)
    for l in range(n_h):
        ax = axes[0][l]
        for i, p in enumerate(PARAMS):
            v = init[l, i]
            frac = float(np.clip((v - lo[i]) / (hi[i] - lo[i]), 0.0, 1.0))
            on_bound = bool(np.isclose(v, lo[i]) or np.isclose(v, hi[i]))
            is_fixed = bool(fixed.get(p))
            ax.barh(i, frac, color="tab:gray" if is_fixed else "tab:orange",
                    edgecolor="r" if on_bound else "k", linewidth=2.0 if on_bound else 0.6)
            ax.text(frac + 0.02, i, f"{v:g}  [{'fixed' if is_fixed else 'free'}]"
                    + ("  ON BOUND" if on_bound else ""), va="center", fontsize=8)
        ax.set_yticks(range(len(PARAMS)))
        ax.set_yticklabels([f"{p}\n[{lo[i]:g}, {hi[i]:g}]" for i, p in enumerate(PARAMS)], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.6)
        ax.set_xlabel("position inside bounds (0 = lower, 1 = upper)")
        ax.set_title(labels[l], fontsize=9, fontweight="bold")
        ax.grid(True, axis="x", ls=":", alpha=0.6)
    fig.suptitle("Initial Waxman-Smits values per horizon (grey = fixed, orange = free, "
                 "red edge = on a bound)", fontsize=11)
    fig.tight_layout()
    result["ws"] = os.path.join(out_dir, "initial_ws_by_horizon.png")
    save_fig(fig, result["ws"], dpi=170, tag="INI-WS")
    return result
