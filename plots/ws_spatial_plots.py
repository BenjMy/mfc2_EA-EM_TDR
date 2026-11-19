"""plots/ws_spatial_plots.py -- maps for the per-station Waxman-Smits calibration (to-do item E).

One figure, one column per horizon:

* top row: the sigma_s field used by Steps 5 / 6 (``mapping.params``, per EM point) with the
  stations drawn on top (same colour scale); a station whose fit sits on a bound gets a red
  cross, a station that was not fitted (too few observations) a hollow marker;
* bottom row: the per-station sigma_s (sorted by x) against the pooled value (dashed line),
  so "the stations differ from the pooled fit" can be read directly.

Pure matplotlib; takes a ``ws_spatial.SpatialMapping``.
"""
from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt  # noqa: E402

from plots._common import ensure_dir  # noqa: E402

SIGMA_S = 3     # index of sigma_s in ``params`` (a, m, sigma_w, sigma_s)


def plot_station_params(mapping, point_x, point_y, path: str, *, param: int = SIGMA_S,
                        label: str = "sigma_s (mS/m)") -> str:
    """Write the figure to ``path`` and return it. ``point_y`` None = 1-D transect (y = 0)."""
    sf = mapping.station_fits
    if sf is None:
        raise ValueError("mapping has no station_fits (not a station-based mapping)")
    px = np.asarray(point_x, dtype=float)
    py = np.zeros_like(px) if point_y is None else np.asarray(point_y, dtype=float)
    n_layers = mapping.params.shape[1]
    field = mapping.params[param]                              # (n_layers, n_points)
    st_vals = sf.params[param]                                 # (n_layers, n_stations)
    pooled = None if mapping.pooled is None else np.asarray(mapping.pooled)[:, param]

    pool = [field[np.isfinite(field)], st_vals[sf.valid]]
    if pooled is not None:
        pool.append(pooled[np.isfinite(pooled)])
    allv = np.concatenate(pool)
    vmin, vmax = (float(allv.min()), float(allv.max())) if allv.size else (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, axes = plt.subplots(2, n_layers, figsize=(4.8 * n_layers, 7.6), squeeze=False)
    order = np.argsort(sf.x, kind="stable")
    for layer in range(n_layers):
        ax = axes[0, layer]
        sc = ax.scatter(px, py, c=field[layer], s=22, cmap="viridis", vmin=vmin, vmax=vmax,
                        marker="s", linewidths=0)
        ok = sf.valid[layer]
        ax.scatter(sf.x[ok], sf.y[ok], c=st_vals[layer][ok], s=110, cmap="viridis", vmin=vmin,
                   vmax=vmax, edgecolors="k", linewidths=1.2, zorder=3)
        if (~ok).any():
            ax.scatter(sf.x[~ok], sf.y[~ok], s=110, facecolors="none", edgecolors="0.4",
                       linewidths=1.2, zorder=3, label="not fitted")
        bd = sf.at_bound[layer]
        if bd.any():
            ax.scatter(sf.x[bd], sf.y[bd], marker="x", s=70, c="red", zorder=4, label="on a bound")
        if (~ok).any() or bd.any():
            ax.legend(loc="best", fontsize=8)
        ax.set_title(f"Horizon {layer}: {getattr(mapping, 'method', '')} field")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        fig.colorbar(sc, ax=ax, label=label, shrink=0.8)

        ax2 = axes[1, layer]
        idx = [i for i in order if ok[i]]
        ax2.bar(np.arange(len(idx)), [st_vals[layer][i] for i in idx], color="tab:blue", alpha=0.85)
        ax2.set_xticks(np.arange(len(idx)))
        ax2.set_xticklabels([sf.stations[i] for i in idx], rotation=90, fontsize=7)
        if pooled is not None:
            ax2.axhline(pooled[layer], color="k", ls="--", lw=1.2, label="pooled")
            ax2.legend(loc="best", fontsize=8)
        ax2.set_ylim(min(0.0, vmin), vmax * 1.05)
        ax2.set_xlabel("station (sorted by x)")
        ax2.set_ylabel(label)
        ax2.set_title(f"Horizon {layer}: fitted stations, median n_obs "
                      f"{int(np.median(sf.n_obs[layer]))}")
    fig.tight_layout()
    ensure_dir(os.path.dirname(path) or ".")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
