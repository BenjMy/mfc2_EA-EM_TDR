"""
plots/ws_distribution.py
========================
Waxman-Smits parameter distribution: TRUTH (synthetic ground truth) vs INITIAL
values used by the calibration, per horizon and per position.

What it shows
-------------
One panel per parameter (a, m, sigma_w, sigma_s). Colour = horizon.
  solid line   truth along the transect (H0 widest; overlapping horizons stay visible) (position index, or x [m] in 1-D)
  dashed line  initial value of the calibration for that horizon (``lab_p0``,
               or ``DEFAULT_WS_P0`` when the scenario sets none). It is
               spatially constant: Step 2/4 start every position from it.
  panel title  [fixed] when the parameter is pinned to its initial value
               (``ws_fixed_params``), [free] when the calibration may move it.

Why not ``data_synthetic.plot_ws_truth_xy_maps``: that figure averages over
layers (hides the depth change of the heterogeneous truth) and has no initial
values. It is kept as is; this module is the per-horizon view.

Inputs   : ``<output_dir>/synthetic_ws_truth_{a,m,sigma_w,sigma_s}.npy``
           (ensemble, n_points, n_layers; only written when ws_xy_variation is
           on), or ``ws_truth_params`` {layer: {param: value}} for a truth
           that is uniform in space.
Output   : ``<output_dir>/synthetic_ws_truth_vs_initial.png`` (path returned)
Called by: run_pipeline.py, Step 0 (synthetic runs).

Array shapes: truth ``(n_points, n_layers, 4)``, initial ``(n_layers, 4)``;
last axis order = ``PARAMS``.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from plots._common import ensure_dir, save_fig

PARAMS = ("a", "m", "sigma_w", "sigma_s")
_LABELS = {
    "a": "a [-]",
    "m": "m [-]",
    "sigma_w": "sigma_w [mS/m]",
    "sigma_s": "sigma_s [mS/m]",
}
# Same default as em_methodology.DEFAULT_WS_P0 (kept literal so plotting does
# not import the core library).
DEFAULT_WS_P0 = (1.5, 1.8, 50.0, 5.0)


def resolve_initial_ws(lab_p0, n_layers: int) -> np.ndarray:
    """Initial WS values as ``(n_layers, 4)`` (order ``PARAMS``).

    ``lab_p0`` may be None (-> ``DEFAULT_WS_P0`` for every horizon), a flat
    4-vector (same for every horizon) or an ``(n_layers, 4)`` matrix.
    """
    if lab_p0 is None:
        return np.tile(np.asarray(DEFAULT_WS_P0, dtype=float), (n_layers, 1))
    arr = np.asarray(lab_p0, dtype=float)
    if arr.ndim == 1:
        if arr.shape != (4,):
            raise ValueError(f"lab_p0 must have 4 values, got shape {arr.shape}")
        return np.tile(arr, (n_layers, 1))
    if arr.shape != (n_layers, 4):
        raise ValueError(
            f"lab_p0 must be (n_layers={n_layers}, 4), got shape {arr.shape}"
        )
    return arr


def load_ws_truth(output_dir: str, realization_index: int = 0,
                  ws_truth_params: dict | None = None,
                  n_points: int | None = None) -> np.ndarray | None:
    """Truth WS parameters as ``(n_points, n_layers, 4)``, or None if unknown.

    Uses the saved ``synthetic_ws_truth_*.npy`` arrays when all four exist;
    otherwise builds a spatially uniform truth from ``ws_truth_params``
    (``{layer_index: {param: value}}``; needs ``n_points``).
    """
    paths = [os.path.join(output_dir, f"synthetic_ws_truth_{p}.npy") for p in PARAMS]
    if all(os.path.exists(p) for p in paths):
        return np.stack([np.load(p)[realization_index] for p in paths], axis=-1)
    if ws_truth_params and n_points:
        layers = sorted(ws_truth_params)
        one = np.array([[ws_truth_params[l][p] for p in PARAMS] for l in layers], dtype=float)
        return np.tile(one[None, :, :], (n_points, 1, 1))
    return None


def describe_truth(truth: np.ndarray, tol: float = 1e-9) -> dict:
    """Which parameters vary along x and with depth: ``{param: (varies_x, varies_depth)}``."""
    out = {}
    for i, p in enumerate(PARAMS):
        t = truth[:, :, i]
        out[p] = (bool(np.ptp(t, axis=0).max() > tol), bool(np.ptp(t, axis=1).max() > tol))
    return out


def plot_ws_truth_vs_initial(output_dir: str, lab_p0=None, ws_fixed_params: dict | None = None,
                             ws_truth_params: dict | None = None, realization_index: int = 0,
                             out_name: str = "synthetic_ws_truth_vs_initial.png") -> str | None:
    """Draw truth vs initial WS distribution; return the PNG path (None if no truth)."""
    x_path = os.path.join(output_dir, "synthetic_x_coords.npy")
    x_coords = np.load(x_path) if os.path.exists(x_path) else None
    n_points = None if x_coords is None else len(x_coords)
    truth = load_ws_truth(output_dir, realization_index, ws_truth_params, n_points)
    if truth is None:
        return None

    n_pts, n_layers, _ = truth.shape
    init = resolve_initial_ws(lab_p0, n_layers)
    fixed = ws_fixed_params or {}
    vary = describe_truth(truth)

    y_path = os.path.join(output_dir, "synthetic_y_coords.npy")
    is_1d = x_coords is not None and len(x_coords) == n_pts and (
        not os.path.exists(y_path) or np.ptp(np.load(y_path)) == 0)
    xs = x_coords if is_1d else np.arange(n_pts)
    xlabel = "x [m]" if is_1d else "point index"

    colors = plt.cm.viridis(np.linspace(0.05, 0.9, n_layers))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), squeeze=False)
    for ax, (i, p) in zip(axes[0], enumerate(PARAMS)):
        # Horizons that share a value overlap: draw H0 widest and lowest, the
        # deepest horizon thinnest on top, so every horizon stays visible.
        for l in range(n_layers):
            w = 1.5 + 2.0 * (n_layers - 1 - l)
            ax.plot(xs, truth[:, l, i], "-", color=colors[l], lw=w, alpha=0.9,
                    solid_capstyle="butt", zorder=2 + l, label=f"H{l} truth")
            ax.axhline(init[l, i], ls=(0, (4, 3)), color=colors[l], lw=1.2,
                       zorder=10 + l, label=f"H{l} initial")
        vx, vz = vary[p]
        state = "fixed" if fixed.get(p) else "free"
        ax.set_title(f"{p} [{state}]  truth varies: x={'yes' if vx else 'no'}, "
                     f"depth={'yes' if vz else 'no'}", fontsize=9, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(_LABELS[p])
        ax.grid(True, ls=":", alpha=0.6)
    axes[0][0].legend(fontsize=6, ncol=2, loc="best")
    fig.suptitle("Waxman-Smits parameters: truth (solid) vs calibration initial value (dashed), "
                 "colour = horizon", fontsize=11)
    fig.tight_layout()
    out = os.path.join(ensure_dir(output_dir), out_name)
    save_fig(fig, out, dpi=180, tag="WS")
    return out
