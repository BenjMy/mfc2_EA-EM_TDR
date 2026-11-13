"""
em_subsample.py
===============
Optional thinning of the (very large) real EM surveys, for testing and for the
"calibrate on probe points only" stage. Pure pandas/numpy: no emagpy import.

Scenario field ``em_subsample`` (``null`` or ``{"mode": "off"}`` = untouched)::

    {"mode": "grid",   "cell_m": 3.0}      # one point per cell_m x cell_m cell
    {"mode": "probes", "radius_m": 5.0}    # only points within radius_m of a TDR probe
    {"mode": "stride", "stride": 10}       # every 10th point
    {"mode": "random", "max_points": 2000, "seed": 0}
    # common optional key: "keep_probe_nearest": true (default) -> the EM point
    # nearest to every TDR probe is always kept, whatever the mode.

Where it runs: ``run_pipeline._load_real_data``, once per survey file, AFTER
``data_real.preprocess_em_survey`` (the rolling mean needs the full sequence)
and BEFORE the inversion / ``geometry_utils.match_epochs_to_reference_grid``.
The processed CSV (regenerated from the raw file on every run) is rewritten
in place, rows keep their original order.

``grid`` snaps to a fixed absolute grid (origin 0 in the survey CRS) and keeps
the point closest to each cell centre, so the SAME cells are chosen in every
epoch and the epochs still overlap; random thinning would not. Keep ``cell_m``
below ``spatial_match_tolerance_m``.

Note: with a lateral-smoothing ``beta > 0`` thinning changes the neighbourhoods
the smoothing acts on; for ``beta = 0`` (default) points are independent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODES = ("off", "grid", "probes", "stride", "random")
_REQUIRED = {"grid": "cell_m", "probes": "radius_m", "stride": "stride", "random": "max_points"}
_KEYS = {"mode", "cell_m", "radius_m", "stride", "max_points", "seed", "keep_probe_nearest"}


def validate_em_subsample(cfg, name: str = "?") -> dict | None:
    """Return a normalised dict (or None when off); raise ValueError on bad input."""
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise ValueError(f"Scenario '{name}': em_subsample must be null or an object, got {cfg!r}.")
    unknown = sorted(set(cfg) - _KEYS)
    if unknown:
        raise ValueError(f"Scenario '{name}': em_subsample has unknown key(s) {unknown}; allowed: {sorted(_KEYS)}.")
    mode = cfg.get("mode", "off")
    if mode not in MODES:
        raise ValueError(f"Scenario '{name}': em_subsample.mode must be one of {MODES}, got {mode!r}.")
    if mode == "off":
        return None
    key = _REQUIRED[mode]
    val = cfg.get(key)
    if not isinstance(val, (int, float)) or isinstance(val, bool) or val <= 0:
        raise ValueError(f"Scenario '{name}': em_subsample.mode='{mode}' needs a positive number '{key}', got {val!r}.")
    if mode in ("stride", "random") and int(val) != val:
        raise ValueError(f"Scenario '{name}': em_subsample.{key} must be an integer, got {val!r}.")
    out = {"mode": mode, key: val, "keep_probe_nearest": bool(cfg.get("keep_probe_nearest", True))}
    if mode == "random":
        out["seed"] = int(cfg.get("seed", 0))
    return out


def select_indices(x, y, cfg: dict, probe_x=None, probe_y=None) -> np.ndarray:
    """Sorted row positions to keep for points (x, y) under a validated ``cfg``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    mode = cfg["mode"]
    have_probes = probe_x is not None and len(probe_x) > 0
    if have_probes:
        px, py = np.asarray(probe_x, dtype=float), np.asarray(probe_y, dtype=float)

    if mode == "grid":
        cell = float(cfg["cell_m"])
        ix, iy = np.floor(x / cell).astype(np.int64), np.floor(y / cell).astype(np.int64)
        d2 = (x - (ix + 0.5) * cell) ** 2 + (y - (iy + 0.5) * cell) ** 2
        order = np.argsort(d2, kind="stable")                       # closest to centre first
        _, first = np.unique(np.stack([ix[order], iy[order]], axis=1), axis=0, return_index=True)
        keep = order[first]
    elif mode == "probes":
        if not have_probes:
            raise ValueError("em_subsample.mode='probes' needs the TDR probe coordinates.")
        r = float(cfg["radius_m"])
        dmin = np.sqrt(((x[:, None] - px[None, :]) ** 2 + (y[:, None] - py[None, :]) ** 2)).min(axis=1)
        keep = np.flatnonzero(dmin <= r)
    elif mode == "stride":
        keep = np.arange(0, n, int(cfg["stride"]))
    elif mode == "random":
        m = min(int(cfg["max_points"]), n)
        keep = np.random.RandomState(int(cfg.get("seed", 0))).choice(n, size=m, replace=False)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if cfg.get("keep_probe_nearest", True) and have_probes:
        nearest = np.sqrt((x[:, None] - px[None, :]) ** 2 + (y[:, None] - py[None, :]) ** 2).argmin(axis=0)
        keep = np.concatenate([keep, nearest])
    keep = np.unique(keep)
    if keep.size == 0:
        raise ValueError("em_subsample kept 0 points; check its parameters.")
    return keep


def subsample_em_csv(csv_path: str, cfg: dict, probe_x=None, probe_y=None) -> dict:
    """Rewrite the emagpy-ready CSV in place keeping the selected rows; return stats."""
    df = pd.read_csv(csv_path)
    keep = select_indices(df["x"].to_numpy(), df["y"].to_numpy(), cfg, probe_x, probe_y)
    df.iloc[keep].to_csv(csv_path, index=False)
    return {"mode": cfg["mode"], "n_before": int(len(df)), "n_after": int(keep.size)}
