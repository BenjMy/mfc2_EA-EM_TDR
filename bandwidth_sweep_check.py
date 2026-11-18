"""Spacing checks for the localized-WS bandwidth sweep (to-do item A).

The Gaspari-Cohn kernel (``em_methodology.gaspari_cohn``) has support ``2*c``: a probe
only gets neighbours with non-zero weight if some other probe lies within ``2*c``.
A sweep whose candidates all stay below half the nearest-neighbour spacing therefore
gives the SAME leave-one-probe-out RMSE as the global fit for every c (every fold
falls back to the pooled fit), and cannot detect spatial structure.

Pure numpy; no emagpy needed.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Candidate c values as multiples of the median nearest-neighbour probe distance.
DEFAULT_FACTORS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)


def unique_probe_locations(x: Sequence[float], y: Sequence[float] | None = None,
                           decimals: int = 6) -> np.ndarray:
    """Unique probe locations as an (n, 1) or (n, 2) array (probes repeat per horizon/epoch)."""
    x = np.asarray(x, dtype=float)
    pts = x[:, None] if y is None else np.column_stack([x, np.asarray(y, dtype=float)])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    return np.unique(np.round(pts, decimals), axis=0)


def nearest_neighbour_distances(x: Sequence[float], y: Sequence[float] | None = None) -> np.ndarray:
    """Distance from each unique probe location to its nearest other probe (empty if < 2)."""
    pts = unique_probe_locations(x, y)
    if len(pts) < 2:
        return np.array([])
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def default_bandwidth_sweep(x: Sequence[float], y: Sequence[float] | None = None,
                            factors: Sequence[float] = DEFAULT_FACTORS) -> list[float]:
    """c candidates = median nearest-neighbour probe distance x ``factors`` (rounded to 0.1)."""
    nn = nearest_neighbour_distances(x, y)
    if nn.size == 0:
        raise ValueError("need at least two distinct probe locations to derive a bandwidth sweep")
    med = float(np.median(nn))
    return [round(med * f, 1) for f in factors]


def sweep_spacing_warning(bandwidths: Sequence[float], x: Sequence[float],
                          y: Sequence[float] | None = None) -> str | None:
    """Message if the sweep cannot see neighbours (None if it is fine or cannot be judged).

    reach = 2 * max(c) (Gaspari-Cohn support). Compared with the nearest-neighbour
    distances of the unique probe locations.
    """
    nn = nearest_neighbour_distances(x, y)
    bw = [float(b) for b in bandwidths]
    if nn.size == 0 or not bw:
        return None
    reach = 2.0 * max(bw)
    nn_min, nn_med = float(nn.min()), float(np.median(nn))
    n_with = int(np.sum(nn < reach))
    if reach <= nn_min:
        return (f"every candidate is too small: the largest c={max(bw):g} reaches {reach:g} "
                f"(kernel support 2c) but the nearest probes are {nn_min:g}-{float(nn.max()):g} "
                f"apart (median {nn_med:g}); no probe has a neighbour, so the LOO-CV RMSE equals "
                f"the global fit for every c. Try c up to about {2 * nn_med:g} or more "
                f"(e.g. {default_bandwidth_sweep(x, y)}).")
    if n_with < 0.5 * len(nn):
        return (f"only {n_with} of {len(nn)} probe locations have a neighbour within the "
                f"largest candidate (c={max(bw):g}, reach {reach:g}; median spacing {nn_med:g}); "
                f"the sweep is weak evidence about spatial structure. Consider larger c, "
                f"e.g. {default_bandwidth_sweep(x, y)}.")
    return None
