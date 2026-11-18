"""Waxman-Smits truth for the synthetic MFC2 twin: constant, per horizon, gradient, two zones.

Pure numpy (no emagpy), so it can be tested in the fast tier. The generator draws theta, then
converts it to bulk EC with the parameters this module returns at each (x, y, horizon):

    sigma_b = theta**m / a * (sigma_w + sigma_s)          (same form as em_methodology)

Modes (``WSTruth.mode``):

``constant``     one parameter set everywhere (the original generator behaviour);
``per_horizon``  one set per horizon (a, m, sigma_w, sigma_s lists, length n_layers), constant in x/y;
``gradient``     sigma_s varies linearly with the normalised x coordinate from ``sigma_s_lo`` (x0)
                 to ``sigma_s_hi`` (x1), all other parameters at their base value;
``two_zones``    sigma_s = ``sigma_s_lo`` for x_norm < ``split`` and ``sigma_s_hi`` beyond (a
                 vertical zone border, same in all horizons).

x_norm = (x - x0) / (x1 - x0) over the field bounding box passed to ``ws_params``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

MODES = ("constant", "per_horizon", "gradient", "two_zones")
PARAM_NAMES = ("a", "m", "sigma_w", "sigma_s")


@dataclass(frozen=True)
class WSTruth:
    mode: str = "constant"
    a: float = 0.5
    m: float = 3.0
    sigma_w: float = 500.0
    sigma_s: float = 100.0
    # per_horizon: one value per horizon (None -> base value repeated)
    per_horizon: dict = field(default_factory=lambda: {
        "a": [0.5, 0.5, 0.4, 0.35],
        "m": [3.0, 3.0, 3.0, 3.0],
        "sigma_w": [500.0, 500.0, 600.0, 700.0],
        "sigma_s": [100.0, 100.0, 100.0, 100.0],
    })
    sigma_s_lo: float = 0.0
    sigma_s_hi: float = 200.0
    split: float = 0.5

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"WSTruth.mode must be one of {MODES}, got {self.mode!r}")
        if not 0.0 < self.split < 1.0:
            raise ValueError("WSTruth.split must be inside (0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)


def as_truth(spec) -> WSTruth:
    """``WSTruth`` | mode name | dict of fields -> ``WSTruth``."""
    if isinstance(spec, WSTruth):
        return spec
    if isinstance(spec, str):
        return WSTruth(mode=spec)
    if isinstance(spec, dict):
        return WSTruth(**spec)
    raise TypeError(f"cannot build a WSTruth from {type(spec).__name__}")


def ws_params(truth: WSTruth, x, y, layer_idx: int, bbox, n_layers: int) -> dict:
    """Parameters at points (x, y) in horizon ``layer_idx``: dict of arrays shaped like ``x``.

    ``bbox`` = (x0, x1, y0, y1) of the field (sets x_norm).
    """
    x = np.asarray(x, dtype=float)
    ones = np.ones_like(x)
    out = {k: getattr(truth, k) * ones for k in PARAM_NAMES}
    if truth.mode == "constant":
        return out
    if not 0 <= layer_idx < n_layers:
        raise IndexError(f"layer_idx {layer_idx} outside 0..{n_layers - 1}")
    if truth.mode == "per_horizon":
        for k in PARAM_NAMES:
            vals = truth.per_horizon.get(k)
            if vals is None:
                continue
            if len(vals) != n_layers:
                raise ValueError(f"per_horizon['{k}'] has {len(vals)} values, need {n_layers}")
            out[k] = float(vals[layer_idx]) * ones
        return out
    x0, x1 = bbox[0], bbox[1]
    xn = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    if truth.mode == "gradient":
        out["sigma_s"] = truth.sigma_s_lo + (truth.sigma_s_hi - truth.sigma_s_lo) * xn
    else:  # two_zones
        out["sigma_s"] = np.where(xn < truth.split, truth.sigma_s_lo, truth.sigma_s_hi)
    return out


def bulk_ec(theta, params: dict) -> np.ndarray:
    """sigma_b [mS/m] = theta**m / a * (sigma_w + sigma_s), theta clipped to [0.01, 1]."""
    th = np.clip(np.asarray(theta, dtype=float), 0.01, 1.0)
    return (th ** params["m"] / params["a"]) * (params["sigma_w"] + params["sigma_s"])


def params_array(truth: WSTruth, x, y, bbox, n_layers: int) -> np.ndarray:
    """(n_points, n_layers, 4) array [a, m, sigma_w, sigma_s] - saved as ground truth."""
    x = np.asarray(x, dtype=float)
    arr = np.empty((x.size, n_layers, len(PARAM_NAMES)))
    for l in range(n_layers):
        p = ws_params(truth, x, y, l, bbox, n_layers)
        arr[:, l, :] = np.stack([p[k] for k in PARAM_NAMES], axis=-1)
    return arr
