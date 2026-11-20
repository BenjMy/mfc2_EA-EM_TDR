"""Export the final SWC maps of a run to one NetCDF file (``output_model/maps.nc``).

Why: the figures of a run are PNGs, so scenarios cannot be compared on one shared colour
scale afterwards. ``maps.nc`` holds the numbers (theta, its uncertainty, the global-WS map,
the coordinates) so ``plot_run_log.py --maps`` can redraw several runs side by side.

Layout (dims ``time`` (1 for a single survey), ``point`` (EM measurement points), ``layer``):

    theta(time, point, layer)           m3/m3, the optimal map (localized WS when active)
    theta_global(time, point, layer)    optional, global-WS map
    theta_std_total / _ec / _ws         optional, sigma of theta (Step 6f)
    x(point), y(point)                  coordinates [m] (UTM in 2-D, x along the transect in 1-D, y = 0)
    layer_depth(layer)                  layer centre [m], when known
    attrs: scenario, run_id, mode, rmse_m3m3, rmse_pct, rsr, rating, K, units

Pure numpy + xarray (optional dependency: ``export_maps`` returns None and says why when it
is missing, so a run is never stopped by it). Tests: tests/test_maps_export.py.
"""
from __future__ import annotations

import os

import numpy as np

MAPS_NAME = "maps.nc"
OPTIONAL_VARS = ("theta_global", "theta_std_total", "theta_std_ec", "theta_std_ws")


def _as3d(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim == 2:                     # (point, layer): single survey
        a = a[None]
    if a.ndim != 3:
        raise ValueError(f"expected (time, point, layer) or (point, layer), got shape {a.shape}")
    return a


def export_maps(path: str, theta, x, y=None, layer_depth=None, extra: dict | None = None,
                attrs: dict | None = None) -> str | None:
    """Write the NetCDF file; returns the path, or None (with a message) when it cannot."""
    try:
        import xarray as xr
    except ImportError:
        print("[MAPS] xarray not installed: maps.nc not written (pip install xarray netcdf4)")
        return None
    try:
        th = _as3d(theta)
        nt, npt, nl = th.shape
        x = np.asarray(x, dtype=float).ravel()
        y = np.zeros(npt) if y is None else np.asarray(y, dtype=float).ravel()
        if x.size != npt or y.size != npt:
            raise ValueError(f"{x.size} x / {y.size} y coordinates for {npt} points")
        dims = ("time", "point", "layer")
        data = {"theta": (dims, th, {"units": "m3/m3", "long_name": "optimal volumetric water content"})}
        for k, v in (extra or {}).items():
            if v is None:
                continue
            arr = np.asarray(v, dtype=float)
            if arr.size == th.size:
                data[k] = (dims, arr.reshape(th.shape), {"units": "m3/m3"})
        coords = {"x": ("point", x, {"units": "m"}), "y": ("point", y, {"units": "m"})}
        if layer_depth is not None and np.size(layer_depth) == nl:
            coords["layer_depth"] = ("layer", np.asarray(layer_depth, dtype=float), {"units": "m"})
        ds = xr.Dataset(data, coords=coords)
        ds.attrs.update({k: (v if isinstance(v, (int, float, str)) else str(v))
                         for k, v in (attrs or {}).items() if v is not None})
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ds.to_netcdf(path)
        return path
    except Exception as exc:  # noqa: BLE001 - housekeeping must not stop a run
        print(f"[MAPS] WARNING: could not write {path!r}: {exc}")
        return None


def load_maps(path: str):
    """Open a ``maps.nc`` as an in-memory xarray.Dataset (None when absent / unreadable)."""
    try:
        import xarray as xr
        with xr.open_dataset(path) as ds:
            return ds.load()
    except Exception:  # noqa: BLE001
        return None


def collect_extra(output_model_dir: str, shape) -> dict:
    """The optional arrays a run saved next to theta_calc.npy, when their shape matches."""
    out = {}
    for name in OPTIONAL_VARS:
        p = os.path.join(output_model_dir, f"{name}.npy")
        if os.path.isfile(p):
            try:
                a = np.load(p)
                if a.size == int(np.prod(shape)):
                    out[name] = a
            except (OSError, ValueError):
                pass
    return out
