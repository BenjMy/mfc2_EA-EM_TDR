"""EC of a calibrated horizon derived from TDR (Level 2 of analysis_pinned_ceiling.py).

Pass 1 of the pinned-horizon experiment calibrates Waxman-Smits (WS) on the TDR
probes. This module turns that calibration into a per-EM-point bulk EC for one
horizon, which pass 2 then pins in the Step 1 inversion (``pin_layers`` +
``pin_source``):

    TDR theta at the probes --(inverse WS, calibrated params)--> EC_bulk at the probes
    --(interpolation along x, or over x-y)--> EC_bulk at every EM point

Pure numpy/scipy, no pipeline import. The inverse of
``em_methodology.waxman_smits_theta``:  sigma_b = (sigma_w + sigma_s) * theta**m / a.
"""
from __future__ import annotations

import numpy as np


def ec_from_theta(theta, a: float, m: float, sigma_w: float, sigma_s: float):
    """Inverse Waxman-Smits: volumetric water content -> bulk EC [mS/m]."""
    theta = np.clip(np.asarray(theta, dtype=float), 1e-6, None)
    return (sigma_w + sigma_s) * theta ** m / a


def _interp_1d(x_obs, v_obs, x_pts):
    order = np.argsort(x_obs)
    return np.interp(x_pts, x_obs[order], v_obs[order])    # constant beyond the ends


def _interp_2d(xy_obs, v_obs, xy_pts):
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    near = NearestNDInterpolator(xy_obs, v_obs)(xy_pts)
    if len(v_obs) < 3:
        return near
    try:
        lin = LinearNDInterpolator(xy_obs, v_obs)(xy_pts)
    except Exception:                       # collinear probes: no triangulation
        return near
    return np.where(np.isnan(lin), near, lin)


def derive_pin_ec(theta_obs, obs_positions, params, point_x, point_y=None,
                  obs_times=None, n_time: int = 1, ec_bounds=None):
    """Bulk EC of ONE horizon at every EM point, from that horizon's TDR data.

    Parameters
    ----------
    theta_obs : (n_obs,) TDR theta of this horizon (train observations only).
    obs_positions : (n_obs,) index of the EM survey point of each probe.
    params : (a, m, sigma_w, sigma_s) calibrated for this horizon.
    point_x, point_y : (n_pts,) coordinates of the EM points (``point_y`` None
        = 1-D transect: interpolate along x only).
    obs_times : (n_obs,) epoch index of each observation (None = one epoch).
    n_time : number of epochs of the result.
    ec_bounds : optional (lo, hi) mS/m; the result is clipped to it so the pin
        stays inside the inversion's conductivity range.

    Returns
    -------
    (n_time, n_pts) array. Several probes at one position, and several
    observations of one probe in one epoch, are averaged in EC space. An epoch
    without observation takes the nearest epoch that has some.
    """
    theta_obs = np.asarray(theta_obs, dtype=float)
    obs_positions = np.asarray(obs_positions, dtype=int)
    point_x = np.asarray(point_x, dtype=float)
    if theta_obs.size == 0:
        raise ValueError("derive_pin_ec: no TDR observation for this horizon "
                         "(an uncalibrated horizon cannot be pinned to TDR).")
    obs_times = (np.zeros(len(theta_obs), dtype=int) if obs_times is None
                 else np.asarray(obs_times, dtype=int))
    ec_obs = ec_from_theta(theta_obs, *params)

    out = np.full((n_time, len(point_x)), np.nan)
    have = sorted(set(obs_times.tolist()))
    for t in range(n_time):
        t_src = t if t in have else min(have, key=lambda h: abs(h - t))
        sel = obs_times == t_src
        pos = np.unique(obs_positions[sel])
        v = np.array([ec_obs[sel & (obs_positions == p)].mean() for p in pos])
        if point_y is None:
            out[t] = _interp_1d(point_x[pos], v, point_x)
        else:
            point_y_arr = np.asarray(point_y, dtype=float)
            xy = np.column_stack([point_x, point_y_arr])
            out[t] = _interp_2d(xy[pos], v, xy)
    if ec_bounds is not None:
        out = np.clip(out, ec_bounds[0], ec_bounds[1])
    return out
