"""ws_spatial.py -- spatially varying Waxman-Smits (WS) calibration on real data.

Implements the numerical part of to-do items A-D and F (see ``to-do.md``):

* **A** ``nearest_neighbour_spacing`` / ``default_bandwidth_sweep`` / ``sweep_below_spacing``:
  the kernel bandwidth sweep is derived from the probe spacing, and a sweep whose candidates
  are all below the nearest-neighbour spacing is flagged (it cannot detect spatial structure:
  no probe has a neighbour inside the kernel, so every LOO-CV equals the global fit).
* **B** ``fit_stations``: one WS fit per station and horizon. Only ``sigma_s`` is free by
  default (about 7 observations per station and horizon identify one parameter, two at most);
  ``a``, ``m``, ``sigma_w`` are held at the pooled value of the horizon, which is also the
  centre of the Gaussian prior (shrinkage towards the pooled fit). Result: ``StationFits``
  (``ws_by_station.csv``) with an at-bound flag.
* **C** ``spread_station_params`` (``voronoi`` | ``idw``) and ``loso_predictions`` /
  ``loso_summary`` / ``choose_method``: compare pooled, kernel, Voronoi and IDW by
  leave-one-STATION-out CV of theta at the held-out station (same metric for all) and pick the
  simplest method not worse than the best by more than one standard error.
* **D** ``build_station_mapping``: the per-point parameters ``(4, n_layers, n_points)`` of a
  station-based method, in the same container as the kernel mapping
  (``localized_mapping.LocalizedMapping``), so Step 5 / Step 6 need no other change.
* **F** ``probe_error_records`` / ``write_probe_errors_csv`` / ``error_metrics``: signed error
  ``theta_calc - theta_TDR`` at every probe observation, in-sample (final calibration) and
  held-out (leave-one-station-out).

Pure numpy; the optimiser is the one of the pooled fit
(``em_methodology._fit_ws_horizon_weighted``), so a station fit uses the same objective,
priors and bounds as the pooled fit. Unit-tested in ``tests/test_ws_spatial.py``.

Method order used for "simplest" (C): pooled < voronoi < idw < kernel.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

import em_methodology as EMmethod
import localized_mapping as lmap

PARAMS = ("a", "m", "sigma_w", "sigma_s")
SPATIAL_CHOICES = ("pooled", "kernel", "voronoi", "idw")
STATION_METHODS = ("voronoi", "idw")
SIMPLICITY_ORDER = ("pooled", "voronoi", "idw", "kernel")
SWEEP_FACTORS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)     # x median nearest-neighbour spacing
MIN_STATION_OBS = 3


# ======================================================================== A: bandwidth sweep
def nearest_neighbour_spacing(x: Sequence[float], y: Sequence[float] | None = None) -> np.ndarray:
    """Distance of every location to its nearest OTHER location (duplicates removed first)."""
    x = np.asarray(x, dtype=float)
    pts = np.column_stack([x, np.zeros_like(x) if y is None else np.asarray(y, dtype=float)])
    pts = np.unique(pts, axis=0)
    if len(pts) < 2:
        return np.array([], dtype=float)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def default_bandwidth_sweep(x: Sequence[float], y: Sequence[float] | None = None,
                            factors: Sequence[float] = SWEEP_FACTORS) -> list[float]:
    """Candidate bandwidths = median nearest-neighbour probe distance x ``factors``.

    Returns ``[]`` when fewer than two distinct probe locations exist (nothing to derive).
    Values are rounded to 0.1 m and de-duplicated.
    """
    nn = nearest_neighbour_spacing(x, y)
    if nn.size == 0:
        return []
    med = float(np.median(nn))
    return sorted({round(med * float(f), 1) for f in factors if med * float(f) > 0})


def sweep_below_spacing(bandwidths: Sequence[float], x: Sequence[float],
                        y: Sequence[float] | None = None) -> tuple[bool, float]:
    """``(all_below, median_spacing)``: True when EVERY candidate is <= the median NN spacing.

    A kernel with c <= spacing has no neighbouring probe inside its support (2c), so the
    leave-one-probe-out CV cannot tell it from the global fit.
    """
    nn = nearest_neighbour_spacing(x, y)
    if nn.size == 0 or len(bandwidths) == 0:
        return False, float("nan")
    med = float(np.median(nn))
    return bool(all(float(b) <= med for b in bandwidths)), med


# ======================================================================== observations
@dataclass
class ProbeObs:
    """Probe observations paired with the EC of the model at the probe (one row each)."""
    station: np.ndarray      # (n,) object, station identity
    pos: np.ndarray          # (n,) int, EM point index of the probe
    time: np.ndarray         # (n,) int, epoch index
    layer: np.ndarray        # (n,) int, horizon
    theta: np.ndarray        # (n,) float, TDR theta
    ec: np.ndarray           # (n,) float, EC of the model at (time, pos, layer)
    x: np.ndarray            # (n,) float, x of the EM point
    y: np.ndarray            # (n,) float, y of the EM point (0 when 1-D)
    weight: np.ndarray       # (n,) float, base weight (mean 1 per call)

    def __len__(self) -> int:
        return len(self.theta)

    def subset(self, mask) -> "ProbeObs":
        m = np.asarray(mask)
        return ProbeObs(*(getattr(self, f)[m] for f in
                          ("station", "pos", "time", "layer", "theta", "ec", "x", "y", "weight")))

    def valid(self) -> np.ndarray:
        return np.isfinite(self.theta) & np.isfinite(self.ec)


def build_probe_obs(avg_model: np.ndarray, *, sample_positions, sample_times, theta_obs,
                    probe_layer_indices, sensor_ids, point_x, point_y=None,
                    theta_obs_std=None) -> ProbeObs:
    """Collect the probe rows from the collapsed EC model ``([n_time,] n_points, n_layers)``.

    ``probe_layer_indices`` None -> horizon 0 (as ``localized_mapping.theta_at_points``).
    ``sensor_ids`` None -> the EM point index is the station identity.
    """
    avg = np.asarray(avg_model, dtype=float)
    pos = np.asarray(sample_positions, dtype=int)
    n = len(pos)
    times = np.asarray(sample_times, dtype=int)
    n_layers = avg.shape[-1]
    layer = (np.zeros(n, dtype=int) if probe_layer_indices is None
             else np.clip(np.asarray(probe_layer_indices, dtype=int), 0, n_layers - 1))
    ec = np.array([avg[times[i], pos[i], layer[i]] if avg.ndim == 3 else avg[pos[i], layer[i]]
                   for i in range(n)], dtype=float)
    station = (np.array([str(p) for p in pos], dtype=object) if sensor_ids is None
               else np.array([str(s) for s in sensor_ids], dtype=object))
    if len(station) != n:
        raise ValueError(f"sensor_ids has {len(station)} entries, expected {n}")
    px = np.asarray(point_x, dtype=float)
    py = np.zeros_like(px) if point_y is None else np.asarray(point_y, dtype=float)
    if theta_obs_std is None:
        w = np.ones(n)
    else:
        std = np.broadcast_to(np.asarray(theta_obs_std, dtype=float), (n,))
        ok = np.isfinite(std) & (std > 0)
        w = np.ones(n) if not ok.all() else (1.0 / std ** 2) / np.mean(1.0 / std ** 2)
    return ProbeObs(station=station, pos=pos, time=times, layer=layer,
                    theta=np.asarray(theta_obs, dtype=float), ec=ec,
                    x=px[pos], y=py[pos], weight=w)


def count_stations(sensor_ids, fallback: int | None = None) -> int:
    """Number of distinct stations actually used (G: the log must not print the config value)."""
    if sensor_ids is None or len(sensor_ids) == 0:
        return int(fallback or 0)
    return len({str(s) for s in sensor_ids})


# ======================================================================== pooled params
def pooled_params_from_fit(global_fit) -> np.ndarray:
    """``(n_layers, 4)`` array [a, m, sigma_w, sigma_s] from a ``PetroFit``."""
    return np.array([[h.a, h.m, h.sigma_w, h.sigma_s] for h in global_fit.horizon_fits], dtype=float)


def fit_pooled(obs: ProbeObs, n_layers: int, *, lab_p0=None, fixed_params=None,
               ws_bounds=None) -> tuple[np.ndarray, np.ndarray]:
    """Pooled per-horizon fit on ``obs`` (what the production Step 2/4 fit does, one model).

    Returns ``(params (n_layers, 4), calibrated (n_layers,) bool)``; a horizon without valid
    observations keeps the prior ``lab_p0``.
    """
    ok = obs.valid()
    # em_methodology wants the EC array, so rebuild a 2-D ``(n_points, n_layers)`` carrier:
    # one row per observation, EC in the observation's horizon column.
    n = int(ok.sum())
    carrier = np.full((1, max(n, 1), n_layers), np.nan)
    if n:
        carrier[0, np.arange(n), obs.layer[ok]] = obs.ec[ok]
    fits = EMmethod.calibrate_petrophysical_relationship(
        models=carrier, sample_positions=list(range(n)), sample_times=[0] * n,
        theta_obs=obs.theta[ok], lab_p0=lab_p0, probe_layer_indices=obs.layer[ok].tolist(),
        fixed_params=fixed_params, ws_bounds=ws_bounds)
    fit = fits[0]
    params = pooled_params_from_fit(fit)
    cal = np.array([bool(getattr(h, "is_calibrated", True)) for h in fit.horizon_fits])
    return params, cal


# ======================================================================== B: per-station fits
@dataclass
class StationFits:
    """One WS fit per (horizon, station); arrays are ``(n_layers, n_stations)``."""
    stations: list
    x: np.ndarray                 # (n_stations,)
    y: np.ndarray                 # (n_stations,)
    params: np.ndarray            # (4, n_layers, n_stations) a, m, sigma_w, sigma_s
    n_obs: np.ndarray             # (n_layers, n_stations) int
    rmse: np.ndarray              # (n_layers, n_stations)
    valid: np.ndarray             # (n_layers, n_stations) bool: fitted (>= MIN_STATION_OBS obs)
    at_bound: np.ndarray          # (n_layers, n_stations) bool: a free parameter sits on a bound
    free: tuple = ("sigma_s",)
    shrinkage: float = 1.0
    notes: list = field(default_factory=list)


def effective_free(free: Sequence[str], fixed_params: dict | None) -> tuple[str, ...]:
    """The per-station free parameters: ``free`` minus whatever the scenario pins."""
    fixed_params = fixed_params or {}
    return tuple(p for p in free if p in PARAMS and not fixed_params.get(p, False))


def fit_stations(obs: ProbeObs, pooled: np.ndarray, *, fixed_params=None, ws_bounds=None,
                 free: Sequence[str] = ("sigma_s",), shrinkage: float = 1.0,
                 min_obs: int = MIN_STATION_OBS) -> StationFits:
    """Fit the WS parameters in ``free`` separately at every station and horizon.

    The other parameters are held at ``pooled[layer]``, which is also the prior centre, so a
    noisy station is pulled towards the pooled value (``shrinkage`` multiplies the prior
    weight; 1 = the pooled fit's own regularisation). A station with fewer than ``min_obs``
    valid observations in a horizon is not fitted (``valid`` False; it takes the pooled value).
    """
    n_layers = pooled.shape[0]
    stations = list(dict.fromkeys(obs.station.tolist()))
    n_st = len(stations)
    free_eff = effective_free(free, fixed_params)
    notes: list = []
    if len(free) and not free_eff:
        notes.append("every requested per-station parameter is fixed by the scenario "
                     "(ws_fixed_params): the station fits equal the pooled fit")
    fixed_station = {p: (p not in free_eff) for p in PARAMS}
    lo, hi = EMmethod.resolve_ws_bounds(ws_bounds)

    params = np.repeat(pooled.T[:, :, None], n_st, axis=2).astype(float)       # (4, L, S)
    n_obs = np.zeros((n_layers, n_st), dtype=int)
    rmse = np.full((n_layers, n_st), np.nan)
    valid = np.zeros((n_layers, n_st), dtype=bool)
    at_bound = np.zeros((n_layers, n_st), dtype=bool)
    sx = np.zeros(n_st)
    sy = np.zeros(n_st)
    ok_all = obs.valid()
    for s_i, st in enumerate(stations):
        of_st = obs.station == st
        if of_st.any():
            sx[s_i], sy[s_i] = obs.x[of_st][0], obs.y[of_st][0]
        for layer in range(n_layers):
            m = of_st & (obs.layer == layer) & ok_all
            n_obs[layer, s_i] = int(m.sum())
            if m.sum() < min_obs or not free_eff:
                continue
            p0 = pooled[layer].copy()
            a, mm, sw, ss, pred = EMmethod._fit_ws_horizon_weighted(
                obs.theta[m], obs.ec[m], obs.weight[m] / np.mean(obs.weight[m]), p0,
                fixed_station, ws_bounds=ws_bounds,
                lambda_reg=0.1 * float(shrinkage))
            params[:, layer, s_i] = [a, mm, sw, ss]
            rmse[layer, s_i] = float(np.sqrt(np.mean((obs.theta[m] - pred) ** 2)))
            valid[layer, s_i] = True
            tol = 1e-6
            vals = (a, mm, sw, ss)
            at_bound[layer, s_i] = any(
                (vals[i] <= lo[i] + tol * max(1.0, abs(lo[i])) or vals[i] >= hi[i] - tol * max(1.0, abs(hi[i])))
                for i, p in enumerate(PARAMS) if p in free_eff)
    return StationFits(stations=stations, x=sx, y=sy, params=params, n_obs=n_obs, rmse=rmse,
                       valid=valid, at_bound=at_bound, free=free_eff, shrinkage=float(shrinkage),
                       notes=notes)


def write_ws_by_station_csv(sf: StationFits, path: str) -> str:
    """``ws_by_station.csv``: station, x, y, horizon, a, m, sigma_w, sigma_s, n_obs, rmse, fitted, at_bound."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["station", "x", "y", "horizon", *PARAMS, "n_obs", "rmse", "fitted", "at_bound"])
        for layer in range(sf.params.shape[1]):
            for s_i, st in enumerate(sf.stations):
                w.writerow([st, f"{sf.x[s_i]:.3f}", f"{sf.y[s_i]:.3f}", layer,
                            *(f"{sf.params[k, layer, s_i]:.6g}" for k in range(4)),
                            int(sf.n_obs[layer, s_i]),
                            ("" if not np.isfinite(sf.rmse[layer, s_i]) else f"{sf.rmse[layer, s_i]:.6f}"),
                            int(sf.valid[layer, s_i]), int(sf.at_bound[layer, s_i])])
    return path


def station_fits_to_dict(sf: StationFits) -> dict:
    """JSON-serialisable record of the per-station values (``stage_calibration.json``)."""
    return {"free": list(sf.free), "shrinkage": sf.shrinkage, "stations": list(sf.stations),
            "x": [float(v) for v in sf.x], "y": [float(v) for v in sf.y],
            "params": {p: sf.params[k].tolist() for k, p in enumerate(PARAMS)},
            "n_obs": sf.n_obs.tolist(), "fitted": sf.valid.tolist(),
            "at_bound": sf.at_bound.tolist(), "notes": list(sf.notes)}


# ======================================================================== C: spreading
def spread_station_params(sf: StationFits, tx: Sequence[float], ty: Sequence[float] | None,
                          method: str, pooled: np.ndarray, *, power: float = 2.0
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Per-target parameters ``(4, n_layers, n_targets)`` and ``fallback (n_layers, n_targets)``.

    ``voronoi``: the value of the nearest fitted station of the horizon. ``idw``: inverse
    distance weights ``1/d**power`` over the fitted stations (a target on a station takes its
    value). Targets of a horizon without any fitted station take the pooled value
    (``fallback`` True). Distances are 2-D Euclidean (1-D when ``ty`` is None).
    """
    if method not in STATION_METHODS:
        raise ValueError(f"method must be one of {STATION_METHODS}, got {method!r}")
    tx = np.asarray(tx, dtype=float)
    ty = np.zeros_like(tx) if ty is None else np.asarray(ty, dtype=float)
    n_layers = sf.params.shape[1]
    params = np.repeat(pooled.T[:, :, None], len(tx), axis=2).astype(float)
    fallback = np.ones((n_layers, len(tx)), dtype=bool)
    for layer in range(n_layers):
        ok = sf.valid[layer]
        if not ok.any():
            continue
        d = np.sqrt((tx[:, None] - sf.x[ok][None, :]) ** 2 + (ty[:, None] - sf.y[ok][None, :]) ** 2)
        vals = sf.params[:, layer, :][:, ok]                         # (4, n_ok)
        if method == "voronoi":
            nearest = np.argmin(d, axis=1)
            params[:, layer, :] = vals[:, nearest]
        else:
            with np.errstate(divide="ignore"):
                wgt = 1.0 / np.maximum(d, 1e-12) ** power
            wgt /= wgt.sum(axis=1, keepdims=True)
            params[:, layer, :] = vals @ wgt.T
        fallback[layer] = False
    return params, fallback


@dataclass
class SpatialMapping(lmap.LocalizedMapping):
    """``LocalizedMapping`` of a station-based method (``bandwidth`` is NaN: no kernel)."""
    method: str = "voronoi"
    station_fits: StationFits | None = None
    power: float = 2.0
    pooled: np.ndarray | None = None          # (n_layers, 4) pooled values (prior centre, fallback)


def build_station_mapping(avg_model: np.ndarray, global_fit, *, method: str, point_x, point_y,
                          sample_positions, sample_times, theta_obs, probe_layer_indices=None,
                          sensor_ids=None, fixed_params=None, theta_obs_std=None, ws_bounds=None,
                          free: Sequence[str] = ("sigma_s",), power: float = 2.0,
                          shrinkage: float = 1.0) -> SpatialMapping:
    """EC -> theta map with per-station WS fits spread over the field (D).

    ``global_fit`` (the Step 4 ``PetroFit`` at K) gives the pooled values: the held parameters,
    the prior centre and the fallback. Pooled output is untouched elsewhere: this is only
    called when ``ws_spatial`` is ``voronoi`` or ``idw``.
    """
    avg = np.asarray(avg_model, dtype=float)
    if avg.ndim not in (2, 3):
        raise ValueError(f"avg_model must be 2-D or 3-D (ensemble axis collapsed), got {avg.shape}")
    n_points = avg.shape[-2]
    if len(point_x) != n_points:
        raise ValueError(f"point_x has {len(point_x)} entries, the EC model has {n_points} points")
    obs = build_probe_obs(avg, sample_positions=sample_positions, sample_times=sample_times,
                          theta_obs=theta_obs, probe_layer_indices=probe_layer_indices,
                          sensor_ids=sensor_ids, point_x=point_x, point_y=point_y,
                          theta_obs_std=theta_obs_std)
    pooled = pooled_params_from_fit(global_fit)
    n_layers = avg.shape[-1]
    if pooled.shape[0] < n_layers:                      # as build_localized_mapping: last horizon repeats
        pooled = np.vstack([pooled, np.repeat(pooled[-1:], n_layers - pooled.shape[0], axis=0)])
    pooled = pooled[:n_layers]
    sf = fit_stations(obs, pooled, fixed_params=fixed_params, ws_bounds=ws_bounds,
                      free=free, shrinkage=shrinkage)
    tx = np.asarray(point_x, dtype=float)
    ty = None if point_y is None else np.asarray(point_y, dtype=float)
    params, fallback = spread_station_params(sf, tx, ty, method, pooled, power=power)
    return SpatialMapping(theta=lmap.theta_from_params(avg, params), bandwidth=float("nan"),
                          params=params, fallback=fallback, localized_fit=None,
                          method=method, station_fits=sf, power=float(power), pooled=pooled)


# ======================================================================== leave-one-station-out
def _predict(ec: np.ndarray, p: np.ndarray) -> np.ndarray:
    """theta = (a EC / (sigma_w + sigma_s))**(1/m), vectorised; ``p`` is (4,) or (4, n)."""
    p = np.asarray(p, dtype=float)
    a, m, sw, ss = p
    ratio = a * np.clip(ec, 1e-6, None) / (sw + ss)
    return np.power(np.clip(ratio, 0.0, None), 1.0 / m)


def loso_predictions(obs: ProbeObs, n_layers: int, *, methods: Sequence[str] = SIMPLICITY_ORDER,
                     bandwidth: float | None = None, avg_model: np.ndarray | None = None,
                     point_x=None, point_y=None, lab_p0=None, fixed_params=None, ws_bounds=None,
                     free: Sequence[str] = ("sigma_s",), power: float = 2.0,
                     shrinkage: float = 1.0) -> dict[str, np.ndarray]:
    """Leave-one-STATION-out predictions of theta for every observation, per method.

    All observations of a station (every epoch and horizon) are removed together, the
    method is refitted on the remaining stations, and evaluated at the held-out station's
    location with the parameters the method assigns there. ``pooled`` is the baseline (a pooled
    fit of the remaining stations). A method that cannot predict at a station (kernel without a
    neighbour within 2c) uses the pooled fit of the remaining stations, never the held-out one,
    so all methods score the SAME observations. ``kernel`` needs ``bandwidth``, ``avg_model``,
    ``point_x`` (and ``point_y`` for 2-D); it is skipped otherwise.

    Returns ``{method: (n_obs,) predicted theta}`` (NaN for observations that are not valid).
    """
    methods = [m for m in methods if m in SIMPLICITY_ORDER]
    if "kernel" in methods and (bandwidth is None or avg_model is None or point_x is None):
        methods = [m for m in methods if m != "kernel"]
    out = {m: np.full(len(obs), np.nan) for m in methods}
    ok_all = obs.valid()
    for st in dict.fromkeys(obs.station.tolist()):
        held = (obs.station == st) & ok_all
        train = ~(obs.station == st)
        if not held.any() or not (train & ok_all).any():
            continue
        tr = obs.subset(train)
        pooled, cal = fit_pooled(tr, n_layers, lab_p0=lab_p0, fixed_params=fixed_params,
                                 ws_bounds=ws_bounds)
        sf = None
        if any(m in STATION_METHODS for m in methods):
            sf = fit_stations(tr, pooled, fixed_params=fixed_params, ws_bounds=ws_bounds,
                              free=free, shrinkage=shrinkage)
        hx, hy = float(obs.x[held][0]), float(obs.y[held][0])
        hpos = int(obs.pos[held][0])
        for method in methods:
            if method == "pooled":
                params = np.repeat(pooled.T[:, :, None], 1, axis=2)
            elif method in STATION_METHODS:
                params, _ = spread_station_params(sf, [hx], [hy], method, pooled, power=power)
            else:                                           # kernel
                fits = EMmethod.calibrate_petrophysical_relationship_localized(
                    models=np.asarray(avg_model, dtype=float)[np.newaxis],
                    sample_positions=obs.pos[train].tolist(), sample_times=obs.time[train].tolist(),
                    theta_obs=obs.theta[train], point_x_coords=point_x, bandwidth=float(bandwidth),
                    point_y_coords=point_y, target_indices=[hpos], lab_p0=lab_p0,
                    probe_layer_indices=obs.layer[train].tolist(), fixed_params=fixed_params,
                    ws_bounds=ws_bounds)
                params = np.repeat(pooled.T[:, :, None], 1, axis=2)
                for layer, hz in enumerate(fits[0].horizon_fits[:n_layers]):
                    loc = np.array([float(np.asarray(v)[0]) for v in (hz.a, hz.m, hz.sigma_w, hz.sigma_s)])
                    if bool(np.asarray(hz.is_calibrated)[0]) and np.isfinite(loc).all():
                        params[:, layer, 0] = loc
            for i in np.flatnonzero(held):
                out[method][i] = float(_predict(obs.ec[i], params[:, obs.layer[i], 0]))
    return out


def loso_summary(obs: ProbeObs, preds: dict[str, np.ndarray]) -> dict[str, dict]:
    """Per method: RMSE, bias, n and the standard error of the RMSE (over held-out stations).

    The SE comes from the per-station mean squared errors: ``se_mse / (2 rmse)``.
    """
    ok = obs.valid()
    summary: dict[str, dict] = {}
    for method, pred in preds.items():
        m = ok & np.isfinite(pred)
        if not m.any():
            summary[method] = {"rmse": float("nan"), "bias": float("nan"), "n": 0, "se": float("nan"),
                               "n_stations": 0}
            continue
        err = pred[m] - obs.theta[m]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        per_station = np.array([np.mean(err[obs.station[m] == s] ** 2)
                                for s in dict.fromkeys(obs.station[m].tolist())])
        se_mse = float(np.std(per_station, ddof=1) / np.sqrt(len(per_station))) if len(per_station) > 1 else float("nan")
        summary[method] = {"rmse": rmse, "bias": float(np.mean(err)), "n": int(m.sum()),
                           "se": (se_mse / (2.0 * rmse)) if rmse > 0 else float("nan"),
                           "n_stations": int(len(per_station))}
    return summary


def choose_method(summary: dict[str, dict], order: Sequence[str] = SIMPLICITY_ORDER) -> tuple[str | None, str]:
    """Simplest method whose RMSE is within one standard error of the best. ``(method, reason)``."""
    scored = {m: s for m, s in summary.items() if np.isfinite(s.get("rmse", np.nan))}
    if not scored:
        return None, "no method could be scored"
    best = min(scored, key=lambda m: scored[m]["rmse"])
    se = scored[best]["se"]
    limit = scored[best]["rmse"] + (se if np.isfinite(se) else 0.0)
    for m in order:
        if m in scored and scored[m]["rmse"] <= limit:
            if m == best:
                return m, f"{m} has the lowest held-out RMSE ({scored[m]['rmse']:.5f})"
            return m, (f"{m} ({scored[m]['rmse']:.5f}) is within one standard error "
                       f"({se:.5f}) of the best, {best} ({scored[best]['rmse']:.5f}), and simpler")
    return best, f"{best} has the lowest held-out RMSE"


# ======================================================================== F: probe errors
PROBE_ERROR_COLUMNS = ("station", "x", "y", "epoch", "horizon", "theta_TDR", "theta_calc",
                       "error", "held_out_flag")


def probe_error_records(obs: ProbeObs, theta_calc: np.ndarray, *, held_out: bool) -> list[dict]:
    """One record per valid probe observation: ``error = theta_calc - theta_TDR``."""
    recs = []
    for i in np.flatnonzero(obs.valid() & np.isfinite(theta_calc)):
        recs.append({"station": str(obs.station[i]), "x": float(obs.x[i]), "y": float(obs.y[i]),
                     "epoch": int(obs.time[i]), "horizon": int(obs.layer[i]),
                     "theta_TDR": float(obs.theta[i]), "theta_calc": float(theta_calc[i]),
                     "error": float(theta_calc[i] - obs.theta[i]), "held_out_flag": bool(held_out)})
    return recs


def in_sample_theta(obs: ProbeObs, params: np.ndarray) -> np.ndarray:
    """theta at the probes from per-point parameters ``(4, n_layers, n_points)``."""
    out = np.full(len(obs), np.nan)
    for i in range(len(obs)):
        if np.isfinite(obs.ec[i]):
            out[i] = float(_predict(obs.ec[i], params[:, obs.layer[i], obs.pos[i]]))
    return out


def error_metrics(records: Sequence[dict], held_out: bool | None = None) -> dict:
    """RMSE and bias overall, per station and per epoch (``held_out`` None = all records)."""
    recs = [r for r in records if held_out is None or r["held_out_flag"] == held_out]
    if not recs:
        return {"n": 0, "rmse": float("nan"), "bias": float("nan"), "by_station": {}, "by_epoch": {}}

    def stats(rs):
        e = np.array([r["error"] for r in rs])
        return {"n": len(e), "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e))}

    out = stats(recs)
    out["by_station"] = {s: stats([r for r in recs if r["station"] == s])
                         for s in dict.fromkeys(r["station"] for r in recs)}
    out["by_epoch"] = {k: stats([r for r in recs if r["epoch"] == k])
                       for k in sorted({r["epoch"] for r in recs})}
    return out


def write_probe_errors_csv(records: Sequence[dict], path: str) -> str:
    """``probe_errors.csv`` (station, x, y, epoch, horizon, theta_TDR, theta_calc, error, held_out_flag)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(PROBE_ERROR_COLUMNS)
        for r in records:
            w.writerow([r["station"], f"{r['x']:.3f}", f"{r['y']:.3f}", r["epoch"], r["horizon"],
                        f"{r['theta_TDR']:.6f}", f"{r['theta_calc']:.6f}", f"{r['error']:.6f}",
                        int(r["held_out_flag"])])
    return path
