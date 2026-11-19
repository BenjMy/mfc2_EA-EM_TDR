"""localized_mapping.py -- convert the optimal EC model to theta with LOCALIZED Waxman-Smits.

By default (``--ws-map-calibration localized``) the water-content maps are no longer built
with ONE global ``(a, m, sigma_w, sigma_s)`` set per horizon. After Step 5 has chosen the
stack depth K, the Gaspari-Cohn kernel-localized fit
(:func:`em_methodology.calibrate_petrophysical_relationship_localized`) is run on the
optimal averaged EC model, and every EM point is converted with the parameters fitted FOR
THAT POINT:

    theta(t, x, layer) = ( a_l(x) * EC(t, x, layer) / (sigma_w_l(x) + sigma_s_l(x)) ) ** (1 / m_l(x))

Nothing is re-inverted: the localized fit only reads the probe-matched EC and the TDR theta
(cheap), and is evaluated at every point.

Fallback to the global fit
--------------------------
The Gaspari-Cohn kernel is exactly zero beyond ``2 * bandwidth``. A point of a horizon with
no TDR observation inside that support cannot be localized; it gets the GLOBAL per-horizon
parameters of the Step 4 fit at K (``selected_output.best_fit``), not the lab prior. The
same applies to a horizon with no probe at all (the global fit there is the prior too) and
to any non-finite localized parameter. ``LocalizedMapping.fallback`` records where.

Opt-out: ``--ws-map-calibration global`` keeps the previous behaviour (one set per horizon).
Without a localized bandwidth (no ``ws_localized_bandwidth`` / sweep / calibration value)
the maps also use the global fit, and the log says why.

Pure numpy; imports ``em_methodology`` only for the localized fit. Unit-tested in
``tests/test_localized_mapping.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import em_methodology as EMmethod

MAP_CALIBRATION_CHOICES = ("localized", "global")
K_SELECTION_CHOICES = ("global", "localized-loo")
DEFAULT_K_SELECTION = "global"
DEFAULT_MAP_CALIBRATION = "localized"
_PARAM_NAMES = ("a", "m", "sigma_w", "sigma_s")


@dataclass
class LocalizedMapping:
    """Result of :func:`build_localized_mapping`.

    ``theta`` has the shape of the EC model it was built from. ``params`` is
    ``(4, n_layers, n_points)`` in the order a, m, sigma_w, sigma_s, AFTER the fallback;
    ``fallback`` is ``(n_layers, n_points)``, True where the global fit was used.
    """
    theta: np.ndarray
    bandwidth: float
    params: np.ndarray
    fallback: np.ndarray
    localized_fit: object

    def fallback_fraction(self) -> list[float]:
        """Per horizon, the fraction of EM points that fell back to the global fit."""
        return [float(np.mean(row)) for row in self.fallback]


def resolve_map_calibration(requested: str | None, bandwidth: float | None) -> tuple[str, str]:
    """``(mode, reason)``: the calibration the maps will use.

    ``requested`` None -> the default (localized). ``reason`` is non-empty only when the
    request had to be downgraded to ``global`` (localized needs a bandwidth).
    """
    mode = DEFAULT_MAP_CALIBRATION if requested is None else str(requested)
    if mode not in MAP_CALIBRATION_CHOICES:
        raise ValueError(f"ws_map_calibration must be one of {MAP_CALIBRATION_CHOICES}, got {mode!r}")
    if mode == "localized" and (bandwidth is None or not np.isfinite(bandwidth) or bandwidth <= 0):
        return "global", ("no localized bandwidth: set ws_localized_bandwidth / "
                          "ws_localized_bandwidth_sweep in the scenario or pass "
                          "--ws-localized-bandwidth")
    return mode, ""


def theta_from_params(ec_grid: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Waxman-Smits EC -> theta with per-(layer, point) parameters.

    ``ec_grid`` is ``([n_time,] n_points, n_layers)``; ``params`` is
    ``(4, n_layers, n_points)``. Same formula as ``step4_select._ec_grid_to_theta``
    (EC clipped at 1e-6, ratio clipped at 0); NaN EC stays NaN; layers without parameters
    get 0, as there.
    """
    ec_grid = np.asarray(ec_grid, dtype=float)
    theta = np.zeros_like(ec_grid)
    for layer in range(min(ec_grid.shape[-1], params.shape[1])):
        a, m, sigma_w, sigma_s = params[:, layer, :]          # each (n_points,)
        ec = np.clip(ec_grid[..., layer], 1e-6, None)         # broadcasts over time
        ratio = a * ec / (sigma_w + sigma_s)
        theta[..., layer] = np.power(np.clip(ratio, 0.0, None), 1.0 / m)
    return theta


def build_localized_mapping(
    avg_model: np.ndarray,
    global_fit,
    *,
    bandwidth: float,
    point_x: Sequence[float],
    point_y: Sequence[float] | None,
    sample_positions: Sequence[int],
    sample_times: Sequence[int],
    theta_obs: np.ndarray,
    probe_layer_indices: Sequence[int] | None = None,
    lab_p0=None,
    fixed_params: dict | None = None,
    theta_obs_std=None,
    ws_bounds: dict | None = None,
) -> LocalizedMapping:
    """Localized fit on ``avg_model`` (the optimal averaged EC model), evaluated at every point.

    Parameters
    ----------
    avg_model : ``([n_time,] n_points, n_layers)``, the ensemble axis already collapsed.
    global_fit : the Step 4 ``PetroFit`` at K; its ``horizon_fits`` are the fallback.
    bandwidth, point_x, point_y, ... : as in
        ``EMmethod.calibrate_petrophysical_relationship_localized`` (``point_y`` None =
        1-D distance along x, else 2-D Euclidean).
    """
    avg = np.asarray(avg_model, dtype=float)
    if avg.ndim not in (2, 3):
        raise ValueError(f"avg_model must be 2-D or 3-D (ensemble axis collapsed), got {avg.shape}")
    n_points = avg.shape[-2]
    if len(point_x) != n_points:
        raise ValueError(f"point_x has {len(point_x)} entries, the EC model has {n_points} points")

    fits = EMmethod.calibrate_petrophysical_relationship_localized(
        models=avg[np.newaxis],                      # length-1 ensemble, as in step5b
        sample_positions=sample_positions, sample_times=sample_times, theta_obs=theta_obs,
        point_x_coords=point_x, bandwidth=bandwidth, point_y_coords=point_y,
        lab_p0=lab_p0, probe_layer_indices=probe_layer_indices,
        fixed_params=fixed_params, theta_obs_std=theta_obs_std, ws_bounds=ws_bounds)
    loc = fits[0]

    n_layers = len(loc.horizon_fits)
    params = np.empty((4, n_layers, n_points), dtype=float)
    fallback = np.zeros((n_layers, n_points), dtype=bool)
    n_global = len(global_fit.horizon_fits)
    for layer, hz in enumerate(loc.horizon_fits):
        g = global_fit.horizon_fits[min(layer, n_global - 1)]
        g_params = np.array([g.a, g.m, g.sigma_w, g.sigma_s], dtype=float)
        # targets are every point, in order (target_indices = arange(n_points))
        local = np.vstack([hz.a, hz.m, hz.sigma_w, hz.sigma_s]).astype(float)
        bad = ~np.asarray(hz.is_calibrated, dtype=bool) | ~np.all(np.isfinite(local), axis=0)
        local[:, bad] = g_params[:, None]
        params[:, layer, :] = local
        fallback[layer] = bad

    return LocalizedMapping(theta=theta_from_params(avg, params), bandwidth=float(bandwidth),
                            params=params, fallback=fallback, localized_fit=loc)


def theta_at_points(mapping: LocalizedMapping, avg_model: np.ndarray,
                    positions: Sequence[int], times: Sequence[int],
                    layers: Sequence[int] | None) -> np.ndarray:
    """theta predicted by the localized parameters at given probes (e.g. held-out ones).

    ``positions`` index the points axis, ``times`` the time axis (time-lapse only),
    ``layers`` the horizon of each probe (None -> horizon 0, as ``ec_to_water_content``).
    """
    avg = np.asarray(avg_model, dtype=float)
    n_layers = mapping.params.shape[1]
    out = np.empty(len(positions), dtype=float)
    for i, pos in enumerate(positions):
        layer = 0 if layers is None else min(max(int(layers[i]), 0), n_layers - 1)
        ec = avg[int(times[i]), int(pos), layer] if avg.ndim == 3 else avg[int(pos), layer]
        a, m, sigma_w, sigma_s = mapping.params[:, layer, int(pos)]
        ratio = a * max(ec, 1e-6) / (sigma_w + sigma_s) if np.isfinite(ec) else np.nan
        out[i] = np.power(max(ratio, 0.0), 1.0 / m) if np.isfinite(ratio) else np.nan
    return out


def resolve_k_selection(requested: str | None, bandwidth: float | None) -> tuple[str, str]:
    """``(mode, reason)``: how the stack depth K is selected.

    ``requested`` None -> the default (``global``). ``localized-loo`` needs a bandwidth;
    without one it is downgraded to ``global`` and ``reason`` says why.
    """
    mode = DEFAULT_K_SELECTION if requested is None else str(requested)
    if mode not in K_SELECTION_CHOICES:
        raise ValueError(f"k_selection must be one of {K_SELECTION_CHOICES}, got {mode!r}")
    if mode == "localized-loo" and (bandwidth is None or not np.isfinite(bandwidth) or bandwidth <= 0):
        return "global", ("no localized bandwidth: set ws_localized_bandwidth / "
                          "ws_localized_bandwidth_sweep in the scenario or pass "
                          "--ws-localized-bandwidth")
    return mode, ""


def localized_loo_rmse(
    avg_model: np.ndarray,
    *,
    bandwidth: float,
    point_x: Sequence[float],
    point_y: Sequence[float] | None,
    sample_positions: Sequence[int],
    sample_times: Sequence[int],
    theta_obs: np.ndarray,
    probe_layer_indices: Sequence[int] | None = None,
    lab_p0=None,
    fixed_params: dict | None = None,
    theta_obs_std=None,
    ws_bounds: dict | None = None,
) -> tuple[float, int, int]:
    """Leave-one-probe-out RMSE of the localized Waxman-Smits fit on ``avg_model``.

    For every distinct probe LOCATION (``sample_positions`` value) all observations at that
    location are removed (all epochs, so a time-lapse probe cannot leak through its other
    epochs), the localized fit is refitted on the remaining probes, and evaluated at the
    held-out location with the parameters fitted FOR THAT POINT. The RMSE is taken over the
    held-out observations.

    A held-out probe with no remaining probe of its horizon within the kernel support (2c),
    or whose EC / parameters are not finite, is NOT scored: it has no localized prediction,
    and substituting the global fit would leak the held-out probe (the global fit saw it).
    Which probes are skipped depends on the geometry and bandwidth only, so the scored set
    is the same for every K and the curves are comparable.

    Returns ``(rmse, n_scored, n_total)``; ``rmse`` is NaN when nothing could be scored.
    """
    avg = np.asarray(avg_model, dtype=float)
    positions = np.asarray(sample_positions, dtype=int)
    times = np.asarray(sample_times, dtype=int)
    theta_obs = np.asarray(theta_obs, dtype=float)
    layers = None if probe_layer_indices is None else np.asarray(probe_layer_indices, dtype=int)
    std = None if theta_obs_std is None else np.broadcast_to(
        np.asarray(theta_obs_std, dtype=float), (len(positions),))
    n_total = len(positions)

    sq_err: list[float] = []
    for pos in np.unique(positions):
        held = positions == pos
        keep = ~held
        if not keep.any():
            continue
        fits = EMmethod.calibrate_petrophysical_relationship_localized(
            models=avg[np.newaxis],
            sample_positions=positions[keep].tolist(), sample_times=times[keep].tolist(),
            theta_obs=theta_obs[keep], point_x_coords=point_x, bandwidth=bandwidth,
            point_y_coords=point_y, target_indices=[int(pos)], lab_p0=lab_p0,
            probe_layer_indices=None if layers is None else layers[keep].tolist(),
            fixed_params=fixed_params,
            theta_obs_std=None if std is None else std[keep], ws_bounds=ws_bounds)
        horizons = fits[0].horizon_fits
        for i in np.flatnonzero(held):
            layer = 0 if layers is None else min(max(int(layers[i]), 0), len(horizons) - 1)
            hz = horizons[layer]
            if not bool(np.asarray(hz.is_calibrated)[0]):
                continue
            a, m, sigma_w, sigma_s = (float(np.asarray(v)[0]) for v in (hz.a, hz.m, hz.sigma_w, hz.sigma_s))
            ec = avg[int(times[i]), int(pos), layer] if avg.ndim == 3 else avg[int(pos), layer]
            if not (np.isfinite(ec) and np.isfinite([a, m, sigma_w, sigma_s]).all()
                    and np.isfinite(theta_obs[i])):
                continue
            ratio = a * max(ec, 1e-6) / (sigma_w + sigma_s)
            sq_err.append((theta_obs[i] - np.power(max(ratio, 0.0), 1.0 / m)) ** 2)

    if not sq_err:
        return float("nan"), 0, n_total
    return float(np.sqrt(np.mean(sq_err))), len(sq_err), n_total


def localized_loo_rmse_by_k(averaged_models: Sequence[np.ndarray], **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """:func:`localized_loo_rmse` for every cumulative stack ``averaged_models[k]`` (K = k + 1).

    ``kwargs`` are those of :func:`localized_loo_rmse`. Returns ``(rmse_by_k, n_scored_by_k)``.
    """
    rmse, n_scored = [], []
    for avg in averaged_models:
        r, n, _ = localized_loo_rmse(avg, **kwargs)
        rmse.append(r)
        n_scored.append(n)
    return np.array(rmse, dtype=float), np.array(n_scored, dtype=int)
