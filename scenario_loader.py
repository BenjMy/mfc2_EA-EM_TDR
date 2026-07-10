#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenario_loader.py
==================
Reads, validates, and returns ensemble configuration scenarios defined in
``ensemble_scenarios.json`` (or any compatible JSON file).

Usage
-----
From the command line (via run_pipeline.py):
    python run_pipeline.py --scenario free_all
    python run_pipeline.py --scenario default --config my_scenarios.json

From Python:
    from scenario_loader import load_scenario
    cfg = load_scenario("mixed_middle_free")

Returned object
---------------
A ``ScenarioConfig`` dataclass with typed, validated fields ready to pass
directly into ``EMmethod.generate_ensemble_em_survey`` and the rest of the
pipeline.  All optional fields have sensible defaults so callers need not
guard against None.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Union


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """
    Typed container for one ensemble scenario.

    Attributes
    ----------
    name : str
        Scenario key as it appears in the JSON file.
    description : str
        Human-readable description (shown in pipeline log).
    output_dir : str
        Root output directory for plots, cache files, and NPY arrays.
    realization_index : int
        Index of the synthetic ground-truth member to use (0-based).
        Ignored for real field data.
    depths0 : list[float]
        Starting layer interface depths [m], length = n_layers - 1.
    fixed_depths : bool | list[bool]
        ``True``  → all interfaces fixed.
        ``False`` → all interfaces free.
        ``list``  → per-interface control (same length as depths0).
    param_grid : dict[str, list]
        Inversion parameter Cartesian grid fed to
        ``generate_ensemble_em_survey``.  Recognised keys: alpha, method,
        forwardModel.
    cond_ranges : list[list[float]]
        Per-layer conductivity starting bounds [mS/m], length = n_layers.
        Each inner list is [min, max] or [single_value].
    forward_model : str
        Default EMagPy forward model key ('CS', 'FSlin', 'FS', …).
        May be overridden per-member by param_grid['forwardModel'].
    gamma : float
        Temporal regularisation weight (time-lapse only).  0.0 = off.
    cache_tag : str
        Suffix appended to the pickle cache filename so different scenarios
        never share a stale cache.
    depths_are_fixed : bool
        Convenience flag derived from ``fixed_depths``: True only when every
        interface is fixed.  Passed to Steps 2 and 4 to control warnings and
        plot labels.
    """
    name:              str
    description:       str
    output_dir:        str
    realization_index: int
    depths0:           list[float]
    fixed_depths:      Union[bool, list[bool]]
    param_grid:        dict
    cond_ranges:       list[list[float]]
    forward_model:     str
    gamma:             float
    cache_tag:         str
    # ------------------------------------------------------------------
    # SWC / TDR sensor network
    # ------------------------------------------------------------------
    num_swc_positions: int              = 4
    """Number of horizontal positions at which SWC sensors are placed.
    Total probes = num_swc_positions × n_layers (one per layer per position).
    Raising this value improves petrophysical calibration coverage but has
    diminishing RMSE returns beyond ~num_measured_points // 2."""

    swc_depths: list[float] | None     = None
    """Explicit sensor installation depths [m].  When None the pipeline uses
    the layer-centre depths derived from depths0.  Override to test
    mis-matched sensor vs layer-boundary scenarios."""

    window_m: float                    = 1.0
    """Half-width of the spatial averaging window used when matching each
    SWC probe to its nearest EM measurement indices [m].  A good starting
    value is half the instrument coil spacing."""

    matching_warn_fraction: float      = 0.5
    """TDR<->EM spatial-matching quality gate (warn level). If the single
    most-shared EM measurement point ends up matched to more than this
    fraction of ALL probes/stations, the pipeline logs a [WARN] in the run
    report (see em_methodology.summarize_probe_matching). Does not stop
    the run. Default 0.5 (i.e. more than half of all probes sharing one
    EM point)."""

    matching_error_fraction: float | None = None
    """TDR<->EM spatial-matching quality gate (error level). If set and the
    largest-group fraction (see matching_warn_fraction) exceeds it, the
    pipeline aborts before Step 2 rather than silently calibrating against
    a near-degenerate spatial match. None (default) disables this hard
    gate; only the warn-level message is ever emitted."""

    # ------------------------------------------------------------------
    # Observation noise
    # ------------------------------------------------------------------
    tdr_noise_std: float               = 0.01
    """Standard deviation of Gaussian noise added to synthetic TDR θ
    readings [m³/m³].  Reflects real-world sensor uncertainty."""

    eca_noise_level: float             = 0.0
    """Fractional Gaussian noise applied to synthetic ECa readings before
    inversion (0.0 = clean, 0.05 = 5 % noise).  Tests inversion robustness
    to instrument noise."""

    homogeneous: bool                  = False
    """Synthetic generation only. When True, disables both spatial gradients
    (XY position and XZ/depth) in generate_time_lapse_dataset so theta is
    spatially uniform across the domain. The temporal trend (progressive
    wetting over timesteps) and per-sample Gaussian noise are still applied,
    so the uniform soil can still evolve over time. Useful for isolating
    temporal-only inversion behaviour without confounding spatial signal.
    Ignored when data_source == 'real'."""

    # ------------------------------------------------------------------
    # Ensemble spread control
    # ------------------------------------------------------------------
    alpha_range_log: list | None       = None
    """If set, overrides param_grid['alpha'] with a log-spaced sweep.
    Format: [min_alpha, max_alpha, n_steps]  e.g. [0.001, 5.0, 8].
    Provides a clean way to control regularisation diversity without
    listing individual values in the JSON."""

    # ------------------------------------------------------------------
    # 2-D grid geometry
    # ------------------------------------------------------------------
    grid_num_x: int                    = 10
    """Number of grid columns in 2-D spatial mode."""

    grid_num_y: int                    = 5
    """Number of grid rows in 2-D spatial mode."""

    # ------------------------------------------------------------------
    # Real-data mode  (only populated when data_source == 'real')
    # ------------------------------------------------------------------
    data_source: str                   = "synthetic"
    """'synthetic' (default) or 'real'.  When 'real', the three paths below
    must be set; the pipeline will refuse to mix synthetic generation with
    real field observations."""

    em_data_dir: str | None            = None
    """Directory containing GEM-2 / EM survey CSV files (real mode only)."""

    tdr_data_dir: str | None           = None
    """Directory containing SoilNet TDR Excel files (real mode only)."""

    coord_file: str | None             = None
    """Sensor coordinates CSV file (real mode only)."""

    time_tolerance_days: float         = 1.0
    """Half-window [days] for matching TDR readings to the nearest EM survey
    date (real mode only)."""

    em_unit: str                       = "ppt"
    """Physical unit of the quadrature/in-phase columns in the EM survey CSV.
    emagpy uses this to convert raw instrument readings to LIN ECa (mS/m):
      'ppm'  — parts per million  (GEM-2, CMD-Explorer, and most multi-
               frequency FD-EMI instruments)
      'ppt'  — parts per thousand (GF Instruments CMD series, default)
    Set to 'ppm' whenever you are loading real GEM-2 survey files.
    Leaving this at 'ppt' with ppm data inflates every ECa value by 1000×."""

    drop_coils: list[str | float] | None = None
    """Coil(s)/frequency channel(s) to exclude from every inversion in this
    scenario (e.g. a noisy GEM-2 frequency). Each entry may be the exact
    emagpy coil name (e.g. 'HCP1.66f425h1'), a frequency in Hz (e.g. 425),
    or a unique coil-name substring (e.g. 'f425') — see
    em_methodology.drop_coils_from_problem for matching rules. Applied
    identically to every ensemble member and to diagnostic re-inversions.
    None (default) drops nothing. A CLI --drop-coils flag, if given, takes
    priority over this field."""

    ws_fixed_params: dict | None = None
    """Holds one or more Waxman-Smits parameters fixed at their prior (lab
    or default) value during Steps 2 and 4 calibration, instead of letting
    all four (a, m, sigma_w, sigma_s) float freely. Keys are any subset of
    'a', 'm', 'sigma_w', 'sigma_s'; each maps to True to fix that parameter.
    E.g. {"m": true, "sigma_s": true} fixes the cementation exponent and
    surface conductivity — both fairly stable, texture-dependent constants
    — and leaves only 'a' and 'sigma_w' free. Recommended whenever TDR
    coverage per horizon is sparse (e.g. only 2 sensor-epoch observations),
    since fitting all 4 unknowns against that few points is underdetermined.
    None (default) leaves every parameter free, unchanged from before."""

    lab_p0: list | None = None
    """Starting-guess / fixed-value Waxman-Smits parameters [a, m, sigma_w,
    sigma_s], used both as the optimizer's starting point for any parameter
    left free AND as the pinned value for any parameter named True in
    ws_fixed_params (e.g. a literature/lab cementation exponent 'm' that
    ws_fixed_params pins the fit to, instead of the generic library default).
    Two accepted shapes:
      - A flat 4-element list [a, m, sigma_w, sigma_s], applied to every
        soil horizon.
      - A list of per-horizon 4-element lists, length == n_layers
        (= len(depths0) + 1), one [a, m, sigma_w, sigma_s] set per horizon.
    None (default) falls back to em_methodology.DEFAULT_WS_P0 =
    [1.5, 1.8, 50.0, 5.0] for every horizon, unchanged from before."""

    # ------------------------------------------------------------------
    # Quality gates (pipeline_improvement_report.md, section 4:
    # "Uncertainty & quality as gates, not just numbers")
    # ------------------------------------------------------------------
    quality_gate_cv_mean_warn: float = 0.3
    """Post-Step-5 quality gate: if the selected solution's mean CV
    (fractional, e.g. 0.3 = 30%) exceeds this, the run's final status is
    downgraded from SUCCESS to a warning state instead of reporting the
    same '✅ SUCCESS' as a clean run. Does not abort the run."""

    quality_gate_cv_high_threshold: float = 1.0
    """CV value (fractional, default 1.0 = 100%) above which a cell counts
    as 'high uncertainty' for quality_gate_cv_high_fraction_warn. Matches
    the paper's own CV>100% 'interpret with caution' guidance."""

    quality_gate_cv_high_fraction_warn: float = 0.3
    """Post-Step-5 quality gate: if more than this fraction of CV-map cells
    exceed quality_gate_cv_high_threshold, the run's final status is
    downgraded to a warning state. Does not abort the run."""

    quality_gate_require_all_horizons_calibrated: bool = True
    """Post-Step-5 quality gate: if True (default) and the selected
    solution has any horizon with no real TDR probe coverage
    (is_calibrated == False), the run's final status is downgraded to a
    warning state. Does not abort the run."""

    depths_are_fixed: bool             = field(init=False)

    def __post_init__(self):
        if isinstance(self.fixed_depths, bool):
            self.depths_are_fixed = self.fixed_depths
        else:
            self.depths_are_fixed = all(self.fixed_depths)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        return len(self.depths0) + 1

    @property
    def cache_filename(self) -> str:
        return f"ensemble_cache_{self.cache_tag}.pkl"

    @property
    def cache_path(self) -> str:
        return os.path.join(self.output_dir, self.cache_filename)

    def summary(self) -> str:
        """One-block log string shown at pipeline start."""
        fd = self.fixed_depths
        fd_str = (
            "all fixed" if fd is True else
            "all free"  if fd is False else
            "[" + ", ".join("fixed" if f else "FREE" for f in fd) + "]"
        )
        members_estimate = (
            len(list(__import__("itertools").product(
                *[v for v in self.cond_ranges if len(v) > 1]
            ))) *
            len(list(__import__("itertools").product(
                *self.param_grid.values()
            ))) if self.param_grid else 1
        )
        # Effective alpha list (explicit or derived from alpha_range_log)
        import numpy as _np
        if self.alpha_range_log is not None:
            lo, hi, n = self.alpha_range_log
            _alphas    = list(_np.logspace(_np.log10(lo), _np.log10(hi), int(n)))
            alpha_str  = f"log-spaced {lo}…{hi} ×{n} → {[round(a, 6) for a in _alphas]}"
        else:
            alpha_str  = str(self.param_grid.get("alpha", "(from param_grid)"))

        swc_str      = str(self.swc_depths) if self.swc_depths is not None else "auto (layer centres)"
        total_probes = self.num_swc_positions * self.n_layers

        lines = [
            f"  Scenario        : {self.name}",
            f"  Description     : {self.description}",
            f"  Data source     : {self.data_source.upper()}",
            f"  Output dir      : {self.output_dir}",
            f"  Layers          : {self.n_layers}  (depths0 = {self.depths0})",
            f"  Interfaces      : {fd_str}",
            f"  Forward model   : {self.forward_model}",
            f"  Gamma (TL)      : {self.gamma}",
            f"  Param grid      : {self.param_grid}",
            f"  Alpha sweep     : {alpha_str}",
            f"  Cond ranges     : {self.cond_ranges}",
            f"  Cache tag       : {self.cache_tag}  →  {self.cache_filename}",
            f"  ── SWC sensors ──────────────────────────────────────",
            f"  Positions       : {self.num_swc_positions}",
            f"  Depths          : {swc_str}",
            f"  Window (m)      : {self.window_m}",
            f"  Total probes    : {total_probes}  ({self.num_swc_positions} pos × {self.n_layers} layers)",
            f"  Matching gate   : warn>{self.matching_warn_fraction*100:.0f}%"
            + (f", error>{self.matching_error_fraction*100:.0f}%" if self.matching_error_fraction is not None else ", error=(disabled)"),
            f"  ── Noise ────────────────────────────────────────────",
            f"  TDR noise σ     : {self.tdr_noise_std} m³/m³",
            f"  ECa noise       : {self.eca_noise_level * 100:.1f} %",
            f"  Homogeneous     : {self.homogeneous}"
            + ("  (spatial gradients disabled, temporal trend + noise kept)" if self.homogeneous else ""),
        ]
        if self.data_source == "real":
            lines += [
                f"  ── Real data ────────────────────────────────────────",
                f"  EM directory    : {self.em_data_dir}",
                f"  TDR directory   : {self.tdr_data_dir}",
                f"  Coordinates     : {self.coord_file}",
                f"  Time tolerance  : ±{self.time_tolerance_days} days",
                f"  EM unit         : {self.em_unit}  "
                f"({'GEM-2 / ppm sensor' if self.em_unit == 'ppm' else 'GF Instruments / ppt sensor'})",
            ]
        lines += [
            f"  Drop coils      : {self.drop_coils if self.drop_coils else '(none)'}",
            f"  WS fixed params : {self.ws_fixed_params if self.ws_fixed_params else '(none — all 4 free)'}",
            f"  WS lab_p0       : {self.lab_p0 if self.lab_p0 else '(none — using DEFAULT_WS_P0 = [1.5, 1.8, 50.0, 5.0])'}",
            f"  ── Quality gates ────────────────────────────────────",
            f"  CV mean warn    : {self.quality_gate_cv_mean_warn}",
            f"  CV high thresh  : {self.quality_gate_cv_high_threshold}",
            f"  CV high frac warn: {self.quality_gate_cv_high_fraction_warn}",
            f"  Require all horizons calibrated: {self.quality_gate_require_all_horizons_calibrated}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_REQUIRED = {"depths0", "fixed_depths", "param_grid", "cond_ranges"}
_SKIP_KEYS = {"_comment", "_schema"}
_VALID_METHODS = {"L-BFGS-B", "CG", "Nelder-Mead", "SLSQP", "TNC","MCMC"}
_VALID_FORWARD_MODELS = {"CS", "FSlin", "FS", "QP"}


def _validate(name: str, raw: dict) -> None:
    """Raise ValueError with a descriptive message on any schema violation."""
    missing = _REQUIRED - raw.keys()
    if missing:
        raise ValueError(
            f"Scenario '{name}' is missing required field(s): {sorted(missing)}"
        )

    depths0 = raw["depths0"]
    if not isinstance(depths0, list) or not all(isinstance(d, (int, float)) for d in depths0):
        raise ValueError(
            f"Scenario '{name}': depths0 must be a list of numbers, got {depths0!r}."
        )
    if len(depths0) == 0:
        raise ValueError(
            f"Scenario '{name}': depths0 must contain at least one interface depth."
        )
    if depths0 != sorted(depths0):
        raise ValueError(
            f"Scenario '{name}': depths0 must be strictly increasing, got {depths0}."
        )

    fd = raw["fixed_depths"]
    if not isinstance(fd, (bool, list)):
        raise ValueError(
            f"Scenario '{name}': fixed_depths must be a bool or list[bool], got {fd!r}."
        )
    if isinstance(fd, list):
        if len(fd) != len(depths0):
            raise ValueError(
                f"Scenario '{name}': fixed_depths list has {len(fd)} entries but "
                f"depths0 has {len(depths0)} interfaces; lengths must match."
            )
        if not all(isinstance(f, bool) for f in fd):
            raise ValueError(
                f"Scenario '{name}': all elements of fixed_depths list must be bool."
            )

    cond_ranges = raw["cond_ranges"]
    n_layers = len(depths0) + 1
    if not isinstance(cond_ranges, list) or len(cond_ranges) != n_layers:
        n_got = len(cond_ranges) if isinstance(cond_ranges, list) else type(cond_ranges).__name__
        raise ValueError(
            f"Scenario '{name}': cond_ranges must have exactly {n_layers} entries "
            f"(one per layer = len(depths0)+1 = {len(depths0)}+1), but got {n_got}.\n"
            f"  depths0     = {depths0}  ->  {n_layers} layers\n"
            f"  cond_ranges has {n_got} entries\n"
            "Tip: whenever you change depths0, update cond_ranges to match. "
            "Each entry is [min_mS_m, max_mS_m] or [single_mS_m]."
        )
    for i, rng in enumerate(cond_ranges):
        if not isinstance(rng, list) or len(rng) not in (1, 2):
            raise ValueError(
                f"Scenario '{name}': cond_ranges[{i}] must be [value] or [min, max], "
                f"got {rng!r}."
            )
        if len(rng) == 2 and rng[0] > rng[1]:
            raise ValueError(
                f"Scenario '{name}': cond_ranges[{i}] min ({rng[0]}) > max ({rng[1]})."
            )

    param_grid = raw["param_grid"]
    if not isinstance(param_grid, dict):
        raise ValueError(
            f"Scenario '{name}': param_grid must be a dict, got {type(param_grid).__name__}."
        )
    for key, vals in param_grid.items():
        if not isinstance(vals, list) or len(vals) == 0:
            raise ValueError(
                f"Scenario '{name}': param_grid['{key}'] must be a non-empty list."
            )
        if key == "method":
            invalid = [v for v in vals if v not in _VALID_METHODS]
            if invalid:
                raise ValueError(
                    f"Scenario '{name}': param_grid['method'] contains unknown "
                    f"optimiser(s) {invalid}. Valid: {sorted(_VALID_METHODS)}."
                )
        if key == "forwardModel":
            invalid = [v for v in vals if v not in _VALID_FORWARD_MODELS]
            if invalid:
                raise ValueError(
                    f"Scenario '{name}': param_grid['forwardModel'] contains unknown "
                    f"model(s) {invalid}. Valid: {sorted(_VALID_FORWARD_MODELS)}."
                )

    gamma = raw.get("gamma", 0.0)
    if not isinstance(gamma, (int, float)) or gamma < 0:
        raise ValueError(
            f"Scenario '{name}': gamma must be a non-negative number, got {gamma!r}."
        )

    # ---- Real-data mode fields ----------------------------------------
    data_source = raw.get("data_source", "synthetic")
    if data_source not in ("synthetic", "real"):
        raise ValueError(
            f"Scenario '{name}': data_source must be 'synthetic' or 'real', "
            f"got {data_source!r}."
        )
    if data_source == "real":
        for field_name in ("em_data_dir", "tdr_data_dir", "coord_file"):
            val = raw.get(field_name)
            if not val or not isinstance(val, str):
                raise ValueError(
                    f"Scenario '{name}': data_source is 'real' but "
                    f"'{field_name}' is missing or empty.  Set it to the "
                    f"path of your {field_name.replace('_', ' ')}."
                )
        ttd = raw.get("time_tolerance_days", 1.0)
        if not isinstance(ttd, (int, float)) or ttd <= 0:
            raise ValueError(
                f"Scenario '{name}': time_tolerance_days must be a positive "
                f"number, got {ttd!r}."
            )

    em_unit = raw.get("em_unit", "ppt")
    if em_unit not in ("ppm", "ppt"):
        raise ValueError(
            f"Scenario '{name}': em_unit must be 'ppm' or 'ppt', "
            f"got {em_unit!r}.  Use 'ppm' for GEM-2 / CMD-Explorer data, "
            f"'ppt' for GF Instruments CMD series."
        )

    drop_coils = raw.get("drop_coils", None)
    if drop_coils is not None:
        _dc_items = [drop_coils] if isinstance(drop_coils, (str, int, float)) else drop_coils
        if not isinstance(_dc_items, list) or not all(
            isinstance(c, (str, int, float)) for c in _dc_items
        ):
            raise ValueError(
                f"Scenario '{name}': drop_coils must be null, a single "
                f"str/number, or a list of str/number coil specs "
                f"(coil name, frequency in Hz, or unique substring), "
                f"got {drop_coils!r}."
            )

    ws_fixed_params = raw.get("ws_fixed_params", None)
    if ws_fixed_params is not None:
        _valid_ws_keys = {"a", "m", "sigma_w", "sigma_s"}
        if not isinstance(ws_fixed_params, dict):
            raise ValueError(
                f"Scenario '{name}': ws_fixed_params must be null or an object "
                f"mapping a subset of {sorted(_valid_ws_keys)} to true/false, "
                f"got {ws_fixed_params!r}."
            )
        _bad_keys = set(ws_fixed_params.keys()) - _valid_ws_keys
        if _bad_keys:
            raise ValueError(
                f"Scenario '{name}': ws_fixed_params has unknown key(s) "
                f"{sorted(_bad_keys)}. Valid keys: {sorted(_valid_ws_keys)}."
            )
        if not all(isinstance(v, bool) for v in ws_fixed_params.values()):
            raise ValueError(
                f"Scenario '{name}': ws_fixed_params values must all be "
                f"true/false, got {ws_fixed_params!r}."
            )

    lab_p0 = raw.get("lab_p0", None)
    if lab_p0 is not None:
        _lab_p0_err = (
            f"Scenario '{name}': lab_p0 must be null, a flat 4-element list "
            f"[a, m, sigma_w, sigma_s], or a list of {n_layers} such "
            f"4-element lists (one per layer, n_layers = len(depths0)+1 = "
            f"{n_layers}), got {lab_p0!r}."
        )
        if not isinstance(lab_p0, list) or len(lab_p0) == 0:
            raise ValueError(_lab_p0_err)
        if all(isinstance(v, (int, float)) for v in lab_p0):
            # Flat [a, m, sigma_w, sigma_s] form
            if len(lab_p0) != 4:
                raise ValueError(_lab_p0_err)
        elif all(isinstance(row, list) for row in lab_p0):
            # Per-horizon [[a, m, sigma_w, sigma_s], ...] form
            if len(lab_p0) != n_layers or not all(
                len(row) == 4 and all(isinstance(v, (int, float)) for v in row)
                for row in lab_p0
            ):
                raise ValueError(_lab_p0_err)
        else:
            raise ValueError(_lab_p0_err)

    # ---- Quality gates (report item 4) ---------------------------------
    for _field_name, _lo, _hi in [
        ("quality_gate_cv_mean_warn", 0.0, None),
        ("quality_gate_cv_high_threshold", 0.0, None),
        ("quality_gate_cv_high_fraction_warn", 0.0, 1.0),
    ]:
        _val = raw.get(_field_name, None)
        if _val is not None:
            if (not isinstance(_val, (int, float)) or _val < _lo
                    or (_hi is not None and _val > _hi)):
                _range_str = f"[{_lo}, {_hi}]" if _hi is not None else f">= {_lo}"
                raise ValueError(
                    f"Scenario '{name}': {_field_name} must be a number "
                    f"{_range_str}, got {_val!r}."
                )
    _req_calib = raw.get("quality_gate_require_all_horizons_calibrated", None)
    if _req_calib is not None and not isinstance(_req_calib, bool):
        raise ValueError(
            f"Scenario '{name}': quality_gate_require_all_horizons_calibrated "
            f"must be a bool, got {_req_calib!r}."
        )

    # ---- SWC sensor network -------------------------------------------
    num_pos = raw.get("num_swc_positions", 4)
    if not isinstance(num_pos, int) or num_pos < 1:
        raise ValueError(
            f"Scenario '{name}': num_swc_positions must be a positive integer, "
            f"got {num_pos!r}."
        )

    swc_depths = raw.get("swc_depths", None)
    if swc_depths is not None:
        if not isinstance(swc_depths, list) or not all(
            isinstance(d, (int, float)) and d > 0 for d in swc_depths
        ):
            raise ValueError(
                f"Scenario '{name}': swc_depths must be a list of positive numbers "
                f"or null, got {swc_depths!r}."
            )

    window_m = raw.get("window_m", 1.0)
    if not isinstance(window_m, (int, float)) or window_m <= 0:
        raise ValueError(
            f"Scenario '{name}': window_m must be a positive number, got {window_m!r}."
        )

    matching_warn_fraction = raw.get("matching_warn_fraction", 0.5)
    if not isinstance(matching_warn_fraction, (int, float)) or not (0.0 < matching_warn_fraction <= 1.0):
        raise ValueError(
            f"Scenario '{name}': matching_warn_fraction must be in (0, 1], "
            f"got {matching_warn_fraction!r}."
        )

    matching_error_fraction = raw.get("matching_error_fraction", None)
    if matching_error_fraction is not None:
        if not isinstance(matching_error_fraction, (int, float)) or not (0.0 < matching_error_fraction <= 1.0):
            raise ValueError(
                f"Scenario '{name}': matching_error_fraction must be null or in "
                f"(0, 1], got {matching_error_fraction!r}."
            )
        if matching_error_fraction < matching_warn_fraction:
            raise ValueError(
                f"Scenario '{name}': matching_error_fraction ({matching_error_fraction}) "
                f"must be >= matching_warn_fraction ({matching_warn_fraction}), or every "
                "run would hit the error gate before the warn gate ever fires."
            )

    # ---- Noise levels -------------------------------------------------
    tdr_noise = raw.get("tdr_noise_std", 0.01)
    if not isinstance(tdr_noise, (int, float)) or tdr_noise < 0:
        raise ValueError(
            f"Scenario '{name}': tdr_noise_std must be >= 0, got {tdr_noise!r}."
        )

    eca_noise = raw.get("eca_noise_level", 0.0)
    if not isinstance(eca_noise, (int, float)) or not (0.0 <= eca_noise <= 1.0):
        raise ValueError(
            f"Scenario '{name}': eca_noise_level must be in [0, 1], got {eca_noise!r}."
        )

    homogeneous = raw.get("homogeneous", False)
    if not isinstance(homogeneous, bool):
        raise ValueError(
            f"Scenario '{name}': homogeneous must be a bool, got {homogeneous!r}."
        )

    # ---- Ensemble spread: alpha log-sweep -----------------------------
    alpha_range = raw.get("alpha_range_log", None)
    if alpha_range is not None:
        if (
            not isinstance(alpha_range, list)
            or len(alpha_range) != 3
            or not all(isinstance(v, (int, float)) for v in alpha_range)
            or alpha_range[0] <= 0
            or alpha_range[1] <= 0
            or alpha_range[0] >= alpha_range[1]
            or not isinstance(alpha_range[2], int)
            or alpha_range[2] < 2
        ):
            raise ValueError(
                f"Scenario '{name}': alpha_range_log must be [min>0, max>min, n_steps>=2], "
                f"got {alpha_range!r}."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_scenario(
    scenario_name: str = "default",
    config_path: str | None = None,
) -> ScenarioConfig:
    """
    Load, validate, and return a single named scenario from the JSON config.

    Parameters
    ----------
    scenario_name : str
        Key to look up in the JSON file.  Use ``list_scenarios()`` to see
        what is available.  Defaults to ``'default'``.
    config_path : str, optional
        Path to the JSON scenario file.  When omitted, looks for
        ``ensemble_scenarios.json`` in the same directory as this module.

    Returns
    -------
    ScenarioConfig
        Fully validated, typed configuration object.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    KeyError
        If ``scenario_name`` is not present in the file.
    ValueError
        If the scenario definition fails any validation check.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ensemble_scenarios.json",
        )

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Scenario config not found at '{config_path}'. "
            "Create or copy 'ensemble_scenarios.json' next to this script."
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        all_scenarios = json.load(fh)

    # Strip meta-keys
    scenarios = {k: v for k, v in all_scenarios.items() if k not in _SKIP_KEYS}

    if scenario_name not in scenarios:
        available = sorted(scenarios.keys())
        raise KeyError(
            f"Scenario '{scenario_name}' not found in '{config_path}'. "
            f"Available scenarios: {available}"
        )

    raw = scenarios[scenario_name]
    _validate(scenario_name, raw)

    # Build the cache tag: use explicit value or fall back to scenario name
    cache_tag = raw.get("cache_tag", scenario_name)

    # Apply alpha_range_log: override param_grid["alpha"] with a log sweep
    import numpy as _np
    param_grid = {k: list(v) for k, v in raw["param_grid"].items()}
    alpha_range_log = raw.get("alpha_range_log", None)
    if alpha_range_log is not None:
        lo, hi, n = alpha_range_log
        param_grid["alpha"] = list(_np.logspace(_np.log10(lo), _np.log10(hi), int(n)))

    # Coil exclusion: accept a single str/int/float or a list thereof in JSON
    _drop_coils_raw = raw.get("drop_coils", None)
    if _drop_coils_raw is None:
        drop_coils = None
    elif isinstance(_drop_coils_raw, (str, int, float)):
        drop_coils = [_drop_coils_raw]
    else:
        drop_coils = list(_drop_coils_raw)

    return ScenarioConfig(
        name              = scenario_name,
        description       = raw.get("description", ""),
        output_dir        = raw.get("output_dir", f"mfc2_outputs/{scenario_name}"),
        realization_index = int(raw.get("realization_index", 0)),
        depths0           = [float(d) for d in raw["depths0"]],
        fixed_depths      = (
            raw["fixed_depths"] if isinstance(raw["fixed_depths"], bool)
            else [bool(f) for f in raw["fixed_depths"]]
        ),
        param_grid        = param_grid,
        cond_ranges       = [[float(x) for x in rng] for rng in raw["cond_ranges"]],
        forward_model     = str(raw.get("forward_model", "CS")),
        gamma             = float(raw.get("gamma", 0.0)),
        cache_tag         = cache_tag,
        # SWC sensor network
        num_swc_positions = int(raw.get("num_swc_positions", 4)),
        swc_depths        = (
            [float(d) for d in raw["swc_depths"]]
            if raw.get("swc_depths") is not None else None
        ),
        window_m          = float(raw.get("window_m", 1.0)),
        matching_warn_fraction  = float(raw.get("matching_warn_fraction", 0.5)),
        matching_error_fraction = (
            float(raw["matching_error_fraction"])
            if raw.get("matching_error_fraction") is not None else None
        ),
        # Noise
        tdr_noise_std     = float(raw.get("tdr_noise_std", 0.01)),
        eca_noise_level   = float(raw.get("eca_noise_level", 0.0)),
        homogeneous       = bool(raw.get("homogeneous", False)),
        # Ensemble spread
        alpha_range_log   = alpha_range_log,
        # 2-D grid
        grid_num_x        = int(raw.get("grid_num_x", 10)),
        grid_num_y        = int(raw.get("grid_num_y", 5)),
        # Real-data mode
        data_source       = str(raw.get("data_source", "synthetic")),
        em_data_dir       = raw.get("em_data_dir", None),
        tdr_data_dir      = raw.get("tdr_data_dir", None),
        coord_file        = raw.get("coord_file", None),
        time_tolerance_days = float(raw.get("time_tolerance_days", 1.0)),
        em_unit           = str(raw.get("em_unit", "ppt")),
        drop_coils        = drop_coils,
        quality_gate_cv_mean_warn = float(raw.get("quality_gate_cv_mean_warn", 0.3)),
        quality_gate_cv_high_threshold = float(raw.get("quality_gate_cv_high_threshold", 1.0)),
        quality_gate_cv_high_fraction_warn = float(raw.get("quality_gate_cv_high_fraction_warn", 0.3)),
        quality_gate_require_all_horizons_calibrated = bool(
            raw.get("quality_gate_require_all_horizons_calibrated", True)
        ),
        ws_fixed_params   = raw.get("ws_fixed_params", None),
        lab_p0            = (
            [[float(v) for v in row] for row in raw["lab_p0"]]
            if (raw.get("lab_p0") is not None and isinstance(raw["lab_p0"][0], list))
            else ([float(v) for v in raw["lab_p0"]] if raw.get("lab_p0") is not None else None)
        ),
    )


def list_scenarios(config_path: str | None = None) -> list[str]:
    """
    Return the names of all scenarios defined in the JSON file.

    Parameters
    ----------
    config_path : str, optional
        Path to the JSON file.  Defaults to ``ensemble_scenarios.json``
        next to this module.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ensemble_scenarios.json",
        )
    with open(config_path, "r", encoding="utf-8") as fh:
        all_scenarios = json.load(fh)
    return [k for k in all_scenarios if k not in _SKIP_KEYS]


def validate_all(config_path: str | None = None) -> None:
    """
    Validate every scenario in the file and print a pass/fail report.
    Useful as a pre-run sanity check.

        python -c "from scenario_loader import validate_all; validate_all()"
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ensemble_scenarios.json",
        )
    with open(config_path, "r", encoding="utf-8") as fh:
        all_scenarios = json.load(fh)
    scenarios = {k: v for k, v in all_scenarios.items() if k not in _SKIP_KEYS}
    print(f"Validating {len(scenarios)} scenario(s) in '{config_path}':")
    all_ok = True
    for name, raw in scenarios.items():
        try:
            _validate(name, raw)
            print(f"  ✓  {name}")
        except (ValueError, TypeError) as exc:
            print(f"  ✗  {name}: {exc}")
            all_ok = False
    if all_ok:
        print("All scenarios valid.")
    else:
        print("One or more scenarios failed validation.")


# ---------------------------------------------------------------------------
# CLI entry point — python scenario_loader.py [scenario_name]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        validate_all()
    elif sys.argv[1] == "--list":
        for name in list_scenarios():
            print(name)
    else:
        cfg = load_scenario(sys.argv[1])
        print(cfg.summary())
