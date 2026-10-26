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

import bisect
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
        ``generate_ensemble_em_survey``.  Recognised keys: alpha, beta,
        method, forwardModel. ``beta`` is EMagPy's lateral (within-epoch,
        neighbouring-profile) smoothing weight -- the spatial counterpart
        to ``alpha`` (per-profile vertical smoothing) and ``gamma``
        (temporal, time-lapse-only smoothing, see the ``gamma`` field
        below). Defaults to ``[0.0]`` (no lateral smoothing) when omitted,
        matching EMagPy's own ``invert(beta=0.0, ...)`` default.
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

    full_range_gradient: bool          = False
    """Synthetic generation only (ignored when homogeneous is True). When
    True, the static XZ (depth) and XY (plan-view) initial-model gradients
    each span the full [theta_dry, theta_wet] range independently, instead
    of the small fixed-amplitude default gradient -- see
    0_generate_synthetic_data.generate_time_lapse_dataset. Combined via
    xy_z_blend below. Default False (small default gradient, unchanged)."""

    xy_z_blend: float                  = 0.5
    """Only used when full_range_gradient is True. Weight in [0, 1] between
    the XY component (1.0 = XY gradient only) and the Z/depth component
    (0.0 = Z gradient only) of the initial model. Default 0.5 (equal
    blend)."""

    invert_z_gradient: bool            = False
    """Only used when full_range_gradient is True. When True, flips the
    depth gradient so the shallowest layer is wet and the deepest is dry
    (i.e. soil dries out uniformly with depth) -- default is dry at the
    surface, wet at depth."""

    random_xy_field: bool              = False
    """Only used when full_range_gradient is True. When True, replaces the
    deterministic linear XY ramp with a smooth, spatially-correlated RANDOM
    field (scattered wet/dry patches, generated as a sum of random Gaussian
    blobs) instead of a single monotonic gradient direction -- see
    0_generate_synthetic_data._generate_random_xy_field. One field is drawn
    per ensemble member and held fixed across all timesteps/layers within
    that member; only the (still spatially-uniform) depth trend and small
    temporal/noise terms continue to vary. Default False (linear ramp,
    unchanged)."""

    random_xy_n_blobs: int             = 8
    """Only used when random_xy_field is True. Number of random Gaussian
    patches summed into the field. More blobs -> busier, more fragmented
    pattern; fewer -> a handful of broad wet/dry zones. Default 8."""

    random_xy_correlation_m: float | None = None
    """Only used when random_xy_field is True. Patch radius [m] (1-sigma).
    None (default) auto-picks ~1/4 of the survey's shorter horizontal
    extent."""

    random_xy_seed: int | None         = None
    """Only used when random_xy_field is True. Seeds the random XY field
    independently of the per-sample noise stream, so the pattern is
    reproducible run-to-run. None (default) derives a seed from 42."""

    ws_xy_variation: bool              = False
    """When True, the TRUTH Waxman-Smits params (a, m, sigma_w, sigma_s)
    vary spatially in XY instead of being spatially uniform, independent
    of the theta gradient above (random_xy_field/full_range_gradient).
    This is the ground-truth petrophysics used to generate the synthetic
    data, not the Step-2 calibration fit. Default False (spatially
    uniform, unchanged)."""

    ws_xy_params: list | None          = None
    """Only used when ws_xy_variation is True. Subset of ["a", "m",
    "sigma_w", "sigma_s"] that varies in XY; every other param stays
    fixed at its lab default. None (default) -> ["m", "sigma_s"]."""

    ws_xy_ranges: dict | None          = None
    """Only used when ws_xy_variation is True. Per-param {"param":
    [min, max]} truth range spanned across the XY field. None (default)
    -> a [0.3,0.7], m [2.0,4.0], sigma_w [300,700], sigma_s [50,200]."""

    ws_xy_pattern: str                 = "linear"
    """Only used when ws_xy_variation is True. "linear" (diagonal x/y
    ramp, same convention as theta) or "random" (smooth spatially-
    correlated random field, independently seeded from the moisture
    pattern). Default "linear"."""

    ws_xy_n_blobs: int                 = 6
    """Only used when ws_xy_pattern is "random". Number of random
    Gaussian patches summed into the WS-parameter field. Default 6."""

    ws_xy_correlation_m: float | None  = None
    """Only used when ws_xy_pattern is "random". Patch radius [m]
    (1-sigma). None (default) auto-picks."""

    ws_xy_seed: int | None             = None
    """Only used when ws_xy_pattern is "random". Seeds the WS-parameter
    field independently of random_xy_seed. None (default) derives a
    seed from 99."""

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

    ws_localized_bandwidth: float | None = None
    """Gaspari-Cohn localization radius 'c' (spatial_calibration_
    alternatives.md §2), same units as the transect x-coordinates (metres
    for UTM/real-data grids). When set, run_pipeline.py additionally runs
    step2.run_petro_localized as a DIAGNOSTIC sibling to the normal Step 2
    calibration -- it does not replace fits_step2 and does not change
    Steps 3-5, which still consume the scalar-per-horizon fit. Produces
    petro_calibration_localized.png (sigma_s(x) per horizon) in output_dir.
    Kernel support is exactly zero beyond 2*c, so c should be at least
    roughly half the spacing between adjacent TDR probes, or every target
    point can fall outside kernel support of any observation. None
    (default) skips the localized diagnostic entirely, unchanged from
    before this field existed."""

    ws_localized_bandwidth_sweep: list[float] | None = None
    """Optional sequence of candidate Gaspari-Cohn bandwidths (same units
    as ws_localized_bandwidth) to evaluate via leave-one-probe-out CV
    (step2.run_bandwidth_sweep) before running the localized diagnostic.
    Evaluated once against the rank-1 (best-RMSE) Step 2 ensemble member.
    The recommended bandwidth overrides ws_localized_bandwidth for the
    localized diagnostic call. None (default) disables the sweep and
    uses ws_localized_bandwidth as-is (or skips the diagnostic if that
    is also None)."""

    # ------------------------------------------------------------------
    # Real-data time-lapse spatial reference grid
    # (pipeline_review.md / real_data_pipeline_fixes.md, issue 2:
    # "spatial reference-grid matching, not row trimming")
    # ------------------------------------------------------------------
    spatial_match_tolerance_m: float = 2.0
    """Real-data time-lapse only. Distance [m] within which two survey
    epochs' measurement points are considered "the same physical location"
    when building the shared spatial reference grid used to assemble
    models_clean across epochs (see
    geometry_utils.match_epochs_to_reference_grid and
    run_pipeline.py Step 1). This REPLACES the old behaviour of trimming
    every epoch to the globally-shortest epoch's point COUNT
    (`arr[:min_pts]`), which silently paired unrelated physical locations
    whenever epochs weren't walked in identical order.  Distinct from
    `window_m` (which governs TDR-probe <-> EM-point matching, a different
    question). Should be set to roughly the GPS/GNSS precision of the EM
    instrument used for real-data surveys; a few metres is typical.
    Ignored for synthetic data (point counts are consistent by
    construction) and for real single-survey (non-time-lapse) runs."""

    # ------------------------------------------------------------------
    # Temperature correction
    # FIX ("temperature correction is currently a complete no-op"):
    # run_pipeline.py previously hardcoded the temperature fed into
    # em_methodology.apply_temperature_correction to exactly T_ref, so the
    # correction factor 1 + alpha*(T - T_ref) was always exactly 1 --
    # the step ran, was logged, and corrected nothing, for every scenario.
    # ------------------------------------------------------------------
    soil_temperature_c: float | list | None = None
    """Explicit soil temperature [deg C] for the temperature correction
    applied to inverted EC before petrophysical calibration (see
    em_methodology.apply_temperature_correction, Eq. 1 of Loiseau et al.).
    Accepts a scalar (uniform temperature for the whole run) or a list
    broadcastable against the model shape per
    apply_temperature_correction's own docstring (e.g. one value per
    layer, or one value per time step for time-lapse data).

    When set, this OVERRIDES any measured per-epoch temperature
    auto-detected from real TDR data (a column containing 'temp' in its
    name -- see run_pipeline.py's _extract_temperature_by_epoch).

    None (default):
      - real-data scenarios fall back to the auto-detected per-epoch
        measured temperature if a 'temp'-named column is found in the TDR
        data; if none is found, the correction is explicitly SKIPPED
        (identity -- NOT silently applied as a no-op the way it used to
        be) and logged as such.
      - synthetic scenarios always skip the correction (there is no
        measured temperature to fall back to); set this explicitly to
        test the correction's effect on synthetic data.
    """

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

    # ------------------------------------------------------------------
    # Train / validation split (Loiseau et al., 2026 Secs. 2.3/2.6)
    # ------------------------------------------------------------------
    train_val_split: str = "off"
    """Split TDR/SWC observations into a training subset (drives Step 2/4
    Waxman-Smits fitting and Step 3/5 ensemble-K selection) and a
    validation subset (held out untouched, reported once in Step 5b) --
    see train_val_split.py. One of 'alternate_sorted' (paper's Larzac
    fallback -- NOT fully independent, both subsets can share a probe or
    time step), 'by_probe' (fully independent, needs a per-observation
    probe id; real-data mode only), 'by_time_block' (fully independent
    w.r.t. time; paper's Avignon approach), 'random' (comparison only), or
    'off' (default -- legacy behaviour, fit and select K on the full
    observation set). A CLI --train-val-split flag, if given, overrides
    this field."""

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
            f"  Full-range grad.: {self.full_range_gradient}"
            + (
                f"  (xy_z_blend={self.xy_z_blend}, "
                f"invert_z={self.invert_z_gradient}, "
                f"random_xy={self.random_xy_field}"
                + (
                    f" [n_blobs={self.random_xy_n_blobs}, "
                    f"corr_m={self.random_xy_correlation_m}, "
                    f"seed={self.random_xy_seed}]"
                    if self.random_xy_field else ""
                )
                + ")"
                if self.full_range_gradient else ""
            ),
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
                f"  Spatial match tol (time-lapse epochs): ±{self.spatial_match_tolerance_m} m",
            ]
        lines += [
            f"  Drop coils      : {self.drop_coils if self.drop_coils else '(none)'}",
            f"  WS fixed params : {self.ws_fixed_params if self.ws_fixed_params else '(none — all 4 free)'}",
            f"  WS xy variation : {self.ws_xy_variation}"
            + (
                f"  (params={self.ws_xy_params if self.ws_xy_params else ['m', 'sigma_s']}, "
                f"ranges={self.ws_xy_ranges if self.ws_xy_ranges else '(defaults)'}, "
                f"pattern={self.ws_xy_pattern}"
                + (
                    f" [n_blobs={self.ws_xy_n_blobs}, "
                    f"corr_m={self.ws_xy_correlation_m}, "
                    f"seed={self.ws_xy_seed}]"
                    if self.ws_xy_pattern == "random" else ""
                )
                + ")"
                if self.ws_xy_variation else ""
            ),
            f"  WS lab_p0       : {self.lab_p0 if self.lab_p0 else '(none — using DEFAULT_WS_P0 = [1.5, 1.8, 50.0, 5.0])'}",
            f"  WS localized c  : {self.ws_localized_bandwidth if self.ws_localized_bandwidth is not None else '(none — localized Step 2 diagnostic disabled)'}",
            f"  WS localized c sweep : {self.ws_localized_bandwidth_sweep if self.ws_localized_bandwidth_sweep is not None else '(none — fixed bandwidth used as-is)'}",
            f"  ── Quality gates ────────────────────────────────────",
            f"  CV mean warn    : {self.quality_gate_cv_mean_warn}",
            f"  CV high thresh  : {self.quality_gate_cv_high_threshold}",
            f"  CV high frac warn: {self.quality_gate_cv_high_fraction_warn}",
            f"  Require all horizons calibrated: {self.quality_gate_require_all_horizons_calibrated}",
            f"  ── Train/val split ──────────────────────────────────",
            f"  Method          : {self.train_val_split}"
            + ("  (legacy: fit + select K on the full observation set)" if self.train_val_split == "off" else ""),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_REQUIRED = {"depths0", "fixed_depths", "param_grid", "cond_ranges"}
_SKIP_KEYS = {"_comment", "_schema"}
# NOTE on this whitelist vs. EMagPy's actual Problem.invert(method=...) options:
# scipy-backed:  L-BFGS-B, TNC, CG, Nelder-Mead
# spotpy-backed (MCMC-family): MCMC, ROPE, DREAM, SCEUA
#   - only ROPE supports alpha/beta/gamma smoothing among the MCMC family;
#     MCMC/DREAM/SCEUA silently ignore it (see EMagPy docs).
# GN is also accepted by invert() (Gauss-Newton, distinct from the separate
# invertGN() helper) but isn't whitelisted here since this pipeline hasn't
# exercised it -- add it if/when needed.
# ANN is deliberately NOT included: it requires tensorflow and takes extra
# kwargs (nsample, noise, annplot) this pipeline's invert_kwargs plumbing
# doesn't currently populate.
# SLSQP is kept for backward compatibility with existing scenarios, though
# it isn't in EMagPy's own documented method list for invert() -- it's
# passed straight through to scipy.optimize.minimize(), which does support
# it, so it likely works, but treat it as unconfirmed/unofficial rather
# than an EMagPy-documented option.
_VALID_METHODS = {"L-BFGS-B", "CG", "Nelder-Mead", "SLSQP", "TNC",
                   "MCMC", "ROPE", "DREAM", "SCEUA"}
_VALID_FORWARD_MODELS = {"CS", "FSlin", "FS", "QP"}


def _horizon_index_from_depth(depth: float, depths0: list[float]) -> int:
    """
    0-based horizon index for a sensor installation depth, given the fixed
    interface depths ``depths0`` (length = n_layers - 1). Mirrors
    ``em_methodology.layer_index_from_depth`` exactly (horizon 0 spans
    ``[0, depths0[0])``, horizon i spans ``[depths0[i-1], depths0[i])``, the
    last horizon spans ``[depths0[-1], +inf)``) but is reimplemented locally
    so ``scenario_loader`` -- a lightweight, dependency-free config module --
    doesn't have to import ``em_methodology`` (and, transitively, ``emagpy``)
    just to validate a JSON scenario file.
    """
    return bisect.bisect_right(depths0, depth)


def _check_swc_depth_horizon_coverage(name: str, depths0: list[float],
                                       swc_depths: list[float] | None) -> None:
    """
    Cross-field validation (pipeline_improvement_report.md, section 8 /
    action item 6): catch a ``swc_depths`` configuration that leaves one or
    more soil horizons implied by ``depths0`` with NO sensor installed in
    them, at scenario-LOAD time -- i.e. before the (potentially 30+ minute)
    Step 1 ensemble inversion ever runs.

    Mirrors the horizon-routing logic used later by
    ``em_methodology.layer_index_from_depth`` /
    ``em_methodology.check_early_horizon_coverage``, but runs here on the
    static scenario config alone. When ``swc_depths`` is ``None`` (the
    common case -- the pipeline auto-derives one sensor depth per layer
    centre from ``depths0``), coverage is complete by construction and this
    is a no-op.

    Raises
    ------
    ValueError
        If any horizon has zero ``swc_depths`` entries mapped to it. An
        uncalibrated horizon isn't a soft degradation here: it silently
        falls back to prior/uncalibrated Waxman-Smits parameters for the
        entire run, so failing fast at load time (rather than only
        surfacing it as an "UNCALIBRATED" horizon deep in the Step 2/4
        output) is worth the strictness.
    """
    if swc_depths is None:
        return

    n_layers = len(depths0) + 1
    covered = {
        min(max(_horizon_index_from_depth(d, depths0), 0), n_layers - 1)
        for d in swc_depths
    }
    uncovered = sorted(set(range(n_layers)) - covered)
    if uncovered:
        raise ValueError(
            f"Scenario '{name}': swc_depths={swc_depths} do not reach all "
            f"{n_layers} horizon(s) implied by depths0={depths0}. "
            f"Horizon(s) {uncovered} have no sensor depth mapped to them and "
            f"can never be calibrated -- they will silently fall back to "
            f"prior/uncalibrated Waxman-Smits parameters in Steps 2/4 for "
            f"the entire run. Fix by either adding a sensor depth inside "
            f"each uncovered horizon's depth range, or by leaving "
            f"swc_depths unset (null) so the pipeline auto-derives one "
            f"depth per layer centre (which always covers every horizon)."
        )


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

    # ---- Train/validation split (Loiseau et al., 2026 Secs. 2.3/2.6) --
    tvs_method = raw.get("train_val_split", "off")
    _valid_tvs_methods = ("alternate_sorted", "by_probe", "by_time_block", "random", "off")
    if tvs_method not in _valid_tvs_methods:
        raise ValueError(
            f"Scenario '{name}': train_val_split must be one of "
            f"{_valid_tvs_methods}, got {tvs_method!r}."
        )
    if tvs_method == "by_probe" and data_source != "real":
        # split_train_val() would just fall back to alternate_sorted with a
        # warning at runtime (synthetic mode has no genuine per-observation
        # probe id), but failing loudly at scenario-load time is clearer
        # than a run silently not doing what the config says it's doing.
        raise ValueError(
            f"Scenario '{name}': train_val_split='by_probe' needs a real "
            f"per-observation probe id, which only exists for "
            f"data_source='real'. Use 'alternate_sorted', 'by_time_block', "
            f"or 'random' for synthetic scenarios."
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
        # FIX (em_unit defaults to 'ppt' but real GEM-2 data is 'ppm'):
        # this pipeline's real-data branch (0_real_data_loader.py's
        # EM_HEADER_MAP / run_pipeline.py's _COLUMN_VARIANTS) only ever
        # ingests GEM-2 multi-frequency quadrature exports
        # (Q_425Hz, Q_1525Hz, ...) -- a 'ppm' instrument. Nothing enforced
        # that previously: the ScenarioConfig dataclass default and this
        # loader's own em_unit fallback below are both 'ppt' (the
        # GF-Instruments convention), so a real-data scenario that simply
        # omitted 'em_unit' silently inverted every ECa value ~1000x too
        # large. Require it to be present explicitly for data_source='real'
        # so a missing/forgotten em_unit fails loudly at scenario-load
        # time, before the (potentially 30+ minute) Step 1 inversion runs,
        # rather than silently corrupting the whole run's EC values.
        if "em_unit" not in raw:
            raise ValueError(
                f"Scenario '{name}': data_source is 'real' but 'em_unit' is not set. "
                f"This pipeline's real-data branch only ingests GEM-2 multi-frequency "
                f"quadrature exports (Q_425Hz, Q_1525Hz, ...), which are a 'ppm' "
                f"instrument -- NOT the GF-Instruments 'ppt' default this loader "
                f"otherwise falls back to. Leaving em_unit unset previously silently "
                f"inflated every inverted ECa value by ~1000x. Add "
                f'"em_unit": "ppm" to this scenario explicitly (or "ppt" only if you '
                f"have verified this scenario's data really is GF-Instruments-format)."
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

    ws_localized_bandwidth = raw.get("ws_localized_bandwidth", None)
    if ws_localized_bandwidth is not None:
        if not isinstance(ws_localized_bandwidth, (int, float)) or ws_localized_bandwidth <= 0:
            raise ValueError(
                f"Scenario '{name}': ws_localized_bandwidth must be null or a "
                f"positive number (Gaspari-Cohn radius c, same units as the "
                f"transect x-coordinates), got {ws_localized_bandwidth!r}."
            )

    ws_localized_bandwidth_sweep = raw.get("ws_localized_bandwidth_sweep", None)
    if ws_localized_bandwidth_sweep is not None:
        if (
            not isinstance(ws_localized_bandwidth_sweep, list)
            or len(ws_localized_bandwidth_sweep) < 2
            or not all(
                isinstance(v, (int, float)) and v > 0
                for v in ws_localized_bandwidth_sweep
            )
        ):
            raise ValueError(
                f"Scenario '{name}': ws_localized_bandwidth_sweep must be null "
                f"or a list of >= 2 positive numbers (candidate Gaspari-Cohn "
                f"radii c, same units as the transect x-coordinates), got "
                f"{ws_localized_bandwidth_sweep!r}."
            )

    # ---- Real-data time-lapse spatial reference grid -------------------
    spatial_tol = raw.get("spatial_match_tolerance_m", None)
    if spatial_tol is not None:
        if not isinstance(spatial_tol, (int, float)) or spatial_tol <= 0:
            raise ValueError(
                f"Scenario '{name}': spatial_match_tolerance_m must be a "
                f"positive number, got {spatial_tol!r}."
            )

    # ---- Temperature correction -----------------------------------------
    soil_temp = raw.get("soil_temperature_c", None)
    if soil_temp is not None:
        _soil_temp_err = (
            f"Scenario '{name}': soil_temperature_c must be null, a single "
            f"number, or a list of numbers, got {soil_temp!r}."
        )
        if isinstance(soil_temp, (int, float)) and not isinstance(soil_temp, bool):
            pass
        elif isinstance(soil_temp, list) and len(soil_temp) > 0 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in soil_temp
        ):
            pass
        else:
            raise ValueError(_soil_temp_err)

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

    # Cross-field check (report item 6 / section 8): swc_depths must reach
    # every horizon implied by depths0, or that horizon can never be
    # calibrated no matter how the ensemble inversion runs.
    if swc_depths is not None:
        _check_swc_depth_horizon_coverage(name, depths0, swc_depths)

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

    full_range_gradient = raw.get("full_range_gradient", False)
    if not isinstance(full_range_gradient, bool):
        raise ValueError(
            f"Scenario '{name}': full_range_gradient must be a bool, got "
            f"{full_range_gradient!r}."
        )

    xy_z_blend = raw.get("xy_z_blend", 0.5)
    if not isinstance(xy_z_blend, (int, float)) or not (0.0 <= xy_z_blend <= 1.0):
        raise ValueError(
            f"Scenario '{name}': xy_z_blend must be in [0, 1], got {xy_z_blend!r}."
        )

    for _bkey in ("invert_z_gradient", "random_xy_field"):
        _bval = raw.get(_bkey, False)
        if not isinstance(_bval, bool):
            raise ValueError(
                f"Scenario '{name}': {_bkey} must be a bool, got {_bval!r}."
            )

    random_xy_n_blobs = raw.get("random_xy_n_blobs", 8)
    if not isinstance(random_xy_n_blobs, int) or random_xy_n_blobs < 1:
        raise ValueError(
            f"Scenario '{name}': random_xy_n_blobs must be a positive int, "
            f"got {random_xy_n_blobs!r}."
        )

    random_xy_correlation_m = raw.get("random_xy_correlation_m", None)
    if random_xy_correlation_m is not None and (
        not isinstance(random_xy_correlation_m, (int, float)) or random_xy_correlation_m <= 0
    ):
        raise ValueError(
            f"Scenario '{name}': random_xy_correlation_m must be > 0 or null, "
            f"got {random_xy_correlation_m!r}."
        )

    ws_xy_variation = raw.get("ws_xy_variation", False)
    if not isinstance(ws_xy_variation, bool):
        raise ValueError(
            f"Scenario '{name}': ws_xy_variation must be a bool, got "
            f"{ws_xy_variation!r}."
        )

    _valid_ws_xy_params = {"a", "m", "sigma_w", "sigma_s"}
    ws_xy_params = raw.get("ws_xy_params", None)
    if ws_xy_params is not None:
        if not isinstance(ws_xy_params, list) or not all(
            isinstance(p, str) for p in ws_xy_params
        ):
            raise ValueError(
                f"Scenario '{name}': ws_xy_params must be null or a list of "
                f"strings from {sorted(_valid_ws_xy_params)}, got {ws_xy_params!r}."
            )
        _bad_ws_xy_params = set(ws_xy_params) - _valid_ws_xy_params
        if _bad_ws_xy_params:
            raise ValueError(
                f"Scenario '{name}': ws_xy_params has unknown Waxman-Smits "
                f"parameter(s) {sorted(_bad_ws_xy_params)}. Valid: "
                f"{sorted(_valid_ws_xy_params)}."
            )

    ws_xy_ranges = raw.get("ws_xy_ranges", None)
    if ws_xy_ranges is not None:
        if not isinstance(ws_xy_ranges, dict):
            raise ValueError(
                f"Scenario '{name}': ws_xy_ranges must be null or an object "
                f"mapping a subset of {sorted(_valid_ws_xy_params)} to "
                f"[min, max], got {ws_xy_ranges!r}."
            )
        _bad_range_keys = set(ws_xy_ranges.keys()) - _valid_ws_xy_params
        if _bad_range_keys:
            raise ValueError(
                f"Scenario '{name}': ws_xy_ranges has unknown key(s) "
                f"{sorted(_bad_range_keys)}. Valid keys: "
                f"{sorted(_valid_ws_xy_params)}."
            )
        for _pname, _prange in ws_xy_ranges.items():
            if (
                not isinstance(_prange, list) or len(_prange) != 2
                or not all(isinstance(x, (int, float)) for x in _prange)
                or _prange[0] >= _prange[1]
            ):
                raise ValueError(
                    f"Scenario '{name}': ws_xy_ranges[{_pname!r}] must be a "
                    f"[min, max] pair with min < max, got {_prange!r}."
                )

    ws_xy_pattern = raw.get("ws_xy_pattern", "linear")
    if ws_xy_pattern not in ("linear", "random"):
        raise ValueError(
            f"Scenario '{name}': ws_xy_pattern must be 'linear' or 'random', "
            f"got {ws_xy_pattern!r}."
        )

    ws_xy_n_blobs = raw.get("ws_xy_n_blobs", 6)
    if not isinstance(ws_xy_n_blobs, int) or ws_xy_n_blobs < 1:
        raise ValueError(
            f"Scenario '{name}': ws_xy_n_blobs must be a positive int, "
            f"got {ws_xy_n_blobs!r}."
        )

    ws_xy_correlation_m = raw.get("ws_xy_correlation_m", None)
    if ws_xy_correlation_m is not None and (
        not isinstance(ws_xy_correlation_m, (int, float)) or ws_xy_correlation_m <= 0
    ):
        raise ValueError(
            f"Scenario '{name}': ws_xy_correlation_m must be > 0 or null, "
            f"got {ws_xy_correlation_m!r}."
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
        full_range_gradient = bool(raw.get("full_range_gradient", False)),
        xy_z_blend        = float(raw.get("xy_z_blend", 0.5)),
        invert_z_gradient = bool(raw.get("invert_z_gradient", False)),
        random_xy_field   = bool(raw.get("random_xy_field", False)),
        random_xy_n_blobs = int(raw.get("random_xy_n_blobs", 8)),
        random_xy_correlation_m = (
            float(raw["random_xy_correlation_m"])
            if raw.get("random_xy_correlation_m") is not None else None
        ),
        random_xy_seed    = (
            int(raw["random_xy_seed"]) if raw.get("random_xy_seed") is not None else None
        ),
        ws_xy_variation   = bool(raw.get("ws_xy_variation", False)),
        ws_xy_params      = (
            list(raw["ws_xy_params"]) if raw.get("ws_xy_params") is not None else None
        ),
        ws_xy_ranges      = (
            {k: [float(v[0]), float(v[1])] for k, v in raw["ws_xy_ranges"].items()}
            if raw.get("ws_xy_ranges") is not None else None
        ),
        ws_xy_pattern     = str(raw.get("ws_xy_pattern", "linear")),
        ws_xy_n_blobs     = int(raw.get("ws_xy_n_blobs", 6)),
        ws_xy_correlation_m = (
            float(raw["ws_xy_correlation_m"])
            if raw.get("ws_xy_correlation_m") is not None else None
        ),
        ws_xy_seed        = (
            int(raw["ws_xy_seed"]) if raw.get("ws_xy_seed") is not None else None
        ),
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
        spatial_match_tolerance_m = float(raw.get("spatial_match_tolerance_m", 2.0)),
        soil_temperature_c = (
            float(raw["soil_temperature_c"])
            if isinstance(raw.get("soil_temperature_c"), (int, float))
            else ([float(v) for v in raw["soil_temperature_c"]]
                  if raw.get("soil_temperature_c") is not None else None)
        ),
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
        ws_localized_bandwidth = (
            float(raw["ws_localized_bandwidth"])
            if raw.get("ws_localized_bandwidth") is not None else None
        ),
        ws_localized_bandwidth_sweep = (
            [float(v) for v in raw["ws_localized_bandwidth_sweep"]]
            if raw.get("ws_localized_bandwidth_sweep") is not None else None
        ),
        train_val_split   = str(raw.get("train_val_split", "off")),
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
