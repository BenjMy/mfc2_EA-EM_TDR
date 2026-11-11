import numpy as np
import os
import pandas as pd
from emagpy import Problem

import geometry_utils

# Fixed coil set shared by the data generator, the exported CSV surveys, and
# anything downstream that needs to know which columns hold ECa readings.
COILS_LIST = ['VCP0.5', 'HCP0.5', 'VCP1.0', 'HCP1.0', 'VCP1.5', 'HCP1.5']

# ---------------------------------------------------------------------------
# Alento field-survey spatial reference (UTM Zone 33N, from 1_ALENTO_040820.csv)
# Used to anchor synthetic coordinates so they are geographically realistic.
# ---------------------------------------------------------------------------
#ALENTO_UTM_ORIGIN_X = 515487.23   # SW corner easting  [m]
#ALENTO_UTM_ORIGIN_Y = 4468201.66  # SW corner northing [m]
#ALENTO_TRANSECT_LENGTH_X = 239.0  # E-W extent of survey area [m]
#ALENTO_TRANSECT_LENGTH_Y = 310.4  # N-S extent of survey area [m]

ORIGIN_X = 0   # SW corner easting  [m]
ORIGIN_Y = 0  # SW corner northing [m]
TRANSECT_LENGTH_X = 100  # E-W extent of survey area [m]
TRANSECT_LENGTH_Y = 100  # N-S extent of survey area [m]

def _build_transect_x_coords(num_measured_points: int, point_spacing: float = 2.0) -> np.ndarray:
    """
    Single source of truth for the along-transect x-coordinates of the
    synthetic measurement points (1-D mode).
    """
    return np.arange(num_measured_points, dtype=float) * point_spacing


def _build_2d_grid_coords(
    num_x: int,
    num_y: int,
    point_spacing: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 2-D regular grid of UTM coordinates.
    """
    xs = np.arange(num_x, dtype=float) * point_spacing
    ys = np.arange(num_y, dtype=float) * point_spacing
    XX, YY = np.meshgrid(xs, ys)          # shape (num_y, num_x)
    x_local = XX.ravel()
    y_local = YY.ravel()
    utm_x = ORIGIN_X + x_local
    utm_y = ORIGIN_Y + y_local
    return x_local, y_local, utm_x, utm_y


def forward_waxman_smits(theta, a, m, sigma_w, sigma_s):
    """
    Ecuación directa de Waxman-Smits: Calcula EC_bulk (mS/m) a partir de Theta.
    EC_bulk = (theta**m / a) * (sigma_w + sigma_s)
    """
    theta = np.clip(theta, 0.01, 1.0)
    return (theta ** m / a) * (sigma_w + sigma_s)


def _generate_random_xy_field(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    n_blobs: int,
    correlation_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Smooth, spatially-correlated random field over the (x_coords, y_coords)
    measurement points, normalized to [0, 1] -- the "more random" XY
    counterpart to the deterministic linear ``xy_frac`` ramp used by the
    plain and ``full_range_gradient`` ratios.

    Built as a sum of ``n_blobs`` random Gaussian "patches" (random center,
    random sign/amplitude, fixed correlation length ``correlation_m``)
    rather than per-point white noise, so the result looks like plausible
    patchy soil texture/moisture heterogeneity (spatially smooth, with
    both wetter- and drier-than-average patches) instead of salt-and-pepper
    noise that would be unphysical and impossible for any real EM/TDR
    survey to resolve.

    Parameters
    ----------
    x_coords, y_coords : ndarray, shape (n_points,)
        Local (or UTM) coordinates of every measurement point. In 1-D mode
        ``y_coords`` is all zeros, which still works fine -- the field just
        varies along x only.
    n_blobs : int
        Number of random Gaussian patches summed together. More blobs ->
        busier, more fragmented pattern; fewer -> a handful of broad
        wet/dry zones.
    correlation_m : float
        Gaussian patch radius [m] (1-sigma). Roughly the size of one
        wet/dry zone. Should be a meaningful fraction of the survey extent
        -- too small relative to the point spacing degenerates toward
        white noise; too large degenerates toward a single smooth blob.
    rng : np.random.Generator
        Seeded generator, so the field is reproducible.

    Returns
    -------
    ndarray, shape (n_points,)
        Field values normalized to [0, 1] (min-max scaled across all
        points), ready to substitute for ``xy_frac`` in the theta_xy
        formula.
    """
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    # Guard against a degenerate (zero-extent) axis, e.g. 1-D mode's
    # all-zero y_coords -- still center blobs at 0 rather than dividing
    # by a zero range anywhere downstream.
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)

    field = np.zeros_like(x_coords, dtype=float)
    for _ in range(n_blobs):
        cx = rng.uniform(x_min - 0.25 * x_span, x_max + 0.25 * x_span)
        cy = rng.uniform(y_min - 0.25 * y_span, y_max + 0.25 * y_span)
        amp = rng.uniform(-1.0, 1.0)   # signed -> both wet and dry patches
        d2 = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
        field += amp * np.exp(-d2 / (2.0 * correlation_m ** 2))

    f_min, f_max = field.min(), field.max()
    if f_max > f_min:
        field = (field - f_min) / (f_max - f_min)
    else:
        # Degenerate case (e.g. n_blobs=0): fall back to a flat mid-value
        # field rather than dividing by zero.
        field = np.full_like(field, 0.5)
    return field


def generate_time_lapse_dataset(
    output_dir: str = "mfc2_outputs",
    point_spacing: float = 2.0,
    mode: str = "1d",
    num_x: int = 10,
    num_y: int = 5,
    depths0: list | None = None,
    cond_ranges: list | None = None,
    homogeneous: bool = False,
    infiltration: bool = False,
    front_z0: float = 0.2,
    front_speed: float = 0.5,
    front_width: float = 0.15,
    theta_dry: float = 0.08,
    theta_wet: float = 0.38,
    full_range_gradient: bool = False,
    xy_z_blend: float = 0.5,
    invert_z_gradient: bool = False,
    random_xy_field: bool = False,
    random_xy_n_blobs: int = 8,
    random_xy_correlation_m: float | None = None,
    random_xy_seed: int | None = None,
    ws_xy_variation: bool = False,
    ws_xy_params: list | None = None,
    ws_xy_ranges: dict | None = None,
    ws_xy_pattern: str = "linear",
    ws_xy_n_blobs: int = 6,
    ws_xy_correlation_m: float | None = None,
    ws_xy_seed: int | None = None,
    ws_layer_params: dict | None = None,
):
    """
    Generate synthetic 4-D subsurface property matrices (EC_bulk, ECa, θ)
    using a realistic linear/spatial gradient from dry to wet in both XY and XZ.

    Parameters
    ----------
    homogeneous : bool, default False
        When True, disables both spatial gradients (XY position and XZ/depth)
        AND per-sample noise, so theta is identical across all measurement
        points and layers at each timestep. Only temporal trend (progressive
        wetting over timesteps) varies. Useful for testing inversion behavior
        on truly uniform soil that evolves over time, without spatial confounding.
    infiltration : bool, default False
        When True, replaces the linear XZ/temporal gradient with a downward-
        moving wetting-front model: at t=0 the shallow subsurface (near
        front_z0) is wet (theta_wet) and everything below is still at the
        pre-event dry state (theta_dry); the front then advances downward by
        front_speed [m] per timestep, so deeper layers (e.g. ~1.2 m) wet up
        progressively over the time-lapse sequence. front_width controls how
        sharp the wetting front is (smaller = sharper step). Takes precedence
        over `homogeneous` if both are set.
    full_range_gradient : bool, default False
        When True (and `infiltration`/`homogeneous` are both False), the
        static XZ (depth) and XY (plan-view) gradients each span the FULL
        [theta_dry, theta_wet] range independently, instead of the small
        fixed-amplitude offsets used by the plain default gradient (which
        only adds up to ~0.12-0.18 vol. fraction on top of a 0.05 baseline).
        The two components are combined with `xy_z_blend`:
            base_theta = xy_z_blend * theta_xy + (1 - xy_z_blend) * theta_z
        where theta_z = theta_dry + (theta_wet - theta_dry) * z_frac
        (z_frac=0 at the shallowest layer, 1 at the deepest -- reverse with
        `invert_z_gradient`), and theta_xy = theta_dry + (theta_wet -
        theta_dry) * xy_frac, with xy_frac = 0.5*x_frac + 0.5*y_frac (2-D
        mode) or x_frac alone (1-D mode) by default -- or a random spatial
        field when `random_xy_field=True`, see below. A small temporal
        trend and noise are still layered on top, same as the default
        gradient.
    xy_z_blend : float, default 0.5
        Only used when `full_range_gradient=True`. Weight in [0, 1] between
        the XY component (1.0 = XY gradient only) and the Z/depth component
        (0.0 = Z gradient only) of the initial model.
    invert_z_gradient : bool, default False
        Only used when `full_range_gradient=True`. When True, flips the
        depth gradient so the shallowest layer is wet and the deepest layer
        is dry (default is dry-at-surface, wet-at-depth) -- i.e. the soil
        dries out uniformly with depth. This depth trend is a function of
        z_frac ONLY (identical at every XY location), so "uniform" here
        means "the same drying rate everywhere in plan view", independent
        of whatever XY pattern (linear or random) is layered on top of it.
    random_xy_field : bool, default False
        Only used when `full_range_gradient=True`. When True, replaces the
        deterministic linear `xy_frac` ramp (a simple diagonal gradient)
        with a smooth, spatially-correlated RANDOM field (sum of Gaussian
        patches, see `_generate_random_xy_field`) -- i.e. scattered wetter/
        drier patches instead of a single monotonic direction. One field is
        drawn per ensemble member (so different ensemble members see a
        different, but equally plausible, spatial pattern) and held fixed
        across all timesteps/layers within that member -- only the
        Z/depth trend and the small temporal/noise terms still evolve.
    random_xy_n_blobs : int, default 8
        Only used when `random_xy_field=True`. Number of random Gaussian
        patches summed into the field -- see `_generate_random_xy_field`.
    random_xy_correlation_m : float or None, default None
        Only used when `random_xy_field=True`. Patch radius [m] -- see
        `_generate_random_xy_field`. `None` (default) auto-picks ~1/4 of
        the survey's shorter horizontal extent (x or y span), a reasonable
        default patch size for most grids.
    random_xy_seed : int or None, default None
        Only used when `random_xy_field=True`. Seeds the random field
        generator independently of the module's global `np.random.seed(42)`
        call, so the XY pattern is reproducible run-to-run without being
        coupled to (or perturbing) the per-sample noise stream. `None`
        (default) derives a seed from the global seed instead (still
        reproducible, just not independently controllable).
    ws_xy_variation : bool, default False
        When True, the TRUTH Waxman-Smits parameters (a, m, sigma_w,
        sigma_s) are no longer spatially uniform (the `_WS_DEFAULTS`
        constant) -- instead, the parameters listed in `ws_xy_params` are
        linearly (or randomly, see `ws_xy_pattern`) interpolated across
        their `ws_xy_ranges` as a function of plan-view (x, y) position,
        independent of and in addition to the existing per-layer (depth)
        variation. This is the "spatially-heterogeneous petrophysics mode"
        the `cond_ranges` warning above refers to. Every parameter NOT
        listed in `ws_xy_params` stays fixed at its `_WS_DEFAULTS` value,
        identical everywhere, same as before.
    ws_xy_params : list of str or None, default None
        Which Waxman-Smits parameters vary spatially when
        `ws_xy_variation=True`. Subset of {"a", "m", "sigma_w", "sigma_s"}.
        `None` (default) -> ["m", "sigma_s"], since the cementation
        exponent and surface conductivity are the parameters most
        plausibly tied to spatially-varying soil texture/lithology, while
        `a` and `sigma_w` are more often treated as near-constant
        lab/pore-water values across a small survey.
    ws_xy_ranges : dict or None, default None
        Per-parameter [min, max] truth range spanned across the XY field,
        e.g. {"sigma_s": [50.0, 200.0]}. Only keys in `ws_xy_params` are
        used. `None` (default) -> `_WS_XY_DEFAULT_RANGES` below. At
        xy_frac=0 a point gets `min`, at xy_frac=1 it gets `max` (same
        xy_frac convention as the moisture field: linear x/y ramp, or a
        random field -- see `ws_xy_pattern`).
    ws_xy_pattern : {"linear", "random"}, default "linear"
        "linear": xy_frac = 0.5*x_frac + 0.5*y_frac (2-D mode) or x_frac
        (1-D mode) -- the same simple diagonal ramp used elsewhere.
        "random": a smooth spatially-correlated random field (sum of
        Gaussian patches, see `_generate_random_xy_field`), drawn with its
        OWN seed stream independent of `random_xy_field`'s moisture
        pattern -- so e.g. moisture and sigma_s can show unrelated spatial
        patterns even if both are randomized. One field per ensemble
        member, held fixed across all layers/timesteps for that member.
    ws_xy_n_blobs : int, default 6
        Only used when `ws_xy_pattern="random"`. See `random_xy_n_blobs`.
    ws_xy_correlation_m : float or None, default None
        Only used when `ws_xy_pattern="random"`. See
        `random_xy_correlation_m`.
    ws_xy_seed : int or None, default None
        Only used when `ws_xy_pattern="random"`. Seeds the WS-parameter
        random field independently of both the global noise seed and
        `random_xy_seed`. `None` -> derived from 99.
    ws_layer_params : dict or None, default None
        Per-layer overrides of the TRUE Waxman-Smits parameters, e.g.
        ``{2: {"a": 0.35, "sigma_w": 700.0}, 3: {...}}`` (layer index ->
        subset of {a, m, sigma_w, sigma_s}). Layers not listed keep
        `_WS_DEFAULTS`. Parameters in `ws_xy_params` are drawn from
        `ws_xy_ranges` instead and take precedence. `None` -> same truth in
        every layer (previous behaviour).
    """

    os.makedirs(output_dir, exist_ok=True)

    if cond_ranges is not None:
        print(
            "[DATA GEN][WARN] cond_ranges was provided but is currently NOT used by "
            "generate_time_lapse_dataset -- the truth Waxman-Smits parameters (a, m, "
            "sigma_w, sigma_s) are fixed by the hardcoded _WS_DEFAULTS below (or, if "
            "ws_xy_variation=True, drawn from ws_xy_ranges instead), regardless of "
            "cond_ranges. Only `theta` (and, with ws_xy_variation, the WS params "
            "themselves) varies spatially/temporally."
        )

    ensemble_size = 4
    n_time = 3

    if depths0 is None:
        layer_bottoms = [0.3, 0.6, 1.0]
    else:
        layer_bottoms = list(depths0)

    num_layers = len(layer_bottoms) + 1
    depth_boundaries = [0.0] + layer_bottoms + [layer_bottoms[-1] * 1.5]

    _WS_DEFAULTS = {
        0: {"a": 0.50, "m": 3.0, "sigma_w": 500.0, "sigma_s": 100.0},
        1: {"a": 0.50, "m": 3.0, "sigma_w": 500.0, "sigma_s": 100.0},
        2: {"a": 0.50, "m": 3.0, "sigma_w": 500.0, "sigma_s": 100.0},
        3: {"a": 0.50, "m": 3.0, "sigma_w": 500.0, "sigma_s": 100.0},
    }
    layer_params = {
        l: _WS_DEFAULTS.get(l, _WS_DEFAULTS[3])
        for l in range(num_layers)
    }
    # Optional per-layer overrides of the TRUE WS parameters (scenario field
    # ``ws_layer_params``, e.g. a two-zone vertical heterogeneity).
    if ws_layer_params:
        for _l, _over in ws_layer_params.items():
            _l = int(_l)
            if _l not in layer_params:
                raise ValueError(
                    f"ws_layer_params: layer {_l} does not exist "
                    f"(this run has {num_layers} layers, 0..{num_layers - 1})."
                )
            _unknown = set(_over) - {"a", "m", "sigma_w", "sigma_s"}
            if _unknown:
                raise ValueError(
                    f"ws_layer_params[{_l}] has unknown parameter(s) {sorted(_unknown)}."
                )
            layer_params[_l] = {**layer_params[_l], **{k: float(v) for k, v in _over.items()}}

    # --- spatially-heterogeneous petrophysics (ws_xy_variation) ---------
    _WS_XY_DEFAULT_RANGES = {
        "a":       [0.30, 0.70],
        "m":       [2.00, 4.00],
        "sigma_w": [300.0, 700.0],
        "sigma_s": [50.0, 200.0],
    }
    _WS_VALID_PARAMS = {"a", "m", "sigma_w", "sigma_s"}
    if ws_xy_variation:
        if ws_xy_params is None:
            ws_xy_params = ["m", "sigma_s"]
        bad = set(ws_xy_params) - _WS_VALID_PARAMS
        if bad:
            raise ValueError(
                f"ws_xy_params contains unknown Waxman-Smits parameter(s) {sorted(bad)}; "
                f"valid choices are {sorted(_WS_VALID_PARAMS)}."
            )
        _ws_ranges = dict(_WS_XY_DEFAULT_RANGES)
        if ws_xy_ranges:
            for k, v in ws_xy_ranges.items():
                if k not in _WS_VALID_PARAMS:
                    raise ValueError(
                        f"ws_xy_ranges key '{k}' is not a valid Waxman-Smits parameter; "
                        f"valid choices are {sorted(_WS_VALID_PARAMS)}."
                    )
                _ws_ranges[k] = v
        _active_ranges = {k: _ws_ranges[k] for k in ws_xy_params}
        print(
            f"[DATA GEN] ws_xy_variation enabled: {ws_xy_params} will vary spatially "
            f"in XY (pattern='{ws_xy_pattern}'), ranges={_active_ranges}"
        )
    else:
        _ws_ranges = None

    coils_list = COILS_LIST

    mode = mode.lower().strip()
    if mode == "2d":
        x_local, y_local, utm_x_grid, utm_y_grid = _build_2d_grid_coords(
            num_x, num_y, point_spacing
        )
        num_measured_points = len(x_local)
        x_coords = x_local
        y_coords = y_local
        grid_shape = (num_y, num_x)
        print(f"[DATA GEN] 2-D mode: {num_x}×{num_y} grid → {num_measured_points} points")
    else:
        num_measured_points = 10
        x_coords = _build_transect_x_coords(num_measured_points, point_spacing)
        y_coords = np.zeros(num_measured_points)
        utm_x_grid = None
        utm_y_grid = None
        grid_shape = None
        print(f"[DATA GEN] 1-D mode: {num_measured_points} transect points")

    ensemble_ec_bulk = []
    ensemble_ec_apparent = []
    ensemble_theta_true = []

    # Computed up-front (single shared implementation, see note near the
    # np.save calls below) so the infiltration branch can look up each
    # layer's true depth [m] while building the per-point profile.
    layer_centers = geometry_utils.layer_centers(layer_bottoms).tolist()

    np.random.seed(42)

    # Pre-calculate spatial coordinate extents for normalization mapping
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Random XY field: one draw per ensemble member (see docstring), using
    # its own seeded Generator so it neither depends on nor perturbs the
    # np.random.seed(42) stream consumed below for per-sample noise.
    if random_xy_field:
        _base_seed = random_xy_seed if random_xy_seed is not None else 42
        if random_xy_correlation_m is None:
            # In 1-D mode y_coords is all zeros (y_max == y_min), which
            # would otherwise collapse this to ~0 via the min() below --
            # use the x extent alone in that case.
            if mode == "2d":
                _shorter_span = min(max(x_max - x_min, 1e-9), max(y_max - y_min, 1e-9))
            else:
                _shorter_span = max(x_max - x_min, 1e-9)
            _correlation_m = 0.25 * _shorter_span
        else:
            _correlation_m = random_xy_correlation_m
        xy_random_fields = [
            _generate_random_xy_field(
                x_coords, y_coords, random_xy_n_blobs, _correlation_m,
                np.random.default_rng(_base_seed + ens_idx),
            )
            for ens_idx in range(ensemble_size)
        ]
    else:
        xy_random_fields = None

    # WS-parameter spatial field: independent seed stream from the moisture
    # random_xy_field above, so the two patterns don't have to coincide.
    if ws_xy_variation and ws_xy_pattern == "random":
        _ws_base_seed = ws_xy_seed if ws_xy_seed is not None else 99
        if ws_xy_correlation_m is None:
            if mode == "2d":
                _ws_correlation_m = 0.25 * min(max(x_max - x_min, 1e-9), max(y_max - y_min, 1e-9))
            else:
                _ws_correlation_m = 0.25 * max(x_max - x_min, 1e-9)
        else:
            _ws_correlation_m = ws_xy_correlation_m
        ws_random_fields = [
            _generate_random_xy_field(
                x_coords, y_coords, ws_xy_n_blobs, _ws_correlation_m,
                np.random.default_rng(_ws_base_seed + ens_idx),
            )
            for ens_idx in range(ensemble_size)
        ]
    else:
        ws_random_fields = None

    # Per-point-per-layer truth WS params actually used (constant across
    # timesteps within an ensemble member, so only tracked here, not saved
    # per-timestep). Only populated/saved when ws_xy_variation is set.
    ensemble_ws_truth = {p: [] for p in ("a", "m", "sigma_w", "sigma_s")} if ws_xy_variation else None

    for ens_idx in range(ensemble_size):
        time_lapse_ec = np.zeros((n_time, num_measured_points, num_layers))
        time_lapse_eca = np.zeros((n_time, num_measured_points, len(coils_list)))
        time_lapse_theta = np.zeros((n_time, num_measured_points, num_layers))
        if ws_xy_variation:
            ws_truth_this_member = {
                p: np.zeros((num_measured_points, num_layers)) for p in ("a", "m", "sigma_w", "sigma_s")
            }

        for t_idx in range(n_time):
            # Gentle temporal trend (progressive wetting over subsequent timesteps)
            t_trend = 0.03 * t_idx
            
            # In homogeneous mode, pre-compute one theta value for the entire timestep
            # (all points and all layers get this same value)
            if homogeneous:
                theta_min = 0.05
                base_theta_homogeneous = theta_min + t_trend
                base_theta_homogeneous = np.clip(base_theta_homogeneous, 0.05, 0.45)
            
            for p_idx in range(num_measured_points):
                current_ec_profile = []

                # Normalized spatial fractions for horizontal (XY) gradient
                x_frac = (x_coords[p_idx] - x_min) / (x_max - x_min) if x_max > x_min else 0.0
                y_frac = (y_coords[p_idx] - y_min) / (y_max - y_min) if y_max > y_min else 0.0

                # Plan-view fraction driving the spatially-varying truth WS
                # params (independent stream from the moisture xy_frac /
                # xy_random_fields above, even though the "linear" pattern
                # uses the same formula).
                if ws_xy_variation:
                    if ws_xy_pattern == "random":
                        ws_frac = ws_random_fields[ens_idx][p_idx]
                    else:
                        ws_frac = (0.5 * x_frac + 0.5 * y_frac) if mode == "2d" else x_frac

                for l_idx in range(num_layers):
                    # Normalized vertical fraction for depth (XZ) gradient
                    z_frac = l_idx / (num_layers - 1) if num_layers > 1 else 0.0

                    if infiltration:
                        # Downward-moving wetting front: at t_idx=0 the front
                        # sits at front_z0 (shallow -> wet, deep -> still dry).
                        # It advances by front_speed [m] per timestep, so a
                        # layer centered at e.g. 1.2 m starts dry and wets up
                        # progressively once the front reaches it. A logistic
                        # (not step) transition avoids an unrealistic sharp
                        # discontinuity, which also helps ERT/EMI inversion
                        # sensitivity tests. Mild XY heterogeneity nudges the
                        # front slightly earlier/later across the transect/grid.
                        z_depth = layer_centers[l_idx]
                        if mode == "2d":
                            het_offset = 0.05 * x_frac + 0.03 * y_frac
                        else:
                            het_offset = 0.08 * x_frac
                        front_z_t = front_z0 + front_speed * t_idx - het_offset
                        wet_fraction = 1.0 / (1.0 + np.exp((z_depth - front_z_t) / front_width))
                        base_theta = theta_dry + (theta_wet - theta_dry) * wet_fraction

                        theta_val = base_theta + np.random.normal(0, 0.01)
                        theta_val = np.clip(theta_val, 0.03, 0.45)
                    elif homogeneous:
                        # Truly homogeneous: use the pre-computed uniform theta for this timestep
                        theta_val = base_theta_homogeneous
                    elif full_range_gradient:
                        # Full [theta_dry, theta_wet] range gradient, independently
                        # in Z (depth) and XY (plan view), blended via xy_z_blend.
                        z_frac_eff = z_frac if invert_z_gradient else (1.0 - z_frac)
                        theta_z = theta_dry + (theta_wet - theta_dry) * z_frac_eff

                        if random_xy_field:
                            # Spatially-correlated random patches instead of a
                            # monotonic ramp -- same value at every layer/
                            # timestep for this point/ensemble member, so the
                            # depth trend above stays the only thing varying
                            # with z.
                            xy_frac = xy_random_fields[ens_idx][p_idx]
                        else:
                            xy_frac = (0.5 * x_frac + 0.5 * y_frac) if mode == "2d" else x_frac
                        theta_xy = theta_dry + (theta_wet - theta_dry) * xy_frac

                        base_theta = (
                            xy_z_blend * theta_xy + (1.0 - xy_z_blend) * theta_z + t_trend
                        )

                        theta_val = base_theta + np.random.normal(0, 0.01)
                        theta_val = np.clip(theta_val, theta_dry, theta_wet)
                    else:
                        # Absolute minimal soil moisture baseline (Dry boundary)
                        theta_min = 0.05

                        # Continuous gradient distribution
                        if mode == "2d":
                            xy_gradient = 0.12 * x_frac + 0.06 * y_frac
                        else:
                            xy_gradient = 0.18 * x_frac

                        xz_gradient = 0.12 * z_frac
                        
                        # Formulate systematic multi-dimensional continuous gradient
                        base_theta = theta_min + xy_gradient + xz_gradient + t_trend

                        # Add fine noise to model standard physical micro-heterogeneities
                        theta_val = base_theta + np.random.normal(0, 0.01)
                        theta_val = np.clip(theta_val, 0.05, 0.45)
                    time_lapse_theta[t_idx, p_idx, l_idx] = theta_val

                    if ws_xy_variation:
                        point_ws = dict(layer_params[l_idx])
                        for pname in ws_xy_params:
                            pmin, pmax = _ws_ranges[pname]
                            point_ws[pname] = pmin + (pmax - pmin) * ws_frac
                        if t_idx == 0:
                            for pname in ("a", "m", "sigma_w", "sigma_s"):
                                ws_truth_this_member[pname][p_idx, l_idx] = point_ws[pname]
                    else:
                        point_ws = layer_params[l_idx]

                    ec_bulk = forward_waxman_smits(
                        theta_val,
                        point_ws["a"],
                        point_ws["m"],
                        point_ws["sigma_w"],
                        point_ws["sigma_s"],
                    )
                    ec_bulk = max(1.0, ec_bulk + np.random.normal(0, 0.1))
                    time_lapse_ec[t_idx, p_idx, l_idx] = ec_bulk
                    current_ec_profile.append(ec_bulk)

                models_input = [np.array([current_ec_profile])]
                depths_input = [np.array([layer_bottoms])]

                k_point = Problem()
                k_point.models = models_input
                k_point.depths = depths_input
                res_forward = k_point.forward(
                    forwardModel="CS",
                    coils=coils_list,
                    models=models_input,
                    depths=depths_input,
                )
                time_lapse_eca[t_idx, p_idx, :] = res_forward[0].values.flatten()

        ensemble_ec_bulk.append(time_lapse_ec)
        ensemble_ec_apparent.append(time_lapse_eca)
        ensemble_theta_true.append(time_lapse_theta)
        if ws_xy_variation:
            for pname in ("a", "m", "sigma_w", "sigma_s"):
                ensemble_ws_truth[pname].append(ws_truth_this_member[pname])

    np.save(os.path.join(output_dir, "synthetic_time_lapse_ec.npy"),    np.array(ensemble_ec_bulk))
    np.save(os.path.join(output_dir, "synthetic_time_lapse_eca.npy"),   np.array(ensemble_ec_apparent))
    np.save(os.path.join(output_dir, "synthetic_time_lapse_theta.npy"), np.array(ensemble_theta_true))
    np.save(os.path.join(output_dir, "synthetic_x_coords.npy"), x_coords)
    np.save(os.path.join(output_dir, "synthetic_y_coords.npy"), y_coords)

    if ws_xy_variation:
        # Per-point-per-layer TRUTH WS params actually used, shape
        # (ensemble_size, num_measured_points, num_layers). Constant across
        # timesteps, so these are ground truth for validating any
        # localized/spatial WS calibration (e.g. step2.run_petro_localized)
        # against the true spatial pattern, not just the fitted one.
        for pname in ("a", "m", "sigma_w", "sigma_s"):
            np.save(
                os.path.join(output_dir, f"synthetic_ws_truth_{pname}.npy"),
                np.array(ensemble_ws_truth[pname]),
            )
    else:
        for pname in ("a", "m", "sigma_w", "sigma_s"):
            _stale_path = os.path.join(output_dir, f"synthetic_ws_truth_{pname}.npy")
            if os.path.exists(_stale_path):
                os.remove(_stale_path)

    if mode == "2d" and utm_x_grid is not None:
        np.save(os.path.join(output_dir, "synthetic_utm_x.npy"), utm_x_grid)
        np.save(os.path.join(output_dir, "synthetic_utm_y.npy"), utm_y_grid)
        np.save(os.path.join(output_dir, "synthetic_grid_shape.npy"), np.array(grid_shape))
    else:
        for _stale in ("synthetic_utm_x.npy", "synthetic_utm_y.npy", "synthetic_grid_shape.npy"):
            _stale_path = os.path.join(output_dir, _stale)
            if os.path.exists(_stale_path):
                os.remove(_stale_path)

    # Single shared implementation (geometry_utils.layer_centers) instead of
    # an inline recomputation — see pipeline_improvement_report.md, section 6
    # / action item 3. This used to be reimplemented independently here, in
    # run_pipeline.py::_layer_centers, and in run_pipeline_short.py; a
    # mismatch between those copies previously caused a Step 6d plotting
    # crash. depth_boundaries is unchanged and still saved/returned below.
    # (layer_centers itself is now computed up-front, before the ensemble
    # loop, so the infiltration branch can use it — see above.)

    np.save(os.path.join(output_dir, "synthetic_layer_centers.npy"),    np.array(layer_centers))
    np.save(os.path.join(output_dir, "synthetic_depth_boundaries.npy"), np.array(depth_boundaries))
    np.save(os.path.join(output_dir, "synthetic_layer_bottoms.npy"),    np.array(layer_bottoms))

    print(f"[DATA GEN] {num_layers} layers derived from depths0={layer_bottoms}")
    if homogeneous:
        print("[DATA GEN] Dataset, theta verdadero y ECa completados exitosamente en modo HOMOGÉNEO "
              "(TODOS los puntos y horizontes tienen theta idéntico, solo varía la tendencia temporal).")
    else:
        print("[DATA GEN] Dataset, theta verdadero y ECa completados exitosamente con gradientes continuos XY y XZ.")

    # Describe which theta-generation branch was actually used (mirrors the
    # if/elif/else precedence in the per-point loop above: infiltration >
    # homogeneous > full_range_gradient > default gradient).
    if infiltration:
        gradient_mode = "infiltration (downward wetting front)"
    elif homogeneous:
        gradient_mode = "homogeneous (spatially uniform theta; temporal trend only)"
    elif full_range_gradient:
        gradient_mode = "full_range_gradient (XY/XZ each span full [theta_dry, theta_wet])"
        if random_xy_field:
            gradient_mode += ", random_xy_field (patchy XY, not linear)"
    else:
        gradient_mode = "default gradient (small fixed-amplitude XY/XZ offset)"

    result = {
        "depth_boundaries":   depth_boundaries,
        "layer_bottoms":      layer_bottoms,
        "layer_centers":      layer_centers,
        "x_coords":           x_coords,
        "y_coords":           y_coords,
        "coils_list":         coils_list,
        "mode":               mode,
        "homogeneous":        homogeneous,
        # --- new: truth petrophysics + generation settings, for reporting ---
        # NOTE: layer_params (below) is the per-LAYER baseline from
        # _WS_DEFAULTS -- with ws_xy_variation=False this is exactly what
        # every point used (identical across x/y/ensemble/time). With
        # ws_xy_variation=True, the params listed in ws_xy_params instead
        # vary continuously across x/y per ensemble member; the actual
        # per-point-per-layer values used are saved to
        # synthetic_ws_truth_{a,m,sigma_w,sigma_s}.npy, not summarized here.
        "ws_truth_params":    {l: dict(p) for l, p in layer_params.items()},
        "ws_xy_variation":    ws_xy_variation,
        "gradient_mode":      gradient_mode,
        "theta_dry":          theta_dry,
        "theta_wet":          theta_wet,
        "cond_ranges_input":  cond_ranges,   # kept for transparency; currently unused (see warning above)
        "cond_ranges_used":   False,
        "ensemble_size":      ensemble_size,
        "n_time":             n_time,
        "point_spacing":      point_spacing,
    }
    if mode == "2d":
        result["utm_x"]      = utm_x_grid
        result["utm_y"]      = utm_y_grid
        result["grid_shape"] = grid_shape
        result["num_x"]      = num_x
        result["num_y"]      = num_y
    if infiltration:
        result["infiltration_params"] = {
            "front_z0": front_z0, "front_speed": front_speed, "front_width": front_width,
        }
    if full_range_gradient:
        result["full_range_gradient_params"] = {
            "xy_z_blend": xy_z_blend, "invert_z_gradient": invert_z_gradient,
            "random_xy_field": random_xy_field,
        }
    if full_range_gradient and random_xy_field:
        result["random_xy_params"] = {
            "n_blobs": random_xy_n_blobs,
            "correlation_m": _correlation_m,
            "seed": _base_seed,
        }
    if ws_xy_variation:
        result["ws_xy_variation_params"] = {
            "varying_params": ws_xy_params,
            "ranges": {k: _ws_ranges[k] for k in ws_xy_params},
            "pattern": ws_xy_pattern,
        }
        if ws_xy_pattern == "random":
            result["ws_xy_variation_params"].update({
                "n_blobs": ws_xy_n_blobs,
                "correlation_m": _ws_correlation_m,
                "seed": _ws_base_seed,
            })
    return result


def export_eca_surveys_to_csv(
    output_dir: str = "mfc2_outputs",
    realization_index: int = 0,
    csv_subdir: str = "synthetic_csv",
    point_spacing: float = 2.0,
    eca_noise_level: float = 0.0,
    tdr_noise_std: float = 0.0,
):
    eca_path   = os.path.join(output_dir, "synthetic_time_lapse_eca.npy")
    theta_path = os.path.join(output_dir, "synthetic_time_lapse_theta.npy")
    if not (os.path.exists(eca_path) and os.path.exists(theta_path)):
        raise FileNotFoundError(f"Could not find .npy files in '{output_dir}'.")

    eca_ensemble   = np.load(eca_path)
    theta_ensemble = np.load(theta_path)

    eca_realization   = eca_ensemble[realization_index]
    theta_realization = theta_ensemble[realization_index]

    n_time, num_measured_points, n_coils = eca_realization.shape

    x_coords = (
        np.load(os.path.join(output_dir, "synthetic_x_coords.npy"))
        if os.path.exists(os.path.join(output_dir, "synthetic_x_coords.npy"))
        else _build_transect_x_coords(num_measured_points, point_spacing)
    )
    y_coords = (
        np.load(os.path.join(output_dir, "synthetic_y_coords.npy"))
        if os.path.exists(os.path.join(output_dir, "synthetic_y_coords.npy"))
        else np.zeros(num_measured_points)
    )

    utm_x_path = os.path.join(output_dir, "synthetic_utm_x.npy")
    utm_y_path = os.path.join(output_dir, "synthetic_utm_y.npy")
    has_utm = os.path.exists(utm_x_path) and os.path.exists(utm_y_path)
    if has_utm:
        utm_x = np.load(utm_x_path)
        utm_y = np.load(utm_y_path)

    rng = np.random.default_rng(seed=0)
    if eca_noise_level > 0.0:
        noise_factor = 1.0 + rng.normal(0.0, eca_noise_level, size=eca_realization.shape)
        eca_noisy = eca_realization * noise_factor
    else:
        eca_noisy = eca_realization

    csv_dir = os.path.join(output_dir, csv_subdir)
    os.makedirs(csv_dir, exist_ok=True)

    csv_paths = []
    for t_idx in range(n_time):
        df = pd.DataFrame(eca_noisy[t_idx], columns=COILS_LIST)
        df.insert(0, "y", y_coords)
        df.insert(0, "x", x_coords)
        if has_utm:
            df.insert(2, "utm_x", utm_x)
            df.insert(3, "utm_y", utm_y)

        csv_path = os.path.join(csv_dir, f"synthetic_survey_t{t_idx:02d}.csv")
        df.to_csv(csv_path, index=False)
        csv_paths.append(csv_path)

    return csv_paths, theta_realization, tdr_noise_std


def plot_ws_truth_xy_maps(output_dir: str = "mfc2_outputs", realization_index: int = 0) -> list:
    """
    Plot the TRUE (ground-truth) Waxman-Smits petrophysical params (a, m,
    sigma_w, sigma_s) as spatial XY maps, for scenarios generated with
    ws_xy_variation=True. No-op (returns []) if none of the
    synthetic_ws_truth_{a,m,sigma_w,sigma_s}.npy files exist -- i.e. the
    dataset was generated with ws_xy_variation=False (the default), so the
    truth petrophysics is spatially uniform and there's nothing to map.

    Each param is averaged across layers before plotting (params NOT in the
    scenario's ws_xy_params stay spatially uniform -- their layer values are
    identical everywhere -- while params IN ws_xy_params are also constant
    across layers by construction, see generate_time_lapse_dataset, so the
    layer-mean is exactly the per-point spatial value either way; it's not
    smoothing over a real depth trend).

    2-D mode: rendered as an imshow heatmap on the (num_y, num_x) survey
    grid, using synthetic_utm_x/y.npy + synthetic_grid_shape.npy if present.
    1-D mode (or missing grid files): rendered as a 1-D scatter/line along
    x_coords, matching plot_synthetic_results' point-indexed style.

    Returns a list of the saved PNG paths (empty if nothing was plotted).
    """
    import matplotlib.pyplot as plt

    _param_labels = {
        "a":       "a (tortuosity factor)",
        "m":       "m (cementation exponent)",
        "sigma_w": "sigma_w [mS/m] (pore-water conductivity)",
        "sigma_s": "sigma_s [mS/m] (surface conductivity)",
    }
    _present = [
        p for p in ("a", "m", "sigma_w", "sigma_s")
        if os.path.exists(os.path.join(output_dir, f"synthetic_ws_truth_{p}.npy"))
    ]
    if not _present:
        return []

    x_coords_path = os.path.join(output_dir, "synthetic_x_coords.npy")
    y_coords_path = os.path.join(output_dir, "synthetic_y_coords.npy")
    x_coords = np.load(x_coords_path) if os.path.exists(x_coords_path) else None
    y_coords = np.load(y_coords_path) if os.path.exists(y_coords_path) else None

    utm_x_path = os.path.join(output_dir, "synthetic_utm_x.npy")
    utm_y_path = os.path.join(output_dir, "synthetic_utm_y.npy")
    grid_shape_path = os.path.join(output_dir, "synthetic_grid_shape.npy")
    has_grid = (
        os.path.exists(utm_x_path) and os.path.exists(utm_y_path)
        and os.path.exists(grid_shape_path)
    )

    fig, axes = plt.subplots(1, len(_present), figsize=(5.5 * len(_present), 5), squeeze=False)
    axes = axes[0]
    saved_paths = []

    for ax, pname in zip(axes, _present):
        arr = np.load(os.path.join(output_dir, f"synthetic_ws_truth_{pname}.npy"))
        # arr shape: (ensemble_size, num_measured_points, num_layers)
        point_vals = arr[realization_index].mean(axis=-1)  # (num_measured_points,)

        if has_grid:
            grid_shape = tuple(np.load(grid_shape_path).tolist())
            utm_x_grid = np.load(utm_x_path)
            utm_y_grid = np.load(utm_y_path)
            value_grid = point_vals.reshape(grid_shape)
            im = ax.imshow(
                value_grid,
                origin="lower",
                aspect="auto",
                extent=[utm_x_grid.min(), utm_x_grid.max(), utm_y_grid.min(), utm_y_grid.max()],
                cmap="viridis",
            )
            ax.set_xlabel("UTM X [m]")
            ax.set_ylabel("UTM Y [m]")
            fig.colorbar(im, ax=ax, label=_param_labels[pname])
        else:
            _x = x_coords if x_coords is not None else np.arange(len(point_vals))
            _y = y_coords if y_coords is not None else np.zeros(len(point_vals))
            sc = ax.scatter(_x, _y, c=point_vals, cmap="viridis", s=80, edgecolors="k", linewidths=0.5)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]" if y_coords is not None and np.any(y_coords != 0) else "")
            fig.colorbar(sc, ax=ax, label=_param_labels[pname])

        ax.set_title(f"Truth {pname}", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "synthetic_ws_truth_xy_maps.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    saved_paths.append(out_path)
    return saved_paths


def plot_synthetic_results(output_dir: str = "mfc2_outputs", realization_index: int = 0) -> None:
    import matplotlib.pyplot as plt

    ec_bulk_ensemble  = np.load(os.path.join(output_dir, "synthetic_time_lapse_ec.npy"))
    eca_ensemble      = np.load(os.path.join(output_dir, "synthetic_time_lapse_eca.npy"))

    ec_bulk     = ec_bulk_ensemble[realization_index]
    ec_apparent = eca_ensemble[realization_index]

    n_time              = ec_bulk.shape[0]
    num_measured_points = ec_bulk.shape[1]
    n_layers            = ec_bulk.shape[2]
    coils               = COILS_LIST

    lb_path = os.path.join(output_dir, "synthetic_layer_bottoms.npy")
    if os.path.exists(lb_path):
        layer_bottoms = np.load(lb_path).tolist()
    else:
        layer_bottoms = [0.3, 0.6, 1.0]

    bottoms = np.asarray(layer_bottoms, dtype=float)
    tops = np.concatenate(([0.0], bottoms[:-1]))
    z_centers = 0.5 * (tops + bottoms)

    if n_layers > len(z_centers):
        last_bottom = layer_bottoms[-1] if len(layer_bottoms) > 0 else 0.0
        half_space_center = last_bottom + 0.4
        z = np.concatenate((z_centers, [half_space_center]))[:n_layers]
    else:
        z = z_centers[:n_layers]

    fig, axes = plt.subplots(1, n_time, figsize=(4 * n_time, 5.5), sharey=True, squeeze=False)
    axes = axes[0]
    colors_pts = plt.cm.plasma(np.linspace(0, 0.85, num_measured_points))

    for t in range(n_time):
        ax = axes[t]
        for p in range(num_measured_points):
            ax.plot(ec_bulk[t, p, :], z, marker="o", color=colors_pts[p], alpha=0.3, linestyle="-")
        spatial_mean = np.mean(ec_bulk[t], axis=0)
        ax.plot(spatial_mean, z, color="tab:red", linestyle="-", linewidth=2.5, label="Mean across positions")
        ax.set_title(f"EC_bulk - Time Step {t+1}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Bulk EC [mS/m]")
        ax.grid(True, linestyle=":", alpha=0.6)
        if t == 0:
            ax.set_ylabel("Depth [m]", fontsize=11)
            ax.invert_yaxis()
            ax.legend(loc="lower left", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "synthetic_ec_bulk_profiles.png"), dpi=200)
    plt.close()

    fig, axes = plt.subplots(n_time, 1, figsize=(9, 2.8 * n_time), sharex=True, squeeze=False)
    axes = axes[:, 0]
    colors_coils = plt.cm.tab10(np.linspace(0, 1, len(coils)))

    for t in range(n_time):
        ax = axes[t]
        for c_idx, coil_name in enumerate(coils):
            ax.plot(range(1, num_measured_points + 1), ec_apparent[t, :, c_idx],
                    marker="s", color=colors_coils[c_idx], linewidth=1.5,
                    label=coil_name if t == 0 else "")
        ax.set_title(f"Apparent EC ($EC_a$) - Time Step {t+1}", fontsize=10, pad=4)
        ax.set_ylabel("$EC_a$ [mS/m]")
        ax.grid(True, linestyle=":", alpha=0.6)
        if t == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, title="Coil Config")

    axes[-1].set_xlabel("Measurement Point along Transect", fontsize=10)
    axes[-1].set_xticks(range(1, num_measured_points + 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "synthetic_eca_surface_data.png"), dpi=200)
    plt.close()

    # WS truth XY maps: no-op (returns []) when ws_xy_variation wasn't used,
    # so this is safe to call unconditionally for every scenario.
    plot_ws_truth_xy_maps(output_dir=output_dir, realization_index=realization_index)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic EM/TDR dataset for the EA-EM pipeline.")
    parser.add_argument("--mode", choices=["1d", "2d"], default="1d")
    parser.add_argument("--out", default="mfc2_outputs")
    parser.add_argument("--num-x", type=int, default=10)
    parser.add_argument("--num-y", type=int, default=5)
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument(
        "--homogeneous",
        action="store_true",
        help="Disable spatial (XY/XZ) gradients so theta is spatially uniform. "
             "Temporal trend and noise are still applied.",
    )
    parser.add_argument(
        "--infiltration",
        action="store_true",
        help="Replace the linear XZ/temporal gradient with a downward-moving "
             "wetting-front model (high theta near the surface at t=0, "
             "progressively wetting deeper layers over time). Overrides "
             "--homogeneous if both are given.",
    )
    parser.add_argument("--front-z0", type=float, default=0.2,
                         help="Wetting-front depth [m] at t=0 (infiltration mode).")
    parser.add_argument("--front-speed", type=float, default=0.5,
                         help="Front advance rate [m/timestep] (infiltration mode).")
    parser.add_argument("--front-width", type=float, default=0.15,
                         help="Transition sharpness [m] of the wetting front (infiltration mode).")
    parser.add_argument("--theta-dry", type=float, default=0.08,
                         help="Pre-event (dry) volumetric water content "
                              "(infiltration and full-range-gradient modes).")
    parser.add_argument("--theta-wet", type=float, default=0.38,
                         help="Post-wetting volumetric water content "
                              "(infiltration and full-range-gradient modes).")
    parser.add_argument(
        "--full-range-gradient", action="store_true",
        help="Static (non-infiltration, non-homogeneous) initial model: XZ "
             "(depth) and XY (plan-view) gradients each span the full "
             "[theta-dry, theta-wet] range independently, combined via "
             "--xy-z-blend. Overrides the plain default gradient; ignored "
             "if --infiltration or --homogeneous is also given.",
    )
    parser.add_argument(
        "--xy-z-blend", type=float, default=0.5, metavar="W",
        help="Weight in [0, 1] between the XY gradient (1.0 = XY only) and "
             "the Z/depth gradient (0.0 = Z only) when --full-range-gradient "
             "is set. Default 0.5 (equal blend).",
    )
    parser.add_argument(
        "--invert-z-gradient", action="store_true",
        help="With --full-range-gradient, flip the depth gradient so the "
             "shallowest layer is wet and the deepest layer is dry "
             "(default: dry at surface, wet at depth).",
    )
    parser.add_argument(
        "--random-xy-field", action="store_true",
        help="With --full-range-gradient, replace the linear XY ramp with a "
             "smooth spatially-correlated RANDOM field (patchy wet/dry "
             "zones) instead of a single monotonic direction. See "
             "--random-xy-n-blobs / --random-xy-correlation-m / "
             "--random-xy-seed.",
    )
    parser.add_argument(
        "--random-xy-n-blobs", type=int, default=8, metavar="N",
        help="Number of random Gaussian patches summed into the field when "
             "--random-xy-field is set. Default 8.",
    )
    parser.add_argument(
        "--random-xy-correlation-m", type=float, default=None, metavar="M",
        help="Patch radius [m] (1-sigma) when --random-xy-field is set. "
             "Default: auto, ~1/4 of the shorter survey extent.",
    )
    parser.add_argument(
        "--random-xy-seed", type=int, default=None, metavar="SEED",
        help="Seed for the random XY field (independent of the module's "
             "internal per-sample noise seed). Default: derived from 42.",
    )
    parser.add_argument("--depths0", type=str, default=None,
                         help="Comma-separated layer-bottom depths [m], e.g. '0.4,0.9,1.5' "
                              "(gives layer centers at 0.2, 0.65, 1.2 m). Defaults to [0.3, 0.6, 1.0].")
    parser.add_argument(
        "--ws-xy-variation", action="store_true",
        help="Make the TRUTH Waxman-Smits parameters (default: m, sigma_s) "
             "vary spatially across x/y instead of being spatially uniform. "
             "See --ws-xy-params / --ws-xy-pattern.",
    )
    parser.add_argument(
        "--ws-xy-params", type=str, default=None, metavar="P1,P2",
        help="Comma-separated subset of {a,m,sigma_w,sigma_s} to vary in XY "
             "when --ws-xy-variation is set. Default: 'm,sigma_s'.",
    )
    parser.add_argument(
        "--ws-xy-pattern", choices=["linear", "random"], default="linear",
        help="'linear': diagonal x/y ramp (same convention as the moisture "
             "gradient). 'random': smooth spatially-correlated random field, "
             "independent of the moisture pattern. Default 'linear'.",
    )
    parser.add_argument(
        "--ws-xy-n-blobs", type=int, default=6, metavar="N",
        help="Number of random Gaussian patches when --ws-xy-pattern=random. Default 6.",
    )
    parser.add_argument(
        "--ws-xy-correlation-m", type=float, default=None, metavar="M",
        help="Patch radius [m] when --ws-xy-pattern=random. Default: auto.",
    )
    parser.add_argument(
        "--ws-xy-seed", type=int, default=None, metavar="SEED",
        help="Seed for the WS-parameter random field. Default: derived from 99.",
    )
    args = parser.parse_args()

    depths0 = (
        [float(d) for d in args.depths0.split(",")]
        if args.depths0 is not None else None
    )

    res = generate_time_lapse_dataset(
        output_dir=args.out,
        point_spacing=args.spacing,
        mode=args.mode,
        num_x=args.num_x,
        num_y=args.num_y,
        depths0=depths0,
        homogeneous=args.homogeneous,
        infiltration=args.infiltration,
        front_z0=args.front_z0,
        front_speed=args.front_speed,
        front_width=args.front_width,
        theta_dry=args.theta_dry,
        theta_wet=args.theta_wet,
        full_range_gradient=args.full_range_gradient,
        xy_z_blend=args.xy_z_blend,
        invert_z_gradient=args.invert_z_gradient,
        random_xy_field=args.random_xy_field,
        random_xy_n_blobs=args.random_xy_n_blobs,
        random_xy_correlation_m=args.random_xy_correlation_m,
        random_xy_seed=args.random_xy_seed,
        ws_xy_variation=args.ws_xy_variation,
        ws_xy_params=(
            [p.strip() for p in args.ws_xy_params.split(",")]
            if args.ws_xy_params is not None else None
        ),
        ws_xy_pattern=args.ws_xy_pattern,
        ws_xy_n_blobs=args.ws_xy_n_blobs,
        ws_xy_correlation_m=args.ws_xy_correlation_m,
        ws_xy_seed=args.ws_xy_seed,
    )

    print(f"[DATA GEN] {len(res['layer_bottoms']) + 1} layers derived from depths0={res['layer_bottoms']}")
    print("[DATA GEN] Dataset, theta verdadero y ECa completados exitosamente.")
    
    plot_synthetic_results(output_dir=args.out, realization_index=0)