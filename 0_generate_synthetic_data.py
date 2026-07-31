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
ALENTO_UTM_ORIGIN_X = 515487.23   # SW corner easting  [m]
ALENTO_UTM_ORIGIN_Y = 4468201.66  # SW corner northing [m]
ALENTO_TRANSECT_LENGTH_X = 239.0  # E-W extent of survey area [m]
ALENTO_TRANSECT_LENGTH_Y = 310.4  # N-S extent of survey area [m]


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
    Build a 2-D regular grid of UTM coordinates anchored to the Alento
    survey bounding box.
    """
    xs = np.arange(num_x, dtype=float) * point_spacing
    ys = np.arange(num_y, dtype=float) * point_spacing
    XX, YY = np.meshgrid(xs, ys)          # shape (num_y, num_x)
    x_local = XX.ravel()
    y_local = YY.ravel()
    utm_x = ALENTO_UTM_ORIGIN_X + x_local
    utm_y = ALENTO_UTM_ORIGIN_Y + y_local
    return x_local, y_local, utm_x, utm_y


def forward_waxman_smits(theta, a, m, sigma_w, sigma_s):
    """
    Ecuación directa de Waxman-Smits: Calcula EC_bulk (mS/m) a partir de Theta.
    EC_bulk = (theta**m / a) * (sigma_w + sigma_s)
    """
    theta = np.clip(theta, 0.01, 1.0)
    return (theta ** m / a) * (sigma_w + sigma_s)


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
        mode) or x_frac alone (1-D mode). A small temporal trend and noise
        are still layered on top, same as the default gradient.
    xy_z_blend : float, default 0.5
        Only used when `full_range_gradient=True`. Weight in [0, 1] between
        the XY component (1.0 = XY gradient only) and the Z/depth component
        (0.0 = Z gradient only) of the initial model.
    invert_z_gradient : bool, default False
        Only used when `full_range_gradient=True`. When True, flips the
        depth gradient so the shallowest layer is wet and the deepest layer
        is dry (default is dry-at-surface, wet-at-depth).
    """
    os.makedirs(output_dir, exist_ok=True)

    if cond_ranges is not None:
        print(
            "[DATA GEN][WARN] cond_ranges was provided but is currently NOT used by "
            "generate_time_lapse_dataset -- the truth Waxman-Smits parameters (a, m, "
            "sigma_w, sigma_s) are fixed by the hardcoded _WS_DEFAULTS below, "
            "identical for every layer/position/time/ensemble member, regardless of "
            "cond_ranges. Only `theta` varies spatially/temporally. This parameter is "
            "reserved for a future spatially-heterogeneous petrophysics mode."
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

    for ens_idx in range(ensemble_size):
        time_lapse_ec = np.zeros((n_time, num_measured_points, num_layers))
        time_lapse_eca = np.zeros((n_time, num_measured_points, len(coils_list)))
        time_lapse_theta = np.zeros((n_time, num_measured_points, num_layers))

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
                        z_frac_eff = (1.0 - z_frac) if invert_z_gradient else z_frac
                        theta_z = theta_dry + (theta_wet - theta_dry) * z_frac_eff

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

                    ec_bulk = forward_waxman_smits(
                        theta_val,
                        layer_params[l_idx]["a"],
                        layer_params[l_idx]["m"],
                        layer_params[l_idx]["sigma_w"],
                        layer_params[l_idx]["sigma_s"],
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

    np.save(os.path.join(output_dir, "synthetic_time_lapse_ec.npy"),    np.array(ensemble_ec_bulk))
    np.save(os.path.join(output_dir, "synthetic_time_lapse_eca.npy"),   np.array(ensemble_ec_apparent))
    np.save(os.path.join(output_dir, "synthetic_time_lapse_theta.npy"), np.array(ensemble_theta_true))
    np.save(os.path.join(output_dir, "synthetic_x_coords.npy"), x_coords)
    np.save(os.path.join(output_dir, "synthetic_y_coords.npy"), y_coords)

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
        # NOTE: layer_params is currently IDENTICAL across all layers (see
        # _WS_DEFAULTS above) -- there is no spatial (x/y) or per-horizon
        # variation of (a, m, sigma_w, sigma_s) in the truth model. Only
        # `theta` varies, via the gradient/infiltration logic below.
        "ws_truth_params":    {l: dict(p) for l, p in layer_params.items()},
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
        }
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
    parser.add_argument("--depths0", type=str, default=None,
                         help="Comma-separated layer-bottom depths [m], e.g. '0.4,0.9,1.5' "
                              "(gives layer centers at 0.2, 0.65, 1.2 m). Defaults to [0.3, 0.6, 1.0].")
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
    )

    print(f"[DATA GEN] {len(res['layer_bottoms']) + 1} layers derived from depths0={res['layer_bottoms']}")
    print("[DATA GEN] Dataset, theta verdadero y ECa completados exitosamente.")
    
    plot_synthetic_results(output_dir=args.out, realization_index=0)