#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2 standalone runner — petrophysical calibration and TDR diagnostics.
"""

from __future__ import annotations

import os
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt

import em_methodology as EMmethod
import geometry_utils


# ==========================================================================
# DIAGNOSTIC PLOTTING FUNCTIONS FOR SYNTHETIC TDR DATA
# ==========================================================================
def plot_petrophysical_calibration(
    ec_values: np.ndarray,
    theta_obs: np.ndarray,
    petro_fit: EMmethod.PetroFit,
    output_path: str = "mfc2_outputs/petro_calibration.png",
    probe_layer_indices: np.ndarray = None  # <-- Añadimos este parámetro
) -> None:
    """
    Plots the cross-plot of Inverted EC vs. Synthetic TDR Water Content using
    the Waxman-Smits model. Points and curves are colored by soil horizon.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    num_horizons = len(petro_fit.horizon_fits)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, num_horizons))

    fig, ax = plt.subplots(figsize=(7, 5.5))  # Un poco más ancho para la leyenda

    # 1. Graficar los PUNTOS MEDIDOS (Scatter) clasificados por horizonte
    if probe_layer_indices is not None and len(probe_layer_indices) == len(ec_values):
        # Mapeamos cada punto a su color de horizonte correspondiente
        for hz_idx in range(num_horizons):
            mask = (probe_layer_indices == hz_idx)
            if np.any(mask):
                ax.scatter(
                    ec_values[mask],
                    theta_obs[mask],
                    color=colors[hz_idx],
                    edgecolor='black',
                    s=80,
                    zorder=4,
                    label=f'Observed Data (Hz {hz_idx})'
                )
    else:
        # Modo de respaldo (Fallback) si no se pasan los índices de las capas
        # Intenta usar la longitud de los arrays internos de los horizontes si coincide
        start_idx = 0
        for hz in petro_fit.horizon_fits:
            n_points = len(hz.observed_theta)
            end_idx = start_idx + n_points

            # Verificación de seguridad para no desbordar ec_values
            if end_idx <= len(ec_values):
                ax.scatter(
                    ec_values[start_idx:end_idx],
                    theta_obs[start_idx:end_idx],
                    color=colors[hz.horizon],
                    edgecolor='black',
                    s=80,
                    zorder=4,
                    label=f'Observed Data (Hz {hz.horizon})'
                )
                start_idx = end_idx
            else:
                # Si las longitudes no cuadran, vuelve al naranja genérico original
                ax.scatter(ec_values, theta_obs, color='darkorange', edgecolor='black',
                           s=75, zorder=3, label='Synthetic TDR Data Pairs')
                break

    # 2. Graficar las LÍNEAS TEÓRICAS de Waxman-Smits
    ec_line = np.linspace(ec_values.min() * 0.8, ec_values.max() * 1.2, 100)

    for hz in petro_fit.horizon_fits:
        theta_line = EMmethod.waxman_smits_theta(
            ec_line, hz.a, hz.m, hz.sigma_w, hz.sigma_s
        )
        is_calibrated = getattr(hz, 'is_calibrated', True)
        if is_calibrated:
            # Per-horizon goodness-of-fit alongside RMSE: Pearson R^2 and
            # p-value between this horizon's observed and model-predicted
            # theta (hz.predicted_theta / hz.observed_theta are the exact
            # pairs the optimizer/RMSE were computed against in
            # em_methodology.calibrate_petrophysical_relationship, so this
            # is consistent with the reported RMSE). "n/a" when there are
            # too few points or either series is constant (e.g. when
            # ws_fixed_params holds all 4 parameters fixed, theta_pred can
            # collapse to identical values and Pearson's r is undefined).
            r2_str, p_str = "n/a", "n/a"
            try:
                from scipy.stats import pearsonr
                obs_hz, pred_hz = hz.observed_theta, hz.predicted_theta
                if (len(obs_hz) >= 2 and np.std(obs_hz) > 0 and np.std(pred_hz) > 0):
                    r_val, p_val = pearsonr(obs_hz, pred_hz)
                    r2_str = f"{r_val**2:.3f}"
                    p_str = f"{p_val:.3g}"
            except Exception as e:
                print(f"[WARN] Could not compute R2/p-value for horizon {hz.horizon}: {e}")

            ax.plot(
                ec_line, theta_line,
                color=colors[hz.horizon], linestyle='--', linewidth=2, zorder=2,
                label=(
                    f'WS Fit Hz {hz.horizon} '
                    f'(RMSE={hz.rmse:.4f}, R²={r2_str}, p={p_str})'
                )
            )
        else:
            # No TDR probe covers this horizon — draw the prior-only curve
            # faint and dotted so it reads clearly as unfit, not a real result.
            ax.plot(
                ec_line, theta_line,
                color=colors[hz.horizon], linestyle=':', linewidth=1.5,
                alpha=0.45, zorder=1,
                label=(
                    f'Hz {hz.horizon} — UNCALIBRATED (prior only, no TDR probe)'
                )
            )

    # Configuración estética
    ax.set_title("Step 2: Petrophysical Parameter Alignment (Waxman-Smits)", fontsize=12, pad=12)
    ax.set_xlabel("Inverted Bulk Electrical Conductivity (EC) [mS/m]", fontsize=10)
    ax.set_ylabel("Volumetric Water Content ($\\theta$) [m$^3$/m$^3$]", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)

    # Desplazamos un poco la leyenda para que no pise las curvas si hay muchos datos
    ax.legend(loc='upper left', frameon=True, fontsize=7, bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Petrophysical regression graph successfully exported to '{output_path}'")
def plot_dual_depth_profile(ec_profile, depths0, petro_fit, output_path, depths_are_fixed=True):
    """
    Plots a dual axis depth profile:
    - Left axis (ax1): Electrical Conductivity (EC) profile from inversion.
    - Right axis (ax2): Volumetric Water Content (VWC) observed vs predicted.

    When ``depths_are_fixed=False`` the Y-axis label notes that the shown
    interface depths are starting values and may not reflect final inverted
    positions (interfaces were free to move during optimisation).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Convertir depths0 a array de NumPy para permitir operaciones matemáticas
    depths_arr = np.asarray(depths0)

    # 1. Calcular los centros de las capas a partir de los límites
    layer_centers = (depths_arr[:-1] + depths_arr[1:]) / 2.0

    # 2. Configurar la figura y el primer eje (Conductividad Eléctrica)
    fig, ax1 = plt.subplots(figsize=(6, 8))

    color_ec = 'tab:blue'
    # Graficar el perfil de EC (promediado si ec_profile tiene múltiples columnas/puntos)
    if ec_profile.ndim > 1:
        ec_mean = np.mean(ec_profile, axis=0)
    else:
        ec_mean = ec_profile

    # Asegurar emparejamiento de dimensiones para la línea de EC
    min_len_ec = min(len(ec_mean), len(layer_centers))
    ax1.plot(ec_mean[:min_len_ec], layer_centers[:min_len_ec], color=color_ec,
             linestyle='-', marker='o', linewidth=2, label='EC Profile')

    ax1.set_xlabel('Electrical Conductivity [mS/m]', color=color_ec, fontsize=10)
    depth_ylabel = (
        'Depth [m]' if depths_are_fixed
        else 'Starting Depth [m] (interfaces may have moved)'
    )
    ax1.set_ylabel(depth_ylabel, fontsize=10)
    ax1.tick_params(axis='x', labelcolor=color_ec)
    ax1.invert_yaxis()  # La profundidad aumenta hacia abajo
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 3. Crear el segundo eje espejo para el Contenido de Agua (VWC)
    ax2 = ax1.twiny()
    color_vwc = 'tab:orange'

    # Extraer de forma segura los datos medidos (observed_theta) desde petro_fit
    theta_obs = getattr(petro_fit, 'observed_theta', None)
    if theta_obs is None:
        raise ValueError("El objeto 'petro_fit' no contiene el atributo 'observed_theta'.")

    # 4. TRUCO DE CONTROL DE DIMENSIONES (Safe Trim/Wrap)
    # Evita el ValueError asegurando que X e Y tengan exactamente el mismo tamaño
    min_len_theta = min(len(theta_obs), len(layer_centers))

    plot_x = theta_obs[:min_len_theta]
    plot_y = layer_centers[:min_len_theta]

    # Graficar los datos observados como puntos dispersos (Scatter)
    ax2.scatter(
        plot_x,
        plot_y,
        color=color_vwc,
        marker='s',
        s=80,
        edgecolor='black',
        zorder=5,
        label='Observed Target Data'
    )

    # Opcional: Si petro_fit también tiene predicciones, puedes graficarlas aquí
    theta_pred = getattr(petro_fit, 'predicted_theta', None)
    if theta_pred is not None:
        ax2.plot(theta_pred[:min_len_theta], plot_y, color=color_vwc,
                 linestyle='--', alpha=0.7, label='Predicted VWC')

    ax2.set_xlabel('Volumetric Water Content [m$^3$/m$^3$]', color=color_vwc, fontsize=10)
    ax2.tick_params(axis='x', labelcolor=color_vwc)

    # 4. Ajustes finales, leyenda y guardado
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_one_to_one_validation(
    theta_obs: np.ndarray,
    theta_pred: np.ndarray,
    output_path: str = "mfc2_outputs/parity_validation.png"
) -> None:
    """Generates a diagnostic parity plot validating prediction mapping."""
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(theta_obs, theta_pred, color='mediumseagreen', edgecolor='black', s=80, zorder=3)

    min_val = min(theta_obs.min(), theta_pred.min()) * 0.95
    max_val = max(theta_obs.max(), theta_pred.max()) * 1.05
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1.5, label='1:1 Parity Line')

    # Overall goodness-of-fit: Pearson R^2 and p-value of predicted vs.
    # observed theta, pooled across all horizons. Needs >=2 non-degenerate
    # pairs and non-constant input; falls back to "n/a" rather than raising,
    # since small/uncalibrated ensembles can easily produce a constant
    # theta_pred (e.g. all-fixed WS params) that makes Pearson's r undefined.
    r2_text = "R$^2$ = n/a"
    p_text = "p = n/a"
    try:
        from scipy.stats import pearsonr
        if len(theta_obs) >= 2 and np.std(theta_obs) > 0 and np.std(theta_pred) > 0:
            r_val, p_val = pearsonr(theta_obs, theta_pred)
            r2_text = f"R$^2$ = {r_val**2:.3f}"
            p_text = f"p = {p_val:.3g}"
    except Exception as e:
        print(f"[WARN] Could not compute R2/p-value for parity plot: {e}")

    ax.text(
        0.03, 0.97, f"{r2_text}\n{p_text}  (n={len(theta_obs)})",
        transform=ax.transAxes, fontsize=9, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='gray'),
    )

    ax.set_title("Step 2 Quality Control: Residual Fit Parity", fontsize=11)
    ax.set_xlabel("Observed TDR Soil Moisture [m$^3$/m$^3$]", fontsize=10)
    ax.set_ylabel("Predicted Model Translation [m$^3$/m$^3$]", fontsize=10)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=8)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT] Parity check graph exported to '{output_path}'")


def plot_one_to_one_validation_by_sensor(
    theta_obs: np.ndarray,
    theta_pred: np.ndarray,
    sample_positions: Sequence[int],
    probe_x_coords: Sequence[float] | None = None,
    probe_y_coords: Sequence[float] | None = None,
    sensor_ids: Sequence[str] | None = None,
    output_path: str = "mfc2_outputs/parity_validation_by_sensor.png",
    min_points_per_sensor: int = 2,
    max_labelled_sensors: int = 20,
) -> None:
    """
    Per-sensor companion to ``plot_one_to_one_validation``.

    The pooled parity plot mixes observations from every EM measurement
    point into one scatter/R^2, which is fine with no localisation but
    becomes misleading once ``sample_positions`` spans many spatial
    locations: the pooled R^2 is really an average over N different
    local relationships, and a location-specific bias (e.g. one probe
    installed in a different soil unit) is invisible in the pooled cloud.

    This renders two panels instead:
      * left  -- the same parity scatter, coloured by sensor (spatial
                 index), so clusters/outlying sensors are visible directly;
      * right -- a horizontal bar chart of Pearson R^2 computed
                 independently per sensor, sorted best-to-worst, with the
                 pooled R^2 drawn as a reference line. Sensors with fewer
                 than ``min_points_per_sensor`` observations are shown
                 greyed-out / "n/a" (R^2 is not meaningful from 1 point).

    Parameters
    ----------
    theta_obs, theta_pred : np.ndarray, shape (n_samples,)
        Same arrays passed to ``plot_one_to_one_validation``
        (``best_fit.observed_theta`` / ``best_fit.predicted_theta``).
    sample_positions : sequence of int, length n_samples
        Spatial index into the EM measurement-point axis for each
        observation -- i.e. the same ``sample_positions`` already passed
        into ``run_petro``/``calibrate_petrophysical_relationship``. Used
        as the grouping key when ``sensor_ids`` is not given.
    probe_x_coords, probe_y_coords : sequence of float, optional
        Coordinates of each EM measurement point (length = number of
        distinct measurement points, indexed by the values in
        ``sample_positions``), e.g. ``point_x_coords``/``point_y_coords``
        from ``run_petro_localized`` or the survey's ``utm_x``/``utm_y``.
        When given (and ``sensor_ids`` is not), sensor labels in the bar
        chart show the physical location instead of a bare integer index.
        Purely cosmetic -- ignored once ``sensor_ids`` is given, since
        real station IDs are a strictly better label.
    sensor_ids : sequence of str, length n_samples, optional
        Real TDR station ID (e.g. ``probe_table['station_name']``, or the
        synthetic ``PROBE_LABELS``) for each individual observation --
        same order/length as ``theta_obs``/``sample_positions``, i.e.
        ``EMmethod.SamplePositionResult.probe_id`` expanded per
        observation, NOT deduplicated per EM point. This is the key
        upgrade over grouping by ``sample_positions`` alone: several TDR
        stations can spatially collapse onto the SAME EM measurement
        point (``extract_sample_positions_from_survey``'s
        ``representative_index`` -- e.g. 11 of 20 stations onto one EM
        point in a real-data run), and grouping by ``sample_positions``
        would silently merge those distinct stations into one bar/color.
        When given, grouping and labeling both switch to the real
        per-station identity instead, so each physical sensor gets its
        own bar even if it shares an EM point with others. ``None``
        (default) falls back to the original ``sample_positions``-based
        grouping, unchanged behaviour from before this argument existed.
    output_path : str, optional
        Where to save the figure.
    min_points_per_sensor : int, optional
        Minimum observations at a sensor to compute its own R^2. Default 2
        (Pearson r needs >=2 non-degenerate pairs); sensors below this are
        still shown in the bar chart, greyed out, so you can see coverage
        gaps rather than have them silently disappear.
    max_labelled_sensors : int, optional
        Above this many distinct sensors, y-axis tick labels are dropped
        from the bar chart (still one bar per sensor, sorted) to keep it
        legible -- the scatter panel remains the way to inspect individual
        points at that point.
    """
    from scipy.stats import pearsonr

    theta_obs = np.asarray(theta_obs, dtype=float)
    theta_pred = np.asarray(theta_pred, dtype=float)
    sample_positions = np.asarray(sample_positions)

    if len(theta_obs) != len(sample_positions):
        print(
            f"[WARN] plot_one_to_one_validation_by_sensor: theta_obs has "
            f"{len(theta_obs)} entries but sample_positions has "
            f"{len(sample_positions)} -- skipping (lengths must match, "
            f"one entry per observation)."
        )
        return

    use_sensor_ids = sensor_ids is not None
    if use_sensor_ids:
        sensor_ids = np.asarray(sensor_ids, dtype=object)
        if len(sensor_ids) != len(theta_obs):
            print(
                f"[WARN] plot_one_to_one_validation_by_sensor: sensor_ids has "
                f"{len(sensor_ids)} entries but theta_obs has {len(theta_obs)} "
                f"-- falling back to sample_positions-based grouping."
            )
            use_sensor_ids = False
        elif any(sid is None for sid in sensor_ids):
            # Missing IDs for some observations (e.g. an older run's
            # position_results without probe_ids attached) -- fall back to
            # "#pos" for just those entries rather than losing the whole
            # per-observation split.
            sensor_ids = np.array(
                [sid if sid is not None else f"#{pos}"
                 for sid, pos in zip(sensor_ids, sample_positions)],
                dtype=object,
            )

    group_key = sensor_ids if use_sensor_ids else sample_positions

    uniq_groups = np.unique(group_key)
    if len(uniq_groups) <= 1:
        print(
            "[PLOT] plot_one_to_one_validation_by_sensor: only one distinct "
            "sensor position present -- nothing to facet, skipping "
            "(the pooled parity_validation.png plot already covers this case)."
        )
        return

    def _sensor_label(group) -> str:
        if use_sensor_ids:
            return str(group)
        pos = group
        if probe_x_coords is not None:
            x = np.asarray(probe_x_coords)[pos]
            if probe_y_coords is not None:
                y = np.asarray(probe_y_coords)[pos]
                return f"#{pos} (x={x:.1f}, y={y:.1f})"
            return f"#{pos} (x={x:.1f})"
        return f"#{pos}"

    # Per-sensor R^2 (Pearson r^2), RMSE, and n.
    rows = []
    for group in uniq_groups:
        mask = group_key == group
        n = int(mask.sum())
        obs_i, pred_i = theta_obs[mask], theta_pred[mask]
        r2 = np.nan
        if n >= min_points_per_sensor and np.std(obs_i) > 0 and np.std(pred_i) > 0:
            try:
                r_val, _ = pearsonr(obs_i, pred_i)
                r2 = r_val ** 2
            except Exception as e:
                print(f"[WARN] Could not compute per-sensor R2 for sensor {group}: {e}")
        rmse = float(np.sqrt(np.mean((obs_i - pred_i) ** 2))) if n > 0 else np.nan
        rows.append({"pos": group, "n": n, "r2": r2, "rmse": rmse})

    # Sorted best-to-worst by R^2 (NaNs -- insufficient coverage -- last).
    rows_sorted = sorted(rows, key=lambda r: (-r["r2"] if not np.isnan(r["r2"]) else 1e9))

    # Pooled R^2, for the reference line (same computation as
    # plot_one_to_one_validation).
    pooled_r2 = np.nan
    if len(theta_obs) >= 2 and np.std(theta_obs) > 0 and np.std(theta_pred) > 0:
        r_val, _ = pearsonr(theta_obs, theta_pred)
        pooled_r2 = r_val ** 2

    fig, (ax_scatter, ax_bar) = plt.subplots(
        1, 2, figsize=(12, max(5, 0.32 * len(uniq_groups) + 2))
    )

    # --- Left: pooled scatter, coloured by sensor -------------------------
    cmap = plt.get_cmap("tab20" if len(uniq_groups) <= 20 else "viridis")
    color_norm = {group: i / max(1, len(uniq_groups) - 1)
                  for i, group in enumerate(uniq_groups)}
    colors = np.array([cmap(color_norm[g]) for g in group_key])

    min_val = min(theta_obs.min(), theta_pred.min()) * 0.95
    max_val = max(theta_obs.max(), theta_pred.max()) * 1.05
    ax_scatter.plot([min_val, max_val], [min_val, max_val],
                     color='red', linestyle='--', linewidth=1.2, zorder=2,
                     label='1:1 Parity Line')
    ax_scatter.scatter(theta_obs, theta_pred, c=colors, edgecolor='black',
                        linewidth=0.4, s=60, zorder=3)
    ax_scatter.set_xlim(min_val, max_val)
    ax_scatter.set_ylim(min_val, max_val)
    ax_scatter.set_xlabel("Observed TDR Soil Moisture [m$^3$/m$^3$]", fontsize=10)
    ax_scatter.set_ylabel("Predicted Model Translation [m$^3$/m$^3$]", fontsize=10)
    ax_scatter.set_title("Coloured by sensor location", fontsize=10)
    ax_scatter.grid(True, linestyle=':', alpha=0.6)
    ax_scatter.legend(loc='lower right', fontsize=8)

    # --- Right: per-sensor R^2 bar chart -----------------------------------
    labels = [_sensor_label(r["pos"]) for r in rows_sorted]
    r2_vals = [r["r2"] for r in rows_sorted]
    y_pos = np.arange(len(rows_sorted))
    bar_colors = [cmap(color_norm[r["pos"]]) if not np.isnan(r["r2"]) else "lightgrey"
                  for r in rows_sorted]
    plot_vals = [0.0 if np.isnan(v) else v for v in r2_vals]

    ax_bar.barh(y_pos, plot_vals, color=bar_colors, edgecolor='black', linewidth=0.4)
    if not np.isnan(pooled_r2):
        ax_bar.axvline(pooled_r2, color='red', linestyle='--', linewidth=1.2,
                        label=f'Pooled R$^2$ = {pooled_r2:.3f}')
    ax_bar.set_yticks(y_pos)
    if len(rows_sorted) <= max_labelled_sensors:
        ax_bar.set_yticklabels(labels, fontsize=7)
    else:
        ax_bar.set_yticklabels([])
        ax_bar.set_ylabel(f"{len(rows_sorted)} sensors (sorted, best to worst)", fontsize=9)
    ax_bar.invert_yaxis()  # best sensor at top
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xlabel("Per-sensor R$^2$", fontsize=10)
    ax_bar.set_title("Goodness-of-fit per sensor", fontsize=10)
    ax_bar.grid(True, axis='x', linestyle=':', alpha=0.6)
    ax_bar.legend(loc='lower right', fontsize=8)

    # Annotate n and "n/a" for under-covered sensors directly on the bars.
    for yi, r in zip(y_pos, rows_sorted):
        label = "n/a" if np.isnan(r["r2"]) else f"n={r['n']}"
        ax_bar.text(0.01, yi, label, va='center', ha='left', fontsize=6.5,
                    color='dimgray')

    fig.suptitle("Step 2 Quality Control: Per-Sensor Residual Fit", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    n_covered = sum(1 for r in rows if not np.isnan(r["r2"]))
    print(
        f"[PLOT] Per-sensor parity check exported to '{output_path}' "
        f"({n_covered}/{len(rows)} sensors had >= {min_points_per_sensor} "
        f"observation(s); pooled R$^2$ = "
        f"{'n/a' if np.isnan(pooled_r2) else f'{pooled_r2:.3f}'})"
    )


def plot_ws_parameter_histograms(fits, best_fit, depths0, lab_p0, output_path):
    """
    Plots histograms of the Waxman-Smits petrophysical parameters
    for each optimized horizon. Prevents ValueError if data range is zero.

    For each parameter/horizon panel, the y-axis shows the bin count and the
    x-axis shows the parameter value. Two reference lines are overlaid:
      - a blue dashed line for the laboratory-optimized parameter (``lab_p0``),
        used as the initial value for the field-scale optimization;
      - a green line for the field-optimized parameter of the selected
        ensemble model (``best_fit``).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- CORRECCIÓN AQUÍ ---
    # Cambiamos las llaves para que coincidan exactamente con tus objetos PetroFit
    param_keys = ['a', 'm', 'sigma_w', 'sigma_s']
    # Display labels for the x-axis of each parameter row (LaTeX-style subscripts).
    param_labels = {
        'a': r'$a$',
        'm': r'$m$',
        'sigma_w': r'$\sigma_w$',
        'sigma_s': r'$\sigma_s$',
    }
    # -----------------------

    num_params = len(param_keys)
    num_horizons = len(best_fit.horizon_fits)

    # Normalize lab_p0 into a (num_horizons, num_params) array, or None if unavailable.
    lab_p0_arr = None
    if lab_p0 is not None:
        lab_p0_arr = np.asarray(lab_p0, dtype=float)
        if lab_p0_arr.ndim == 1:
            lab_p0_arr = np.tile(lab_p0_arr, (num_horizons, 1))

    fig, axes = plt.subplots(num_params, num_horizons, figsize=(3 * num_horizons, 2.5 * num_params), squeeze=False)

    # El resto del bucle se mantiene exactamente igual...
    for p_idx, key in enumerate(param_keys):
        for h_idx in range(num_horizons):
            ax = axes[p_idx][h_idx]

            try:
                data = [getattr(f.horizon_fits[h_idx], key) for f in fits if hasattr(f, 'horizon_fits')]
            except (AttributeError, IndexError):
                data = []

            data_arr = np.asarray(data, dtype=float)

            if data_arr.size == 0:
                num_bins = 1
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')

            # Verificamos si todos los valores son virtualmente idénticos (tolerancia flotante)
            elif np.all(np.isclose(data_arr, data_arr[0], atol=1e-5)):
                num_bins = 1

            else:
                # Regla por defecto si hay una variabilidad física real
                num_bins = max(6, len(data_arr) // 5)
            # ----------------------------------------------------------

            ax.hist(data_arr, bins=num_bins, color='steelblue', edgecolor='white', alpha=0.75)

            # Say what a collapsed panel means instead of leaving a bare bar on
            # an arbitrary axis: no spread, and/or members stuck on a bound.
            if data_arr.size:
                _lo = EMmethod.WS_BOUNDS[0][p_idx]
                _hi = EMmethod.WS_BOUNDS[1][p_idx]
                _tol = 1e-6 * max(1.0, abs(_hi))
                _n = data_arr.size
                _n_lo = int(np.sum(data_arr <= _lo + _tol))
                _n_hi = int(np.sum(data_arr >= _hi - _tol))
                _notes = []
                if np.all(np.isclose(data_arr, data_arr[0], atol=1e-5)):
                    _pad = max(abs(float(data_arr[0])) * 0.05, 1e-3)
                    ax.set_xlim(float(data_arr[0]) - _pad, float(data_arr[0]) + _pad)
                    _notes.append(f"no spread (all = {float(data_arr[0]):.4g})")
                if _n_lo:
                    _notes.append(f"{_n_lo}/{_n} at lower bound {_lo:g}")
                if _n_hi:
                    _notes.append(f"{_n_hi}/{_n} at upper bound {_hi:g}")
                if _notes:
                    ax.text(0.97, 0.95, "\n".join(_notes), transform=ax.transAxes,
                            ha='right', va='top', fontsize=7, color='firebrick',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))

            # Blue dashed line: laboratory-optimized parameter (initial value for
            # the field-scale optimization).
            if lab_p0_arr is not None:
                try:
                    lab_val = lab_p0_arr[h_idx, p_idx]
                    ax.axvline(lab_val, color='royalblue', linestyle='--', linewidth=2)
                except IndexError:
                    pass

            # Green line: field-optimized parameter for the selected ensemble model.
            try:
                best_val = getattr(best_fit.horizon_fits[h_idx], key)
                ax.axvline(best_val, color='forestgreen', linestyle='-', linewidth=2)
            except (AttributeError, IndexError):
                pass

            if p_idx == 0:
                ax.set_title(f'Horizon {h_idx + 1}', fontsize=10, fontweight='bold')
            if h_idx == 0:
                ax.set_ylabel('Count', fontsize=10)

            ax.set_xlabel(param_labels.get(key, key), fontsize=11, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.tick_params(axis='both', labelsize=9)

    # Shared figure-level legend explaining the two reference lines.
    legend_handles = [
        plt.Line2D([0], [0], color='royalblue', linestyle='--', linewidth=2,
                   label='Laboratory-optimized (initial) parameters'),
        plt.Line2D([0], [0], color='forestgreen', linestyle='-', linewidth=2,
                   label='Field-optimized parameters (selected ensemble model)'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
        frameon=False,
    )

    fig.suptitle(
        "Blue dashed lines show laboratory-optimized parameters used as initial values\n"
        "for the field-scale optimization. Green lines show field-optimized parameters\n"
        "obtained for the selected ensemble model. (For interpretation of the references\n"
        "to colour in this figure legend, the reader is referred to the web version of\n"
        "this article.)",
        fontsize=8, y=1.02,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_localized_sigma_s_profile(
    localized_fit: "EMmethod.LocalizedPetroFit",
    probe_x_coords: Sequence[float] | None = None,
    probe_layer_indices: Sequence[int] | None = None,
    params: Sequence[str] = ("sigma_s",),
    output_path: str = "mfc2_outputs/petro_calibration_localized.png",
) -> None:
    """
    Diagnostic for kernel-localized calibration (spatial_calibration_alternatives.md
    §2): plots one or more WS parameters varying along the transect per
    horizon, instead of the single flat value ``plot_petrophysical_calibration``
    shows for the un-localized fit. This is the plot that actually makes the
    point of localizing visible — a flat line would mean the kernel made no
    difference; a curve that dips toward each nearby probe's own value and
    relaxes toward the horizon-wide behaviour far from any probe is the
    expected, useful result.

    Historically this only ever plotted sigma_s. It now accepts ``params``
    so sigma_w can be plotted alongside it — important whenever a scenario's
    ``ws_fixed_params`` leaves BOTH sigma_w and sigma_s free (e.g. the
    default 2d/2d_dense_tdr scenarios): since they trade off against each
    other in the WS forward model (theta ~ (sigma_w + sigma_s)), a
    sigma_s-only plot can't show whether the localization is actually
    pushing the non-identifiability into sigma_w instead. a/m are also
    accepted for completeness (e.g. if a scenario frees them), though they
    don't share sigma_w/sigma_s's [mS/m] units — mixing dimensionless and
    conductivity params on one axis is visually confusing, so prefer
    calling this once per unit-compatible group rather than passing all
    four at once.

    Note: if the fit was produced with 2-D Euclidean (x, y) localization
    (``point_y_coords`` given), this plot still projects onto x only —
    two targets at the same x but different y can show different values
    at the same x position on this plot. Fine for a 1-D transect; for a
    genuine 2-D grid, treat this as a diagnostic slice, not the full
    picture, and inspect ``localized_fit.horizon_fits[i].target_y``
    alongside it if a true 2-D map is needed (see ``plot_localized_sigma_s_map``,
    which already supports any of a/m/sigma_w/sigma_s via its own
    ``param`` argument).

    Un-calibrated targets (``is_calibrated=False`` — no observation fell
    within ``2*bandwidth`` of that point) are drawn as a faint dotted
    segment, matching how ``plot_petrophysical_calibration`` marks
    uncalibrated horizons.

    Parameters
    ----------
    localized_fit : EMmethod.LocalizedPetroFit
        Typically the best/selected ensemble member's localized fit.
    probe_x_coords : sequence of float, optional
        x-coordinates of the TDR probes, drawn as vertical reference lines
        so it's visually clear where the real data anchors the curve.
    probe_layer_indices : sequence of int, optional
        Horizon index for each probe in ``probe_x_coords`` — used so each
        probe's marker only appears on its own horizon's panel. If omitted,
        every probe line is drawn on every panel.
    params : sequence of str
        Which fitted parameter(s) to plot, each one of ``"a"``, ``"m"``,
        ``"sigma_w"``, ``"sigma_s"``. Default ``("sigma_s",)`` — unchanged
        behaviour from before this argument existed. Pass e.g.
        ``("sigma_w", "sigma_s")`` to overlay both on the same axes (they
        share [mS/m] units, so this is the natural pairing).
    output_path : str
        Where to save the figure.
    """
    _valid_params = ("a", "m", "sigma_w", "sigma_s")
    params = tuple(params)
    for p in params:
        if p not in _valid_params:
            raise ValueError(f"params entries must be one of {_valid_params}, got {p!r}.")
    if not params:
        raise ValueError("params must contain at least one parameter name.")

    _param_labels = {
        "a": "$a$", "m": "$m$",
        "sigma_w": "$\\sigma_w$ [mS/m]", "sigma_s": "$\\sigma_s$ [mS/m]",
    }
    _param_symbols = {"a": "$a$", "m": "$m$", "sigma_w": "$\\sigma_w$", "sigma_s": "$\\sigma_s$"}
    # Distinct linestyle/marker per parameter so overlaid curves stay
    # visually distinguishable even though they share the horizon's color.
    _param_styles = {
        "sigma_s": dict(linestyle='-', marker='o'),
        "sigma_w": dict(linestyle='--', marker='s'),
        "a": dict(linestyle='-.', marker='^'),
        "m": dict(linestyle=':', marker='D'),
    }

    num_horizons = len(localized_fit.horizon_fits)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, num_horizons))

    fig, axes = plt.subplots(
        num_horizons, 1, figsize=(8, 3 * num_horizons), squeeze=False, sharex=True,
    )

    for h_idx, hz in enumerate(localized_fit.horizon_fits):
        ax = axes[h_idx][0]
        x = hz.target_x
        calibrated = hz.is_calibrated
        order = np.argsort(x)
        x_sorted, cal_sorted = x[order], calibrated[order]

        for p_i, param in enumerate(params):
            y = getattr(hz, param)
            y_sorted = y[order]

            # Draw calibrated stretches solid, uncalibrated stretches
            # faint/dotted, by masking rather than filtering, so gaps in
            # coverage are visible as gaps rather than silently bridged.
            y_cal = np.where(cal_sorted, y_sorted, np.nan)
            y_uncal = np.where(~cal_sorted, y_sorted, np.nan)

            style = _param_styles.get(param, dict(linestyle='-', marker='o'))
            ax.plot(x_sorted, y_cal, color=colors[h_idx], linewidth=2,
                    markersize=4, label=f'Localized {_param_symbols[param]}(x) — Hz {h_idx}',
                    **style)
            ax.plot(x_sorted, y_uncal, color=colors[h_idx], linewidth=1.5,
                    linestyle=':', alpha=0.45, marker='x', markersize=5,
                    label=(f'Uncalibrated (no obs within kernel support)'
                           if p_i == 0 else None))

        if probe_x_coords is not None:
            for p_i, px in enumerate(probe_x_coords):
                if (probe_layer_indices is not None
                        and probe_layer_indices[p_i] != h_idx):
                    continue
                ax.axvline(px, color='gray', linestyle='--', alpha=0.5,
                           linewidth=1)

        if len(params) == 1:
            ax.set_ylabel(_param_labels[params[0]], fontsize=10)
        else:
            ax.set_ylabel(" / ".join(_param_labels[p] for p in params), fontsize=10)
        ax.set_title(
            f'Horizon {h_idx} — bandwidth c={hz.bandwidth:g}', fontsize=10,
        )
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best', fontsize=8)

    axes[-1][0].set_xlabel('Along-transect position x', fontsize=10)
    _param_join = ", ".join(_param_symbols[p] for p in params)
    fig.suptitle(
        f'Step 2 (localized): {_param_join} varying continuously along the '
        'transect per horizon, instead of one flat value per horizon',
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    _param_names = "/".join(params)
    print(f"[PLOT] Localized {_param_names}(x) profile exported to '{output_path}'")


def plot_localized_sigma_s_map(
    localized_fit: "EMmethod.LocalizedPetroFit",
    probe_x_coords: Sequence[float] | None = None,
    probe_y_coords: Sequence[float] | None = None,
    probe_layer_indices: Sequence[int] | None = None,
    param: str = "sigma_s",
    output_path: str = "mfc2_outputs/petro_calibration_localized_map.png",
) -> None:
    """
    True 2-D map of a kernel-localized WS parameter (default sigma_s), one
    panel per horizon, plotting against BOTH ``target_x`` and ``target_y`` --
    the full-picture companion to ``plot_localized_sigma_s_profile``, which
    only projects onto x and can't distinguish two targets at the same x but
    different y.

    Requires the fit to have been produced with ``point_y_coords`` (2-D
    Euclidean localization) -- i.e. every ``horizon_fits[i].target_y`` is
    not ``None``. Raises a clear error otherwise rather than silently
    drawing a meaningless 1-D map.

    Uncalibrated targets (``is_calibrated=False``) are drawn as small gray
    x-markers instead of being colored by value, so missing coverage reads
    as a gap rather than a misleadingly-colored point.

    Parameters
    ----------
    localized_fit : EMmethod.LocalizedPetroFit
        Typically the best/selected ensemble member's localized fit, from a
        call to ``run_petro_localized(..., point_y_coords=...)``.
    probe_x_coords, probe_y_coords : sequence of float, optional
        TDR probe coordinates, overlaid as black triangles so it's visually
        clear where the real observations anchor the map.
    probe_layer_indices : sequence of int, optional
        Horizon index for each probe -- restricts each probe marker to its
        own horizon's panel. If omitted, every probe is shown on every panel.
    param : str
        Which fitted parameter to map: one of ``"a"``, ``"m"``,
        ``"sigma_w"``, ``"sigma_s"``. Default ``"sigma_s"`` (the parameter
        the rest of this module's diagnostics focus on).
    output_path : str
        Where to save the figure.
    """
    if param not in ("a", "m", "sigma_w", "sigma_s"):
        raise ValueError(f"param must be one of a/m/sigma_w/sigma_s, got {param!r}.")

    num_horizons = len(localized_fit.horizon_fits)
    for hz in localized_fit.horizon_fits:
        if hz.target_y is None:
            raise ValueError(
                "plot_localized_sigma_s_map requires a fit produced with "
                "point_y_coords (2-D Euclidean localization) -- this fit's "
                "target_y is None (1-D along-transect localization was "
                "used). Use plot_localized_sigma_s_profile instead."
            )

    fig, axes = plt.subplots(
        1, num_horizons, figsize=(5.5 * num_horizons + 1.2, 5),
        squeeze=False, sharex=True, sharey=True,
    )

    # Shared color scale across horizons (only over calibrated targets) so
    # panels are visually comparable rather than each auto-scaling to its
    # own range.
    all_vals = np.concatenate([
        getattr(hz, param)[hz.is_calibrated] for hz in localized_fit.horizon_fits
        if np.any(hz.is_calibrated)
    ]) if any(np.any(hz.is_calibrated) for hz in localized_fit.horizon_fits) else np.array([0.0, 1.0])
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    _param_labels = {
        "a": "$a$", "m": "$m$", "sigma_w": "$\\sigma_w$ [mS/m]",
        "sigma_s": "$\\sigma_s$ [mS/m]",
    }

    sc = None
    for h_idx, hz in enumerate(localized_fit.horizon_fits):
        ax = axes[0][h_idx]
        vals = getattr(hz, param)
        cal = hz.is_calibrated

        if np.any(cal):
            sc = ax.scatter(
                hz.target_x[cal], hz.target_y[cal], c=vals[cal],
                cmap="viridis", vmin=vmin, vmax=vmax, s=28,
                edgecolors="none",
            )
        if np.any(~cal):
            ax.scatter(
                hz.target_x[~cal], hz.target_y[~cal], marker="x",
                color="gray", s=18, alpha=0.5, label="Uncalibrated",
            )

        if probe_x_coords is not None and probe_y_coords is not None:
            for p_i, (px, py) in enumerate(zip(probe_x_coords, probe_y_coords)):
                if (probe_layer_indices is not None
                        and probe_layer_indices[p_i] != h_idx):
                    continue
                ax.scatter([px], [py], marker="^", color="black", s=45,
                           zorder=5, label="TDR probe" if p_i == 0 else None)

        ax.set_title(f"Horizon {h_idx} — bandwidth c={hz.bandwidth:g}", fontsize=10)
        ax.set_xlabel("x", fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.4)
        if h_idx == 0:
            ax.set_ylabel("y", fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=7)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, fraction=0.035, pad=0.02)
        cbar.set_label(_param_labels[param], fontsize=10)

    fig.suptitle(
        f"Step 2 (localized, 2-D): {_param_labels[param]} varying over the "
        "full survey grid per horizon",
        fontsize=10, y=1.02,
    )
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Localized {param}(x, y) map exported to '{output_path}'")


# ==========================================================================
# STANDALONE EXECUTION ENGINE
# ==========================================================================

def _extract_ec_for_plots(
    models: np.ndarray,
    sample_positions: Sequence[int],
    sample_times: Sequence[int],
    probe_layer_indices: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract matched (σ_b, layer) pairs for the best ensemble member (index 0),
    mirroring the routing logic inside
    ``EMmethod.calibrate_petrophysical_relationship`` exactly so the plotted
    points correspond to what was actually fitted.

    Works for both 3D ``(ensemble_size, num_measured_points, num_layers)`` and
    4D ``(ensemble_size, n_time, num_measured_points, num_layers)`` arrays.
    For 4D data each probe's σ_b is read at its own survey time step
    (``sample_times[i]``), matching the calibration engine's behaviour.

    Returns
    -------
    ec_extracted : np.ndarray, shape (n_calibration_samples,)
        Flat σ_b values in the same order as ``best_fit.observed_theta``.
    spatial_profile : np.ndarray, shape (num_measured_points, num_layers)
        Time-averaged (or single-time) EC for the best ensemble member —
        used for the dual-depth profile plot.
    kept_indices : np.ndarray, shape (n_calibration_samples,)
        Index into the ORIGINAL ``sample_positions``/``sample_times``/
        ``probe_layer_indices`` (length = len(sample_positions)) that each
        element of ``ec_extracted`` came from — includes both the horizon
        routing AND the NaN-observation drop (real-data time-lapse
        reference-grid gaps), so it's guaranteed to match
        ``best_fit.observed_theta`` exactly. Use this — NOT
        ``geometry_utils.route_array_to_horizons`` — to route any other
        per-observation array (``sample_positions``, ``probe_layer_indices``,
        per-probe coordinates) so it lines up with the fitted values:
        ``routed = np.asarray(other_array)[kept_indices]``.
    """
    timelapse = models.ndim == 4

    if timelapse:
        # best member: (n_time, num_measured_points, num_layers)
        best_member = models[0]
        # Spatial profile for depth plot: mean across time steps
        spatial_profile = best_member.mean(axis=0)   # (num_measured_points, num_layers)
        num_layers = models.shape[3]
    else:
        best_member = models[0]                       # (num_measured_points, num_layers)
        spatial_profile = best_member
        num_layers = models.shape[2]

    # Horizon routing (+ NaN-observation drop) now delegates to the single
    # shared implementation in geometry_utils.py (pipeline_improvement_
    # report.md, section 6 / action item 3) instead of an independently
    # reimplemented copy of the same strided/modulo/explicit-index logic
    # used by em_methodology.calibrate_petrophysical_relationship and by
    # step4_select.py::_extract_ec_at_probes.
    ec_extracted, _horizon_slices, kept_indices = geometry_utils.route_ec_to_horizons(
        ec_array=best_member,
        sample_positions=sample_positions,
        sample_times=sample_times,
        probe_layer_indices=probe_layer_indices,
        num_layers=num_layers,
    )
    return ec_extracted, spatial_profile, kept_indices



def run_petro(
    models: np.ndarray,
    sample_positions: Sequence[int],
    sample_times: Sequence[int],
    theta_obs: np.ndarray,
    depths0: list[float] | None = None,
    output_dir: str = "mfc2_outputs",
    generate_plots: bool = True,
    probe_layer_indices: Sequence[int] | None = None,
    depths_are_fixed: bool = True,
    fixed_params: dict[str, bool] | None = None,
    probe_depths_reflect_fitted_interfaces: bool = False,
    lab_p0: np.ndarray | list | None = None,
    theta_obs_std: float | Sequence[float] | None = None,
    sensor_ids: Sequence[str] | None = None,
) -> list[EMmethod.PetroFit]:
    """
    Run STEP 2: Calibrate petrophysical function using Option A spatial layouts.

    Supports both single-time (3D) and time-lapse (4D) model arrays.

    Parameters
    ----------
    models : np.ndarray
        Shape ``(ensemble_size, num_measured_points, num_layers)`` for
        single-time surveys, or
        ``(ensemble_size, n_time, num_measured_points, num_layers)`` for
        time-lapse surveys.
    sample_positions : sequence of int
        Spatial index into the ``num_measured_points`` axis for each TDR probe.
    sample_times : sequence of int
        0-based time-step index for each TDR observation. Pass all-zeros for
        single-time surveys. For time-lapse surveys each entry selects which
        survey epoch the corresponding ``theta_obs`` reading was collected at.
    theta_obs : np.ndarray, shape (n_samples,)
        Observed volumetric water content [m³/m³].
    depths0 : list of float, optional
        Layer interface depths [m]. Used for labelling and depth plots.
        When ``depths_are_fixed=False`` these are the *starting* positions
        and may not reflect where interfaces ended up after inversion.
    output_dir : str
        Directory for diagnostic output plots.
    generate_plots : bool
        When True, produce all diagnostic figures.
    probe_layer_indices : sequence of int, optional
        0-based horizon index for each TDR probe, same order as
        ``sample_positions``/``theta_obs``. Pass this whenever probe
        installation depths are known (via
        ``EMmethod.extract_sample_positions_from_survey(..., probe_depths=...,
        depths0=...)`` + ``EMmethod.sample_layer_indices_from_results``) so
        each observation is routed to the horizon it actually came from,
        instead of the spatial-index stride/modulo heuristic.
    depths_are_fixed : bool, optional
        Must match the ``FIXED_DEPTHS`` setting used in Step 1.  When
        ``False``, a message is propagated through calibration (the
        horizon-to-probe routing is approximate) and the depth-profile
        plot Y-axis label is updated to reflect that the shown interface
        depths are starting values, not final inverted positions.
        Default ``True`` (classic fixed-depth behaviour).
    fixed_params : dict[str, bool], optional
        Forwarded unchanged to ``EMmethod.calibrate_petrophysical_relationship``.
        Holds one or more of ``'a'``, ``'m'``, ``'sigma_w'``, ``'sigma_s'``
        fixed at their prior (lab/default) value instead of letting all four
        float — useful here too since Step 2 fits against the same sparse
        per-horizon TDR observations as Step 4. ``None`` (default) leaves
        everything free, unchanged from before.
    probe_depths_reflect_fitted_interfaces : bool, optional
        Forwarded unchanged to ``EMmethod.calibrate_petrophysical_relationship``.
        Set to ``True`` when ``probe_layer_indices`` was computed against
        the ensemble's actual fitted interface depths (Step 1's
        ``depths_clean`` / ``k.depths[0]``) rather than the static starting
        ``depths0`` — this downgrades the free-depth message from "[WARN]"
        to "[INFO]" since the routing already accounts for interface
        movement. Default ``False`` (routing assumed to be against the
        static starting ``depths0``).
    lab_p0 : array-like, optional
        Forwarded unchanged to ``EMmethod.calibrate_petrophysical_relationship``.
        Starting-guess / fixed-value Waxman-Smits parameters ``[a, m,
        sigma_w, sigma_s]``, either one set applied to every horizon
        (shape ``(4,)``) or one set per horizon (shape
        ``(num_layers, 4)``). Used both as the optimizer's starting point
        for free parameters AND as the pinned value for any parameter
        named in ``fixed_params``. ``None`` (default) falls back to
        ``EMmethod.DEFAULT_WS_P0 = [1.5, 1.8, 50.0, 5.0]`` for every
        horizon, unchanged from before.
    theta_obs_std : float or sequence of float, optional
        FIX ("TDR measurement uncertainty is collected but never used to
        weight the calibration"): forwarded unchanged to
        ``EMmethod.calibrate_petrophysical_relationship``, which uses it to
        inverse-variance-weight each observation's contribution to the
        per-horizon data-misfit term (renormalised to mean 1, so the
        overall misfit scale — and therefore ``lambda_reg`` — doesn't shift
        purely because weighting was turned on). Typically the scenario's
        ``tdr_noise_std`` for synthetic data; ``None`` (default, and what
        real-data runs should generally pass, since this pipeline doesn't
        yet have a genuine per-probe real-data noise estimate — see
        ``run_pipeline.py``'s Step 2/4 call sites) disables weighting
        entirely, i.e. no behaviour change from before this parameter
        existed.
    """
    os.makedirs(output_dir, exist_ok=True)

    timelapse = models.ndim == 4
    if timelapse:
        n_time = models.shape[1]
        print(f"[STEP 2] Time-lapse mode: {n_time} survey epochs detected.")
    else:
        print("[STEP 2] Single-time mode.")

    # Core calibration — handles both 3D and 4D internally
    fits = EMmethod.calibrate_petrophysical_relationship(
        models=models,
        sample_positions=list(sample_positions),
        sample_times=list(sample_times),
        theta_obs=theta_obs,
        depths0=depths0,
        probe_layer_indices=probe_layer_indices,
        depths_are_fixed=depths_are_fixed,
        fixed_params=fixed_params,
        probe_depths_reflect_fitted_interfaces=probe_depths_reflect_fitted_interfaces,
        lab_p0=lab_p0,
        theta_obs_std=theta_obs_std,
    )
    print(f"[STEP 2] Completed parameter extraction for {len(fits)} ensemble members.")

    # Print per-horizon Waxman-Smits parameters for the best-ranked fit
    best_fit = fits[0]
    print("[STEP 2] Best-fit Waxman-Smits parameters per horizon:")
    for hz in best_fit.horizon_fits:
        print(
            f"  Horizon {hz.horizon}: "
            f"a={hz.a:.4f}, m={hz.m:.4f}, "
            f"sigma_w={hz.sigma_w:.2f} mS/m, sigma_s={hz.sigma_s:.2f} mS/m, "
            f"RMSE={hz.rmse:.4f}"
        )

    if generate_plots:
        print("[STEP 2] Generating diagnostic plots...")

        # Extract EC values and a spatial profile for plotting.
        # For 4D arrays: each probe's σ_b is read at its own survey time step,
        # and the depth profile uses the time-averaged EC of the best member.
        ec_extracted, spatial_profile, kept_indices = _extract_ec_for_plots(
            models=models,
            sample_positions=sample_positions,
            sample_times=sample_times,
            probe_layer_indices=probe_layer_indices,
        )

        plot_petrophysical_calibration(
            ec_values=ec_extracted,
            theta_obs=best_fit.observed_theta,
            petro_fit=best_fit,
            output_path=os.path.join(output_dir, "petro_calibration.png"),
        )

        if depths0 is not None:
            # spatial_profile is always (num_measured_points, num_layers) here
            plot_dual_depth_profile(
                ec_profile=spatial_profile,
                depths0=depths0,
                petro_fit=best_fit,
                output_path=os.path.join(output_dir, "tdr_ec_depth_profile.png"),
                depths_are_fixed=depths_are_fixed,
            )

        plot_one_to_one_validation(
            theta_obs=best_fit.observed_theta,
            theta_pred=best_fit.predicted_theta,
            output_path=os.path.join(output_dir, "parity_validation.png"),
        )

        # Per-sensor companion: only meaningful once observations actually
        # come from more than one spatial location (i.e. --mode 2d / any
        # localisation run with several probes). No-op (prints and returns)
        # for the classic single-location / 1-D case, so this is safe to
        # leave on unconditionally.
        #
        # IMPORTANT: best_fit.observed_theta/predicted_theta are routed to
        # HORIZON order AND have any non-finite-EC observation dropped
        # (see geometry_utils.route_ec_to_horizons) -- so sample_positions
        # must be gathered via the SAME kept_indices _extract_ec_for_plots
        # just computed above, not re-routed independently, or sensor
        # labels could silently line up with the wrong observations (or,
        # once any NaN-drop has happened, with the wrong COUNT of
        # observations entirely).
        routed_sample_positions = np.asarray(sample_positions)[kept_indices]
        routed_sensor_ids = (
            np.asarray(sensor_ids, dtype=object)[kept_indices]
            if sensor_ids is not None else None
        )
        plot_one_to_one_validation_by_sensor(
            theta_obs=best_fit.observed_theta,
            theta_pred=best_fit.predicted_theta,
            sample_positions=routed_sample_positions,
            sensor_ids=routed_sensor_ids,
            output_path=os.path.join(output_dir, "parity_validation_by_sensor.png"),
        )

    return fits


def plot_petrophysical_calibration_localized_overlay(
    ec_values: np.ndarray,
    theta_obs: np.ndarray,
    petro_fit: "EMmethod.PetroFit",
    localized_fit: "EMmethod.LocalizedPetroFit",
    probe_x_coords: Sequence[float],
    probe_layer_indices: Sequence[int] | None = None,
    probe_y_coords: Sequence[float] | None = None,
    output_path: str = "mfc2_outputs/petro_calibration_localized_overlay.png",
) -> None:
    """
    Probe-level companion to ``petro_calibration.png``: draws the single
    global Waxman-Smits curve per horizon (``petro_fit`` — exactly what
    ``plot_petrophysical_calibration`` shows) together with, for every
    probe, the WS curve implied by ITS OWN nearest kernel-localized target
    (``localized_fit``) — a thin, semi-transparent line in the same
    horizon color. A tight fan hugging the global curve means localization
    barely changed anything at that probe; a fan that spreads out means
    the local fit found meaningfully different parameters nearby.

    This is deliberately different from ``plot_localized_sigma_s_profile``/
    ``plot_localized_sigma_s_map``, which show a parameter varying in
    SPACE (x, or x/y). This instead shows the effect in the SAME
    EC-vs-theta axes as the original calibration cross-plot, anchored at
    each probe's actual observation — i.e. "how much does localizing
    change the curve THIS probe's own data point sits on", rather than
    "how does sigma_s vary along the transect".

    Parameters
    ----------
    ec_values, theta_obs : np.ndarray
        Same convention already used by ``plot_petrophysical_calibration``
        for its own ``ec_values``/``theta_obs`` (i.e. exactly what you'd
        pass there — ``ec_extracted`` from ``_extract_ec_for_plots`` and
        ``best_fit.observed_theta``). ``probe_layer_indices`` must be the
        SAME array already used to mask ``ec_values``/``theta_obs`` in
        that function, since this reuses the identical masking convention
        to route each probe to its horizon and its own local target.
    petro_fit : EMmethod.PetroFit
        The global (un-localized) fit — same object passed to
        ``plot_petrophysical_calibration``.
    localized_fit : EMmethod.LocalizedPetroFit
        Typically the same-ranked ensemble member's localized fit (e.g.
        ``fits_step2_localized[0]`` alongside ``fits_step2[0]``).
    probe_x_coords : sequence of float
        Each probe's x-coordinate, in the SAME order/convention as
        ``probe_layer_indices`` above — e.g. what ``run_petro_localized``
        already computes internally as
        ``np.asarray(point_x_coords)[sample_positions]``.
    probe_layer_indices : sequence of int, optional
        Horizon index for each probe, same order as ``probe_x_coords``.
        Required to route probes to the right horizon and find each
        probe's nearest local target; if omitted, the overlay is skipped
        and only the global curves are drawn (with a printed warning),
        same fallback spirit as ``plot_petrophysical_calibration``.
    probe_y_coords : sequence of float, optional
        Each probe's y-coordinate, same order, when the localized fit
        used 2-D Euclidean localization (i.e.
        ``localized_fit.horizon_fits[i].target_y`` is not None). Ignored
        for 1-D along-transect localization.
    output_path : str
        Where to save the figure.

    Notes
    -----
    ``probe_x_coords``/``probe_y_coords``/``probe_layer_indices`` MUST be
    in the same horizon-routed order as ``ec_values``/``theta_obs`` (i.e.
    already passed through ``geometry_utils.route_array_to_horizons`` with
    the same ``probe_layer_indices``/``num_layers``, as ``run_petro_localized``
    now does before calling this). If a future caller passes them in plain
    per-observation order instead, the symptom is scatter points and local
    curves that visibly don't correspond to the same probes.
    """
    num_horizons = len(petro_fit.horizon_fits)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, num_horizons))
    fig, ax = plt.subplots(figsize=(7.5, 6))

    ec_values = np.asarray(ec_values, dtype=float)
    theta_obs = np.asarray(theta_obs, dtype=float)
    ec_line_full = np.linspace(ec_values.min() * 0.8, ec_values.max() * 1.2, 100)

    if probe_layer_indices is None:
        print(
            "[PLOT][WARN] plot_petrophysical_calibration_localized_overlay: "
            "probe_layer_indices not given -- cannot route probes to "
            "per-horizon local targets. Drawing global curves only."
        )
        probe_layer_indices_arr = None
    else:
        probe_layer_indices_arr = np.asarray(probe_layer_indices)

    probe_x_coords = np.asarray(probe_x_coords, dtype=float)
    use_2d = probe_y_coords is not None
    if use_2d:
        probe_y_coords = np.asarray(probe_y_coords, dtype=float)

    local_by_horizon = {hz.horizon: hz for hz in localized_fit.horizon_fits}

    for hz in petro_fit.horizon_fits:
        h_idx = hz.horizon
        color = colors[h_idx]

        # Global (un-localized) curve -- bold, exactly as in
        # plot_petrophysical_calibration.
        theta_global = EMmethod.waxman_smits_theta(
            ec_line_full, hz.a, hz.m, hz.sigma_w, hz.sigma_s
        )
        is_calibrated = getattr(hz, 'is_calibrated', True)
        ax.plot(
            ec_line_full, theta_global, color=color, linewidth=2.5, zorder=3,
            linestyle=('-' if is_calibrated else ':'),
            alpha=(1.0 if is_calibrated else 0.45),
            label=(
                f'Global fit — Hz {h_idx} (a={hz.a:.2f}, m={hz.m:.2f}, '
                f'\u03c3_w={hz.sigma_w:.1f}, \u03c3_s={hz.sigma_s:.1f})'
                if is_calibrated else
                f'Global fit — Hz {h_idx} (UNCALIBRATED, prior only)'
            ),
        )

        if probe_layer_indices_arr is None:
            continue

        mask = probe_layer_indices_arr == h_idx
        if not np.any(mask):
            continue

        # Observed data points for this horizon.
        ax.scatter(
            ec_values[mask], theta_obs[mask], color=color, edgecolor='black',
            s=70, zorder=4, label=f'Observed data — Hz {h_idx}',
        )

        loc_hz = local_by_horizon.get(h_idx)
        if loc_hz is None:
            continue

        first_local_label_used = False
        for p_i in np.where(mask)[0]:
            px = probe_x_coords[p_i]
            if use_2d and loc_hz.target_y is not None:
                py = probe_y_coords[p_i]
                dist = np.sqrt((loc_hz.target_x - px) ** 2 + (loc_hz.target_y - py) ** 2)
            else:
                dist = np.abs(loc_hz.target_x - px)
            nearest = int(np.argmin(dist))

            if not loc_hz.is_calibrated[nearest]:
                continue  # falls back to the prior -- skip to avoid clutter

            theta_local = EMmethod.waxman_smits_theta(
                ec_line_full, loc_hz.a[nearest], loc_hz.m[nearest],
                loc_hz.sigma_w[nearest], loc_hz.sigma_s[nearest],
            )
            ax.plot(
                ec_line_full, theta_local, color=color, linewidth=0.8,
                linestyle='--', alpha=0.5, zorder=2,
                label=(f'Local per-probe fits — Hz {h_idx}'
                       if not first_local_label_used else None),
            )
            first_local_label_used = True

    ax.set_title(
        "Step 2 (localized overlay): global fit vs. each probe's own "
        "kernel-localized fit", fontsize=11, pad=12,
    )
    ax.set_xlabel("Inverted Bulk Electrical Conductivity (EC) [mS/m]", fontsize=10)
    ax.set_ylabel("Volumetric Water Content ($\\theta$) [m$^3$/m$^3$]", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', frameon=True, fontsize=7, bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Localized overlay calibration graph exported to '{output_path}'")


def run_petro_localized(
    models: np.ndarray,
    sample_positions: Sequence[int],
    sample_times: Sequence[int],
    theta_obs: np.ndarray,
    point_x_coords: Sequence[float],
    bandwidth: float,
    point_y_coords: Sequence[float] | None = None,
    depths0: list[float] | None = None,
    output_dir: str = "mfc2_outputs",
    generate_plots: bool = True,
    probe_layer_indices: Sequence[int] | None = None,
    target_indices: Sequence[int] | None = None,
    fixed_params: dict[str, bool] | None = None,
    lab_p0: np.ndarray | list | None = None,
    theta_obs_std: float | Sequence[float] | None = None,
    global_fit: "EMmethod.PetroFit | None" = None,
    sensor_ids: Sequence[str] | None = None,
) -> list["EMmethod.LocalizedPetroFit"]:
    """
    Step 2, kernel-localized variant (spatial_calibration_alternatives.md
    §2) — sibling to ``run_petro``, not a replacement. Where ``run_petro``
    fits one scalar [a, m, sigma_w, sigma_s] per horizon and broadcasts it
    to every EM point, this fits a SEPARATE parameter set at every target
    location, weighting every TDR observation in that horizon by a
    Gaspari-Cohn kernel of its distance to the target instead of a hard
    per-position bucket. Horizon routing (probe_layer_indices / the
    strided heuristic) is unchanged — only the spatial pooling WITHIN a
    horizon differs from ``run_petro``.

    Call this instead of ``run_petro`` when you want sigma_s (etc.) to vary
    continuously along the transect rather than being flat per horizon —
    in particular for the sparse-TDR-coverage case where most EM points
    have no local probe at all and currently just inherit the pure global
    value with no spatial information from nearby probes.

    Parameters
    ----------
    models, sample_positions, sample_times, theta_obs, depths0, output_dir,
    generate_plots, probe_layer_indices, fixed_params, lab_p0, theta_obs_std
        Same meaning as in ``run_petro``.
    point_x_coords : sequence of float, length num_measured_points
        Along-transect x-coordinate of every EM measured point (the same
        axis ``sample_positions`` indexes into). Forwarded to
        ``EMmethod.calibrate_petrophysical_relationship_localized``.
    point_y_coords : sequence of float, length num_measured_points, optional
        Cross-transect y-coordinate of every EM measured point. When given,
        localization uses 2-D Euclidean distance (x, y) instead of the
        signed 1-D x-difference — use this for genuine 2-D survey grids
        (``mode == "2d"``) so probes that are close in y but far in x (or
        vice versa) aren't ignored. ``None`` (default) keeps the original
        1-D-transect behaviour. Forwarded unchanged to
        ``EMmethod.calibrate_petrophysical_relationship_localized``.
    bandwidth : float
        Localization radius ``c`` (same units as ``point_x_coords``). No
        default — pick empirically (sweep a few candidates, check
        validation RMSE), the same way you'd have picked a fixed-shrinkage
        constant.
    target_indices : sequence of int, optional
        Which EM points to produce a localized fit for. Defaults to every
        point in ``point_x_coords``.
    global_fit : EMmethod.PetroFit, optional
        The un-localized fit for the SAME ensemble member (typically
        ``fits_step2[0]``, i.e. the rank-1 member also used as
        ``best_fit`` here). When given, additionally generates
        ``petro_calibration_localized_overlay.png`` via
        ``plot_petrophysical_calibration_localized_overlay`` — each
        probe's own local WS curve overlaid on the single global curve, in
        the same EC-vs-theta axes as ``petro_calibration.png``. ``None``
        (default) skips this — unchanged behaviour from before this
        parameter existed.

    Returns
    -------
    list of EMmethod.LocalizedPetroFit, length ensemble_size

    Notes
    -----
    The existing per-horizon diagnostic plots (``plot_petrophysical_calibration``,
    ``plot_dual_depth_profile``, ``plot_one_to_one_validation``,
    ``plot_ws_parameter_histograms``) all assume one scalar parameter set
    per horizon and are NOT called here — they don't generalize cleanly to
    a per-target array. Instead this generates
    ``plot_localized_sigma_s_profile`` (now covering whichever of
    sigma_w/sigma_s are free per ``fixed_params``, not sigma_s alone), and,
    when ``global_fit`` is passed, ``plot_petrophysical_calibration_localized_overlay``
    — the probe-anchored companion to ``petro_calibration.png``. If you also
    want the flat-value diagnostics for comparison, run ``run_petro``
    alongside this on the same data and view both sets of plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    timelapse = models.ndim == 4
    if timelapse:
        n_time = models.shape[1]
        print(f"[STEP 2 LOCALIZED] Time-lapse mode: {n_time} survey epochs detected.")
    else:
        print("[STEP 2 LOCALIZED] Single-time mode.")

    if point_y_coords is not None:
        print(f"[STEP 2 LOCALIZED] Using bandwidth c={bandwidth:g}, "
              f"2-D Euclidean (x, y) distance.")
    else:
        print(f"[STEP 2 LOCALIZED] Using bandwidth c={bandwidth:g}, "
              f"1-D along-transect (x) distance.")

    fits = EMmethod.calibrate_petrophysical_relationship_localized(
        models=models,
        sample_positions=list(sample_positions),
        sample_times=list(sample_times),
        theta_obs=theta_obs,
        point_x_coords=point_x_coords,
        point_y_coords=point_y_coords,
        bandwidth=bandwidth,
        target_indices=target_indices,
        depths0=depths0,
        probe_layer_indices=probe_layer_indices,
        fixed_params=fixed_params,
        lab_p0=lab_p0,
        theta_obs_std=theta_obs_std,
    )
    print(f"[STEP 2 LOCALIZED] Completed localized parameter extraction for "
          f"{len(fits)} ensemble members.")

    best_fit = fits[0]
    print("[STEP 2 LOCALIZED] Best-fit sigma_s(x) range per horizon "
          "(over calibrated targets only):")
    for hz in best_fit.horizon_fits:
        if np.any(hz.is_calibrated):
            s_cal = hz.sigma_s[hz.is_calibrated]
            print(
                f"  Horizon {hz.horizon}: sigma_s in "
                f"[{s_cal.min():.2f}, {s_cal.max():.2f}] mS/m "
                f"across {hz.is_calibrated.sum()}/{len(hz.is_calibrated)} "
                f"calibrated target(s), mean RMSE="
                f"{np.nanmean(hz.rmse):.4f}"
            )
        else:
            print(f"  Horizon {hz.horizon}: no target within kernel support "
                  f"of any observation — every point fell back to the prior.")

    if generate_plots:
        print("[STEP 2 LOCALIZED] Generating diagnostic plot...")

        probe_x_coords = np.asarray(point_x_coords)[np.asarray(list(sample_positions))]
        probe_y_coords = (
            np.asarray(point_y_coords)[np.asarray(list(sample_positions))]
            if point_y_coords is not None else None
        )

        # Which conductivity params to show on the sigma_s(x) profile: both
        # sigma_w and sigma_s when both are free (the case that actually
        # motivated adding this -- their non-identifiability trade-off is
        # invisible if only sigma_s is plotted), otherwise whichever single
        # one is free, falling back to sigma_s alone if both happen to be
        # fixed (matches the original default so the plot still renders
        # something rather than nothing).
        _fixed = fixed_params or {}
        _free_conductivity_params = tuple(
            p for p in ("sigma_w", "sigma_s") if not _fixed.get(p, False)
        )
        if not _free_conductivity_params:
            _free_conductivity_params = ("sigma_s",)

        plot_localized_sigma_s_profile(
            localized_fit=best_fit,
            probe_x_coords=probe_x_coords,
            probe_layer_indices=probe_layer_indices,
            params=_free_conductivity_params,
            output_path=os.path.join(output_dir, "petro_calibration_localized.png"),
        )

        if point_y_coords is not None:
            # Mirror the profile plot's parameter selection above: render one
            # map per free conductivity parameter, so sigma_w isn't silently
            # dropped when both sigma_w and sigma_s are free (see comment on
            # _free_conductivity_params above -- same non-identifiability
            # rationale applies here). plot_localized_sigma_s_map only takes
            # a single `param` per call (unlike the profile function's
            # `params` list), so loop and suffix the filename per parameter.
            # When exactly one param is free, keep the original unsuffixed
            # filename so existing downstream references don't break.
            for _p in _free_conductivity_params:
                if len(_free_conductivity_params) == 1:
                    _map_filename = "petro_calibration_localized_map.png"
                else:
                    _map_filename = f"petro_calibration_localized_map_{_p}.png"
                plot_localized_sigma_s_map(
                    localized_fit=best_fit,
                    probe_x_coords=probe_x_coords,
                    probe_y_coords=probe_y_coords,
                    probe_layer_indices=probe_layer_indices,
                    param=_p,
                    output_path=os.path.join(output_dir, _map_filename),
                )

        if global_fit is not None:
            ec_extracted, _, kept_indices = _extract_ec_for_plots(
                models=models,
                sample_positions=sample_positions,
                sample_times=sample_times,
                probe_layer_indices=probe_layer_indices,
            )
            # BUGFIX: ec_extracted/global_fit.observed_theta are routed to
            # HORIZON order AND have any non-finite-EC observation dropped
            # by _extract_ec_for_plots/route_ec_to_horizons (real-data
            # time-lapse reference-grid gaps -- see
            # geometry_utils.route_ec_to_horizons), i.e. concatenated
            # per-horizon, NaN-dropped, rather than left in original
            # per-observation order/count. probe_x_coords/probe_y_coords/
            # probe_layer_indices above, however, are still in ORIGINAL
            # per-observation order/count (built directly from
            # sample_positions/probe_layer_indices, never routed). Passing
            # them through as-is made
            # plot_petrophysical_calibration_localized_overlay's internal
            # `mask = probe_layer_indices == h_idx` select the wrong
            # elements (or, once any NaN-drop occurred, a mask of the wrong
            # LENGTH entirely -- a hard IndexError) out of the routed
            # ec_values/theta_obs arrays. Gather all three with
            # kept_indices -- NOT geometry_utils.route_array_to_horizons,
            # which only replicates the routing, not the NaN-drop -- so
            # every array passed in below shares one consistent,
            # length-matched order with ec_extracted/global_fit.observed_theta.
            probe_x_coords_routed = np.asarray(probe_x_coords)[kept_indices]
            probe_y_coords_routed = (
                np.asarray(probe_y_coords)[kept_indices]
                if probe_y_coords is not None else None
            )
            probe_layer_indices_routed = (
                np.asarray(probe_layer_indices)[kept_indices]
                if probe_layer_indices is not None else None
            )
            plot_petrophysical_calibration_localized_overlay(
                ec_values=ec_extracted,
                theta_obs=global_fit.observed_theta,
                petro_fit=global_fit,
                localized_fit=best_fit,
                probe_x_coords=probe_x_coords_routed,
                probe_y_coords=probe_y_coords_routed,
                probe_layer_indices=probe_layer_indices_routed,
                output_path=os.path.join(output_dir, "petro_calibration_localized_overlay.png"),
            )
        else:
            print(
                "[STEP 2 LOCALIZED] global_fit not passed -- skipping "
                "petro_calibration_localized_overlay.png (pass fits_step2[0] "
                "as global_fit to enable it)."
            )

    return fits


# ==========================================================================
# BANDWIDTH SELECTION — leave-one-probe-out CV sweep
# ==========================================================================

def plot_bandwidth_sweep(
    sweep_results: list["EMmethod.BandwidthSweepResult"],
    output_path: str = "mfc2_outputs/bandwidth_sweep.png",
) -> None:
    """
    Plots pooled LOO-CV RMSE vs. candidate bandwidth, with the recommended
    (minimum-RMSE) bandwidth marked. A secondary panel shows the number of
    folds that fell back to the prior per bandwidth (large at small
    bandwidths is expected -- see ``EMmethod.sweep_localization_bandwidth``).
    """
    bw = np.array([r.bandwidth for r in sweep_results])
    rmse = np.array([r.loo_rmse for r in sweep_results])
    n_fallback = np.array([r.n_fallback for r in sweep_results])
    n_folds = np.array([r.n_folds for r in sweep_results])

    valid = ~np.isnan(rmse)
    best_idx = int(np.argmin(np.where(valid, rmse, np.inf))) if np.any(valid) else 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)

    ax1.plot(bw, rmse, marker='o', color='tab:blue', linewidth=1.8)
    ax1.axvline(bw[best_idx], color='tab:red', linestyle='--', alpha=0.7,
                label=f'Recommended c={bw[best_idx]:g} '
                      f'(LOO-CV RMSE={rmse[best_idx]:.5f})')
    ax1.set_ylabel('Pooled LOO-CV RMSE [m³/m³]', fontsize=10)
    ax1.set_title('Localization bandwidth sweep — leave-one-probe-out validation',
                   fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='best', fontsize=9)

    ax2.bar(bw, n_fallback, width=(bw.max() - bw.min()) * 0.04 if len(bw) > 1 else 1.0,
            color='tab:orange', alpha=0.7, label='Folds falling back to prior')
    ax2.set_xlabel('Bandwidth c', fontsize=10)
    ax2.set_ylabel('# LOO folds\nfalling back to prior', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='best', fontsize=8)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Bandwidth sweep exported to '{output_path}'")


def run_bandwidth_sweep(
    sigma_b_reference: np.ndarray,
    sample_positions: Sequence[int],
    sample_times: Sequence[int],
    theta_obs: np.ndarray,
    point_x_coords: Sequence[float],
    bandwidths: Sequence[float],
    point_y_coords: Sequence[float] | None = None,
    output_dir: str = "mfc2_outputs",
    generate_plots: bool = True,
    probe_layer_indices: Sequence[int] | None = None,
    lab_p0: np.ndarray | list | None = None,
    fixed_params: dict[str, bool] | None = None,
    theta_obs_std: float | Sequence[float] | None = None,
) -> tuple[list["EMmethod.BandwidthSweepResult"], float]:
    """
    Driver for ``EMmethod.sweep_localization_bandwidth`` — runs the
    leave-one-probe-out CV sweep across ``bandwidths``, prints a per-
    candidate summary, plots the sweep curve, and returns the recommended
    bandwidth (minimum pooled LOO-CV RMSE).

    Run this ONCE up front (against a single reference model, e.g. Step
    2's rank-1 ensemble member) to pick ``ws_localized_bandwidth`` for the
    scenario config — not as part of the per-run pipeline, since it's a
    one-off tuning decision rather than something that needs to be redone
    every pipeline run.

    Parameters
    ----------
    sigma_b_reference, sample_positions, sample_times, theta_obs,
    point_x_coords, point_y_coords, probe_layer_indices, lab_p0,
    fixed_params, theta_obs_std
        Same meaning as in ``EMmethod.sweep_localization_bandwidth`` /
        ``run_petro_localized``.
    bandwidths : sequence of float
        Candidate values to sweep, e.g. ``np.linspace(1.0, 20.0, 10)``.

    Returns
    -------
    sweep_results : list of EMmethod.BandwidthSweepResult
    recommended_bandwidth : float
        The bandwidth with the lowest pooled LOO-CV RMSE among candidates
        that produced at least one valid fold.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[BANDWIDTH SWEEP] Evaluating {len(list(bandwidths))} candidate "
          f"bandwidth(s) via leave-one-probe-out CV"
          f"{' (2-D Euclidean)' if point_y_coords is not None else ' (1-D x)'}.")

    sweep_results = EMmethod.sweep_localization_bandwidth(
        sigma_b_reference=sigma_b_reference,
        sample_positions=sample_positions,
        sample_times=sample_times,
        theta_obs=theta_obs,
        point_x_coords=point_x_coords,
        point_y_coords=point_y_coords,
        bandwidths=bandwidths,
        probe_layer_indices=probe_layer_indices,
        lab_p0=lab_p0,
        fixed_params=fixed_params,
        theta_obs_std=theta_obs_std,
    )

    for r in sweep_results:
        print(
            f"  c={r.bandwidth:8.3f}  LOO-CV RMSE={r.loo_rmse:.5f} m³/m³  "
            f"({r.n_folds} valid folds, {r.n_fallback} fell back to prior)"
        )

    valid = [r for r in sweep_results if not np.isnan(r.loo_rmse)]
    if not valid:
        raise RuntimeError(
            "[BANDWIDTH SWEEP] Every candidate bandwidth produced zero valid "
            "LOO folds -- check probe_layer_indices / horizon routing."
        )
    best = min(valid, key=lambda r: r.loo_rmse)
    print(f"[BANDWIDTH SWEEP] Recommended bandwidth: c={best.bandwidth:g} "
          f"(pooled LOO-CV RMSE={best.loo_rmse:.5f} m³/m³)")

    if generate_plots:
        plot_bandwidth_sweep(
            sweep_results,
            output_path=os.path.join(output_dir, "bandwidth_sweep.png"),
        )

    return sweep_results, best.bandwidth