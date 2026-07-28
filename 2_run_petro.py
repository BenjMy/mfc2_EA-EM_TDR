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
                    f'(a={hz.a:.2f}, m={hz.m:.2f}, '
                    f'σ_w={hz.sigma_w:.1f}, σ_s={hz.sigma_s:.1f}, '
                    f'RMSE={hz.rmse:.4f}, R²={r2_str}, p={p_str})'
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
                    f'Hz {hz.horizon} — UNCALIBRATED (prior only, no TDR probe): '
                    f'a={hz.a:.2f}, m={hz.m:.2f}, '
                    f'σ_w={hz.sigma_w:.1f}, σ_s={hz.sigma_s:.1f}'
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

# ==========================================================================
# STANDALONE EXECUTION ENGINE
# ==========================================================================

def _extract_ec_for_plots(
    models: np.ndarray,
    sample_positions: Sequence[int],
    sample_times: Sequence[int],
    probe_layer_indices: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
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

    # Horizon routing now delegates to the single shared implementation in
    # geometry_utils.py (pipeline_improvement_report.md, section 6 / action
    # item 3) instead of an independently reimplemented copy of the same
    # strided/modulo/explicit-index logic used by
    # em_methodology.calibrate_petrophysical_relationship and by
    # 4_run_select_assess.py::_extract_ec_at_probes.
    ec_extracted, _horizon_slices = geometry_utils.route_ec_to_horizons(
        ec_array=best_member,
        sample_positions=sample_positions,
        sample_times=sample_times,
        probe_layer_indices=probe_layer_indices,
        num_layers=num_layers,
    )
    return ec_extracted, spatial_profile



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
        ec_extracted, spatial_profile = _extract_ec_for_plots(
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

    return fits