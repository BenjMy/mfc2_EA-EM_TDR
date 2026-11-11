#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 3 standalone runner — inverted model averaging and diagnostics.
"""

from __future__ import annotations

import os
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt

import em_methodology as EMmethod


# ==========================================================================
# DIAGNOSTIC PLOTTING FUNCTIONS FOR STEP 3 AVERAGING
# ==========================================================================

def plot_step3_rmse_evolution(fits_step2: Sequence[EMmethod.PetroFit], output_dir: str) -> None:
    """Renders an optimization plot sorting all individual model calibration fits."""
    rmse_values = sorted([f.rmse for f in fits_step2])
    
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(range(1, len(rmse_values) + 1), rmse_values, marker='o', linestyle='-', 
            color='tab:purple', alpha=0.8, label="Ensemble Member Fits")
    
    ax.set_title("Step 3 Input Vector: Sorted Base Inversion Quality Metrics", fontsize=11, pad=12)
    ax.set_xlabel("Sorted Model Index (Ascending Error Linearity)", fontsize=10)
    ax.set_ylabel("Linear Alignment Regression Residuals [RMSE]", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc="upper left")
    
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "step3_rmse_distribution.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[PLOT] Step 3 optimization error curve exported to '{plot_path}'")

def plot_step3_averaged_profiles(spatial_profiles, depths0, output_path):
    """
    Definitive fix for list of arrays matching:
    [array(3, 10, 4), array(3, 10, 4), ...] -> Ensemble of 4 models.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # 1. Convertir la lista de arrays en un único bloque 4D de NumPy
    # Shape resultante: (4, 3, 10, 4) -> (modelos, tiempos, puntos, capas)
    profiles = np.stack(spatial_profiles, axis=0)
    depths_arr = np.asarray(depths0)
    
    # 2. Calcular los centros de las 4 capas usando depths0
    layer_centers = (depths_arr[:-1] + depths_arr[1:]) / 2.0
    num_layers = len(layer_centers)  # Debería ser 4 si depths0 tiene 5 límites

    # 3. PROMEDIAR EL ENSAMBLE (Eje 0 - los 4 modelos)
    # Shape resultante: (3, 10, 4) -> (tiempos, puntos, capas)
    ensemble_mean = np.mean(profiles, axis=0)
    
    n_time = ensemble_mean.shape[0]        # 3 instantes de tiempo
    num_points = ensemble_mean.shape[1]    # 10 puntos de medición
    num_layers_data = ensemble_mean.shape[2] # 4 horizontes reales de datos

    # 4. Sincronización final de seguridad para el eje Y
    if num_layers != num_layers_data:
        # Si hay discrepancia entre depths0 y los datos, generamos un eje lineal basado en los datos
        y_plot = np.linspace(depths_arr[0], depths_arr[-1], num_layers_data)
    else:
        y_plot = layer_centers

    # 5. Configurar subplots (1 panel por cada instante de tiempo)
    #    Reserve a slim right-hand strip for a shared colorbar instead of a
    #    per-point legend: with num_points in the hundreds (e.g. a dense
    #    real-data EM transect), one "Point N" legend entry per point used
    #    to blow up to a column wider than the plot itself. Point identity
    #    is encoded as color -> colorbar instead, with the index axis
    #    spelled out (point index along transect), so it stays readable at
    #    any num_points.
    fig, axes = plt.subplots(
        1, n_time, figsize=(4 * n_time + 1.1, 6), sharey=True,
        gridspec_kw={'right': 0.88},
    )
    if n_time == 1:
        axes = [axes]

    # Paleta de colores para diferenciar los puntos del transecto
    colors = plt.cm.plasma(np.linspace(0, 0.85, num_points))

    for t_idx in range(n_time):
        ax = axes[t_idx]
        
        # Extraer datos de este tiempo. Shape: (num_points, num_layers)
        time_data = ensemble_mean[t_idx, :, :]
        
        # A. Graficar el perfil vertical para cada uno de los puntos
        #    (color-coded by index; no per-point legend label)
        for p_idx in range(num_points):
            x_plot = time_data[p_idx, :]
            
            ax.plot(
                x_plot, 
                y_plot, 
                color=colors[p_idx], 
                linestyle='-', 
                marker='o', 
                markersize=3,
                alpha=0.5,
                linewidth=0.9,
            )

        # B. Graficar la media espacial de todo el lote para este tiempo específico
        spatial_mean = np.mean(time_data, axis=0)
        ax.plot(
            spatial_mean, 
            y_plot, 
            color='black', 
            linestyle='--', 
            linewidth=2.5, 
            zorder=5,
            label='Spatial Mean'
        )

        # Detalles estéticos por panel
        ax.set_xlabel('Bulk EC [mS/m]', fontsize=10)
        ax.set_title(f'Time Step {t_idx + 1}', fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        # Tiny legend with only the Spatial Mean line — individual points
        # are explained by the shared colorbar instead, so this never grows
        # with num_points.
        ax.legend(loc='lower left', fontsize=7, frameon=True)

        if t_idx == 0:
            ax.set_ylabel('Depth [m]', fontsize=11, fontweight='bold')
            ax.invert_yaxis()  # Invertimos el eje vertical solo en el primer panel

    # Single shared colorbar (point index along transect) instead of a
    # per-point legend entry. mappable carries no real data array; norm/cmap
    # alone is enough to draw the colorbar.
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.plasma,
        norm=plt.Normalize(vmin=1, vmax=max(num_points, 1)),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label('Point index (transect order)', fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    if os.path.isdir(output_path):
        actual_output = os.path.join(output_path, "step3_averaged_time_profiles.png")
    else:
        actual_output = output_path

    plt.savefig(actual_output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Step 3 temporal/spatial profiles successfully exported to '{actual_output}'")
    
    
# ==========================================================================
# DRIVER EXECUTION INTERFACE
# ==========================================================================

def run_average(
    models: np.ndarray,
    fits_step2: Sequence[EMmethod.PetroFit],
    depths0: list[float] | None = None,
    output_dir: str = "mfc2_outputs",
    generate_plots: bool = True
) -> tuple[list[np.ndarray], np.ndarray]:
    """Run STEP 3: rank and cumulatively average the ensemble models."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Compute rankings and cumulative matrix averages
    averaged_models, ranking = EMmethod.average_inverted_models(
        models=models,
        fits=fits_step2,
    )
    print(f"[STEP 3] Averaging complete. N={len(models)} models ranked. "
          f"Rank-1 model index: {ranking[0]}, rank-N model index: {ranking[-1]}.")

    # 2. Render diagnostic plots
    if generate_plots:
        print("[STEP 3] Initiating diagnostic graphics generation pipeline...")
        plot_step3_rmse_evolution(fits_step2, output_dir)
        
        if depths0 is not None:
            plot_step3_averaged_profiles(averaged_models, depths0, output_dir)
        else:
            print("[WARN] Skipping staircase profile plot: 'depths0' boundaries missing.")

    return averaged_models, ranking