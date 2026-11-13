"""Ground-truth TDR depth profiles (synthetic runs, Step 0).

plot_truth_tdr(output_dir, theta_true_realization, depths0, fname) -> str
    theta (n_time, n_pos, n_layer) or (n_pos, n_layer), before any petrophysical
    / EM forward transform. Returns the saved figure path.

Called by run_pipeline.main (Step 0) and run_and_score.
"""
from __future__ import annotations

import os

import numpy as np

import geometry_utils


def plot_truth_tdr(output_dir: str, theta_true_realization, depths0,
                    fname: str = "truth_tdr_profiles.png"):
    """
    Plot the ground-truth TDR (volumetric water content, theta) depth
    profiles for the exported realization, BEFORE any pedophysical/EM
    forward transformation. This is the theta that gets pedophysically
    converted to EC_bulk and then forward-modeled into the exported ECa
    surveys, so it's the reference point for sanity-checking that chain.

    Accepts theta_true_realization shaped (n_time, n_pos, n_layer) or
    (n_pos, n_layer) for a single-survey (non-time-lapse) run.

    Ported from run_pipeline_short.py (the synthetic-only minimal runner)
    so the full pipeline gets the same truth-check plot.
    """
    import matplotlib.pyplot as plt

    theta = np.asarray(theta_true_realization, dtype=float)
    if theta.ndim == 2:
        theta = theta[np.newaxis, ...]  # -> (1, n_pos, n_layer)
    if theta.ndim != 3:
        raise ValueError(
            f"[plot_truth_tdr] Unexpected theta_true_realization shape "
            f"{theta.shape}; expected (n_time, n_pos, n_layer) or (n_pos, n_layer)."
        )
    n_time, n_pos, n_layer = theta.shape

    # 1. Get centers for the explicitly defined bounded layers
    z_centers = geometry_utils.layer_centers(depths0)

    # 2. If there is an infinite half-space (n_layer > len(depths0)), append a representative depth
    if n_layer > len(z_centers):
        last_bottom = depths0[-1] if len(depths0) > 0 else 0.0
        # Extrapolate the final layer's plotting depth (e.g., 0.4m below the last boundary)
        half_space_center = last_bottom + 0.4
        z = np.concatenate((z_centers, [half_space_center]))[:n_layer]
    else:
        z = z_centers[:n_layer]

    fig, axes = plt.subplots(1, n_time, figsize=(4.5 * n_time, 5),
                              squeeze=False, sharey=True)
    axes = axes[0]

    for t in range(n_time):
        ax = axes[t]
        theta_t = theta[t]  # (n_pos, n_layer)
        for p in range(n_pos):
            ax.plot(theta_t[p], z, "o-", lw=1, alpha=0.30, color="tab:blue")
        ax.plot(theta_t.mean(axis=0), z, "o-", lw=2.5, color="tab:red",
                 label="Mean across positions")
        ax.invert_yaxis()
        ax.set_xlabel(r"$\theta$ (TDR truth) [m$^3$/m$^3$]")
        ax.set_title(f"t = {t}" if n_time > 1 else "TDR ground truth")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Depth [m]")

    fig.suptitle("Synthetic TDR Ground Truth — Volumetric Water Content Profiles",
                  fontsize=12)
    fig.tight_layout()
    fig_path = os.path.join(output_dir, fname)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] TDR ground-truth profiles     -> {fig_path}")
    return fig_path
