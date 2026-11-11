#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train/validation diagnostic plots (Loiseau et al. 2026, Secs. 2.3 & 2.6).

Called from run_pipeline.py's Step 5b, once a --train-val-split method
other than 'off' is active. Everything here is a forward-only, no-refit
evaluation -- these plots exist to show whether the K chosen using ONLY
`split.train` (Steps 2-5) also generalises to `split.val`, which Steps
2-5 never see. Written to ``<OUTPUT_DIR>/train_val/``.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_rmse_vs_k(
    train_rmse_by_k: np.ndarray,
    val_rmse_by_k: np.ndarray,
    selected_K: int,
    output_dir: str,
    fname: str = "rmse_vs_K.png",
) -> str:
    """
    Train vs. validation RMSE across every candidate ensemble stack depth K.

    The classic model-selection diagnostic: train RMSE should fall (or
    plateau) roughly monotonically as K grows, since Step 4 keeps
    refitting on more averaged data; validation RMSE bottoming out and
    then climbing again is the signature of Step 5's K selection starting
    to overfit the training subset. ``selected_K`` -- picked as the
    argmin of the TRAIN curve only, before this val curve was ever
    computed -- is marked on both curves so a diverging val curve at that
    exact K is visually obvious.

    Parameters
    ----------
    train_rmse_by_k, val_rmse_by_k : np.ndarray, shape (n_K,)
        RMSE at each stack depth K = 1..n_K, index-aligned with each
        other and with ``averaged_models`` / ``fits_step4``.
    selected_K : int
        1-based K chosen by Step 5.
    output_dir : str
        Directory to save into (typically ``<OUTPUT_DIR>/train_val``).

    Returns
    -------
    str
        Path the figure was saved to.
    """
    os.makedirs(output_dir, exist_ok=True)

    train_rmse_by_k = np.asarray(train_rmse_by_k, dtype=float)
    val_rmse_by_k = np.asarray(val_rmse_by_k, dtype=float)
    n_K = len(train_rmse_by_k)
    x = np.arange(1, n_K + 1)

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)

    ax.plot(
        x, train_rmse_by_k, color="tab:blue", linestyle="-", marker="o",
        markersize=4, linewidth=1.4, label="Train RMSE",
    )
    ax.plot(
        x, val_rmse_by_k, color="tab:orange", linestyle="-", marker="s",
        markersize=4, linewidth=1.4, label="Validation RMSE",
    )

    if 1 <= selected_K <= n_K:
        ax.axvline(selected_K, color="gray", linestyle=":", linewidth=1.2, zorder=1)
        if np.isfinite(train_rmse_by_k[selected_K - 1]):
            ax.plot(
                selected_K, train_rmse_by_k[selected_K - 1], color="tab:blue",
                marker="o", markersize=11, fillstyle="none", markeredgewidth=1.8, zorder=4,
            )
        if np.isfinite(val_rmse_by_k[selected_K - 1]):
            ax.plot(
                selected_K, val_rmse_by_k[selected_K - 1], color="tab:orange",
                marker="s", markersize=11, fillstyle="none", markeredgewidth=1.8, zorder=4,
            )
        ax.text(
            selected_K, ax.get_ylim()[1], f" selected K={selected_K}",
            fontsize=8.5, color="dimgray", va="top", ha="left",
        )

    ax.set_title(
        "Train vs. Validation RMSE Across Ensemble Stack Depth K",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Ensemble stack depth K", fontsize=10)
    ax.set_ylabel(r"RMSE [m$^3$/m$^3$]", fontsize=10)
    if n_K > 1:
        ax.set_xticks(x if n_K <= 20 else np.linspace(1, n_K, 10, dtype=int))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="none")

    fig.tight_layout()
    fig_path = os.path.join(output_dir, fname)
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"[PLOT] Train/val RMSE-vs-K curve -> {fig_path}")
    return fig_path


def _r2_p_text(obs: np.ndarray, pred: np.ndarray) -> tuple[str, str]:
    """
    Guarded Pearson R^2 / p-value text, matching
    ``step2_petro.plot_one_to_one_validation``'s own fallback behaviour
    (needs >=2 non-degenerate pairs and non-constant input; "n/a"
    otherwise rather than raising).
    """
    r2_text, p_text = "R$^2$ = n/a", "p = n/a"
    try:
        from scipy.stats import pearsonr
        if len(obs) >= 2 and np.std(obs) > 0 and np.std(pred) > 0:
            r_val, p_val = pearsonr(obs, pred)
            r2_text = f"R$^2$ = {r_val**2:.3f}"
            p_text = f"p = {p_val:.3g}"
    except Exception as e:
        print(f"[WARN] Could not compute R2/p-value for train/val parity plot: {e}")
    return r2_text, p_text


def plot_parity_train_val(
    train_obs: np.ndarray,
    train_pred: np.ndarray,
    val_obs: np.ndarray,
    val_pred: np.ndarray,
    output_dir: str,
    fname: str = "parity_train_vs_val.png",
) -> str:
    """
    Side-by-side observed-vs-predicted parity plots for the train and
    validation subsets, at the selected K, sharing one x/y range so the
    two panels are directly comparable at a glance.

    Mirrors ``step2_petro.plot_one_to_one_validation`` (same 1:1 line,
    same R^2/p annotation style), but split into two panels rather than
    one overlay: the validation subset is usually much smaller than
    train, and overlaying the two tends to bury the (more diagnostically
    important) validation points under a denser train point cloud.

    Parameters
    ----------
    train_obs, train_pred : np.ndarray
        Observed/predicted theta for the TRAIN subset, at the selected K
        (e.g. ``fits_step4[selected_output.K - 1].observed_theta`` /
        ``.predicted_theta``).
    val_obs, val_pred : np.ndarray
        Observed/predicted theta for the VALIDATION subset, evaluated
        with the train-fitted parameters pinned (e.g.
        ``val_fit.observed_theta`` / ``val_fit.predicted_theta``).
    output_dir : str
        Directory to save into (typically ``<OUTPUT_DIR>/train_val``).

    Returns
    -------
    str
        Path the figure was saved to.
    """
    os.makedirs(output_dir, exist_ok=True)

    train_obs = np.asarray(train_obs, dtype=float)
    train_pred = np.asarray(train_pred, dtype=float)
    val_obs = np.asarray(val_obs, dtype=float)
    val_pred = np.asarray(val_pred, dtype=float)

    all_vals = np.concatenate([train_obs, train_pred, val_obs, val_pred])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size:
        min_val, max_val = float(all_vals.min()) * 0.95, float(all_vals.max()) * 1.05
    else:
        min_val, max_val = 0.0, 1.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150, sharex=True, sharey=True)

    panels = [
        (axes[0], train_obs, train_pred, "Train", "tab:blue"),
        (axes[1], val_obs, val_pred, "Validation", "tab:orange"),
    ]
    for ax, obs, pred, label, color in panels:
        finite = np.isfinite(obs) & np.isfinite(pred)
        obs_f, pred_f = obs[finite], pred[finite]

        ax.scatter(obs_f, pred_f, color=color, edgecolor="black", s=70, zorder=3, alpha=0.85)
        ax.plot(
            [min_val, max_val], [min_val, max_val], color="red", linestyle="--",
            linewidth=1.5, label="1:1 Parity Line",
        )
        r2_text, p_text = _r2_p_text(obs_f, pred_f)
        ax.text(
            0.03, 0.97, f"{r2_text}\n{p_text}  (n={len(obs_f)})", transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="gray"),
        )
        ax.set_title(f"{label} Subset", fontsize=11)
        ax.set_xlabel("Observed TDR Soil Moisture [m$^3$/m$^3$]", fontsize=10)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=8)

    axes[0].set_ylabel("Predicted Model Translation [m$^3$/m$^3$]", fontsize=10)
    fig.suptitle(
        "Train vs. Validation Parity (Held-Out Check, Sec. 2.6)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig_path = os.path.join(output_dir, fname)
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"[PLOT] Train/val parity comparison -> {fig_path}")
    return fig_path
