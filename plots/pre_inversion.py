"""
plots/pre_inversion.py
======================
Diagnostic figures drawn from the raw / filtered EM survey and from the Step 1
inversion ensemble. Moved out of ``em_methodology`` (refactoring Phase 5) so the
core library no longer imports matplotlib; function bodies are unchanged.

Functions
---------
plot_step1_ensemble_single_file(...)  ensemble of inverted EC profiles, one survey file
plot_step1_ensemble_by_point(...)     same, one panel per measured point
plot_raw_apparent_conductivity(...)   raw ECa per coil (before filtering)
plot_filtered_apparent_conductivity(...)  ECa after coil dropping / filtering

Called from run_pipeline.py (Step 1). Inputs are arrays/CSV paths, outputs are
PNG files in ``output_dir``; nothing is returned except the saved path(s).
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from emagpy import Problem

from em_methodology import drop_coils_from_problem


def plot_step1_ensemble_single_file(
    models: np.ndarray,
    depths0: Sequence[float],
    output_dir: str = "mfc2_outputs",
    depths_are_fixed: bool = True,
    depths_ensemble: np.ndarray | None = None,
) -> None:
    """
    Generates staircase validation plots tracking individual model lines
    and central tendencies directly against interface boundaries.

    FIX (depth-aware plotting): previously, every ensemble member's EC
    staircase was drawn against the *static* ``depths0`` starting grid,
    even when ``fixedDepths=False`` allowed EMagPy to fit different
    interface positions per member (returned in ``k.depths[0]``, shape
    ``(num_measured_points, num_layers-1)``). That meant free/mixed-depth
    scenarios were visually indistinguishable from fixed-depth ones — the
    staircase always broke at the same y-values regardless of what the
    inversion actually found.

    ``depths_ensemble``, when supplied, fixes this: each member's step plot
    is now drawn using its OWN fitted depths (averaged across measurement
    points), so real depth spread across the ensemble becomes visible.

    NaN-safety: ``models`` (and ``depths_ensemble``) may contain NaN cells
    for real-data time-lapse runs whose spatial reference grid isn't fully
    covered by every epoch (see
    ``geometry_utils.match_epochs_to_reference_grid``). All spatial
    reductions below use ``nanmean``/``nanmedian`` so a partially-covered
    point never turns an otherwise-valid profile into an all-NaN one.

    Parameters
    ----------
    depths_ensemble : np.ndarray, optional
        Per-member fitted interface depths, shape
        ``(ensemble_size, num_measured_points, num_layers-1)``, as captured
        from EMagPy's ``k.depths[0]`` for each inversion. Required to show
        true depth spread when ``depths_are_fixed=False``. When ``None``
        (default), falls back to the previous static-``depths0``-grid
        behaviour for backward compatibility.
    """
    num_members, num_positions, num_layers = models.shape
    max_depth_view = depths0[-1] + 0.5

    use_member_depths = depths_ensemble is not None and not depths_are_fixed
    if use_member_depths:
        expected_shape = (num_members, num_positions, num_layers - 1)
        if depths_ensemble.shape != expected_shape:
            print(f"[WARN] depths_ensemble shape {depths_ensemble.shape} != expected "
                  f"{expected_shape}; falling back to static depths0 grid for plotting.")
            use_member_depths = False

    fig, ax = plt.subplots(figsize=(6, 7))

    for i in range(num_members):
        profile_1d = np.nanmean(models[i], axis=0)
        ec_padded = np.append(profile_1d, profile_1d[-1])

        if use_member_depths:
            member_depths = np.nanmean(depths_ensemble[i], axis=0)  # (num_layers-1,)
            plot_depths_i = [0.0] + list(member_depths) + [max_depth_view]
        else:
            plot_depths_i = [0.0] + list(depths0) + [max_depth_view]

        ax.step(ec_padded, plot_depths_i, where='pre', alpha=0.15, color='teal')

    ensemble_median = np.nanmedian(models, axis=(0, 1))
    median_padded = np.append(ensemble_median, ensemble_median[-1])

    if use_member_depths:
        # Median fitted depth per interface, across members and positions
        median_depths = np.nanmedian(np.nanmean(depths_ensemble, axis=1), axis=0)
        median_plot_depths = [0.0] + list(median_depths) + [max_depth_view]
    else:
        median_plot_depths = [0.0] + list(depths0) + [max_depth_view]

    # ADDED: mean + min/max envelope across the WHOLE ensemble (all members
    # x all spatial points collapsed together, same axes as the median
    # above). The envelope makes it visually explicit how tight/loose the
    # ensemble agreement is at each depth, rather than relying on the eye
    # to judge overlap density from 288 semi-transparent spaghetti lines.
    # NOTE: this collapses across spatial position just like the member
    # lines/median above -- if individual points behave very differently
    # from one another (e.g. one point ~homogeneous with depth, another
    # strongly layered), that will NOT be visible here; it will show up as
    # a wide min/max band instead. Use plot_step1_ensemble_by_point()
    # (same output_dir, 'step1_ensemble_spread_by_point.png') to see
    # whether a shared feature like the 0.3-0.6 m conductive layer is
    # real and shared, or a spatial-averaging artifact.
    ensemble_mean = np.nanmean(models, axis=(0, 1))
    ensemble_min  = np.nanmin(models, axis=(0, 1))
    ensemble_max  = np.nanmax(models, axis=(0, 1))
    mean_padded = np.append(ensemble_mean, ensemble_mean[-1])
    min_padded  = np.append(ensemble_min, ensemble_min[-1])
    max_padded  = np.append(ensemble_max, ensemble_max[-1])

    ax.fill_betweenx(median_plot_depths, min_padded, max_padded, step='post',
                      alpha=0.18, color='teal', linewidth=0,
                      label='Ensemble min–max')
    ax.step(mean_padded, median_plot_depths, where='pre', color='navy',
            linewidth=1.6, linestyle='-', label='Ensemble Mean')

    ax.step(median_padded, median_plot_depths, where='pre', color='black', linewidth=2.2,
            linestyle='--', label='Ensemble Median')

    if depths_are_fixed:
        title = "Step 1 Ensemble: Inverted EC vs Fixed Layer Depths"
        ylabel = "Fixed Layer Depth [m]"
    elif use_member_depths:
        title = "Step 1 Ensemble: Inverted EC vs Actual Fitted Layer Depths"
        ylabel = "Fitted Layer Depth [m]"
        for j, d in enumerate(depths0):
            ax.axhline(d, color='darkorange', linestyle=':', linewidth=0.8, alpha=0.4,
                       label='Starting depth0' if j == 0 else None)
    else:
        # No depths_ensemble was supplied — fall back to the old approximate view
        title = "Step 1 Ensemble: Inverted EC (layer interfaces were free to move)"
        ylabel = "Starting Layer Depth [m] (interfaces may have shifted)"
        for d in depths0:
            ax.axhline(d, color='darkorange', linestyle=':', linewidth=0.8, alpha=0.6)

    ax.set_title(title, fontsize=11, pad=15)
    ax.set_xlabel("Electrical Conductivity ($EC$) [mS/m]", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_yticks(list(depths0) + [0.0])
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "step1_ensemble_spread.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_step1_ensemble_by_point(
    models: np.ndarray,
    depths0: Sequence[float],
    output_dir: str = "mfc2_outputs",
    max_points: int = 12,
    point_labels: Sequence[str] | None = None,
) -> None:
    """
    Small-multiples companion to ``plot_step1_ensemble_single_file``: one
    subplot PER SPATIAL POINT, each showing that point's own median EC
    profile ± min/max across ensemble members, instead of the single-file
    plot's average over ALL points.

    Why this exists
    ----------------
    ``plot_step1_ensemble_single_file`` computes every member's profile as
    ``nanmean(models[i], axis=0)`` -- averaged across spatial points before
    it's ever plotted. That can manufacture the appearance of a shared
    feature (e.g. "every model agrees on a more-conductive layer between
    0.3 and 0.6 m") even when it's really an artifact of blending together
    points with genuinely different depth behaviour -- e.g. one point close
    to homogeneous with depth, another strongly layered. This function
    keeps points separate so you can tell the two cases apart.

    NaN-safety: matches ``plot_step1_ensemble_single_file`` -- uses
    ``nanmedian``/``nanmin``/``nanmax`` throughout, so points left as NaN
    for some ensemble members (e.g. real-data time-lapse spatial-reference-
    grid gaps, see ``geometry_utils.match_epochs_to_reference_grid``) don't
    corrupt the points that do have data.

    Parameters
    ----------
    models : np.ndarray
        Shape ``(ensemble_size, num_measured_points, num_layers)`` -- same
        array passed to ``plot_step1_ensemble_single_file``.
    max_points : int, optional
        If ``num_measured_points`` exceeds this, an evenly-spaced subsample
        of ``max_points`` is plotted instead of all of them, to keep the
        figure readable. Default 12.
    point_labels : sequence of str, optional
        Custom label per selected point (e.g. ``"x=2.0, y=4.0"``). When
        ``None`` (default), points are labeled by their index into
        ``models``'s spatial axis.
    """
    num_members, num_positions, num_layers = models.shape
    max_depth_view = depths0[-1] + 0.5
    plot_depths = [0.0] + list(depths0) + [max_depth_view]

    if num_positions > max_points:
        point_idx = np.linspace(0, num_positions - 1, max_points).round().astype(int)
        point_idx = np.unique(point_idx)
    else:
        point_idx = np.arange(num_positions)
    n_show = len(point_idx)

    ncols = min(4, n_show)
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.6 * nrows),
                              sharex=True, sharey=True, squeeze=False)

    for ax_i, p in enumerate(point_idx):
        ax = axes.flat[ax_i]
        point_profiles = models[:, p, :]  # (ensemble_size, num_layers)

        med = np.nanmedian(point_profiles, axis=0)
        mn  = np.nanmin(point_profiles, axis=0)
        mx  = np.nanmax(point_profiles, axis=0)
        med_p = np.append(med, med[-1])
        mn_p  = np.append(mn, mn[-1])
        mx_p  = np.append(mx, mx[-1])

        ax.fill_betweenx(plot_depths, mn_p, mx_p, step='post', alpha=0.25,
                          color='teal', linewidth=0)
        ax.step(med_p, plot_depths, where='pre', color='black', linewidth=1.6)

        label = point_labels[ax_i] if point_labels is not None else f"point {p}"
        ax.set_title(label, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    for ax_i in range(n_show, nrows * ncols):
        axes.flat[ax_i].axis("off")

    axes[0, 0].invert_yaxis()  # shared y-axis, so this flips all subplots
    for ax in axes[:, 0]:
        ax.set_ylabel("Layer Depth [m]", fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("EC [mS/m]", fontsize=9)

    fig.suptitle("Step 1 Ensemble Spread by Spatial Point (median ± min/max)",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "step1_ensemble_spread_by_point.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_raw_apparent_conductivity(fname: str, output_dir: str = "mfc2_outputs",
                                   em_unit: str = "ppt", drop_coils=None) -> None:
    """
    Plot raw apparent conductivity / quadrature data from a survey CSV.

    Strategy
    --------
    1. Try emagpy's k.show(dist=True) first (works for synthetic CSVs with
       standard coil column names like 'VCP0.5', 'HCP1.0', etc.).
    2. If that produces a blank plot (axes data range < 0.5 m or all-zero
       ECa), fall back to a direct pandas read and plot every numeric column
       that is not an x/y/elevation coordinate.  This handles real GEM-2
       quadrature CSVs whose column names (e.g. 'HCP1.66f425h1_quad') are
       not recognised by emagpy's internal show() method.

    Parameters
    ----------
    drop_coils : None, str, int, float, or sequence thereof, optional
        Coil(s)/frequency channel(s) that are excluded from the actual
        ensemble inversion elsewhere in the pipeline (see
        ``drop_coils_from_problem``). This raw plot intentionally still
        *shows* their signal — useful for judging why a channel looked
        noisy enough to exclude — but the matching panel(s) are flagged
        with a red "EXCLUDED FROM INVERSION" annotation/border so it's
        visually unambiguous that this channel did not feed Step 1+.
        Default ``None`` (no flagging).
    """
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "raw_eca_transect.png")

    # Resolve drop_coils into a set of exact coil names up front, via a
    # dry-run match against this survey's loaded coils, so both the
    # emagpy-native and the direct-CSV fallback path can flag the same
    # panel(s) consistently — without ever actually removing data from
    # this diagnostic plot.
    dropped_coil_names: list[str] = []
    if drop_coils is not None:
        try:
            _k_probe = Problem()
            _k_probe.createSurvey(fname, unit=em_unit)
            dropped_coil_names = drop_coils_from_problem(_k_probe, drop_coils, dry_run=True)
        except Exception as _e:
            print(f"[PLOT][WARN] Could not resolve drop_coils for annotation "
                  f"({_e}); raw ECa plot will be unflagged.")

    def _is_dropped(label: str) -> bool:
        """True if `label` (an axis title/column name) corresponds to one
        of the resolved dropped coils — matched loosely since emagpy/column
        labels may carry suffixes like '_quad' that the bare coil name
        (e.g. 'HCP1.66f425h1') doesn't have."""
        label_l = label.lower()
        return any(name.lower() in label_l for name in dropped_coil_names)

    # ── Attempt 1: emagpy native show() ────────────────────────────────
    try:
        k = Problem()
        k.createSurvey(fname, unit=em_unit)
        k.show(dist=True)
        fig = plt.gcf()

        # Detect blank plot: emagpy leaves a default ±0.05 x-range when it
        # finds no recognisable coil columns.
        axes = fig.get_axes()
        is_blank = True
        for ax in axes:
            xlim = ax.get_xlim()
            if abs(xlim[1] - xlim[0]) > 0.5:   # real survey data is >metres wide
                is_blank = False
                break

        if not is_blank:
            if dropped_coil_names:
                _flag_dropped_axes(axes, dropped_coil_names, _is_dropped)
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[PLOT] Raw ECa transect (emagpy) → '{plot_path}'")
            return

        plt.close(fig)
        print("[PLOT][WARN] emagpy show() produced a blank plot — "
              "falling back to direct CSV plot.")

    except Exception as e:
        plt.close("all")
        print(f"[PLOT][WARN] emagpy show() failed ({e}) — "
              "falling back to direct CSV plot.")

    # ── Attempt 2: direct pandas CSV plot ──────────────────────────────
    try:
        df = pd.read_csv(fname)
        df.columns = df.columns.str.strip()

        # Identify the along-transect x column
        x_col = next((c for c in df.columns if c.lower() in ("x", "dist", "distance")), None)

        # Signal columns: numeric, not coordinate/elevation
        skip = {"x", "y", "utm_x", "utm_y", "elevation", "gpsd", "alt",
                 "lat", "lon", "latitude", "longitude"}
        signal_cols = [
            c for c in df.columns
            if c.lower() not in skip
            and pd.api.types.is_numeric_dtype(df[c])
            and c != x_col
        ]

        if not signal_cols:
            print(f"[PLOT][WARN] No plottable signal columns found in '{fname}'. "
                  "Skipping raw ECa plot.")
            return

        n_cols = len(signal_cols)
        ncols_grid = min(n_cols, 3)
        n_rows = int(np.ceil(n_cols / ncols_grid))
        fig, axes = plt.subplots(n_rows, ncols_grid,
                                 figsize=(5 * ncols_grid, 3.5 * n_rows),
                                 squeeze=False)

        x_vals = df[x_col].to_numpy(dtype=float) if x_col else np.arange(len(df))

        for idx, col in enumerate(signal_cols):
            r, c = divmod(idx, ncols_grid)
            ax = axes[r][c]
            is_dropped = _is_dropped(col)
            ax.plot(x_vals, df[col].to_numpy(dtype=float),
                    linewidth=1.0,
                    color="tab:red" if is_dropped else f"C{idx % 10}",
                    alpha=0.85)
            ax.set_title(col, fontsize=8,
                        color="tab:red" if is_dropped else "black",
                        fontweight="bold" if is_dropped else "normal")
            ax.set_xlabel("Distance [m]" if x_col else "Index", fontsize=8)
            ax.set_ylabel("Signal", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.tick_params(labelsize=7)
            if is_dropped:
                for spine in ax.spines.values():
                    spine.set_edgecolor("tab:red")
                    spine.set_linewidth(2.0)
                ax.text(
                    0.5, 0.95, "EXCLUDED FROM INVERSION",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=7, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="tab:red", alpha=0.85),
                )

        # Hide unused panels
        for idx in range(n_cols, n_rows * ncols_grid):
            r, c = divmod(idx, ncols_grid)
            axes[r][c].set_visible(False)

        fig.suptitle(
            f"Raw Survey Signal — {os.path.basename(fname)}"
            + (f"  (red = excluded from inversion: {', '.join(dropped_coil_names)})"
               if dropped_coil_names else ""),
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Raw survey signals (direct CSV) → '{plot_path}'")

    except Exception as e2:
        plt.close("all")
        print(f"[PLOT][WARN] Direct CSV plot also failed: {e2}. "
              "Raw ECa plot skipped.")


def plot_filtered_apparent_conductivity(fname: str, output_dir: str = "mfc2_outputs",
                                        em_unit: str = "ppt", drop_coils=None,
                                        prefilter_fname: str | None = None) -> None:
    """
    Plot the survey signal AFTER the load-time cleaning chain
    (``preprocess_em_survey``: drop_coils -> filterRange -> filterRepeated ->
    filterPercentile -> rollingMean). Companion to
    ``plot_raw_apparent_conductivity``; writes ``filtered_eca_transect.png``.

    Differences from the raw plot
    -----------------------------
    * Coils excluded via ``drop_coils`` are REMOVED from the figure (the
      raw plot keeps them and flags them in red), so the panels shown here
      are exactly the channels that feed the inversion.
    * If ``prefilter_fname`` is given and exists (the pre-filter snapshot
      written by ``_load_real_data``), its signal is drawn in light grey
      behind each filtered line, so the effect of the range / repeated /
      percentile / rolling-mean stages is visible panel by panel. Point
      counts before -> after are shown in each panel.

    Parameters
    ----------
    fname : str
        Filtered, emagpy-ready survey CSV (comma-separated, ``x``/``y``/
        ``elevation`` + coil columns).
    drop_coils : None, str, int, float, or sequence thereof, optional
        Same spec as ``drop_coils_from_problem``; matching coil columns are
        omitted from the plot.
    prefilter_fname : str or None, optional
        Same-format CSV captured before filtering, used for the grey
        overlay. ``None`` (or a missing file) plots the filtered data only.
    """
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "filtered_eca_transect.png")

    dropped_coil_names: list[str] = []
    if drop_coils is not None:
        try:
            _k_probe = Problem()
            _k_probe.createSurvey(fname, unit=em_unit)
            dropped_coil_names = drop_coils_from_problem(_k_probe, drop_coils, dry_run=True)
        except Exception as _e:
            print(f"[PLOT][WARN] Could not resolve drop_coils for filtered plot "
                  f"({_e}); dropped channels will NOT be hidden.")

    def _is_dropped(label: str) -> bool:
        label_l = label.lower()
        return any(name.lower() in label_l for name in dropped_coil_names)

    try:
        df = pd.read_csv(fname)
        df.columns = df.columns.str.strip()

        df_pre = None
        if prefilter_fname and os.path.exists(prefilter_fname):
            df_pre = pd.read_csv(prefilter_fname)
            df_pre.columns = df_pre.columns.str.strip()

        x_col = next((c for c in df.columns if c.lower() in ("x", "dist", "distance")), None)
        skip = {"x", "y", "utm_x", "utm_y", "elevation", "gpsd", "alt",
                "lat", "lon", "latitude", "longitude"}
        signal_cols = [
            c for c in df.columns
            if c.lower() not in skip
            and pd.api.types.is_numeric_dtype(df[c])
            and c != x_col
            and not _is_dropped(c)
        ]
        if not signal_cols:
            print(f"[PLOT][WARN] No plottable signal columns left in '{fname}' "
                  "after dropping coils. Filtered ECa plot skipped.")
            return

        n_cols = len(signal_cols)
        ncols_grid = min(n_cols, 3)
        n_rows = int(np.ceil(n_cols / ncols_grid))
        fig, axes = plt.subplots(n_rows, ncols_grid,
                                 figsize=(5 * ncols_grid, 3.5 * n_rows),
                                 squeeze=False)

        x_vals = df[x_col].to_numpy(dtype=float) if x_col else np.arange(len(df))
        x_pre = None
        if df_pre is not None:
            x_pre = (df_pre[x_col].to_numpy(dtype=float)
                     if x_col and x_col in df_pre.columns else np.arange(len(df_pre)))

        for idx, col in enumerate(signal_cols):
            r, c = divmod(idx, ncols_grid)
            ax = axes[r][c]
            n_pre_txt = ""
            if df_pre is not None and col in df_pre.columns:
                ax.plot(x_pre, df_pre[col].to_numpy(dtype=float), linewidth=0.8,
                        color="0.7", alpha=0.9, label="before filtering", zorder=1)
                n_pre_txt = f"{len(df_pre)} → "
            ax.plot(x_vals, df[col].to_numpy(dtype=float), linewidth=1.1,
                    color=f"C{idx % 10}", alpha=0.95, label="after filtering", zorder=2)
            ax.set_title(f"{col}  ({n_pre_txt}{len(df)} pts)", fontsize=8)
            ax.set_xlabel("Distance [m]" if x_col else "Index", fontsize=8)
            ax.set_ylabel("Signal", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.tick_params(labelsize=7)
            if idx == 0 and df_pre is not None:
                ax.legend(fontsize=7, loc="best")

        for idx in range(n_cols, n_rows * ncols_grid):
            r, c = divmod(idx, ncols_grid)
            axes[r][c].set_visible(False)

        fig.suptitle(
            f"Filtered Survey Signal — {os.path.basename(fname)}"
            + (f"  (dropped, not shown: {', '.join(dropped_coil_names)})"
               if dropped_coil_names else ""),
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Filtered survey signals → '{plot_path}'")

    except Exception as e:
        plt.close("all")
        print(f"[PLOT][WARN] Filtered ECa plot failed: {e}. Skipped.")


def _flag_dropped_axes(axes, dropped_coil_names: list[str], is_dropped_fn) -> None:
    """
    Best-effort flagging of dropped-coil panel(s) on a figure produced by
    emagpy's own ``Problem.show()``. emagpy's per-axis titles typically
    include the coil name, so this matches on title text; if emagpy ever
    changes its labeling and nothing matches, this silently flags nothing
    (the raw plot still renders normally — annotation is best-effort only,
    never load-bearing for the actual exclusion which is enforced
    separately in the real inversion).
    """
    any_flagged = False
    for ax in axes:
        title = ax.get_title()
        if title and is_dropped_fn(title):
            any_flagged = True
            ax.set_title(title, color="tab:red", fontweight="bold", fontsize=ax.title.get_fontsize())
            for spine in ax.spines.values():
                spine.set_edgecolor("tab:red")
                spine.set_linewidth(2.0)
            ax.text(
                0.5, 0.95, "EXCLUDED FROM INVERSION",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="tab:red", alpha=0.85),
            )
    if not any_flagged and dropped_coil_names:
        print(f"[PLOT][WARN] Could not match dropped coil(s) "
              f"{dropped_coil_names} to any emagpy show() panel title — "
              "annotation skipped (exclusion itself is unaffected).")
