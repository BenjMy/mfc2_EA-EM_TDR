"""Soil-water-content (SWC) uncertainty from the ensemble.

RMSE / MAE are *metrics* (one number, how far theta_calc is from the TDR
probes or from the truth).  This module gives an *uncertainty*: for every
(time, point, layer) a standard deviation, so the result reads
``theta = 0.30 +- 0.10 m3/m3``.

Definition (same weights and Bessel-like correction as the CV map of
``step4_select``, paper Eq. 6-8): take the K best ensemble members chosen in
Step 5, convert each member's bulk EC to theta, and take the RMSE-weighted
mean and standard deviation across the members.  Two flavours:

``std_ec``     members' EC through the FINAL (Step 4) Waxman-Smits fit.
               Spread caused by the EM inversion only.
``std_total``  every member's EC through ITS OWN Step 2 calibration.
               EM-inversion spread + Waxman-Smits calibration spread.
``std_ws``     sqrt(max(total^2 - ec^2, 0)): the calibration share.

What it does NOT include: model error of the Waxman-Smits law itself, TDR
noise beyond what the calibration fit propagates, and anything that is common
to all members (a shared bias).  So it is a spread, not a guaranteed
confidence interval; for synthetic runs ``uncertainty_coverage`` tells whether
the +-1/2 sigma bands actually contain the truth.
"""
from __future__ import annotations

import os

import numpy as np


def ec_grid_to_theta(ec_grid, fit):
    """Waxman-Smits EC -> theta, per horizon (last axis = layers).

    Same formula as ``step4_select`` (theta = (a*EC/(sigma_w+sigma_s))**(1/m)).
    Layers without a horizon fit give 0; NaN EC stays NaN.
    """
    ec_grid = np.asarray(ec_grid, dtype=float)
    out = np.zeros_like(ec_grid)
    for l_idx in range(ec_grid.shape[-1]):
        if l_idx >= len(fit.horizon_fits):
            continue
        hf = fit.horizon_fits[l_idx]
        ec = np.clip(ec_grid[..., l_idx], 1e-6, None)
        ratio = hf.a * ec / (hf.sigma_w + hf.sigma_s)
        out[..., l_idx] = np.power(np.clip(ratio, 0.0, None), 1.0 / hf.m)
    out[~np.isfinite(ec_grid)] = np.nan
    return out


def rmse_weights(member_rmse):
    """RMSE-normalised weights (best -> 1), as in step4_select (Eq. 6-7)."""
    r = np.asarray(member_rmse, dtype=float)
    n = len(r)
    spread = r.max() - r.min()
    if n == 1 or spread < 1e-15:
        return np.full(n, 1.0 / n)
    norm = (r - r.max()) / (r.min() - r.max())
    tot = norm.sum()
    return norm / tot if tot >= 1e-15 else np.full(n, 1.0 / n)


def weighted_mean_std(x, w):
    """Weighted mean and Bessel-like std over axis 0 (K members)."""
    k = x.shape[0]
    wv = np.asarray(w, dtype=float).reshape((k,) + (1,) * (x.ndim - 1))
    mean = np.sum(wv * x, axis=0)
    if k == 1:
        return mean, np.zeros_like(mean)
    std = np.sqrt((k / (k - 1)) * np.sum(wv * (x - mean) ** 2, axis=0))
    return mean, std


def theta_ensemble_uncertainty(models, ranking, K, member_fits, final_fit):
    """Per-cell theta mean and std from the top-K members.

    Parameters
    ----------
    models       (N, ..., n_layers) member EC, same axis order as ``ranking``.
    ranking      member indices, best first (``average_inverted_models``).
    K            optimal stack depth chosen in Step 5.
    member_fits  per-member Step 2 ``PetroFit`` list, indexed like ``models``
                 (NOT the per-K Step 4 list).
    final_fit    the Step 4 fit of the chosen K (``fits_step4[K-1]``).

    Returns dict: theta_mean, std_total, std_ec, std_ws (all ``models.shape[1:]``).
    """
    idx = np.asarray(ranking)[:K]
    w = rmse_weights([member_fits[i].rmse for i in idx])
    ec = np.asarray(models, dtype=float)[idx]
    th_total = np.stack([ec_grid_to_theta(ec[j], member_fits[i]) for j, i in enumerate(idx)])
    th_ec = np.stack([ec_grid_to_theta(ec[j], final_fit) for j in range(len(idx))])
    mean_total, std_total = weighted_mean_std(th_total, w)
    mean_ec, std_ec = weighted_mean_std(th_ec, w)
    return {
        "theta_mean": mean_ec,            # = RMSE-weighted theta under the final fit
        "theta_mean_total": mean_total,
        "std_total": std_total,
        "std_ec": std_ec,
        "std_ws": np.sqrt(np.clip(std_total ** 2 - std_ec ** 2, 0.0, None)),
        "weights": w,
    }


def uncertainty_coverage(theta_true, theta_mean, std, nsigmas=(1.0, 2.0)):
    """Synthetic check: fraction of cells with |truth-mean| <= n*std.

    ~0.68 / ~0.95 would mean a calibrated Gaussian uncertainty; far lower
    means the ensemble spread under-states the real error (a shared bias the
    members cannot see).  Also returns the mean error and mean std.
    """
    err = np.asarray(theta_mean, float) - np.asarray(theta_true, float)
    ok = np.isfinite(err) & np.isfinite(std)
    out = {f"within_{n:g}sigma": float(np.mean(np.abs(err[ok]) <= n * std[ok]))
           for n in nsigmas}
    out.update(rmse=float(np.sqrt(np.mean(err[ok] ** 2))),
               mean_std=float(np.mean(std[ok])), n=int(ok.sum()))
    return out


def plot_swc_uncertainty(theta_mean, std_total, std_ec, layer_centers, output_dir,
                         theta_true=None, scenario_name="default", K=None,
                         coverage=None):
    """Figure ``solution/solution_swc_uncertainty.png``.

    One row per time step (3-D inputs: time, points, layers).  Left: spatial
    mean profile with +-1 sigma bands (dark = EM inversion only, light = plus
    calibration), per-point profiles faint, truth dashed.  Right: map of the
    total sigma (points x layers) in m3/m3.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _as3d(a):
        a = np.asarray(a, float)
        return a[np.newaxis] if a.ndim == 2 else a
    th, st, se = _as3d(theta_mean), _as3d(std_total), _as3d(std_ec)
    nt, npts, nl = th.shape
    z = np.asarray(layer_centers, float)[:nl]
    fig, axes = plt.subplots(nt, 2, figsize=(10.5, 4.2 * nt), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.1]})
    for t in range(nt):
        ax, ax2 = axes[t]
        m = np.nanmean(th[t], axis=0)
        s_tot = np.nanmean(st[t], axis=0)
        s_ec = np.nanmean(se[t], axis=0)
        for p in range(npts):
            ax.plot(th[t, p], z, color="0.7", lw=0.8, zorder=1)
        ax.fill_betweenx(z, m - s_tot, m + s_tot, color="tab:blue", alpha=0.25,
                         label=r"$\pm1\sigma$ EM + calibration", zorder=2)
        ax.fill_betweenx(z, m - s_ec, m + s_ec, color="tab:blue", alpha=0.5,
                         label=r"$\pm1\sigma$ EM inversion only", zorder=3)
        ax.plot(m, z, "k-o", lw=2, label="ensemble mean (spatial mean)", zorder=4)
        if theta_true is not None:
            tt = np.asarray(theta_true, float)
            tt = tt[t] if tt.ndim == 3 else tt
            ax.plot(np.nanmean(tt, axis=0), z, "--x", color="crimson", lw=2,
                    label="truth (spatial mean)", zorder=5)
        ax.set_xlabel(r"$\theta$ [m$^3$/m$^3$]")
        ax.set_ylabel("Depth [m]")
        ax.invert_yaxis()
        ax.grid(ls=":", alpha=0.6)
        ax.set_title(f"SWC +- uncertainty, time step {t + 1}", fontsize=10, fontweight="bold")
        if t == 0:
            ax.legend(fontsize=7, loc="lower left")
        im = ax2.imshow(st[t].T, aspect="auto", cmap="viridis", origin="upper",
                        extent=[-0.5, npts - 0.5, nl - 0.5, -0.5])
        ax2.set_yticks(range(nl))
        ax2.set_yticklabels([f"{v:.2f}" for v in z])
        ax2.set_xlabel("measurement point index")
        ax2.set_ylabel("layer centre [m]")
        ax2.set_title(r"$\sigma_\theta$ (total) [m$^3$/m$^3$]", fontsize=10)
        fig.colorbar(im, ax=ax2, fraction=0.046)
    sub = f"  K={K}" if K is not None else ""
    if coverage:
        sub += (f"  |  truth within 1/2 sigma: {coverage['within_1sigma']:.0%} / "
                f"{coverage['within_2sigma']:.0%}  (68% / 95% if calibrated)")
    fig.suptitle(f"Final SWC uncertainty (ensemble spread) - {scenario_name}{sub}", fontsize=11)
    fig.tight_layout()
    sol = os.path.join(output_dir, "solution")
    os.makedirs(sol, exist_ok=True)
    path = os.path.join(sol, "solution_swc_uncertainty.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
