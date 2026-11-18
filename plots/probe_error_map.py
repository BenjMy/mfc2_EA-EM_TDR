"""Probe error map: signed error theta_calc - theta_TDR at every TDR probe (to-do item F).

Post-hoc and cheap: it only reads arrays a finished run already wrote to ``output_model/``
(no emagpy, no inversion), so it also works for ``--stage calibrate`` runs.

    python -m plots.probe_error_map <run_dir> [--x x.npy --y y.npy] [--held-out held.npy]

writes next to the run (``<run_dir>/probe_errors/``):
    probe_errors.csv              station, x, y, epoch, horizon, theta_TDR, theta_calc, error, held_out_flag
    probe_errors_by_station.csv   per station and horizon: n, RMSE, bias
    probe_errors_by_epoch.csv     per epoch and horizon: n, RMSE, bias
    probe_error_map.png           plan view, one panel per (horizon, epoch), colour = signed error

Inputs (all in ``<run_dir>/output_model``):
    tdr_train_theta / _positions / _times / _layers   the calibration observations
    theta_calc                                        (time, points, layers)
    synthetic_x_coords / synthetic_y_coords           optional (else --x/--y, else the point index)
    held_out                                          optional bool per observation; default all False

``theta_calc`` is the final model, so without ``held_out`` the map is IN-SAMPLE. For a
held-out map pass the flags of a train/val or leave-one-station-out run.
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plots._common import ensure_dir, save_fig  # noqa: E402

COLUMNS = ["station", "x", "y", "epoch", "horizon", "theta_TDR", "theta_calc", "error", "held_out_flag"]


def build_probe_error_table(theta_obs, theta_calc, positions, times, layers, x, y=None,
                            held_out=None) -> dict[str, np.ndarray]:
    """One row per observation. ``theta_calc`` is (time, points, layers) or (points, layers).

    station = index of the EM point the probe is matched to (probes matched to the same
    point share a station id). error = theta_calc - theta_TDR.
    """
    theta_obs = np.asarray(theta_obs, dtype=float)
    pos, tim, lay = (np.asarray(a, dtype=int) for a in (positions, times, layers))
    calc = np.asarray(theta_calc, dtype=float)
    if calc.ndim == 2:
        calc = calc[None]
    if not (len(theta_obs) == len(pos) == len(tim) == len(lay)):
        raise ValueError("theta_obs, positions, times and layers must have the same length")
    if tim.size and (tim.max() >= calc.shape[0] or pos.max() >= calc.shape[1] or lay.max() >= calc.shape[2]):
        raise ValueError(f"indices out of range for theta_calc {calc.shape}")
    x = np.asarray(x, dtype=float)
    yv = np.zeros_like(x) if y is None else np.asarray(y, dtype=float)
    pred = calc[tim, pos, lay]
    held = np.zeros(len(theta_obs), bool) if held_out is None else np.asarray(held_out, bool)
    return {"station": pos, "x": x[pos], "y": yv[pos], "epoch": tim, "horizon": lay,
            "theta_TDR": theta_obs, "theta_calc": pred, "error": pred - theta_obs,
            "held_out_flag": held}


def _stats(err: np.ndarray) -> tuple[int, float, float]:
    ok = np.isfinite(err)
    n = int(ok.sum())
    return (n, float(np.sqrt(np.mean(err[ok] ** 2))), float(np.mean(err[ok]))) if n else (0, np.nan, np.nan)


def summarise(table: dict[str, np.ndarray], by: str) -> list[dict]:
    """RMSE and bias per (``by`` in {"station", "epoch"}, horizon)."""
    rows = []
    for key in np.unique(table[by]):
        for hz in np.unique(table["horizon"]):
            m = (table[by] == key) & (table["horizon"] == hz)
            if m.any():
                n, rmse, bias = _stats(table["error"][m])
                rows.append({by: int(key), "horizon": int(hz), "n": n, "RMSE": rmse, "bias": bias})
    return rows


def write_csv(path: str, rows: Sequence[dict], columns: Sequence[str]) -> str:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in columns})
    return path


def table_rows(table: dict[str, np.ndarray]) -> list[dict]:
    n = len(table["error"])
    return [{c: (bool(table[c][i]) if c == "held_out_flag" else table[c][i].item()) for c in COLUMNS}
            for i in range(n)]


def plot_probe_error_map(table: dict[str, np.ndarray], output_path: str,
                         held_out_only: bool = False, title: str = "") -> str:
    """Plan view per (horizon, epoch); diverging colour map centred on 0, shared limits."""
    m = table["held_out_flag"] if held_out_only else np.ones(len(table["error"]), bool)
    if not m.any():
        raise ValueError("no observations to plot" + (" (no held-out flags set)" if held_out_only else ""))
    sel = {k: v[m] for k, v in table.items()}
    horizons, epochs = np.unique(sel["horizon"]), np.unique(sel["epoch"])
    vmax = float(np.nanmax(np.abs(sel["error"]))) or 1e-6
    one_d = np.allclose(sel["y"], sel["y"][0])
    fig, axes = plt.subplots(len(horizons), len(epochs),
                             figsize=(2.6 * len(epochs) + 1.6, 2.4 * len(horizons) + 0.8),
                             squeeze=False, sharex=True, sharey=True)
    sc = None
    for i, hz in enumerate(horizons):
        for j, ep in enumerate(epochs):
            ax = axes[i, j]
            k = (sel["horizon"] == hz) & (sel["epoch"] == ep)
            yy = np.zeros(int(k.sum())) if one_d else sel["y"][k]
            if k.any():
                sc = ax.scatter(sel["x"][k], yy, c=sel["error"][k], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                                s=70, edgecolors="k", linewidths=0.4)
                for xs, ys, st in zip(sel["x"][k], yy, sel["station"][k]):
                    ax.annotate(str(int(st)), (xs, ys), fontsize=6, xytext=(3, 3), textcoords="offset points")
                _, rmse, bias = _stats(sel["error"][k])
                ax.set_title(f"H{hz}, epoch {ep}\nRMSE {rmse:.3f}, bias {bias:+.3f}", fontsize=8)
            else:
                ax.set_title(f"H{hz}, epoch {ep}\nno data", fontsize=8)
            ax.grid(alpha=0.3)
            if one_d:
                ax.set_yticks([])
            if i == len(horizons) - 1:
                ax.set_xlabel("x [m]", fontsize=8)
            if j == 0 and not one_d:
                ax.set_ylabel("y [m]", fontsize=8)
    if sc is not None:
        fig.colorbar(sc, ax=axes, label="theta_calc - theta_TDR [m$^3$/m$^3$]", shrink=0.85)
    fig.suptitle(title or ("held-out probes" if held_out_only else "all probes (in-sample unless flagged)"),
                 fontsize=10)
    ensure_dir(os.path.dirname(os.path.abspath(output_path)))
    save_fig(fig, output_path, tag="PROBE-ERR")
    return output_path


def run_from_output_model(run_dir: str, x_path: str | None = None, y_path: str | None = None,
                          held_out_path: str | None = None, out_dir: str | None = None) -> dict[str, str]:
    """Read a finished run, write the CSVs and the map(s); returns {name: path}."""
    om = os.path.join(run_dir, "output_model")
    ld = lambda n: np.load(os.path.join(om, n + ".npy"))        # noqa: E731
    need = ["tdr_train_theta", "tdr_train_positions", "tdr_train_times", "tdr_train_layers", "theta_calc"]
    miss = [n for n in need if not os.path.exists(os.path.join(om, n + ".npy"))]
    if miss:
        raise FileNotFoundError(f"{om} lacks {miss} (run made before the Step 5 arrays were saved?)")
    theta_calc = ld("theta_calc")
    n_pts = theta_calc.shape[-2]

    def coord(path, default_name, fallback):
        p = path or os.path.join(om, default_name + ".npy")
        return np.load(p) if os.path.exists(p) else fallback

    x = coord(x_path, "synthetic_x_coords", np.arange(n_pts, dtype=float))
    y = coord(y_path, "synthetic_y_coords", None)
    held = np.load(held_out_path) if held_out_path else (
        ld("held_out") if os.path.exists(os.path.join(om, "held_out.npy")) else None)
    tab = build_probe_error_table(ld("tdr_train_theta"), theta_calc, ld("tdr_train_positions"),
                                  ld("tdr_train_times"), ld("tdr_train_layers"), x, y, held)
    out = ensure_dir(out_dir or os.path.join(run_dir, "probe_errors"))
    paths = {
        "probe_errors": write_csv(os.path.join(out, "probe_errors.csv"), table_rows(tab), COLUMNS),
        "by_station": write_csv(os.path.join(out, "probe_errors_by_station.csv"),
                                summarise(tab, "station"), ["station", "horizon", "n", "RMSE", "bias"]),
        "by_epoch": write_csv(os.path.join(out, "probe_errors_by_epoch.csv"),
                              summarise(tab, "epoch"), ["epoch", "horizon", "n", "RMSE", "bias"]),
        "map": plot_probe_error_map(tab, os.path.join(out, "probe_error_map.png")),
    }
    if tab["held_out_flag"].any():
        paths["map_held_out"] = plot_probe_error_map(tab, os.path.join(out, "probe_error_map_held_out.png"),
                                                     held_out_only=True)
    return paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--x"), ap.add_argument("--y"), ap.add_argument("--held-out")
    ap.add_argument("--out-dir")
    a = ap.parse_args(argv)
    for k, v in run_from_output_model(a.run_dir, a.x, a.y, a.held_out, a.out_dir).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
