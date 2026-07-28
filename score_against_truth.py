#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_against_truth.py
=======================
Compares the pipeline's recovered soil-moisture field (captured by
``run_and_score.py`` into an ``.npz``) against the KNOWN ground truth used
to build the synthetic-real-mirror dataset (``generate_synthetic_real_mirror.py``).

Ground truth is recomputed on demand from the exact same ``theta_truth()``
function used to generate the data -- imported directly from
``generate_synthetic_real_mirror.py`` rather than re-implemented here, so
the two can never silently drift apart. The noise term is set to zero for
scoring (we want the pipeline's estimate compared to the true mean field,
not to one noisy draw of it).

This is the number that answers "is the spatial map still consistent" even
at epochs/horizons with NO direct TDR coverage -- which is the whole point
of a synthetic ground-truth twin (real data can never tell you this).

Usage
-----
    python3 score_against_truth.py \\
        --capture ./pipeline_capture.npz \\
        --generator-dir /path/to/dir/containing/generate_synthetic_real_mirror.py \\
        --out ./scoring_summary.csv

Output
------
* A per-(epoch, horizon) CSV: RMSE, bias, R^2, n_points, calibrated (bool),
  had_tdr_this_epoch (bool -- epoch 0 in the real dates below is when TDR
  coverage exists; adapt EPOCHS_WITH_TDR if you restrict TDR to a subset).
* A console summary split by "calibrated horizon" vs "uncalibrated horizon"
  -- this contrast is the headline number for the depth-extrapolation
  scenario (deep horizons below 15/30 cm TDR reach get no direct
  calibration; this quantifies exactly how much that costs you).
"""

from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd


def _load_generator_module(generator_dir: str):
    path = os.path.join(generator_dir, "generate_synthetic_real_mirror.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"generate_synthetic_real_mirror.py not found in {generator_dir}")
    spec = importlib.util.spec_from_file_location("gen_mirror", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]
    if len(obs) == 0:
        return dict(rmse=float("nan"), bias=float("nan"), r2=float("nan"), n=0)
    resid = pred - obs
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    bias = float(np.mean(resid))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return dict(rmse=rmse, bias=bias, r2=r2, n=int(len(obs)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True, help="Output of run_and_score.py")
    ap.add_argument("--generator-dir", required=True,
                     help="Directory containing generate_synthetic_real_mirror.py")
    ap.add_argument("--out", default="./scoring_summary.csv")
    args = ap.parse_args()

    gen = _load_generator_module(args.generator_dir)
    data = np.load(args.capture)

    theta_calc = data["theta_calc"]     # (n_time, n_points, n_layers)
    ref_x, ref_y = data["ref_x"], data["ref_y"]
    is_calibrated = data["is_calibrated"] if "is_calibrated" in data else None
    cv_map = data["cv_map"] if "cv_map" in data else None

    if theta_calc.ndim == 2:
        theta_calc = theta_calc[np.newaxis, ...]
        if cv_map is not None and cv_map.ndim == 2:
            cv_map = cv_map[np.newaxis, ...]

    n_time, n_points, n_layers = theta_calc.shape
    if len(ref_x) != n_points:
        raise ValueError(
            f"reference grid has {len(ref_x)} points but theta_calc has "
            f"{n_points} -- capture file is inconsistent, re-run run_and_score.py."
        )
    if n_time > len(gen.REAL_EM_EPOCHS):
        raise ValueError(
            f"theta_calc has {n_time} epochs but generator only defines "
            f"{len(gen.REAL_EM_EPOCHS)} -- did you regenerate the data with "
            "a different epoch list than what produced this capture?"
        )

    rows = []
    for t in range(n_time):
        iso_date = gen.REAL_EM_EPOCHS[t][2]
        for l in range(n_layers):
            truth = gen.theta_truth(
                ref_x, ref_y, layer_idx=l, iso_date=iso_date,
                rng=np.random.default_rng(0), noise_std=0.0,  # noise-free mean field
            )
            pred = theta_calc[t, :, l]
            m = _metrics(truth, pred)
            row = dict(
                epoch=t, date=iso_date, horizon=l,
                depth_range=_horizon_label(l, gen.DEPTHS0),
                calibrated=(bool(is_calibrated[l]) if is_calibrated is not None
                            and l < len(is_calibrated) else None),
                **m,
            )
            if cv_map is not None:
                cv_l = cv_map[t, :, l]
                row["mean_cv"] = float(np.nanmean(cv_l))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[SCORE] wrote {args.out}\n")

    print(df.to_string(index=False,
                        float_format=lambda v: f"{v:8.4f}" if isinstance(v, float) else str(v)))

    if is_calibrated is not None:
        print("\n=== Calibrated vs. uncalibrated horizons (pooled across all epochs) ===")
        for calib_flag in (True, False):
            sub = df[df["calibrated"] == calib_flag]
            if sub.empty:
                continue
            label = "CALIBRATED (TDR reached this horizon)" if calib_flag else \
                    "UNCALIBRATED (no TDR at this horizon -- prior-only WS params)"
            print(f"\n  {label}")
            print(f"    mean RMSE : {sub['rmse'].mean():.4f} m3/m3   "
                  f"(range {sub['rmse'].min():.4f}-{sub['rmse'].max():.4f})")
            print(f"    mean bias : {sub['bias'].mean():+.4f} m3/m3")
            print(f"    mean R2   : {sub['r2'].mean():.4f}")
            if "mean_cv" in sub.columns:
                print(f"    mean CV   : {sub['mean_cv'].mean():.4f}")

    print("\n[SCORE] Done. This CSV is the raw material for the paper's "
          "depth-extrapolation / density-tradeoff figures.")


def _horizon_label(layer_idx: int, depths0: list[float]) -> str:
    bounds = [0.0] + list(depths0) + [None]
    lo = bounds[layer_idx]
    hi = bounds[layer_idx + 1]
    return f"{lo:.2f}-{hi:.2f} m" if hi is not None else f">{lo:.2f} m"


if __name__ == "__main__":
    main()
