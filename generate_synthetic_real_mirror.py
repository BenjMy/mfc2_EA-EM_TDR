#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_synthetic_real_mirror.py
==================================
Builds a synthetic "twin" of the real MFC2 field deployment: same 7 EM
survey dates, same per-epoch point-count/footprint asymmetry (including
the degraded 15/12/2020 epoch), same 24 TDR station coordinates, same
15 cm / 30 cm TDR depths -- but with a KNOWN ground-truth soil-moisture
field, so pipeline output can be scored against truth the way real field
data never allows.

Design decisions (documented, not hidden)
------------------------------------------
* Point counts are DOWNSAMPLED from the real per-epoch counts (up to
  ~98k) via `--scale` / `--max-points`, preserving the real epochs'
  RELATIVE asymmetry (epoch 3 stays ~1% the size of the others). This is
  deliberate, not a shortcut: the real pipeline runs `filterRepeated`
  before every inversion specifically because raw GPS logs are full of
  near-duplicate points, so matching the literal raw count is not what
  determines inversion quality -- effective spatial density is. Full-scale
  point counts (`--scale 1.0`) are supported but will be slow (emagpy's
  forward model is called once per point; see the loop below) and are
  meant to be run in a real compute environment, not this sandbox.
* Quadrature (ppm) values are produced by forward-modelling EC_bulk with
  emagpy's own `Problem.forward(forwardModel="CS", ...)` (same call
  pattern as `data_synthetic.py`), converting the resulting
  ECa [mS/m] back to GEM-2-style quadrature ppm via emagpy's own
  `eca2Q` -- the exact inverse of what `Survey.readDF(unit='ppm')` does
  on the way in. This round-trips exactly through the real pipeline's own
  conversion, so nothing about the "realism" of the raw files is fudged.
* Ground truth = smooth spatial field (fixed, deterministic) x seasonal
  cycle (cosine, wet-winter / dry-summer keyed to the real survey dates)
  x depth trend, plus i.i.d. noise. Petrophysical (Waxman-Smits) truth
  parameters are held CONSTANT across space and time -- matching this
  pipeline's own calibration assumption (a single per-horizon WS fit is
  assumed to hold everywhere/everywhen) -- so any recovery error measured
  downstream is attributable to sampling/coverage, not to violating an
  assumption the pipeline itself makes.

Outputs
-------
<outdir>/EMI_MFC2_synthetic/*.csv         raw EM survey files (semicolon-delimited)
<outdir>/daily_MFC_synthetic/MTX##_*.xlsx one file per TDR station
<outdir>/ground_truth/*.npy               truth arrays, NOT fed to the pipeline
<outdir>/mfc2_real_synthetic_mirror.json  scenario_loader-compatible JSON scenario
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

from emagpy import Problem
from emagpy.invertHelper import eca2Q

# ---------------------------------------------------------------------------
# Ground-truth facts about the real deployment (from the uploaded EM survey
# index and SOILNET_MFC2_coordinates_sensors.csv). Kept as plain data so a
# future update to the real inventory is a one-line diff.
# ---------------------------------------------------------------------------

# REAL_EM_EPOCHS = [
#     # (raw_index, ddmmyy, iso_date, n_points_real, min_x, max_x, min_y, max_y, min_z, max_z)
#     (1, "040820", "2020-08-04", 97830, 515487.23, 515726.21, 4468201.66, 4468512.02, 443.4, 490.6),
#     (2, "291020", "2020-10-29", 32351, 515489.20, 515699.03, 4468209.23, 4468489.60, 444.5, 484.78),
#     (3, "151220", "2020-12-15",   494, 515655.93, 515706.45, 4468226.35, 4468265.98, 464.6, 470.49),  # degraded epoch
#     (4, "050221", "2021-02-05", 62095, 515509.32, 515788.86, 4468206.11, 4468551.28, 434.81, 513.09),
#     (5, "300421", "2021-04-30", 23712, 515493.59, 515704.36, 4468209.43, 4468494.22, 434.41, 480.39),
#     (6, "050821", "2021-08-05", 51196, 515487.23, 515792.96, 4468210.18, 4468537.53, 439.2, 507.9),
#     (7, "161221", "2021-12-16", 46357, 515486.95, 515788.28, 4468219.42, 4468532.52, 441.1, 495.7),
# ]


REAL_EM_EPOCHS = [
    # (raw_index, ddmmyy, iso_date, n_points_real, min_x, max_x, min_y, max_y, min_z, max_z)
    (1, "040820", "2020-08-04", 9783, 515487.23, 515726.21, 4468201.66, 4468512.02, 443.4, 490.6),
    (2, "291020", "2020-10-29", 3235, 515489.20, 515699.03, 4468209.23, 4468489.60, 444.5, 484.78),
    (3, "151220", "2020-12-15",   49, 515655.93, 515706.45, 4468226.35, 4468265.98, 464.6, 470.49),  # degraded epoch
    (4, "050221", "2021-02-05", 6209, 515509.32, 515788.86, 4468206.11, 4468551.28, 434.81, 513.09),
    (5, "300421", "2021-04-30", 2371, 515493.59, 515704.36, 4468209.43, 4468494.22, 434.41, 480.39),
    (6, "050821", "2021-08-05", 5119, 515487.23, 515792.96, 4468210.18, 4468537.53, 439.2, 507.9),
    (7, "161221", "2021-12-16", 4635, 515486.95, 515788.28, 4468219.42, 4468532.52, 441.1, 495.7),
]


GEM2_COILS = [
    "HCP1.66f425h1", "HCP1.66f1525h1", "HCP1.66f5325h1",
    "HCP1.66f18325h1", "HCP1.66f63025h1", "HCP1.66f92775h1",
]
GEM2_COIL_SEP = 1.66  # metres, shared by all six GEM-2 channels above

DEPTHS0 = [0.3, 0.6, 1.0]          # matches ensemble_scenarios.json "2d_real"
N_LAYERS = len(DEPTHS0) + 1
# Constant-in-space-and-time WS truth parameters (matches "2d_real"'s lab_p0)
WS_A, WS_M, WS_SIGMA_W, WS_SIGMA_S = 0.5, 3.0, 500.0, 100.0

THETA_MIN, THETA_MAX = 0.05, 0.45


# ---------------------------------------------------------------------------
# Ground-truth soil-moisture field
# ---------------------------------------------------------------------------

def _field_bbox():
    xs = [e[4] for e in REAL_EM_EPOCHS] + [e[5] for e in REAL_EM_EPOCHS]
    ys = [e[6] for e in REAL_EM_EPOCHS] + [e[7] for e in REAL_EM_EPOCHS]
    return min(xs), max(xs), min(ys), max(ys)


_X0, _X1, _Y0, _Y1 = _field_bbox()


def _raw_field(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    """Unnormalized field shape, evaluated on already-normalized [0,1]
    coordinates. Factored out so the normalizing constant below and
    spatial_field() itself always agree on the exact same formula."""
    return (
        0.5 * np.sin(2.1 * np.pi * xn + 0.4) * np.cos(1.3 * np.pi * yn)
        + 0.3 * np.cos(3.7 * np.pi * (xn - 0.3 * yn))
        - 0.4 * ((xn - 0.65) ** 2 + (yn - 0.35) ** 2)  # broad wetter/drier lobe
    )


def _compute_field_max(n: int = 20000) -> float:
    """Global normalizing constant for spatial_field(), computed ONCE over
    a dense fixed grid spanning the full normalized [0, 1]^2 domain -- not
    over whatever batch of points happens to be passed to spatial_field()
    at call time. This is what makes the field genuinely fixed: a single
    TDR-station call (one point) and a 150-point EM-epoch call now divide
    by the same constant instead of each renormalizing to their own batch."""
    side = int(np.ceil(np.sqrt(n)))
    xn, yn = np.meshgrid(np.linspace(0.0, 1.0, side), np.linspace(0.0, 1.0, side))
    return float(np.max(np.abs(_raw_field(xn.ravel(), yn.ravel()))))


_FIELD_MAX = _compute_field_max()


def spatial_field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Smooth, deterministic spatial heterogeneity in [-1, 1], fixed for
    the whole synthetic twin (soil texture / drainage pattern doesn't
    move in time). Built from a couple of broad sinusoids anchored to the
    real field extent -- not fitted to anything, just a plausible smooth
    pattern with more than one length scale.

    Normalized against the FIXED global _FIELD_MAX (see _compute_field_max),
    not against the min/max of whatever x, y batch is passed in here. This
    is what guarantees TDR stations (called one point at a time) and EM
    epochs (called with tens to hundreds of points) sample the same
    underlying field on the same scale."""
    xn = (x - _X0) / max(_X1 - _X0, 1e-6)
    yn = (y - _Y0) / max(_Y1 - _Y0, 1e-6)
    field = _raw_field(xn, yn)
    return np.clip(field / _FIELD_MAX, -1.0, 1.0)


def seasonal_offset(iso_date: str) -> float:
    """Cosine seasonal cycle in theta-units, peak (wettest) ~mid-January,
    trough (driest) ~mid-July -- matches the real survey cadence spanning
    two full winters and two summers (Aug'20 -> Dec'21)."""
    doy = pd.Timestamp(iso_date).dayofyear
    peak_doy = 15  # mid-January
    return 0.10 * np.cos(2 * np.pi * (doy - peak_doy) / 365.25)


def theta_truth_deterministic(x, y, layer_idx: int, iso_date: str) -> np.ndarray:
    """Noise-free ground-truth volumetric water content at given (x, y),
    one layer, one date. Depth trend: modestly wetter with depth (z_frac
    in [0,1]). Pure function of its inputs -- no RNG involved -- so it can
    be recomputed identically at scoring time, or saved once and reused."""
    z_frac = layer_idx / max(N_LAYERS - 1, 1)
    base = 0.22
    spatial_amp = 0.10
    depth_amp = 0.05
    theta = (
        base
        + spatial_amp * spatial_field(x, y)
        + depth_amp * z_frac
        + seasonal_offset(iso_date)
    )
    return np.clip(theta, THETA_MIN, THETA_MAX)


def theta_truth(x, y, layer_idx: int, iso_date: str, rng: np.random.Generator,
                 noise_std: float = 0.015) -> np.ndarray:
    """Noisy synthetic OBSERVATION of theta_truth_deterministic (what the
    EM forward model / TDR readings actually see). NOT the value to save
    as ground truth -- save theta_truth_deterministic(...) for that, since
    this draws fresh noise from `rng` on every call and can't be
    reproduced after the fact."""
    theta = theta_truth_deterministic(x, y, layer_idx, iso_date)
    theta = theta + rng.normal(0.0, noise_std, size=np.shape(x))
    return np.clip(theta, THETA_MIN, THETA_MAX)


def forward_waxman_smits(theta: np.ndarray) -> np.ndarray:
    """sigma_bulk = (theta**m / a) * (sigma_w + sigma_s)  -- the
    multiplicative "simplified Waxman-Smits" form used throughout this
    codebase (formation factor F = a/theta**m, sigma_bulk = (sigma_w +
    sigma_s) / F).

    VERIFIED against em_methodology.py's actual forward relation: this is
    the exact algebraic inverse of em_methodology.waxman_smits_theta,
    theta = (a * sigma_b / (sigma_w + sigma_s)) ** (1/m)
    -> sigma_b = (theta**m / a) * (sigma_w + sigma_s).
    Matches data_synthetic.py's own forward_waxman_smits, which
    uses the same multiplicative form. An earlier version of this function
    used an ADDITIVE surface-conduction term ((theta**m/a)*sigma_w +
    sigma_s) instead -- that is the inverse of a different equation
    (theta = (a*sigma_b/sigma_w)**(1/m), with sigma_s tacked on
    afterward), NOT of what em_methodology.py's calibration engine
    actually fits. Using the additive form here would have meant the
    synthetic ground truth was generated under a different constitutive
    relationship than the one the pipeline calibrates against, biasing
    every calibrated/uncalibrated RMSE comparison in a way that is
    indistinguishable from a genuine sensor-density effect -- exactly the
    kind of mismatch this synthetic twin exists to avoid. sigma_s does
    still persist as theta -> 0 in this form (it doesn't vanish, it's
    just co-scaled with sigma_w rather than added on unscaled); this
    matches the "simplified Waxman-Smits" form standard in the applied
    EMI/ERT soil-moisture literature this pipeline draws on, not the
    literal Waxman & Smits (1968) S_w**n / S_w**(n-1) split, which
    em_methodology.py's own model does not implement either."""
    theta_c = np.clip(theta, 0.01, 1.0)
    return (theta_c ** WS_M / WS_A) * (WS_SIGMA_W + WS_SIGMA_S)


# ---------------------------------------------------------------------------
# EM survey generation
# ---------------------------------------------------------------------------

def _sample_points(n: int, x0, x1, y0, y1, rng: np.random.Generator):
    x = rng.uniform(x0, x1, size=n)
    y = rng.uniform(y0, y1, size=n)
    return x, y


def generate_em_surveys(outdir: str, scale: float, max_points: int,
                         eca_noise_level: float, seed: int) -> list[dict]:
    em_dir = os.path.join(outdir, "EMI_MFC2_synthetic")
    gt_dir = os.path.join(outdir, "ground_truth")
    os.makedirs(em_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    records = []

    for raw_idx, ddmmyy, iso_date, n_real, x0, x1, y0, y1, z0, z1 in REAL_EM_EPOCHS:
        rng = np.random.default_rng(seed + raw_idx)
        n_points = int(np.clip(round(n_real * scale), 20, max_points))
        x, y = _sample_points(n_points, x0, x1, y0, y1, rng)
        elev = rng.uniform(z0, z1, size=n_points)

        # Deterministic ground truth (no noise) -- saved for scoring, NOT
        # fed to the pipeline. This is the recoverable signal; the noisy
        # theta_layers below (used for forward-modeling EC) additionally
        # includes i.i.d. micro-variability that no inversion could recover
        # anyway, so it isn't a meaningful scoring target.
        theta_layers_det = np.stack(
            [theta_truth_deterministic(x, y, l, iso_date) for l in range(N_LAYERS)],
            axis=1,
        )  # (n_points, n_layers)
        gt_stem = f"em_epoch{raw_idx}_{iso_date}"
        np.save(os.path.join(gt_dir, f"{gt_stem}_theta.npy"), theta_layers_det)
        np.save(os.path.join(gt_dir, f"{gt_stem}_xy.npy"), np.stack([x, y], axis=1))
        np.save(os.path.join(gt_dir, f"{gt_stem}_depths.npy"), np.array(DEPTHS0))

        # theta per layer -> EC per layer -> forward EM per point.
        theta_layers = np.stack(
            [theta_truth(x, y, l, iso_date, rng) for l in range(N_LAYERS)], axis=1
        )  # (n_points, n_layers)
        ec_layers = forward_waxman_smits(theta_layers)  # mS/m

        eca_mS_m = np.zeros((n_points, len(GEM2_COILS)), dtype=float)
        depths_row = np.array(DEPTHS0, dtype=float)
        for i in range(n_points):
            k = Problem()
            models_input = [ec_layers[i:i + 1, :]]
            depths_input = [depths_row[np.newaxis, :]]
            k.models = models_input
            k.depths = depths_input
            res = k.forward(forwardModel="CS", coils=GEM2_COILS,
                             models=models_input, depths=depths_input)
            eca_mS_m[i, :] = res[0].values.flatten()

        if eca_noise_level > 0:
            eca_mS_m *= (1.0 + rng.normal(0.0, eca_noise_level, size=eca_mS_m.shape))

        # ECa [mS/m] -> quadrature ppm, exact inverse of Survey.readDF(unit='ppm')
        q_ppm = np.zeros_like(eca_mS_m)
        for c_idx, coil in enumerate(GEM2_COILS):
            freq = float(coil.split("f")[1].split("h")[0])
            eca_Sm = eca_mS_m[:, c_idx] / 1e3
            q_ppm[:, c_idx] = eca2Q(eca_Sm, GEM2_COIL_SEP, freq) * 1e6

        freq_labels = ["425Hz", "1525Hz", "5325Hz", "18325Hz", "63025Hz", "92775Hz"]
        df = pd.DataFrame({"X": x, "Y": y, "GPSalt": elev})
        for c_idx, lbl in enumerate(freq_labels):
            df[f"Q_{lbl}"] = q_ppm[:, c_idx]

        fname = f"{raw_idx}_ALENTO_{ddmmyy}.csv"
        fpath = os.path.join(em_dir, fname)
        df.to_csv(fpath, sep=";", index=False)

        print(f"[EM] epoch {raw_idx} {iso_date}: {n_points} pts "
              f"(real={n_real}, scale={scale}) -> {fpath}")
        records.append({
            "raw_index": raw_idx, "iso_date": iso_date, "path": fpath,
            "n_points": n_points, "bbox": (x0, x1, y0, y1),
        })

    return records


# ---------------------------------------------------------------------------
# TDR station generation
# ---------------------------------------------------------------------------

def generate_tdr_stations(outdir: str, coord_csv: str, start_date: str,
                           end_date: str, freq_days: int, tdr_noise_std: float,
                           missing_frac: float, seed: int) -> pd.DataFrame:
    tdr_dir = os.path.join(outdir, "daily_MFC_synthetic")
    gt_dir = os.path.join(outdir, "ground_truth")
    os.makedirs(tdr_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    coords = pd.read_csv(coord_csv, delimiter=";", comment="[")
    coords.columns = coords.columns.str.strip()
    coords["Pos"] = coords["Pos"].str.strip()

    dates = pd.date_range(start_date, end_date, freq=f"{freq_days}D")
    np.save(os.path.join(gt_dir, "tdr_dates.npy"),
            np.array([d.strftime("%Y-%m-%d") for d in dates]))
    rng = np.random.default_rng(seed + 9000)

    for _, row in coords.iterrows():
        pos = row["Pos"]  # "Unit N"
        unit_num = int(pos.replace("Unit", "").strip())
        x, y = float(row["x"]), float(row["y"])

        # NOTE (unverified assumption): 15cm TDR -> layer_idx=0 (0-0.3m),
        # 30cm TDR -> layer_idx=1 (0.3-0.6m) treats each probe as a point
        # sample AT that depth. Real TDR rods more often integrate the
        # average moisture from the surface down to rod length (so a
        # "30cm" probe = 0-30cm average, not the 30-60cm layer). Confirm
        # this matches how em_methodology.py associates TDR depths with
        # its own layer boundaries before trusting truth-vs-recovered
        # comparisons at the 30cm depth.
        #
        # theta_truth is vectorized over x/y but takes a single scalar
        # iso_date, so dates are looped individually here -- this is the
        # only call, consuming rng draws once per date/layer (the removed
        # duplicate call above used to silently burn an extra len(dates)
        # draws from `rng` before this loop even started).
        theta15 = np.array([
            theta_truth(np.array([x]), np.array([y]), 0, d.strftime("%Y-%m-%d"),
                        rng, tdr_noise_std)[0] for d in dates
        ])
        theta30 = np.array([
            theta_truth(np.array([x]), np.array([y]), 1, d.strftime("%Y-%m-%d"),
                        rng, tdr_noise_std)[0] for d in dates
        ])

        # Deterministic ground truth (no noise), saved for scoring.
        theta15_det = np.array([
            theta_truth_deterministic(np.array([x]), np.array([y]), 0,
                                       d.strftime("%Y-%m-%d"))[0] for d in dates
        ])
        theta30_det = np.array([
            theta_truth_deterministic(np.array([x]), np.array([y]), 1,
                                       d.strftime("%Y-%m-%d"))[0] for d in dates
        ])
        np.save(os.path.join(gt_dir, f"tdr_unit{unit_num:02d}_theta15.npy"), theta15_det)
        np.save(os.path.join(gt_dir, f"tdr_unit{unit_num:02d}_theta30.npy"), theta30_det)

        df = pd.DataFrame({
            "Year": dates.year, "Month": dates.month, "Day": dates.day,
            "Hour": 12, "Min": 0, "Sec": 0,
            "SWC_15cm_Avg": theta15 * 100.0,  # stored as %, matches real-file convention
            "SWC_30cm_Avg": theta30 * 100.0,
        })

        if missing_frac > 0:
            mask = rng.random(len(df)) < missing_frac
            df.loc[mask, ["SWC_15cm_Avg", "SWC_30cm_Avg"]] = np.nan

        fname = f"MTX{unit_num:02d}_synthetic.xlsx"
        fpath = os.path.join(tdr_dir, fname)
        df.to_excel(fpath, index=False)
        print(f"[TDR] {pos} ({unit_num}) -> {fpath}  ({len(dates)} readings, "
              f"{int((df['SWC_15cm_Avg'].isna()).sum())} gaps @15cm)")

    return coords


# ---------------------------------------------------------------------------
# Scenario JSON
# ---------------------------------------------------------------------------

def write_scenario_json(outdir: str, em_dir: str, tdr_dir: str, coord_file: str) -> str:
    scenario = {
        "mfc2_real_synthetic_mirror": {
            "description": (
                "Synthetic twin of the real MFC2 deployment: real EM survey "
                "dates/point-count asymmetry (incl. the degraded 15/12/2020 "
                "epoch) and real 24-station TDR coordinates at 15/30 cm, "
                "with known ground truth for scoring."
            ),
            "output_dir": "mfc2_outputs/mfc2_real_synthetic_mirror",
            "data_source": "real",
            "em_data_dir": em_dir,
            "em_unit": "ppm",
            "tdr_data_dir": tdr_dir,
            "coord_file": coord_file,
            "time_tolerance_days": 3.0,
            "realization_index": 0,
            "depths0": DEPTHS0,
            "ws_fixed_params": {"a": True, "m": True, "sigma_w": False, "sigma_s": False},
            "lab_p0": [WS_A, WS_M, WS_SIGMA_W, WS_SIGMA_S],
            "fixed_depths": True,
            "param_grid": {"method": ["MCMC"], "forwardModel": ["CS"]},
            "alpha_range_log": [0.001, 5.0, 9],
            "cond_ranges": [[0.01, 300.0]] * N_LAYERS,
            "gamma": 0.0,
            "window_m": 3.0,
            "spatial_match_tolerance_m": 3.0,
            "tdr_noise_std": 0.015,
            "eca_noise_level": 0.03,
            "cache_tag": "mfc2_real_synthetic_mirror",
        }
    }
    import json
    path = os.path.join(outdir, "mfc2_real_synthetic_mirror.json")
    with open(path, "w") as f:
        json.dump(scenario, f, indent=2)
    print(f"[SCENARIO] wrote {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coord-csv", required=True,
                     help="Path to SOILNET_MFC2_coordinates_sensors.csv (real, reused as-is).")
    ap.add_argument("--outdir", default="./synthetic_real_mirror")
    ap.add_argument("--scale", type=float, default=0.01,
                     help="Fraction of the REAL per-epoch point count to simulate "
                          "(default 0.01 -- fast smoke-test scale; use up to 1.0 "
                          "for a full-resolution run in a real compute environment).")
    ap.add_argument("--max-points", type=int, default=150,
                     help="Hard cap on points per EM epoch regardless of --scale "
                          "(keeps the smoke test fast).")
    ap.add_argument("--eca-noise", type=float, default=0.03)
    ap.add_argument("--tdr-noise-std", type=float, default=0.015)
    ap.add_argument("--tdr-freq-days", type=int, default=1,
                     help="Days between synthetic TDR readings (1 = daily, matching "
                          "real SoilNet loggers).")
    ap.add_argument("--tdr-missing-frac", type=float, default=0.02,
                     help="Fraction of synthetic TDR readings randomly set to NaN, "
                          "mirroring real logger gaps.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    em_records = generate_em_surveys(
        args.outdir, scale=args.scale, max_points=args.max_points,
        eca_noise_level=args.eca_noise, seed=args.seed,
    )
    start_date = REAL_EM_EPOCHS[0][2]
    end_date = REAL_EM_EPOCHS[-1][2]
    generate_tdr_stations(
        args.outdir, args.coord_csv, start_date, end_date,
        freq_days=args.tdr_freq_days, tdr_noise_std=args.tdr_noise_std,
        missing_frac=args.tdr_missing_frac, seed=args.seed,
    )

    em_dir = os.path.join(args.outdir, "EMI_MFC2_synthetic")
    tdr_dir = os.path.join(args.outdir, "daily_MFC_synthetic")
    write_scenario_json(args.outdir, em_dir, tdr_dir, os.path.abspath(args.coord_csv))

    print(f"\n[DONE] {len(em_records)} EM epochs, "
          f"{len(pd.read_csv(args.coord_csv, delimiter=';', comment='['))} TDR stations "
          f"written under '{args.outdir}'.")


if __name__ == "__main__":
    main()
