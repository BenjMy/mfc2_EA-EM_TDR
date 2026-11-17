#!/usr/bin/env python3
"""TDR-pinned experiment, two passes (Level 2 of analysis_pinned_ceiling.py).

Pass 1 : the normal pipeline on ``--scenario`` (all layers free), calibrating
         Waxman-Smits on the (train) TDR probes.
Derive : EC of the pinned horizon at every EM point from the TDR theta and the
         calibrated WS (``pin_from_tdr``) -> ``<pass1>/output_model/pin_ec_tdr.npy``.
Pass 2 : ``<scenario>_pintdr`` = the scenario with ``pin_layers`` /
         ``pin_source='file'`` pointing at that file, run through the same pipeline.

    python run_pinned_tdr.py --scenario 1d_surface_free2 --mode 1d --num-swc-positions 5
    python run_pinned_tdr.py --scenario 1d_surface_free2 --pass1-dir <existing run dir>   # skip pass 1
    python run_pinned_tdr.py --scenario 1d_surface_free2 --derive-only --pass1-dir <dir>

Not done here: per-member EC draws (honest uncertainty, Level 3).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

import pin_from_tdr

HERE = os.path.dirname(os.path.abspath(__file__))
PIN_FILE = "pin_ec_tdr.npy"
NEEDED = ("ws_params", "tdr_train_theta", "tdr_train_positions", "tdr_train_times",
          "tdr_train_layers", "synthetic_x_coords", "synthetic_y_coords",
          "synthetic_time_lapse_theta")


def build_pin_file(output_model_dir: str, horizon: int = 0, ec_bounds=None,
                   out_path: str | None = None) -> str:
    """Derive the pinned-horizon EC from a finished pass-1 ``output_model`` folder."""
    d = output_model_dir
    missing = [n for n in NEEDED if not os.path.exists(os.path.join(d, n + ".npy"))]
    if missing:
        raise FileNotFoundError(
            f"{d} lacks {missing}: pass 1 must be a run made with this version "
            f"(it saves the calibration arrays in Step 5).")
    ld = lambda n: np.load(os.path.join(d, n + ".npy"))      # noqa: E731
    params, layers = ld("ws_params"), ld("tdr_train_layers")
    is_cal = os.path.join(d, "is_calibrated.npy")
    if os.path.exists(is_cal) and not np.load(is_cal)[horizon]:
        raise ValueError(f"horizon {horizon} is not calibrated: nothing to pin to TDR.")
    sel = layers == horizon
    x, y = ld("synthetic_x_coords"), ld("synthetic_y_coords")
    n_time, n_layers = ld("synthetic_time_lapse_theta").shape[1], params.shape[0]
    ec = pin_from_tdr.derive_pin_ec(
        ld("tdr_train_theta")[sel], ld("tdr_train_positions")[sel], params[horizon], x,
        point_y=None if np.allclose(y, y[0]) else y,
        obs_times=ld("tdr_train_times")[sel], n_time=n_time, ec_bounds=ec_bounds)
    full = np.full((n_time, len(x), n_layers), np.nan)
    full[:, :, horizon] = ec
    out_path = out_path or os.path.join(d, PIN_FILE)
    np.save(out_path, full)
    return out_path


def latest_run_dir(scenario: str, config: str | None = None) -> str:
    import scenario_loader as sl
    cfg = sl.load_scenario(scenario, config) if config else sl.load_scenario(scenario)
    runs = sorted(glob.glob(os.path.join(HERE, cfg.output_dir, "[0-9]*_*")))
    if not runs:
        raise FileNotFoundError(f"no run folder under {cfg.output_dir}")
    return runs[-1]


def write_pass2_config(scenario: str, pin_file: str, horizon: int, out_json: str,
                       base_config: str | None = None) -> str:
    import scenario_loader as sl
    base = base_config or os.path.join(HERE, "ensemble_scenarios.json")
    raw = json.load(open(base, encoding="utf-8"))
    cfg = sl.load_scenario(scenario, base_config) if base_config else sl.load_scenario(scenario)
    name = f"{scenario}_pintdr"
    n_layers = len(cfg.depths0) + 1
    raw[name] = {
        "extends": scenario,
        "description": f"Pass 2 of the TDR-pinned experiment: '{scenario}' with horizon "
                       f"{horizon} pinned to the EC derived from TDR + calibrated WS ({pin_file}).",
        "output_dir": cfg.output_dir + "_pintdr", "cache_tag": f"{cfg.cache_tag}_pintdr",
        "pin_layers": [i == horizon for i in range(n_layers)],
        "pin_source": "file", "pin_ec_file": os.path.abspath(pin_file),
    }
    json.dump(raw, open(out_json, "w", encoding="utf-8"), indent=2)
    return name


def _run(scenario, config, extra):
    cmd = [sys.executable, os.path.join(HERE, "run_pipeline.py"), "--scenario", scenario] + extra
    if config:
        cmd += ["--config", config]
    print("[pinned-tdr] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scenario", required=True, help="pass-1 scenario (all layers free)")
    ap.add_argument("--horizon", type=int, default=0, help="horizon to pin (default 0)")
    ap.add_argument("--pass1-dir", help="finished pass-1 run folder (skips pass 1)")
    ap.add_argument("--derive-only", action="store_true", help="stop after writing the pin file")
    ap.add_argument("--config", help="scenario JSON (default ensemble_scenarios.json)")
    ap.add_argument("--ec-max", type=float, default=None, help="clip the pin to <= this mS/m")
    a, extra = ap.parse_known_args(argv)        # unknown args (--mode, --num-swc-positions) go to run_pipeline
    if not a.pass1_dir:
        _run(a.scenario, a.config, extra)
    p1 = a.pass1_dir or latest_run_dir(a.scenario, a.config)
    pin = build_pin_file(os.path.join(p1, "output_model"), a.horizon,
                         (0.0, a.ec_max) if a.ec_max else None)
    print(f"[pinned-tdr] pin EC written: {pin}")
    if a.derive_only:
        return
    cfg_json = os.path.join(p1, "pass2_scenarios.json")
    name = write_pass2_config(a.scenario, pin, a.horizon, cfg_json, a.config)
    _run(name, cfg_json, extra)


if __name__ == "__main__":
    main()
