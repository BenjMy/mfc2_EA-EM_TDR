"""Slow tier: run the pipeline on the synthetic MFC2 twin through the scenario matrix.

Off by default (marker ``slow``); run with ``pytest -m slow tests/test_pipeline_mfc2_synthetic.py``.
Needs the real emagpy. Each scenario of ``ensemble_scenarios_synthetic_mfc2.json`` is run as a
subprocess (``--quick --quick-epochs 3``) on a small dataset with extra EM points next to every
TDR station, so the same code path as real data is exercised.

Checked per scenario: exit code 0, report ends SUCCESS, run_manifest.json and theta_calc.npy
exist and theta_calc has finite values. The calibrate -> map split is tested on the base scenario.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.slow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "ensemble_scenarios_synthetic_mfc2.json")
DATASET = "mfc2_synthetic"

SCENARIOS = [k for k in json.load(open(CONFIG)) if not k.startswith("_")]
# FSlin inversion on CS-made data would be an inverse-crime-free test only with a FSlin truth.
NEEDS_FSLIN_TRUTH = {"mfc2_syn_fslin"}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    pytest.importorskip("emagpy")
    import generate_synthetic_real_mirror as gen
    root = tmp_path_factory.mktemp("mfc2_slow")
    paths = {}
    for tag, fm in (("cs", "CS"), ("fslin", "FSlin")):
        out = gen.build_dataset(str(root / f"data_{tag}"), scale=0.002, max_points=40, seed=42,
                                tdr_freq_days=14, n_stations=8, near_station_points=2,
                                truth_forward_model=fm, output_dir=str(root / "runs"))
        dfile = root / f"datasets_{tag}.json"
        gen.register_built_dataset(out, name=DATASET, datasets_file=str(dfile))
        paths[tag] = (out, str(dfile))
    # scenario file with output_dir redirected to the temp dir
    cfg = json.load(open(CONFIG))
    for name, sc in cfg.items():
        if not name.startswith("_"):
            sc["output_dir"] = str(root / "runs" / name)
    cfg_path = root / "scenarios.json"
    cfg_path.write_text(json.dumps(cfg))
    return dict(root=root, config=str(cfg_path), paths=paths)


def _run(synth, scenario, *extra, truth="cs"):
    _, dfile = synth["paths"][truth]
    env = dict(os.environ, EA_EM_DATASETS_FILE=dfile, MPLBACKEND="Agg")
    cmd = [sys.executable, os.path.join(ROOT, "run_pipeline.py"), "--data-source", "real", "--scenario", scenario,
           "--config", synth["config"], "--dataset", DATASET, *extra]
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=3600)


def _run_dirs(synth, scenario):
    return sorted(glob.glob(os.path.join(str(synth["root"]), "runs", scenario + "*")))


def _find(synth, scenario, pattern):
    hits = []
    for d in _run_dirs(synth, scenario):
        hits += glob.glob(os.path.join(d, "**", pattern), recursive=True)
    return sorted(hits)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_runs_to_success(synth, scenario):
    truth = "fslin" if scenario in NEEDS_FSLIN_TRUTH else "cs"
    r = _run(synth, scenario, "--quick", "--quick-epochs", "3", truth=truth)
    assert r.returncode == 0, r.stderr[-3000:] + r.stdout[-2000:]

    reports = _find(synth, scenario, "pipeline_report_*.md")
    assert reports, "no pipeline report written"
    assert "SUCCESS" in open(reports[-1], encoding="utf-8").read()

    assert _find(synth, scenario, "run_manifest.json"), "run_manifest.json missing"

    thetas = _find(synth, scenario, "theta_calc.npy")
    assert thetas, "theta_calc.npy missing"
    theta = np.load(thetas[-1])
    assert np.isfinite(theta).any(), "theta_calc has no finite value"
    assert np.nanmin(theta) >= 0.0 and np.nanmax(theta) <= 1.0


def test_calibrate_then_map_split(synth):
    base = "mfc2_syn_base"
    cal = _run(synth, base, "--stage", "calibrate")
    assert cal.returncode == 0, cal.stderr[-3000:] + cal.stdout[-2000:]
    calib = _find(synth, base, "stage_calibration.json")
    assert calib, "stage_calibration.json missing after --stage calibrate"
    info = json.load(open(calib[-1]))
    assert info, "empty stage_calibration.json"

    mp = _run(synth, base, "--stage", "map", "--calibration", calib[-1])
    assert mp.returncode == 0, mp.stderr[-3000:] + mp.stdout[-2000:]
    assert _find(synth, base, "theta_calc.npy"), "map stage wrote no theta_calc.npy"
