import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pin_from_tdr as pin  # noqa: E402
import run_pinned_tdr as rp  # noqa: E402
import scenario_loader as sl  # noqa: E402

PARAMS = np.array([[0.5, 1.8, 500., 10.], [0.5, 2.0, 500., 5.]] + [[0.5, 2.0, 500., 5.]] * 2)


def _fake_pass1(tmp_path, calibrated=(True, False, False, False)):
    d = tmp_path / "output_model"
    d.mkdir()
    x = np.arange(0., 20., 2.)
    np.save(d / "ws_params.npy", PARAMS)
    np.save(d / "tdr_train_theta.npy", np.array([.15, .30, .20, .25, .1, .2]))
    np.save(d / "tdr_train_positions.npy", np.array([1, 7, 1, 7, 1, 7]))
    np.save(d / "tdr_train_times.npy", np.array([0, 0, 1, 1, 2, 2]))
    np.save(d / "tdr_train_layers.npy", np.zeros(6, dtype=int))
    np.save(d / "synthetic_x_coords.npy", x)
    np.save(d / "synthetic_y_coords.npy", np.zeros_like(x))
    np.save(d / "synthetic_time_lapse_theta.npy", np.zeros((4, 3, 10, 4)))
    np.save(d / "is_calibrated.npy", np.array(calibrated))
    return str(d)


def test_build_pin_file_shape_and_values(tmp_path):
    path = rp.build_pin_file(_fake_pass1(tmp_path))
    arr = np.load(path)
    assert arr.shape == (3, 10, 4)                      # (n_time, n_pts, n_layers)
    assert np.isnan(arr[:, :, 1:]).all()                # only the pinned layer is filled
    expect = pin.ec_from_theta([.15, .30], *PARAMS[0])
    assert np.allclose(arr[0, [1, 7], 0], expect)


def test_uncalibrated_horizon_refused(tmp_path):
    d = _fake_pass1(tmp_path)
    with pytest.raises(ValueError):
        rp.build_pin_file(d, horizon=1)


def test_old_pass1_without_calibration_arrays_is_explained(tmp_path):
    (tmp_path / "output_model").mkdir()
    with pytest.raises(FileNotFoundError, match="ws_params"):
        rp.build_pin_file(str(tmp_path / "output_model"))


def test_pass2_config_loads_and_pins(tmp_path):
    cfg_json = str(tmp_path / "s.json")
    name = rp.write_pass2_config("1d_surface_free2", "/x/pin.npy", 0, cfg_json)
    cfg = sl.load_scenario(name, cfg_json)
    base = sl.load_scenario("1d_surface_free2")
    assert cfg.pin_layers == [True, False, False, False]
    assert cfg.pin_source == "file" and cfg.pin_ec_file == "/x/pin.npy"
    assert cfg.output_dir != base.output_dir and cfg.depths0 == base.depths0


def test_file_source_requires_path():
    raw = json.load(open(os.path.join(ROOT, "ensemble_scenarios.json")))
    raw["bad"] = {"extends": "1d_surface_free2", "pin_layers": [True, False, False, False],
                  "pin_source": "file"}
    p = os.path.join(os.path.dirname(__file__), "_tmp_bad.json")
    json.dump(raw, open(p, "w"))
    try:
        with pytest.raises(ValueError, match="pin_ec_file"):
            sl.load_scenario("bad", p)
    finally:
        os.remove(p)
