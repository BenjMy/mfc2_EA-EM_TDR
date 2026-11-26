"""Truth Waxman-Smits curve in the Step 2 EC-theta plots (synthetic runs)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import em_methodology as em
import step2_petro as s2

T = {"a": 0.5, "m": 3.0, "sigma_w": 500.0, "sigma_s": 100.0}


def _labels(monkeypatch, tmp_path, **kwargs):
    """Legend labels and truth lines of a two-horizon petro_calibration.png."""
    theta = np.array([0.1, 0.2, 0.3, 0.35])
    ec = theta ** 3.0 / 0.5 * 600.0
    hz = [em.HorizonWaxmanSmitsParams(horizon=h, a=1.0, m=2.5, sigma_w=400.0, sigma_s=80.0, rmse=0.01,
                                      predicted_theta=theta[:2], observed_theta=theta[:2], is_calibrated=True)
          for h in (0, 1)]
    monkeypatch.setattr(s2.plt, "close", lambda *a: None)
    s2.plot_petrophysical_calibration(ec, theta, em.PetroFit(model_index=0, horizon_fits=hz),
                                      output_path=str(tmp_path / "p.png"), **kwargs)
    lines = [l for l in plt.gca().lines if l.get_label().startswith("Truth")]
    out = [l.get_label() for l in lines], lines
    plt.close("all")
    return out


def test_no_truth_on_real_data(monkeypatch, tmp_path):
    assert _labels(monkeypatch, tmp_path)[0] == []
    assert _labels(monkeypatch, tmp_path, ws_truth={})[0] == []


def test_homogeneous_truth_is_one_curve_of_the_true_relationship(monkeypatch, tmp_path):
    labels, lines = _labels(monkeypatch, tmp_path, ws_truth={0: T, 1: T, 2: T})
    assert len(labels) == 1 and labels[0].startswith("Truth Hz 0,1 (a=0.50, m=3.00")
    x, y = lines[0].get_data()
    assert y == pytest.approx(em.waxman_smits_theta(np.asarray(x), 0.5, 3.0, 500.0, 100.0))


def test_layered_and_baseline_only_truth(monkeypatch, tmp_path):
    labels, _ = _labels(monkeypatch, tmp_path, ws_truth={"0": T, "1": {**T, "a": 0.35}},
                        ws_truth_baseline_only=True)
    assert len(labels) == 2 and all(l.startswith("Truth baseline Hz") for l in labels)
