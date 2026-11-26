"""Tests of the pure functions of ws_param_check.py and diagnose_ec_inversion.py (no inversion run)."""
import numpy as np
import pytest

import ws_param_check as wpc


# ----------------------------------------------------------------------------
# ws_param_check
# ----------------------------------------------------------------------------
def test_theta_from_ec_inverts_ec_bulk():
    th = np.linspace(0.05, 0.45, 9)
    ec = wpc.ec_bulk(th, 0.5, 3.0, 500.0, 100.0)
    np.testing.assert_allclose(wpc.theta_from_ec(ec, 0.5, 3.0, 500.0, 100.0), th, rtol=1e-12)


def test_a_sigma_w_sigma_s_are_degenerate_through_C():
    """Same C = a/(sigma_w+sigma_s) and m  ->  identical EC (why a, sw, ss cannot be told apart)."""
    d = wpc.degeneracy_check((0.5, 3.0, 500.0, 100.0), (1.0, 3.0, 1100.0, 100.0))
    assert d < 1e-9
    # a different m is not degenerate
    assert wpc.degeneracy_check((0.5, 3.0, 500.0, 100.0), (0.5, 2.0, 500.0, 100.0)) > 1.0


def test_baseline_floor_theta_matches_plan():
    """Plan section 6: with (0.5, 3, 500, 100) any theta below ~0.094 is at the 1 mS/m floor."""
    th = wpc.theta_at_ec(wpc.EC_FLOOR, 0.5, 3.0, 500.0, 100.0)
    assert th == pytest.approx(0.094, abs=5e-4)


def test_param_effect_sign_and_order():
    eff = wpc.param_effect(0.5, 3.0, 500.0, 100.0)
    assert eff["m+10%"] > 0 > eff["m-10%"]          # larger m -> larger theta for the same EC
    assert eff["a+10%"] > 0 > eff["a-10%"]
    assert eff["sw+10%"] < 0 < eff["sw-10%"]
    # m is the strongest lever, sigma_s the weakest
    assert abs(eff["m+10%"]) > abs(eff["a+10%"]) > abs(eff["ss+10%"])


# ----------------------------------------------------------------------------
# diagnose_ec_inversion (needs data_synthetic -> emagpy)
# ----------------------------------------------------------------------------
pytest.importorskip("emagpy")
import diagnose_ec_inversion as dei  # noqa: E402


def test_resample_identity_for_fixed_depths():
    ec = np.array([[2.0, 5.0, 10.0, 30.0], [1.0, 1.0, 4.0, 8.0]])
    depths = np.tile(dei.DEPTHS0, (2, 1))
    np.testing.assert_allclose(dei.resample_to_truth_layers(ec, depths), ec)


def test_resample_is_depth_weighted_when_interface_moves():
    # one point; first interface at 0.2 instead of 0.3: truth layer 0 (0-0.3 m) mixes
    # 0.2 m of layer 0 (EC 2) and 0.1 m of layer 1 (EC 8)  ->  (0.2*2 + 0.1*8) / 0.3 = 4
    ec = np.array([[2.0, 8.0, 8.0, 8.0]])
    depths = np.array([[0.2, 0.6, 1.0]])
    out = dei.resample_to_truth_layers(ec, depths)
    assert out[0, 0] == pytest.approx(4.0)
    assert out[0, 1] == pytest.approx(8.0)


def test_layer_metrics_perfect_and_biased():
    true = np.array([[2.0, 5.0], [4.0, 10.0], [8.0, 20.0]])
    perfect = dei.layer_metrics(true, true)
    assert all(r["ratio"] == pytest.approx(1.0) and r["rmse_log10"] == pytest.approx(0) for r in perfect)
    biased = dei.layer_metrics(true * 5, true)
    assert all(r["ratio"] == pytest.approx(5.0) for r in biased)
    assert all(r["corr_log"] == pytest.approx(1.0) for r in biased)   # bias, not scatter


def test_truth_reproduces_generated_eca():
    """Generator and EMagPy forward agree: the control that rules out a convention mismatch."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        np.random.seed(0)
        csv, ec, eca, _ = dei.make_truth(tmp, seed=0, noise=0.0, t_idx=0)
        sim = dei.forward_eca(ec, np.tile(dei.DEPTHS0, (ec.shape[0], 1)), "CS")
        np.testing.assert_allclose(sim, eca, rtol=1e-6)
