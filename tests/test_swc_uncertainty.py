import types
import numpy as np
import swc_uncertainty as su


def _fit(a=0.5, m=2.0, sw=500.0, ss=100.0, rmse=0.02, n_layers=3):
    hf = types.SimpleNamespace(a=a, m=m, sigma_w=sw, sigma_s=ss)
    return types.SimpleNamespace(horizon_fits=[hf] * n_layers, rmse=rmse)


def _ec_for(theta, fit):   # inverse Waxman-Smits for building test data
    hf = fit.horizon_fits[0]
    return theta ** hf.m * (hf.sigma_w + hf.sigma_s) / hf.a


def test_ec_to_theta_roundtrip_and_nan():
    f = _fit()
    th = np.array([[0.1, 0.3, 0.4]])
    ec = _ec_for(th, f)
    np.testing.assert_allclose(su.ec_grid_to_theta(ec, f), th, rtol=1e-6)
    ec[0, 1] = np.nan
    assert np.isnan(su.ec_grid_to_theta(ec, f)[0, 1])


def test_identical_members_zero_uncertainty():
    f = _fit()
    ec = np.broadcast_to(_ec_for(np.full((2, 4, 3), 0.3), f), (5, 2, 4, 3)).copy()
    u = su.theta_ensemble_uncertainty(ec, np.arange(5), 5, [_fit()] * 5, f)
    assert np.allclose(u["std_total"], 0) and np.allclose(u["std_ec"], 0)
    np.testing.assert_allclose(u["theta_mean"], 0.3, rtol=1e-6)


def test_ec_spread_known_value_and_k1():
    f = _fit()
    ths = np.array([0.2, 0.3, 0.4])            # equal rmse -> equal weights
    ec = np.stack([_ec_for(np.full((4, 3), t), f) for t in ths])
    fits = [_fit(rmse=0.02)] * 3
    u = su.theta_ensemble_uncertainty(ec, np.arange(3), 3, fits, f)
    np.testing.assert_allclose(u["std_ec"], np.std(ths, ddof=1), rtol=1e-6)
    np.testing.assert_allclose(u["std_total"], u["std_ec"], rtol=1e-6)  # same fits
    assert np.allclose(u["std_ws"], 0)
    u1 = su.theta_ensemble_uncertainty(ec, np.arange(3), 1, fits, f)
    assert np.allclose(u1["std_total"], 0)


def test_calibration_spread_adds_to_total():
    f = _fit()
    ec = np.broadcast_to(_ec_for(np.full((4, 3), 0.3), f), (3, 4, 3)).copy()
    fits = [_fit(m=2.0), _fit(m=2.5), _fit(m=3.0)]   # members disagree on m
    u = su.theta_ensemble_uncertainty(ec, np.arange(3), 3, fits, f)
    assert np.allclose(u["std_ec"], 0)
    assert np.all(u["std_total"] > 0) and np.allclose(u["std_ws"], u["std_total"])


def test_weights_favour_best_and_sum_to_one():
    w = su.rmse_weights([0.01, 0.02, 0.05])
    assert np.isclose(w.sum(), 1) and w[0] > w[1] > w[2]
    assert np.allclose(su.rmse_weights([0.03, 0.03]), 0.5)


def test_ranking_selects_top_k_only():
    f = _fit()
    ths = np.array([0.30, 0.30, 0.45])
    ec = np.stack([_ec_for(np.full((4, 3), t), f) for t in ths])
    u = su.theta_ensemble_uncertainty(ec, np.array([0, 1, 2]), 2, [_fit()] * 3, f)
    assert np.allclose(u["std_ec"], 0)          # outlier member 2 excluded


def test_coverage():
    rng = np.random.default_rng(0)
    truth = np.zeros(20000)
    mean = rng.normal(0, 1, truth.shape)
    c = su.uncertainty_coverage(truth, mean, np.ones_like(mean))
    assert abs(c["within_1sigma"] - 0.68) < 0.02 and abs(c["within_2sigma"] - 0.95) < 0.02


def test_plot_writes_png(tmp_path):
    th = np.full((2, 4, 3), 0.3); sd = np.full_like(th, 0.05)
    p = su.plot_swc_uncertainty(th, sd, sd * 0.5, [0.1, 0.4, 0.8], str(tmp_path),
                                theta_true=th + 0.01, coverage={"within_1sigma": .7, "within_2sigma": .9})
    import os; assert os.path.getsize(p) > 1000
    # 2-D (single survey) input also accepted
    su.plot_swc_uncertainty(th[0], sd[0], sd[0], [0.1, 0.4, 0.8], str(tmp_path))
