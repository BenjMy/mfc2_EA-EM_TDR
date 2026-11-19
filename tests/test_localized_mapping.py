"""localized_mapping.py + steps.step5_selection.run_step5_localized_mapping.

Pure numpy (emagpy is stubbed by conftest). The synthetic horizon has exact Waxman-Smits
data and the prior equal to the truth, so the localized optimiser sits at its minimum and
the expected maps are known in closed form.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

import localized_mapping as lmap
from step4_select import SelectionOutput
from steps import step5_selection as s5

TRUTH = (1.0, 2.0, 100.0, 20.0)          # a, m, sigma_w, sigma_s
X = np.arange(11) * 10.0                 # 11 points, 10 m apart
EC = np.linspace(20.0, 120.0, 11)        # mS/m


def ws_theta(ec, a, m, sw, ss):
    return (a * np.asarray(ec, dtype=float) / (sw + ss)) ** (1.0 / m)


def global_fit(params=TRUTH):
    a, m, sw, ss = params
    return SimpleNamespace(horizon_fits=[SimpleNamespace(a=a, m=m, sigma_w=sw, sigma_s=ss, is_calibrated=True)])


class Rep:
    def __init__(self):
        self.logs, self.metrics = [], {}

    def log(self, msg):
        self.logs.append(msg)

    def metric(self, k, v):
        self.metrics[k] = v


def build(avg, positions, times=None, bandwidth=30.0, gfit=None, p0=TRUTH, y=None):
    positions = list(positions)
    times = [0] * len(positions) if times is None else list(times)
    ec_at = [avg[t, p, 0] if avg.ndim == 3 else avg[p, 0] for t, p in zip(times, positions)]
    theta_obs = ws_theta(ec_at, *TRUTH)
    return lmap.build_localized_mapping(
        avg, gfit or global_fit(), bandwidth=bandwidth, point_x=X, point_y=y,
        sample_positions=positions, sample_times=times, theta_obs=theta_obs,
        probe_layer_indices=[0] * len(positions), lab_p0=np.array(p0, dtype=float))


# ---------------------------------------------------------------- resolve
def test_resolve_default_is_localized_with_bandwidth():
    assert lmap.resolve_map_calibration(None, 5.0) == ("localized", "")
    assert lmap.resolve_map_calibration("localized", 5.0) == ("localized", "")


def test_resolve_downgrades_without_bandwidth():
    mode, why = lmap.resolve_map_calibration(None, None)
    assert mode == "global" and "bandwidth" in why


def test_resolve_global_is_global_and_silent():
    assert lmap.resolve_map_calibration("global", 5.0) == ("global", "")
    assert lmap.resolve_map_calibration("global", None) == ("global", "")


def test_resolve_rejects_unknown_mode():
    with pytest.raises(ValueError):
        lmap.resolve_map_calibration("nearest", 5.0)


# ---------------------------------------------------------------- mapping
def test_exact_data_gives_the_closed_form_map():
    avg = EC[:, None].copy()                                  # (n_points, 1)
    m = build(avg, positions=[1, 3, 5, 7])
    assert m.theta.shape == avg.shape
    assert np.allclose(m.theta[:, 0], ws_theta(EC, *TRUTH), rtol=1e-6)


def test_points_without_a_probe_in_support_use_the_global_fit():
    avg = EC[:, None].copy()
    other = (1.3, 2.2, 90.0, 15.0)                            # global differs from the localized fits
    m = build(avg, positions=[0, 1], bandwidth=5.0, gfit=global_fit(other))
    assert not m.fallback[0, 0] and not m.fallback[0, 1]      # probes themselves: localized
    assert m.fallback[0, 5:].all()                            # >= 40 m from any probe, support = 10 m
    assert np.allclose(m.params[:, 0, 5:], np.array(other)[:, None])
    assert m.fallback_fraction()[0] == pytest.approx(float(m.fallback[0].mean()))
    # the fallback is the GLOBAL fit, not the lab prior
    assert np.allclose(m.theta[5:, 0], ws_theta(EC[5:], *other), rtol=1e-9)


def test_time_lapse_model_keeps_its_shape_and_time_axis():
    avg = np.stack([EC, 1.2 * EC, 0.8 * EC])[:, :, None]      # (3, n_points, 1)
    m = build(avg, positions=[2, 4, 6], times=[0, 1, 2])
    assert m.theta.shape == avg.shape
    for t, f in enumerate((1.0, 1.2, 0.8)):
        assert np.allclose(m.theta[t, :, 0], ws_theta(f * EC, *TRUTH), rtol=1e-6)


def test_nan_ec_stays_nan_and_does_not_poison_other_points():
    avg = EC[:, None].copy()
    avg[9, 0] = np.nan
    m = build(avg, positions=[1, 3, 5, 7])
    assert np.isnan(m.theta[9, 0])
    assert np.isfinite(np.delete(m.theta[:, 0], 9)).all()


def test_two_dimensional_distance_is_used_when_y_is_given():
    avg = EC[:, None].copy()
    y = np.zeros(11)
    y[5:] = 500.0                                             # far away in y only
    m = build(avg, positions=[0, 1, 2], bandwidth=30.0, y=y)
    assert not m.fallback[0, :3].any()
    assert m.fallback[0, 5:].all()                            # close in x to nothing, far in y


def test_localized_follows_a_heterogeneous_field_better_than_one_set():
    # left probes obey a=0.8, right probes a=1.6; the "global" set is the mid value
    rng_ec = np.array([30.0, 60.0, 90.0])
    pos = [0, 1, 2, 8, 9, 10]
    avg = np.zeros((11, 1))
    avg[pos, 0] = np.concatenate([rng_ec, rng_ec])
    a_true = np.array([0.8] * 3 + [1.6] * 3)
    theta_obs = ws_theta(avg[pos, 0], 1.0, 2.0, 100.0, 20.0) * np.sqrt(a_true)  # theta ~ a**(1/m), m=2
    mid = (1.2, 2.0, 100.0, 20.0)
    m = lmap.build_localized_mapping(
        avg, global_fit(mid), bandwidth=15.0, point_x=X, point_y=None, sample_positions=pos,
        sample_times=[0] * 6, theta_obs=theta_obs, probe_layer_indices=[0] * 6,
        lab_p0=np.array(mid))
    loc = lmap.theta_at_points(m, avg, pos, [0] * 6, [0] * 6)
    glob = ws_theta(avg[pos, 0], *mid)
    assert np.sqrt(np.mean((loc - theta_obs) ** 2)) < np.sqrt(np.mean((glob - theta_obs) ** 2))


def test_theta_at_points_matches_the_map_at_the_same_points():
    avg = np.stack([EC, 1.2 * EC])[:, :, None]
    m = build(avg, positions=[2, 4, 6], times=[0, 1, 0])
    got = lmap.theta_at_points(m, avg, [3, 5], [1, 0], [0, 0])
    assert np.allclose(got, [m.theta[1, 3, 0], m.theta[0, 5, 0]])


def test_wrong_point_axis_length_is_an_error():
    with pytest.raises(ValueError):
        lmap.build_localized_mapping(
            EC[:, None], global_fit(), bandwidth=10.0, point_x=X[:5], point_y=None,
            sample_positions=[1], sample_times=[0], theta_obs=np.array([0.2]),
            probe_layer_indices=[0])


# ---------------------------------------------------------------- Step 5 hook
def _selected(avg):
    return SelectionOutput(K=1, rmse=0.0, cv_map=np.zeros(avg.shape), best_fit=global_fit(),
                           theta_calc=ws_theta(avg, *TRUTH)[None, ...])


def _hook(sel, avg, requested, bandwidth, x=X):
    rep = Rep()
    out = s5.run_step5_localized_mapping(
        rep, selected_output=sel, averaged_models=[avg], requested_mode=requested,
        bandwidth=bandwidth, bandwidth_source="test", point_x=x, point_y=None,
        sample_positions=[1, 3, 5], sample_times=[0, 0, 0],
        theta_obs=ws_theta(avg[[1, 3, 5], 0], *TRUTH), probe_layer_indices=[0, 0, 0],
        lab_p0=np.array(TRUTH), fixed_params=None, theta_obs_std=None, ws_bounds=None)
    return out, rep


def test_hook_default_switches_theta_calc_and_keeps_the_global_map():
    avg = EC[:, None].copy()
    sel = _selected(avg)
    before = sel.theta_calc.copy()
    out, rep = _hook(sel, avg, None, 30.0)
    assert out is not None and sel.ws_map_mode == "localized"
    assert sel.theta_calc.shape == before.shape == (1, 11, 1)
    assert np.allclose(sel.theta_calc_global, before)
    assert rep.metrics["Step 5 — WS calibration behind the maps"].startswith("localized")


def test_hook_global_request_leaves_the_selection_untouched():
    avg = EC[:, None].copy()
    sel = _selected(avg)
    before = sel.theta_calc.copy()
    out, _ = _hook(sel, avg, "global", 30.0)
    assert out is None and sel.ws_map_mode == "global" and sel.localized_mapping is None
    assert np.array_equal(sel.theta_calc, before)


def test_hook_without_bandwidth_falls_back_to_global_and_says_why():
    avg = EC[:, None].copy()
    sel = _selected(avg)
    out, rep = _hook(sel, avg, None, None)
    assert out is None and sel.ws_map_mode == "global"
    assert any("bandwidth" in line for line in rep.logs)


def test_hook_failure_keeps_the_run_on_global_maps():
    avg = EC[:, None].copy()
    sel = _selected(avg)
    before = sel.theta_calc.copy()
    out, rep = _hook(sel, avg, None, 30.0, x=X[:4])          # wrong length -> ValueError inside
    assert out is None and sel.ws_map_mode == "global"
    assert np.array_equal(sel.theta_calc, before)
    assert any("[WARN]" in line for line in rep.logs)


def test_persist_writes_the_localized_arrays(tmp_path):
    avg = EC[:, None].copy()
    sel = _selected(avg)
    _hook(sel, avg, None, 30.0)
    s5.persist_step5_result(
        Rep(), selected_output=sel, averaged_models=[avg], theta_obs=np.array([0.1]),
        sample_positions=[1], sample_times=[0], probe_layer_indices=[0],
        output_model_dir=str(tmp_path))
    for name in ("theta_calc", "theta_calc_global", "ws_localized_params",
                 "ws_localized_fallback", "ws_localized_bandwidth"):
        assert os.path.exists(tmp_path / f"{name}.npy"), name
    assert np.load(tmp_path / "ws_localized_params.npy").shape == (4, 1, 11)
    assert float(np.load(tmp_path / "ws_localized_bandwidth.npy")) == 30.0
