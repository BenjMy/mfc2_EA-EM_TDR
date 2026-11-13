"""Waxman-Smits forward/inverse round trip (em_methodology.waxman_smits_theta)."""
import numpy as np
import pytest

import em_methodology as em

PARAMS = [
    (0.5, 3.0, 500.0, 100.0),   # synthetic truth defaults
    (1.0, 1.8, 50.0, 5.0),
    (0.35, 2.0, 700.0, 50.0),
]


def _ec_from_theta(theta, a, m, sigma_w, sigma_s):
    """Analytic inverse: sigma_b = (sigma_w + sigma_s) * theta**m / a  [mS/m]."""
    return (sigma_w + sigma_s) * np.asarray(theta) ** m / a


@pytest.mark.parametrize("a,m,sw,ss", PARAMS)
def test_theta_ec_theta_round_trip(a, m, sw, ss):
    theta = np.linspace(0.05, 0.45, 9)
    back = em.waxman_smits_theta(_ec_from_theta(theta, a, m, sw, ss), a, m, sw, ss)
    np.testing.assert_allclose(back, theta, rtol=1e-10)


def test_monotonic_in_ec_and_shape_preserved():
    sigma_b = np.linspace(1.0, 200.0, 40).reshape(4, 10)
    theta = em.waxman_smits_theta(sigma_b, 0.5, 3.0, 500.0, 100.0)
    assert theta.shape == sigma_b.shape
    assert np.all(np.diff(theta.ravel()) > 0)


def test_non_positive_ec_is_guarded_not_nan():
    theta = em.waxman_smits_theta(np.array([0.0, -5.0]), 0.5, 3.0, 500.0, 100.0)
    assert np.all(np.isfinite(theta)) and np.all(theta >= 0)


def test_default_bounds_contain_default_start_and_override_is_partial():
    lo, hi = em.resolve_ws_bounds(None)
    p0 = em.DEFAULT_WS_P0
    assert all(l <= p <= h for l, p, h in zip(lo, p0, hi))
    lo2, hi2 = em.resolve_ws_bounds({"a": [0.2, 5.0]})
    assert (lo2[0], hi2[0]) == (0.2, 5.0)
    assert (lo2[1:], hi2[1:]) == (lo[1:], hi[1:])
