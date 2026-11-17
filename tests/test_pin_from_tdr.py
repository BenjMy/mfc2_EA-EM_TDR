import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import em_methodology as EM  # noqa: E402
import pin_from_tdr as pin  # noqa: E402

P = (0.5, 1.8, 500.0, 10.0)


def test_ec_from_theta_inverts_waxman_smits():
    theta = np.array([0.05, 0.2, 0.35])
    ec = pin.ec_from_theta(theta, *P)
    assert np.allclose(EM.waxman_smits_theta(ec, *P), theta)


def test_exact_at_probes_and_linear_between_1d():
    x = np.arange(0.0, 20.0, 2.0)               # 10 EM points
    theta = np.array([0.15, 0.30])
    out = pin.derive_pin_ec(theta, [1, 7], P, x)
    ec = pin.ec_from_theta(theta, *P)
    assert out.shape == (1, 10)
    assert np.allclose(out[0, [1, 7]], ec)
    assert np.allclose(out[0, 4], ec.mean())     # x=8 is midway between x=2 and x=14
    assert np.allclose(out[0, 0], ec[0]) and np.allclose(out[0, 9], ec[1])  # constant ends


def test_single_probe_gives_constant_profile():
    out = pin.derive_pin_ec([0.2], [3], P, np.arange(6.0))
    assert np.allclose(out, out[0, 0])


def test_two_probes_at_one_position_are_averaged_in_ec_space():
    out = pin.derive_pin_ec([0.1, 0.3], [2, 2], P, np.arange(5.0))
    assert np.allclose(out, pin.ec_from_theta([0.1, 0.3], *P).mean())


def test_epochs_and_missing_epoch_uses_nearest():
    x = np.arange(4.0)
    out = pin.derive_pin_ec([0.1, 0.2], [0, 0], P, x, obs_times=[0, 2], n_time=3)
    ec = pin.ec_from_theta([0.1, 0.2], *P)
    assert out.shape == (3, 4)
    assert np.allclose(out[0], ec[0]) and np.allclose(out[2], ec[1])
    assert np.allclose(out[1], ec[0])            # tie -> first epoch


def test_bounds_clip():
    out = pin.derive_pin_ec([0.5], [0], P, np.arange(3.0), ec_bounds=(1.0, 50.0))
    assert out.max() == 50.0


def test_2d_nearest_with_two_probes_and_linear_with_three():
    x = np.array([0., 1., 2., 0., 1., 2.]); y = np.array([0., 0., 0., 1., 1., 1.])
    two = pin.derive_pin_ec([0.1, 0.3], [0, 2], P, x, y)
    assert two.shape == (1, 6) and np.isfinite(two).all()
    three = pin.derive_pin_ec([0.1, 0.3, 0.2], [0, 2, 4], P, x, y)
    assert np.isfinite(three).all()
    ec = pin.ec_from_theta([0.1, 0.3, 0.2], *P)
    assert np.allclose(three[0, [0, 2, 4]], ec)


def test_no_observation_raises():
    with pytest.raises(ValueError):
        pin.derive_pin_ec([], [], P, np.arange(3.0))
