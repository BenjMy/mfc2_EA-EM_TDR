"""ws_bounds: scenario field validation, resolve_ws_bounds, and that the Step 2
optimizer can reach an 'a' below the default lower bound (0.5) only when asked."""
import json
import os

import numpy as np
import pytest

import em_methodology as EMmethod
import scenario_loader as sl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH_JSON = os.path.join(ROOT, "ensemble_scenarios.json")


def _scenario_with(tmp_path, **over):
    with open(SYNTH_JSON, encoding="utf-8") as fh:
        raw = json.load(fh)["1d_full_depth_ref"]
    raw.update(over)
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"s": raw}), encoding="utf-8")
    return str(p)


def test_resolve_defaults_and_partial_override():
    lo, hi = EMmethod.resolve_ws_bounds(None)
    assert (lo, hi) == (list(EMmethod.WS_BOUNDS[0]), list(EMmethod.WS_BOUNDS[1]))
    lo, hi = EMmethod.resolve_ws_bounds({"a": [0.2, 5.0]})
    assert lo[0] == 0.2 and hi[0] == 5.0
    assert lo[1:] == list(EMmethod.WS_BOUNDS[0])[1:]
    assert hi[1:] == list(EMmethod.WS_BOUNDS[1])[1:]
    with pytest.raises(ValueError):
        EMmethod.resolve_ws_bounds({"a": [2.0, 1.0]})
    with pytest.raises(ValueError):
        EMmethod.resolve_ws_bounds({"zeta": [0.0, 1.0]})


def test_loader_accepts_and_rejects(tmp_path):
    cfg = sl.load_scenario("s", config_path=_scenario_with(tmp_path, ws_bounds={"a": [0.2, 5]}))
    assert cfg.ws_bounds == {"a": [0.2, 5.0]}
    assert sl.load_scenario("s", config_path=_scenario_with(tmp_path)).ws_bounds is None
    for bad in ({"a": [1.0, 0.5]}, {"a": [0.2]}, {"x": [0, 1]}, {}, [0.2, 5], {"a": ["0", "1"]}):
        with pytest.raises(ValueError):
            sl.load_scenario("s", config_path=_scenario_with(tmp_path, ws_bounds=bad))


def test_free4_scenarios_use_wider_a_bounds_and_start_off_bound():
    for name in ("1d_full_free4", "1d_surface_free4",
                 "1d_full_hetero_free4", "1d_surface_hetero_free4"):
        cfg = sl.load_scenario(name, config_path=SYNTH_JSON)
        lo, hi = EMmethod.resolve_ws_bounds(cfg.ws_bounds)
        a0 = cfg.lab_p0[0]
        assert lo[0] < 0.35 and lo[0] < a0 < hi[0], (name, lo[0], a0, hi[0])


def _synthetic_horizon(a_true=0.3, m=2.0, sw=500.0, ss=100.0, n=12, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.1, 0.4, n)
    sigma_b = (theta ** m / a_true) * (sw + ss)
    return theta, sigma_b


def _fit_a(ws_bounds):
    theta, sigma_b = _synthetic_horizon()
    models = np.tile(sigma_b[None, :, None], (1, 1, 1))           # (ens, pts, 1 layer)
    fits = EMmethod.calibrate_petrophysical_relationship(
        models=models, sample_positions=list(range(len(theta))),
        sample_times=[0] * len(theta), theta_obs=theta,
        lab_p0=np.array([1.0, 2.0, 500.0, 100.0]), ws_bounds=ws_bounds,
    )
    return fits[0].horizon_fits[0].a


def test_optimizer_reaches_a_below_default_bound_only_with_override():
    default_bounds_a = _fit_a(None)
    wide_a = _fit_a({"a": [0.2, 5.0]})
    assert default_bounds_a >= EMmethod.WS_BOUNDS[0][0] - 1e-9     # clipped at 0.5
    assert wide_a < EMmethod.WS_BOUNDS[0][0] - 1e-3                # free to go below
    assert wide_a >= 0.2 - 1e-9
