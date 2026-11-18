"""steps/step2_localized_calibration.run_step2_localized_diagnostic with stubs (no emagpy, no inversion)."""
from types import SimpleNamespace

import numpy as np

from steps import step2_localized_calibration as s2l


class Reporter:
    def __init__(self):
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)


CALIB = [SimpleNamespace(probe_x=float(i), probe_y=2.0 * i, probe_depth=0.1 * (i + 1)) for i in range(3)]
FIT = [SimpleNamespace(horizon_fits=["h0", "h1"])]


def _install(monkeypatch, fail_in=None):
    """Stub the fit and the three plots; return the dict recording what each received."""
    seen = {"plots": []}

    def fit(**kw):
        seen["fit"] = kw
        if fail_in == "fit":
            raise RuntimeError("fit boom")
        return FIT

    def plot(name):
        def f(**kw):
            if fail_in == name:
                raise RuntimeError(name + " boom")
            seen["plots"].append((name, kw))
        return f

    monkeypatch.setattr(s2l.step2, "run_petro_localized", fit)
    for n in ("plot_calibration_density_kernel", "plot_ws_params_xz", "plot_ws_params_xy"):
        monkeypatch.setattr(s2l.pcd, n, plot(n))
    return seen


def _call(rep, mode="1d", bandwidth=12.5, from_sweep=True, localized_y=None, **over):
    kw = dict(mode=mode, bandwidth=bandwidth, bandwidth_from_sweep=from_sweep, localized_y=localized_y,
              models_clean=["m0", "m1"], sample_positions=[0, 1], sample_times=[0, 0],
              theta_obs=np.zeros(2), x_transect=np.arange(3.0), depths0=[0.1, 0.3],
              effective_depths0=[0.1, 0.25], output_dir="/out", probe_layer_indices=[0, 1],
              fixed_params={"m": 1.5}, lab_p0="p0", theta_obs_std=0.02,
              fits_step2=[SimpleNamespace(rmse=0.02)], sample_sensor_ids=["a", "b"],
              ws_bounds={"a": [0.2, 5]}, survey_df="survey", calib_position_results=CALIB)
    kw.update(over)
    return s2l.run_step2_localized_diagnostic(rep, **kw)


def test_skipped_when_no_bandwidth(monkeypatch):
    seen = _install(monkeypatch)
    rep = Reporter()
    assert _call(rep, bandwidth=None) is None
    assert "fit" not in seen and not seen["plots"]
    assert any("not set -- skipping" in m for m in rep.lines)


def test_success_passes_arguments_and_returns_fits(monkeypatch):
    seen = _install(monkeypatch)
    rep = Reporter()
    out = _call(rep)
    assert out is FIT
    kw = seen["fit"]
    assert kw["bandwidth"] == 12.5 and kw["depths0"] == [0.1, 0.3] and kw["models"] == ["m0", "m1"]
    assert kw["global_fit"] is not None and kw["sensor_ids"] == ["a", "b"] and kw["output_dir"] == "/out"
    assert kw["point_y_coords"] is None and kw["ws_bounds"] == {"a": [0.2, 5]}
    names = [n for n, _ in seen["plots"]]
    assert names == ["plot_calibration_density_kernel", "plot_ws_params_xz"]   # no XY in 1-D
    xz = dict(seen["plots"])["plot_ws_params_xz"]
    assert xz["depths0"] == [0.1, 0.25]                    # effective depths, not the static ones
    assert list(xz["probe_x"]) == [0.0, 1.0, 2.0]
    assert np.allclose(xz["probe_depths"], [0.1, 0.2, 0.3])
    assert any("recommended by LOO-CV sweep" in m for m in rep.lines)
    assert any("Diagnostic complete" in m for m in rep.lines)


def test_fixed_bandwidth_wording_and_2d_xy_plot(monkeypatch):
    seen = _install(monkeypatch)
    rep = Reporter()
    y = np.arange(3.0)
    _call(rep, mode="2d", from_sweep=False, localized_y=y)
    assert any("fixed scenario value" in m and "2-D Euclidean" in m for m in rep.lines)
    assert seen["fit"]["point_y_coords"] is y
    names = [n for n, _ in seen["plots"]]
    assert names[-1] == "plot_ws_params_xy"
    assert list(dict(seen["plots"])["plot_ws_params_xy"]["probe_y"]) == [0.0, 2.0, 4.0]


def test_fit_failure_is_a_warning_and_returns_none(monkeypatch):
    seen = _install(monkeypatch, fail_in="fit")
    rep = Reporter()
    assert _call(rep) is None
    assert not seen["plots"]
    assert any("[WARN] Localized diagnostic failed: fit boom" in m for m in rep.lines)


def test_plot_failure_is_a_warning_and_never_raises(monkeypatch):
    _install(monkeypatch, fail_in="plot_ws_params_xz")
    rep = Reporter()
    _call(rep)
    assert any("[WARN] Localized diagnostic failed: plot_ws_params_xz boom" in m for m in rep.lines)
