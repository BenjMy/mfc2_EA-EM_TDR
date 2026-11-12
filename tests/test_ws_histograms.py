"""plot_ws_parameter_histograms: collapsed / at-bound panels are annotated."""
import os
import types

import numpy as np

import em_methodology as EMmethod
import step2_petro


def _fit(a_vals, m_vals):
    hz = [types.SimpleNamespace(a=a, m=m, sigma_w=500.0 + 10 * i, sigma_s=100.0, rmse=0.01)
          for i, (a, m) in enumerate(zip(a_vals, m_vals))]
    return types.SimpleNamespace(horizon_fits=hz)


def test_histogram_runs_and_annotates_bound(tmp_path):
    rng = np.random.default_rng(0)
    # 2 horizons: H0 'a' spreads, H1 'a' stuck on its lower bound (0.5)
    fits = [_fit([0.5 + abs(rng.normal(0, 0.1)), EMmethod.WS_BOUNDS[0][0]],
                 rng.normal(2, 0.1, 2)) for _ in range(20)]
    out = tmp_path / "h.png"
    texts = []
    import matplotlib.axes
    orig = matplotlib.axes.Axes.text

    def spy(self, *a, **k):
        texts.append(a[2] if len(a) > 2 else k.get("s", ""))
        return orig(self, *a, **k)

    matplotlib.axes.Axes.text = spy
    try:
        step2_petro.plot_ws_parameter_histograms(
            fits, fits[0], [0.3], np.array([0.5, 3.0, 500.0, 100.0]), str(out))
    finally:
        matplotlib.axes.Axes.text = orig
    assert os.path.getsize(out) > 0
    assert any("20/20 at lower bound 0.5" in t for t in texts)      # H1 'a'
    assert any("no spread" in t for t in texts)                      # H1 'a' and sigma_s
