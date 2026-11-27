"""plot_em_diagnostics_real: epoch matching must scale to ~1e5 points and keep its tolerance rule."""
import inspect

import numpy as np
import pandas as pd

import plot_em_diagnostics_real as pd_real


def _df(x, y, v):
    return pd.DataFrame({"x": x, "y": y, "f": v})


def test_match_applies_tolerance_and_uses_nearest_point():
    ref = _df([0.0, 10.0, 20.0], [0, 0, 0], [1.0, 2.0, 3.0])
    ep = _df([0.5, 10.2, 50.0], [0, 0, 0], [10.0, 20.0, 30.0])      # nothing within 2 m of x = 20
    x, y, m = pd_real._match_epochs_to_reference([ref, ep], ["f"], tolerance_m=2.0)
    assert m[1][:, 0].tolist()[:2] == [10.0, 20.0] and np.isnan(m[1][2, 0])
    assert m[0][:, 0].tolist() == [1.0, 2.0, 3.0]                     # the reference matches itself


def test_match_scales_to_real_survey_size():
    rng = np.random.default_rng(0)
    n = 106_342                                                        # the real map run: dense matrix = 169 GiB
    xy = rng.uniform(0, 500, (n, 2))
    a, b = _df(xy[:, 0], xy[:, 1], np.arange(n, dtype=float)), _df(xy[:, 0] + 0.01, xy[:, 1], np.arange(n, dtype=float))
    x, y, m = pd_real._match_epochs_to_reference([a, b], ["f"], tolerance_m=2.0)
    assert m[1].shape == (n, 1) and np.isfinite(m[1]).mean() > 0.99


def test_filter_repeated_is_called_with_the_real_keyword():
    src = inspect.getsource(pd_real)
    assert "filterRepeated(index" not in src and "filterRepeated(tolerance=" in src
