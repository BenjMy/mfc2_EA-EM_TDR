"""plots.truth_tdr.plot_truth_tdr: shapes, half-space depth, file written."""
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from plots.truth_tdr import plot_truth_tdr


DEPTHS = np.array([0.2, 0.5, 1.0])  # 3 bounded layers + half-space = 4 layers


@pytest.mark.parametrize("shape", [(2, 5, 4), (5, 4)])
def test_writes_png_for_time_lapse_and_single_survey(tmp_path, shape):
    theta = np.random.default_rng(0).uniform(0.1, 0.4, size=shape)
    path = plot_truth_tdr(str(tmp_path), theta, DEPTHS)
    assert os.path.isfile(path)
    assert os.path.basename(path) == "truth_tdr_profiles.png"


def test_bad_ndim_raises(tmp_path):
    with pytest.raises(ValueError, match="Unexpected theta_true_realization shape"):
        plot_truth_tdr(str(tmp_path), np.zeros(4), DEPTHS)


def test_still_reachable_from_run_pipeline():
    import run_pipeline
    assert run_pipeline.plot_truth_tdr is plot_truth_tdr
