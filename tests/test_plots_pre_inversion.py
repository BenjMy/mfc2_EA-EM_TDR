"""Smoke tests for plots/pre_inversion.py (moved out of em_methodology, Phase 5)."""
import os

import numpy as np

from plots import pre_inversion


def test_step1_ensemble_plots_write_pngs(tmp_path):
    models = np.random.default_rng(0).uniform(5, 60, (6, 8, 3))  # (ensemble, points, layers)
    pre_inversion.plot_step1_ensemble_single_file(models, [0.3, 0.6], output_dir=str(tmp_path))
    pre_inversion.plot_step1_ensemble_by_point(models, [0.3, 0.6], output_dir=str(tmp_path), max_points=4)
    assert os.path.getsize(tmp_path / "step1_ensemble_spread.png") > 0
    assert os.path.getsize(tmp_path / "step1_ensemble_spread_by_point.png") > 0


def test_core_library_does_not_import_matplotlib():
    import em_methodology
    assert not hasattr(em_methodology, "plt")
