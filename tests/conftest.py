"""Make the repo root importable and force a headless matplotlib backend."""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# The em_methodology tests only need the module to import; stub emagpy when
# the real package is not installed (e.g. CI, cloud sandbox).
try:
    import emagpy  # noqa: F401
except ImportError:
    import types

    _stub = types.ModuleType("emagpy")

    class Problem:  # minimal placeholder, never instantiated by the tests
        pass

    _stub.Problem = Problem
    _stub.__is_stub__ = True
    sys.modules["emagpy"] = _stub


# ---------------------------------------------------------------------------
# Synthetic MFC2-like dataset (session fixture, generated once into a temp dir)
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


def _real_emagpy() -> bool:
    return not getattr(sys.modules.get("emagpy"), "__is_stub__", False)


@pytest.fixture(scope="session")
def mfc2_synthetic(tmp_path_factory):
    """Small deterministic synthetic MFC2 dataset registered as ``mfc2_synthetic``.

    Returns the dict of ``build_dataset`` (``em_dir``, ``tdr_dir``, ``coord_file``,
    ``ground_truth_dir``, ``scenario_json``, ``n_em_epochs``) plus ``root`` (the temp
    directory), ``run_dir`` (where a pipeline run should write) and ``env`` (environment for a
    subprocess: ``EA_EM_DATASETS_FILE`` points to a JSON that registers the dataset, so
    ``python run_pipeline.py --dataset mfc2_synthetic`` takes the same path as real data).
    Skipped when the real emagpy (forward model) is missing. Fixed seed, about 4 s.
    """
    pytest.importorskip("emagpy")
    if not _real_emagpy():
        pytest.skip("needs the real emagpy (forward model)")
    import generate_synthetic_real_mirror as gen

    root = tmp_path_factory.mktemp("mfc2_synthetic")
    run_dir = root / "runs"
    out = gen.build_dataset(str(root / "data"), scale=0.01, max_points=150, seed=42,
                            tdr_freq_days=7, n_stations=20, output_dir=str(run_dir))
    datasets_file = root / "EA_EM_datasets.json"
    gen.register_built_dataset(out, datasets_file=str(datasets_file))
    env = dict(os.environ, EA_EM_DATASETS_FILE=str(datasets_file), MPLBACKEND="Agg")
    return dict(out, root=str(root), run_dir=str(run_dir), env=env,
                datasets_file=str(datasets_file))
