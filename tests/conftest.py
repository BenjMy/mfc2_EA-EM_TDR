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
