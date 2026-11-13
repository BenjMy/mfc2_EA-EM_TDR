"""Helpers shared by the pipeline step modules (no dependency on run_pipeline)."""
import os


def resolve_npy_path(filename: str, output_model_dir: str, output_dir: str) -> str:
    """
    Resolve the path to a ``.npy`` array, preferring ``output_model_dir``
    (where this pipeline now consolidates arrays) and falling back to the
    legacy top-level ``output_dir`` location for backward compatibility
    with older runs/caches.
    """
    preferred = os.path.join(output_model_dir, filename)
    if os.path.isfile(preferred):
        return preferred
    return os.path.join(output_dir, filename)
