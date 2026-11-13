"""Shared plotting helpers: one copy of ``ensure_dir`` and ``save_fig``.

Used by the plot_*.py modules (previously each module had its own copy).

ensure_dir(path)                      -> path   (creates the folder if needed)
save_fig(fig, path, dpi=180, tag=...) -> None   (savefig + close + log line)
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt


def ensure_dir(path: str) -> str:
    """Create ``path`` (and parents) if missing and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def save_fig(fig: plt.Figure, path: str, dpi: int = 180, tag: str = "DIAG") -> None:
    """Save ``fig`` to ``path`` (tight bbox), close it, and print ``[<tag>] saved``.

    ``tag`` is only the prefix of the log line (modules keep their own prefix,
    e.g. ``DIAG`` or ``SOLUTION``).
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    [{tag}] saved → {os.path.relpath(path)}")
