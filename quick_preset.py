"""
quick_preset.py -- the ``--quick`` smoke-test preset (real-data runs).

Step 1 dispatches one inversion task per
    (alpha x forward model x starting-conductivity combination x epoch)
(see README "Cost").  For ``real_single_base`` that is
    5 alphas x 2 forward models x 4 starting combinations x 7 epochs = 280.

``--quick`` shrinks the ensemble to check that a configuration runs end to
end (and to eyeball the plots) in minutes instead of hours:

    one alpha  +  one forward model            -> 280 -> 28 tasks
    ... and ``--quick-epochs 3``               -> 280 -> 12 tasks

It does NOT touch the data: no EM point is dropped (that is what
``em_subsample`` is for) and the TDR probes are unchanged.  It DOES shrink
the ensemble, which is the scientific object of the method, so a quick run
is a smoke test, never a result.  Everything downstream of Step 1 still
runs, but on a degenerate ensemble (one alpha, one forward model) the
rank/stack/uncertainty statistics are not meaningful.

Choices made here (deterministic, printed at run time):
  * alpha          -> the MIDDLE element of the scenario's alpha list.  The
                      list is a log sweep (``alpha_range_log``), so the
                      middle is the geometric centre: neither the
                      under- nor the over-regularised end.
  * forward model  -> the FIRST element of the scenario's list (the
                      scenario's primary model, e.g. FSlin before CS).
  * epochs         -> ``k`` indices evenly spaced over the survey series,
                      always including the first and the last.
Starting-conductivity combinations, ``method``, ``beta`` and ``gamma`` are
left exactly as the scenario / CLI define them.
"""
from __future__ import annotations

import os

import numpy as np


def reduce_param_grid(param_grid: dict) -> tuple[dict, dict]:
    """Return ``(reduced_grid, info)``; never mutates *param_grid*.

    ``info`` lists, per reduced axis, the full list and what was kept, for
    logging.  Axes with a single value (or absent) are left alone.
    """
    grid = dict(param_grid)
    info: dict[str, dict] = {}

    alphas = list(grid.get("alpha", []))
    if len(alphas) > 1:
        kept = alphas[len(alphas) // 2]
        info["alpha"] = {"n_full": len(alphas), "kept": kept}
        grid["alpha"] = [kept]

    fms = list(grid.get("forwardModel", []))
    if len(fms) > 1:
        kept = fms[0]
        info["forwardModel"] = {"n_full": len(fms), "kept": kept}
        grid["forwardModel"] = [kept]

    return grid, info


def select_epochs(n_total: int, k: int | None) -> list[int]:
    """Indices of ``k`` evenly spaced epochs out of ``n_total`` (first and
    last always included).  ``k=None`` or ``k >= n_total`` keeps them all.
    """
    if n_total < 1:
        raise ValueError("select_epochs: n_total must be >= 1")
    if k is None or k >= n_total:
        return list(range(n_total))
    if k < 1:
        raise ValueError(f"--quick-epochs must be >= 1 (got {k})")
    if k == 1:
        return [0]
    idx = np.unique(np.round(np.linspace(0, n_total - 1, k)).astype(int))
    return [int(i) for i in idx]


def subset_epochs(raw_em_files: list, em_dates: list, observations: list,
                  keep: list[int]) -> tuple[list, list, list]:
    """Restrict a prepared real dataset to the epochs in *keep*.

    ``observations`` carry a ``time_index`` into ``em_dates``; it is
    remapped onto the new, contiguous 0..len(keep)-1 numbering and
    observations of dropped epochs are removed.  Inputs are not mutated.
    """
    keep = sorted(set(keep))
    new_index = {old: new for new, old in enumerate(keep)}
    files = [raw_em_files[i] for i in keep]
    dates = [em_dates[i] for i in keep]
    obs = [dict(o, time_index=new_index[o["time_index"]])
           for o in observations if o["time_index"] in new_index]
    return files, dates, obs


def quick_cache_path(cache_path: str, quick_epochs: int | None) -> str:
    """Cache file used by a quick run.

    The full-run cache sits at a fixed per-scenario path.  A quick run has
    a different config fingerprint, so pointing it at that same file would
    treat the (expensive) full cache as stale and then OVERWRITE it.  Quick
    runs therefore get their own file next to it.
    """
    root, ext = os.path.splitext(cache_path)
    tag = "_quick" + (f"_e{quick_epochs}" if quick_epochs else "")
    return f"{root}{tag}{ext}"


def count_tasks(param_grid: dict, n_start_combos: int, n_epochs: int) -> int:
    """Number of Step-1 tasks for a grid (for the pre-run banner)."""
    n = 1
    for key in ("alpha", "beta", "gamma", "method", "forwardModel"):
        n *= max(len(param_grid.get(key, [None])), 1)
    return n * n_start_combos * n_epochs
