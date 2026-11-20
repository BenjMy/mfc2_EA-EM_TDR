"""Put an RMSE of soil water content (SWC) in context.

An RMSE of 0.05 m3/m3 means nothing on its own: is it large or small? It depends on
how much the measured SWC itself varies in the dataset. This module gives three
numbers next to the raw RMSE:

* ``rmse_pct``   RMSE in % vol (m3/m3 x 100) -- the headline, human-readable number;
* ``rsr``        RMSE / std of the measured SWC (RMSE-observations standard deviation
                 ratio, Moriasi et al. 2007) -- the normalised number: 0 = perfect,
                 1 = no better than predicting the mean of the observations;
* ``nrmse_mean`` RMSE / mean of the measured SWC, in % (relative error).

plus ``skill`` = 1 - rsr**2 (Nash-Sutcliffe-like: share of the observed variance
explained when RMSE is the error of the model) and a plain-language ``rating``.

How to read the rating (RSR, after Moriasi et al. 2007):

    RSR <= 0.50   very good
    0.50-0.60     good
    0.60-0.70     satisfactory
    > 0.70        unsatisfactory (error comparable to the natural spread of SWC)
    >= 1.00       no skill (not better than the mean of the probes)

Absolute reference points (m3/m3): a calibrated TDR/capacitance probe is itself good
to about 0.02-0.03; remote-sensing SWC products target about 0.04 (SMAP mission
requirement). So RMSE below ~0.03 is at the sensor-noise level, 0.03-0.06 is typical
for a good field map, and above ~0.06 deserves a close look.

Caveat: RSR needs a spread in the observations. With a short, dry-only or
single-position dataset the std is tiny and RSR explodes; ``rating`` then says
"std too small" instead of a misleading number (see ``MIN_STD``).

Pure numpy, no emagpy: unit-tested in tests/test_rmse_context.py.
"""
from __future__ import annotations

import math
import re

import numpy as np

MIN_STD = 0.01          # m3/m3: below this the observed spread is too small for a ratio
RSR_BANDS = ((0.50, "very good"), (0.60, "good"), (0.70, "satisfactory"))

# Columns added to the run logs (see run_log.RUN_LOG_COLUMNS).
CONTEXT_COLUMNS = ["rmse_pct", "swc_obs_std", "rsr", "rmse_rating"]


def parse_rmse(value) -> float:
    """First number of a metric string such as ``'0.031212 m³/m³'`` (nan if none)."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    return float(m.group(0)) if m else float("nan")


def rating_from_rsr(rsr: float, obs_std: float | None = None) -> str:
    """Plain-language rating of an RSR (see module docstring)."""
    if obs_std is not None and (not np.isfinite(obs_std) or obs_std < MIN_STD):
        return "std too small"
    if rsr is None or not np.isfinite(rsr):
        return "n/a"
    if rsr >= 1.0:
        return "no skill"
    for upper, label in RSR_BANDS:
        if rsr <= upper:
            return label
    return "unsatisfactory"


def rmse_context(rmse, theta_obs) -> dict:
    """Context numbers for ``rmse`` [m3/m3] against the measured SWC ``theta_obs``.

    ``rmse`` may be a float or the metric string. ``theta_obs`` is any array of the
    observed SWC (NaN ignored). Returns a dict with floats (nan when undefined):
    rmse, rmse_pct, obs_mean, obs_std, obs_min, obs_max, n_obs, rsr, nrmse_mean_pct,
    skill, rating.
    """
    r = parse_rmse(rmse)
    obs = np.asarray(theta_obs, dtype=float).ravel()
    obs = obs[np.isfinite(obs)]
    out = {"rmse": r, "rmse_pct": r * 100.0, "n_obs": int(obs.size),
           "obs_mean": float("nan"), "obs_std": float("nan"),
           "obs_min": float("nan"), "obs_max": float("nan"),
           "rsr": float("nan"), "nrmse_mean_pct": float("nan"),
           "skill": float("nan"), "rating": "n/a"}
    if obs.size < 2:
        return out
    # population std (ddof=0): matches the definition of RMSE (divide by n)
    out.update(obs_mean=float(obs.mean()), obs_std=float(obs.std()),
               obs_min=float(obs.min()), obs_max=float(obs.max()))
    if out["obs_mean"] > 0:
        out["nrmse_mean_pct"] = r / out["obs_mean"] * 100.0
    if out["obs_std"] > 0:
        out["rsr"] = r / out["obs_std"]
        out["skill"] = 1.0 - out["rsr"] ** 2
    out["rating"] = rating_from_rsr(out["rsr"], out["obs_std"])
    return out


def metric_lines(ctx: dict) -> list[tuple[str, str]]:
    """(name, text) pairs for ``reporter.metric`` -- the report/manifest entries."""
    f = lambda v, p: "n/a" if not math.isfinite(v) else f"{v:.{p}f}"
    return [
        ("Final RMSE (% vol)", f"{f(ctx['rmse_pct'], 2)} % vol"),
        ("SWC observed std [m³/m³]",
         f"{f(ctx['obs_std'], 4)}  (mean {f(ctx['obs_mean'], 4)}, "
         f"range {f(ctx['obs_min'], 3)}-{f(ctx['obs_max'], 3)}, n={ctx['n_obs']})"),
        ("Normalised RMSE (RMSE/std, RSR)", f(ctx["rsr"], 3)),
        ("Relative RMSE (RMSE/mean obs)", f"{f(ctx['nrmse_mean_pct'], 1)} %"),
        ("RMSE rating", ctx["rating"]),
    ]


def prefixed_metric_lines(prefix: str, ctx: dict) -> list[tuple[str, str]]:
    """Compact metrics for a secondary RMSE (e.g. ``'Validation'`` = held-out probes)."""
    f = lambda v, p: "n/a" if not math.isfinite(v) else f"{v:.{p}f}"
    return [
        (f"{prefix} RMSE (% vol)", f"{f(ctx['rmse_pct'], 2)} % vol"),
        (f"{prefix} RSR (RMSE/std of its own SWC obs)",
         f"{f(ctx['rsr'], 3)}  (obs std {f(ctx['obs_std'], 4)}, n={ctx['n_obs']})"),
        (f"{prefix} RMSE rating", ctx["rating"]),
    ]


def log_fields(ctx: dict) -> dict:
    """Values for the run-log CSV columns ``CONTEXT_COLUMNS`` (strings)."""
    f = lambda v, p: "" if not math.isfinite(v) else f"{v:.{p}f}"
    return {"rmse_pct": f(ctx["rmse_pct"], 3), "swc_obs_std": f(ctx["obs_std"], 5),
            "rsr": f(ctx["rsr"], 4), "rmse_rating": ctx["rating"] if ctx["rating"] != "n/a" else ""}


def log_fields_from_rmse(final_rmse) -> dict:
    """Backfill for old log rows: only the %vol column is derivable from the RMSE alone."""
    r = parse_rmse(final_rmse)
    return {"rmse_pct": "" if not math.isfinite(r) else f"{r * 100.0:.3f}",
            "swc_obs_std": "", "rsr": "", "rmse_rating": ""}
