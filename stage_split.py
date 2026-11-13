"""
stage_split.py -- ``--stage calibrate`` / ``--stage map`` for real-data runs.

Why: calibration (Waxman-Smits per horizon, ranking of the ensemble, choice of
the stack depth K, localized bandwidth) only reads the EM points matched to the
TDR probes, while the maps need every point.  A single run does both, so the
whole ensemble (alphas x forward models x starting conductivities x epochs) is
inverted on all points although most members end up unused.

    stage      EM points                 ensemble                 output
    ---------  ------------------------  -----------------------  -------------------------
    full       as the scenario says      full grid                everything (default, as before)
    calibrate  probe neighbourhoods only full grid                Steps 0-5 + stage_calibration.json
    map        as the scenario says      ONLY the K selected      Steps 0-6 + drift vs calibration
               (all, or grid-thinned)    members

``stage_calibration.json`` is the hand-over between the two runs.  It stores
the K top-ranked members (alpha, beta, gamma, method, forward model, starting
conductivities), the selected K, the Waxman-Smits parameters of the selected
stack, the localized-WS bandwidth and enough run metadata to refuse an
incompatible map run (different horizons, conductivity ranges or surveys).

How the map stage "applies" the calibration: it inverts only the selected
members, then runs Steps 2-5 unchanged on that reduced ensemble.  Steps 2-5 are
cheap (they only read the probe-matched points) and K is the arg-min of the
RMSE over the cumulative top-ranked stacks, so on the K selected members the
selection reproduces the same K (the stacks 1..K are identical prefixes).  The
re-derived K and WS parameters are compared with the stored ones and any drift
is reported (:func:`drift_report`), instead of silently re-using numbers that
belong to a slightly different probe<->EM matching.
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime

import numpy as np

STAGES = ("full", "calibrate", "map")
CALIBRATION_FILENAME = "stage_calibration.json"
SCHEMA_VERSION = 1
DEFAULT_PROBE_RADIUS_M = 5.0   # README: nearest EM point of a probe is 1.1-3.2 m away
WS_KEYS = ("a", "m", "sigma_w", "sigma_s")


# ----------------------------------------------------------------- helpers
def _f(x) -> float:
    """float rounded to 12 significant digits: stable across a JSON round trip."""
    return float(f"{float(x):.12g}")


def member_key(alpha, beta, gamma, method, forward_model, conds0) -> tuple:
    """Hashable identity of an ensemble member (one inversion setting)."""
    return (_f(alpha), _f(beta), _f(gamma), str(method), str(forward_model),
            tuple(_f(c) for c in conds0))


def key_from_meta(meta: dict) -> tuple:
    return member_key(meta["alpha"], meta["beta"], meta["gamma"],
                      meta["method"], meta["forwardModel"], meta["conds0"])


def key_from_combo(combo) -> tuple:
    """``combo_list`` entry of run_pipeline: (alpha, beta, gamma, method, fm, conds0)."""
    alpha, beta, gamma, method, f_model, conds0 = combo
    return member_key(alpha, beta, gamma, method, f_model, conds0)


def stage_label(stage: str) -> str:
    """Suffix of the run folder: '' for the default stage, '_calibrate' / '_map'."""
    return "" if stage in (None, "full") else f"_{stage}"


def stage_cache_path(cache_path: str, stage: str) -> str:
    """Own cache file per stage so a calibrate/map run never overwrites (or is
    mistaken for) the full-run cache that sits at the fixed per-scenario path."""
    if stage in (None, "full"):
        return cache_path
    root, ext = os.path.splitext(cache_path)
    return f"{root}_{stage}{ext}"


def calibrate_em_subsample(em_subsample: dict | None) -> tuple[dict, bool]:
    """EM thinning to use in ``--stage calibrate``: probe neighbourhoods only.

    A scenario that already thins with ``mode='probes'`` keeps its own radius;
    anything else (off, grid, stride, random) is replaced by ``probes`` with the
    default radius.  Returns ``(config, replaced)``.
    """
    if em_subsample and str(em_subsample.get("mode", "off")).lower() == "probes":
        return dict(em_subsample), False
    return {"mode": "probes", "radius_m": DEFAULT_PROBE_RADIUS_M,
            "keep_probe_nearest": True}, True


# ------------------------------------------------------------ the artifact
def build_calibration_record(*, scenario: str, run_id: str, output_dir: str,
                             ranking, param_metadata: list, rmse_by_member,
                             selected_K: int, selected_rmse: float,
                             ws_by_horizon: list[dict],
                             localized_bandwidth: float | None,
                             bandwidth_from_sweep: bool,
                             depths0, cond_ranges, em_files: list[str],
                             em_subsample: dict | None, quick: bool,
                             n_probe_obs: int) -> dict:
    """Assemble the JSON-serialisable hand-over record.

    ``ranking[0]`` is the best member; the first ``selected_K`` entries are the
    members the map stage will invert.
    """
    ranking = [int(i) for i in ranking]
    if not 1 <= selected_K <= len(ranking):
        raise ValueError(f"selected_K={selected_K} outside 1..{len(ranking)}")
    if len(param_metadata) != len(rmse_by_member):
        raise ValueError("param_metadata and rmse_by_member differ in length "
                         f"({len(param_metadata)} vs {len(rmse_by_member)}).")
    members = []
    for rank, idx in enumerate(ranking[:selected_K], start=1):
        meta = param_metadata[idx]
        members.append({
            "rank": rank,
            "ensemble_index": idx,
            "alpha": _f(meta["alpha"]), "beta": _f(meta["beta"]),
            "gamma": _f(meta["gamma"]), "method": str(meta["method"]),
            "forwardModel": str(meta["forwardModel"]),
            "conds0": [_f(c) for c in meta["conds0"]],
            "rmse": float(rmse_by_member[idx]),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scenario": scenario,
        "run_id": run_id,
        "run_dir": output_dir,
        "quick": bool(quick),
        "em_subsample": em_subsample,
        "n_ensemble_total": len(param_metadata),
        "n_probe_obs": int(n_probe_obs),
        "selected_K": int(selected_K),
        "selected_rmse": float(selected_rmse),
        "members": members,
        "ws_by_horizon": ws_by_horizon,
        "localized_bandwidth": (None if localized_bandwidth is None
                                else float(localized_bandwidth)),
        "bandwidth_from_sweep": bool(bandwidth_from_sweep),
        "depths0": [float(d) for d in np.atleast_1d(depths0)],
        "cond_ranges": [[float(lo), float(hi)] for lo, hi in cond_ranges],
        "em_files": [os.path.basename(str(f)) for f in em_files],
    }


def ws_rows_from_fit(horizon_fits) -> list[dict]:
    """PetroFit.horizon_fits -> plain dicts (one per horizon)."""
    rows = []
    for h in horizon_fits:
        rows.append({
            "horizon": int(h.horizon),
            "is_calibrated": bool(h.is_calibrated),
            "a": float(h.a), "m": float(h.m),
            "sigma_w": float(h.sigma_w), "sigma_s": float(h.sigma_s),
            "rmse": (float(h.rmse) if h.rmse is not None else None),
        })
    return rows


def write_calibration(output_dir: str, record: dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, CALIBRATION_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return path


def resolve_calibration_path(arg: str | None, search_dir: str | None) -> str:
    """Path of the calibration file for ``--stage map``.

    ``arg`` may be the JSON file, a calibrate run folder, or ``None`` / ``"latest"``
    (newest ``*_calibrate`` folder directly under ``search_dir``).
    """
    if arg and arg.lower() != "latest":
        path = os.path.join(arg, CALIBRATION_FILENAME) if os.path.isdir(arg) else arg
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--calibration: no such file '{path}'.")
        return path
    cands = []
    if search_dir:
        cands = glob.glob(os.path.join(search_dir, "*_calibrate", CALIBRATION_FILENAME))
    if not cands:
        raise FileNotFoundError(
            "--stage map needs a calibration: run the same scenario with "
            "'--stage calibrate' first, or pass '--calibration <run folder or "
            f"{CALIBRATION_FILENAME}>'. Searched: "
            f"{os.path.join(search_dir or '?', '*_calibrate')}")
    return max(cands, key=os.path.getmtime)


def read_calibration(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    missing = [k for k in ("schema_version", "selected_K", "members", "ws_by_horizon",
                           "depths0", "cond_ranges", "em_files") if k not in rec]
    if missing:
        raise ValueError(f"'{path}' is not a valid calibration file (missing {missing}).")
    if rec["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"'{path}': schema_version {rec['schema_version']} "
                         f"unsupported (expected {SCHEMA_VERSION}).")
    if len(rec["members"]) != rec["selected_K"]:
        raise ValueError(f"'{path}': {len(rec['members'])} members stored but "
                         f"selected_K={rec['selected_K']}.")
    return rec


def record_member_keys(record: dict) -> list[tuple]:
    """Identity of each stored member, sorted (used in the cache fingerprint)."""
    return sorted(key_from_meta(m) for m in record["members"])


# ------------------------------------------------------- the map stage side
def select_members(combo_list: list, record: dict) -> tuple[list, list[tuple]]:
    """Keep the ``combo_list`` entries that the calibration selected.

    Returns ``(kept, missing)``: ``missing`` are stored members that do not
    exist in the current grid (grid or scenario changed since the calibration).
    Order of ``combo_list`` is preserved.
    """
    wanted = set(record_member_keys(record))
    kept = [c for c in combo_list if key_from_combo(c) in wanted]
    found = {key_from_combo(c) for c in kept}
    return kept, sorted(wanted - found)


def check_compatibility(record: dict, *, depths0, cond_ranges,
                        em_files: list[str], scenario: str) -> tuple[list[str], list[str]]:
    """``(errors, warnings)`` comparing a calibration with the current run.

    Errors make the map run meaningless (different horizons / ranges / surveys);
    warnings are informational (other scenario name, quick calibration, ...).
    """
    errors, warnings = [], []
    if not np.allclose(np.atleast_1d(depths0), record["depths0"]):
        errors.append(f"depths0 differ: calibration {record['depths0']} vs "
                      f"now {[float(d) for d in np.atleast_1d(depths0)]}.")
    cr_now = [[float(lo), float(hi)] for lo, hi in cond_ranges]
    if not np.allclose(np.asarray(cr_now, float), np.asarray(record["cond_ranges"], float)):
        errors.append(f"cond_ranges differ: calibration {record['cond_ranges']} vs now {cr_now}.")
    now_files = [os.path.basename(str(f)) for f in em_files]
    if now_files != list(record["em_files"]):
        errors.append("EM survey files/epochs differ from the calibration run "
                      f"(calibration: {record['em_files']}; now: {now_files}). If the "
                      "calibration used --quick-epochs, the map run needs the same epochs.")
    if scenario != record.get("scenario"):
        warnings.append(f"scenario is '{scenario}', calibration came from "
                        f"'{record.get('scenario')}'.")
    if record.get("quick"):
        warnings.append("the calibration was a --quick run (reduced ensemble): "
                        "its selected members are a smoke-test choice, not a result.")
    return errors, warnings


def drift_report(record: dict, K_now: int, ws_now: list[dict],
                 rel_tol: float = 0.05) -> dict:
    """Compare the K / WS parameters re-derived in the map run with the stored ones.

    A parameter 'drifts' when |new - old| > ``rel_tol`` * max(|old|, 1e-9).  Some
    drift is expected when the map run matches probes to a denser point set;
    a changed K or large drift means the calibration no longer describes the
    map run and it should be redone.
    """
    rows, max_rel = [], 0.0
    old_by_h = {r["horizon"]: r for r in record["ws_by_horizon"]}
    for new in ws_now:
        old = old_by_h.get(new["horizon"])
        if old is None or not (old.get("is_calibrated") and new.get("is_calibrated")):
            continue
        for k in WS_KEYS:
            o, n = float(old[k]), float(new[k])
            rel = abs(n - o) / max(abs(o), 1e-9)
            max_rel = max(max_rel, rel)
            rows.append({"horizon": new["horizon"], "param": k, "calibration": o,
                         "map_run": n, "rel_diff": rel, "drift": rel > rel_tol})
    k_changed = int(K_now) != int(record["selected_K"])
    return {"K_calibration": int(record["selected_K"]), "K_map": int(K_now),
            "k_changed": k_changed, "rows": rows, "max_rel": max_rel,
            "rel_tol": rel_tol,
            "warn": k_changed or any(r["drift"] for r in rows)}
