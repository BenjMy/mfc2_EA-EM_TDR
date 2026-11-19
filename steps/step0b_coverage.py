"""Pipeline Step 0b: early TDR/SWC horizon coverage check. Phase 6 of refactoring.md.

Moved verbatim out of ``run_pipeline.main()``. Runs BEFORE the potentially 30+ minute
Step 1 ensemble inversion; it needs only ``depths0`` and the known probe/sensor
installation depths (available right after Step 0, real or synthetic). It catches a
depths0 / sensor-depth mismatch that would otherwise only surface as an
"UNCALIBRATED" horizon after the full inversion has run. ``main()`` keeps the
``reporter.step``/``end_step`` bracket. No import of ``run_pipeline`` (circular).
"""
from __future__ import annotations

import em_methodology as EMmethod


def run_step0b_coverage(reporter, *, data_source, observations_list, cfg,
                        synthetic_geometry, depths0):
    """Check that every horizon of ``depths0`` has a TDR/SWC sensor depth; never aborts.

    ``data_source == "real"``: probe depths come from ``observations_list``
    (dicts with a ``"depth"`` key). Otherwise they come from ``cfg.swc_depths`` when set,
    else ``synthetic_geometry["layer_centers"]``. Returns the
    ``EMmethod.check_early_horizon_coverage`` result.
    """
    if data_source == "real":
        _early_probe_depths = sorted({obs["depth"] for obs in observations_list})
    else:
        _early_swc_depths_cfg = (
            getattr(cfg, "swc_depths", None) if cfg is not None else None
        )
        _early_probe_depths = (
            _early_swc_depths_cfg if _early_swc_depths_cfg is not None
            else synthetic_geometry["layer_centers"]
        )
    _coverage_check = EMmethod.check_early_horizon_coverage(
        probe_depths=_early_probe_depths, depths0=depths0,
    )
    reporter.log(_coverage_check.message)
    reporter.metric("Early coverage — level", _coverage_check.level)
    reporter.metric("Early coverage — uncovered horizons", len(_coverage_check.uncovered_horizons))
    if _coverage_check.level == "WARN":
        reporter.log(
            "[PIPELINE][WARN] Proceeding despite incomplete horizon coverage -- "
            "the affected horizon(s) will fall back to prior/uncalibrated "
            "Waxman-Smits parameters in Steps 2/4. Consider fixing depths0 or "
            "the SWC/TDR sensor depths before committing to a full ensemble run "
            "(re-run with --list / scenario_loader.py <name> to check quickly)."
        )
    return _coverage_check
