"""
Core Methodology Library: Option A Architecture

Tracks and analyzes multi-inversion electromagnetic (EM) ensembles using either:

  Single-time survey  →  3D layout: (ensemble_size, num_measured_points, num_layers)
  Time-lapse survey   →  4D layout: (ensemble_size, n_time, num_measured_points, num_layers)

Designed to process standalone or time-lapse 1D/2D transect surveys with fixed
interface depths.  Time-lapse mode is triggered automatically when a list of
survey filenames is passed instead of a single string.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence, Callable

import numpy as np
from emagpy import Problem

import geometry_utils


# ==========================================================================
# 1. CORE DATA STRUCTURES & CONTAINERS
# ==========================================================================

@dataclass
class HorizonWaxmanSmitsParams:
    """
    Fitted Waxman-Smits parameters for a single soil horizon.

    Model: θ = (a · σ_b / (σ_w + σ_s)) ^ (1/m)

    Parameters
    ----------
    horizon : int
        0-based layer index.
    a : float
        Tortuosity factor.
    m : float
        Cementation exponent.
    sigma_w : float
        Pore water conductivity [mS/m].
    sigma_s : float
        Surface conductivity [mS/m].
    rmse : float
        Root-mean-square error of θ prediction on calibration samples.
        ``NaN`` when ``is_calibrated`` is ``False`` (no TDR probe observations
        were available for this horizon, so there is nothing to compute an
        error against).
    predicted_theta : np.ndarray
        Predicted volumetric water content at calibration sample positions.
        Empty when ``is_calibrated`` is ``False``.
    observed_theta : np.ndarray
        Observed (TDR) volumetric water content at calibration sample positions.
        Empty when ``is_calibrated`` is ``False``.
    is_calibrated : bool
        ``True`` when at least one TDR probe was actually installed in (and
        routed to) this horizon, so the Waxman-Smits parameters were fitted
        against real observations. ``False`` when no probe mapped to this
        horizon: in that case ``a/m/sigma_w/sigma_s`` are simply the
        starting-guess (``p0``) values carried forward unfit, and should be
        treated as an uncalibrated prior rather than a genuine calibration
        result.
    """
    horizon: int
    a: float
    m: float
    sigma_w: float
    sigma_s: float
    rmse: float
    predicted_theta: np.ndarray
    observed_theta: np.ndarray
    is_calibrated: bool = True


@dataclass
class PetroFit:
    """
    Container for Waxman-Smits calibration results for one ensemble member.

    Holds a list of per-horizon fitted parameters (one HorizonWaxmanSmitsParams
    per soil layer), plus a convenience aggregate RMSE across all horizons.

    Shape notes
    -----------
    horizon_fits : list of length num_layers
        Each element covers calibration at one soil horizon.
    """
    model_index: int
    horizon_fits: list[HorizonWaxmanSmitsParams]

    @property
    def rmse(self) -> float:
        """
        Mean RMSE across **calibrated** horizons only — used for ensemble
        ranking (Steps 3/4).

        Horizons with no TDR probe coverage (``is_calibrated=False``) carry
        ``rmse=NaN`` and are excluded via ``np.nanmean`` so an uncalibrated
        horizon's fabricated fit can never influence which ensemble members
        rank best. Returns ``NaN`` only in the degenerate case where *no*
        horizon in this member was calibrated.
        """
        rmses = [h.rmse for h in self.horizon_fits]
        if all(np.isnan(r) for r in rmses):
            return float("nan")
        return float(np.nanmean(rmses))

    @property
    def n_calibrated_horizons(self) -> int:
        """Number of horizons in this member that had real TDR coverage."""
        return sum(1 for h in self.horizon_fits if h.is_calibrated)

    @property
    def params(self) -> np.ndarray:
        """
        Convenience stacked view: shape (num_layers, 4) → [a, m, σ_w, σ_s] per row.
        """
        return np.array([[h.a, h.m, h.sigma_w, h.sigma_s] for h in self.horizon_fits])

    @property
    def predicted_theta(self) -> np.ndarray:
        """Concatenated predicted θ across all horizons."""
        return np.concatenate([h.predicted_theta for h in self.horizon_fits])

    @property
    def observed_theta(self) -> np.ndarray:
        """Concatenated observed θ across all horizons."""
        return np.concatenate([h.observed_theta for h in self.horizon_fits])


@dataclass
class SelectedModel:
    """
    Result container for Step 5 structural validation assessment.
    Maintains best stacking rank K, optimized cross-sections, and spatial uncertainty matrices.
    """
    K: int
    rmse: float
    ec_model: np.ndarray  # Shape: ([n_time,] num_measured_points, num_layers)
    cv_map: np.ndarray    # Shape: ([n_time,] num_measured_points, num_layers)
    ranking: np.ndarray = field(repr=False)
    coverage: np.ndarray | None = field(default=None, repr=False)
    # Shape: ([n_time,] num_measured_points, num_layers), values in [-1, 1].
    # coverage > 0  → more than half the K members gave a valid EC (well constrained).
    # coverage ≤ 0  → fewer than half returned a valid EC (hatched in plots).


# ==========================================================================
# 2. SPATIAL MATCHING: TDR PROBE LOCATIONS → EM MEASUREMENT INDICES
# ==========================================================================

@dataclass
class SamplePositionResult:
    """
    Output of ``extract_sample_positions_from_survey`` for one TDR probe.

    Attributes
    ----------
    probe_x : float
        Along-transect coordinate of the TDR probe [m or CRS units].
    probe_y : float
        Cross-transect coordinate (0.0 for 1-D transects).
    window_m : float
        Half-width of the spatial averaging window actually applied [m].
    indices : list[int]
        Row indices into the EM measurement array (axis-1 of ``models``)
        that fall within the window.  Length ≥ 1 (falls back to nearest
        point when no measurement sits within the window).
    representative_index : int
        Single representative index used when a scalar ``sample_positions``
        list is needed (index of measurement closest to ``probe_x``).
    ec_mean : np.ndarray, shape (num_layers,)
        Mean inverted EC across ``indices`` for the first ensemble member.
        Used to diagnose the extraction quality.
    ec_std : np.ndarray, shape (num_layers,)
        Standard deviation of inverted EC across ``indices`` — proxy for
        the spatial integration uncertainty described in Benoit et al. (2019).
    probe_depth : float
        Installation depth of the TDR probe [m]. ``nan`` when no depth was
        supplied to ``extract_sample_positions_from_survey``.
    layer_index : int
        0-based soil-horizon index this probe belongs to, derived from
        ``probe_depth`` and the fixed interface depths ``depths0``. ``-1``
        when no depth/``depths0`` pair was supplied (caller must then rely
        on the spatial-index-only heuristic in
        ``calibrate_petrophysical_relationship``).
    used_fallback : bool
        ``True`` when no EM measurement fell within ``window_m`` of this
        probe and the match was instead forced to the single nearest EM
        point (with a ``[WARN]`` printed at match time). This is a
        materially different, weaker match than a genuine window match —
        it indicates the EM survey geometry may not actually cover this
        probe's location — so callers (reporting, plotting, QA gates)
        should distinguish it rather than treating every
        ``representative_index`` as equally trustworthy.
    probe_id : str or None
        Human-readable identifier for this probe (e.g. the real TDR
        station name/``Unit`` from ``data_real.load_tdr_dataframe``
        for real-data runs, or a synthetic label like ``"P3"``). ``None``
        when the caller didn't supply ``probe_ids`` to
        ``extract_sample_positions_from_survey`` — diagnostic plots that
        read this field should fall back to their existing unlabeled
        behaviour in that case rather than erroring.
    """
    probe_x: float
    probe_y: float
    window_m: float
    indices: list[int]
    representative_index: int
    ec_mean: np.ndarray
    ec_std: np.ndarray
    probe_depth: float = float("nan")
    layer_index: int = -1
    used_fallback: bool = False
    probe_id: str | None = None


def layer_index_from_depth(probe_depth: float, depths0: Sequence[float]) -> int:
    """
    Map a TDR probe's installation depth to its 0-based soil-horizon index.

    ``depths0`` gives the fixed interface depths (length ``num_layers - 1``)
    used in ``setInit``/``generate_ensemble_single_file``. Horizon 0 spans
    ``[0, depths0[0])``, horizon ``i`` spans ``[depths0[i-1], depths0[i])``,
    and the last horizon spans ``[depths0[-1], +inf)``.

    Parameters
    ----------
    probe_depth : float
        TDR probe installation depth [m], positive downward.
    depths0 : sequence of float
        Layer interface depths [m], strictly increasing.

    Returns
    -------
    int
        0-based horizon index in ``[0, len(depths0)]``.
    """
    return int(np.searchsorted(np.asarray(depths0, dtype=float), probe_depth, side="right"))


def _probe_distances(
    x_em: np.ndarray,
    y_em: np.ndarray,
    x_probe: float,
    y_probe: float,
) -> tuple[np.ndarray, bool]:
    """
    Compute distances from every EM measurement to a single TDR probe.

    Decision rule
    -------------
    * If ``y_probe != 0.0`` **or** ``y_em`` contains non-zero values the survey
      has a real cross-transect spread → use full Euclidean (XY) distance.
    * Otherwise (pure 1-D transect, all y = 0) → use |Δx| only, which is
      numerically identical but makes the 1-D case explicit in diagnostics.

    Returns
    -------
    dist : np.ndarray, shape (num_measured_points,)
        Distance from each EM measurement to the probe [same units as CRS].
    used_2d : bool
        True when Euclidean XY distance was applied (logged by the caller).
    """
    has_y_spread = (y_probe != 0.0) or np.any(y_em != 0.0)
    if has_y_spread:
        dist = np.sqrt((x_em - x_probe) ** 2 + (y_em - y_probe) ** 2)
        return dist, True
    else:
        return np.abs(x_em - x_probe), False


def extract_sample_positions_from_survey(
    survey_df,                          # k.surveys[0].df  (pandas DataFrame)
    models: np.ndarray,                 # shape (ensemble_size, num_measured_points, num_layers)
    probe_x_coords: Sequence[float],    # along-transect x of each TDR sensor [same CRS as df['x']]
    window_m: float = 1.0,             # half-width of horizontal averaging window [m]
    probe_y_coords: Sequence[float] | None = None,  # cross-transect y; triggers XY distance automatically
    probe_depths: Sequence[float] | None = None,    # installation depth [m] of each TDR probe
    depths0: Sequence[float] | None = None,         # fixed layer interface depths [m]
    probe_ids: Sequence[str] | None = None,         # human-readable ID per probe, same order
) -> list[SamplePositionResult]:
    """
    Match TDR probe field coordinates to EM measurement indices, applying a
    spatial averaging window as described in the article (Benoit et al., 2019;
    Dimech et al., 2023).

    Workflow
    --------
    For each TDR probe at coordinate ``x_probe``:

    1. Compute the distance from every EM measurement to the probe.
       For 1-D transects this is ``|x_em - x_probe|``; for 2-D surveys
       set ``use_2d_distance=True`` to use full Euclidean distance.

    2. Keep all EM measurements within ``window_m`` of the probe.
       If none fall within the window, fall back to the single nearest point
       (with a warning) — this mirrors the ERT approach of choosing the
       integration width that best matches the water content.

    3. Average the inverted EC across those indices (first ensemble member)
       to give ``ec_mean`` and ``ec_std`` per layer. ``ec_std`` is the spatial
       integration uncertainty analogous to the ERT standard deviation over
       the integration width.

    4. Return the index of the measurement *closest* to the probe as
       ``representative_index`` — this is the scalar passed to
       ``calibrate_petrophysical_relationship`` as ``sample_positions``.
       When multiple indices fall in the window the closest one is used to
       avoid biasing the pairing toward one edge of the window.

    Parameters
    ----------
    survey_df : pd.DataFrame
        ``k.surveys[0].df`` after ``createSurvey`` (and optionally
        ``convertFromNMEA``).  Must contain columns ``'x'`` and ``'y'``.
    models : np.ndarray, shape (ensemble_size, num_measured_points, num_layers)
        Inverted EC ensemble from ``generate_ensemble_single_file``.
    probe_x_coords : sequence of float
        Along-transect x coordinates of the TDR sensors in the same CRS
        as ``survey_df['x']``.
    window_m : float, optional
        Half-width of horizontal averaging window in metres (default 1.0 m).
        The article recommends testing several widths and choosing the one
        that best minimises RMSE against observed θ.  Start with the EM
        instrument's nominal footprint (~coil spacing / 2) and tune.
    probe_y_coords : sequence of float, optional
        Cross-transect y coordinates of each TDR probe.  When any value is
        non-zero, or when ``survey_df`` contains non-zero ``'y'`` values,
        full Euclidean XY distance is used automatically.  Pass all zeros
        (or omit) for pure 1-D transects — the |Δx|-only path is then taken.
    probe_depths : sequence of float, optional
        Installation depth [m] of each TDR probe, same order as
        ``probe_x_coords``. When supplied together with ``depths0``, each
        probe is mapped to its true soil horizon via ``layer_index_from_depth``
        and the result's ``layer_index`` field is populated. This is what
        lets ``calibrate_petrophysical_relationship`` route each observation
        to the horizon it actually came from instead of relying on a
        spatial-index stride/modulo heuristic.
    depths0 : sequence of float, optional
        Fixed layer interface depths [m] (same array passed to ``setInit``).
        Required alongside ``probe_depths`` to compute ``layer_index``;
        ignored if ``probe_depths`` is not given.
    probe_ids : sequence of str, optional
        Human-readable identifier for each probe, same order/length as
        ``probe_x_coords`` (e.g. real TDR station names from
        ``probe_table['station_name']``, or synthetic ``PROBE_LABELS``).
        Stored on each result's ``.probe_id`` so downstream diagnostic
        plots (``plot_probe_matching``, ``plot_calibration_density_*``)
        can label markers by sensor instead of by anonymous index. When
        omitted, ``.probe_id`` is ``None`` for every result and those
        plots fall back to their existing unlabeled behaviour.

    Returns
    -------
    list of SamplePositionResult
        One entry per TDR probe, in the same order as ``probe_x_coords``.

    Notes
    -----
    EM spatial resolution is governed by the instrument coil spacing and
    orientation (typically 0.32–3.66 m for Dualem / CMD instruments), while
    TDR sensors integrate over a few centimetres.  A window of 0.5–2× the
    coil spacing is a reasonable starting point (Calamita et al., 2012).
    """
    import pandas as pd

    if (probe_depths is None) != (depths0 is None):
        raise ValueError(
            "probe_depths and depths0 must be supplied together — got "
            f"probe_depths={'set' if probe_depths is not None else 'None'}, "
            f"depths0={'set' if depths0 is not None else 'None'}. "
            "Pass both to enable depth-aware layer routing, or neither to "
            "fall back to the spatial-index heuristic."
        )

    x_em = survey_df['x'].to_numpy(dtype=float)
    y_em = survey_df['y'].to_numpy(dtype=float) if 'y' in survey_df.columns else np.zeros_like(x_em)

    probe_x_coords = list(probe_x_coords)
    if probe_y_coords is None:
        probe_y_coords = [0.0] * len(probe_x_coords)

    if probe_depths is not None:
        probe_depths = list(probe_depths)
        if len(probe_depths) != len(probe_x_coords):
            raise ValueError(
                f"probe_depths has {len(probe_depths)} entries but "
                f"probe_x_coords has {len(probe_x_coords)}; each TDR probe "
                "needs exactly one depth."
            )
    else:
        probe_depths = [float("nan")] * len(probe_x_coords)

    if probe_ids is not None:
        probe_ids = list(probe_ids)
        if len(probe_ids) != len(probe_x_coords):
            raise ValueError(
                f"probe_ids has {len(probe_ids)} entries but "
                f"probe_x_coords has {len(probe_x_coords)}; each TDR probe "
                "needs exactly one ID (or pass probe_ids=None to skip labeling)."
            )
    else:
        probe_ids = [None] * len(probe_x_coords)

    # Use only the first ensemble member, first time step for diagnostic EC statistics.
    # For 4-D (time-lapse): models[0] is (n_time, num_measured_points, num_layers);
    # take time step 0 for the spatial window diagnostics.
    if models.ndim == 4:
        first_model = models[0, 0]  # (num_measured_points, num_layers)
    else:
        first_model = models[0]     # (num_measured_points, num_layers)

    results: list[SamplePositionResult] = []
    _2d_logged = False

    for x_probe, y_probe, depth_probe, id_probe in zip(
        probe_x_coords, probe_y_coords, probe_depths, probe_ids
    ):
        dist, used_2d = _probe_distances(x_em, y_em, x_probe, y_probe)
        if used_2d and not _2d_logged:
            print("[METHODOLOGY] Non-zero Y coordinates detected — using XY Euclidean "
                  "distance for TDR influence window.")
            _2d_logged = True

        within = np.where(dist <= window_m)[0]

        used_fallback = False
        if len(within) == 0:
            nearest = int(np.argmin(dist))
            used_fallback = True
            print(
                f"[WARN] No EM measurement within {window_m} m of probe at x={x_probe:.2f}. "
                f"Falling back to nearest point (index {nearest}, dist={dist[nearest]:.2f} m). "
                f"Consider increasing window_m."
            )
            within = np.array([nearest])

        # Representative index: closest measurement to the probe within the window
        rep_idx = int(within[np.argmin(dist[within])])

        ec_window = first_model[within, :]        # shape (n_in_window, num_layers)
        ec_mean   = ec_window.mean(axis=0)        # shape (num_layers,)
        ec_std    = ec_window.std(axis=0) if len(within) > 1 else np.zeros(first_model.shape[1])

        layer_idx = layer_index_from_depth(depth_probe, depths0) if depths0 is not None else -1

        results.append(SamplePositionResult(
            probe_x=float(x_probe),
            probe_y=float(y_probe),
            window_m=window_m,
            indices=within.tolist(),
            representative_index=rep_idx,
            ec_mean=ec_mean,
            ec_std=ec_std,
            probe_depth=float(depth_probe),
            layer_index=layer_idx,
            used_fallback=used_fallback,
            probe_id=(str(id_probe) if id_probe is not None else None),
        ))

    return results


def sample_layer_indices_from_results(
    results: list[SamplePositionResult],
) -> list[int]:
    """
    Convenience extractor: returns the per-probe ``layer_index`` list ready
    to pass into ``calibrate_petrophysical_relationship`` as
    ``probe_layer_indices``. Entries are ``-1`` for any probe whose depth
    wasn't supplied to ``extract_sample_positions_from_survey``.
    """
    return [r.layer_index for r in results]


def sample_positions_from_results(
    results: list[SamplePositionResult],
) -> list[int]:
    """
    Convenience extractor: returns the scalar ``representative_index`` list
    ready to pass into ``calibrate_petrophysical_relationship`` as
    ``sample_positions``.
    """
    return [r.representative_index for r in results]


@dataclass
class ProbeMatchingSummary:
    """
    Aggregate diagnostic summary of how a batch of TDR probes matched onto
    EM measurement-point indices (see ``extract_sample_positions_from_survey``).

    Surfaces the many-to-one collapse and window-fallback situations that
    ``extract_sample_positions_from_survey`` already knows about internally
    (via ``representative_index`` / ``used_fallback`` on each
    ``SamplePositionResult``) but previously never reported anywhere other
    than an easy-to-miss per-probe ``print()``.

    Attributes
    ----------
    n_probes : int
        Total number of TDR probes matched.
    n_distinct_em_points : int
        Number of distinct EM ``representative_index`` values used across
        all probes. Small relative to ``n_probes`` means many probes are
        being calibrated against essentially the same EC reading.
    largest_group_size : int
        Size of the largest group of probes sharing the same
        ``representative_index`` (i.e. the worst-case collapse).
    largest_group_em_index : int
        The EM measurement-point index responsible for ``largest_group_size``.
    largest_group_fraction : float
        ``largest_group_size / n_probes`` — the fraction of ALL probes that
        collapse onto that single worst-case EM point. This is the quantity
        the warning/error thresholds below are compared against.
    n_fallback_matches : int
        Number of probes for which no EM measurement fell within
        ``window_m`` and the match was forced to the nearest point instead
        of a genuine window match (``SamplePositionResult.used_fallback``).
    group_sizes : dict[int, int]
        Full histogram: EM index -> number of probes matched to it.
    level : str
        ``"OK"``, ``"WARN"``, or ``"ERROR"`` depending on whether
        ``largest_group_fraction`` exceeds ``warn_fraction_threshold`` /
        ``error_fraction_threshold``.
    message : str
        Human-readable summary line, ready to log/report as-is.
    """
    n_probes: int
    n_distinct_em_points: int
    largest_group_size: int
    largest_group_em_index: int
    largest_group_fraction: float
    n_fallback_matches: int
    group_sizes: dict[int, int]
    level: str
    message: str


def summarize_probe_matching(
    results: list[SamplePositionResult],
    warn_fraction_threshold: float = 0.5,
    error_fraction_threshold: float | None = None,
) -> ProbeMatchingSummary:
    """
    Summarize a batch of TDR-probe-to-EM-point matches produced by
    ``extract_sample_positions_from_survey``, so many-to-one collapses and
    window-fallback matches are visible as a single reportable object
    instead of requiring per-probe log inspection.

    Parameters
    ----------
    results : list of SamplePositionResult
        Output of ``extract_sample_positions_from_survey`` for one batch of
        probes (e.g. all synthetic SWC probes, or all real TDR stations).
    warn_fraction_threshold : float, optional
        If the largest single-EM-point group contains more than this
        fraction of ALL probes, ``level`` is set to ``"WARN"``. Default 0.5
        (i.e. more than half of all probes sharing one EM point).
    error_fraction_threshold : float or None, optional
        If given and the largest-group fraction exceeds it, ``level`` is
        set to ``"ERROR"`` instead of ``"WARN"`` — callers can use this to
        fail fast rather than silently proceeding with degraded
        calibration quality. ``None`` (default) never escalates to ERROR;
        the pipeline only warns.

    Returns
    -------
    ProbeMatchingSummary
    """
    from collections import Counter

    n_probes = len(results)
    if n_probes == 0:
        return ProbeMatchingSummary(
            n_probes=0, n_distinct_em_points=0, largest_group_size=0,
            largest_group_em_index=-1, largest_group_fraction=0.0,
            n_fallback_matches=0, group_sizes={}, level="OK",
            message="[MATCHING] No probes to summarize.",
        )

    rep_indices = [r.representative_index for r in results]
    counts = Counter(rep_indices)
    group_sizes = dict(counts)

    n_distinct = len(counts)
    largest_em_idx, largest_size = max(counts.items(), key=lambda kv: kv[1])
    largest_fraction = largest_size / n_probes

    n_fallback = sum(1 for r in results if r.used_fallback)

    level = "OK"
    if error_fraction_threshold is not None and largest_fraction > error_fraction_threshold:
        level = "ERROR"
    elif largest_fraction > warn_fraction_threshold:
        level = "WARN"

    message = (
        f"[MATCHING] {n_distinct} distinct EM point(s) used out of {n_probes} probe(s). "
        f"Largest group: {largest_size} probe(s) -> EM index {largest_em_idx} "
        f"({largest_fraction*100:.1f}% of all probes). "
        f"Window-fallback (nearest-point) matches: {n_fallback}/{n_probes}."
    )
    if level == "WARN":
        message += (
            f"  [WARN] largest-group fraction {largest_fraction*100:.1f}% exceeds "
            f"warn threshold {warn_fraction_threshold*100:.1f}% — many probes are "
            f"effectively calibrated against one EM reading."
        )
    elif level == "ERROR":
        message += (
            f"  [ERROR] largest-group fraction {largest_fraction*100:.1f}% exceeds "
            f"error threshold {error_fraction_threshold*100:.1f}%."
        )

    return ProbeMatchingSummary(
        n_probes=n_probes,
        n_distinct_em_points=n_distinct,
        largest_group_size=largest_size,
        largest_group_em_index=int(largest_em_idx),
        largest_group_fraction=largest_fraction,
        n_fallback_matches=n_fallback,
        group_sizes=group_sizes,
        level=level,
        message=message,
    )


def _is_timelapse(fnames) -> bool:
    """Return True when *fnames* is a non-string sequence (i.e. a list of files)."""
    return not isinstance(fnames, str)


def load_survey_positions(fnames, targetProjection: str | None = None,
                          em_unit: str = "ppt"):
    """
    Load EM measurement positions from one survey CSV (or the first time-step
    of a time-lapse stack) without running inversion.

    Returns the survey DataFrame with columns ``'x'`` and ``'y'`` in the
    requested projection (or raw Longitude/Latitude when no projection given).

    Parameters
    ----------
    fnames : str or list of str
        Single survey file path **or** list of time-step survey file paths.
        When a list is supplied the first file is loaded after ``trimSurveys``
        so position coordinates are consistent with the trimmed ensemble.
    targetProjection : str, optional
        EPSG code (e.g. ``'EPSG:27700'``) to reproject NMEA coordinates.
        Pass the same value used in ``generate_ensemble_em_survey``.
    em_unit : str, optional
        Physical unit of the quadrature/in-phase columns: ``'ppm'`` for GEM-2
        instruments, ``'ppt'`` for GF Instruments (default).  Passed directly
        to ``createSurvey`` / ``createTimeLapseSurvey`` so ECa conversion is
        correct even in this position-only load.

    Returns
    -------
    pd.DataFrame
        The survey measurement table (``k.surveys[0].df``).
    """
    k = Problem()
    if _is_timelapse(fnames):
        k.createTimeLapseSurvey(list(fnames), targetProjection=targetProjection,
                                unit=em_unit)
        k.trimSurveys()
    else:
        k.createSurvey(fnames, targetProjection=targetProjection, unit=em_unit)
    return k.surveys[0].df


def extract_conductivity_pstd(pstd_array: np.ndarray, num_layers: int) -> np.ndarray:
    """
    Slice out the per-layer CONDUCTIVITY columns of an emagpy ``k.pstds[i]``
    array (MCMC/ROPE/DREAM/SCEUA-based inversion methods only; all-zero for
    every other method).

    Background
    ----------
    For MCMC-family methods, ``Problem.invert()`` runs a real posterior
    sampler (via ``spotpy``) at every measurement point, but then collapses
    it down to just two things per point: the lowest-misfit sample (which
    becomes ``k.models[i]`` / ``k.depths[i]``) and a per-FREE-PARAMETER
    standard deviation computed from the best 10% of the chain (``k.pstds[i]``,
    shape ``(num_measured_points, n_free_depths + n_free_conductivities)``).
    The raw posterior chain itself is never retained by emagpy.

    Column layout of ``k.pstds[i]`` is ``[free_depth_stds..., conductivity_stds]``
    (this mirrors the ``vd``/``vc`` ordering used internally by
    ``Problem.invert()``). Since this pipeline never sets ``fixedConds`` in
    any ``k.setInit()`` call, EVERY conductivity layer is always free, so the
    trailing ``num_layers`` columns of ``k.pstds[i]`` are always exactly the
    per-layer conductivity posterior std -- regardless of whether depths
    were fixed or free (which only changes how many LEADING columns exist).
    Slicing the trailing ``num_layers`` columns therefore always gives a
    result directly comparable, index-for-index, to a model/EC array shaped
    ``(num_measured_points, num_layers)``.

    Parameters
    ----------
    pstd_array : np.ndarray
        ``k.pstds[i]`` for one epoch, shape ``(num_measured_points,
        n_free_params)``, OR any array carrying that as its trailing two
        axes -- e.g. this pipeline's own ``pstds_clean``, shaped
        ``(ensemble_size, [n_time,] num_measured_points, n_free_params)``.
        Any number of leading (ensemble/time) axes is accepted unchanged.
        ``n_free_params >= num_layers`` whenever this pipeline's own
        ``k.setInit()`` convention (all conductivities free) holds.
    num_layers : int
        Number of soil layers (== conductivity columns == ``cond_ranges`` length).

    Returns
    -------
    np.ndarray, same leading shape as ``pstd_array`` but with the trailing
    axis reduced to ``num_layers``
        Per-layer conductivity MCMC posterior std [mS/m]. All-zero for
        non-MCMC-family inversion methods (matches emagpy's own convention:
        ``self.pstds`` is zero-initialised and only the MCMC branch ever
        writes non-zero values into it). NaN-filled if the trailing axis
        has fewer than ``num_layers`` columns, which would indicate an
        emagpy version/convention mismatch rather than normal operation.
    """
    pstd_array = np.asarray(pstd_array, dtype=float)
    if pstd_array.ndim < 2 or pstd_array.shape[-1] < num_layers:
        return np.full(pstd_array.shape[:-1] + (num_layers,), np.nan, dtype=float)
    return pstd_array[..., -num_layers:]


# ==========================================================================
# 3. ENSEMBLE GENERATION UTILITIES
# ==========================================================================

def drop_coils_from_problem(k: "Problem", drop_coils, dry_run: bool = False) -> list[str]:
    """
    Remove one or more coils/frequency channels from an already-loaded
    emagpy ``Problem`` before inversion (thin wrapper around emagpy's own
    ``Problem.removeCoil()``).

    Typical use: a specific GEM-2 frequency channel (e.g. 425 Hz) is too
    noisy and should be excluded from the inversion entirely.

    Must be called **after** ``k.createSurvey()`` / ``k.createTimeLapseSurvey()``
    (coils are only known once a survey is loaded) and **before**
    ``k.setInit()`` / ``k.invert()``.

    Parameters
    ----------
    k : emagpy.Problem
        Problem instance with at least one survey already loaded.
    drop_coils : None, str, int, float, or sequence thereof
        Coil(s) to remove. Each entry may be:

        * the exact emagpy coil name, e.g. ``'HCP1.66f425h1'``
          (see ``k.coils`` for the loaded names),
        * a frequency in Hz, e.g. ``425`` or ``425.0``
          (matched against ``k.freqs``), or
        * a substring that uniquely identifies one coil name,
          e.g. ``'f425'`` or ``'425'``.

        Pass ``None`` (default) to remove nothing — a no-op, so this can
        always be called unconditionally.
    dry_run : bool, optional
        If True, only resolve which coil names match ``drop_coils`` and
        return them — do **not** call ``k.removeCoil()``. Used by raw-data
        diagnostic plots (e.g. ``plot_raw_apparent_conductivity``) to find
        out which panel(s) correspond to a dropped channel, while still
        showing that channel's raw signal for visual comparison. Default
        ``False`` (actually removes the coil, as before).

    Returns
    -------
    list of str
        The exact coil names that were removed (for logging/reporting).

    Raises
    ------
    ValueError
        If a requested spec matches zero or more-than-one loaded coil.
    """
    if drop_coils is None:
        return []
    if isinstance(drop_coils, (str, int, float)):
        drop_coils = [drop_coils]

    removed: list[str] = []
    for spec in drop_coils:
        coil_name = None
        if isinstance(spec, (int, float)) and not isinstance(spec, bool):
            matches = [c for c, f in zip(k.coils, k.freqs) if np.isclose(f, float(spec))]
            if len(matches) == 1:
                coil_name = matches[0]
            elif len(matches) > 1:
                raise ValueError(
                    f"drop_coils frequency {spec} Hz matches multiple coils "
                    f"{matches}; use the exact coil name instead."
                )
        else:
            spec_str = str(spec)
            if spec_str in k.coils:
                coil_name = spec_str
            else:
                matches = [c for c in k.coils if spec_str.lower() in c.lower()]
                if len(matches) == 1:
                    coil_name = matches[0]
                elif len(matches) > 1:
                    raise ValueError(
                        f"drop_coils spec '{spec}' matches multiple coils "
                        f"{matches}; use the exact coil name instead."
                    )
        if coil_name is None:
            raise ValueError(
                f"drop_coils spec '{spec}' did not match any loaded coil. "
                f"Available coils: {k.coils}"
            )
        if not dry_run:
            k.removeCoil(coilName=coil_name)
        removed.append(coil_name)
    return removed


def build_conductivity_ensemble(
    cond_ranges: Sequence[Sequence[float]],
) -> list[list[float]]:
    """
    Computes a full Cartesian-product grid expansion of starting conductivities.
    Dynamically scales to any layer architecture defined by the parameter input bounds.
    """
    return [list(combo) for combo in itertools.product(*cond_ranges)]


def generate_ensemble_em_survey(
    fnames,
    param_grid: dict,
    cond_ranges: Sequence[Sequence[float]],
    depths0: Sequence[float],
    fixed_depths: bool | Sequence[bool] = True,
    targetProjection: str | None = None,
    forwardModel: str = "CS",
    gamma: float = 0.0,
    verbose: bool = False,
    em_unit: str = "ppt",
    drop_coils=None,
    inversion_mode: str = "independent",
    spatial_match_tolerance_m: float = 1e-6,
) -> tuple[np.ndarray, list[dict]]:
    """
    Step 1 Core Engine: Runs sequential inversions for one ensemble member set.

    Supports **both** single-time and time-lapse EM surveys:

    * **Single file** (``fnames`` is a ``str``):
      Always uses ``createSurvey`` → ``invert``.
      Returns ``(ensemble_size, num_measured_points, num_layers)``.

    * **Time-lapse** (``fnames`` is a ``list[str]``), two modes:

      ``inversion_mode='independent'`` (default)
          Each survey file is inverted separately with ``createSurvey``.
          No ``trimSurveys``, no temporal coupling.  Results from every
          epoch are matched onto a shared UNION reference grid by actual
          (x, y) coordinate via
          ``geometry_utils.match_epochs_to_reference_grid`` -- the same
          matcher ``run_pipeline.py``'s parallel worker uses -- rather than
          a row-count/row-position clip. An epoch that doesn't cover a
          given reference point gets NaN there; no epoch's own points are
          ever reordered, subsetted, or silently paired with an unrelated
          location under a shared index. See
          ``spatial_match_tolerance_m`` below. Prefer this mode for real
          field data where survey paths differ across epochs, or whenever
          you do not want temporal regularisation.

      ``inversion_mode='timelapse'``
          Uses ``createTimeLapseSurvey`` → ``trimSurveys`` → ``invert``
          with ``gamma`` temporal smoothing.  emagpy's ``trimSurveys``
          truncates every epoch to the shortest survey's row count and
          aligns them by row index (not by coordinate).  Suitable only
          when surveys were walked in the same spatial order each epoch.
          Returns ``(ensemble_size, n_time, num_measured_points, num_layers)``.

    Parameters
    ----------
    fnames : str or list of str
        Single survey CSV path or ordered list of time-step CSV paths.
    param_grid : dict
        Mapping of inversion parameter name → list of candidate values.
        All combinations are enumerated (Cartesian product).
    cond_ranges : sequence of [min, max]
        Per-layer starting conductivity bounds [mS/m]. Two values per layer;
        both endpoints are included as starting conditions.
    depths0 : sequence of float
        Initial layer interface depths [m] (length = num_layers − 1).
        When ``fixed_depths`` is ``True`` (default) or all-True these values
        are kept constant during inversion.  When any interface is unfixed the
        inversion is free to adjust it, allowing the data to inform layering.
    fixed_depths : bool or sequence of bool, optional
        Controls which interface depths are held constant during inversion.

        * ``True``  (default) — all interfaces fixed; classic behaviour.
        * ``False``           — all interfaces free (fully flexible layering).
        * sequence of bool, length = ``len(depths0)`` — mixed mode: fix only
          the interfaces listed as ``True``, e.g. ``[True, False, True]``
          fixes the first and last boundaries but lets the middle one float.

        When interfaces are free, ``depths0`` still provides their starting
        values; the inversion then perturbs them to minimise misfit.
    targetProjection : str, optional
        EPSG code for coordinate reprojection.
    forwardModel : str, optional
        EMagPy forward model key (default ``'CS'``).
    gamma : float, optional
        Temporal smoothing factor between surveys.  Only meaningful when
        ``inversion_mode='timelapse'``; silently ignored in independent mode.
        ``0.0`` (default) disables temporal regularisation.
    inversion_mode : {'independent', 'timelapse'}, optional
        Controls how multiple survey files are handled when ``fnames`` is a
        list (time-lapse context).  Default ``'independent'``.  See class
        docstring for the distinction.  Ignored when ``fnames`` is a single
        string (always ``createSurvey`` in that case).
    verbose : bool, optional
        Print progress messages.
    em_unit : str, optional
        Physical unit of quadrature/in-phase columns in the survey CSV.
        ``'ppm'`` for GEM-2 instruments, ``'ppt'`` for GF Instruments
        (default).  Passed to ``createSurvey`` / ``createTimeLapseSurvey``
        so the LIN-ECa conversion uses the correct divisor (1e6 vs 1e3).
    drop_coils : None, str, int, float, or sequence thereof, optional
        Coil(s)/frequency channel(s) to exclude from the survey before
        inversion (e.g. a noisy GEM-2 frequency). See
        ``drop_coils_from_problem`` for accepted formats (exact coil name,
        frequency in Hz, or a unique substring like ``'f425'``). Applied
        identically to every ensemble member. Default ``None`` (no coils
        dropped).
    spatial_match_tolerance_m : float, optional
        Only used when ``inversion_mode='independent'`` and ``fnames`` is a
        list. Distance tolerance (metres) for matching one epoch's point to
        another's as "the same location" when building the shared reference
        grid. Default ``1e-6`` m is appropriate for synthetic data, whose
        coordinates are bit-identical across epochs by construction (only
        float round-trip noise needs covering). For real field data, pass a
        value matching your GPS positional uncertainty (e.g. ``2.0`` m),
        the same way ``run_pipeline.py`` uses ``SPATIAL_MATCH_TOLERANCE_M``.

    Returns
    -------
    models : np.ndarray
        Single: ``(ensemble_size, num_measured_points, num_layers)``
        Time-lapse: ``(ensemble_size, n_time, num_measured_points, num_layers)``
    param_sets : list of dict
        Per-member metadata (keys: inversion params + ``'conds0'``).
    """
    timelapse = _is_timelapse(fnames)
    n_layers = len(cond_ranges)
    if len(depths0) != n_layers - 1:
        raise ValueError(
            f"depths0 must contain exactly {n_layers - 1} interface depths "
            f"for {n_layers} structural layers, got {len(depths0)}."
        )

    # Resolve fixed_depths → a plain list[bool] of length len(depths0)
    n_interfaces = len(depths0)
    if isinstance(fixed_depths, bool):
        fixed_depths_list: list[bool] = [fixed_depths] * n_interfaces
    else:
        fixed_depths_list = list(fixed_depths)
        if len(fixed_depths_list) != n_interfaces:
            raise ValueError(
                f"fixed_depths has {len(fixed_depths_list)} entries but "
                f"depths0 has {n_interfaces} interfaces; lengths must match."
            )

    if verbose:
        free_indices = [i for i, f in enumerate(fixed_depths_list) if not f]
        if not free_indices:
            print("[METHODOLOGY] All layer interfaces fixed (classic mode).")
        elif len(free_indices) == n_interfaces:
            print("[METHODOLOGY] All layer interfaces free — depths will be optimised.")
        else:
            print(
                f"[METHODOLOGY] Mixed depth mode: interfaces {free_indices} are free "
                f"(depths0 values used as starting points)."
            )

    cond_sets = build_conductivity_ensemble(cond_ranges)
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]

    inversion_param_sets = (
        [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]
        if keys else [{}]
    )

    param_sets: list[dict] = []
    for conds0, inv_params in itertools.product(cond_sets, inversion_param_sets):
        member = dict(inv_params)
        member["conds0"] = conds0
        param_sets.append(member)

    _mode_label = (
        "single-file" if not timelapse else
        f"time-lapse/{inversion_mode}"
    )
    if verbose:
        print(f"[METHODOLOGY] Computing {len(param_sets)} {_mode_label} grid-expansion inversions...")
        if timelapse and inversion_mode == "independent":
            print("[METHODOLOGY] independent mode: each epoch inverted separately "
                  "(no trimSurveys, no temporal coupling).")
        elif timelapse and inversion_mode == "timelapse":
            print(f"[METHODOLOGY] timelapse mode: createTimeLapseSurvey + "
                  f"trimSurveys + gamma={gamma}.")

    models: list[np.ndarray] = []

    # Per-layer (min, max) box constraints for the L-BFGS-B solver, derived
    # from the same cond_ranges used to seed conds0. Without this, EMagPy's
    # k.invert() falls back to its own internal default bnds (unrelated to
    # cond_ranges), silently capping every inverted EC value regardless of
    # how wide cond_ranges is configured.
    bnds_list = [tuple(r) for r in cond_ranges]

    for member in param_sets:
        conds0 = member["conds0"]
        invert_kwargs = {k: v for k, v in member.items() if k != "conds0"}
        invert_kwargs.setdefault("forwardModel", forwardModel)
        invert_kwargs.setdefault("bnds", bnds_list)

        if timelapse and inversion_mode == "timelapse":
            # ── True time-lapse: all epochs loaded together, trimmed to a
            #    common row count, inverted jointly with gamma smoothing ───
            k = Problem()
            k.createTimeLapseSurvey(list(fnames), targetProjection=targetProjection,
                                    unit=em_unit)
            k.trimSurveys()
            if drop_coils is not None:
                drop_coils_from_problem(k, drop_coils)
            k.setInit(
                depths0=list(depths0),
                conds0=list(conds0),
                fixedDepths=fixed_depths_list,
            )
            k.invert(gamma=gamma, **invert_kwargs)
            # k.models: list of n_time arrays, each (n_pts_trimmed, n_layers)
            member_array = np.stack([np.asarray(m) for m in k.models], axis=0)
            # shape: (n_time, n_pts_trimmed, num_layers)

        elif timelapse and inversion_mode == "independent":
            # ── Independent: one createSurvey per epoch, no trimSurveys ────
            # This mirrors exactly what run_pipeline.py's parallel worker does.
            #
            # FIX ("spatial reference-grid matching, not row trimming"):
            # this used to do `arr[:min_pts]` -- clipping every epoch to the
            # shortest epoch's row COUNT and pairing "row i of epoch A" with
            # "row i of epoch B" purely by array position. That silently
            # mispairs physically different locations whenever epochs differ
            # in point count or walk order, and throws away every point
            # beyond the shortest epoch's count even though those points
            # were successfully inverted. Now matched by actual (x, y)
            # coordinate onto a shared union reference grid, exactly like
            # run_pipeline.py's Step 1 -- see that module's "SPATIAL
            # ASSEMBLY OF THE ENSEMBLE ACROSS EPOCHS" section.
            per_epoch: list[np.ndarray] = []
            epoch_xy: list[tuple[np.ndarray, np.ndarray]] = []
            for fname in fnames:
                k = Problem()
                k.createSurvey(fname, targetProjection=targetProjection, unit=em_unit)
                if drop_coils is not None:
                    drop_coils_from_problem(k, drop_coils)
                k.setInit(
                    depths0=list(depths0),
                    conds0=list(conds0),
                    fixedDepths=fixed_depths_list,
                )
                k.invert(**invert_kwargs)
                model_arr = np.asarray(k.models[0])  # (n_pts_t, n_layers)

                _pos_df = load_survey_positions(
                    fname, targetProjection=targetProjection, em_unit=em_unit,
                )
                _x_epoch = _pos_df["x"].to_numpy(dtype=float)
                _y_epoch = (
                    _pos_df["y"].to_numpy(dtype=float)
                    if "y" in _pos_df.columns else np.zeros_like(_x_epoch)
                )
                if len(_x_epoch) != model_arr.shape[0]:
                    # Position table and inverted model disagree on row
                    # count for this epoch -- guard defensively (truncate
                    # the shorter of the two) rather than mis-pairing
                    # coordinates with EC values.
                    _n_use = min(len(_x_epoch), model_arr.shape[0])
                    _x_epoch, _y_epoch = _x_epoch[:_n_use], _y_epoch[:_n_use]
                    model_arr = model_arr[:_n_use]

                per_epoch.append(model_arr)
                epoch_xy.append((_x_epoch, _y_epoch))

            ref_x, ref_y, matched_indices = geometry_utils.match_epochs_to_reference_grid(
                epoch_xy, tolerance_m=spatial_match_tolerance_m,
            )
            n_ref = len(ref_x)
            n_layers_out = per_epoch[0].shape[-1]
            member_array = np.full((len(per_epoch), n_ref, n_layers_out), np.nan)
            for t_idx, arr in enumerate(per_epoch):
                idx_map = matched_indices[t_idx]
                valid = idx_map >= 0
                if np.any(valid):
                    member_array[t_idx, valid] = arr[idx_map[valid]]
            # shape: (n_time, n_ref, num_layers) -- union reference grid;
            # NaN where a given epoch didn't cover that reference point.
            # NaN-safe downstream (same convention as run_pipeline.py).

        else:
            # ── Single file ─────────────────────────────────────────────────
            k = Problem()
            k.createSurvey(fnames, targetProjection=targetProjection, unit=em_unit)
            if drop_coils is not None:
                drop_coils_from_problem(k, drop_coils)
            k.setInit(
                depths0=list(depths0),
                conds0=list(conds0),
                fixedDepths=fixed_depths_list,
            )
            k.invert(**invert_kwargs)
            member_array = np.asarray(k.models[0])
            # shape: (num_measured_points, num_layers)

        models.append(member_array)

    # Stack across ensemble members
    models_array = np.stack(models, axis=0)
    # Single:          (ensemble_size, num_measured_points, num_layers)
    # Time-lapse both: (ensemble_size, n_time, num_measured_points, num_layers)
    return models_array, param_sets


# Keep backward-compatible alias
def generate_ensemble_single_file(
    fname: str,
    param_grid: dict,
    cond_ranges: Sequence[Sequence[float]],
    depths0: Sequence[float],
    fixed_depths: bool | Sequence[bool] = True,
    targetProjection: str | None = None,
    forwardModel: str = "CS",
    verbose: bool = False,
    drop_coils=None,
) -> tuple[np.ndarray, list[dict]]:
    """Backward-compatible wrapper — delegates to ``generate_ensemble_em_survey``."""
    return generate_ensemble_em_survey(
        fnames=fname,
        param_grid=param_grid,
        cond_ranges=cond_ranges,
        depths0=depths0,
        fixed_depths=fixed_depths,
        targetProjection=targetProjection,
        forwardModel=forwardModel,
        verbose=verbose,
        drop_coils=drop_coils,
    )


# ==========================================================================
# 4. SENSITIVITY AND VARIANCE ENGINES
# ==========================================================================

def compute_param_sensitivity(
    models: np.ndarray,
    param_metadata: list[dict]
) -> dict[str, dict[str, np.ndarray]]:
    """
    Quantifies inversion performance variance isolated by processing parameters.
    Aggregates spatial variance maps across measurement locations to yield clean
    layer dominance metrics.

    Handles both:
    * **3D** ``(ensemble_size, num_measured_points, num_layers)`` — single-time
    * **4D** ``(ensemble_size, n_time, num_measured_points, num_layers)`` — time-lapse;
      the time axis is mean-collapsed before computing sensitivity so the result
      is always a per-layer vector.
    """
    target_keys = ["alpha", "beta", "method", "forwardModel"]
    results = {}

    # Collapse time axis for time-lapse arrays → uniform 3D view
    if models.ndim == 4:
        work = models.mean(axis=1)  # (ensemble_size, num_measured_points, num_layers)
    else:
        work = models

    for key in target_keys:
        indices = [i for i, p in enumerate(param_metadata) if key in p]
        if indices:
            sub_stack = work[indices]        # (sub_ensemble, positions, layers)
            std_map = np.std(sub_stack, axis=0)            # (positions, layers)
            mean_layer_sensitivity = np.mean(std_map, axis=0)  # (layers,)
            results[key] = {"__sensitivity__": mean_layer_sensitivity}

    return results


# ==========================================================================
# 4B. TEMPERATURE CORRECTION  (Step 1 -> Step 2 boundary)
# ==========================================================================

def apply_temperature_correction(
    models: np.ndarray,
    temp_profiles,
    alpha: float = 0.023,
    T_ref: float = 20.0,
) -> np.ndarray:
    """
    FIX 3.6 — Temperature-correct inverted EC models before petrophysical
    calibration, per Loiseau et al. (2026), Eq. 1.

    The paper defines the correction in the resistivity domain:

        rho_corr = rho / [1 + alpha * (T - T_ref)]                      (Eq. 1)

    This codebase's ``models`` arrays are bulk electrical CONDUCTIVITY
    [mS/m] (sigma = 1/rho), not resistivity. Substituting sigma = 1/rho into
    Eq. 1 and solving for sigma_corr gives the algebraically equivalent
    conductivity-domain correction implemented here:

        sigma_corr = sigma * [1 + alpha * (T - T_ref)]

    Parameters
    ----------
    models : np.ndarray
        Inverted EC ensemble straight out of Step 1
        (``generate_ensemble_em_survey``):
          * single-survey -> ``(ensemble_size, num_measured_points, num_layers)``
          * time-lapse    -> ``(ensemble_size, n_time, num_measured_points, num_layers)``
    temp_profiles : float, array-like, or np.ndarray
        Soil temperature [deg C], broadcastable against ``models.shape[1:]``
        (every axis except the leading ensemble axis -- every ensemble
        member sees the same field temperature). Common shapes:
          * scalar                   -> one uniform T for everything
          * ``(num_layers,)``        -> one T per horizon (single-survey, or
                                          reused at every time step in
                                          time-lapse mode)
          * ``(n_time, num_layers)`` -> one T per (survey date, horizon),
                                          constant across measured positions
                                          -- the typical case when
                                          thermocouples are installed per
                                          depth but not per position
          * exact ``models.shape[1:]`` -> fully explicit per
                                          (time, position, layer) field
    alpha : float, optional
        Temperature coefficient [deg C^-1]. Default 0.023, the natural-soils
        value used by Loiseau et al. (2026) (after Travelletti et al., 2012).
    T_ref : float, optional
        Reference temperature [deg C] that all models are corrected to.
        Default 20.0, matching the paper's field application.

    Returns
    -------
    np.ndarray
        Temperature-corrected EC ensemble, same shape as ``models``.

    Notes
    -----
    Directionality follows directly from the paper's resistivity-domain
    formula via sigma = 1/rho (see derivation above) -- it has not been
    independently re-derived against an instrument calibration sheet. If a
    sanity check against known field behaviour (e.g. comparing surveys taken
    at very different soil temperatures, before/after correction) suggests
    the sign is backwards for your site, flip the sign of ``alpha`` rather
    than editing the formula.

    If your number of survey dates happens to numerically equal your number
    of measured positions, disambiguate a ``(n_time, num_layers)`` array from
    a ``(num_points, num_layers)`` array by instead passing the fully
    explicit ``models.shape[1:]``-shaped array -- shape alone can't tell
    them apart in that edge case.
    """
    models = np.asarray(models, dtype=float)
    T = np.asarray(temp_profiles, dtype=float)
    spatial_shape = models.shape[1:]   # everything except the ensemble axis

    # The one genuinely ambiguous case for plain NumPy broadcasting: a
    # (n_time, num_layers) temperature array for 4-D time-lapse models.
    # Plain broadcasting aligns shapes from the right, so (n_time, num_layers)
    # would try to broadcast n_time against num_measured_points instead of
    # against the time axis. Insert the skipped position axis explicitly.
    if (
        models.ndim == 4
        and T.ndim == 2
        and T.shape == (spatial_shape[0], spatial_shape[2])
    ):
        T = T[:, np.newaxis, :]   # (n_time, 1, num_layers)

    try:
        correction = 1.0 + alpha * (T - T_ref)
        corrected = models * correction   # ensemble axis broadcasts automatically
    except ValueError as exc:
        raise ValueError(
            f"temp_profiles with shape {np.asarray(temp_profiles).shape} cannot "
            f"be broadcast against the model's per-member shape {spatial_shape}. "
            "Provide a scalar, a (num_layers,) array, a (n_time, num_layers) "
            f"array for time-lapse data, or an array with the exact shape "
            f"{spatial_shape}."
        ) from exc

    return corrected


# ==========================================================================
# 5. WAXMAN-SMITS PETROPHYSICAL MODEL
# ==========================================================================

def waxman_smits_theta(sigma_b: np.ndarray, a: float, m: float,
                        sigma_w: float, sigma_s: float) -> np.ndarray:
    """
    Waxman-Smits (1968) forward model: inverted EC → volumetric water content.

    Derivation
    ----------
    Formation factor:  F = a / θ^m
    WS bulk EC:        σ_b = (σ_w + σ_s) / F  →  F = (σ_w + σ_s) / σ_b
    Solving for θ:     θ = (a · σ_b / (σ_w + σ_s)) ^ (1/m)

    Parameters
    ----------
    sigma_b : array-like
        Bulk electrical conductivity [mS/m] from EM inversion.
    a : float
        Tortuosity factor (> 0, commonly 1.0–2.0 for soils).
    m : float
        Cementation exponent (> 1, commonly 1.3–2.5 for soils).
    sigma_w : float
        Pore water conductivity [mS/m] (> 0).
    sigma_s : float
        Surface conductivity [mS/m] (≥ 0).

    Returns
    -------
    theta : np.ndarray
        Predicted volumetric water content [m³/m³].
    """
    sigma_b = np.asarray(sigma_b, dtype=float)
    denom = sigma_w + sigma_s
    # Guard: avoid division by zero and negative base in power
    ratio = np.clip(a * sigma_b / denom, 1e-12, None)
    return ratio ** (1.0 / m)


# Default initial parameters [a, m, σ_w, σ_s] per horizon — can be overridden
# with laboratory-calibrated values passed into calibrate_petrophysical_relationship.
DEFAULT_WS_P0 = np.array([1.5, 1.8, 50.0, 5.0])

# Parameter bounds for scipy.optimize.curve_fit
#   a      : (0.5,  5.0)
#   m      : (1.0,  4.0)
#   σ_w    : (1.0, 1000.0)
#   σ_s    : (0.0,  200.0)
WS_BOUNDS = (
    [0.5,  1.0,   1.0,  0.0],
    [5.0,  4.0, 1000.0, 200.0],
)


def resolve_ws_bounds(ws_bounds: dict | None = None) -> tuple[list[float], list[float]]:
    """Effective Waxman-Smits bounds in the ``WS_BOUNDS`` layout
    ``([lo_a, lo_m, lo_sw, lo_ss], [hi_a, hi_m, hi_sw, hi_ss])``.

    ``ws_bounds`` is an optional ``{param: [lo, hi]}`` override of any subset
    of ``a``, ``m``, ``sigma_w``, ``sigma_s`` (the scenario's ``ws_bounds``
    field); unlisted parameters keep their ``WS_BOUNDS`` default. ``None``
    returns the defaults unchanged.
    """
    lo, hi = list(WS_BOUNDS[0]), list(WS_BOUNDS[1])
    if ws_bounds:
        names = ("a", "m", "sigma_w", "sigma_s")
        unknown = set(ws_bounds) - set(names)
        if unknown:
            raise ValueError(f"ws_bounds: unknown parameter(s) {sorted(unknown)}.")
        for i, n in enumerate(names):
            if n in ws_bounds:
                l, h = (float(v) for v in ws_bounds[n])
                if not l < h:
                    raise ValueError(f"ws_bounds[{n!r}] needs lo < hi, got {ws_bounds[n]!r}.")
                lo[i], hi[i] = l, h
    return lo, hi


# ==========================================================================
# 6. PETROPHYSICAL ALIGNMENT MODULES  (horizon-aware)
# ==========================================================================

def calibrate_petrophysical_relationship(
    models: np.ndarray,
    sample_positions: Sequence[int],
    sample_times: Sequence[int],       # retained for API compatibility
    theta_obs: np.ndarray,
    depths0: Sequence[float] | None = None,
    lab_p0: np.ndarray | None = None,
    probe_layer_indices: Sequence[int] | None = None,
    depths_are_fixed: bool = True,
    fixed_params: dict[str, bool] | None = None,
    warm_start_continuation: bool = False,
    probe_depths_reflect_fitted_interfaces: bool = False,
    theta_obs_std: float | Sequence[float] | None = None,
    ws_bounds: dict | None = None,
) -> list[PetroFit]:
    """
    Step 2 Core Engine: Fits Waxman-Smits parameters **separately per soil horizon**
    for every ensemble member using a Bayesian Gaussian Prior penalty system.

    Horizon assignment
    ------------------
    Each soil layer (column index in the (num_measured_points, num_layers) model
    matrix) is treated as an independent horizon. By default (``probe_layer_indices=None``)
    ``theta_obs`` is assigned to horizons via a spatial-index stride/modulo
    heuristic. When ``probe_layer_indices`` is supplied each observation is
    routed to the horizon its probe was actually installed in.

    Parameters
    ----------
    models : np.ndarray
        Shape ``(ensemble_size, num_measured_points, num_layers)`` for single-time,
        or ``(ensemble_size, n_time, num_measured_points, num_layers)`` for
        time-lapse surveys.  For time-lapse data each sample observation is
        extracted at the time step given by ``sample_times[i]``, not from a
        time-averaged view, so that theta_obs at different campaign dates is
        matched to the correct EC snapshot.
    sample_positions : sequence of int
        Spatial indices into the ``num_measured_points`` axis at which TDR
        sensors sit.
    sample_times : sequence of int
        Time-step indices (0-based) for each TDR observation.  For single-time
        surveys pass all zeros.  For time-lapse surveys each entry selects
        which ``k.models[t]`` slice the corresponding theta_obs was measured at.
    theta_obs : np.ndarray, shape (n_samples,)
        Observed volumetric water content at ``sample_positions``.
    depths0 : sequence of float, optional
        Layer boundary depths [m].  Used only for logging/labelling.
    lab_p0 : np.ndarray, shape (num_layers, 4) or (4,), optional
        Laboratory-derived initial parameter guesses [a, m, σ_w, σ_s] per
        horizon.  Falls back to ``DEFAULT_WS_P0`` when not provided.
    probe_layer_indices : sequence of int, optional
        0-based horizon index for each entry in ``sample_positions``/``theta_obs``.
    depths_are_fixed : bool, optional
        Set to ``False`` when Step 1 was run with free (or partially free)
        layer interfaces.  When ``False`` and ``probe_layer_indices`` is not
        ``None``, a message is emitted flagging that the horizon-to-probe
        pairing is approximate, because per-layer EC values are unaffected
        by interface movement but the horizon *label* attached to each
        probe may not match the horizon it ended up in after the inversion
        moved the interfaces.  Exactly what that message says depends on
        ``probe_depths_reflect_fitted_interfaces`` (see below).  When
        ``probe_layer_indices`` is ``None`` the strided heuristic is used
        regardless, so no message is issued.  Default ``True`` (classic
        fixed-depth behaviour).
    probe_depths_reflect_fitted_interfaces : bool, optional
        Set to ``True`` when the caller already computed ``probe_layer_indices``
        using the ensemble's actual **fitted** interface depths (e.g. the
        per-member/per-position depths captured in Step 1's
        ``depths_clean``/``k.depths[0]``, aggregated into an
        ensemble-representative value) instead of the static *starting*
        ``depths0``.  This is the fix suggested by the original warning text
        ("inspect individual member interface depths before assigning
        probes to horizons"): routing against fitted depths is a
        substantially better approximation of where each probe actually
        ended up than routing against the unmoved starting depths, even
        though a single ensemble-aggregated value still isn't exactly
        per-member. When ``True`` the emitted message is downgraded from a
        "[WARN]" to an "[INFO]" and its wording reflects that fitted depths
        were used. Has no effect when ``depths_are_fixed`` is ``True`` or
        ``probe_layer_indices`` is ``None``. Default ``False`` (routing
        assumed to be against the static starting ``depths0``, the original
        behaviour).
    fixed_params : dict[str, bool], optional
        Holds one or more of the four Waxman-Smits parameters
        (``'a'``, ``'m'``, ``'sigma_w'``, ``'sigma_s'``) fixed at their
        prior value (``lab_p0`` if given, else ``DEFAULT_WS_P0``) instead of
        letting the optimizer adjust them. Implemented by pinning that
        parameter's L-BFGS-B bound to ``(p0, p0)``, so the underlying 4-D
        objective is unchanged but effectively degenerates to fewer free
        dimensions. Typical use: with only ~2 observations per horizon,
        fitting all 4 parameters is underdetermined; fixing ``m``
        (cementation exponent) and ``sigma_s`` (surface conductivity) — both
        fairly stable, texture-dependent constants — to literature/lab
        values and letting only ``a`` and ``sigma_w`` float turns a 2-data
        vs. 4-unknown problem into a much better-constrained 2-vs-2 fit.
        Keys not present, or the whole dict omitted (``None``, default),
        are left free — fully backward compatible.
    ws_bounds : dict, optional
        ``{param: [lo, hi]}`` override of the optimizer bounds for any subset
        of ``a``/``m``/``sigma_w``/``sigma_s`` (see ``resolve_ws_bounds``);
        unlisted parameters keep ``WS_BOUNDS``. ``None`` (default) keeps
        ``WS_BOUNDS`` unchanged.
    warm_start_continuation : bool, optional
        When ``True``, each ensemble member's per-horizon optimizer is
        seeded (``x0``) from the *previous* member's converged parameters
        for that horizon, instead of always restarting from the fixed prior
        ``p0``. The Gaussian prior penalty term itself still centres on the
        original ``p0`` regardless — only the optimizer's starting point
        changes. This only makes physical sense when the ensemble axis is
        an ordered sequence where consecutive members are expected to be
        similar, e.g. cumulative stack depth K = 1, 2, …, N in
        ``recalibrate_petrophysical_relationship`` (Step 4). Continuation
        smooths out the "different local minimum per K" jitter that
        otherwise produces a spiky, two-band RMSE-vs-K curve when the
        per-horizon fit is this underdetermined. Leave ``False`` (default)
        for Step 2's raw per-ensemble-member calibration, where members are
        unordered parameter-grid combinations and continuity across them
        has no meaning.
    theta_obs_std : float or sequence of float, optional
        FIX ("TDR measurement uncertainty is collected but never used to
        weight the calibration"): standard deviation [m³/m³] of ``theta_obs``
        measurement noise -- e.g. the scenario's ``tdr_noise_std`` -- either
        one value applied to every observation, or one value per observation
        (same length/order as ``theta_obs``). When given, each observation's
        contribution to the per-horizon data-misfit term (see
        ``_log_ratio_objective`` below) is weighted by
        ``1 / theta_obs_std**2``, renormalised so the weights average to 1
        across each horizon's own observations -- a noisier probe reading
        counts for proportionally less, a more precise one for
        proportionally more, without changing the overall scale of the
        misfit term (so ``lambda_reg`` below doesn't need retuning purely
        because weighting was turned on). This is a simplification, not a
        full propagation of theta-measurement noise through the
        Waxman-Smits forward model into EC-space (that would additionally
        depend on the candidate ``m`` at each optimizer iteration); it only
        captures *relative* confidence between observations. ``None``
        (default) applies uniform weighting, i.e. no behaviour change from
        before this parameter existed. Observations dropped by the
        NaN-observation filter below have their weight dropped too (the
        same mask is applied to both).

    Returns
    -------
    list of PetroFit, length ensemble_size
        Each PetroFit holds a ``horizon_fits`` list with one
        ``HorizonWaxmanSmitsParams`` per layer.
    """
    from scipy.optimize import minimize  # noqa: F401 — used in _log_ratio_objective below

    fixed_params = fixed_params or {}
    _PARAM_ORDER = ("a", "m", "sigma_w", "sigma_s")
    _ws_lo, _ws_hi = resolve_ws_bounds(ws_bounds)
    if fixed_params:
        _fixed_names = [k for k in _PARAM_ORDER if fixed_params.get(k)]
        if _fixed_names:
            print(f"[STEP 2] Fixed Waxman-Smits parameters (held at prior, not optimized): "
                  f"{_fixed_names}")

    # ---- Handle 4-D time-lapse arrays ----------------------------------------
    # Build a 2D look-up: sigma_b_obs[i] = EC at (time=sample_times[i],
    # position=sample_positions[i]) for the current ensemble member.
    # For 3-D (single-time) arrays this simplifies to sigma_b_obs[i] =
    # model[sample_positions[i], layer].
    timelapse_4d = models.ndim == 4

    if timelapse_4d:
        ensemble_size = models.shape[0]
        num_layers    = models.shape[3]
        num_positions = models.shape[2]
    else:
        ensemble_size, num_positions, num_layers = models.shape
    sample_positions = list(sample_positions)
    n_samples = len(sample_positions)

    # theta_obs_std -> a per-observation array (Fix "TDR measurement
    # uncertainty ... never used to weight the calibration"). Scalar
    # broadcasts to every observation; a sequence must match theta_obs
    # 1:1. None (default) disables weighting entirely -- see
    # _log_ratio_objective below.
    if theta_obs_std is not None:
        theta_obs_std_arr = np.broadcast_to(
            np.asarray(theta_obs_std, dtype=float), (n_samples,)
        ).copy()
        if np.any(theta_obs_std_arr <= 0) or np.any(~np.isfinite(theta_obs_std_arr)):
            print(
                "[STEP 2][WARN] theta_obs_std contains non-positive or non-finite "
                "value(s) -- observation weighting disabled for this call "
                "(falling back to uniform weights) rather than dividing by zero."
            )
            theta_obs_std_arr = None
    else:
        theta_obs_std_arr = None

    if probe_layer_indices is not None:
        probe_layer_indices = list(probe_layer_indices)
        if len(probe_layer_indices) != n_samples:
            raise ValueError(
                f"probe_layer_indices has {len(probe_layer_indices)} entries "
                f"but sample_positions/theta_obs has {n_samples}; each TDR "
                "probe needs exactly one layer index."
            )
        if not depths_are_fixed:
            if probe_depths_reflect_fitted_interfaces:
                print(
                    "[INFO] Step 2: the inversion ran with free (or partially free) "
                    "layer interfaces, but probe_layer_indices were routed against "
                    "the ensemble's actual FITTED interface depths (not the static "
                    "starting depths0), so the horizon assigned to each TDR probe "
                    "already reflects where the interfaces ended up. This is still "
                    "a single ensemble-aggregated routing rather than an exact "
                    "per-member assignment, so treat it as a good approximation "
                    "rather than an exact one."
                )
            else:
                print(
                    "[WARN] Step 2: probe_layer_indices were derived from the starting "
                    "depths0 values, but the inversion ran with free (or partially free) "
                    "layer interfaces. Interface positions may have shifted during "
                    "optimisation, so the horizon assigned to each TDR probe is "
                    "approximate. Consider setting probe_layer_indices=None to fall "
                    "back to the strided heuristic, or pass "
                    "probe_depths_reflect_fitted_interfaces=True after routing probes "
                    "against the ensemble's fitted interface depths (depths_clean / "
                    "k.depths[0]) instead of the static starting depths0."
                )

    # Build per-horizon initial guesses
    if lab_p0 is None:
        p0_per_horizon = np.tile(DEFAULT_WS_P0, (num_layers, 1))
    else:
        lab_p0 = np.asarray(lab_p0, dtype=float)
        if lab_p0.ndim == 1:
            p0_per_horizon = np.tile(lab_p0, (num_layers, 1))
        else:
            p0_per_horizon = lab_p0  # shape (num_layers, 4)

    fits: list[PetroFit] = []

    # Per-horizon warm-start state: last converged [a, m, sigma_w, sigma_s]
    # for that horizon, carried across ensemble-axis iterations when
    # warm_start_continuation=True (see docstring). None until the first
    # successful fit for that horizon.
    _continuation_x0: list[np.ndarray | None] = [None] * num_layers

    for idx in range(ensemble_size):
        horizon_fits: list[HorizonWaxmanSmitsParams] = []

        for layer in range(num_layers):
            # Extract σ_b at this layer for each sample observation.
            # For time-lapse 4-D arrays each probe reading is paired with the
            # EC snapshot at its own survey time (sample_times[i]).
            if timelapse_4d:
                sigma_b_full = np.array(
                    [models[idx, sample_times[i], pos, layer]
                     for i, pos in enumerate(sample_positions)],
                    dtype=float,
                )
            else:
                sigma_b_full = np.array(
                    [models[idx, pos, layer] for pos in sample_positions],
                    dtype=float,
                )

            is_calibrated_layer = True

            if probe_layer_indices is not None:
                # Explicit per-probe horizon routing — each TDR probe's
                # installation depth has already been mapped to its soil
                # layer (via layer_index_from_depth), so we know exactly
                # which observations belong to this horizon.
                indices = [i for i, li in enumerate(probe_layer_indices) if li == layer]
                if not indices:
                    # No TDR probe was ever installed in this horizon. Do NOT
                    # borrow observations from other horizons. Instead
                    # mark this horizon explicitly uncalibrated.
                    print(
                        f"[WARN] No TDR probes mapped to layer {layer} via "
                        f"probe_layer_indices. Leaving this horizon UNCALIBRATED "
                        f"(prior parameters only, excluded from RMSE ranking) "
                        f"instead of fitting against unrelated observations."
                    )
                    is_calibrated_layer = False
                    theta_layer = np.array([], dtype=float)
                    sigma_b_layer = np.array([], dtype=float)
                    std_layer = None
                else:
                    theta_layer = theta_obs[indices]
                    sigma_b_layer = sigma_b_full[indices]
                    std_layer = theta_obs_std_arr[indices] if theta_obs_std_arr is not None else None
            elif n_samples >= num_layers:
                # Assign theta_obs to this horizon using a strided slice so
                # each layer gets a distinct, non-overlapping subset where
                # possible.
                step = n_samples // num_layers
                start = layer * step
                end = start + step if layer < num_layers - 1 else n_samples
                theta_layer = theta_obs[start:end]
                sigma_b_layer = sigma_b_full[start:end]
                std_layer = theta_obs_std_arr[start:end] if theta_obs_std_arr is not None else None
            else:
                # Wrap-around: assign each observation to the closest layer
                indices = [i for i in range(n_samples) if i % num_layers == layer]
                if not indices:
                    indices = list(range(n_samples))   # fallback: use all
                theta_layer = theta_obs[indices]
                sigma_b_layer = sigma_b_full[indices]
                std_layer = theta_obs_std_arr[indices] if theta_obs_std_arr is not None else None

            # ----------------------------------------------------------------
            # NaN-observation filter (real-data time-lapse spatial reference
            # grid). See geometry_utils.match_epochs_to_reference_grid and
            # run_pipeline.py Step 1: real GEM-2 survey epochs are now matched
            # onto a shared UNION spatial grid instead of being trimmed to a
            # common row count, so a probe/epoch pair whose reference-grid
            # point wasn't actually covered by that particular epoch shows up
            # here as sigma_b_full == NaN, rather than being silently paired
            # with an unrelated EC value the way the old row-count trim did.
            # Drop those observations from THIS horizon's fit instead of
            # letting a NaN reach the optimizer (the log-ratio misfit term
            # below would otherwise turn a single NaN into a NaN objective
            # for the whole horizon, silently discarding every OTHER valid
            # observation too).
            # ----------------------------------------------------------------
            if is_calibrated_layer and sigma_b_layer.size > 0:
                _finite = np.isfinite(sigma_b_layer)
                if not np.all(_finite):
                    _n_dropped = int((~_finite).sum())
                    print(
                        f"[STEP 2][INFO] Layer {layer}: dropping {_n_dropped} "
                        f"observation(s) with no EC value at their survey epoch "
                        f"(unmatched on the time-lapse spatial reference grid) "
                        f"out of {sigma_b_layer.size} candidate(s)."
                    )
                    theta_layer = theta_layer[_finite]
                    sigma_b_layer = sigma_b_layer[_finite]
                    if std_layer is not None:
                        std_layer = std_layer[_finite]

                if sigma_b_layer.size == 0:
                    print(
                        f"[WARN] Layer {layer}: every candidate observation was "
                        f"NaN (unmatched on the time-lapse spatial reference grid). "
                        f"Leaving this horizon UNCALIBRATED for this ensemble member."
                    )
                    is_calibrated_layer = False

            # Per-observation weights for the data-misfit term (Fix "TDR
            # measurement uncertainty ... never used to weight the
            # calibration"): inverse-variance, renormalised to mean 1 so the
            # overall misfit SCALE is unchanged whether or not weighting is
            # active -- lambda_reg below doesn't need retuning purely
            # because theta_obs_std was supplied. Falls back to uniform
            # weights (no-op) when theta_obs_std wasn't given, or doesn't
            # survive the NaN filter above cleanly.
            if (std_layer is not None and std_layer.size == theta_layer.size
                    and theta_layer.size > 0 and np.all(np.isfinite(std_layer))
                    and np.all(std_layer > 0)):
                _inv_var = 1.0 / (std_layer ** 2)
                obs_weights = _inv_var / np.mean(_inv_var)
            else:
                obs_weights = np.ones_like(theta_layer)

            p0 = p0_per_horizon[layer]

            if not is_calibrated_layer:
                # No real observations for this horizon — do not run the
                # optimizer at all. Carry forward the starting-guess parameters.
                a, m, sigma_w, sigma_s = p0
                theta_pred = np.array([], dtype=float)
                rmse = float("nan")
            else:
                try:
                    from scipy.optimize import minimize

                    # --- Regularization Configuration ---
                    # standard deviations defining how far the optimizer can stray from p0
                    PRIOR_STDS = {
                        'a': 0.5,        # Range (0.5, 5)
                        'm': 0.5,        # Range (1.0, 4.0)
                        'sigma_w': 150.0,# Range (1, 1000)
                        'sigma_s': 30.0  # Range (0, 200)
                    }
                    lambda_reg = 0.1  # Regularization weight factor

                    def _log_ratio_objective(params):
                        a_, m_, sigma_w_, sigma_s_ = params
                        # Avoid non-physical values during optimisation
                        a_ = max(a_, 1e-6)
                        m_ = max(m_, 1e-6)
                        sigma_w_ = max(sigma_w_, 1e-6)
                        sigma_s_ = max(sigma_s_, 0.0)

                        # 1. Standard Data Misfit (Paper Eq. 3)
                        #
                        # FIX (physics: log-misfit singularity at sigma_b~1 mS/m):
                        # previously this divided by log_ert
                        # (sum(|Δlog10 / log10(sigma_ert)|)), which blows up for
                        # ANY observation whose bulk EC is close to 1 mS/m
                        # (log10(1) = 0) -- squarely inside the 1-300 mS/m band
                        # this pipeline's own em_unit sanity check treats as
                        # typical GEM-2 field-soil EC. A single such observation
                        # could dominate the whole horizon's misfit regardless of
                        # fit quality elsewhere. Sum-of-squared-log10-differences
                        # is the standard misfit for EC/resistivity data spanning
                        # orders of magnitude (no singularity), and puts
                        # data_misfit on the same sum-of-squares footing as the
                        # Gaussian prior_penalty term below (previously one term
                        # was L1-relative and the other L2 -- an apples-to-oranges
                        # combination). NOTE: this changes the absolute scale of
                        # data_misfit relative to the old formula, so lambda_reg
                        # below may warrant re-tuning against real calibration
                        # data; obs_weights (see above) are renormalised to mean 1
                        # specifically so turning TDR-noise weighting on/off
                        # doesn't ALSO shift this scale.
                        theta_clip = np.clip(theta_layer, 1e-6, 1.0)
                        sigma_ws = (theta_clip ** m_ / a_) * (sigma_w_ + sigma_s_)
                        sigma_ert = np.clip(sigma_b_layer, 1e-10, None)
                        log_ert = np.log10(sigma_ert)
                        log_ws  = np.log10(np.clip(sigma_ws, 1e-10, None))
                        # FIX (TDR measurement uncertainty never used to weight
                        # the calibration): obs_weights is inverse-variance from
                        # theta_obs_std, renormalised to mean 1 (all-ones, i.e. a
                        # no-op, when theta_obs_std wasn't supplied) -- see its
                        # computation above, right before this optimizer call.
                        data_misfit = np.sum(obs_weights * (log_ert - log_ws) ** 2)

                        # 2. Gaussian Prior Penalty Term Matrix
                        prior_penalty = (
                            ((a_ - p0[0]) / PRIOR_STDS['a']) ** 2 +
                            ((m_ - p0[1]) / PRIOR_STDS['m']) ** 2 +
                            ((sigma_w_ - p0[2]) / PRIOR_STDS['sigma_w']) ** 2 +
                            ((sigma_s_ - p0[3]) / PRIOR_STDS['sigma_s']) ** 2
                        )

                        # Total combined loss
                        return data_misfit + (lambda_reg * prior_penalty)

                    # Bounds: a∈(0.5,5), m∈(1.0,4.0), σ_w∈(1,1000), σ_s∈(0,200)
                    # Parameters listed in fixed_params are pinned to a
                    # degenerate (p0, p0) bound instead of their normal
                    # range, so L-BFGS-B cannot move them -- the objective
                    # stays 4-D but effectively has fewer free dimensions.
                    # This is how e.g. m and sigma_s can be held at
                    # literature/lab values while a and sigma_w still float.
                    _bounds_full = [
                        (_ws_lo[0], _ws_hi[0]),   # a
                        (_ws_lo[1], _ws_hi[1]),   # m
                        (_ws_lo[2], _ws_hi[2]),   # σ_w
                        (_ws_lo[3], _ws_hi[3]),   # σ_s
                    ]
                    bounds_lbfgsb = [
                        (p0[i], p0[i]) if fixed_params.get(_PARAM_ORDER[i]) else _bounds_full[i]
                        for i in range(4)
                    ]

                    # Starting point for the optimizer. When
                    # warm_start_continuation is enabled and a previous
                    # ensemble-axis member already converged for this
                    # horizon, seed from THAT solution instead of the fixed
                    # prior p0 -- this smooths the fit trajectory across an
                    # ordered ensemble axis (e.g. Step 4's K = 1, 2, ..., N)
                    # instead of letting L-BFGS-B land in a different local
                    # minimum at every step. The prior penalty above still
                    # centres on p0 regardless of x0. Fixed parameters are
                    # always reset to their exact p0 value here so they sit
                    # dead-centre in their pinned (p0, p0) bound.
                    if warm_start_continuation and _continuation_x0[layer] is not None:
                        x0 = _continuation_x0[layer].copy()
                    else:
                        x0 = p0.copy()
                    for i in range(4):
                        if fixed_params.get(_PARAM_ORDER[i]):
                            x0[i] = p0[i]

                    result = minimize(
                        _log_ratio_objective,
                        x0=x0,
                        method="L-BFGS-B",
                        bounds=bounds_lbfgsb,
                        options={"maxiter": 5000, "ftol": 1e-12},
                    )
                    popt = result.x if result.success else p0.copy()
                    _continuation_x0[layer] = np.asarray(popt, dtype=float).copy()
                except Exception:
                    popt = p0.copy()   # safe fallback to initial guess

                a, m, sigma_w, sigma_s = popt
                theta_pred = waxman_smits_theta(sigma_b_layer, a, m, sigma_w, sigma_s)
                rmse = float(np.sqrt(np.mean((theta_pred - theta_layer) ** 2)))

            horizon_fits.append(
                HorizonWaxmanSmitsParams(
                    horizon=layer,
                    a=float(a),
                    m=float(m),
                    sigma_w=float(sigma_w),
                    sigma_s=float(sigma_s),
                    rmse=rmse,
                    predicted_theta=theta_pred,
                    observed_theta=theta_layer,
                    is_calibrated=is_calibrated_layer,
                )
            )

        fits.append(PetroFit(model_index=idx, horizon_fits=horizon_fits))

    return fits


# ==========================================================================
# 6b. SPATIAL LOCALIZATION: KERNEL-WEIGHTED PER-TARGET CALIBRATION
#
# See spatial_calibration_alternatives.md, §2 ("Kernel-weighted /
# localization"). Generalizes calibrate_petrophysical_relationship above:
# instead of one scalar [a, m, sigma_w, sigma_s] per horizon broadcast to
# every EM point (fine when a horizon has direct TDR coverage everywhere,
# but silently falls back to the pure-prior/global value at every EM point
# with no nearby probe), this refits SEPARATELY at every target location,
# weighting every TDR observation in that horizon by a smooth,
# compact-support function of its distance to the target instead of a hard
# "belongs to this position" bucket.
#
# bandwidth (``c``) -> inf reproduces calibrate_petrophysical_relationship's
# single global-per-horizon fit exactly (every observation gets spatial
# weight 1 regardless of distance). bandwidth -> 0 collapses to
# nearest-neighbour/IDW (only the closest observation(s) keep nonzero
# weight). Horizon routing (probe_layer_indices / the strided heuristic) is
# UNCHANGED and still a hard partition by depth -- localization only mixes
# observations WITHIN the same horizon, across space, never across depth.
# ==========================================================================

def gaspari_cohn(dist: np.ndarray | float, c: float) -> np.ndarray:
    """
    Gaspari-Cohn compact-support localization kernel (Gaspari & Cohn, 1999).

    Standard taper used for covariance/observation localization in ensemble
    data assimilation (ensemble Kalman filter): weight is 1 at zero
    distance, decays smoothly, and is EXACTLY zero for ``|dist| > 2c`` --
    unlike a Gaussian kernel's infinite tail, this lets callers skip
    far-away observations entirely rather than summing over many
    vanishingly small but nonzero weights.

    Parameters
    ----------
    dist : np.ndarray or float
        Signed or unsigned distance(s) from the target location, same
        units as ``c``. Only ``|dist|`` matters. Any shape. Works
        unchanged for 2-D (or N-D) Euclidean distances -- e.g.
        ``np.sqrt(dx**2 + dy**2)`` -- since those are already
        nonnegative, so ``np.abs`` is a no-op; only the caller's distance
        computation needs to change to go from 1-D to 2-D localization.
    c : float
        Localization radius. Support (nonzero weight) extends to ``2*c``;
        ``c`` itself is where the taper is roughly half-strength. Chosen
        empirically -- exactly as ad hoc as fixed-shrinkage's
        ``shrinkage_k`` was -- e.g. by sweeping a few candidate values and
        checking validation RMSE.

    Returns
    -------
    np.ndarray, same shape as ``dist`` (or 0-d if ``dist`` was a scalar).
        Weights in ``[0, 1]``.

    Notes
    -----
    ``c -> inf`` sends every weight to (approximately) 1 -- recovers a pure
    global fit. ``c -> 0`` sends every weight except at ``dist == 0`` to 0
    -- recovers nearest-neighbour.
    """
    dist_arr = np.atleast_1d(np.asarray(dist, dtype=float))
    if c <= 0:
        raise ValueError(f"gaspari_cohn: bandwidth c must be > 0, got {c!r}.")

    r = np.abs(dist_arr) / c
    w = np.zeros_like(r)

    m1 = r <= 1.0
    r1 = r[m1]
    w[m1] = (-0.25 * r1 ** 5 + 0.5 * r1 ** 4 + 0.625 * r1 ** 3
              - (5.0 / 3.0) * r1 ** 2 + 1.0)

    m2 = (r > 1.0) & (r <= 2.0)
    r2 = r[m2]
    with np.errstate(divide="ignore", invalid="ignore"):
        w[m2] = (r2 ** 5 / 12.0 - 0.5 * r2 ** 4 + 0.625 * r2 ** 3
                  + (5.0 / 3.0) * r2 ** 2 - 5.0 * r2 + 4.0 - 2.0 / (3.0 * r2))

    w = np.clip(w, 0.0, None)
    return w if np.asarray(dist).ndim > 0 else w[0]


@dataclass
class LocalizedHorizonWaxmanSmitsParams:
    """
    Kernel-localized Waxman-Smits fit for one soil horizon: one parameter
    set PER TARGET LOCATION along the transect, instead of
    ``HorizonWaxmanSmitsParams``'s single scalar broadcast to every EM point.

    Parameters
    ----------
    horizon : int
        0-based layer index (same meaning as ``HorizonWaxmanSmitsParams.horizon``).
    target_indices : np.ndarray, shape (n_targets,)
        Indices into the ``num_measured_points`` axis this horizon was
        localized to (typically every EM point, but can be a subset).
    target_x : np.ndarray, shape (n_targets,)
        Along-transect x-coordinate of each target (from ``point_x_coords``).
    target_y : np.ndarray, shape (n_targets,), or None
        Cross-transect y-coordinate of each target (from
        ``point_y_coords``), when 2-D Euclidean localization was used.
        ``None`` for the default 1-D-transect case.
    a, m, sigma_w, sigma_s : np.ndarray, shape (n_targets,)
        Fitted Waxman-Smits parameters AT EACH TARGET. Where a target had no
        TDR observation within the kernel's support (``2*bandwidth``), the
        corresponding entry falls back to the prior ``p0`` for this horizon
        (see ``is_calibrated``).
    rmse : np.ndarray, shape (n_targets,)
        Unweighted RMSE of this target's fitted params against ALL of this
        horizon's observed theta (same convention as
        ``HorizonWaxmanSmitsParams.rmse`` -- not just the nearby ones), so
        per-horizon RMSE stays comparable to the un-localized fit. ``NaN``
        wherever ``is_calibrated`` is ``False`` for that target.
    is_calibrated : np.ndarray of bool, shape (n_targets,)
        ``False`` at any target whose kernel support contained zero
        effective weight (no TDR observation of this horizon was close
        enough to matter, given ``bandwidth``) -- that target's params are
        the unfit prior, same spirit as
        ``HorizonWaxmanSmitsParams.is_calibrated`` but per-target rather
        than per-horizon.
    bandwidth : float
        The ``c`` passed to ``gaspari_cohn`` for this fit.
    """
    horizon: int
    target_indices: np.ndarray
    target_x: np.ndarray
    a: np.ndarray
    m: np.ndarray
    sigma_w: np.ndarray
    sigma_s: np.ndarray
    rmse: np.ndarray
    is_calibrated: np.ndarray
    bandwidth: float
    target_y: np.ndarray | None = None


@dataclass
class LocalizedPetroFit:
    """
    Container for kernel-localized Waxman-Smits calibration results for one
    ensemble member -- the localized analogue of ``PetroFit``.
    """
    model_index: int
    horizon_fits: list[LocalizedHorizonWaxmanSmitsParams]

    @property
    def rmse(self) -> float:
        """
        Mean RMSE across every calibrated (horizon, target) pair. Provided
        for parity with ``PetroFit.rmse``; how Steps 3/4's ensemble ranking
        should aggregate a per-target RMSE array into a single per-member
        score (mean over targets? worst-target? per-cell, per §5) is a
        separate design decision -- this is just a simple default.
        """
        all_rmse = np.concatenate([
            np.where(h.is_calibrated, h.rmse, np.nan) for h in self.horizon_fits
        ]) if self.horizon_fits else np.array([])
        if all_rmse.size == 0 or np.all(np.isnan(all_rmse)):
            return float("nan")
        return float(np.nanmean(all_rmse))


def _fit_ws_horizon_weighted(
    theta_layer: np.ndarray,
    sigma_b_layer: np.ndarray,
    obs_weights: np.ndarray,
    p0: np.ndarray,
    fixed_params: dict[str, bool],
    x0_start: np.ndarray | None = None,
    ws_bounds: dict | None = None,
) -> tuple[float, float, float, float, np.ndarray]:
    """
    Single Waxman-Smits horizon fit for an arbitrary observation-weight
    vector. Extracted from ``calibrate_petrophysical_relationship``'s inner
    optimizer block (same objective, priors, bounds, and fallback
    behaviour) so ``calibrate_petrophysical_relationship_localized`` can
    call it once per target location with a different ``obs_weights`` each
    time, instead of duplicating the optimizer setup.

    Parameters
    ----------
    theta_layer, sigma_b_layer : np.ndarray, shape (n_obs,)
        This horizon's observed theta and paired bulk EC.
    obs_weights : np.ndarray, shape (n_obs,)
        Per-observation weight fed into the data-misfit term. Callers are
        responsible for any renormalisation convention (e.g. mean 1) --
        this function uses whatever is passed in as-is.
    p0 : np.ndarray, shape (4,)
        Prior/starting guess ``[a, m, sigma_w, sigma_s]``. Always used as
        the *prior center* for the regularization term, regardless of
        ``x0_start``.
    fixed_params : dict[str, bool]
        Same meaning as in ``calibrate_petrophysical_relationship``.
    x0_start : np.ndarray, shape (4,), optional
        Optimizer starting point. Defaults to ``p0`` (the old cold-start
        behaviour) when not given. Callers doing localized fits over
        spatially ordered targets can pass the previous target's
        converged solution here to warm-start the optimizer -- this does
        not change the prior (``p0`` still anchors the regularization
        term) or any other outward behaviour, it only changes where
        L-BFGS-B begins searching.

    Returns
    -------
    a, m, sigma_w, sigma_s : float
    theta_pred : np.ndarray, shape (n_obs,)
        Predicted theta at every ``sigma_b_layer`` entry using the fitted
        params (computed against ALL observations, not just the
        nonzero-weight ones, so RMSE stays comparable across bandwidths).
    """
    from scipy.optimize import minimize

    _PARAM_ORDER = ("a", "m", "sigma_w", "sigma_s")
    PRIOR_STDS = {"a": 0.5, "m": 0.5, "sigma_w": 150.0, "sigma_s": 30.0}
    lambda_reg = 0.1
    _LN10 = np.log(10.0)

    theta_clip = np.clip(theta_layer, 1e-6, 1.0)
    ln_theta_clip = np.log(theta_clip)
    sigma_ert = np.clip(sigma_b_layer, 1e-10, None)
    log_ert = np.log10(sigma_ert)

    def _log_ratio_objective_and_grad(params):
        a_, m_, sigma_w_, sigma_s_ = params
        a = max(a_, 1e-6)
        m = max(m_, 1e-6)
        sigma_w = max(sigma_w_, 1e-6)
        sigma_s = max(sigma_s_, 0.0)

        # Subgradient of the max()/clip() reparameterization: 1 where the
        # raw optimizer parameter is inside its clamp, 0 where it's been
        # clamped (matches the value the un-clamped objective used).
        mask_a = 1.0 if a_ > 1e-6 else 0.0
        mask_m = 1.0 if m_ > 1e-6 else 0.0
        mask_sw = 1.0 if sigma_w_ > 1e-6 else 0.0
        mask_ss = 1.0 if sigma_s_ > 0.0 else 0.0

        S = sigma_w + sigma_s
        T = theta_clip ** m
        sigma_ws = T * S / a
        sigma_ws_clipped = np.clip(sigma_ws, 1e-10, None)
        log_ws = np.log10(sigma_ws_clipped)

        resid = log_ert - log_ws
        data_misfit = np.sum(obs_weights * resid ** 2)

        prior_penalty = (
            ((a - p0[0]) / PRIOR_STDS["a"]) ** 2
            + ((m - p0[1]) / PRIOR_STDS["m"]) ** 2
            + ((sigma_w - p0[2]) / PRIOR_STDS["sigma_w"]) ** 2
            + ((sigma_s - p0[3]) / PRIOR_STDS["sigma_s"]) ** 2
        )
        obj = data_misfit + (lambda_reg * prior_penalty)

        # --- analytic gradient ---
        active = sigma_ws > 1e-10
        dlogws_dsigmaws = np.where(active, 1.0 / (sigma_ws_clipped * _LN10), 0.0)
        dmisfit_dsigmaws = -2.0 * obs_weights * resid * dlogws_dsigmaws

        dsigmaws_da = -sigma_ws / a
        dsigmaws_dm = sigma_ws * ln_theta_clip
        dsigmaws_dsw = T / a
        dsigmaws_dss = T / a

        dmisfit_da = np.sum(dmisfit_dsigmaws * dsigmaws_da)
        dmisfit_dm = np.sum(dmisfit_dsigmaws * dsigmaws_dm)
        dmisfit_dsw = np.sum(dmisfit_dsigmaws * dsigmaws_dsw)
        dmisfit_dss = np.sum(dmisfit_dsigmaws * dsigmaws_dss)

        dprior_da = 2.0 * (a - p0[0]) / PRIOR_STDS["a"] ** 2
        dprior_dm = 2.0 * (m - p0[1]) / PRIOR_STDS["m"] ** 2
        dprior_dsw = 2.0 * (sigma_w - p0[2]) / PRIOR_STDS["sigma_w"] ** 2
        dprior_dss = 2.0 * (sigma_s - p0[3]) / PRIOR_STDS["sigma_s"] ** 2

        grad = np.array([
            (dmisfit_da + lambda_reg * dprior_da) * mask_a,
            (dmisfit_dm + lambda_reg * dprior_dm) * mask_m,
            (dmisfit_dsw + lambda_reg * dprior_dsw) * mask_sw,
            (dmisfit_dss + lambda_reg * dprior_dss) * mask_ss,
        ])
        return obj, grad

    _ws_lo, _ws_hi = resolve_ws_bounds(ws_bounds)
    _bounds_full = [(_ws_lo[i], _ws_hi[i]) for i in range(4)]
    bounds_lbfgsb = [
        (p0[i], p0[i]) if fixed_params.get(_PARAM_ORDER[i]) else _bounds_full[i]
        for i in range(4)
    ]

    x0 = (x0_start.copy() if x0_start is not None else p0.copy())
    for i in range(4):
        if fixed_params.get(_PARAM_ORDER[i]):
            x0[i] = p0[i]
        else:
            # Warm-start values must still respect this fit's own bounds
            # (a previous target's converged solution could in principle
            # sit outside them in edge cases).
            lo, hi = bounds_lbfgsb[i]
            x0[i] = min(max(x0[i], lo), hi)

    try:
        result = minimize(
            _log_ratio_objective_and_grad,
            x0=x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds_lbfgsb,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        popt = result.x if result.success else p0.copy()
    except Exception:
        popt = p0.copy()

    a, m, sigma_w, sigma_s = popt
    theta_pred = waxman_smits_theta(sigma_b_layer, a, m, sigma_w, sigma_s)
    return float(a), float(m), float(sigma_w), float(sigma_s), theta_pred


def calibrate_petrophysical_relationship_localized(
    models: np.ndarray,
    sample_positions: Sequence[int],
    sample_times: Sequence[int],
    theta_obs: np.ndarray,
    point_x_coords: Sequence[float],
    bandwidth: float,
    point_y_coords: Sequence[float] | None = None,
    target_indices: Sequence[int] | None = None,
    depths0: Sequence[float] | None = None,
    lab_p0: np.ndarray | None = None,
    probe_layer_indices: Sequence[int] | None = None,
    fixed_params: dict[str, bool] | None = None,
    theta_obs_std: float | Sequence[float] | None = None,
    ws_bounds: dict | None = None,
) -> list[LocalizedPetroFit]:
    """
    Kernel-localized (Gaspari-Cohn-weighted) generalization of
    ``calibrate_petrophysical_relationship`` -- see
    ``spatial_calibration_alternatives.md`` §2.

    Fits Waxman-Smits parameters separately per soil horizon (exactly as
    the un-localized function does -- horizon routing is unchanged and
    still a hard depth partition) AND separately per spatial target
    location within that horizon. Every TDR observation belonging to the
    horizon contributes to every target's fit, weighted by
    ``gaspari_cohn(dist to target, bandwidth)`` multiplied into the
    existing ``theta_obs_std``-derived weight, instead of a hard
    "belongs to this position" bucket.

    ``bandwidth -> inf`` reproduces ``calibrate_petrophysical_relationship``'s
    single global-per-horizon fit exactly. ``bandwidth -> 0`` collapses to
    nearest-neighbour.

    Parameters
    ----------
    models, sample_positions, sample_times, theta_obs, depths0, lab_p0,
    probe_layer_indices, fixed_params, theta_obs_std
        Same meaning as in ``calibrate_petrophysical_relationship``. Horizon
        routing (``probe_layer_indices`` if given, else the strided
        modulo heuristic) is reused unchanged.
    point_x_coords : sequence of float, length num_measured_points
        Along-transect x-coordinate of EVERY EM measured point (the same
        axis ``sample_positions`` indexes into). When ``point_y_coords`` is
        NOT given, ``gaspari_cohn`` distances are the signed 1-D difference
        along this axis. Typically the same coordinate array used to build
        ``probe_x_coords`` in ``extract_sample_positions_from_survey``.
    bandwidth : float
        The localization radius ``c`` passed to ``gaspari_cohn``, same
        units as ``point_x_coords`` (and ``point_y_coords``, if given). No
        default -- pick empirically by sweeping a few candidate values and
        checking validation RMSE, the same way ``shrinkage_k`` would have
        been tuned.
    point_y_coords : sequence of float, length num_measured_points, optional
        Cross-transect y-coordinate of every EM measured point, same axis
        and ordering as ``point_x_coords``. When given, ``gaspari_cohn``
        distances become the 2-D Euclidean distance
        ``sqrt((x_obs - x_target)**2 + (y_obs - y_target)**2)`` instead of
        the signed 1-D x-difference -- required for genuine 2-D survey
        grids (``mode == "2d"``, real ``utm_x``/``utm_y`` geometry), where
        localizing on x alone would ignore probes that are close in y but
        far in x (or vice versa). ``None`` (default) keeps the original
        1-D-transect behaviour unchanged (``y`` implicitly 0 for every
        point, so 1-D distance and 2-D distance coincide).
    target_indices : sequence of int, optional
        Which EM points to produce a localized fit for. Defaults to every
        point in ``point_x_coords`` (``range(len(point_x_coords))``) -- i.e.
        a full per-EM-point σ_s(x[, y]) field. Pass a coarser subset to save
        compute if a per-point fit isn't needed everywhere.

    Returns
    -------
    list of LocalizedPetroFit, length ensemble_size
    """
    fixed_params = fixed_params or {}
    _PARAM_ORDER = ("a", "m", "sigma_w", "sigma_s")

    point_x_coords = np.asarray(point_x_coords, dtype=float)

    timelapse_4d = models.ndim == 4
    if timelapse_4d:
        ensemble_size = models.shape[0]
        num_layers = models.shape[3]
        num_positions = models.shape[2]
    else:
        ensemble_size, num_positions, num_layers = models.shape

    if len(point_x_coords) != num_positions:
        raise ValueError(
            f"point_x_coords has {len(point_x_coords)} entries but models "
            f"has {num_positions} measured points along that axis; every "
            "measured point needs an x-coordinate for localization."
        )

    # Optional Euclidean (x, y) localization -- see point_y_coords docstring
    # above. use_2d=False keeps the exact original signed-1-D-distance path
    # (gaspari_cohn(obs_x - target_x, bandwidth)) untouched.
    if point_y_coords is not None:
        point_y_coords = np.asarray(point_y_coords, dtype=float)
        if len(point_y_coords) != num_positions:
            raise ValueError(
                f"point_y_coords has {len(point_y_coords)} entries but "
                f"models has {num_positions} measured points along that "
                "axis; every measured point needs a y-coordinate when "
                "point_y_coords is provided."
            )
        use_2d = True
    else:
        use_2d = False

    sample_positions = list(sample_positions)
    n_samples = len(sample_positions)
    sample_x = point_x_coords[np.asarray(sample_positions, dtype=int)]
    sample_y = (
        point_y_coords[np.asarray(sample_positions, dtype=int)] if use_2d else None
    )

    if target_indices is None:
        target_indices = np.arange(num_positions)
    else:
        target_indices = np.asarray(target_indices, dtype=int)
    target_x_all = point_x_coords[target_indices]
    target_y_all = point_y_coords[target_indices] if use_2d else None
    n_targets = len(target_indices)

    if theta_obs_std is not None:
        theta_obs_std_arr = np.broadcast_to(
            np.asarray(theta_obs_std, dtype=float), (n_samples,)
        ).copy()
        if np.any(theta_obs_std_arr <= 0) or np.any(~np.isfinite(theta_obs_std_arr)):
            print(
                "[STEP 2 LOCALIZED][WARN] theta_obs_std contains non-positive "
                "or non-finite value(s) -- observation weighting disabled for "
                "this call (falling back to uniform base weights)."
            )
            theta_obs_std_arr = None
    else:
        theta_obs_std_arr = None

    if probe_layer_indices is not None:
        probe_layer_indices = list(probe_layer_indices)
        if len(probe_layer_indices) != n_samples:
            raise ValueError(
                f"probe_layer_indices has {len(probe_layer_indices)} entries "
                f"but sample_positions/theta_obs has {n_samples}."
            )

    if lab_p0 is None:
        p0_per_horizon = np.tile(DEFAULT_WS_P0, (num_layers, 1))
    else:
        lab_p0 = np.asarray(lab_p0, dtype=float)
        p0_per_horizon = (
            np.tile(lab_p0, (num_layers, 1)) if lab_p0.ndim == 1 else lab_p0
        )

    fits: list[LocalizedPetroFit] = []

    for idx in range(ensemble_size):
        horizon_fits: list[LocalizedHorizonWaxmanSmitsParams] = []

        for layer in range(num_layers):
            if timelapse_4d:
                sigma_b_full = np.array(
                    [models[idx, sample_times[i], pos, layer]
                     for i, pos in enumerate(sample_positions)],
                    dtype=float,
                )
            else:
                sigma_b_full = np.array(
                    [models[idx, pos, layer] for pos in sample_positions],
                    dtype=float,
                )

            # ---- Horizon routing: unchanged hard depth partition ---------
            if probe_layer_indices is not None:
                indices = [i for i, li in enumerate(probe_layer_indices) if li == layer]
            elif n_samples >= num_layers:
                step = n_samples // num_layers
                start = layer * step
                end = start + step if layer < num_layers - 1 else n_samples
                indices = list(range(start, end))
            else:
                indices = [i for i in range(n_samples) if i % num_layers == layer]
                if not indices:
                    indices = list(range(n_samples))

            p0 = p0_per_horizon[layer]

            if not indices:
                print(
                    f"[STEP 2 LOCALIZED][WARN] No TDR probes mapped to layer "
                    f"{layer}. Every target in this horizon falls back to "
                    f"the prior (uncalibrated)."
                )
                horizon_fits.append(LocalizedHorizonWaxmanSmitsParams(
                    horizon=layer,
                    target_indices=target_indices.copy(),
                    target_x=target_x_all.copy(),
                    target_y=target_y_all.copy() if use_2d else None,
                    a=np.full(n_targets, p0[0]),
                    m=np.full(n_targets, p0[1]),
                    sigma_w=np.full(n_targets, p0[2]),
                    sigma_s=np.full(n_targets, p0[3]),
                    rmse=np.full(n_targets, np.nan),
                    is_calibrated=np.zeros(n_targets, dtype=bool),
                    bandwidth=bandwidth,
                ))
                continue

            theta_layer = theta_obs[indices]
            sigma_b_layer = sigma_b_full[indices]
            obs_x_layer = sample_x[indices]
            obs_y_layer = sample_y[indices] if use_2d else None
            std_layer = (
                theta_obs_std_arr[indices] if theta_obs_std_arr is not None else None
            )

            _finite = np.isfinite(sigma_b_layer)
            if not np.all(_finite):
                theta_layer = theta_layer[_finite]
                sigma_b_layer = sigma_b_layer[_finite]
                obs_x_layer = obs_x_layer[_finite]
                if obs_y_layer is not None:
                    obs_y_layer = obs_y_layer[_finite]
                if std_layer is not None:
                    std_layer = std_layer[_finite]

            # Base (non-spatial) per-observation weight -- same
            # inverse-variance-from-theta_obs_std convention as the
            # un-localized function.
            if (std_layer is not None and std_layer.size == theta_layer.size
                    and theta_layer.size > 0 and np.all(np.isfinite(std_layer))
                    and np.all(std_layer > 0)):
                _inv_var = 1.0 / (std_layer ** 2)
                base_weights = _inv_var / np.mean(_inv_var)
            else:
                base_weights = np.ones_like(theta_layer)

            a_arr = np.full(n_targets, np.nan)
            m_arr = np.full(n_targets, np.nan)
            sigma_w_arr = np.full(n_targets, np.nan)
            sigma_s_arr = np.full(n_targets, np.nan)
            rmse_arr = np.full(n_targets, np.nan)
            calibrated_arr = np.zeros(n_targets, dtype=bool)

            # Warm-start tracker for this horizon: target_x_all is spatially
            # ordered, so neighbouring targets under the same
            # Gaspari-Cohn-weighted fit tend to have very similar optima.
            # Seeding x0 with the previous target's converged solution
            # (instead of restarting from the prior every time) cuts
            # iteration counts substantially once the loop "warms up".
            # This never changes the prior itself -- only where L-BFGS-B
            # starts searching -- and resets to p0 whenever a target has
            # no in-range observations, so a bad neighbourhood can't poison
            # later targets.
            _prev_popt = None

            _target_y_iter = target_y_all if use_2d else [None] * n_targets
            for t_i, (target_x, target_y) in enumerate(zip(target_x_all, _target_y_iter)):
                if theta_layer.size == 0:
                    a_arr[t_i], m_arr[t_i], sigma_w_arr[t_i], sigma_s_arr[t_i] = p0
                    continue

                if use_2d:
                    # Euclidean (x, y) distance -- see point_y_coords
                    # docstring. gaspari_cohn only ever uses |dist|, so a
                    # nonnegative Euclidean distance is a drop-in
                    # replacement for the signed 1-D difference below.
                    dist = np.sqrt(
                        (obs_x_layer - target_x) ** 2 + (obs_y_layer - target_y) ** 2
                    )
                else:
                    dist = obs_x_layer - target_x
                spatial_w = gaspari_cohn(dist, bandwidth)
                combined_w_raw = base_weights * spatial_w

                if np.sum(combined_w_raw) < 1e-8:
                    # No observation of this horizon falls within the
                    # kernel's support (2*bandwidth) of this target -- same
                    # spirit as "no TDR probes mapped to this horizon", just
                    # per-target instead of per-horizon. Fall back to prior.
                    a_arr[t_i], m_arr[t_i], sigma_w_arr[t_i], sigma_s_arr[t_i] = p0
                    _prev_popt = None
                    continue

                # Renormalise to mean 1 (same convention as base_weights)
                # so lambda_reg's scale doesn't need retuning as bandwidth
                # changes how many effective observations feed each target.
                combined_w = combined_w_raw / np.mean(combined_w_raw)

                a, m, sigma_w, sigma_s, theta_pred = _fit_ws_horizon_weighted(
                    theta_layer, sigma_b_layer, combined_w, p0, fixed_params,
                    x0_start=_prev_popt, ws_bounds=ws_bounds,
                )
                a_arr[t_i], m_arr[t_i], sigma_w_arr[t_i], sigma_s_arr[t_i] = (
                    a, m, sigma_w, sigma_s,
                )
                _prev_popt = np.array([a, m, sigma_w, sigma_s])
                # RMSE against ALL of this horizon's observations (not just
                # the nearby ones), so it stays comparable across targets
                # and bandwidths -- same convention as the un-localized RMSE.
                rmse_arr[t_i] = float(np.sqrt(np.mean((theta_pred - theta_layer) ** 2)))
                calibrated_arr[t_i] = True

            horizon_fits.append(LocalizedHorizonWaxmanSmitsParams(
                horizon=layer,
                target_indices=target_indices.copy(),
                target_x=target_x_all.copy(),
                target_y=target_y_all.copy() if use_2d else None,
                a=a_arr, m=m_arr, sigma_w=sigma_w_arr, sigma_s=sigma_s_arr,
                rmse=rmse_arr, is_calibrated=calibrated_arr, bandwidth=bandwidth,
            ))

        fits.append(LocalizedPetroFit(model_index=idx, horizon_fits=horizon_fits))

    return fits


# ==========================================================================
# BANDWIDTH SELECTION -- leave-one-probe-out cross-validation
#
# The in-sample RMSE stored on LocalizedHorizonWaxmanSmitsParams.rmse is NOT
# a fair way to pick ``bandwidth``: as bandwidth -> 0, a target sitting at
# (or very near) an observation's own location gets that observation's
# weight -> 1 and everyone else's -> 0, so the "fit" just memorizes that
# single point and its RMSE goes to ~0 for the wrong reason (pure
# overfitting, not genuine skill). Leave-one-probe-out (LOO) CV fixes this
# the standard way: refit with observation i EXCLUDED, predict AT
# observation i's own location, and score against its true theta_obs. This
# is exactly the check the docstrings above already point to ("pick c the
# same way you'd have picked shrinkage_k -- by sweeping candidates and
# checking validation RMSE"); this function performs that sweep.
# ==========================================================================

@dataclass
class BandwidthSweepResult:
    """
    LOO-CV result for one candidate bandwidth.

    Attributes
    ----------
    bandwidth : float
    loo_rmse : float
        Root-mean-square leave-one-out residual across every TDR
        observation, ALL horizons pooled (the number that should be
        compared across candidate bandwidths).
    loo_rmse_per_horizon : dict[int, float]
        Same, broken out per horizon (NaN if a horizon had < 2 observations
        and so has no valid LOO fold).
    n_folds : int
        Number of observations that produced a valid LOO prediction (i.e.
        excluding folds where every OTHER observation fell outside the
        kernel's support and the fold fell back to the prior).
    n_fallback : int
        Number of folds that fell back to the prior p0 (no other
        observation within ``2*bandwidth`` of the held-out point) -- large
        values at small bandwidths are expected and are themselves useful
        signal that the bandwidth is too small for the probe spacing.
    """
    bandwidth: float
    loo_rmse: float
    loo_rmse_per_horizon: dict
    n_folds: int
    n_fallback: int


def sweep_localization_bandwidth(
    sigma_b_reference: np.ndarray,
    sample_positions: Sequence[int],
    sample_times: Sequence[int],
    theta_obs: np.ndarray,
    point_x_coords: Sequence[float],
    bandwidths: Sequence[float],
    point_y_coords: Sequence[float] | None = None,
    probe_layer_indices: Sequence[int] | None = None,
    num_layers: int | None = None,
    lab_p0: np.ndarray | None = None,
    fixed_params: dict[str, bool] | None = None,
    theta_obs_std: float | Sequence[float] | None = None,
    ws_bounds: dict | None = None,
) -> list[BandwidthSweepResult]:
    """
    Leave-one-probe-out cross-validated RMSE of the kernel-localized
    Waxman-Smits fit, for each candidate bandwidth in ``bandwidths`` -- the
    principled way to pick ``bandwidth``, instead of eyeballing the
    in-sample ``LocalizedHorizonWaxmanSmitsParams.rmse``.

    For each horizon, each candidate bandwidth, and each observation ``i``
    belonging to that horizon: refit Waxman-Smits using every OTHER
    observation in the horizon (``i`` excluded), weighted by
    ``gaspari_cohn(dist to i's own location, bandwidth)``, then predict
    theta at ``i`` (via the forward model, using ``i``'s own bulk EC) and
    compare against ``theta_obs[i]``. Pooling these held-out residuals
    across every observation (all horizons) gives one LOO-CV RMSE per
    bandwidth, directly comparable across candidates.

    Because this excludes the held-out point itself, it does NOT reward
    ``bandwidth -> 0`` memorization the way naive in-sample RMSE would --
    at very small bandwidths, most folds will instead fall back to the
    prior (see ``n_fallback``) once no other observation is close enough,
    which typically makes LOO-CV RMSE at small bandwidths worse, not
    better. Expect a U- or L-shaped curve: too small -> too few effective
    neighbours per fold (prior fallback, high error); too large ->
    approaches the global fit's own RMSE; a interior minimum (if one
    exists) is the recommended ``bandwidth``.

    Parameters
    ----------
    sigma_b_reference : np.ndarray
        ONE ensemble member's inverted bulk-EC model, shape
        ``(num_positions, num_layers)`` (single-time) or
        ``(n_time, num_positions, num_layers)`` (time-lapse) -- i.e. what
        ``models[idx]`` would be inside
        ``calibrate_petrophysical_relationship_localized``. Bandwidth
        selection is a one-off tuning step, not part of the per-ensemble-
        member pipeline, so this takes a single reference slice (e.g. the
        rank-1 / best-RMSE member from Step 2, or an ensemble mean) rather
        than the full ensemble -- sweeping many candidate bandwidths times
        many LOO folds times every ensemble member would be far more
        compute than this decision needs.
    sample_positions, sample_times, theta_obs, point_x_coords,
    point_y_coords, probe_layer_indices, lab_p0, fixed_params,
    theta_obs_std
        Same meaning as in ``calibrate_petrophysical_relationship_localized``.
    bandwidths : sequence of float
        Candidate values of ``c`` to evaluate, e.g.
        ``np.linspace(1.0, 20.0, 10)``.
    num_layers : int, optional
        Number of soil horizons. Inferred from ``sigma_b_reference``'s last
        axis if omitted.

    Returns
    -------
    list of BandwidthSweepResult, same length and order as ``bandwidths``.
    """
    fixed_params = fixed_params or {}

    point_x_coords = np.asarray(point_x_coords, dtype=float)
    use_2d = point_y_coords is not None
    if use_2d:
        point_y_coords = np.asarray(point_y_coords, dtype=float)

    timelapse_4d = sigma_b_reference.ndim == 3
    if num_layers is None:
        num_layers = sigma_b_reference.shape[-1]

    sample_positions = list(sample_positions)
    n_samples = len(sample_positions)
    sample_x = point_x_coords[np.asarray(sample_positions, dtype=int)]
    sample_y = point_y_coords[np.asarray(sample_positions, dtype=int)] if use_2d else None

    if theta_obs_std is not None:
        theta_obs_std_arr = np.broadcast_to(
            np.asarray(theta_obs_std, dtype=float), (n_samples,)
        ).copy()
        if np.any(theta_obs_std_arr <= 0) or np.any(~np.isfinite(theta_obs_std_arr)):
            theta_obs_std_arr = None
    else:
        theta_obs_std_arr = None

    if probe_layer_indices is not None:
        probe_layer_indices = list(probe_layer_indices)

    if lab_p0 is None:
        p0_per_horizon = np.tile(DEFAULT_WS_P0, (num_layers, 1))
    else:
        lab_p0_arr = np.asarray(lab_p0, dtype=float)
        p0_per_horizon = (
            np.tile(lab_p0_arr, (num_layers, 1)) if lab_p0_arr.ndim == 1 else lab_p0_arr
        )

    results: list[BandwidthSweepResult] = []

    for c in bandwidths:
        all_resid = []
        per_horizon_resid: dict = {}
        n_fallback = 0

        for layer in range(num_layers):
            if timelapse_4d:
                sigma_b_full = np.array(
                    [sigma_b_reference[sample_times[i], pos, layer]
                     for i, pos in enumerate(sample_positions)],
                    dtype=float,
                )
            else:
                sigma_b_full = np.array(
                    [sigma_b_reference[pos, layer] for pos in sample_positions],
                    dtype=float,
                )

            if probe_layer_indices is not None:
                indices = [i for i, li in enumerate(probe_layer_indices) if li == layer]
            elif n_samples >= num_layers:
                step = n_samples // num_layers
                start = layer * step
                end = start + step if layer < num_layers - 1 else n_samples
                indices = list(range(start, end))
            else:
                indices = [i for i in range(n_samples) if i % num_layers == layer]
                if not indices:
                    indices = list(range(n_samples))

            if len(indices) < 2:
                # Can't leave one out of a horizon with 0 or 1 observation.
                continue

            p0 = p0_per_horizon[layer]
            theta_layer = theta_obs[indices]
            sigma_b_layer = sigma_b_full[indices]
            obs_x_layer = sample_x[indices]
            obs_y_layer = sample_y[indices] if use_2d else None
            std_layer = (
                theta_obs_std_arr[indices] if theta_obs_std_arr is not None else None
            )

            _finite = np.isfinite(sigma_b_layer)
            if not np.all(_finite):
                theta_layer = theta_layer[_finite]
                sigma_b_layer = sigma_b_layer[_finite]
                obs_x_layer = obs_x_layer[_finite]
                if obs_y_layer is not None:
                    obs_y_layer = obs_y_layer[_finite]
                if std_layer is not None:
                    std_layer = std_layer[_finite]
            n_h = theta_layer.size
            if n_h < 2:
                continue

            if (std_layer is not None and std_layer.size == n_h
                    and np.all(np.isfinite(std_layer)) and np.all(std_layer > 0)):
                _inv_var = 1.0 / (std_layer ** 2)
                base_weights_full = _inv_var / np.mean(_inv_var)
            else:
                base_weights_full = np.ones(n_h)

            horizon_resid = []

            for i in range(n_h):
                mask = np.ones(n_h, dtype=bool)
                mask[i] = False

                target_x = obs_x_layer[i]
                target_y = obs_y_layer[i] if use_2d else None

                if use_2d:
                    dist = np.sqrt(
                        (obs_x_layer[mask] - target_x) ** 2
                        + (obs_y_layer[mask] - target_y) ** 2
                    )
                else:
                    dist = obs_x_layer[mask] - target_x

                spatial_w = gaspari_cohn(dist, c)
                combined_w_raw = base_weights_full[mask] * spatial_w

                if np.sum(combined_w_raw) < 1e-8:
                    a, m, sigma_w, sigma_s = p0
                    n_fallback += 1
                else:
                    combined_w = combined_w_raw / np.mean(combined_w_raw)
                    a, m, sigma_w, sigma_s, _ = _fit_ws_horizon_weighted(
                        theta_layer[mask], sigma_b_layer[mask], combined_w,
                        p0, fixed_params, ws_bounds=ws_bounds,
                    )

                theta_pred_i = waxman_smits_theta(
                    np.array([sigma_b_layer[i]]), a, m, sigma_w, sigma_s
                )[0]
                resid = float(theta_pred_i - theta_layer[i])
                horizon_resid.append(resid)
                all_resid.append(resid)

            per_horizon_resid[layer] = (
                float(np.sqrt(np.mean(np.square(horizon_resid))))
                if horizon_resid else float("nan")
            )

        pooled_rmse = (
            float(np.sqrt(np.mean(np.square(all_resid)))) if all_resid else float("nan")
        )
        results.append(BandwidthSweepResult(
            bandwidth=float(c),
            loo_rmse=pooled_rmse,
            loo_rmse_per_horizon=per_horizon_resid,
            n_folds=len(all_resid),
            n_fallback=n_fallback,
        ))

    return results


def ec_to_water_content_localized(
    ec_values: np.ndarray,
    fit: "LocalizedPetroFit",
    point_indices: Sequence[int],
    probe_layer_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Localized analogue of ``ec_to_water_content``: converts EC to θ using
    each point's OWN target-specific Waxman-Smits parameters instead of one
    scalar broadcast across the whole horizon.

    Parameters
    ----------
    ec_values : np.ndarray, shape (n_samples,)
        Bulk EC values to convert.
    fit : LocalizedPetroFit
        Output of ``calibrate_petrophysical_relationship_localized`` for
        one ensemble member (or an averaged model).
    point_indices : sequence of int, shape (n_samples,)
        For each entry in ``ec_values``, the EM measured-point index (into
        the same axis as ``point_x_coords``/``target_indices`` used when
        fitting) whose localized parameters should be used. This is the
        piece that's new relative to ``ec_to_water_content`` -- every point
        can now draw a DIFFERENT σ_s (etc.), not just a different horizon.
    probe_layer_indices : sequence of int, optional
        0-based horizon index per entry, same meaning as in
        ``ec_to_water_content``. ``None`` uses horizon 0 for every point
        (safe fallback for single-horizon surveys).

    Returns
    -------
    theta_pred : np.ndarray, shape (n_samples,)
    """
    ec_values = np.asarray(ec_values, dtype=float)
    n_samples = len(ec_values)
    point_indices = np.asarray(point_indices, dtype=int)
    if len(point_indices) != n_samples:
        raise ValueError(
            f"point_indices has {len(point_indices)} entries but ec_values "
            f"has {n_samples}; lengths must match."
        )

    if probe_layer_indices is None:
        probe_layer_indices = [0] * n_samples
    else:
        probe_layer_indices = list(probe_layer_indices)
        if len(probe_layer_indices) != n_samples:
            raise ValueError(
                f"probe_layer_indices has {len(probe_layer_indices)} entries "
                f"but ec_values has {n_samples}; lengths must match."
            )

    theta_pred = np.empty(n_samples, dtype=float)
    num_horizons = len(fit.horizon_fits)

    # Per-horizon lookup: EM point index -> position within that horizon's
    # target_indices array, built once per horizon rather than per sample.
    _lookup_cache: dict[int, dict[int, int]] = {}

    for i, (ec_val, layer_idx) in enumerate(zip(ec_values, probe_layer_indices)):
        layer_idx = min(max(int(layer_idx), 0), num_horizons - 1)
        hz = fit.horizon_fits[layer_idx]

        if layer_idx not in _lookup_cache:
            _lookup_cache[layer_idx] = {
                int(pt): pos for pos, pt in enumerate(hz.target_indices)
            }
        lookup = _lookup_cache[layer_idx]

        pt = int(point_indices[i])
        if pt not in lookup:
            raise ValueError(
                f"EM point index {pt} was not among the target_indices this "
                f"horizon ({layer_idx}) was localized to -- pass it in "
                "target_indices when calling "
                "calibrate_petrophysical_relationship_localized, or restrict "
                "point_indices to points that were."
            )
        pos = lookup[pt]
        theta_pred[i] = waxman_smits_theta(
            np.array([ec_val]), hz.a[pos], hz.m[pos], hz.sigma_w[pos], hz.sigma_s[pos],
        )[0]

    return theta_pred


# ==========================================================================
# 7. CUMULATIVE MODEL AVERAGING ENGINE
# ==========================================================================

def _compute_rmse_weights(rmse_ranked: np.ndarray) -> np.ndarray:
    """
    Compute normalised RMSE importance weights for a ranked sub-ensemble.

    Implements Eq. 6–7 of Loiseau et al. (2026):

        RMSEnorm,i = (RMSE_i − RMSE_max) / (RMSE_min − RMSE_max)   [Eq. 6]
        w_i = RMSEnorm,i / Σ_j RMSEnorm,j                           [Eq. 7]

    Normalisation maps the best model (lowest RMSE) to 1 and the worst to 0.
    Weights are then re-normalised so they sum to 1 within the sub-ensemble of
    size r.  Because the denominator Σ RMSEnorm,j changes every time a new
    model enters the stack, this function must be called fresh at each rank r.

    Parameters
    ----------
    rmse_ranked : np.ndarray, shape (r,)
        RMSE values for the first ``r`` models **in rank order** (best first,
        i.e. ascending RMSE).  Must contain at least one element.

    Returns
    -------
    weights : np.ndarray, shape (r,)
        Non-negative weights that sum to 1.0.  Better-fitting models (lower
        RMSE) receive larger weights.

    Notes
    -----
    When all RMSE values in the sub-ensemble are identical — including the
    degenerate r = 1 case where min == max — Eq. 6 produces all-zero
    normalised values and the Eq. 7 denominator is zero.  The function falls
    back to equal weights (1/r), which is the mathematical limit of the
    weighting scheme as the RMSE spread tends to zero.
    """
    r = len(rmse_ranked)
    rmse_min = rmse_ranked.min()
    rmse_max = rmse_ranked.max()

    spread = rmse_max - rmse_min
    if spread < 1e-15:
        # All errors identical (or r=1) → equal weights
        return np.full(r, 1.0 / r, dtype=float)

    # Eq. 6 — best model → 1, worst model → 0
    rmse_norm = (rmse_ranked - rmse_max) / (rmse_min - rmse_max)   # shape (r,)

    total = rmse_norm.sum()
    if total < 1e-15:
        # Numerically degenerate after normalisation → equal weights
        return np.full(r, 1.0 / r, dtype=float)

    # Eq. 7 — normalise so weights sum exactly to 1
    return rmse_norm / total                                          # shape (r,)


def average_inverted_models(
    models: np.ndarray,
    fits: Sequence[PetroFit],
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Step 3 Core Engine — cumulative RMSE-weighted ensemble averaging.

    Ranks the N inverted models by increasing RMSE (best first) and builds
    N cumulative weighted averages ρ_r (r = 1 … N) following Loiseau et al.
    (2026), Eq. 5–7:

        ρ_r = Σ_{i=1}^{r}  w_i · ρ_i

    where the weights w_i are derived from each model's normalised RMSE via
    ``_compute_rmse_weights`` (Eq. 6–7).  Lower-error models always contribute
    more than higher-error ones.  The goal is to average out inversion
    artefacts that are inconsistent with the observed soil-water content data.

    Works for both:
      * 3-D arrays → ``(ensemble_size, num_measured_points, num_layers)``
      * 4-D arrays → ``(ensemble_size, n_time, num_measured_points, num_layers)``

    Averaging always operates on the first (ensemble) axis; all remaining
    spatial/temporal dimensions are preserved unchanged.

    Note on NaN cells (real-data time-lapse spatial reference grid)
    -----------------------------------------------------------------
    Since ``run_pipeline.py`` Step 1 now matches real-data time-lapse
    epochs onto a shared spatial reference grid instead of trimming to a
    common point COUNT (see ``geometry_utils.match_epochs_to_reference_grid``),
    ``models`` may contain NaN at ``(t, point)`` combinations that a given
    survey epoch didn't actually cover. That NaN pattern is identical for
    every ensemble member at a fixed ``(t, point)`` (it is a function of the
    epoch/point only, not of which inversion parameters produced the
    member), so the plain weighted sum below already propagates NaN
    correctly cell-by-cell without needing a NaN-aware reduction: a cell is
    NaN in ``rho_r`` if and only if it was NaN for every member being
    averaged, i.e. if and only if that reference-grid point genuinely has
    no data at that epoch.

    Parameters
    ----------
    models : np.ndarray
        Inverted EC ensemble.  First axis = ensemble members; remaining axes
        = spatial / temporal dimensions.
    fits : sequence of PetroFit
        One calibration result per ensemble member, same order as ``models``.
        Only the scalar ``fit.rmse`` attribute is used.

    Returns
    -------
    averaged_models : list of np.ndarray
        Length-N list where ``averaged_models[r-1]`` is the weighted average
        of the r best-fitting inverted models (same shape as ``models[0]``).
        ``averaged_models[0]``  → single best model (r = 1, weight = 1).
        ``averaged_models[-1]`` → weighted average of all N models (r = N).
    ranking : np.ndarray, shape (N,)
        Original indices into ``models`` / ``fits`` sorted by increasing RMSE.
        ``ranking[0]`` is the best member; ``ranking[-1]`` is the worst.
        Passed to Step 5 to retrieve the top-K subset for CV calculation.

    Algorithm
    ---------
    For each cumulative rank r = 1 … N:

    1. Slice the RMSE values of the first r ranked models.
    2. Recompute normalised weights via ``_compute_rmse_weights`` (Eq. 6–7).
       Weights are recomputed at every r because the normalisation denominator
       Σ RMSEnorm,j changes as new models enter the stack.
    3. Build the weighted sum using NumPy broadcasting:

           w_view  shape: (r, 1, 1, …)   — one extra dim per spatial axis
           models  shape: (r, *spatial)
           result  shape: (*spatial)       = ρ_r

    Notes
    -----
    FIX-1 (vs. previous version): the original code used a plain running sum
    divided by rank r, giving every model equal weight 1/r regardless of
    fit quality.  This version implements the paper's Eq. 6–7 exactly.
    """
    # ------------------------------------------------------------------ #
    # 1. Rank models from best (lowest RMSE) to worst                     #
    # ------------------------------------------------------------------ #
    rmse_values = np.array([f.rmse for f in fits], dtype=float)
    ranking     = np.argsort(rmse_values)           # ascending RMSE order

    N = len(ranking)
    averaged_models: list[np.ndarray] = []

    # ------------------------------------------------------------------ #
    # 2. Build cumulative weighted averages r = 1 … N                    #
    # ------------------------------------------------------------------ #
    for r in range(1, N + 1):

        # Indices and RMSE values for the first r ranked models
        top_r_indices = ranking[:r]                         # shape (r,)
        top_r_rmse    = rmse_values[top_r_indices]          # shape (r,)

        # RMSE-normalised importance weights — recomputed at every r
        weights = _compute_rmse_weights(top_r_rmse)         # shape (r,)

        # Broadcast weights over all non-ensemble spatial/temporal dims.
        # models[top_r_indices] has shape (r, *spatial_dims).
        # w_view needs shape      (r, 1, 1, …) to broadcast correctly.
        n_extra_dims = models.ndim - 1
        w_view = weights.reshape((r,) + (1,) * n_extra_dims)

        rho_r = np.sum(w_view * models[top_r_indices].astype(float), axis=0)
        averaged_models.append(rho_r)

    return averaged_models, ranking


# ==========================================================================
# 8. RECALIBRATION, SELECTION & UNCERTAINTY MANAGEMENT
# ==========================================================================

def recalibrate_petrophysical_relationship(
    averaged_models: Sequence[np.ndarray],
    sample_positions: Sequence[int],
    sample_times: Sequence[int],
    theta_obs: np.ndarray,
    depths0: Sequence[float] | None = None,
    lab_p0: np.ndarray | None = None,
    probe_layer_indices: Sequence[int] | None = None,
    depths_are_fixed: bool = True,
    fixed_params: dict[str, bool] | None = None,
    warm_start_continuation: bool = True,
    probe_depths_reflect_fitted_interfaces: bool = False,
    theta_obs_std: float | Sequence[float] | None = None,
    ws_bounds: dict | None = None,
) -> list[PetroFit]:
    """
    Step 4 Core Engine: Re-applies Waxman-Smits regressions across cumulative
    stacked matrix horizons to find optimal data integration boundaries.

    ``depths_are_fixed`` is forwarded to ``calibrate_petrophysical_relationship``
    so the free-depth message is emitted here too when interfaces were not
    fixed during Step 1.

    ``probe_depths_reflect_fitted_interfaces`` is forwarded unchanged -- set
    to ``True`` when ``probe_layer_indices`` was computed against the
    ensemble's fitted interface depths rather than the static starting
    ``depths0``, so the free-depth message downgrades from "[WARN]" to
    "[INFO]" and describes the routing accurately. See
    ``calibrate_petrophysical_relationship`` for details.

    ``fixed_params`` is forwarded unchanged -- see
    ``calibrate_petrophysical_relationship`` for the fixed-parameter mechanism
    (pin one or more of a/m/sigma_w/sigma_s to their prior value).

    ``theta_obs_std`` is forwarded unchanged -- see
    ``calibrate_petrophysical_relationship`` for the TDR-noise observation
    weighting mechanism it enables.

    ``warm_start_continuation`` defaults to ``True`` here (unlike Step 2's
    raw calibration, which defaults to ``False``): ``averaged_models`` is
    stacked in ascending stack-depth order (index r-1 = K = r), so each
    step's converged fit is a meaningful, physically-similar starting point
    for the next. With only a handful of TDR observations per horizon, this
    is what keeps the per-K RMSE trajectory from jumping between unrelated
    local minima -- see the module-level notes on the Step 4 recalibration
    RMSE-vs-K curve. Pass ``False`` to restore the old always-restart-from-p0
    behaviour.
    """
    stacked_averages = np.stack(averaged_models, axis=0)
    return calibrate_petrophysical_relationship(
        models=stacked_averages,
        sample_positions=sample_positions,
        sample_times=sample_times,
        theta_obs=theta_obs,
        depths0=depths0,
        lab_p0=lab_p0,
        probe_layer_indices=probe_layer_indices,
        depths_are_fixed=depths_are_fixed,
        fixed_params=fixed_params,
        warm_start_continuation=warm_start_continuation,
        probe_depths_reflect_fitted_interfaces=probe_depths_reflect_fitted_interfaces,
        theta_obs_std=theta_obs_std,
        ws_bounds=ws_bounds,
    )


def compute_coverage_map(
    models: np.ndarray,
    top_k_indices: np.ndarray,
    ec_min: float = 0.1,
    ec_max: float = 5000.0,
) -> np.ndarray:
    """
    Compute the normalised coverage map for the top-K ensemble subset.

    Coverage quantifies how well each (spatial position, layer) cell is
    constrained by the ensemble: it is defined as the fraction of the K
    selected members that produced a physically plausible EC value at that
    cell, normalised to the range [-1, 1] so that:

        coverage = 2 · (n_valid / K) − 1

    Interpretation
    --------------
    +1.0  All K members returned a valid EC  → fully constrained cell.
     0.0  Half of the members failed         → marginal constraint.
    −1.0  No member returned a valid EC      → unconstrained / masked cell.

    The plotting layer hatchs cells where ``coverage ≤ 0`` (i.e. fewer than
    half the ensemble members produced a physically plausible result).

    A member's EC value at a cell is considered **invalid** when it is:
    * NaN or infinite (numerical failure -- this is also how a real-data
      time-lapse reference-grid point with no coverage at this epoch,
      see ``geometry_utils.match_epochs_to_reference_grid``, is naturally
      folded into "unconstrained" here without any special-casing),
    * below ``ec_min`` [mS/m] (non-physical, below instrument noise floor), or
    * above ``ec_max`` [mS/m] (non-physical saturation / forward-model blow-up).

    Parameters
    ----------
    models : np.ndarray
        Full raw ensemble, shape
        ``(ensemble_size, [n_time,] n_pts, n_layers)``.
    top_k_indices : np.ndarray, shape (K,)
        Indices of the K best-ranked members (from ``ranking[:K]``).
    ec_min : float, optional
        Minimum physically plausible EC [mS/m]. Default 0.1 mS/m.
    ec_max : float, optional
        Maximum physically plausible EC [mS/m]. Default 5000 mS/m.

    Returns
    -------
    coverage : np.ndarray
        Shape matches ``models[0]`` (i.e. ``([n_time,] n_pts, n_layers)``).
        Values in [-1, 1]; cells with ``coverage ≤ 0`` are poorly constrained.
    """
    K = len(top_k_indices)
    subset = models[top_k_indices].astype(float)   # (K, [n_time,] n_pts, n_layers)

    valid = (
        np.isfinite(subset) &
        (subset >= ec_min) &
        (subset <= ec_max)
    )                                               # (K, [n_time,] n_pts, n_layers)

    n_valid = valid.sum(axis=0).astype(float)       # ([n_time,] n_pts, n_layers)

    # Normalise to [-1, 1]: 0 valid → -1, all valid → +1
    coverage = 2.0 * (n_valid / K) - 1.0

    return coverage


def select_and_assess_model(
    models: np.ndarray,
    averaged_models: Sequence[np.ndarray],
    ranking: np.ndarray,
    recalibration_fits: Sequence[PetroFit],
) -> SelectedModel:
    """
    Step 5 Core Engine — select optimal stack depth K and compute weighted CV map.

    Identifies the averaged model ρ_K with the lowest RMSE among the N
    recalibrated ensemble models (Step 4 fits), then quantifies spatial
    reliability via the coefficient of variation (CV) following Loiseau et al.
    (2026), Eq. 8:

        CV_c = sd_c / ρ_c × 100

    where ρ_c and sd_c are the *weighted* mean and weighted standard deviation
    across the K constituent models, using the same RMSE-normalised weights
    (Eq. 6–7) applied in Step 3.

    Parameters
    ----------
    models : np.ndarray
        Original inverted EC ensemble, shape
        ``(ensemble_size, num_measured_points, num_layers)`` or
        ``(ensemble_size, n_time, num_measured_points, num_layers)``.
        Used to retrieve the top-K individual models for CV computation.
    averaged_models : sequence of np.ndarray
        Cumulative averaged models from Step 3, length N.
        ``averaged_models[r-1]`` is the weighted average of the r best models.
    ranking : np.ndarray, shape (N,)
        Indices into ``models`` sorted by ascending RMSE (best first).
        Produced by ``average_inverted_models``.
    recalibration_fits : sequence of PetroFit
        Step 4 re-calibration results, one per averaged model (same order as
        ``averaged_models``).  The RMSE from these fits drives K selection.

    Returns
    -------
    SelectedModel
        Container holding K, min RMSE, the selected EC model, the weighted
        CV map, and the ranking array.

    Notes
    -----
    FIX-2 (vs. previous version): the original code used ``np.mean`` and
    ``np.std`` (unweighted) over the top-K individual models.  The paper
    requires the *weighted* mean ρ_c and the *weighted* standard deviation
    sd_c (Eq. 8), using the same importance weights w_i derived from RMSE
    normalisation (Eq. 6–7).  This version recomputes those weights here
    for the final K-model subset and applies them to both statistics.

    CV interpretation (Loiseau et al., 2026, Sec. 2.5):
      CV > 100 % → high uncertainty; interpret resistivity / θ with caution.
    """
    # ------------------------------------------------------------------ #
    # 1. Identify the optimal stack depth K (lowest Step-4 RMSE)         #
    # ------------------------------------------------------------------ #
    rmse_values  = np.array([f.rmse for f in recalibration_fits], dtype=float)
    best_r_index = int(np.argmin(rmse_values))
    K            = best_r_index + 1

    ec_K   = averaged_models[best_r_index]
    rmse_K = float(rmse_values[best_r_index])

    # ------------------------------------------------------------------ #
    # 2. Retrieve the K individual models that make up ρ_K               #
    # ------------------------------------------------------------------ #
    top_k_indices = ranking[:K]                        # shape (K,)
    top_k_models  = models[top_k_indices].astype(float)  # (K, *spatial)

    # ------------------------------------------------------------------ #
    # 3. Recompute RMSE-normalised weights for the top-K subset (Eq. 6–7)#
    #    (same formula as _compute_rmse_weights, inlined here to avoid    #
    #     a cross-function dependency on the private helper)              #
    # ------------------------------------------------------------------ #
    top_k_rmse = rmse_values[top_k_indices]            # ascending order

    rmse_min = top_k_rmse.min()
    rmse_max = top_k_rmse.max()
    spread   = rmse_max - rmse_min

    if spread < 1e-15 or K == 1:
        # All errors identical or single model → equal weights
        weights = np.full(K, 1.0 / K, dtype=float)
    else:
        rmse_norm = (top_k_rmse - rmse_max) / (rmse_min - rmse_max)   # best→1, worst→0
        total     = rmse_norm.sum()
        weights   = rmse_norm / total if total >= 1e-15 else np.full(K, 1.0 / K)

    # ------------------------------------------------------------------ #
    # 4. Weighted mean ρ_c and weighted standard deviation sd_c (Eq. 8)  #
    # ------------------------------------------------------------------ #
    # Broadcast weights over all spatial/temporal trailing dimensions.
    n_extra = top_k_models.ndim - 1
    w_view  = weights.reshape((K,) + (1,) * n_extra)       # (K, 1, 1, …)

    cell_mean = np.sum(w_view * top_k_models, axis=0)       # weighted mean ρ_c

    # Weighted standard deviation with Bessel-like correction for K > 1:
    #   sd_c = sqrt( K/(K-1) · Σ w_i · (ρ_i − ρ_c)² )
    # When K == 1 the std is zero by definition.
    if K == 1:
        cell_std = np.zeros_like(cell_mean)
    else:
        sq_dev   = w_view * (top_k_models - cell_mean) ** 2   # (K, *spatial)
        cell_std = np.sqrt((K / (K - 1)) * np.sum(sq_dev, axis=0))

    # ------------------------------------------------------------------ #
    # 5. CV map (Eq. 8) — expressed as a fraction (not %)                #
    #    High CV (> 1.0, i.e. > 100 %) flags high-uncertainty zones.     #
    #    A NaN cell_mean (all top-K members NaN at this cell -- e.g. a   #
    #    real-data time-lapse reference-grid point with no coverage at   #
    #    this epoch) is preserved as NaN here rather than folded into    #
    #    0.0, so it is never mistaken for "zero uncertainty".            #
    # ------------------------------------------------------------------ #
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_map = np.where(np.abs(cell_mean) > 1e-5,
                          cell_std / cell_mean,
                          np.where(np.isfinite(cell_mean), 0.0, np.nan))

    # ------------------------------------------------------------------ #
    # 6. Coverage map — fraction of top-K members with valid EC per cell  #
    # Normalised to [-1, +1]: > 0 → well constrained, ≤ 0 → poorly       #
    # constrained (hatched in plots).                                     #
    # ------------------------------------------------------------------ #
    coverage = compute_coverage_map(models, top_k_indices)

    return SelectedModel(K=K, rmse=rmse_K, ec_model=ec_K, cv_map=cv_map,
                         ranking=ranking, coverage=coverage)


def ec_to_water_content(
    ec_values: np.ndarray,
    fit: PetroFit,
    probe_layer_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Convert bulk EC values to volumetric water content θ using a fitted
    ``PetroFit`` object (Waxman-Smits model, per-horizon parameters).

    This is the public convenience wrapper that ``step4_select.py``
    calls when converting EC predictions to θ for RMSE evaluation.  It
    applies each horizon's calibrated [a, m, σ_w, σ_s] parameters to the
    corresponding EC observations, routing them by horizon index.

    Parameters
    ----------
    ec_values : np.ndarray, shape (n_samples,)
        Bulk electrical conductivity values [mS/m] extracted from an
        averaged model at the probe locations.  One value per observation
        (i.e. one scalar per probe per time step).
    fit : PetroFit
        Calibrated petrophysical result for one ensemble member (or averaged
        model).  ``fit.horizon_fits`` provides per-horizon [a, m, σ_w, σ_s].
    probe_layer_indices : sequence of int, optional
        0-based horizon index for each entry in ``ec_values``, in the same
        order.  When supplied, each EC value is converted using the parameters
        of its own horizon.  When ``None``, all values are converted using the
        parameters of horizon 0 (safe fallback for single-horizon surveys or
        when horizon routing is not available).

    Returns
    -------
    theta_pred : np.ndarray, shape (n_samples,)
        Predicted volumetric water content [m³/m³], one value per input EC
        observation. Entries where ``ec_values`` is NaN (e.g. a probe/epoch
        combination with no EC value on the time-lapse spatial reference
        grid) come out as NaN as well -- callers computing an RMSE against
        these should use a NaN-aware reduction (``np.nanmean`` /
        ``np.nanstd``) rather than ``np.mean``/``np.std``.

    Notes
    -----
    FIX-9: this function was called in ``step4_select.py`` but did not
    exist in ``em_methodology.py``.  The equivalent computation was buried
    inside ``calibrate_petrophysical_relationship`` with no public entry point.

    Examples
    --------
    >>> import numpy as np
    >>> # Minimal smoke-test (see integration test below for full coverage)
    >>> ec = np.array([50.0, 120.0, 200.0])
    >>> # fit.horizon_fits[0] must have .a, .m, .sigma_w, .sigma_s attributes
    """
    ec_values = np.asarray(ec_values, dtype=float)
    n_samples  = len(ec_values)
    theta_pred = np.empty(n_samples, dtype=float)

    num_horizons = len(fit.horizon_fits)

    if probe_layer_indices is None:
        # No horizon routing → use horizon 0 for all samples (safe fallback)
        hz = fit.horizon_fits[0]
        theta_pred = waxman_smits_theta(ec_values, hz.a, hz.m, hz.sigma_w, hz.sigma_s)
    else:
        probe_layer_indices = list(probe_layer_indices)
        if len(probe_layer_indices) != n_samples:
            raise ValueError(
                f"probe_layer_indices has {len(probe_layer_indices)} entries "
                f"but ec_values has {n_samples}; lengths must match."
            )
        for i, (ec_val, layer_idx) in enumerate(zip(ec_values, probe_layer_indices)):
            # Clamp to valid horizon range in case of edge mismatch
            layer_idx = min(max(int(layer_idx), 0), num_horizons - 1)
            hz = fit.horizon_fits[layer_idx]
            theta_pred[i] = waxman_smits_theta(
                np.array([ec_val]), hz.a, hz.m, hz.sigma_w, hz.sigma_s
            )[0]

    return theta_pred


# ==========================================================================
# 9. QUALITY GATES — uncertainty & calibration coverage as pass/warn signals
#    (pipeline_improvement_report.md, section 4: "Uncertainty & quality as
#    gates, not just numbers")
#
#    Two independent checks live here:
#
#    * check_early_horizon_coverage()  — run right after data loading
#      (Step 0), BEFORE the expensive Step 1 ensemble inversion. Uses only
#      depths0 and known probe/sensor installation depths, so it costs
#      nothing and can catch a depths0/swc_depths mismatch that would
#      otherwise only surface as an "UNCALIBRATED" horizon after a 30+
#      minute inversion run.
#
#    * evaluate_quality_gates()  — run after Step 5 (selection/CV mapping).
#      Turns the CV map and per-horizon calibration status that the
#      pipeline already computes into a single pass/warn verdict, so a run
#      with e.g. 30% of cells above CV=1.0 or half the horizons
#      uncalibrated no longer reports the same "SUCCESS" status as a clean
#      run. Never aborts the run by itself -- see run_pipeline.py's
#      PipelineReporter.finish(quality_level=...) for how the verdict is
#      surfaced in the final report status.
# ==========================================================================

@dataclass
class HorizonCoverageResult:
    """
    Result of ``check_early_horizon_coverage`` — maps every known TDR/SWC
    probe depth to its 0-based soil horizon and flags any horizon with zero
    probes mapped to it.

    Attributes
    ----------
    n_layers : int
        Total number of soil horizons (= len(depths0) + 1).
    covered_horizons, uncovered_horizons : list[int]
        Horizon indices with >=1 / 0 probe depths mapped to them.
    horizon_probe_counts : dict[int, int]
        Per-horizon probe-depth count (horizon index -> count).
    level : str
        "OK" or "WARN".
    message : str
        Human-readable summary line, ready to log/report as-is.
    """
    n_layers: int
    covered_horizons: list[int]
    uncovered_horizons: list[int]
    horizon_probe_counts: dict[int, int]
    level: str
    message: str


def check_early_horizon_coverage(
    probe_depths: Sequence[float],
    depths0: Sequence[float],
) -> HorizonCoverageResult:
    """
    Early (pre-inversion) check: will every soil horizon implied by
    ``depths0`` actually have at least one TDR/SWC probe installed in it?

    This mirrors the horizon-routing logic used later by
    ``calibrate_petrophysical_relationship`` (via ``layer_index_from_depth``),
    but is deliberately standalone and cheap: it needs only the static
    ``depths0`` and the set of known probe/sensor installation depths --
    neither requires the EM survey to have been inverted. Calling this
    immediately after Step 0 (real or synthetic data loading) surfaces a
    depths0/sensor-depth mismatch (report item 4, second suggestion) BEFORE
    the pipeline spends 30+ minutes on Step 1, rather than only discovering
    it as an "UNCALIBRATED" horizon in the Step 2/4 output.

    Parameters
    ----------
    probe_depths : sequence of float
        Every known TDR/SWC sensor installation depth [m] that will be used
        for calibration (may contain duplicates -- e.g. one depth per
        spatial position -- duplicates only affect the reported count, not
        the coverage verdict).
    depths0 : sequence of float
        Fixed/starting layer interface depths [m], length = n_layers - 1.

    Returns
    -------
    HorizonCoverageResult
    """
    n_layers = len(depths0) + 1
    counts: dict[int, int] = {i: 0 for i in range(n_layers)}
    for d in probe_depths:
        idx = layer_index_from_depth(d, depths0)
        idx = min(max(idx, 0), n_layers - 1)
        counts[idx] += 1

    covered = [i for i in range(n_layers) if counts[i] > 0]
    uncovered = [i for i in range(n_layers) if counts[i] == 0]
    level = "WARN" if uncovered else "OK"

    if level == "OK":
        message = (
            f"[COVERAGE] OK — all {n_layers} horizon(s) have >=1 TDR/SWC "
            f"probe depth mapped. Per-horizon counts: {counts}."
        )
    else:
        message = (
            f"[COVERAGE][WARN] {len(uncovered)}/{n_layers} horizon(s) have NO "
            f"TDR/SWC probe depth mapped to them (depths0={list(depths0)}): "
            f"uncovered horizon(s) {uncovered}. These horizons cannot be "
            f"calibrated no matter how the ensemble inversion runs -- they "
            f"will fall back to prior/uncalibrated Waxman-Smits parameters "
            f"in Steps 2/4. Fix depths0 or the SWC/TDR sensor depths before "
            f"committing to a full run if calibration of these horizons "
            f"matters. Per-horizon probe counts: {counts}."
        )

    return HorizonCoverageResult(
        n_layers=n_layers,
        covered_horizons=covered,
        uncovered_horizons=uncovered,
        horizon_probe_counts=counts,
        level=level,
        message=message,
    )


@dataclass
class QualityGateResult:
    """
    Result of ``evaluate_quality_gates`` — the post-Step-5 pass/warn verdict
    on the final selected solution's uncertainty (CV map) and calibration
    coverage (which horizons actually got fit against real TDR data).

    Attributes
    ----------
    level : str
        "OK" or "WARN".
    cv_mean : float
        Mean of the (fractional, not %) CV map at the optimal stack depth.
    cv_high_fraction : float
        Fraction of CV-map cells exceeding ``high_cv_threshold``.
    high_cv_threshold : float
        The CV value (fraction, e.g. 1.0 = 100%) above which a cell is
        counted as "high uncertainty" for ``cv_high_fraction``.
    n_calibrated_horizons, n_total_horizons : int
        How many of the solution's soil horizons had real TDR coverage
        (``PetroFit.n_calibrated_horizons`` / total horizon count).
    reasons : list[str]
        Human-readable list of every threshold that was exceeded (empty
        when level == "OK").
    message : str
        Single summary line, ready to log/report as-is.
    """
    level: str
    cv_mean: float
    cv_high_fraction: float
    high_cv_threshold: float
    n_calibrated_horizons: int
    n_total_horizons: int
    reasons: list[str]
    message: str


def evaluate_quality_gates(
    cv_map: np.ndarray,
    best_fit: "PetroFit",
    cv_mean_warn_threshold: float = 0.3,
    cv_high_threshold: float = 1.0,
    cv_high_fraction_warn_threshold: float = 0.3,
    require_all_horizons_calibrated: bool = True,
) -> QualityGateResult:
    """
    Evaluate the final selected solution's CV map and per-horizon
    calibration status against configurable thresholds, turning numbers
    that were already being computed (but only ever printed/plotted) into
    an explicit pass/warn signal (report item 4: "Uncertainty & quality as
    gates, not just numbers").

    Triggers ``level="WARN"`` when ANY of the following hold:

    * ``mean(cv_map) > cv_mean_warn_threshold``
    * ``fraction(cv_map > cv_high_threshold) > cv_high_fraction_warn_threshold``
    * ``require_all_horizons_calibrated`` is True and at least one horizon
      in ``best_fit.horizon_fits`` has ``is_calibrated == False``

    This function NEVER aborts the run and never raises -- it only produces
    a verdict for the caller (typically ``run_pipeline.py``) to log and to
    fold into the final report status via
    ``PipelineReporter.finish(quality_level=...)``, so a degraded-but-
    completed run is visibly distinguishable from a clean one instead of
    both reporting the same "✅ SUCCESS".

    Parameters
    ----------
    cv_map : np.ndarray
        The selected solution's Coefficient-of-Variation map (fractional,
        e.g. ``selected_output.cv_map`` from Step 4/5). NaNs are ignored.
    best_fit : PetroFit
        The petrophysical fit for the selected solution (typically
        ``selected_output.best_fit``); its ``horizon_fits`` list is used to
        count calibrated vs. uncalibrated horizons.
    cv_mean_warn_threshold : float, optional
        Mean-CV warn threshold (fractional, default 0.3 = 30%).
    cv_high_threshold : float, optional
        CV value (fractional) above which a cell counts as "high
        uncertainty" for the fraction check. Default 1.0 (100%), matching
        the paper's own CV>100% "interpret with caution" guidance.
    cv_high_fraction_warn_threshold : float, optional
        Warn if more than this fraction of cells exceed ``cv_high_threshold``.
        Default 0.3 (30% of cells).
    require_all_horizons_calibrated : bool, optional
        When True (default), any uncalibrated horizon in the selected
        solution triggers a WARN.

    Returns
    -------
    QualityGateResult
    """
    reasons: list[str] = []
    cv_arr = np.asarray(cv_map, dtype=float)
    cv_mean = float(np.nanmean(cv_arr)) if cv_arr.size else float("nan")
    cv_high_fraction = float(np.nanmean(cv_arr > cv_high_threshold)) if cv_arr.size else 0.0

    if np.isfinite(cv_mean) and cv_mean > cv_mean_warn_threshold:
        reasons.append(
            f"mean CV {cv_mean:.3f} exceeds warn threshold {cv_mean_warn_threshold:.3f}"
        )
    if cv_high_fraction > cv_high_fraction_warn_threshold:
        reasons.append(
            f"{cv_high_fraction*100:.1f}% of cells have CV > {cv_high_threshold:.2f} "
            f"(warn threshold {cv_high_fraction_warn_threshold*100:.1f}%)"
        )

    n_total = len(best_fit.horizon_fits)
    n_calibrated = sum(1 for h in best_fit.horizon_fits if h.is_calibrated)
    if require_all_horizons_calibrated and n_calibrated < n_total:
        uncalibrated = [h.horizon for h in best_fit.horizon_fits if not h.is_calibrated]
        reasons.append(
            f"{n_total - n_calibrated}/{n_total} horizon(s) uncalibrated "
            f"(no TDR probe coverage): horizon(s) {uncalibrated}"
        )

    level = "WARN" if reasons else "OK"
    if level == "OK":
        message = (
            f"[QUALITY GATE] OK — mean CV={cv_mean:.3f}, "
            f"CV>{cv_high_threshold:.2f} fraction={cv_high_fraction*100:.1f}%, "
            f"{n_calibrated}/{n_total} horizons calibrated."
        )
    else:
        message = (
            "[QUALITY GATE][WARN] Final solution quality flagged: "
            + "; ".join(reasons) + "."
        )

    return QualityGateResult(
        level=level,
        cv_mean=cv_mean,
        cv_high_fraction=cv_high_fraction,
        high_cv_threshold=cv_high_threshold,
        n_calibrated_horizons=n_calibrated,
        n_total_horizons=n_total,
        reasons=reasons,
        message=message,
    )
