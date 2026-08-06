#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
em_sensitivity.py
==================
McNeill (1980) cumulative EM response kernels + depth-weighted layer->point
comparison. Closed-form (computed once, offline) -- the EM analogue of the
ERT paper's fixed horizontal integration window, but physically grounded
per coil geometry instead of empirically tuned by RMSE sweep.

Caveat: valid in the low-induction-number (LIN) limit over a
layered/half-space assumption. Treat results from forwardModel="CS" (or the
higher GEM-2 frequencies, where LIN degrades) as approximate -- a
diagnostic, not a re-derivation of the inversion.

Real vs. synthetic coil geometry (see run_pipeline.py wiring): the real-data
path uses a single geometry (HCP, 1.66 m, six frequencies) so one weight
vector is exact for that branch. Synthetic scenarios instead sweep six
distinct (orientation, spacing) pairs feeding the SAME inversion
(``0_generate_synthetic_data.COILS_LIST``), and there is no single
physically-correct way to collapse six different footprints into one
kernel -- the inversion itself already blends their information
nonlinearly. ``layer_weights_multi_coil`` below is an explicitly-labelled
equal-weighted approximation for that case, not a claim about the array's
true joint sensitivity; see the "representative-coil" alternative in
``layer_weights_multi_coil``'s docstring if a narrower, single-purpose
diagnostic is preferred instead.
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np


def parse_coil_spacing_m(coil_name: str) -> tuple[float | None, str | None]:
    """
    'HCP1.66f425h1_quad' -> (1.66, 'HCP')  (real-data coil-alias format,
    see the coil alias table in run_pipeline.py).
    'VCP0.5' -> (0.5, 'VCP')  (synthetic ``COILS_LIST`` format -- no
    frequency suffix, since synthetic coil names encode geometry only).

    Returns (None, None) with a warning if the name matches neither
    convention -- update the regex if either naming scheme changes.
    """
    m = re.match(r"^([A-Za-z]+)([\d.]+)(?:f.*)?$", coil_name)
    if not m:
        print(f"[EM_SENS][WARN] Could not parse coil spacing from '{coil_name}'")
        return None, None
    orientation, spacing = m.group(1).upper(), float(m.group(2))
    return spacing, orientation


def phi_hcp(z_norm: np.ndarray) -> np.ndarray:
    """McNeill (1980) cumulative response fraction ABOVE depth z, HCP coils."""
    return 1.0 / np.sqrt(4.0 * z_norm**2 + 1.0)


def phi_vcp(z_norm: np.ndarray) -> np.ndarray:
    """McNeill (1980) cumulative response fraction ABOVE depth z, VCP coils."""
    return np.sqrt(4.0 * z_norm**2 + 1.0) - 2.0 * z_norm


def layer_weights(
    depths0: Sequence[float], coil_spacing_m: float, orientation: str = "HCP",
) -> np.ndarray:
    """
    depths0 : layer BOTTOM depths [m] -- same convention as this pipeline's
              DEPTHS0/EFFECTIVE_DEPTHS0 -- plus an implicit half-space below
              depths0[-1].

    Returns an array of length len(depths0)+1 (finite layers + half-space)
    summing to 1: weight_i = fraction of the coil's total secondary-field
    response attributable to layer i, given phi(0)=1 (all response from
    the surface down) and phi decreasing monotonically with depth.
    """
    phi = phi_hcp if orientation.upper() == "HCP" else phi_vcp
    edges = np.array([0.0] + list(depths0)) / coil_spacing_m
    phi_edges = phi(edges)
    w_layers = -np.diff(phi_edges)        # response contributed by each finite layer
    w_half_space = phi_edges[-1]          # remainder, from below depths0[-1]
    weights = np.append(w_layers, w_half_space)
    return weights / weights.sum()        # renormalize (guards float error only)


def layer_weights_multi_coil(
    depths0: Sequence[float], coil_names: Sequence[str],
) -> np.ndarray:
    """
    Equal-weighted average of ``layer_weights`` across every distinct
    (spacing, orientation) pair parsed out of ``coil_names`` -- the
    "multi-coil average mode" for synthetic scenarios (see module
    docstring). NOT the array's true joint sensitivity (the inversion
    blends the six geometries' information nonlinearly, not by averaging
    their individual footprints); this is an approximation, worth labelling
    as one wherever it's reported.

    Alternative if you want a narrower, single-purpose diagnostic instead:
    pick the single coil whose spacing is closest to your shallowest
    layer's thickness (the layer you most care about resolving) and call
    ``layer_weights`` directly with that one geometry ("representative-coil
    mode") rather than averaging all six.

    Coil names that fail to parse (``parse_coil_spacing_m`` returns
    ``(None, None)``) are skipped with a warning, not silently dropped.
    """
    seen: dict[tuple[float, str], None] = {}
    for name in coil_names:
        spacing, orientation = parse_coil_spacing_m(name)
        if spacing is None:
            continue
        seen[(spacing, orientation)] = None

    if not seen:
        raise ValueError(
            f"[EM_SENS] None of the provided coil_names {list(coil_names)!r} "
            "parsed to a valid (spacing, orientation) pair -- cannot compute "
            "multi-coil weights."
        )

    per_coil_weights = [
        layer_weights(depths0, coil_spacing_m=spacing, orientation=orientation)
        for spacing, orientation in seen
    ]
    stacked = np.stack(per_coil_weights, axis=0)
    averaged = stacked.mean(axis=0)
    return averaged / averaged.sum()  # renormalize (guards float error only)


def effective_theta(theta_layers: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    theta_layers : (..., n_layers) array of per-layer theta_calc (last axis
                   = layer, matching `weights`'s length).
    weights      : (n_layers,), from layer_weights() / layer_weights_multi_coil(),
                   summing to 1.

    Returns the depth-weighted "effective EM theta" seen by the coil at
    each remaining index (e.g. per position, per time step) -- the number
    to compare against a point TDR reading instead of a single hard-routed
    layer's theta_calc.
    """
    return np.tensordot(theta_layers, weights, axes=([-1], [0]))
