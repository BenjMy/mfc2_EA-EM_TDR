#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train/validation split for TDR-vs-ERT/EM calibration pairs.

Mirrors Loiseau et al. (2026), Sections 2.3 and 2.6: training data drives
petrophysical fitting AND ensemble-K selection; validation data is held out
and touched only once, at the end, to report an unbiased RMSE/r.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SplitIndices:
    train: np.ndarray   # int indices into the flattened obs arrays
    val: np.ndarray
    method: str
    independent: bool   # False if train/val share probes or time steps


def split_train_val(
    theta_obs: np.ndarray,
    sample_positions: np.ndarray,
    sample_times: np.ndarray,
    probe_layer_indices: np.ndarray | None = None,
    probe_id: np.ndarray | None = None,
    method: str = "alternate_sorted",
    seed: int = 0,
) -> SplitIndices:
    """
    Parameters
    ----------
    theta_obs, sample_positions, sample_times, probe_layer_indices
        Same flattened, same-length arrays already passed into
        ``step2.run_petro`` today (one entry per sensor-epoch pair).
    probe_id
        Optional per-observation probe/pit identifier. Required for
        ``method="by_probe"``. Without it, a fully independent split
        isn't possible and the function falls back to ``alternate_sorted``
        with a loud warning.
    method
        - "alternate_sorted"  : sort by theta_obs, take every other value
                                 (what the paper used at data-poor Larzac;
                                 NOT fully independent -- both subsets can
                                 contain the same probe/time step).
        - "by_probe"          : hold out whole probes for validation
                                 (needs probe_id). Fully independent.
        - "by_time_block"     : hold out whole time steps for validation
                                 (what the paper used at Avignon: "using
                                 data from every other time step").
                                 Fully independent w.r.t. time, not probes.
        - "random"            : uniform random 50/50 split, for comparison
                                 only -- not recommended as your primary
                                 split (doesn't guard against pseudo-
                                 replication the way the other methods do).
    """
    n = len(theta_obs)
    rng = np.random.default_rng(seed)

    if method == "by_probe":
        if probe_id is None:
            print(
                "[SPLIT][WARN] method='by_probe' requested but no probe_id "
                "given -- falling back to 'alternate_sorted' (not fully "
                "independent). Pass probe_id once you can identify which "
                "observation came from which physical probe/pit."
            )
            return split_train_val(
                theta_obs, sample_positions, sample_times,
                probe_layer_indices, probe_id, method="alternate_sorted",
                seed=seed,
            )
        uniq_probes = np.unique(probe_id)
        rng.shuffle(uniq_probes)
        half = max(1, len(uniq_probes) // 2)
        train_probes, val_probes = uniq_probes[:half], uniq_probes[half:]
        train = np.where(np.isin(probe_id, train_probes))[0]
        val = np.where(np.isin(probe_id, val_probes))[0]
        return SplitIndices(train, val, method, independent=True)

    if method == "by_time_block":
        uniq_times = np.unique(sample_times)
        train_times = uniq_times[0::2]
        val_times = uniq_times[1::2]
        train = np.where(np.isin(sample_times, train_times))[0]
        val = np.where(np.isin(sample_times, val_times))[0]
        return SplitIndices(train, val, method, independent=True)

    if method == "random":
        idx = rng.permutation(n)
        half = n // 2
        return SplitIndices(idx[:half], idx[half:], method, independent=False)

    # default: "alternate_sorted" -- matches the paper's Larzac fallback
    order = np.argsort(theta_obs)
    train = order[0::2]
    val = order[1::2]
    return SplitIndices(np.sort(train), np.sort(val), method, independent=False)


def subset(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Small helper so call sites read as ``subset(theta_obs, split.train)``."""
    return arr[idx] if arr is not None else None
