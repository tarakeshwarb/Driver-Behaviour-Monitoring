"""
sequences.py — Sequence Construction and Train/Test Split Utilities
====================================================================
Provides helpers for:
  - Converting feature DataFrames into ordered HMM observation sequences
  - Chronological train/val/test splitting at the sequence level
  - Window-length sensitivity experiments
  - Normalisation applied correctly per split (no leakage)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sequence assembly
# ---------------------------------------------------------------------------

def build_sequences_from_features(
    feature_df: pd.DataFrame,
    groupby: str = "trip_id",
    sort_col: str = "t_start",
) -> tuple[list[np.ndarray], list[int], list[dict]]:
    """
    Group windowed feature rows into ordered observation sequences.

    Each trip's windows are sorted by t_start to preserve temporal order.
    Sequences are NEVER shuffled.

    Parameters
    ----------
    feature_df : Combined feature DataFrame (all trips)
    groupby    : Column to group by ('trip_id' recommended)
    sort_col   : Column to sort within each group ('t_start')

    Returns
    -------
    sequences : list of (N_i, D) float64 arrays — one per trip
    lengths   : list of sequence lengths N_i
    meta      : list of dicts with trip-level metadata
    """
    sequences, lengths, meta = [], [], []

    for key, grp in feature_df.groupby(groupby):
        grp = grp.sort_values(sort_col)
        obs = grp[FEATURE_COLUMNS].values.astype(np.float64)

        # Drop windows with any NaN/Inf
        valid = np.all(np.isfinite(obs), axis=1)
        obs = obs[valid]

        if len(obs) < 3:
            logger.warning("Skipping short sequence '%s' (%d windows)", key, len(obs))
            continue

        sequences.append(obs)
        lengths.append(len(obs))

        first = grp.iloc[0]
        meta.append({
            "trip_id": str(key),
            "label":   first.get("label",  "UNKNOWN"),
            "road":    first.get("road",   "UNKNOWN"),
            "driver":  first.get("driver", "UNKNOWN"),
            "n_windows": len(obs),
        })

    logger.info(
        "Built %d sequences, total %d windows", len(sequences), sum(lengths)
    )
    return sequences, lengths, meta


# ---------------------------------------------------------------------------
# Normalisation that respects train/test boundaries
# ---------------------------------------------------------------------------

class SequenceNormaliser:
    """
    Fits StandardScaler on training sequences only, then transforms all splits.

    Usage
    -----
    normaliser = SequenceNormaliser()
    train_seqs = normaliser.fit_transform(train_seqs)
    val_seqs   = normaliser.transform(val_seqs)
    test_seqs  = normaliser.transform(test_seqs)
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, sequences: list[np.ndarray]) -> "SequenceNormaliser":
        X = np.vstack(sequences)
        self.scaler.fit(X)
        self._fitted = True
        return self

    def transform(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        if not self._fitted:
            raise RuntimeError("SequenceNormaliser must be fitted before transform.")
        return [self.scaler.transform(s) for s in sequences]

    def fit_transform(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        return self.fit(sequences).transform(sequences)

    def inverse_transform(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        return [self.scaler.inverse_transform(s) for s in sequences]


# ---------------------------------------------------------------------------
# Chronological split at sequence level
# ---------------------------------------------------------------------------

def split_sequences_chronologically(
    sequences:  list[np.ndarray],
    meta:       list[dict],
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    gap:        int   = 2,
) -> tuple:
    """
    Split sequences chronologically by trip_id (which embeds date/time).

    Parameters
    ----------
    sequences  : list of observation arrays
    meta       : corresponding list of metadata dicts
    train_frac : fraction for training
    val_frac   : fraction for validation
    gap        : number of sequences to skip around boundaries

    Returns
    -------
    (train_seqs, val_seqs, test_seqs,
     train_meta, val_meta, test_meta)
    """
    # Sort by trip_id (YYYYMMDDHHMMSS prefix ensures chronological order)
    order = sorted(range(len(meta)), key=lambda i: meta[i].get("trip_id", ""))
    seqs_sorted = [sequences[i] for i in order]
    meta_sorted = [meta[i]      for i in order]

    n = len(seqs_sorted)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_seqs = seqs_sorted[:n_train]
    train_meta = meta_sorted[:n_train]

    v_start = n_train + gap
    v_end   = v_start + n_val
    val_seqs = seqs_sorted[v_start:v_end]
    val_meta = meta_sorted[v_start:v_end]

    t_start  = v_end + gap
    test_seqs = seqs_sorted[t_start:]
    test_meta = meta_sorted[t_start:]

    logger.info(
        "Split: %d train | %d val | %d test sequences",
        len(train_seqs), len(val_seqs), len(test_seqs),
    )
    return train_seqs, val_seqs, test_seqs, train_meta, val_meta, test_meta


# ---------------------------------------------------------------------------
# Window-length sensitivity experiment helper
# ---------------------------------------------------------------------------

def prepare_window_experiment(
    trips: list[dict],
    window_seconds_list: list[float],
) -> dict[float, dict]:
    """
    For each window duration, extract features and build sequences.

    Returns a dict mapping window_seconds → {sequences, meta, feature_df}.
    Used in the hyperparameter sensitivity experiment in notebooks.
    """
    from src.features import extract_all_features, WindowConfig

    results = {}
    for ws in window_seconds_list:
        cfg = WindowConfig(window_seconds=ws)
        feat_df = extract_all_features(trips, cfg)
        if feat_df.empty:
            continue
        seqs, lengths, meta = build_sequences_from_features(feat_df)
        results[ws] = {
            "sequences":  seqs,
            "lengths":    lengths,
            "meta":       meta,
            "feature_df": feat_df,
        }
        logger.info("Window=%.1f s: %d sequences, %d total windows",
                    ws, len(seqs), sum(lengths))
    return results

