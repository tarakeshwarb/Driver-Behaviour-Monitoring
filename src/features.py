"""
features.py — Feature Extraction Module
========================================
Converts synchronised, preprocessed sensor DataFrames into
time-windowed feature vectors suitable for HMM training.

Feature set (per window):
  ax_mean, ax_std, ax_max, ax_min       — longitudinal acceleration
  ay_mean, ay_std, ay_max, ay_min       — lateral acceleration
  az_mean, az_std                        — vertical acceleration
  jerk_long_mean, jerk_long_std, jerk_long_max  — longitudinal jerk
  jerk_lat_mean,  jerk_lat_std,  jerk_lat_max   — lateral jerk
  speed_mean, speed_std                  — speed (m/s)
  speed_diff_mean, speed_diff_max        — speed change per window
  ax_variance, ay_variance               — variance (driving stability)

Jerk: J(t) = [a(t) − a(t−Δt)] / Δt  (m/s³)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WindowConfig:
    """Parameters controlling time-window construction."""
    window_seconds: float = 3.0   # window duration in seconds
    overlap: float = 0.5          # fractional overlap (0 = no overlap)
    imu_fs: float = 10.0          # IMU sampling rate (Hz)
    min_samples: int = 10         # minimum samples per window


FEATURE_COLUMNS = [
    "ax_mean", "ax_std", "ax_max", "ax_min",
    "ay_mean", "ay_std", "ay_max", "ay_min",
    "az_mean", "az_std",
    "jerk_long_mean", "jerk_long_std", "jerk_long_max",
    "jerk_lat_mean",  "jerk_lat_std",  "jerk_lat_max",
    "speed_mean", "speed_std",
    "speed_diff_mean", "speed_diff_max",
    "ax_variance", "ay_variance",
]


# ---------------------------------------------------------------------------
# Jerk calculation
# ---------------------------------------------------------------------------

def compute_jerk(accel: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute jerk from an acceleration signal.

    J(t) = [a(t) − a(t−Δt)] / Δt

    Parameters
    ----------
    accel : 1-D acceleration array (m/s² or g)
    dt    : sampling interval in seconds

    Returns
    -------
    Jerk array (same length; first element = 0).
    """
    if len(accel) < 2:
        return np.zeros_like(accel)
    jerk = np.empty_like(accel)
    jerk[0] = 0.0
    jerk[1:] = np.diff(accel) / dt
    return jerk


# ---------------------------------------------------------------------------
# Single-window feature extraction
# ---------------------------------------------------------------------------

def extract_window_features(
    window: pd.DataFrame,
    dt: float = 0.1,  # 1/fs
) -> dict[str, float]:
    """
    Extract the full feature vector from a single time window.

    Parameters
    ----------
    window : DataFrame slice with IMU + GPS columns
    dt     : sampling interval (seconds)

    Returns
    -------
    Dict mapping feature name → scalar value.
    """
    ax = window["ax_filt"].values if "ax_filt" in window.columns else window["ax_raw"].values
    ay = window["ay_filt"].values if "ay_filt" in window.columns else window["ay_raw"].values
    az = window["az_filt"].values if "az_filt" in window.columns else window["az_raw"].values

    jerk_long = compute_jerk(ax, dt)
    jerk_lat  = compute_jerk(ay, dt)

    speed = window["speed_ms"].values if "speed_ms" in window.columns else np.zeros(len(window))
    speed_diff = np.diff(speed) if len(speed) > 1 else np.array([0.0])

    # Guard against all-NaN windows
    def _safe(fn, arr, fallback=0.0):
        try:
            v = fn(arr[np.isfinite(arr)])
            return float(v) if np.isfinite(v) else fallback
        except Exception:
            return fallback

    return {
        # Longitudinal acceleration
        "ax_mean": _safe(np.mean, ax),
        "ax_std":  _safe(np.std,  ax),
        "ax_max":  _safe(np.max,  ax),
        "ax_min":  _safe(np.min,  ax),
        # Lateral acceleration
        "ay_mean": _safe(np.mean, ay),
        "ay_std":  _safe(np.std,  ay),
        "ay_max":  _safe(np.max,  ay),
        "ay_min":  _safe(np.min,  ay),
        # Vertical acceleration
        "az_mean": _safe(np.mean, az),
        "az_std":  _safe(np.std,  az),
        # Jerk — longitudinal
        "jerk_long_mean": _safe(np.mean, jerk_long),
        "jerk_long_std":  _safe(np.std,  jerk_long),
        "jerk_long_max":  _safe(lambda a: np.max(np.abs(a)), jerk_long),
        # Jerk — lateral
        "jerk_lat_mean": _safe(np.mean, jerk_lat),
        "jerk_lat_std":  _safe(np.std,  jerk_lat),
        "jerk_lat_max":  _safe(lambda a: np.max(np.abs(a)), jerk_lat),
        # Speed
        "speed_mean": _safe(np.mean, speed),
        "speed_std":  _safe(np.std,  speed),
        # Speed change (braking / acceleration magnitude)
        "speed_diff_mean": _safe(np.mean, speed_diff),
        "speed_diff_max":  _safe(lambda a: np.max(np.abs(a)), speed_diff),
        # Variance (driving stability)
        "ax_variance": _safe(np.var, ax),
        "ay_variance": _safe(np.var, ay),
    }


# ---------------------------------------------------------------------------
# Sliding-window sequence builder
# ---------------------------------------------------------------------------

def build_windows(
    merged: pd.DataFrame,
    cfg: WindowConfig | None = None,
) -> Iterator[tuple[dict[str, float], float, float]]:
    """
    Yield (feature_dict, window_start_ts, window_end_ts) for each window.

    Parameters
    ----------
    merged : preprocessed + synchronised DataFrame (must have 'timestamp')
    cfg    : WindowConfig

    Yields
    ------
    (feature_dict, t_start, t_end)
    """
    if cfg is None:
        cfg = WindowConfig()

    dt = 1.0 / cfg.imu_fs
    step_seconds = cfg.window_seconds * (1.0 - cfg.overlap)
    t_start = merged["timestamp"].iloc[0]
    t_end_max = merged["timestamp"].iloc[-1]

    while t_start + cfg.window_seconds <= t_end_max:
        t_end = t_start + cfg.window_seconds
        mask = (merged["timestamp"] >= t_start) & (merged["timestamp"] < t_end)
        window = merged[mask]

        if len(window) >= cfg.min_samples:
            feat = extract_window_features(window, dt)
            yield feat, t_start, t_end

        t_start += step_seconds


# ---------------------------------------------------------------------------
# Full trip → feature DataFrame
# ---------------------------------------------------------------------------

def trip_to_features(
    trip_dict: dict,
    cfg: WindowConfig | None = None,
) -> pd.DataFrame:
    """
    Convert a preprocessed trip dict (from preprocessing.preprocess_trip)
    into a DataFrame of windowed features with metadata.

    Parameters
    ----------
    trip_dict : dict from preprocessing.preprocess_trip
    cfg       : WindowConfig

    Returns
    -------
    DataFrame with FEATURE_COLUMNS + meta columns:
      trip_id, label, road, driver, t_start, t_end, window_mid
    """
    if cfg is None:
        cfg = WindowConfig()

    merged = trip_dict["merged"]
    rows = []

    for feat, t0, t1 in build_windows(merged, cfg):
        row = feat.copy()
        row["trip_id"]    = trip_dict.get("trip_id", "unknown")
        row["label"]      = trip_dict.get("label",   "UNKNOWN")
        row["road"]       = trip_dict.get("road",    "UNKNOWN")
        row["driver"]     = trip_dict.get("driver",  "UNKNOWN")
        row["t_start"]    = t0
        row["t_end"]      = t1
        row["window_mid"] = (t0 + t1) / 2.0
        rows.append(row)

    if not rows:
        logger.warning("No windows extracted from trip %s", trip_dict.get("trip_id"))
        return pd.DataFrame()

    return pd.DataFrame(rows)


def extract_all_features(
    trips: list[dict],
    cfg: WindowConfig | None = None,
) -> pd.DataFrame:
    """
    Extract features from a list of preprocessed trip dicts.

    Returns
    -------
    Concatenated feature DataFrame with one row per time window.
    """
    if cfg is None:
        cfg = WindowConfig()

    frames = []
    for trip in trips:
        df = trip_to_features(trip, cfg)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    logger.info(
        "Feature extraction complete: %d windows from %d trips",
        len(result), len(trips),
    )
    return result


# ---------------------------------------------------------------------------
# Sequence grouping (preserving temporal order)
# ---------------------------------------------------------------------------

def group_into_sequences(
    feature_df: pd.DataFrame,
    groupby: str = "trip_id",
) -> tuple[list[np.ndarray], list[int], list[dict]]:
    """
    Group windowed feature rows into ordered sequences for HMM training.

    IMPORTANT: Windows within each trip are sorted by t_start to preserve
    temporal ordering. Sequences are NEVER shuffled.

    Parameters
    ----------
    feature_df : DataFrame from extract_all_features
    groupby    : column to group by ('trip_id' recommended)

    Returns
    -------
    sequences  : list of (N_i, D) arrays — one array per sequence
    lengths    : list of sequence lengths N_i
    meta       : list of dicts with trip metadata
    """
    sequences = []
    lengths   = []
    meta      = []

    for key, grp in feature_df.groupby(groupby):
        grp = grp.sort_values("t_start")
        obs = grp[FEATURE_COLUMNS].values.astype(np.float64)

        # Drop rows with NaN in any feature
        valid_mask = np.all(np.isfinite(obs), axis=1)
        obs = obs[valid_mask]

        if len(obs) < 3:
            logger.warning("Skipping short sequence '%s' (%d samples)", key, len(obs))
            continue

        sequences.append(obs)
        lengths.append(len(obs))

        first_row = grp.iloc[0]
        meta.append({
            "trip_id": key,
            "label":   first_row.get("label",  "UNKNOWN"),
            "road":    first_row.get("road",   "UNKNOWN"),
            "driver":  first_row.get("driver", "UNKNOWN"),
            "n_windows": len(obs),
        })

    logger.info(
        "Grouped %d sequences; total windows = %d",
        len(sequences), sum(lengths),
    )
    return sequences, lengths, meta

