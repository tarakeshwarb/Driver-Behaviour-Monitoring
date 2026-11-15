"""
preprocessing.py — UAH-DriveSet Data Preprocessing Module
===========================================================
Handles:
  - Loading RAW_ACCELEROMETERS.txt and RAW_GPS.txt
  - Timestamp-based sensor synchronisation (GPS ~1 Hz, IMU ~10 Hz)
  - Missing-value detection and handling
  - Butterworth low-pass noise filtering
  - Robust z-score normalisation (fitted on train split only)
  - Outlier detection via IQR fences
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column schemas (from UAH-DriveSet README)
# ---------------------------------------------------------------------------

ACCEL_COLS = [
    "timestamp",   # seconds since route start
    "status",      # 1 = valid
    "ax_raw",      # longitudinal accel (g) — raw
    "ay_raw",      # lateral accel (g) — raw
    "az_raw",      # vertical accel (g) — raw
    "ax_filt",     # longitudinal accel (g) — filtered
    "ay_filt",     # lateral accel (g) — filtered
    "az_filt",     # vertical accel (g) — filtered
    "ax_grav",     # gravity-removed longitudinal
    "ay_grav",     # gravity-removed lateral
    "az_grav",     # gravity-removed vertical
]

GPS_COLS = [
    "timestamp",   # seconds since route start
    "speed_kmh",   # vehicle speed (km/h)
    "latitude",
    "longitude",
    "altitude_m",
    "num_satellites",
    "hdop",        # horizontal dilution of precision
    "bearing_deg",
    "accel_mss",   # longitudinal acceleration (m/s²) from GPS Doppler
    "lane_lateral",# lateral lane offset (experimental)
    "lane_change", # lane change flag
    "overtaking",  # overtaking flag
]


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

def load_accelerometers(path: str | Path) -> pd.DataFrame:
    """Load RAW_ACCELEROMETERS.txt into a tidy DataFrame."""
    df = pd.read_csv(
        path, sep=r"\s+", header=None, names=ACCEL_COLS,
        na_values=["nan", "NaN", "inf", "-inf"]
    )
    df = df[df["status"] == 1].drop(columns=["status"]).reset_index(drop=True)
    logger.debug("Loaded %d IMU rows from %s", len(df), path)
    return df


def load_gps(path: str | Path) -> pd.DataFrame:
    """Load RAW_GPS.txt into a tidy DataFrame."""
    df = pd.read_csv(
        path, sep=r"\s+", header=None, names=GPS_COLS,
        na_values=["nan", "NaN", "inf", "-inf"]
    )
    logger.debug("Loaded %d GPS rows from %s", len(df), path)
    return df


def load_events_inertial(path: str | Path) -> pd.DataFrame:
    """
    Load EVENTS_INERTIAL.txt.

    Format: timestamp  event_type  severity  lat  lon  datetime
      event_type: 2=acceleration, -2=braking, 1=turning-left,
                  -1=turning-right (inferred from EVENTS_LIST docs)
    """
    cols = ["timestamp", "event_type", "severity", "latitude", "longitude", "datetime"]
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
        df["event_name"] = df["event_type"].map({
            2: "sudden_acceleration",
            -2: "hard_braking",
            1:  "sharp_turn_left",
            -1: "sharp_turn_right",
        }).fillna("unknown")
        return df
    except Exception:
        return pd.DataFrame(columns=cols + ["event_name"])


def load_events_lane_changes(path: str | Path) -> pd.DataFrame:
    """Load EVENTS_LIST_LANE_CHANGES.txt."""
    cols = ["timestamp", "direction", "lat", "lon", "confidence", "threshold"]
    try:
        return pd.read_csv(path, sep=r"\s+", header=None, names=cols)
    except Exception:
        return pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# Synchronisation
# ---------------------------------------------------------------------------

def synchronise_sensors(
    imu: pd.DataFrame,
    gps: pd.DataFrame,
    method: str = "nearest",
) -> pd.DataFrame:
    """
    Merge IMU (~10 Hz) and GPS (~1 Hz) on timestamp using
    nearest-neighbour interpolation.

    Strategy
    --------
    1. Sort both DataFrames by timestamp.
    2. Interpolate GPS columns onto IMU timestamps (linear interpolation
       for numeric columns; nearest for categorical flags).
    3. Return one row per IMU sample with GPS values attached.

    Parameters
    ----------
    imu     : DataFrame with 'timestamp' column (10 Hz)
    gps     : DataFrame with 'timestamp' column (1 Hz)
    method  : 'nearest' or 'linear'

    Returns
    -------
    Merged DataFrame indexed by IMU timestamps.
    """
    imu = imu.sort_values("timestamp").reset_index(drop=True)
    gps = gps.sort_values("timestamp").reset_index(drop=True)

    gps_interp = {}
    flag_cols = {"lane_change", "overtaking"}

    numeric_gps = [c for c in gps.columns if c != "timestamp"]

    for col in numeric_gps:
        if col in flag_cols:
            # Nearest-neighbour for binary flags
            idx = np.searchsorted(gps["timestamp"].values, imu["timestamp"].values, side="left")
            idx = np.clip(idx, 0, len(gps) - 1)
            gps_interp[col] = gps[col].values[idx]
        else:
            gps_interp[col] = np.interp(
                imu["timestamp"].values,
                gps["timestamp"].values,
                gps[col].values,
            )

    merged = imu.copy()
    for col, values in gps_interp.items():
        merged[col] = values

    logger.debug("Synchronised %d IMU rows with GPS data", len(merged))
    return merged


# ---------------------------------------------------------------------------
# Noise filtering
# ---------------------------------------------------------------------------

def butterworth_lowpass(
    data: np.ndarray,
    cutoff_hz: float = 2.0,
    fs_hz: float = 10.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    data      : 1-D signal array
    cutoff_hz : Filter cut-off frequency (Hz)
    fs_hz     : Sampling frequency (Hz)
    order     : Filter order

    Returns
    -------
    Filtered signal (same length as input).
    """
    nyq = 0.5 * fs_hz
    normal_cutoff = cutoff_hz / nyq
    normal_cutoff = min(normal_cutoff, 0.99)  # guard Nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    # pad if too short
    padlen = min(len(data) - 1, 3 * max(len(a), len(b)))
    if len(data) < 15:
        return data  # too short to filter
    return filtfilt(b, a, data, padlen=padlen)


def filter_imu_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    cutoff_hz: float = 2.0,
    fs_hz: float = 10.0,
) -> pd.DataFrame:
    """Apply Butterworth low-pass filter to specified IMU columns."""
    if columns is None:
        columns = ["ax_filt", "ay_filt", "az_filt"]
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = butterworth_lowpass(df[col].values, cutoff_hz, fs_hz)
    return df


# ---------------------------------------------------------------------------
# Missing values & outliers
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle NaN/Inf values:
      - Replace ±Inf with NaN.
      - Forward-fill then backward-fill gaps ≤ 5 consecutive rows.
      - Drop rows where critical columns remain NaN.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    # Forward/backward fill for short gaps
    df = df.ffill(limit=5).bfill(limit=5)
    critical = ["ax_filt", "ay_filt", "az_filt", "timestamp"]
    present_critical = [c for c in critical if c in df.columns]
    before = len(df)
    df = df.dropna(subset=present_critical).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        logger.warning("Dropped %d rows with NaN in critical columns", removed)
    return df


def clip_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str],
    k: float = 3.0,
) -> pd.DataFrame:
    """
    Clip outliers to [Q1 − k·IQR, Q3 + k·IQR] for listed columns.

    Parameters
    ----------
    df      : Input DataFrame
    columns : Column names to clip
    k       : Multiplier for IQR fence (default 3.0 = very conservative)
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        clipped = df[col].clip(lower, upper)
        n_outliers = (df[col] != clipped).sum()
        if n_outliers > 0:
            logger.debug("Clipped %d outliers in '%s'", n_outliers, col)
        df[col] = clipped
    return df


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class SensorNormaliser:
    """
    Wrapper around sklearn StandardScaler that records which columns
    were fitted, preventing data leakage from test sets.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns
        self.scaler = StandardScaler()
        self._fitted = False
        self._fitted_columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> "SensorNormaliser":
        cols = self.columns or [c for c in df.columns if df[c].dtype in [np.float64, np.float32]]
        self.scaler.fit(df[cols])
        self._fitted = True
        self._fitted_columns = cols
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("SensorNormaliser must be fitted before transform.")
        df = df.copy()
        df[self._fitted_columns] = self.scaler.transform(df[self._fitted_columns])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_trip(
    trip_dir: str | Path,
    normaliser: Optional[SensorNormaliser] = None,
    fit_normaliser: bool = False,
    cutoff_hz: float = 2.0,
    imu_fs: float = 10.0,
) -> dict:
    """
    Full preprocessing pipeline for a single trip directory.

    Returns
    -------
    dict with keys:
      'merged'   : synchronised + filtered + cleaned DataFrame
      'events'   : inertial event annotations DataFrame
      'lane_ev'  : lane-change event DataFrame
      'trip_id'  : trip folder name
      'label'    : driving behaviour label (NORMAL/AGGRESSIVE/DROWSY)
      'road'     : road type (MOTORWAY/SECONDARY)
    """
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    # Parse label and road from folder name: Date-Dist-Driver-Label-Road
    parts = trip_id.split("-")
    label = parts[3] if len(parts) > 3 else "UNKNOWN"
    # Normalise NORMAL1/NORMAL2 → NORMAL
    label = label.replace("NORMAL1", "NORMAL").replace("NORMAL2", "NORMAL")
    road  = parts[4] if len(parts) > 4 else "UNKNOWN"

    # Load raw files
    imu_path = trip_dir / "RAW_ACCELEROMETERS.txt"
    gps_path = trip_dir / "RAW_GPS.txt"

    imu = load_accelerometers(imu_path)
    gps = load_gps(gps_path)

    # Synchronise on timestamps
    merged = synchronise_sensors(imu, gps)

    # Noise filter IMU channels
    imu_cols_to_filter = ["ax_filt", "ay_filt", "az_filt",
                          "ax_raw", "ay_raw", "az_raw"]
    merged = filter_imu_columns(merged, imu_cols_to_filter, cutoff_hz, imu_fs)

    # Missing values
    merged = handle_missing_values(merged)

    # Outlier clipping (conservative k=3)
    clip_cols = ["ax_filt", "ay_filt", "az_filt", "speed_kmh", "accel_mss"]
    merged = clip_outliers_iqr(merged, [c for c in clip_cols if c in merged.columns])

    # Normalisation (only if requested / fitted externally)
    if normaliser is not None:
        if fit_normaliser:
            norm_cols = ["ax_filt", "ay_filt", "az_filt", "speed_kmh"]
            norm_cols = [c for c in norm_cols if c in merged.columns]
            normaliser.columns = norm_cols
            merged = normaliser.fit_transform(merged)
        else:
            merged = normaliser.transform(merged)

    # Metadata
    events   = load_events_inertial(trip_dir / "EVENTS_INERTIAL.txt")
    lane_ev  = load_events_lane_changes(trip_dir / "EVENTS_LIST_LANE_CHANGES.txt")

    # Speed in m/s for feature extraction
    if "speed_kmh" in merged.columns:
        merged["speed_ms"] = merged["speed_kmh"] / 3.6

    return {
        "merged":   merged,
        "events":   events,
        "lane_ev":  lane_ev,
        "trip_id":  trip_id,
        "label":    label,
        "road":     road,
    }


def load_all_trips(
    dataset_root: str | Path,
    drivers: list[str] | None = None,
) -> list[dict]:
    """
    Load and preprocess all trips from UAH-DriveSet.

    Parameters
    ----------
    dataset_root : Path to .../UAH-DriveSet/
    drivers      : Optional list of driver folders, e.g. ['D1','D2'].
                   If None, loads all D1–D6.

    Returns
    -------
    List of trip dicts (one per trip folder).
    """
    dataset_root = Path(dataset_root)
    if drivers is None:
        drivers = sorted([d.name for d in dataset_root.iterdir()
                          if d.is_dir() and d.name.startswith("D")])

    trips = []
    for driver in drivers:
        driver_dir = dataset_root / driver
        if not driver_dir.exists():
            logger.warning("Driver folder not found: %s", driver_dir)
            continue
        for trip_dir in sorted(driver_dir.iterdir()):
            if not trip_dir.is_dir():
                continue
            imu_file = trip_dir / "RAW_ACCELEROMETERS.txt"
            gps_file = trip_dir / "RAW_GPS.txt"
            if not imu_file.exists() or not gps_file.exists():
                continue
            try:
                trip = preprocess_trip(trip_dir)
                trip["driver"] = driver
                trips.append(trip)
                logger.info("Loaded trip: %s (%s/%s)", trip["trip_id"], trip["label"], trip["road"])
            except Exception as exc:
                logger.error("Failed to load %s: %s", trip_dir, exc)
    return trips
