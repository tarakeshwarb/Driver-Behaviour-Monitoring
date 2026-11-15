"""
event_detection.py — Driving Event Detection Module
=====================================================
Translates HMM hidden-state sequences + raw sensor features into
human-interpretable driving events with timestamps and locations.

Event types detected:
  - sudden_acceleration  : high positive longitudinal jerk / speed gain
  - hard_braking         : high negative longitudinal acceleration
  - sharp_turn           : high lateral acceleration magnitude
  - unstable_driving     : high combined variance in ax and ay
  - lane_change          : from GPS lane-change flag

Each detected event is matched against reference annotations from
EVENTS_INERTIAL.txt using a configurable temporal tolerance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds (tuned from UAH-DriveSet paper and sensor units)
# ---------------------------------------------------------------------------

@dataclass
class EventThresholds:
    """
    Sensor thresholds for event classification calibrated to UAH-DriveSet IMU data.

    Units:
      - Accelerations in g (1 g ≈ 9.81 m/s²)
      - Jerk in g/s
      - Speed in m/s
    """
    # Hard braking: strong negative longitudinal acceleration
    hard_braking_ax: float = -0.08           # g
    # Sudden acceleration: strong positive longitudinal value
    sudden_accel_ax: float = 0.045           # g
    # Sharp turn: lateral acceleration magnitude
    sharp_turn_ay:   float = 0.06            # g
    # Jerk threshold for sudden events
    jerk_threshold:  float = 0.04            # g/s
    # Variance threshold for unstable driving
    variance_threshold: float = 0.002        # g²
    # Temporal matching tolerance (seconds) for reference comparison
    event_match_tolerance_s: float = 10.0

    # Minimum consecutive windows to form an event
    min_consecutive_windows: int = 1


DEFAULT_THRESHOLDS = EventThresholds()


# ---------------------------------------------------------------------------
# Event data structure
# ---------------------------------------------------------------------------

@dataclass
class DrivingEvent:
    """A single detected driving event."""
    event_type:  str    # 'sudden_acceleration', 'hard_braking', etc.
    t_start:     float  # seconds since route start
    t_end:       float  # seconds since route start
    severity:    str    # 'low', 'medium', 'high'
    latitude:    float = 0.0
    longitude:   float = 0.0
    ax_peak:     float = 0.0  # peak longitudinal acceleration
    ay_peak:     float = 0.0  # peak lateral acceleration
    jerk_peak:   float = 0.0  # peak jerk value
    description: str   = ""
    matched_ref: bool  = False  # matched against reference annotation

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start

    @property
    def mid_time(self) -> float:
        return (self.t_start + self.t_end) / 2.0


# ---------------------------------------------------------------------------
# Threshold-based event detection (baseline)
# ---------------------------------------------------------------------------

def detect_events_threshold(
    feature_df: pd.DataFrame,
    merged_df:  pd.DataFrame,
    thresholds: EventThresholds | None = None,
) -> list[DrivingEvent]:
    """
    Detect driving events using fixed sensor thresholds.

    This is the baseline method used for comparison with HMM-based detection.

    Parameters
    ----------
    feature_df : windowed feature DataFrame (from features.py)
    merged_df  : raw merged sensor DataFrame (for GPS lookup)
    thresholds : EventThresholds (defaults to DEFAULT_THRESHOLDS)

    Returns
    -------
    List of DrivingEvent objects sorted by t_start.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    events: list[DrivingEvent] = []

    for _, row in feature_df.iterrows():
        t_start = row["t_start"]
        t_end   = row["t_end"]

        # Lookup GPS position at mid-window time
        lat, lon = _get_gps_at_time(merged_df, (t_start + t_end) / 2.0)

        ax_mean = row.get("ax_mean", 0.0)
        ay_mean = row.get("ay_mean", 0.0)
        ax_min  = row.get("ax_min",  0.0)
        ax_max  = row.get("ax_max",  0.0)
        ay_max  = row.get("ay_max",  0.0)
        jerk_long_max = row.get("jerk_long_max", 0.0)
        jerk_lat_max  = row.get("jerk_lat_max",  0.0)
        ax_var  = row.get("ax_variance", 0.0)
        ay_var  = row.get("ay_variance", 0.0)

        # --- Hard braking ---
        if ax_min < thresholds.hard_braking_ax:
            severity = _classify_severity(abs(ax_min), 0.08, 0.12, 0.18)
            events.append(DrivingEvent(
                event_type="hard_braking",
                t_start=t_start, t_end=t_end,
                severity=severity,
                latitude=lat, longitude=lon,
                ax_peak=ax_min,
                jerk_peak=jerk_long_max,
                description=f"Hard braking detected (ax_min={ax_min:.3f} g)",
            ))

        # --- Sudden acceleration ---
        elif ax_max > thresholds.sudden_accel_ax:
            severity = _classify_severity(ax_max, 0.045, 0.07, 0.12)
            events.append(DrivingEvent(
                event_type="sudden_acceleration",
                t_start=t_start, t_end=t_end,
                severity=severity,
                latitude=lat, longitude=lon,
                ax_peak=ax_max,
                jerk_peak=jerk_long_max,
                description=f"Sudden acceleration (ax_max={ax_max:.3f} g)",
            ))

        # --- Sharp turn ---
        if abs(ay_max) > thresholds.sharp_turn_ay:
            direction = "left" if ay_mean > 0 else "right"
            severity = _classify_severity(abs(ay_max), 0.06, 0.09, 0.14)
            events.append(DrivingEvent(
                event_type=f"sharp_turn_{direction}",
                t_start=t_start, t_end=t_end,
                severity=severity,
                latitude=lat, longitude=lon,
                ay_peak=ay_max,
                jerk_peak=jerk_lat_max,
                description=f"Sharp {direction} turn (ay_max={ay_max:.3f} g)",
            ))

        # --- Unstable driving ---
        if ax_var > thresholds.variance_threshold and ay_var > thresholds.variance_threshold:
            severity = _classify_severity(
                (ax_var + ay_var) / 2, thresholds.variance_threshold,
                thresholds.variance_threshold * 2, thresholds.variance_threshold * 4
            )
            events.append(DrivingEvent(
                event_type="unstable_driving",
                t_start=t_start, t_end=t_end,
                severity=severity,
                latitude=lat, longitude=lon,
                description=f"Unstable driving (ax_var={ax_var:.4f}, ay_var={ay_var:.4f})",
            ))

    events.sort(key=lambda e: e.t_start)
    return events


# ---------------------------------------------------------------------------
# HMM-based event detection
# ---------------------------------------------------------------------------

def detect_events_hmm(
    feature_df:    pd.DataFrame,
    merged_df:     pd.DataFrame,
    state_seq:     np.ndarray,
    state_to_label: dict[int, str],
    thresholds:    EventThresholds | None = None,
) -> list[DrivingEvent]:
    """
    Detect driving events using HMM state sequence + sensor thresholds.

    An event is flagged when:
      (a) The HMM assigns a non-NORMAL state, AND
      (b) Sensor features exceed safety thresholds.

    This is a combined approach: HMM provides temporal context;
    thresholds provide physical interpretability.

    Parameters
    ----------
    feature_df    : windowed feature DataFrame
    merged_df     : raw merged sensor DataFrame
    state_seq     : (T,) HMM state indices for each window
    state_to_label: mapping from state index to label string
    thresholds    : EventThresholds

    Returns
    -------
    List of DrivingEvent objects.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # First, get threshold-based detections as candidates
    candidates = detect_events_threshold(feature_df, merged_df, thresholds)

    # Map each candidate to its HMM state label
    if len(state_seq) == 0 or len(feature_df) == 0:
        return candidates

    # Build time → state lookup
    times   = feature_df["t_start"].values
    n_times = min(len(times), len(state_seq))
    time_state = {times[i]: state_to_label.get(state_seq[i], "NORMAL")
                  for i in range(n_times)}

    confirmed: list[DrivingEvent] = []
    for ev in candidates:
        # Find the state label closest to event start time
        if times[:n_times].size > 0:
            idx = int(np.argmin(np.abs(times[:n_times] - ev.t_start)))
            hmm_label = state_to_label.get(state_seq[idx], "NORMAL")
        else:
            hmm_label = "NORMAL"

        # Confirm events that agree with HMM or are very high severity
        if hmm_label != "NORMAL" or ev.severity == "high":
            ev.description += f" [HMM: {hmm_label}]"
            confirmed.append(ev)

    confirmed.sort(key=lambda e: e.t_start)
    return confirmed


# ---------------------------------------------------------------------------
# Reference annotation matching
# ---------------------------------------------------------------------------

def match_events_to_reference(
    detected:   list[DrivingEvent],
    reference:  pd.DataFrame,
    tolerance_s: float = 5.0,
) -> tuple[list[DrivingEvent], float]:
    """
    Match detected events against reference EVENTS_INERTIAL annotations.

    A detection is counted as a true positive if there is a reference event
    of the same broad type within ``tolerance_s`` seconds.

    Parameters
    ----------
    detected    : list of detected DrivingEvent objects
    reference   : DataFrame from preprocessing.load_events_inertial
    tolerance_s : temporal tolerance for matching (seconds)

    Returns
    -------
    (matched_events, event_recall)
      matched_events : detected events with matched_ref flag updated
      event_recall   : fraction of reference events detected
    """
    if reference.empty or not detected:
        return detected, 0.0

    # Group detected events by broad type
    type_map = {
        "sudden_acceleration": [2],
        "hard_braking":        [-2],
        "sharp_turn_left":     [1],
        "sharp_turn_right":    [-1],
    }

    ref_matched = set()  # indices of matched reference events
    for ev in detected:
        ref_types = type_map.get(ev.event_type, [])
        ref_subset = reference[reference["event_type"].isin(ref_types)]
        for ref_idx, ref_row in ref_subset.iterrows():
            if ref_idx in ref_matched:
                continue
            if abs(ev.mid_time - ref_row["timestamp"]) <= tolerance_s:
                ev.matched_ref = True
                ref_matched.add(ref_idx)
                break

    n_ref = len(reference)
    recall = len(ref_matched) / n_ref if n_ref > 0 else 0.0
    return detected, recall


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _get_gps_at_time(merged_df: pd.DataFrame, t: float) -> tuple[float, float]:
    """Return (lat, lon) from merged DataFrame nearest to timestamp t."""
    if merged_df.empty or "latitude" not in merged_df.columns:
        return 0.0, 0.0
    idx = int(np.argmin(np.abs(merged_df["timestamp"].values - t)))
    return float(merged_df["latitude"].iloc[idx]), float(merged_df["longitude"].iloc[idx])


def _classify_severity(value: float, low: float, medium: float, high: float) -> str:
    """Classify a numeric magnitude into low/medium/high."""
    if value >= high:
        return "high"
    elif value >= medium:
        return "medium"
    return "low"


def events_to_dataframe(events: list[DrivingEvent]) -> pd.DataFrame:
    """Convert a list of DrivingEvent objects to a tidy DataFrame."""
    if not events:
        return pd.DataFrame()
    return pd.DataFrame([{
        "event_type":  e.event_type,
        "t_start":     e.t_start,
        "t_end":       e.t_end,
        "duration_s":  e.duration_s,
        "severity":    e.severity,
        "latitude":    e.latitude,
        "longitude":   e.longitude,
        "ax_peak":     e.ax_peak,
        "ay_peak":     e.ay_peak,
        "jerk_peak":   e.jerk_peak,
        "matched_ref": e.matched_ref,
        "description": e.description,
    } for e in events])
