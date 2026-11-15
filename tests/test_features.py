"""
test_features.py — Unit Tests for Feature Extraction
======================================================
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    WindowConfig,
    compute_jerk,
    extract_window_features,
    trip_to_features,
    FEATURE_COLUMNS,
)


@pytest.fixture
def sample_window():
    """A synthetic 3-second window at 10 Hz (30 rows)."""
    rng = np.random.default_rng(0)
    n = 30
    return pd.DataFrame({
        "timestamp": np.linspace(0.0, 2.9, n),
        "ax_filt":   rng.normal(0.0, 0.05, n),
        "ay_filt":   rng.normal(0.0, 0.03, n),
        "az_filt":   rng.normal(-1.0, 0.02, n),
        "speed_ms":  np.linspace(10.0, 12.0, n),
    })


@pytest.fixture
def sample_trip_dict(sample_window):
    """Minimal trip dict for trip_to_features."""
    # Extend to 100 rows so we get multiple windows
    rng = np.random.default_rng(1)
    n = 100
    merged = pd.DataFrame({
        "timestamp": np.linspace(0.0, 9.9, n),
        "ax_filt":   rng.normal(0.0, 0.05, n),
        "ay_filt":   rng.normal(0.0, 0.03, n),
        "az_filt":   rng.normal(-1.0, 0.02, n),
        "speed_ms":  np.linspace(10.0, 15.0, n),
        "latitude":  np.full(n, 40.5),
        "longitude": np.full(n, -3.4),
    })
    return {
        "merged":  merged,
        "events":  pd.DataFrame(),
        "lane_ev": pd.DataFrame(),
        "trip_id": "20151110175712-16km-D1-NORMAL1-SECONDARY",
        "label":   "NORMAL",
        "road":    "SECONDARY",
        "driver":  "D1",
    }


# ---------------------------------------------------------------------------

def test_compute_jerk_shape(sample_window):
    accel = sample_window["ax_filt"].values
    jerk  = compute_jerk(accel, dt=0.1)
    assert jerk.shape == accel.shape, "Jerk must have same length as acceleration"


def test_compute_jerk_first_zero(sample_window):
    accel = sample_window["ax_filt"].values
    jerk  = compute_jerk(accel, dt=0.1)
    assert jerk[0] == 0.0, "First jerk value must be 0 (no predecessor)"


def test_compute_jerk_finite():
    accel = np.array([0.0, 0.1, 0.2, 0.0, -0.1])
    jerk  = compute_jerk(accel, dt=0.1)
    assert np.all(np.isfinite(jerk)), "All jerk values must be finite"


def test_extract_window_features_keys(sample_window):
    feats = extract_window_features(sample_window, dt=0.1)
    for col in FEATURE_COLUMNS:
        assert col in feats, f"Feature '{col}' missing from extract_window_features output"


def test_extract_window_features_finite(sample_window):
    feats = extract_window_features(sample_window, dt=0.1)
    for name, val in feats.items():
        assert np.isfinite(val), f"Feature '{name}' is not finite: {val}"


def test_trip_to_features_returns_dataframe(sample_trip_dict):
    cfg = WindowConfig(window_seconds=3.0, overlap=0.5)
    df  = trip_to_features(sample_trip_dict, cfg)
    assert isinstance(df, pd.DataFrame), "trip_to_features must return a DataFrame"
    assert len(df) > 0, "Should extract at least one window from 10-second trip"


def test_trip_to_features_has_all_feature_columns(sample_trip_dict):
    df = trip_to_features(sample_trip_dict)
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Column '{col}' missing from feature DataFrame"


def test_trip_to_features_temporal_order(sample_trip_dict):
    df = trip_to_features(sample_trip_dict)
    t_starts = df["t_start"].values
    assert np.all(np.diff(t_starts) >= 0), "Windows must be in non-decreasing temporal order"


def test_trip_to_features_metadata(sample_trip_dict):
    df = trip_to_features(sample_trip_dict)
    assert df["label"].iloc[0] == "NORMAL"
    assert df["trip_id"].iloc[0] == sample_trip_dict["trip_id"]

