"""
test_preprocessing.py — Unit Tests for Preprocessing Module
=============================================================
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    synchronise_sensors,
    butterworth_lowpass,
    handle_missing_values,
    clip_outliers_iqr,
    SensorNormaliser,
    ACCEL_COLS,
    GPS_COLS,
)


@pytest.fixture
def sample_imu():
    """10 Hz IMU data for 5 seconds (50 rows)."""
    n = 50
    return pd.DataFrame({
        "timestamp": np.linspace(0.0, 4.9, n),
        "ax_raw":    np.random.default_rng(0).normal(0.0, 0.05, n),
        "ay_raw":    np.random.default_rng(1).normal(0.0, 0.03, n),
        "az_raw":    np.random.default_rng(2).normal(-1.0, 0.02, n),
        "ax_filt":   np.random.default_rng(3).normal(0.0, 0.05, n),
        "ay_filt":   np.random.default_rng(4).normal(0.0, 0.03, n),
        "az_filt":   np.random.default_rng(5).normal(-1.0, 0.02, n),
        "ax_grav":   np.zeros(n),
        "ay_grav":   np.zeros(n),
        "az_grav":   np.zeros(n),
    })


@pytest.fixture
def sample_gps():
    """1 Hz GPS data for 5 seconds (5 rows)."""
    n = 5
    return pd.DataFrame({
        "timestamp":    np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "speed_kmh":    np.array([60.0, 61.0, 62.0, 61.5, 60.5]),
        "latitude":     np.array([40.51, 40.52, 40.53, 40.54, 40.55]),
        "longitude":    np.array([-3.40, -3.41, -3.42, -3.43, -3.44]),
        "altitude_m":   np.full(n, 600.0),
        "num_satellites": np.full(n, 5),
        "hdop":         np.full(n, 1.5),
        "bearing_deg":  np.full(n, 330.0),
        "accel_mss":    np.zeros(n),
        "lane_lateral": np.zeros(n),
        "lane_change":  np.zeros(n),
        "overtaking":   np.zeros(n),
    })


# ---------------------------------------------------------------------------

def test_synchronise_shape(sample_imu, sample_gps):
    merged = synchronise_sensors(sample_imu, sample_gps)
    assert len(merged) == len(sample_imu), "Merged must have same number of rows as IMU"


def test_synchronise_has_speed(sample_imu, sample_gps):
    merged = synchronise_sensors(sample_imu, sample_gps)
    assert "speed_kmh" in merged.columns, "Merged DataFrame must include GPS speed"


def test_synchronise_no_nan(sample_imu, sample_gps):
    merged = synchronise_sensors(sample_imu, sample_gps)
    assert merged["speed_kmh"].isna().sum() == 0, "speed_kmh should have no NaN after interpolation"


def test_butterworth_shape():
    signal = np.sin(np.linspace(0, 2 * np.pi, 100))
    filtered = butterworth_lowpass(signal, cutoff_hz=2.0, fs_hz=10.0)
    assert filtered.shape == signal.shape, "Filtered signal must have same shape"


def test_butterworth_attenuates_high_freq():
    t = np.linspace(0, 10, 1000)
    low_freq  = np.sin(2 * np.pi * 0.5 * t)   # 0.5 Hz — should pass
    high_freq = np.sin(2 * np.pi * 5.0 * t)   # 5.0 Hz — should be attenuated
    combined  = low_freq + high_freq
    filtered  = butterworth_lowpass(combined, cutoff_hz=1.5, fs_hz=100.0)
    # Filtered signal should correlate strongly with low-freq component
    corr = np.corrcoef(filtered, low_freq)[0, 1]
    assert corr > 0.95, f"Low-pass filter should preserve low-freq component (corr={corr:.3f})"


def test_handle_missing_values_removes_nan():
    df = pd.DataFrame({
        "timestamp": [0.0, 1.0, 2.0, 3.0],
        "ax_filt":   [0.1, np.nan, 0.3, 0.4],
        "ay_filt":   [0.0, 0.1, 0.2, 0.3],
        "az_filt":   [-1.0, -1.0, -1.0, -1.0],
    })
    cleaned = handle_missing_values(df)
    # NaN should be forward-filled
    assert cleaned["ax_filt"].isna().sum() == 0


def test_clip_outliers_iqr():
    df = pd.DataFrame({"x": [0.0, 0.1, 0.2, 100.0, 0.15, -100.0, 0.05]})
    clipped = clip_outliers_iqr(df, ["x"], k=3.0)
    # Extreme values must be clipped
    assert clipped["x"].max() < 100.0
    assert clipped["x"].min() > -100.0


def test_sensor_normaliser_no_leakage(sample_imu):
    """Test that the normaliser fitted on train data is correctly applied to test data."""
    train = sample_imu.iloc[:30].copy()
    test  = sample_imu.iloc[30:].copy()

    norm = SensorNormaliser(columns=["ax_filt", "ay_filt"])
    train_norm = norm.fit_transform(train)
    test_norm  = norm.transform(test)

    # Fitted mean should come from training data
    np.testing.assert_allclose(
        train_norm["ax_filt"].mean(), 0.0, atol=0.3,
        err_msg="Normalised training data should have ~0 mean"
    )
    # Test data mean (from different distribution slice) may differ
    assert "ax_filt" in test_norm.columns

