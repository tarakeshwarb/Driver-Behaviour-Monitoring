"""
data_loader.py — High-level Dataset Loading Utilities
======================================================
Convenience wrappers that orchestrate preprocessing.load_all_trips
and add dataset-level statistics reporting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import load_all_trips as _load_all_trips

logger = logging.getLogger(__name__)

DATASET_ROOT_DEFAULT = Path("data/raw/UAH-DriveSet")


def load_dataset(
    dataset_root: str | Path | None = None,
    drivers: list[str] | None = None,
) -> list[dict]:
    """
    Load and preprocess all UAH-DriveSet trips.

    Parameters
    ----------
    dataset_root : path to .../UAH-DriveSet/ folder
    drivers      : e.g. ['D1','D2']; None loads all six

    Returns
    -------
    List of preprocessed trip dicts.
    """
    if dataset_root is None:
        dataset_root = DATASET_ROOT_DEFAULT
    trips = _load_all_trips(dataset_root, drivers)
    logger.info("Loaded %d trips from %s", len(trips), dataset_root)
    return trips


def dataset_summary(trips: list[dict]) -> pd.DataFrame:
    """
    Generate a summary table of the loaded dataset.

    Returns
    -------
    DataFrame with one row per trip: trip_id, driver, label, road,
    n_imu_samples, n_gps_samples, duration_s, n_events
    """
    rows = []
    for t in trips:
        merged = t["merged"]
        events = t["events"]
        rows.append({
            "trip_id":       t["trip_id"],
            "driver":        t.get("driver", "?"),
            "label":         t["label"],
            "road":          t["road"],
            "n_imu_samples": len(merged),
            "duration_s":    float(merged["timestamp"].max() - merged["timestamp"].min())
                             if not merged.empty else 0.0,
            "n_ref_events":  len(events),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "trip_id", "driver", "label", "road",
            "n_imu_samples", "duration_s", "n_ref_events"
        ])
    return pd.DataFrame(rows)


def label_distribution(trips: list[dict]) -> pd.DataFrame:
    """Return counts and percentages per driving-behaviour label."""
    if not trips:
        return pd.DataFrame(columns=["count", "percent"])
    labels = [t["label"] for t in trips]
    counts = pd.Series(labels).value_counts()
    pcts   = (counts / len(labels) * 100).round(1)
    return pd.DataFrame({"count": counts, "percent": pcts})


def driver_summary(trips: list[dict]) -> pd.DataFrame:
    """Return per-driver trip counts and label breakdown."""
    if not trips:
        return pd.DataFrame()
    rows = []
    for t in trips:
        rows.append({"driver": t.get("driver", "?"), "label": t["label"]})
    df = pd.DataFrame(rows)
    return df.groupby(["driver", "label"]).size().unstack(fill_value=0)
