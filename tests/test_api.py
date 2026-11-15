"""
tests/test_api.py — Unit Tests for FastAPI Telemetry & Inference Server
========================================================================
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_model_info():
    res = client.get("/model/info")
    assert res.status_code == 200
    data = res.json()
    assert "n_states" in data
    assert data["n_states"] == 4


def test_list_trips():
    res = client.get("/trips")
    assert res.status_code == 200
    data = res.json()
    assert "trips" in data
    assert len(data["trips"]) > 0


def test_predict_and_telemetry():
    # Pick first available trip
    trips_res = client.get("/trips")
    first_trip = trips_res.json()["trips"][0]
    trip_id = first_trip["trip_id"]

    # 1. Predict
    pred_res = client.post("/predict", json={"trip_id": trip_id})
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert "dominant_label" in pred_data
    assert "n_windows" in pred_data

    # 2. Telemetry
    tel_res = client.get(f"/telemetry/{trip_id}")
    assert tel_res.status_code == 200
    tel_data = tel_res.json()
    assert "timestamps" in tel_data
    assert "ax_filt" in tel_data
    assert "gps_route" in tel_data
    assert len(tel_data["timestamps"]) > 0


def test_metrics_all():
    res = client.get("/metrics/all")
    assert res.status_code == 200
    data = res.json()
    assert "viterbi_comparison" in data
    assert "classification_report" in data


def test_static_dashboard_html():
    res = client.get("/")
    assert res.status_code == 200
    assert "Driver Behaviour" in res.text
    assert "Safety Telemetry Monitor" in res.text

