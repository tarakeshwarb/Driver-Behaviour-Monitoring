"""
api/main.py — FastAPI Inference & Telemetry Backend
====================================================
Serves REST endpoints and static files for the single-page dashboard:

  GET  /health              — service health check
  GET  /model/info          — model parameters and states
  GET  /trips               — list all trips with driver/behavior metadata
  POST /predict             — run HMM inference on a trip
  GET  /telemetry/{trip_id} — full synchronized sensor streams & GPS path
  GET  /events/{trip_id}    — detected driving events with locations
  GET  /metrics/all         — academic evaluation benchmarks (Viterbi, F1, Recall)
  GET  /                    — serves dashboard/index.html
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_trip
from src.features import extract_all_features, WindowConfig, compute_jerk
from src.sequences import build_sequences_from_features
from src.hmm_model import load_model, predict_trip
from src.event_detection import (
    detect_events_hmm,
    events_to_dataframe,
    DEFAULT_THRESHOLDS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-Powered Driver Behaviour Monitoring API",
    description="Sequential Driving Telemetry Ingestion, Gaussian HMM Inference, and Event Detection",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

MODEL_PATH = PROJECT_ROOT / "results" / "model.pkl"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "UAH-DriveSet"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
MODEL_CMP_DIR = PROJECT_ROOT / "results" / "model_comparison"

_model = None
_state_map = None
_trip_cache: dict[str, dict] = {}


def _load_model_if_needed():
    global _model, _state_map
    if _model is None:
        if MODEL_PATH.exists():
            _model, _state_map = load_model(MODEL_PATH)
            logger.info("Gaussian HMM loaded from %s", MODEL_PATH)
        else:
            logger.warning("Model file not found at %s", MODEL_PATH)


@app.on_event("startup")
async def startup_event():
    _load_model_if_needed()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    trip_id: str
    driver: Optional[str] = None


class PredictResponse(BaseModel):
    trip_id: str
    dominant_label: str
    n_windows: int
    log_prob: float
    n_events: int


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
@app.get("/healthz")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model/info")
def model_info():
    _load_model_if_needed()
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "n_states": _model.n_components,
        "covariance_type": _model.covariance_type,
        "n_features": _model.means_.shape[1],
        "state_map": _state_map,
        "model_path": str(MODEL_PATH),
    }


@app.get("/trips")
def list_trips():
    """List all available trips in the UAH-DriveSet with metadata."""
    trips = []
    if not DATASET_ROOT.exists():
        return {"trips": [], "count": 0}
    for driver_dir in sorted(DATASET_ROOT.iterdir()):
        if not driver_dir.is_dir() or not driver_dir.name.startswith("D"):
            continue
        for trip_dir in sorted(driver_dir.iterdir()):
            if not trip_dir.is_dir():
                continue
            imu_ok = (trip_dir / "RAW_ACCELEROMETERS.txt").exists()
            gps_ok = (trip_dir / "RAW_GPS.txt").exists()
            if imu_ok and gps_ok:
                parts = trip_dir.name.split("-")
                label = parts[3] if len(parts) > 3 else "UNKNOWN"
                label = label.replace("NORMAL1", "NORMAL").replace("NORMAL2", "NORMAL")
                road = parts[4] if len(parts) > 4 else "UNKNOWN"
                trips.append({
                    "trip_id": trip_dir.name,
                    "driver": driver_dir.name,
                    "label": label,
                    "road": road,
                })
    return {"trips": trips, "count": len(trips)}


def _resolve_trip_dir(trip_id: str, driver: Optional[str] = None) -> tuple[Path, str]:
    if driver:
        candidate = DATASET_ROOT / driver / trip_id
        if candidate.exists():
            return candidate, driver
    for d in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        candidate = DATASET_ROOT / d / trip_id
        if candidate.exists():
            return candidate, d
    raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found.")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Run sequential HMM inference and event detection on a trip."""
    _load_model_if_needed()
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")

    trip_dir, driver = _resolve_trip_dir(request.trip_id, request.driver)
    trip = preprocess_trip(trip_dir)
    trip["driver"] = driver

    cfg = WindowConfig(window_seconds=3.0, overlap=0.5)
    feat_df = extract_all_features([trip], cfg)
    seqs, _, _ = build_sequences_from_features(feat_df)
    if not seqs:
        raise HTTPException(status_code=422, detail="Failed to construct observation sequences.")

    obs = seqs[0]
    result = predict_trip(_model, obs, _state_map, use_custom_viterbi=True)
    events = detect_events_hmm(feat_df, trip["merged"], result["state_seq"], _state_map)
    events_df = events_to_dataframe(events)

    _trip_cache[request.trip_id] = {
        "result": result,
        "events_df": events_df,
        "feat_df": feat_df,
        "merged": trip["merged"],
        "trip_id": request.trip_id,
        "driver": driver,
        "label": trip["label"],
        "road": trip["road"],
    }

    return PredictResponse(
        trip_id=request.trip_id,
        dominant_label=result["dominant_label"],
        n_windows=len(obs),
        log_prob=float(result["log_prob"]),
        n_events=len(events_df),
    )


@app.get("/telemetry/{trip_id}")
def get_telemetry(trip_id: str):
    """
    Return full time-series telemetry, GPS trajectory, and decoded HMM state intervals.
    """
    _load_model_if_needed()
    if trip_id not in _trip_cache:
        # Run prediction first if not cached
        predict(PredictRequest(trip_id=trip_id))

    cached = _trip_cache[trip_id]
    merged = cached["merged"]
    result = cached["result"]
    events_df = cached["events_df"]
    feat_df = cached["feat_df"]

    # Downsample slightly if extremely long to maintain 60 FPS in browser
    step = 2 if len(merged) > 3000 else 1
    sub = merged.iloc[::step].copy()

    dt = 0.1 * step
    jerk_long = compute_jerk(sub["ax_filt"].values, dt=dt)
    jerk_lat = compute_jerk(sub["ay_filt"].values, dt=dt)

    gps_points = []
    if "latitude" in sub.columns and sub["latitude"].notna().any():
        for lat, lon in zip(sub["latitude"], sub["longitude"]):
            if np.isfinite(lat) and np.isfinite(lon):
                gps_points.append([float(lat), float(lon)])

    # Construct window intervals for state overlay
    state_intervals = []
    lbl_seq = result["label_seq"]
    state_seq = result["state_seq"]
    t_starts = feat_df["t_start"].values
    t_ends = feat_df["t_end"].values

    for i in range(min(len(t_starts), len(lbl_seq))):
        state_intervals.append({
            "t_start": round(float(t_starts[i]), 2),
            "t_end": round(float(t_ends[i]), 2),
            "state_idx": int(state_seq[i]),
            "label": str(lbl_seq[i]),
        })

    events_list = events_df.to_dict(orient="records") if not events_df.empty else []

    duration_s = float(merged["timestamp"].max() - merged["timestamp"].min()) if not merged.empty else 0.0

    return {
        "trip_id": trip_id,
        "driver": cached["driver"],
        "ground_truth_label": cached["label"],
        "dominant_label": result["dominant_label"],
        "log_prob": float(result["log_prob"]),
        "duration_s": round(duration_s, 1),
        "timestamps": sub["timestamp"].round(2).tolist(),
        "ax_filt": sub["ax_filt"].round(4).tolist(),
        "ay_filt": sub["ay_filt"].round(4).tolist(),
        "az_filt": sub["az_filt"].round(4).tolist(),
        "speed_kmh": sub["speed_kmh"].round(1).tolist() if "speed_kmh" in sub.columns else [],
        "speed_ms": sub["speed_ms"].round(2).tolist() if "speed_ms" in sub.columns else [],
        "jerk_long": [round(float(v), 4) for v in jerk_long],
        "jerk_lat": [round(float(v), 4) for v in jerk_lat],
        "gps_route": gps_points,
        "state_intervals": state_intervals,
        "events": events_list,
    }


@app.get("/events/{trip_id}")
def get_events(trip_id: str):
    """Retrieve detected events for a trip."""
    if trip_id not in _trip_cache:
        predict(PredictRequest(trip_id=trip_id))
    df = _trip_cache[trip_id]["events_df"]
    return {"trip_id": trip_id, "events": df.to_dict(orient="records") if not df.empty else []}


@app.get("/metrics/all")
def get_all_metrics():
    """Retrieve precomputed experimental evaluation results and benchmarks."""
    response = {}

    viterbi_csv = MODEL_CMP_DIR / "viterbi_comparison.csv"
    if viterbi_csv.exists():
        response["viterbi_comparison"] = pd.read_csv(viterbi_csv).to_dict(orient="records")

    report_csv = METRICS_DIR / "classification_report.csv"
    if report_csv.exists():
        df_rep = pd.read_csv(report_csv)
        response["classification_report"] = df_rep.to_dict(orient="records")

    event_json = METRICS_DIR / "event_metrics.json"
    if event_json.exists():
        with open(event_json) as f:
            response["event_metrics"] = json.load(f)

    baseline_json = METRICS_DIR / "baseline_comparison.json"
    if baseline_json.exists():
        with open(baseline_json) as f:
            response["baseline_comparison"] = json.load(f)

    sweep_csv = METRICS_DIR / "hyperparameter_sweep.csv"
    if sweep_csv.exists():
        response["hyperparameter_sweep"] = pd.read_csv(sweep_csv).to_dict(orient="records")

    return response


# ---------------------------------------------------------------------------
# Mount Static Dashboard at root "/"
# ---------------------------------------------------------------------------
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="static")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
