"""
scripts/run_experiments.py — End-to-End ML Experiment Runner
=============================================================
Runs the complete research pipeline:
  1. Load all UAH-DriveSet trips
  2. Dataset statistics & summary
  3. Preprocessing & feature extraction
  4. Chronological sequence splitting & leakage-free normalization
  5. HMM training & state mapping
  6. Custom Viterbi vs hmmlearn comparison
  7. Event detection & reference matching
  8. Model evaluation (Accuracy, Macro F1, Confusion Matrix, Event Recall)
  9. Hyperparameter sensitivity sweep (K=3, 4, 5)
  10. Generation of all publication-quality figures and metric tables
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_all_trips
from src.features import extract_all_features, WindowConfig, FEATURE_COLUMNS
from src.sequences import (
    build_sequences_from_features,
    SequenceNormaliser,
    split_sequences_chronologically,
)
from src.hmm_model import (
    train_gaussian_hmm,
    build_state_label_map,
    predict_trip,
    save_model,
    sweep_n_states,
)
from src.custom_viterbi import GaussianHMMParams, compare_with_reference
from src.event_detection import (
    detect_events_hmm,
    detect_events_threshold,
    events_to_dataframe,
    match_events_to_reference,
)
from src.evaluation import (
    evaluate_classification,
    compute_event_recall,
    evaluate_viterbi_comparison,
    evaluate_hmm_vs_threshold,
)
from src.data_loader import dataset_summary, label_distribution, driver_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_experiments")

# Paths
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "UAH-DriveSet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODEL_CMP_DIR = RESULTS_DIR / "model_comparison"
REPORT_DIR = PROJECT_ROOT / "report"

for d in [PROCESSED_DIR, FIG_DIR, METRICS_DIR, MODEL_CMP_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def main():
    logger.info("=== Phase 1: Loading Dataset ===")
    trips = load_all_trips(DATASET_ROOT)
    logger.info("Loaded %d trips.", len(trips))

    summary_df = dataset_summary(trips)
    summary_df.to_csv(METRICS_DIR / "dataset_summary.csv", index=False)
    
    label_df = label_distribution(trips)
    label_df.to_csv(METRICS_DIR / "label_distribution.csv")

    drv_df = driver_summary(trips)
    drv_df.to_csv(METRICS_DIR / "driver_summary.csv")

    # Figure 1: Label distribution
    plt.figure(figsize=(7, 4))
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = plt.bar(label_df.index, label_df["count"], color=colors, edgecolor="black", alpha=0.85)
    plt.title("UAH-DriveSet Driving Behavior Distribution", fontsize=12, fontweight="bold")
    plt.ylabel("Trip Count")
    plt.grid(axis="y", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.3, f"{int(h)}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dataset_label_distribution.png", dpi=300)
    plt.close()

    logger.info("=== Phase 2 & 3: Feature Extraction ===")
    cfg = WindowConfig(window_seconds=3.0, overlap=0.5)
    feat_df = extract_all_features(trips, cfg)
    feat_df.to_csv(PROCESSED_DIR / "features_all.csv", index=False)
    logger.info("Extracted %d feature windows.", len(feat_df))

    # Figure 2: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = feat_df[FEATURE_COLUMNS].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, cbar=True)
    plt.title("Sensor Feature Correlation Heatmap (3s Windows)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_correlation_heatmap.png", dpi=300)
    plt.close()

    logger.info("=== Phase 4: Sequence Construction & Splitting ===")
    sequences, lengths, meta = build_sequences_from_features(feat_df)
    (train_seqs, val_seqs, test_seqs,
     train_meta, val_meta, test_meta) = split_sequences_chronologically(sequences, meta, train_frac=0.70, val_frac=0.15, gap=1)

    logger.info("Chronological split: %d train, %d val, %d test sequences", len(train_seqs), len(val_seqs), len(test_seqs))

    # Leakage-free normalization: fit on train only!
    normaliser = SequenceNormaliser()
    train_seqs_norm = normaliser.fit_transform(train_seqs)
    val_seqs_norm = normaliser.transform(val_seqs)
    test_seqs_norm = normaliser.transform(test_seqs)

    logger.info("=== Phase 5: HMM Training & Hyperparameter Sweep ===")
    val_labels = [m["label"] for m in val_meta]
    sweep_df = sweep_n_states(train_seqs_norm, val_seqs_norm, val_labels, n_states_list=[3, 4, 5], covariance_type="diag")
    sweep_df.to_csv(METRICS_DIR / "hyperparameter_sweep.csv", index=False)

    # Figure 3: Hyperparameter Sweep
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(sweep_df["n_states"], sweep_df["accuracy"] * 100, "o-", color="#2980b9", lw=2)
    axes[0].set_title("Validation Accuracy (%)", fontweight="bold")
    axes[0].set_xlabel("Number of States (K)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(sweep_df["n_states"], sweep_df["aic"], "s-", color="#e74c3c", lw=2)
    axes[1].set_title("Akaike Information Criterion (AIC)", fontweight="bold")
    axes[1].set_xlabel("Number of States (K)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(sweep_df["n_states"], sweep_df["bic"], "^-", color="#27ae60", lw=2)
    axes[2].set_title("Bayesian Information Criterion (BIC)", fontweight="bold")
    axes[2].set_xlabel("Number of States (K)")
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hyperparameter_sweep.png", dpi=300)
    plt.close()

    # Train final model with K=4 (balanced between complexity and expressiveness)
    chosen_k = 4
    model = train_gaussian_hmm(train_seqs_norm, n_states=chosen_k, covariance_type="diag", n_iter=100, random_state=42)
    train_labels = [m["label"] for m in train_meta]
    state_map = build_state_label_map(model, train_seqs_norm, train_labels)
    save_model(model, state_map, RESULTS_DIR / "model.pkl")

    logger.info("=== Phase 6: Custom Viterbi vs hmmlearn Comparison ===")
    viterbi_cmp_df = evaluate_viterbi_comparison(model, test_seqs_norm, n_sequences=len(test_seqs_norm))
    viterbi_cmp_df.to_csv(MODEL_CMP_DIR / "viterbi_comparison.csv", index=False)
    logger.info("Custom Viterbi Mean Agreement: %.2f%%", viterbi_cmp_df["sequence_agreement"].mean() * 100)

    # Figure 4: Viterbi Timing & Agreement
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(viterbi_cmp_df))
    w = 0.35
    axes[0].bar(x - w/2, viterbi_cmp_df["custom_time_ms"], w, label="Custom Viterbi", color="#3498db")
    axes[0].bar(x + w/2, viterbi_cmp_df["ref_time_ms"], w, label="hmmlearn decode()", color="#95a5a6")
    axes[0].set_ylabel("Execution Time (ms)")
    axes[0].set_title("Inference Latency Comparison", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Trip {i+1}" for i in x])
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, viterbi_cmp_df["sequence_agreement"] * 100, color="#2ecc71", edgecolor="black", alpha=0.85)
    axes[1].set_ylabel("Agreement (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Sequence Reconstruction Agreement", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Trip {i+1}" for i in x])
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(viterbi_cmp_df["sequence_agreement"] * 100):
        axes[1].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "viterbi_timing_comparison.png", dpi=300)
    plt.close()

    logger.info("=== Phase 7: Evaluation & Event Detection ===")
    test_labels = [m["label"] for m in test_meta]
    hmm_preds = []
    hmm_results = []
    all_detected_events = []
    all_ref_events = []

    for obs, m in zip(test_seqs_norm, test_meta):
        res = predict_trip(model, obs, state_map, use_custom_viterbi=True)
        hmm_preds.append(res["dominant_label"])
        hmm_results.append(res)

        # Retrieve trip for GPS & reference events
        t_obj = next(t for t in trips if t["trip_id"] == m["trip_id"])
        trip_feat = feat_df[feat_df["trip_id"] == m["trip_id"]]
        events = detect_events_hmm(trip_feat, t_obj["merged"], res["state_seq"], state_map)
        ev_df = events_to_dataframe(events)
        if not ev_df.empty:
            ev_df["trip_id"] = m["trip_id"]
            all_detected_events.append(ev_df)

        ref_ev = t_obj["events"]
        if not ref_ev.empty:
            ref_ev = ref_ev.copy()
            ref_ev["trip_id"] = m["trip_id"]
            all_ref_events.append(ref_ev)

    eval_dict = evaluate_classification(test_labels, hmm_preds)
    eval_dict["report"].to_csv(METRICS_DIR / "classification_report.csv")

    # Figure 5: Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = eval_dict["confusion_matrix"]
    labels = eval_dict["label_names"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"HMM Driving Behavior Confusion Matrix\n(Accuracy: {eval_dict['accuracy']:.1%}, Macro F1: {eval_dict['macro_f1']:.1%})", fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # Event Recall evaluation
    det_combined = pd.concat(all_detected_events, ignore_index=True) if all_detected_events else pd.DataFrame()
    ref_combined = pd.concat(all_ref_events, ignore_index=True) if all_ref_events else pd.DataFrame()
    event_eval = compute_event_recall(det_combined, ref_combined, tolerance_s=5.0)

    with open(METRICS_DIR / "event_metrics.json", "w") as f:
        json.dump(event_eval, f, indent=2)

    logger.info("Event Recall: %.2f%% (Detected: %d, Reference: %d, Matched: %d)",
                event_eval["recall"] * 100, event_eval["n_detected"], event_eval["n_reference"], event_eval["n_matched"])

    # Figure 6: Sensor Traces & HMM States for a representative trip
    sample_trip_id = test_meta[0]["trip_id"]
    sample_trip = next(t for t in trips if t["trip_id"] == sample_trip_id)
    merged = sample_trip["merged"]
    feat_sample = feat_df[feat_df["trip_id"] == sample_trip_id]
    sample_res = hmm_results[0]

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    t = merged["timestamp"].values
    axes[0].plot(t, merged["ax_filt"], color="#2980b9", lw=1, label="Longitudinal Accel ax (g)")
    axes[0].axhline(0.20, color="red", ls="--", lw=0.8, label="Sudden Accel Threshold")
    axes[0].axhline(-0.25, color="darkred", ls="--", lw=0.8, label="Hard Braking Threshold")
    axes[0].set_ylabel("ax (g)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, merged["ay_filt"], color="#e67e22", lw=1, label="Lateral Accel ay (g)")
    axes[1].axhline(0.20, color="orange", ls="--", lw=0.8, label="Turn Threshold")
    axes[1].axhline(-0.20, color="orange", ls="--", lw=0.8)
    axes[1].set_ylabel("ay (g)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    spd = merged["speed_ms"].values if "speed_ms" in merged.columns else (merged["speed_kmh"]/3.6).values
    axes[2].plot(t, spd, color="#27ae60", lw=1, label="Speed (m/s)")
    axes[2].set_ylabel("Speed (m/s)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)

    # Plot HMM State sequence
    t_starts = feat_sample["t_start"].values
    t_ends = feat_sample["t_end"].values
    lbl_seq = sample_res["label_seq"]
    color_map = {"NORMAL": "#2ecc71", "DROWSY": "#f39c12", "AGGRESSIVE": "#e74c3c"}
    for ts, te, lbl in zip(t_starts, t_ends, lbl_seq):
        axes[3].axvspan(ts, te, color=color_map.get(lbl, "#95a5a6"), alpha=0.45)
    axes[3].set_ylabel("HMM State")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_yticks([])
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=l) for l, c in color_map.items()]
    axes[3].legend(handles=handles, loc="upper right", fontsize=8)
    axes[3].grid(alpha=0.3)

    fig.suptitle(f"Sensor Readings & Decoded HMM Driving States ({sample_trip_id})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sensor_traces_and_hmm_states.png", dpi=300)
    plt.close()

    # Figure 7: GPS Route & Events
    if "latitude" in merged.columns and merged["latitude"].notna().any():
        plt.figure(figsize=(8, 7))
        plt.plot(merged["longitude"], merged["latitude"], color="#34495e", lw=2, label="GPS Path")
        plt.scatter(merged["longitude"].iloc[0], merged["latitude"].iloc[0], color="green", s=100, marker="^", label="Start", zorder=5)
        plt.scatter(merged["longitude"].iloc[-1], merged["latitude"].iloc[-1], color="red", s=100, marker="v", label="End", zorder=5)

        trip_events = det_combined[det_combined["trip_id"] == sample_trip_id] if not det_combined.empty else pd.DataFrame()
        if not trip_events.empty:
            plt.scatter(trip_events["longitude"], trip_events["latitude"], color="#e74c3c", s=70, marker="X", label="Detected Events", zorder=6)

        plt.title(f"GPS Route & Event Markers ({sample_trip_id})", fontsize=11, fontweight="bold")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(loc="upper left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "trip_gps_route_events.png", dpi=300)
        plt.close()

    # Baseline comparison
    threshold_events_per_trip = []
    for m in test_meta:
        t_obj = next(t for t in trips if t["trip_id"] == m["trip_id"])
        trip_feat = feat_df[feat_df["trip_id"] == m["trip_id"]]
        evs = detect_events_threshold(trip_feat, t_obj["merged"])
        threshold_events_per_trip.append(evs)

    baseline_cmp = evaluate_hmm_vs_threshold(hmm_results, threshold_events_per_trip, test_labels, state_map)
    with open(METRICS_DIR / "baseline_comparison.json", "w") as f:
        json.dump(baseline_cmp, f, indent=2)

    logger.info("=== Final Results Summary ===")
    logger.info("HMM Accuracy: %.2f%% vs Baseline: %.2f%%", baseline_cmp["hmm_accuracy"]*100, baseline_cmp["baseline_accuracy"]*100)
    logger.info("HMM Macro F1: %.2f%% vs Baseline: %.2f%%", baseline_cmp["hmm_macro_f1"]*100, baseline_cmp["baseline_macro_f1"]*100)
    logger.info("Experiments complete! All artifacts saved.")


if __name__ == "__main__":
    main()

