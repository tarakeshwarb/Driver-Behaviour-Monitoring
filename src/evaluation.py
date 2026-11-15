"""
evaluation.py — Model Evaluation Module
=========================================
Implements evaluation metrics for the driver behaviour HMM:

  Metrics
  -------
  - Accuracy           : overall classification correctness
  - Macro F1           : mean F1 treating all classes equally
  - Per-class P/R/F1   : detailed breakdown
  - Confusion matrix   : heatmap-ready
  - Event recall       : fraction of reference events detected

  Comparisons
  -----------
  - Custom Viterbi vs hmmlearn reference
  - HMM vs threshold baseline
  - Window-length sensitivity (3 s vs 5 s)
  - Driver-independent generalisation

All results are returned as plain Python dicts or DataFrames for
easy serialisation and notebook display.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.custom_viterbi import GaussianHMMParams, compare_with_reference

logger = logging.getLogger(__name__)

LABELS = ["NORMAL", "DROWSY", "AGGRESSIVE"]


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------

def evaluate_classification(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict:
    """
    Compute accuracy, macro F1, per-class P/R/F1, and confusion matrix.

    Parameters
    ----------
    y_true  : ground-truth label list
    y_pred  : predicted label list
    labels  : class label ordering (default: NORMAL, DROWSY, AGGRESSIVE)

    Returns
    -------
    Dict with keys:
      accuracy, macro_f1, report (DataFrame), confusion_matrix (ndarray),
      label_names
    """
    if labels is None:
        labels = LABELS

    # Only keep labels that actually appear
    present_labels = [l for l in labels if l in set(y_true) | set(y_pred)]

    acc     = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=present_labels,
                        average="macro", zero_division=0)
    cm      = confusion_matrix(y_true, y_pred, labels=present_labels)
    report  = classification_report(
        y_true, y_pred, labels=present_labels,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report).T

    logger.info("Accuracy: %.3f | Macro F1: %.3f", acc, macro_f1)

    return {
        "accuracy":        acc,
        "macro_f1":        macro_f1,
        "report":          report_df,
        "confusion_matrix": cm,
        "label_names":     present_labels,
    }


# ---------------------------------------------------------------------------
# Event recall evaluation
# ---------------------------------------------------------------------------

def compute_event_recall(
    detected_events_df: pd.DataFrame,
    reference_events_df: pd.DataFrame,
    tolerance_s: float = 5.0,
) -> dict:
    """
    Compute event-level recall against reference annotations.

    A detection is a true positive if there is a reference event of the
    same broad type within ``tolerance_s`` seconds of the detected event.

    Parameters
    ----------
    detected_events_df  : DataFrame from event_detection.events_to_dataframe
    reference_events_df : DataFrame from preprocessing.load_events_inertial
    tolerance_s         : temporal matching window (seconds)

    Returns
    -------
    Dict with: recall, n_detected, n_reference, n_matched, precision, f1
    """
    type_map = {
        "sudden_acceleration": [2],
        "hard_braking":        [-2],
        "sharp_turn_left":     [1],
        "sharp_turn_right":    [-1],
    }

    if reference_events_df.empty:
        return {"recall": 0.0, "n_detected": len(detected_events_df),
                "n_reference": 0, "n_matched": 0, "precision": 0.0, "f1": 0.0}

    n_ref  = len(reference_events_df)
    matched_ref_indices  = set()
    matched_det_indices  = set()

    for det_idx, det_row in detected_events_df.iterrows():
        ref_types = type_map.get(det_row["event_type"], [])
        ref_subset = reference_events_df[
            reference_events_df["event_type"].isin(ref_types)
        ]
        det_mid = (det_row["t_start"] + det_row["t_end"]) / 2.0
        for ref_idx, ref_row in ref_subset.iterrows():
            if ref_idx in matched_ref_indices:
                continue
            if abs(det_mid - ref_row["timestamp"]) <= tolerance_s:
                matched_ref_indices.add(ref_idx)
                matched_det_indices.add(det_idx)
                break

    n_matched   = len(matched_ref_indices)
    n_detected  = len(detected_events_df)
    recall      = n_matched / n_ref     if n_ref     > 0 else 0.0
    precision   = n_matched / n_detected if n_detected > 0 else 0.0
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)

    return {
        "recall":      recall,
        "precision":   precision,
        "f1":          f1,
        "n_detected":  n_detected,
        "n_reference": n_ref,
        "n_matched":   n_matched,
    }


# ---------------------------------------------------------------------------
# Viterbi comparison
# ---------------------------------------------------------------------------

def evaluate_viterbi_comparison(
    model,
    test_sequences: list[np.ndarray],
    n_sequences:    int = 5,
) -> pd.DataFrame:
    """
    Compare custom Viterbi implementation vs hmmlearn's reference decoder.

    Runs both decoders on ``n_sequences`` test sequences and collects:
      - Sequence agreement (fraction of matching state labels)
      - Log-probability difference
      - Timing (ms)

    Parameters
    ----------
    model          : fitted hmmlearn.hmm.GaussianHMM
    test_sequences : list of observation arrays to compare on
    n_sequences    : maximum number of sequences to compare

    Returns
    -------
    DataFrame with per-sequence comparison results.
    """
    params = GaussianHMMParams.from_hmmlearn(model)
    rows = []

    for i, obs in enumerate(test_sequences[:n_sequences]):
        result = compare_with_reference(params, obs, model, use_vectorised=True)
        rows.append({
            "sequence_idx":        i,
            "length":              len(obs),
            "custom_log_prob":     result.custom_log_prob,
            "ref_log_prob":        result.ref_log_prob,
            "log_prob_diff":       result.log_prob_diff,
            "sequence_agreement":  result.sequence_agreement,
            "sequences_match":     result.sequences_match,
            "custom_time_ms":      result.custom_time_s * 1000,
            "ref_time_ms":         result.ref_time_s    * 1000,
        })
        logger.info(
            "Seq %d (len=%d): agreement=%.1f%%, Δlog_prob=%.4f",
            i, len(obs), result.sequence_agreement * 100, result.log_prob_diff,
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Threshold baseline comparison
# ---------------------------------------------------------------------------

def evaluate_hmm_vs_threshold(
    hmm_results:       list[dict],   # list of predict_trip() outputs
    threshold_events:  list,          # list of DrivingEvent lists
    y_true_labels:     list[str],
    state_to_label:    dict[int, str],
) -> dict:
    """
    Compare HMM classification accuracy vs a simple threshold-based baseline.

    Threshold baseline: classify each trip as AGGRESSIVE if any high-severity
    event was detected, DROWSY if the speed variance is high, else NORMAL.

    Parameters
    ----------
    hmm_results      : list of predict_trip() result dicts
    threshold_events : list of event lists (one per trip)
    y_true_labels    : ground-truth labels
    state_to_label   : HMM state-to-label map

    Returns
    -------
    Dict with hmm_accuracy, baseline_accuracy, hmm_macro_f1, baseline_macro_f1
    """
    hmm_pred = [r["dominant_label"] for r in hmm_results]

    # Simple threshold baseline: aggressive if any "high" event detected
    baseline_pred = []
    for events in threshold_events:
        if any(getattr(e, "severity", "") == "high" and
               "turn" not in getattr(e, "event_type", "")
               for e in events):
            baseline_pred.append("AGGRESSIVE")
        elif len(events) > 3:
            baseline_pred.append("DROWSY")
        else:
            baseline_pred.append("NORMAL")

    hmm_acc      = accuracy_score(y_true_labels, hmm_pred)
    baseline_acc = accuracy_score(y_true_labels, baseline_pred)
    hmm_f1       = f1_score(y_true_labels, hmm_pred,
                             labels=LABELS, average="macro", zero_division=0)
    baseline_f1  = f1_score(y_true_labels, baseline_pred,
                             labels=LABELS, average="macro", zero_division=0)

    return {
        "hmm_accuracy":       hmm_acc,
        "baseline_accuracy":  baseline_acc,
        "hmm_macro_f1":       hmm_f1,
        "baseline_macro_f1":  baseline_f1,
        "improvement_acc":    hmm_acc - baseline_acc,
        "improvement_f1":     hmm_f1  - baseline_f1,
    }


# ---------------------------------------------------------------------------
# Full experiment summary
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResults:
    """Container for all evaluation results from a full experiment run."""
    classification:    dict = field(default_factory=dict)
    event_recall:      dict = field(default_factory=dict)
    viterbi_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    hmm_vs_baseline:   dict = field(default_factory=dict)
    sweep_results:     pd.DataFrame = field(default_factory=pd.DataFrame)
    n_states_chosen:   int = 5
    window_seconds:    float = 3.0
    notes:             str = ""

    def summary(self) -> str:
        lines = [
            "=== Experiment Results Summary ===",
            f"  HMM states: {self.n_states_chosen}  |  Window: {self.window_seconds} s",
            "",
            "  Classification",
            f"    Accuracy:  {self.classification.get('accuracy',  0):.3f}",
            f"    Macro F1:  {self.classification.get('macro_f1',  0):.3f}",
            "",
            "  Event Detection",
            f"    Recall:    {self.event_recall.get('recall',    0):.3f}",
            f"    Precision: {self.event_recall.get('precision', 0):.3f}",
            f"    F1:        {self.event_recall.get('f1',        0):.3f}",
            "",
        ]
        if not self.viterbi_comparison.empty:
            mean_agree = self.viterbi_comparison["sequence_agreement"].mean()
            lines += [
                "  Viterbi Comparison",
                f"    Mean agreement: {mean_agree:.1%}",
            ]
        if self.hmm_vs_baseline:
            lines += [
                "",
                "  HMM vs Baseline",
                f"    HMM acc:      {self.hmm_vs_baseline.get('hmm_accuracy',      0):.3f}",
                f"    Baseline acc: {self.hmm_vs_baseline.get('baseline_accuracy',  0):.3f}",
                f"    Δ accuracy:   {self.hmm_vs_baseline.get('improvement_acc',    0):+.3f}",
            ]
        return "\n".join(lines)
