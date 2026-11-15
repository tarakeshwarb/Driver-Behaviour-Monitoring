"""
hmm_model.py — Gaussian HMM Training and Inference Module
===========================================================
Wraps hmmlearn.hmm.GaussianHMM with:
  - Chronological train/validation/test splitting
  - Multi-sequence training (all trips concatenated with length tracking)
  - State-to-label mapping via majority vote from labelled sequences
  - Hyperparameter comparison (n_states sweep)
  - Model persistence (pickle-based)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.metrics import accuracy_score

from src.custom_viterbi import GaussianHMMParams, viterbi_decode_vectorised

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

LABEL_TO_INT = {"NORMAL": 0, "DROWSY": 1, "AGGRESSIVE": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


# ---------------------------------------------------------------------------
# Train / validation / test chronological split
# ---------------------------------------------------------------------------

def chronological_split(
    trips: list[dict],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    gap_windows: int   = 5,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split trip list chronologically into train / validation / test sets.

    Trips are sorted by their folder name (which starts with YYYYMMDDHHMMSS)
    to ensure temporal ordering. A gap of ``gap_windows`` trips is enforced
    around boundaries to prevent data leakage from sliding windows.

    Parameters
    ----------
    trips       : list of preprocessed trip dicts
    train_ratio : fraction of trips for training (default 0.70)
    val_ratio   : fraction for validation (default 0.15)
    gap_windows : number of trips to skip around split boundaries

    Returns
    -------
    (train_trips, val_trips, test_trips)
    """
    sorted_trips = sorted(trips, key=lambda t: t.get("trip_id", ""))
    n = len(sorted_trips)

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = sorted_trips[:n_train]
    val   = sorted_trips[n_train + gap_windows: n_train + gap_windows + n_val]
    test  = sorted_trips[n_train + gap_windows + n_val + gap_windows:]

    logger.info(
        "Split: %d train, %d val, %d test trips (gap=%d)",
        len(train), len(val), len(test), gap_windows,
    )
    return train, val, test


# ---------------------------------------------------------------------------
# HMM training
# ---------------------------------------------------------------------------

def train_gaussian_hmm(
    sequences: list[np.ndarray],
    n_states: int = 5,
    covariance_type: str = "diag",
    n_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
) -> hmm.GaussianHMM:
    """
    Train a Gaussian HMM on a list of observation sequences.

    Multi-sequence training is performed by concatenating all sequences
    and passing the lengths list to hmmlearn, which correctly handles
    sequence boundaries in the Baum-Welch E-step.

    Parameters
    ----------
    sequences       : list of (N_i, D) arrays — one per trip
    n_states        : number of hidden states
    covariance_type : 'diag', 'full', 'tied', 'spherical'
    n_iter          : maximum EM iterations
    tol             : convergence tolerance on log-likelihood
    random_state    : reproducibility seed

    Returns
    -------
    Fitted GaussianHMM model.
    """
    if not sequences:
        raise ValueError("No sequences provided for training.")

    X = np.vstack(sequences)
    lengths = [len(s) for s in sequences]

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X, lengths)

    score = model.score(X, lengths) / len(X)
    logger.info(
        "HMM trained: K=%d, cov=%s, log-likelihood/sample=%.4f",
        n_states, covariance_type, score,
    )
    return model


# ---------------------------------------------------------------------------
# State → label mapping
# ---------------------------------------------------------------------------

def build_state_label_map(
    model: hmm.GaussianHMM,
    sequences: list[np.ndarray],
    labels: list[str],
) -> dict[int, str]:
    """
    Map anonymous HMM states to driving-behaviour labels via majority vote.

    For each trip sequence, decode the state sequence and accumulate a
    vote histogram per state for each driving-behaviour label.

    Parameters
    ----------
    model     : Fitted GaussianHMM
    sequences : list of observation sequences (one per labelled trip)
    labels    : corresponding driving-behaviour label per sequence

    Returns
    -------
    Dict mapping state_index → label string
    """
    K = model.n_components
    vote_matrix = np.zeros((K, len(LABEL_TO_INT)))

    for obs, label in zip(sequences, labels):
        if label not in LABEL_TO_INT:
            continue
        label_idx = LABEL_TO_INT[label]
        try:
            _, state_seq = model.decode(obs, algorithm="viterbi")
        except Exception as exc:
            logger.warning("Decode failed for sequence of length %d: %s", len(obs), exc)
            continue
        for state in state_seq:
            vote_matrix[state, label_idx] += 1

    state_to_label = {}
    for k in range(K):
        if vote_matrix[k].sum() > 0:
            best_class = int(np.argmax(vote_matrix[k]))
            state_to_label[k] = INT_TO_LABEL[best_class]
        else:
            state_to_label[k] = "NORMAL"

    logger.info("State-label mapping: %s", state_to_label)
    return state_to_label


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_trip(
    model: hmm.GaussianHMM,
    obs_seq: np.ndarray,
    state_to_label: dict[int, str],
    use_custom_viterbi: bool = True,
) -> dict:
    """
    Run Viterbi inference on a single trip sequence.

    Parameters
    ----------
    model              : Fitted GaussianHMM
    obs_seq            : (T, D) observation array
    state_to_label     : mapping from state index to label string
    use_custom_viterbi : use our custom implementation (default True)

    Returns
    -------
    Dict with keys: state_seq, label_seq, log_prob, dominant_label
    """
    if use_custom_viterbi:
        params = GaussianHMMParams.from_hmmlearn(model)
        state_seq, log_prob = viterbi_decode_vectorised(params, obs_seq)
    else:
        log_prob, state_seq = model.decode(obs_seq, algorithm="viterbi")
        state_seq = np.array(state_seq, dtype=np.int32)

    label_seq = np.array([state_to_label.get(s, "NORMAL") for s in state_seq])
    unique, counts = np.unique(label_seq, return_counts=True)
    dominant_label = unique[np.argmax(counts)]

    return {
        "state_seq":      state_seq,
        "label_seq":      label_seq,
        "log_prob":       log_prob,
        "dominant_label": dominant_label,
    }


# ---------------------------------------------------------------------------
# Hyperparameter sweep (n_states)
# ---------------------------------------------------------------------------

def sweep_n_states(
    train_sequences: list[np.ndarray],
    val_sequences:   list[np.ndarray],
    val_labels:      list[str],
    n_states_list:   list[int] | None = None,
    covariance_type: str = "diag",
) -> pd.DataFrame:
    """
    Train HMMs with different n_states and compare on validation set.

    Returns DataFrame with: n_states, ll_train, ll_val, accuracy, aic, bic
    """
    if n_states_list is None:
        n_states_list = [3, 4, 5]

    results = []
    X_train = np.vstack(train_sequences)
    lengths_train = [len(s) for s in train_sequences]
    X_val   = np.vstack(val_sequences)
    lengths_val   = [len(s) for s in val_sequences]
    n_features = train_sequences[0].shape[1]

    for K in n_states_list:
        try:
            model = train_gaussian_hmm(train_sequences, n_states=K,
                                       covariance_type=covariance_type)
            ll_train = model.score(X_train, lengths_train) / len(X_train)
            ll_val   = model.score(X_val,   lengths_val)   / len(X_val)

            if covariance_type == "diag":
                n_params = (K - 1) + K * (K - 1) + 2 * K * n_features
            else:
                n_params = (K - 1) + K * (K - 1) + K * n_features + K * n_features * (n_features + 1) // 2

            ll_total = model.score(X_val, lengths_val)
            aic = 2 * n_params - 2 * ll_total
            bic = n_params * np.log(len(X_val)) - 2 * ll_total

            state_map = build_state_label_map(model, val_sequences, val_labels)
            y_pred, y_true = [], []
            for obs, label in zip(val_sequences, val_labels):
                res = predict_trip(model, obs, state_map)
                y_pred.append(res["dominant_label"])
                y_true.append(label)

            acc = accuracy_score(y_true, y_pred)
            results.append({"n_states": K, "ll_train": ll_train, "ll_val": ll_val,
                             "accuracy": acc, "aic": aic, "bic": bic})
            logger.info("K=%d | acc=%.3f | AIC=%.1f | BIC=%.1f", K, acc, aic, bic)

        except Exception as exc:
            logger.error("Sweep failed for K=%d: %s", K, exc)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: hmm.GaussianHMM, state_map: dict, path: str | Path) -> None:
    """Persist model and state map to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "state_map": state_map}, f)
    logger.info("Model saved to %s", path)


def load_model(path: str | Path) -> tuple[hmm.GaussianHMM, dict]:
    """Load model and state map from disk."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["state_map"]
