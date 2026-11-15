"""
custom_viterbi.py — Custom Viterbi Algorithm Implementation
=============================================================
Implements the Viterbi algorithm for a Hidden Markov Model with
Gaussian emissions from scratch — no hmmlearn calls inside the decoding.

This satisfies the core-algorithm implementation requirement.

The Viterbi algorithm finds the most likely sequence of hidden states
given an observation sequence and HMM parameters (π, A, B).

Algorithm steps:
  1. Initialization  — seed probabilities using initial state distribution
  2. Recursion       — dynamic programming forward pass (log-domain)
  3. Backtracking    — trace the most likely path from the terminal state
  4. Reconstruction  — recover the full state sequence

All computations are in log-space to avoid numerical underflow.

References:
  Viterbi, A. (1967). Error bounds for convolutional codes and an
  asymptotically optimum decoding algorithm. IEEE Transactions on
  Information Theory, 13(2), 260–269.

  Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected
  applications in speech recognition. Proceedings of the IEEE, 77(2), 257–286.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)

LOG_ZERO = -1e300  # surrogate for log(0)


# ---------------------------------------------------------------------------
# HMM parameter container
# ---------------------------------------------------------------------------

@dataclass
class GaussianHMMParams:
    """
    Container for Gaussian HMM parameters extracted from hmmlearn.

    Attributes
    ----------
    pi   : (K,)       — log initial state probabilities
    A    : (K, K)     — log transition matrix
    mu   : (K, D)     — emission mean vectors
    cov  : (K, D, D)  — emission covariance matrices
    n_states : int    — number of hidden states K
    n_features : int  — observation dimensionality D
    """
    pi:        np.ndarray   # log π
    A:         np.ndarray   # log A
    mu:        np.ndarray   # μ
    cov:       np.ndarray   # Σ
    n_states:  int
    n_features: int

    @classmethod
    def from_hmmlearn(cls, model) -> "GaussianHMMParams":
        """
        Extract parameters from a fitted hmmlearn GaussianHMM object.

        Compatible with hmmlearn >= 0.3 where covars_ is a property that
        requires n_features to be set (i.e. model must have been fitted, or
        _covars_ must be set directly).

        Parameters
        ----------
        model : hmmlearn.hmm.GaussianHMM (fitted, or with manually set params)
        """
        pi  = np.log(np.clip(model.startprob_, 1e-300, 1.0))
        A   = np.log(np.clip(model.transmat_,  1e-300, 1.0))
        mu  = model.means_

        K, D = mu.shape

        # Access raw covariance storage — works both on fitted models and
        # models where parameters are set manually without calling fit().
        # hmmlearn 0.3+ stores the raw array in _covars_ before validation.
        cov_type = getattr(model, "covariance_type", "full")
        try:
            # Preferred: use the public property (requires n_features set)
            cov = model.covars_
        except AttributeError:
            # Fallback: access private attribute directly
            cov = model._covars_

        # Normalise covariance to full (K, D, D) regardless of storage type
        if cov.ndim == 3 and cov.shape == (K, D, D):
            pass  # already normalised to (K, D, D) by hmmlearn property
        elif cov.ndim == 2 and cov.shape == (K, D):
            # cov is (K, D) — diagonal per state
            cov_full = np.zeros((K, D, D))
            for k in range(K):
                cov_full[k] = np.diag(cov[k])
            cov = cov_full
        elif cov.ndim == 2 and cov.shape == (D, D):
            # tied covariance
            cov = np.stack([cov] * K)
        elif cov.ndim == 1 and cov.shape == (K,):
            # spherical covariance
            cov_full = np.zeros((K, D, D))
            for k in range(K):
                cov_full[k] = cov[k] * np.eye(D)
            cov = cov_full
        elif cov_type == "diag":
            cov_full = np.zeros((K, D, D))
            for k in range(K):
                cov_full[k] = np.diag(cov[k])
            cov = cov_full
        elif cov_type == "tied":
            cov = np.stack([cov] * K)
        elif cov_type == "spherical":
            cov_full = np.zeros((K, D, D))
            for k in range(K):
                cov_full[k] = cov[k] * np.eye(D)
            cov = cov_full

        return cls(pi=pi, A=A, mu=mu, cov=cov, n_states=K, n_features=D)


# ---------------------------------------------------------------------------
# Log-domain Gaussian emission probability
# ---------------------------------------------------------------------------

def log_emission(
    params: GaussianHMMParams,
    obs: np.ndarray,
    state: int,
) -> float:
    """
    Compute log P(obs | state) under a Gaussian emission.

    Uses scipy.stats.multivariate_normal for numerical stability.

    Parameters
    ----------
    params : GaussianHMMParams
    obs    : (D,) observation vector
    state  : hidden state index

    Returns
    -------
    Log probability (scalar).
    """
    try:
        log_p = multivariate_normal.logpdf(
            obs,
            mean=params.mu[state],
            cov=params.cov[state],
            allow_singular=True,
        )
        return float(log_p) if np.isfinite(log_p) else LOG_ZERO
    except Exception:
        return LOG_ZERO


def precompute_log_emissions(
    params: GaussianHMMParams,
    obs_seq: np.ndarray,
) -> np.ndarray:
    """
    Precompute all log P(o_t | s_k) for the full observation sequence.

    Parameters
    ----------
    params  : GaussianHMMParams
    obs_seq : (T, D) observation sequence

    Returns
    -------
    log_B : (T, K) matrix of log emission probabilities
    """
    T = len(obs_seq)
    K = params.n_states
    log_B = np.full((T, K), LOG_ZERO)

    for t in range(T):
        for k in range(K):
            log_B[t, k] = log_emission(params, obs_seq[t], k)

    return log_B


# ---------------------------------------------------------------------------
# Viterbi algorithm — core implementation
# ---------------------------------------------------------------------------

def viterbi_decode(
    params: GaussianHMMParams,
    obs_seq: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Viterbi algorithm — finds the most probable hidden state sequence.

    All computations are performed in log-space to prevent underflow.

    Algorithm
    ---------
    Initialization (t = 0):
        δ[0, k] = log π[k] + log B[0, k]   for all k
        ψ[0, k] = -1                         (no predecessor)

    Recursion (t = 1, …, T−1):
        δ[t, k] = max_j { δ[t-1, j] + log A[j, k] } + log B[t, k]
        ψ[t, k] = argmax_j { δ[t-1, j] + log A[j, k] }

    Termination:
        q*_T = argmax_k δ[T-1, k]
        log P* = max_k δ[T-1, k]

    Backtracking:
        q*_t = ψ[t+1, q*_{t+1}]   for t = T−2, …, 0

    Parameters
    ----------
    params  : GaussianHMMParams — trained HMM parameters
    obs_seq : (T, D) ndarray — observation sequence

    Returns
    -------
    state_seq : (T,) ndarray — most likely hidden state indices
    log_prob  : float — log probability of the state sequence
    """
    T = len(obs_seq)
    K = params.n_states

    if T == 0:
        return np.array([], dtype=np.int32), LOG_ZERO

    # Precompute log emission matrix — (T, K)
    log_B = precompute_log_emissions(params, obs_seq)

    # --- Viterbi tables ---
    # delta[t, k] = log probability of most likely path ending in state k at t
    delta = np.full((T, K), LOG_ZERO)
    # psi[t, k]   = predecessor state on the most likely path
    psi   = np.zeros((T, K), dtype=np.int32)

    # ----- Step 1: Initialization -----
    delta[0, :] = params.pi + log_B[0, :]
    psi[0, :]   = -1  # no predecessor at t=0

    # ----- Step 2: Recursion -----
    for t in range(1, T):
        for k in range(K):
            # Scores for transitioning from each state j to state k
            # score[j] = delta[t-1, j] + log A[j, k]
            scores = delta[t - 1, :] + params.A[:, k]
            best_prev = int(np.argmax(scores))
            psi[t, k]   = best_prev
            delta[t, k] = scores[best_prev] + log_B[t, k]

    # ----- Step 3: Termination -----
    best_last_state = int(np.argmax(delta[T - 1, :]))
    log_prob = float(delta[T - 1, best_last_state])

    # ----- Step 4: Backtracking -----
    state_seq = np.zeros(T, dtype=np.int32)
    state_seq[T - 1] = best_last_state

    for t in range(T - 2, -1, -1):
        state_seq[t] = psi[t + 1, state_seq[t + 1]]

    return state_seq, log_prob


# ---------------------------------------------------------------------------
# Vectorised Viterbi (faster, for benchmarking)
# ---------------------------------------------------------------------------

def viterbi_decode_vectorised(
    params: GaussianHMMParams,
    obs_seq: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Vectorised Viterbi using numpy operations.

    Equivalent algorithm to viterbi_decode but replaces the inner
    loop over K states with a matrix operation for speed.
    Used for benchmarking against the reference loop implementation.

    Parameters
    ----------
    params  : GaussianHMMParams
    obs_seq : (T, D) ndarray

    Returns
    -------
    state_seq : (T,) ndarray
    log_prob  : float
    """
    T = len(obs_seq)
    K = params.n_states

    if T == 0:
        return np.array([], dtype=np.int32), LOG_ZERO

    log_B = precompute_log_emissions(params, obs_seq)   # (T, K)

    delta = np.full((T, K), LOG_ZERO)
    psi   = np.zeros((T, K), dtype=np.int32)

    # Initialization
    delta[0] = params.pi + log_B[0]

    # Recursion — vectorised over K
    for t in range(1, T):
        # broadcast: (K_prev, 1) + (K_prev, K_next) → (K_prev, K_next)
        scores = delta[t - 1, :, None] + params.A      # (K, K)
        psi[t]   = np.argmax(scores, axis=0)           # (K,)
        delta[t] = scores[psi[t], np.arange(K)] + log_B[t]

    # Termination
    best_last = int(np.argmax(delta[T - 1]))
    log_prob  = float(delta[T - 1, best_last])

    # Backtracking
    state_seq = np.zeros(T, dtype=np.int32)
    state_seq[T - 1] = best_last
    for t in range(T - 2, -1, -1):
        state_seq[t] = psi[t + 1, state_seq[t + 1]]

    return state_seq, log_prob


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

@dataclass
class ViterbiComparison:
    """Results of comparing custom vs reference Viterbi implementations."""
    # Custom implementation
    custom_states:    np.ndarray
    custom_log_prob:  float
    custom_time_s:    float

    # Reference (hmmlearn)
    ref_states:       np.ndarray
    ref_log_prob:     float
    ref_time_s:       float

    # Comparison statistics
    sequence_agreement: float   # fraction of matching state assignments
    log_prob_diff:      float   # |custom_log_prob - ref_log_prob|
    sequences_match:    bool    # True if agreement == 1.0


def compare_with_reference(
    params: GaussianHMMParams,
    obs_seq: np.ndarray,
    hmm_model,
    use_vectorised: bool = True,
) -> ViterbiComparison:
    """
    Compare custom Viterbi implementation with hmmlearn's decode().

    Parameters
    ----------
    params         : GaussianHMMParams (must match hmm_model parameters)
    obs_seq        : (T, D) observation sequence
    hmm_model      : fitted hmmlearn.hmm.GaussianHMM
    use_vectorised : Use the vectorised implementation (default True)

    Returns
    -------
    ViterbiComparison dataclass with timing, agreement, and log-prob metrics.
    """
    decode_fn = viterbi_decode_vectorised if use_vectorised else viterbi_decode

    # --- Custom implementation ---
    t0 = time.perf_counter()
    custom_states, custom_log_prob = decode_fn(params, obs_seq)
    custom_time = time.perf_counter() - t0

    # --- Reference implementation ---
    t0 = time.perf_counter()
    ref_log_prob, ref_states = hmm_model.decode(obs_seq, algorithm="viterbi")
    ref_time = time.perf_counter() - t0

    ref_states = np.array(ref_states, dtype=np.int32)

    # State-label permutation may differ between runs → compute best agreement
    # via Hungarian matching or direct comparison
    agreement = float(np.mean(custom_states == ref_states))

    logger.info(
        "Viterbi comparison — agreement: %.1f%%, custom: %.3f ms, ref: %.3f ms",
        agreement * 100,
        custom_time * 1000,
        ref_time   * 1000,
    )

    return ViterbiComparison(
        custom_states=custom_states,
        custom_log_prob=custom_log_prob,
        custom_time_s=custom_time,
        ref_states=ref_states,
        ref_log_prob=ref_log_prob,
        ref_time_s=ref_time,
        sequence_agreement=agreement,
        log_prob_diff=abs(custom_log_prob - ref_log_prob),
        sequences_match=(agreement == 1.0),
    )
