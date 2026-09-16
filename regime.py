from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def hmm_filtered_states(model, X: np.ndarray) -> np.ndarray:
    """Causal forward filter P(S_t | X_1..X_t), not Viterbi smoothing."""
    n_states = model.n_components
    reg = 1e-6 * np.eye(model.covars_[0].shape[0])
    log_emit = np.column_stack([
        multivariate_normal.logpdf(X, mean=model.means_[k], cov=model.covars_[k] + reg)
        for k in range(n_states)
    ])
    log_start = np.log(model.startprob_ + 1e-300)
    log_trans = np.log(model.transmat_ + 1e-300)
    states = np.zeros(len(X), dtype=int)
    log_alpha = log_start + log_emit[0]
    log_alpha -= logsumexp(log_alpha)
    states[0] = np.argmax(log_alpha)
    for t in range(1, len(X)):
        log_alpha = logsumexp(log_alpha[:, None] + log_trans, axis=0) + log_emit[t]
        log_alpha -= logsumexp(log_alpha)
        states[t] = np.argmax(log_alpha)
    return states


def label_states_by_forward_return(train_states: np.ndarray, forward_returns, labels: list[str]) -> dict[int, str]:
    """Labels states using only forward returns contained inside the training window."""
    if len(train_states) < 2:
        raise ValueError("Need at least two training observations")
    fr = np.asarray(forward_returns)
    n = min(len(train_states) - 1, len(fr) - 1)
    states = train_states[:n]
    vals = fr[:n]
    means = {s: np.nanmean(vals[states == s]) if np.any(states == s) else np.nan for s in range(len(labels))}
    ordered = sorted(means, key=lambda s: means[s] if np.isfinite(means[s]) else -np.inf, reverse=True)
    return {ordered[i]: labels[i] for i in range(len(labels))}
