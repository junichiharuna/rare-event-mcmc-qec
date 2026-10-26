"""Convergence diagnostics for multi-chain MCMC runs."""

import numpy as np


def split_rhat(chains) -> float:
    """Classic split-R̂ (Gelman-Rubin) over a (K, N) array of K chains.

    Each chain is split into first/last halves (2K sequences; the middle sample
    is dropped when N is odd), and R̂ = sqrt(var_plus / W) with
    W = mean within-sequence variance, B/n = between-sequence mean variance,
    var_plus = (n-1)/n * W + B/n. Burn-in removal is the caller's job.
    This is the single implementation; compare/analyze_p2_diag_table.py imports it
    (its private copy, which dropped the last sample on odd N and returned nan for
    a single chain, was removed in favour of this one).
    R̂ near 1 is necessary (not sufficient) for convergence.
    """
    x = np.asarray(chains, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"chains must be 2-D (K, N), got shape {x.shape}")
    k, n_full = x.shape
    if k < 2:
        raise ValueError("need at least 2 chains")
    half = n_full // 2
    if half < 2:
        raise ValueError("chains too short to split")
    seqs = np.concatenate([x[:, :half], x[:, n_full - half:]], axis=0)  # (2K, half)
    n = seqs.shape[1]
    w = seqs.var(axis=1, ddof=1).mean()
    b_over_n = seqs.mean(axis=1).var(ddof=1)
    if w == 0.0:
        return 1.0 if b_over_n == 0.0 else float("inf")
    var_plus = (n - 1) / n * w + b_over_n
    return float(np.sqrt(var_plus / w))
