"""Splitting (p-ladder) estimator for the logical error rate p_L = P(faulty).

Pure post-processor: consumes integer weight traces (one per ladder rung, e.g.
from run_rimc_mixed) and a high-p anchor (p_L from direct MC), and returns
p_L(p) along the ladder by telescoping adjacent-measure importance ratios in
log space.

  p_L(p) = Z_faulty(p) = sum_{e faulty} p^w (1-p)^(N-w),
  Z_faulty(p_to)/Z_faulty(p_from)
      = E_{pi_{p_from}}[ (p_to/p_from)^w * ((1-p_to)/(1-p_from))^(N-w) ].
"""
from __future__ import annotations

import math

import numpy as np


def _logsumexp(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return -math.inf
    m = float(a.max())
    if not math.isfinite(m):
        return m
    return m + math.log(float(np.sum(np.exp(a - m))))


def _log_factors(trace_w, n, p_from, p_to):
    """Per-sample log reweighting factor f_t for the measure change p_from -> p_to."""
    w = np.asarray(trace_w, dtype=float)
    return (w * (math.log(p_to) - math.log(p_from))
            + (n - w) * (math.log1p(-p_to) - math.log1p(-p_from)))


def log_ratio(trace_w, n, p_from, p_to, n_batches=20):
    """Return (log_r, log_r_se): log of E_{pi_{p_from}}[factor] and its log-space SE.

    log_r estimates log[Z_faulty(p_to)/Z_faulty(p_from)]. SE via batch means
    (accounts for MCMC autocorrelation); delta method maps the factor-mean SE
    into log space.
    """
    f = _log_factors(trace_w, n, p_from, p_to)
    log_r = _logsumexp(f) - math.log(len(f))
    nb = min(n_batches, len(f))
    if nb < 2:
        return log_r, 0.0
    batches = np.array_split(f, nb)
    batch_means = np.array(
        [math.exp(_logsumexp(b) - math.log(len(b))) for b in batches])
    mean_factor = float(batch_means.mean())
    sem = float(batch_means.std(ddof=1)) / math.sqrt(nb)
    log_r_se = sem / mean_factor if mean_factor > 0 else float("nan")
    return log_r, log_r_se


def reweight_ess(trace_w, n, p_from, p_to):
    """Kish effective sample size of the reweighting factors (overlap diagnostic).

    Equals len(trace_w) when p_from == p_to; small values flag coarse spacing.
    """
    f = _log_factors(trace_w, n, p_from, p_to)
    return math.exp(2.0 * _logsumexp(f) - _logsumexp(2.0 * f))


def estimate_pl_ladder(rungs, anchor, n, n_batches=20):
    """Telescope p_L down a high->low p ladder from a high-p anchor.

    rungs: list of (p_k, trace_w_k), ordered high->low p.
    anchor: (p_0, log_pl0, log_pl0_se); p_0 must equal rungs[0][0].

    The step k-1 -> k ratio is estimated from the trace at the broader rung
    (k-1). Returns one dict per rung:
      {"p", "log_pl", "log_pl_se", "p_L", "p_L_se", "ess_rung"}
    where p_L = exp(log_pl), p_L_se = p_L * log_pl_se (delta method), and
    ess_rung is the overlap for the step leaving that rung (nan for the last).
    """
    p0, log_pl0, log_pl0_se = anchor
    if rungs[0][0] != p0:
        raise ValueError(f"anchor p {p0} != first rung p {rungs[0][0]}")
    m = len(rungs)
    log_pl = log_pl0
    log_var = log_pl0_se ** 2
    rows = []
    for k in range(m):
        p_k = rungs[k][0]
        if k >= 1:
            p_prev, trace_prev = rungs[k - 1]
            log_r, log_r_se = log_ratio(trace_prev, n, p_prev, p_k, n_batches)
            log_pl += log_r
            log_var += log_r_se ** 2
        log_pl_se = math.sqrt(log_var)
        p_L = math.exp(log_pl)
        if k < m - 1:
            ess = reweight_ess(rungs[k][1], n, p_k, rungs[k + 1][0])
        else:
            ess = float("nan")
        rows.append({"p": p_k, "log_pl": log_pl, "log_pl_se": log_pl_se,
                     "p_L": p_L, "p_L_se": p_L * log_pl_se, "ess_rung": ess})
    return rows


def estimate_pl_ladder_bar(rungs, anchor, n):
    """Bravyi-Vargo splitting: telescope p_L with Bennett's acceptance ratio (BAR).

    This is the original Bravyi-Vargo estimator (2013): the splitting identity
    p_L(p_t) = p_L(p_1) * prod_j R_j, with each adjacent ratio
    R_k = Z(p_k)/Z(p_{k-1}) = p_L(p_k)/p_L(p_{k-1}) estimated by the two-state
    Bennett acceptance-ratio method from samples at BOTH levels (forward work
    from the broader rung k-1, reverse work from rung k). NOT the multistate
    MBAR generalisation.

    Z_faulty(p) == p_L(p), so telescoping from the direct-MC anchor at the top
    rung yields p_L down the ladder. The work uses the full reduced-potential
    difference u_{p_k} - u_{p_{k-1}} (the constant (N-w) term is kept).

    rungs: list of (p_k, trace_w_k), ordered high->low p; rungs[0][0] == anchor p.
    anchor: (p_0, log_pl0, log_pl0_se).
    Returns one dict per rung: {"p", "log_pl", "log_pl_se", "p_L", "p_L_se"}.
    """
    from pymbar.other_estimators import bar

    p0, log_pl0, log_pl0_se = anchor
    if rungs[0][0] != p0:
        raise ValueError(f"anchor p {p0} != first rung p {rungs[0][0]}")

    log_pl = log_pl0
    log_var = log_pl0_se ** 2
    rows = []
    for k in range(len(rungs)):
        p_k = rungs[k][0]
        if k >= 1:
            p_prev = rungs[k - 1][0]
            w_prev = np.asarray(rungs[k - 1][1], dtype=float)  # samples at A = p_prev
            w_cur = np.asarray(rungs[k][1], dtype=float)       # samples at B = p_k
            # dU(w) = u_B(w) - u_A(w), u_p(w) = -[w log p + (N-w) log(1-p)]
            a = math.log(p_prev) - math.log(p_k)
            b = math.log1p(-p_prev) - math.log1p(-p_k)
            w_F = w_prev * a + (n - w_prev) * b        # forward work (A -> B) on A samples
            w_R = -(w_cur * a + (n - w_cur) * b)       # reverse work (B -> A) on B samples
            res = bar(w_F, w_R)
            dF = float(res["Delta_f"])                  # F_B - F_A = -log R_k
            dF_se = float(res["dDelta_f"])
            log_pl += -dF                                # log_pl_k = log_pl_{k-1} + log R_k
            log_var += dF_se ** 2
        log_pl_se = math.sqrt(log_var)
        p_L = math.exp(log_pl)
        rows.append({"p": p_k, "log_pl": log_pl, "log_pl_se": log_pl_se,
                     "p_L": p_L, "p_L_se": p_L * log_pl_se})
    return rows
