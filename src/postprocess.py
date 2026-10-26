"""Chain post-processing: burn-in removal, thinning, ESS, and pooled statistics.

Estimators in this repo historically consumed full weight traces from step 0. These
helpers make the usual post-processing explicit and uniform:

    burned, info = burn_in(traces, method="tau")     # drop the transient
    st = chain_stats(traces, burn="tau")             # one-call summary of (K, N)

Conventions: traces are (N,) or (K, N) float arrays of a scalar observable (typically
the cold-replica weight w). Multi-chain cuts are COMMON across chains so the array
stays rectangular. tau is the Geyer-truncated IAT from `metrics.iat`, clamped to >= 1
(as `compare/analyze_p2_diag_table.naive_se` already does) so ESS can never exceed N.
"""
import math

import numpy as np

from diagnostics import split_rhat
from metrics import iat

_TAU_ITERS = 5      # iterations of the burn-in fixed-point loop
_MIN_TAIL = 8       # never estimate tau on a shorter remainder


def _as_2d(traces):
    """-> ((K, N) float array, was_1d)."""
    x = np.asarray(traces, dtype=float)
    if x.ndim == 1:
        return x[None, :], True
    if x.ndim != 2:
        raise ValueError(f"traces must be 1-D (N,) or 2-D (K, N), got shape {x.shape}")
    return x, False


def _tau_median(x, start=0):
    """Median over chains of iat(chain[start:]), clamped to >= 1."""
    tail = x[:, start:]
    if tail.shape[1] < _MIN_TAIL:
        return float(tail.shape[1])
    return float(np.median([max(iat(row), 1.0) for row in tail]))


def burn_in(traces, method="frac", frac=0.5, c=10.0, max_frac=0.5):
    """Drop the initial transient of a trace. -> (burned, info), same rank as input.

    method="frac": drop the first `frac` fraction (n_dropped = int(frac * N)).
    method="tau":  drop the first c*tau_hat samples. tau_hat cannot be read off the
      raw trace -- a transient inflates the IAT arbitrarily -- so it is estimated
      ITERATIVELY: tau on the trace after the current provisional cut sets the next
      cut, repeated to a fixed point (`_TAU_ITERS` = 5 passes; a transient that decays
      on the autocorrelation timescale converges in 2-3). For (K, N) the median tau
      across chains sets one common cut, keeping the array rectangular.

    The cut is clamped to `max_frac * N`. A fixed point sitting AT that clamp means the
    c*tau rule wants to throw away more than the trace has, i.e. burn-in cannot rescue
    this chain (the trapping regime); `info["capped"]` is then True. The iteration can
    also fail to converge at all -- it oscillates when the transient is LONGER than
    c*tau, which the rule is blind to by construction -- and that case likewise falls
    back to the cap with capped=True, independent of `_TAU_ITERS`.

    info: n_dropped, tau_hat (final, post-cut, >= 1), capped (bool), method.
    """
    x, was_1d = _as_2d(traces)
    n = x.shape[1]
    cap = int(max_frac * n)
    capped = False

    if method == "frac":
        cut = min(int(frac * n), n)   # explicit request; max_frac applies to "tau" only
    elif method == "tau":
        cut, converged = 0, False
        for _ in range(_TAU_ITERS):
            tau = _tau_median(x, cut)
            new = min(int(math.ceil(c * tau)), cap)
            if new == cut:
                converged = True
                break
            cut = new
        if not converged:
            # No fixed point: the cut oscillates between the cap (tau measured with
            # the transient still in) and a small value (tau measured without it),
            # which happens when the transient is LONGER than c*tau. Neither end is
            # right, so fall back to the cap -- the conservative answer, flagged.
            cut = cap
        capped = cut >= cap and cap > 0
    else:
        raise ValueError(f"unknown burn-in method {method!r} (use 'frac' or 'tau')")

    burned = x[:, cut:]
    info = dict(n_dropped=int(cut), tau_hat=_tau_median(burned),
                capped=bool(capped), method=method)
    return (burned[0] if was_1d else burned), info


def thin(traces, stride=None):
    """Keep every `stride`-th sample. -> (thinned, stride), same rank as input.

    stride=None derives max(1, round(tau_hat)) from the trace itself (median tau
    across chains for (K, N)), i.e. roughly one independent sample per kept point.
    """
    x, was_1d = _as_2d(traces)
    if stride is None:
        stride = max(1, int(round(_tau_median(x))))
    stride = int(stride)
    if stride < 1:
        raise ValueError("stride must be >= 1")
    out = x[:, ::stride]
    return (out[0] if was_1d else out), stride


def ess(trace):
    """Effective sample size N / tau_hat. For a (K, N) array: sum of per-chain ESS."""
    x, _ = _as_2d(trace)
    n = x.shape[1]
    return float(sum(n / max(iat(row), 1.0) for row in x))


def chain_stats(traces, burn="frac", frac=0.5, c=10.0, thin_by_tau=False):
    """Canonical post-burn summary of a (K, N) multi-chain run. -> dict.

    Keys:
      ew_pool    mean of the per-chain means (equal weight per chain)
      sd_cross   std over chain means, ddof=1 -- the repo-wide `ew_std` convention
      se_cross   sd_cross / sqrt(K): SE of ew_pool from cross-chain scatter
      tau_median median post-burn per-chain IAT
      ess_total  sum_k n_used / max(tau_k, 1) -- per-chain tau, not the median
      se_ess     within-chain ESS-corrected SE of the pooled mean,
                 sqrt(pooled_var / ess_total), with pooled_var the variance of all
                 post-burn samples (ddof=1). For a well-mixed run se_ess and se_cross
                 estimate the same quantity and should agree; a large gap between them
                 is itself a symptom (chains that disagree beyond their own noise).
      rhat       split_rhat on the post-burn traces (nan if K < 2)
      burn_info  the burn_in() info dict
      stride     thinning stride (None unless thin_by_tau=True)
      n_used     post-burn (post-thin) samples PER CHAIN
    """
    x = np.asarray(traces, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"chain_stats needs a 2-D (K, N) array, got shape {x.shape}")

    burned, info = burn_in(x, method=burn, frac=frac, c=c)
    stride = None
    if thin_by_tau:
        burned, stride = thin(burned)

    k, n_used = burned.shape
    means = burned.mean(axis=1)
    sd_cross = float(means.std(ddof=1)) if k > 1 else 0.0
    taus = [max(iat(row), 1.0) for row in burned]
    ess_total = float(sum(n_used / t for t in taus))
    pooled_var = float(np.var(burned, ddof=1))
    return dict(
        ew_pool=float(means.mean()),
        sd_cross=sd_cross,
        se_cross=sd_cross / math.sqrt(k) if k > 1 else 0.0,
        tau_median=float(np.median(taus)),
        ess_total=ess_total,
        se_ess=math.sqrt(pooled_var / ess_total) if ess_total > 0 else float("nan"),
        rhat=split_rhat(burned) if k > 1 else float("nan"),
        burn_info=info,
        stride=stride,
        n_used=int(n_used),
    )
