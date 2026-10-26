"""Splitting p_L estimator: ratio, overlap, and ladder recovery (MCMC-free)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "compare"))

from css_code import make_toric_code, make_mwpm_decoder
from exact_ew import faulty_weight_counts, exact_ew_from_counts
import splitting_pl as S

_C_FAULTY = None
_N = None


def _d3():
    """Exact faulty weight-count DOS c_faulty[w] and n for toric d=3 (cached)."""
    global _C_FAULTY, _N
    if _C_FAULTY is None:
        code = make_toric_code(3)
        dec = make_mwpm_decoder(code)
        _C_FAULTY, _N = faulty_weight_counts(code, dec)[:2]
    return _C_FAULTY, _N


def _draw(c_faulty, n, p, size, rng):
    """Independent weight samples from the exact marginal P(w) ∝ c_faulty[w] p^w (1-p)^(n-w)."""
    ws = np.arange(n + 1)
    valid = c_faulty > 0
    logu = np.full(n + 1, -np.inf)
    logu[valid] = (np.log(c_faulty[valid].astype(float))
                   + ws[valid] * math.log(p) + (n - ws[valid]) * math.log1p(-p))
    mx = logu[valid].max()
    probs = np.where(valid, np.exp(logu - mx), 0.0)
    probs /= probs.sum()
    return rng.choice(ws, size=size, p=probs).astype(np.int32)


def test_log_ratio_recovers_known_ratio():
    c, n = _d3()
    p_from, p_to = 0.05, 0.02
    trace = _draw(c, n, p_from, 1_000_000, np.random.default_rng(0))
    log_r, log_r_se = S.log_ratio(trace, n, p_from, p_to)
    exact = (exact_ew_from_counts(c, n, p_to)[2]
             / exact_ew_from_counts(c, n, p_from)[2])
    assert abs(math.exp(log_r) - exact) / exact < 0.03
    assert log_r_se > 0.0


def test_ratio_zero_when_p_equal():
    c, n = _d3()
    trace = _draw(c, n, 0.03, 10_000, np.random.default_rng(1))
    log_r, log_r_se = S.log_ratio(trace, n, 0.03, 0.03)
    assert log_r == 0.0
    assert log_r_se == 0.0


def test_reweight_ess_full_when_p_equal():
    c, n = _d3()
    trace = _draw(c, n, 0.03, 5_000, np.random.default_rng(2))
    ess = S.reweight_ess(trace, n, 0.03, 0.03)
    assert abs(ess - len(trace)) < 1e-6


def test_estimate_pl_ladder_recovers_exact_d3():
    c, n = _d3()
    ladder = [0.1, 0.03, 0.01, 0.003, 0.001]
    rng = np.random.default_rng(3)
    rungs = [(p, _draw(c, n, p, 1_000_000, rng)) for p in ladder]
    pl0 = exact_ew_from_counts(c, n, ladder[0])[2]
    anchor = (ladder[0], math.log(pl0), 0.0)
    rows = S.estimate_pl_ladder(rungs, anchor, n)
    assert [r["p"] for r in rows] == ladder
    exact_last = exact_ew_from_counts(c, n, ladder[-1])[2]
    assert abs(rows[-1]["p_L"] - exact_last) / exact_last < 0.05
    assert rows[-1]["log_pl_se"] > rows[1]["log_pl_se"] > 0.0
    assert rows[-1]["p_L_se"] > 0.0
    assert math.isnan(rows[-1]["ess_rung"])


def test_estimate_pl_ladder_bar_recovers_exact_d3():
    c, n = _d3()
    ladder = [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001]
    rng = np.random.default_rng(7)
    rungs = [(p, _draw(c, n, p, 200_000, rng)) for p in ladder]
    pl0 = exact_ew_from_counts(c, n, ladder[0])[2]
    anchor = (ladder[0], math.log(pl0), 0.0)
    rows = S.estimate_pl_ladder_bar(rungs, anchor, n)
    assert [r["p"] for r in rows] == ladder
    # top rung equals the anchor exactly (no ratio applied)
    assert abs(rows[0]["p_L"] - pl0) / pl0 < 1e-12
    exact_last = exact_ew_from_counts(c, n, ladder[-1])[2]
    assert abs(rows[-1]["p_L"] - exact_last) / exact_last < 0.05
    # log-space SE accumulates down the ladder (absolute p_L_se shrinks with p_L)
    assert rows[-1]["log_pl_se"] > rows[1]["log_pl_se"] > 0.0
    assert rows[-1]["p_L_se"] > 0.0


def test_ladder_rejects_anchor_mismatch():
    import pytest
    c, n = _d3()
    rng = np.random.default_rng(4)
    rungs = [(0.1, _draw(c, n, 0.1, 1000, rng)),
             (0.05, _draw(c, n, 0.05, 1000, rng))]
    with pytest.raises(ValueError):
        S.estimate_pl_ladder(rungs, (0.2, -1.0, 0.0), n)
