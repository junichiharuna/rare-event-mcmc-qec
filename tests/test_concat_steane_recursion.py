import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "compare"))
from css_code import make_steane_code, _STEANE_HZ, _steane_lookup
from concat_steane_recursion import single_block_logical_prob, concat_pl


def _brute_single_block(p):
    import itertools
    code = make_steane_code()
    total = 0.0
    for bits in itertools.product([0, 1], repeat=7):
        e = np.array(bits, dtype=np.int8)
        w = int(e.sum())
        synd = (code.hz @ e) % 2
        corr = _steane_lookup(synd)
        residual = (e ^ corr) % 2
        if ((code.lz @ residual) % 2).any():
            total += (p ** w) * ((1 - p) ** (7 - w))
    return total


def test_single_block_matches_bruteforce():
    for p in [0.05, 0.1, 0.2]:
        assert abs(single_block_logical_prob(p) - _brute_single_block(p)) < 1e-12


def test_concat_pl_is_iterated_Q():
    p = 0.1
    q1 = single_block_logical_prob(p)
    q2 = single_block_logical_prob(q1)
    assert abs(concat_pl(p, 1) - q1) < 1e-12
    assert abs(concat_pl(p, 2) - q2) < 1e-12


def test_concat_pl_below_threshold_decreases_with_level():
    # well below the ~0.13 threshold, higher levels suppress P_L
    p = 0.02
    assert concat_pl(p, 3) < concat_pl(p, 2) < concat_pl(p, 1)


def test_recursion_matches_direct_mc_level2():
    import direct_mc
    from concat_steane_recursion import concat_pl

    p = 0.05
    res = direct_mc.run_condition(
        "bit_flip", "concat_steane", 2, p,
        n_target=400, m_max=4_000_000, base_seed=1, n_cores=1, decoder_kind="auto",
    )
    pl_mc = res["p_L"]
    pl_se = res["p_L_se"]
    pl_analytic = concat_pl(p, 2)
    # agree within 4 sigma of the MC estimate
    assert abs(pl_mc - pl_analytic) < 4 * pl_se + 1e-9
