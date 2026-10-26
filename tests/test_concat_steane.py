import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from css_code import CSSCode, make_steane_code  # later symbols imported per-test
# so the file still collects between TDD tasks (names that do not exist yet are
# imported inside the test that first uses them).


def test_concatenate_steane_level2_shape_and_css():
    from css_code import concatenate_css
    inner = make_steane_code()
    code = concatenate_css(make_steane_code(), inner)
    assert code.n == 49
    # 7 blocks x 3 inner + 3 lifted outer = 24 X-stabs (and same for Z)
    assert code.hx.shape == (24, 49)
    assert code.hz.shape == (24, 49)
    assert code.lx.shape == (1, 49)
    assert code.lz.shape == (1, 49)
    assert np.all((code.hx @ code.hz.T) % 2 == 0)
    assert ((code.lx @ code.lz.T) % 2).item() == 1
    assert np.all((code.hx @ code.lz.T) % 2 == 0)
    assert np.all((code.hz @ code.lx.T) % 2 == 0)


def test_concatenate_requires_k1_inner():
    from css_code import concatenate_css
    inner = make_steane_code()
    bad_outer = CSSCode(
        hx=inner.hx, hz=inner.hz,
        lx=np.vstack([inner.lx, inner.lx]),  # fake k=2
        lz=np.vstack([inner.lz, inner.lz]),
        n=7,
    )
    # inner must be k=1; an inner with k!=1 is rejected
    with pytest.raises(ValueError):
        concatenate_css(make_steane_code(), bad_outer)


def test_steane_shape_and_css():
    code = make_steane_code()
    assert code.n == 7
    assert code.hx.shape == (3, 7)
    assert code.hz.shape == (3, 7)
    # one logical qubit: n - rank(hx) - rank(hz) = 7 - 3 - 3 = 1
    assert code.lx.shape[0] == 1
    assert code.lz.shape[0] == 1
    # CSS commutation
    assert np.all((code.hx @ code.hz.T) % 2 == 0)
    # logical anticommutes with its partner, commutes with stabilizers
    assert ((code.lx @ code.lz.T) % 2).item() == 1
    assert np.all((code.hx @ code.lz.T) % 2 == 0)
    assert np.all((code.hz @ code.lx.T) % 2 == 0)
    assert code.hx.dtype == np.int8


def test_concatenated_steane_levels():
    from css_code import make_concatenated_steane_code
    c1 = make_concatenated_steane_code(1)
    assert c1.n == 7
    c2 = make_concatenated_steane_code(2)
    assert c2.n == 49 and c2.lx.shape == (1, 49)
    assert np.all((c2.hx @ c2.hz.T) % 2 == 0)
    c3 = make_concatenated_steane_code(3)
    assert c3.n == 343 and c3.lx.shape == (1, 343)
    assert np.all((c3.hx @ c3.hz.T) % 2 == 0)


def test_concatenated_steane_rejects_zero():
    from css_code import make_concatenated_steane_code
    with pytest.raises(ValueError):
        make_concatenated_steane_code(0)


def _all_weight_le(n, w):
    import itertools
    for k in range(w + 1):
        for combo in itertools.combinations(range(n), k):
            e = np.zeros(n, dtype=np.int8)
            for i in combo:
                e[i] = 1
            yield e


def test_hard_decoder_corrects_all_weight1_level2():
    from css_code import make_concatenated_steane_code, make_concat_steane_hard_decoder
    code = make_concatenated_steane_code(2)
    dec = make_concat_steane_hard_decoder(2)
    for i in range(code.n):
        e = np.zeros(code.n, dtype=np.int8)
        e[i] = 1
        synd = (code.hz @ e) % 2
        corr = dec.decode_syndrome(synd)
        assert not dec.is_faulty(e, corr), f"weight-1 error on qubit {i} not corrected"


def test_hard_decoder_consistent_with_code_syndrome():
    from css_code import make_concatenated_steane_code, make_concat_steane_hard_decoder
    code = make_concatenated_steane_code(2)
    dec = make_concat_steane_hard_decoder(2)
    # the returned correction must reproduce the measured syndrome
    rng = np.random.default_rng(0)
    for _ in range(50):
        e = (rng.random(code.n) < 0.1).astype(np.int8)
        synd = (code.hz @ e) % 2
        corr = dec.decode_syndrome(synd)
        assert np.all((code.hz @ corr) % 2 == synd)


def test_direct_mc_dispatch_concat_steane():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import direct_mc
    code, decoder = direct_mc._make_code_and_decoder(
        "bit_flip", "concat_steane", 2, 0.05, decoder_kind="auto"
    )
    assert code.n == 49
    # decoder corrects a weight-1 error
    e = np.zeros(49, dtype=np.int8); e[0] = 1
    corr = decoder.decode_syndrome((code.hz @ e) % 2)
    assert not decoder.is_faulty(e, corr)
    assert "concat_steane" in direct_mc._CODE_ID


def _run_concat(**kernel):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from runners import run_rimc_mixed
    from css_code import make_concatenated_steane_code, make_concat_steane_hard_decoder
    code = make_concatenated_steane_code(2)
    decoder = make_concat_steane_hard_decoder(2)
    dist, trace = run_rimc_mixed(
        0.05, 200, code=code, decoder=decoder, seed=0, **kernel
    )
    assert isinstance(dist, dict) and len(dist) > 0
    assert all(w >= 0 for w in dist) and abs(sum(dist.values()) - 1.0) < 1e-6
    assert trace.ndim == 1 and trace.size > 0
    return dist, trace


def test_run_rimc_mixed_concat_steane_obf():
    # OBF baseline
    _run_concat(p_single=1.0, p_zanella=0.0, n_replicas=1)


def test_run_rimc_mixed_concat_steane_zpt():
    # default Z+PT kernel (pure Zanella + parallel tempering)
    _run_concat(p_zanella=1.0, n_replicas=4)


def test_run_rimc_mixed_concat_steane_cluster():
    # Z+PT+H: cluster-exchange (the paper's de-biasing move) on the new code
    _run_concat(p_zanella=0.7, p_cluster=0.3, n_replicas=4)
