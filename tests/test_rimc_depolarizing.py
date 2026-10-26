import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from runners import run_rimc_depolarizing


def test_invalid_probabilities_raise():
    with pytest.raises(ValueError, match="sum to 1.0"):
        run_rimc_depolarizing(
            1e-2, 100, p_single=0.5, p_zanella=0.5, p_stab=0.2, p_logic=0.0
        )


def test_returns_correct_types():
    dist, trace = run_rimc_depolarizing(
        1e-2, 200, n_replicas=2, steps_per_swap=10, seed=42
    )
    assert isinstance(dist, dict)
    assert isinstance(trace, np.ndarray)
    assert trace.dtype == np.int32
    assert len(trace) == 200


def test_distribution_sums_to_one():
    dist, _ = run_rimc_depolarizing(1e-2, 200, n_replicas=2, seed=42)
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_all_weights_nonneg():
    dist, trace = run_rimc_depolarizing(1e-2, 200, n_replicas=2, seed=42)
    assert all(w >= 0 for w in dist.keys())
    assert np.all(trace >= 0)


def test_stab_only():
    dist, trace = run_rimc_depolarizing(
        1e-2,
        100,
        p_single=0.0,
        p_zanella=0.0,
        p_stab=1.0,
        p_logic=0.0,
        n_replicas=2,
        steps_per_swap=10,
        seed=0,
    )
    assert len(trace) == 100


def test_explicit_code_and_decoder():
    from css_code import make_toric_code, make_depolarizing_mwpm_decoder

    code = make_toric_code(3)
    dec = make_depolarizing_mwpm_decoder(code)
    dist, trace = run_rimc_depolarizing(
        1e-2, 200, code=code, decoder=dec, n_replicas=2, seed=0
    )
    assert isinstance(dist, dict)
    assert len(trace) == 200


def test_return_swap_stats_shape():
    dist, trace, swaps = run_rimc_depolarizing(
        1e-2,
        200,
        n_replicas=4,
        steps_per_swap=10,
        seed=42,
        return_swap_stats=True,
    )
    assert isinstance(dist, dict)
    assert isinstance(trace, np.ndarray)
    swaps = np.asarray(swaps, dtype=float)
    assert swaps.shape == (3,)  # n_replicas - 1 rungs
    assert np.all(swaps >= 0.0) and np.all(swaps <= 1.0)


def test_return_swap_stats_default_is_two_tuple():
    out = run_rimc_depolarizing(1e-2, 200, n_replicas=2, seed=42)
    assert len(out) == 2  # unchanged when flag absent


def test_return_swap_stats_no_pt_empty():
    dist, trace, swaps = run_rimc_depolarizing(
        1e-2,
        200,
        n_replicas=1,
        seed=42,
        return_swap_stats=True,
    )
    assert isinstance(swaps, np.ndarray)
    assert swaps.shape == (0,)  # n_replicas-1 == 0 rungs, no swaps


def _zh(**kw):
    return dict(p_single=0.0, p_zanella=0.7, p_cluster=0.3, cluster_k=2, n_replicas=1, **kw)


def test_cache_cap_does_not_change_trace():
    from css_code import make_toric_code, make_depolarizing_mwpm_decoder
    code = make_toric_code(3)
    dec = make_depolarizing_mwpm_decoder(code)
    _, t_ref = run_rimc_depolarizing(5e-2, 3000, code=code, decoder=dec, seed=3, **_zh())
    _, t_cap = run_rimc_depolarizing(5e-2, 3000, code=code, decoder=dec, seed=3,
                                     cache_cap=1, **_zh())
    assert np.array_equal(t_ref, t_cap)


def test_cache_cap_bounds_the_lookup_and_counts_decodes():
    from css_code import make_toric_code, make_depolarizing_mwpm_decoder
    code = make_toric_code(3)
    dec = make_depolarizing_mwpm_decoder(code)
    _, _, n_free = run_rimc_depolarizing(5e-2, 3000, code=code, decoder=dec, seed=3,
                                         return_decode_count=True, **_zh())
    _, _, n_cap = run_rimc_depolarizing(5e-2, 3000, code=code, decoder=dec, seed=3,
                                        cache_cap=4, return_decode_count=True, **_zh())
    assert n_cap >= n_free          # clears force re-decodes, never fewer


def test_record_trace_false_keeps_the_distribution():
    from css_code import make_toric_code, make_depolarizing_mwpm_decoder
    code = make_toric_code(3)
    dec = make_depolarizing_mwpm_decoder(code)
    d1, t1 = run_rimc_depolarizing(5e-2, 2000, code=code, decoder=dec, seed=1, **_zh())
    d2, t2 = run_rimc_depolarizing(5e-2, 2000, code=code, decoder=dec, seed=1,
                                   record_trace=False, **_zh())
    assert d1 == d2 and len(t1) == 2000 and len(t2) == 0
