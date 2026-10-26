import inspect
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from runners import run_rimc_mixed, run_rimc_depolarizing


def test_invalid_probabilities_raise():
    with pytest.raises(ValueError, match="sum to 1.0"):
        run_rimc_mixed(1e-2, 100, p_single=0.5, p_zanella=0.5, p_stab=0.2, p_logic=0.0)


def test_default_kernel_is_pure_z_pt():
    # default = Zanella + 4-replica PT; StabFlip/LogicFlip/cluster/hotOBF opt-in
    sig = inspect.signature(run_rimc_mixed)
    assert sig.parameters["p_zanella"].default == 1.0
    assert sig.parameters["p_stab"].default == 0.0
    assert sig.parameters["p_logic"].default == 0.0
    assert sig.parameters["p_single"].default == 0.0
    assert sig.parameters["p_cluster"].default == 0.0
    assert sig.parameters["hot_p_zanella"].default is None
    assert sig.parameters["n_replicas"].default == 4


def test_depolarizing_default_kernel_is_pure_z_pt():
    sig = inspect.signature(run_rimc_depolarizing)
    assert sig.parameters["p_zanella"].default == 1.0
    assert sig.parameters["p_stab"].default == 0.0
    assert sig.parameters["p_logic"].default == 0.0


def test_returns_correct_types():
    dist, trace = run_rimc_mixed(1e-2, 200, n_replicas=2, steps_per_swap=10, seed=42)
    assert isinstance(dist, dict)
    assert isinstance(trace, np.ndarray)
    assert trace.dtype == np.int32
    assert len(trace) == 200


def test_distribution_sums_to_one():
    dist, _ = run_rimc_mixed(1e-2, 200, n_replicas=2, seed=42)
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_single_kernel_only():
    dist, trace = run_rimc_mixed(
        1e-2,
        100,
        p_single=1.0,
        p_zanella=0.0,
        p_stab=0.0,
        p_logic=0.0,
        n_replicas=2,
        steps_per_swap=10,
        seed=0,
    )
    assert len(trace) == 100
    assert isinstance(dist, dict)


def test_stab_only():
    dist, trace = run_rimc_mixed(
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


def test_all_weights_nonneg():
    dist, trace = run_rimc_mixed(1e-2, 200, n_replicas=2, seed=42)
    assert all(w >= 0 for w in dist.keys())
    assert np.all(trace >= 0)


def test_explicit_code_and_decoder():
    from css_code import make_toric_code, make_mwpm_decoder

    code = make_toric_code(3)
    dec = make_mwpm_decoder(code)
    dist, trace = run_rimc_mixed(
        1e-2, 200, code=code, decoder=dec, n_replicas=2, seed=0
    )
    assert isinstance(dist, dict)
    assert len(trace) == 200


def test_default_equals_explicit():
    from css_code import make_toric_code, make_mwpm_decoder

    dist1, trace1 = run_rimc_mixed(1e-2, 200, n_replicas=2, seed=42)
    code = make_toric_code(3)
    dist2, trace2 = run_rimc_mixed(
        1e-2, 200, code=code, decoder=make_mwpm_decoder(code), n_replicas=2, seed=42
    )
    assert trace1.tolist() == trace2.tolist()


def test_return_swap_stats_shape():
    dist, trace, swap_rates = run_rimc_mixed(
        1e-2,
        200,
        n_replicas=3,
        steps_per_swap=10,
        seed=42,
        return_swap_stats=True,
    )
    assert isinstance(swap_rates, np.ndarray)
    assert swap_rates.shape == (2,)  # n_replicas - 1
    assert np.all(swap_rates >= 0.0)
    assert np.all(swap_rates <= 1.0)


def test_return_swap_stats_no_pt():
    dist, trace, swap_rates = run_rimc_mixed(
        1e-2,
        200,
        p_zanella=1.0,
        p_single=0.0,
        p_stab=0.0,
        p_logic=0.0,
        n_replicas=1,
        return_swap_stats=True,
        seed=42,
    )
    assert swap_rates.shape == (0,)


def test_randomize_init_runs_and_returns_correct_shape():
    from css_code import make_toric_code, make_mwpm_decoder

    code = make_toric_code(3)
    dec = make_mwpm_decoder(code)
    dist, trace = run_rimc_mixed(
        1e-2,
        200,
        code=code,
        decoder=dec,
        n_replicas=2,
        seed=42,
        randomize_init=True,
    )
    assert isinstance(dist, dict)
    assert isinstance(trace, np.ndarray)
    assert len(trace) == 200
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_randomize_init_false_matches_default():
    # randomize_init=False must be identical to the default (backward compat)
    dist1, trace1 = run_rimc_mixed(1e-2, 200, n_replicas=2, seed=7)
    dist2, trace2 = run_rimc_mixed(
        1e-2, 200, n_replicas=2, seed=7, randomize_init=False
    )
    assert trace1.tolist() == trace2.tolist()


def test_randomize_init_different_seeds_produce_different_traces():
    # With toric d=3 (n_logic=2) and randomize_init=True,
    # two different seeds should produce different weight traces.
    from css_code import make_toric_code, make_mwpm_decoder

    code = make_toric_code(3)
    dec = make_mwpm_decoder(code)
    _, trace_a = run_rimc_mixed(
        1e-2,
        200,
        code=code,
        decoder=dec,
        n_replicas=1,
        seed=0,
        randomize_init=True,
    )
    _, trace_b = run_rimc_mixed(
        1e-2,
        200,
        code=code,
        decoder=dec,
        n_replicas=1,
        seed=1,
        randomize_init=True,
    )
    assert trace_a.tolist() != trace_b.tolist()


# ── hot_p_zanella tests ────────────────────────────────────────────────────


def test_hot_replica_obf_no_error():
    """hot_p_zanella=0.0 should run without error and return valid output."""
    from css_code import make_toric_code, make_mwpm_decoder

    code = make_toric_code(3)
    decoder = make_mwpm_decoder(code)
    dist, trace = run_rimc_mixed(
        0.01, 500, n_replicas=4, hot_p_zanella=0.0, code=code, decoder=decoder, seed=42
    )
    assert isinstance(dist, dict)
    assert len(trace) > 0


def test_hot_replica_obf_convergence():
    """hot_p_zanella=0.0: cold replica E[w] converges to ~2.261 for d=3 p=0.01."""
    from css_code import make_toric_code, make_mwpm_decoder

    code = make_toric_code(3)
    decoder = make_mwpm_decoder(code)
    dist, trace = run_rimc_mixed(
        0.01, 5000, n_replicas=4, hot_p_zanella=0.0, code=code, decoder=decoder, seed=42
    )
    ew = sum(w * p for w, p in dist.items())
    assert 2.0 < ew < 2.6  # loose bound, exact=2.261


def test_hot_p_zanella_none_unchanged():
    """hot_p_zanella=None must give same result as not passing it."""
    from css_code import make_toric_code, make_mwpm_decoder

    code = make_toric_code(3)
    decoder = make_mwpm_decoder(code)
    dist1, trace1 = run_rimc_mixed(0.01, 500, code=code, decoder=decoder, seed=99)
    dist2, trace2 = run_rimc_mixed(
        0.01, 500, code=code, decoder=decoder, seed=99, hot_p_zanella=None
    )
    assert dist1 == dist2
    np.testing.assert_array_equal(trace1, trace2)


def test_hot_replica_obf_kernel_sum_no_error():
    """hot_p_zanella=0.0: hot kernel probs sum to 1.0 — no ValueError raised."""
    from css_code import make_toric_code, make_mwpm_decoder

    code = make_toric_code(3)
    decoder = make_mwpm_decoder(code)
    # Default p_stab=0.2, p_logic=0.2 => hot_p_single = 0.6; 0.2+0.2+0.6 = 1.0
    dist, trace = run_rimc_mixed(
        0.01, 200, n_replicas=4, hot_p_zanella=0.0, code=code, decoder=decoder, seed=0
    )
    assert isinstance(dist, dict)


# ── ones_arr invariant test ────────────────────────────────────────────────


# ── cluster exchange tests ─────────────────────────────────────────────────


def test_cluster_exchange_preserves_weight():
    """Cluster exchange must never change the weight."""
    import kernels as K
    import numpy as np
    from css_code import make_toric_code, make_mwpm_decoder

    rng = np.random.default_rng(7)
    code = make_toric_code(5)  # d=5, n=50
    decoder = make_mwpm_decoder(code)
    lookup = {}

    # Start from a faulty state
    error = code.lx[0].astype(np.int8)
    weight = int(np.sum(error))
    syndrome = (code.hz @ error % 2).astype(np.uint8)
    lookup[tuple(syndrome)] = decoder.decode_syndrome(syndrome)
    hz_cols = code.hz.T.astype(np.uint8)

    original_weight = weight
    for _ in range(100):
        new_error, new_weight, new_syndrome = K._cluster_exchange_step(
            error, weight, syndrome, code.n, hz_cols, 2, decoder, lookup, rng
        )
        # Weight must be preserved
        assert new_weight == original_weight
        # If accepted, verify by direct count
        if not np.array_equal(new_error, error):
            assert int(np.sum(new_error)) == original_weight
        error, weight, syndrome = new_error, new_weight, new_syndrome


def test_cluster_exchange_convergence():
    """p_cluster=0.2 should still converge to correct E[w]."""
    from css_code import make_toric_code, make_mwpm_decoder

    code = make_toric_code(3)
    decoder = make_mwpm_decoder(code)
    # p_zanella=0.4, p_cluster=0.2, rest same
    dist, trace = run_rimc_mixed(
        0.01,
        5000,
        p_zanella=0.4,
        p_stab=0.2,
        p_logic=0.2,
        p_single=0.0,
        p_cluster=0.2,
        n_replicas=1,
        code=code,
        decoder=decoder,
        seed=42,
    )
    ew = sum(w * p for w, p in dist.items())
    assert 1.8 < ew < 2.8


def test_cluster_exchange_backward_compat():
    """p_cluster=0.0 (default) must give same result as before."""
    from css_code import make_toric_code, make_mwpm_decoder

    code = make_toric_code(3)
    decoder = make_mwpm_decoder(code)
    dist1, trace1 = run_rimc_mixed(0.01, 500, code=code, decoder=decoder, seed=5)
    dist2, trace2 = run_rimc_mixed(
        0.01, 500, code=code, decoder=decoder, seed=5, p_cluster=0.0
    )
    assert dist1 == dist2


def test_cluster_exchange_k_too_large_skips():
    """When weight < k, step should return unchanged state."""
    import kernels as K
    from css_code import make_toric_code, make_mwpm_decoder

    rng = np.random.default_rng(0)
    code = make_toric_code(3)
    decoder = make_mwpm_decoder(code)
    lookup = {}
    error = code.lx[0].astype(np.int8)
    weight = int(np.sum(error))
    syndrome = (code.hz @ error % 2).astype(np.uint8)
    lookup[tuple(syndrome)] = decoder.decode_syndrome(syndrome)
    hz_cols = code.hz.T.astype(np.uint8)

    new_error, new_weight, new_syndrome = K._cluster_exchange_step(
        error, weight, syndrome, code.n, hz_cols, weight + 10, decoder, lookup, rng
    )
    np.testing.assert_array_equal(new_error, error)


# ── ones_arr invariant (from F) ────────────────────────────────────────────


def test_zanella_ones_arr_invariant_steps():
    import kernels as K
    from css_code import make_toric_code, make_mwpm_decoder

    rng = np.random.default_rng(0)
    code = make_toric_code(3)
    decoder = make_mwpm_decoder(code)
    lookup = {}

    error = code.lx[0].astype(np.int8)
    weight = int(np.sum(error))
    syndrome = (code.hz @ error % 2).astype(np.uint8)
    ones_arr = np.where(error == 1)[0].astype(np.int32)

    n = code.n
    alpha = np.sqrt((1 - 0.01) / 0.01)
    beta = np.sqrt(0.01 / (1 - 0.01))

    for _ in range(200):
        error, weight, syndrome, ones_arr = K._mixed_step_with_single(
            error,
            weight,
            syndrome,
            ones_arr,
            n,
            np.log(0.01 / 0.99),
            alpha,
            beta,
            alpha - beta,
            code.hx,
            code.hx.shape[0],
            code.lx,
            code.lx.shape[0],
            code.hz.T.astype(np.uint8),
            0.2,
            0.2,
            0.0,
            decoder,
            lookup,
            rng,
        )
        expected = set(np.where(error == 1)[0].tolist())
        actual = set(ones_arr.tolist())
        assert expected == actual, f"ones_arr mismatch: {actual} != {expected}"


# ── return_all_traces tests (D) ────────────────────────────────────────────


def test_return_all_traces_shape():
    from css_code import make_toric_code, make_mwpm_decoder

    code = make_toric_code(3)
    decoder = make_mwpm_decoder(code)
    result = run_rimc_mixed(
        0.01,
        500,
        n_replicas=4,
        code=code,
        decoder=decoder,
        seed=42,
        return_all_traces=True,
    )
    dist, trace, all_traces, log_p1mp_arr = result
    assert all_traces.shape == (4, len(trace))
    assert log_p1mp_arr.shape == (4,)
    np.testing.assert_array_equal(all_traces[0], trace)  # replica 0 matches trace


def test_return_all_traces_backward_compat():
    from css_code import make_toric_code, make_mwpm_decoder

    code = make_toric_code(3)
    decoder = make_mwpm_decoder(code)
    result = run_rimc_mixed(
        0.01, 500, n_replicas=4, code=code, decoder=decoder, seed=42
    )
    assert len(result) == 2  # (dist, trace) only


def test_return_all_traces_with_swap_stats():
    """return_all_traces=True + return_swap_stats=True appends (all_traces, log_p1mp) at end."""
    from css_code import make_toric_code, make_mwpm_decoder

    code = make_toric_code(3)
    decoder = make_mwpm_decoder(code)
    result = run_rimc_mixed(
        0.01,
        500,
        n_replicas=4,
        code=code,
        decoder=decoder,
        seed=42,
        return_swap_stats=True,
        return_all_traces=True,
    )
    assert len(result) == 5  # dist, trace, swap_rates, all_traces, log_p1mp_arr
    dist, trace, swap_rates, all_traces, log_p1mp_arr = result
    assert all_traces.shape == (4, len(trace))
    assert log_p1mp_arr.shape == (4,)
    np.testing.assert_array_equal(all_traces[0], trace)
