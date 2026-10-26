import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from css_code import make_toric_code, make_depolarizing_mwpm_decoder  # noqa: E402
from kernels import _depolarizing_cluster_step  # noqa: E402


def _setup(d=3, seed=0):
    code = make_toric_code(d)
    dec = make_depolarizing_mwpm_decoder(code)
    rng = np.random.default_rng(seed)
    n = code.n
    ex = np.zeros(n, dtype=np.int8)
    ez = np.zeros(n, dtype=np.int8)
    # seed a few non-identity qubits (mixed Paulis): X, Y, Z
    ex[0] = 1
    ex[1] = 1
    ez[1] = 1
    ez[2] = 1
    sz = (code.hz @ ex % 2).astype(np.uint8)
    sx = (code.hx @ ez % 2).astype(np.uint8)
    w = int(np.sum(ex | ez))
    return code, dec, rng, ex, ez, w, sz, sx


def test_cluster_step_preserves_support():
    code, dec, rng, ex, ez, w, sz, sx = _setup()
    hz_cols = code.hz.T.astype(np.uint8)
    hx_cols = code.hx.T.astype(np.uint8)
    lookup_x, lookup_z = {}, {}
    for _ in range(500):
        nex, nez, nw, nsz, nsx = _depolarizing_cluster_step(
            ex, ez, w, sz, sx, code.n, 1, hz_cols, hx_cols,
            dec, lookup_x, lookup_z, rng,
        )
        assert nw == w
        assert int(np.sum(nex | nez)) == w
        ex, ez, w, sz, sx = nex, nez, nw, nsz, nsx


def test_cluster_step_guard_when_too_few_occupied():
    code, dec, rng, ex, ez, w, sz, sx = _setup()
    ex[:] = 0
    ez[:] = 0
    ex[0] = 1  # support = 1
    sz = (code.hz @ ex % 2).astype(np.uint8)
    sx = (code.hx @ ez % 2).astype(np.uint8)
    hz_cols = code.hz.T.astype(np.uint8)
    hx_cols = code.hx.T.astype(np.uint8)
    out = _depolarizing_cluster_step(
        ex, ez, 1, sz, sx, code.n, 2, hz_cols, hx_cols, dec, {}, {}, rng,
    )
    assert np.array_equal(out[0], ex) and np.array_equal(out[1], ez)  # unchanged


def test_cluster_step_syndrome_matches_recompute():
    code, dec, rng, ex, ez, w, sz, sx = _setup(seed=3)
    hz_cols = code.hz.T.astype(np.uint8)
    hx_cols = code.hx.T.astype(np.uint8)
    lookup_x, lookup_z = {}, {}
    for _ in range(300):
        ex, ez, w, sz, sx = _depolarizing_cluster_step(
            ex, ez, w, sz, sx, code.n, 1, hz_cols, hx_cols,
            dec, lookup_x, lookup_z, rng,
        )
        assert np.array_equal(sz, (code.hz @ ex % 2).astype(np.uint8))
        assert np.array_equal(sx, (code.hx @ ez % 2).astype(np.uint8))


def test_depolarizing_step_accepts_p_cluster():
    from kernels import _depolarizing_step

    code, dec, rng, ex, ez, w, sz, sx = _setup(seed=1)
    hz_cols = code.hz.T.astype(np.uint8)
    hx_cols = code.hx.T.astype(np.uint8)
    log_p1mp = np.log(0.01 / 3.0) - np.log(1 - 0.01)
    alpha = np.sqrt(3 * (1 - 0.01) / 0.01)
    beta = np.sqrt(0.01 / (3 * (1 - 0.01)))
    out = _depolarizing_step(
        ex, ez, w, sz, sx, code.n, log_p1mp, alpha, beta,
        code.hx, code.hx.shape[0], code.hz, code.hz.shape[0],
        code.lx, code.lz, code.lx.shape[0], hz_cols, hx_cols,
        0.0, 0.0, 0.0,  # p_stab, p_logic, p_single
        dec, {}, {}, rng,
        p_cluster=1.0, cluster_k=2,  # always take the cluster branch
    )
    assert int(np.sum(out[0] | out[1])) == w  # support preserved by the cluster branch


def test_run_rimc_depolarizing_zh_runs():
    from runners import run_rimc_depolarizing

    code = make_toric_code(3)
    dec = make_depolarizing_mwpm_decoder(code)
    dist, trace = run_rimc_depolarizing(
        0.05, 2000, p_zanella=0.7, p_cluster=0.3, cluster_k=2,
        n_replicas=1, code=code, decoder=dec, seed=0,
    )
    assert sum(dist.values()) > 0.99  # normalized faulty support distribution
    assert len(trace) > 0


def test_zh_depolarizing_matches_direct_mc_d3():
    import direct_mc
    from runners import run_rimc_depolarizing

    code = make_toric_code(3)
    dec = make_depolarizing_mwpm_decoder(code)
    p = 0.08

    # MCMC-independent reference: direct (rejection) MC mean support weight
    truth = direct_mc.run_condition(
        noise="depolarizing", code_type="toric", d=3, p=p,
        n_target=10000, m_max=10_000_000, decoder_kind="auto",
    )
    truth_ew = truth["ew_mean"]

    dist, _ = run_rimc_depolarizing(
        p, 150_000, p_zanella=0.7, p_cluster=0.3, cluster_k=2,
        n_replicas=1, code=code, decoder=dec, seed=0,
    )
    ew_mcmc = sum(w * f for w, f in dist.items())
    assert abs(ew_mcmc - truth_ew) < 0.1  # Z+H reproduces the truth within MC error
