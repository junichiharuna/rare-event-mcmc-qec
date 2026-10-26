"""Weight-stratified reference under DEPOLARIZING noise (spec 2026-10-07 §2).

Support weight w ~ Binomial(n, p); given the support, Pauli types are i.i.d. uniform on
{X, Y, Z}; so P_L = sum_w B(w; n, p) f_w^dep with f_w^dep = Pr(faulty | |supp| = w)."""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import direct_mc  # noqa: E402
import subset_sampling as ss  # noqa: E402


def test_plan_strata_counts_pauli_types():
    # class sizes 54, 1377, 22032, 247860: the cap sits between the last two
    plan = ss.plan_strata(18, 1, 4, enum_cap=10**5, type_mult=3)
    assert [(w, mode) for w, mode, _ in plan] == [(1, "enum"), (2, "enum"), (3, "enum"), (4, "mc")]
    assert plan[1][2] == math.comb(18, 2) * 9
    # bit-flip default unchanged
    assert ss.plan_strata(18, 1, 4, enum_cap=10**6)[3][2] == math.comb(18, 4)


def test_depol_ctx_has_both_matchings_and_the_runner_decoder():
    ctx = ss.build_ctx("depolarizing", "toric", 5, 0.01)
    assert ctx.noise == "depolarizing"
    assert ctx.matching is not None and ctx.matching_z is not None
    assert hasattr(ctx.decoder, "decode_x")
    ctx_bb = ss.build_ctx("depolarizing", "bb72", 0, 0.01)
    code_ref, dec_ref = direct_mc._make_code_and_decoder("depolarizing", "bb72", 0, 0.01)
    assert type(ctx_bb.decoder).__name__ == type(dec_ref).__name__
    assert np.array_equal(ctx_bb.code.hz, code_ref.hz)


def test_depol_decoder_is_the_runner_decoder():
    # Review focus 1: the reference and the chains must share the decoder (prior = p).
    sys.path.insert(0, os.path.join(ROOT, "cluster_scripts"))
    import paper_codes as PC
    ctx = ss.build_ctx("depolarizing", "bb_l9m6", 0, 1e-3)
    _, dec = PC.build("bb_l9m6", "depolarizing", 1e-3)
    assert type(ctx.decoder).__name__ == type(dec).__name__ == "_DepolarizingBPOSDDecoder"
    ctx = ss.build_ctx("depolarizing", "asymmetric_toric", "3x11", 1e-3)
    assert ctx.n == 66 and ctx.matching_z is not None


def test_generic_path_matches_fast_path_depol():
    ctx_fast = ss.build_ctx("depolarizing", "toric", 5, 0.01)
    ctx_gen = ss.build_ctx("depolarizing", "toric", 5, 0.01, force_generic=True)
    rng = np.random.default_rng(5)
    w, b = 5, 400   # w = 3 < d = 5 yields no failure in 400 shots; w = 5 yields ~30
    idx = np.ascontiguousarray(
        np.argpartition(rng.random((b, ctx_fast.n)), w - 1, axis=1)[:, :w].astype(np.int32))
    types = rng.integers(0, 3, size=(b, w)).astype(np.int8)
    m_fast = ss.fail_mask(idx, ctx_fast, types)
    m_gen = ss.fail_mask(idx, ctx_gen, types)
    assert np.array_equal(m_fast, m_gen) and m_fast.any()


def test_verify_against_repo_depol():
    ctx = ss.build_ctx("depolarizing", "toric", 5, 0.01)
    assert ss.verify_against_repo(ctx, n_shots=300, seed=7) is True


def test_toric_d3_depol_matches_direct_mc():
    p = 0.02
    res = ss.run_code("depolarizing", "toric", 3, p, w_max=5, enum_cap=10**7, n_max=1000,
                      workers=1, seed=1, cache_cap=0, verify=False, progress=False)
    assert all(r["mode"] == "enum" for r in res["fw_rows"])
    assert res["fw_rows"][0]["n_samples"] == math.comb(18, res["w_lo"]) * 3 ** res["w_lo"]
    rows = ss.rows_for_p_grid(res["fw_rows"], res["n"], [p],
                              dict(noise="depolarizing", code_type="toric", d="3",
                                   decoder=res["decoder"], w_max=res["w_max"]))
    row = rows[0]
    assert row["noise"] == "depolarizing"
    assert row["p_L_se"] <= ss.tail_bound(18, 5, p) + 1e-15      # floored at the tail only
    ref = direct_mc.run_condition("depolarizing", "toric", 3, p, n_target=10**9,
                                  m_max=60_000, n_cores=1)
    assert ref["n_faulty"] > 200
    assert abs(row["p_L"] - ref["p_L"]) < 3 * ref["p_L_se"]
    assert abs(row["ew_mean"] - ref["ew_mean"]) < 3 * ref["ew_se"]


def test_theorem_assert_holds_for_depol_mwpm():
    res = ss.run_code("depolarizing", "toric", 5, 0.01, w_max=3, enum_cap=10**7, n_max=500,
                      workers=1, seed=2, cache_cap=0, verify=False, progress=False)
    assert res["w_assert"] == 2
    assert all(r["n_fail"] == 0 for r in res["fw_rows"] if r["w"] <= 2)
