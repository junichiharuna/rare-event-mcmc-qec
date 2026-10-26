"""Tests for src/subset_sampling.py (code-agnostic weight-stratified P_L).

Section "検証" of docs/superpowers/specs/2026-08-28-subset-sampling-design.md.
Everything here runs serially (workers=1) and on tiny budgets so the whole file
stays well under a minute.
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import direct_mc  # noqa: E402
import subset_sampling as ss  # noqa: E402


# ── d-spec parsing (asymmetric toric "d1xd2") ──────────────────────────────
def test_d_spec_parsing():
    assert ss.parse_d("3x5") == "3x5"
    assert ss.parse_d(5) == "5"
    assert ss.parse_d("5") == "5"
    assert ss.d_parts("3x5") == (3, 5)
    assert ss.d_parts(7) == (7,)
    assert ss.code_distance("asymmetric_toric", "3x5") == 3
    assert ss.code_distance("toric", 5) == 5
    assert ss.code_distance("bb144", 0) == 12


def test_asymmetric_toric_d_string_builds_the_rectangular_code():
    ctx = ss.build_ctx("bit_flip", "asymmetric_toric", "3x5", 0.01)
    assert ctx.n == 2 * 3 * 5
    assert ctx.matching is not None  # MWPM fast path
    # square spec must agree with direct_mc's own asymmetric_toric(d, d)
    ctx_sq = ss.build_ctx("bit_flip", "asymmetric_toric", "5x5", 0.01)
    code_ref, _ = direct_mc._make_code_and_decoder("bit_flip", "asymmetric_toric", 5, 0.01)
    assert ctx_sq.n == code_ref.n


# ── stratum planning ───────────────────────────────────────────────────────
def test_plan_strata_switches_to_mc_above_the_cap():
    plan = ss.plan_strata(98, 3, 6, 5_000_000)
    assert [(w, mode) for w, mode, _ in plan] == [
        (3, "enum"), (4, "enum"), (5, "mc"), (6, "mc")]
    assert plan[0][2] == math.comb(98, 3)
    # w_max is clipped at n
    assert [w for w, _, _ in ss.plan_strata(18, 1, 99, 10**9)] == list(range(1, 19))


# ── generic (per-syndrome) path == pymatching decode_batch fast path ───────
def test_generic_path_matches_fast_path_on_mwpm_code():
    ctx_fast = ss.build_ctx("bit_flip", "toric", 5, 0.01)
    ctx_gen = ss.build_ctx("bit_flip", "toric", 5, 0.01, force_generic=True)
    assert ctx_fast.matching is not None
    assert ctx_gen.matching is None

    rng = np.random.default_rng(11)
    w, b = 3, 400
    idx = np.argpartition(rng.random((b, ctx_fast.n)), w - 1, axis=1)[:, :w]
    idx = np.ascontiguousarray(idx.astype(np.int32))

    m_fast = ss.fail_mask(idx, ctx_fast)
    m_gen = ss.fail_mask(idx, ctx_gen)
    assert np.array_equal(m_fast, m_gen)
    assert m_fast.any(), "expected some weight-3 failures on d=5 (test would be vacuous)"
    assert ss.fail_count(idx, ctx_fast) == int(m_fast.sum())


def test_verify_against_repo_mwpm():
    ctx = ss.build_ctx("bit_flip", "toric", 5, 0.01)
    assert ss.verify_against_repo(ctx, n_shots=300, seed=7) is True


# ── toric d=3: full enumeration vs direct MC ───────────────────────────────
def test_toric_d3_full_enumeration_matches_direct_mc():
    p = 0.05
    res = ss.run_code(
        "bit_flip", "toric", 3, p, w_max=18, enum_cap=10**9, n_max=1000,
        target_fail=10, pilot_n=100, workers=1, seed=1, cache_cap=0, verify=False)
    fw = res["fw_rows"]
    assert all(r["mode"] == "enum" for r in fw)
    assert res["n"] == 18
    # theorem: MWPM corrects every weight-(d-1)/2 = weight-1 error
    assert [r["f_w"] for r in fw if r["w"] == 1] == [0.0]

    est = ss.synthesize(18, fw, p)
    assert est["p_L_se"] == 0.0  # fully enumerated -> zero statistical variance
    assert ss.tail_bound(18, 18, p) == 0.0

    ref = direct_mc.run_condition(
        "bit_flip", "toric", 3, p, n_target=10**9, m_max=60_000, n_cores=1,
        decoder_kind="mwpm")
    for key, se_key in (("p_L", "p_L_se"), ("ew_mean", "ew_se")):
        sig = (est[key] - ref[key]) / math.hypot(est[se_key], ref[se_key])
        assert abs(sig) < 3.0, (
            f"{key}: exact {est[key]:.6g} vs directMC {ref[key]:.6g} -> {sig:+.2f} sigma")


# ── bb72 / BP-OSD generic path ─────────────────────────────────────────────
def test_bb72_bposd_generic_path_runs():
    ctx = ss.build_ctx("bit_flip", "bb72", 0, 0.01)
    assert ctx.matching is None  # BP-OSD -> no pymatching fast path
    n_seen, n_fail = ss.enum_stratum(ctx, 1, 0, ctx.n)
    assert n_seen == ctx.n
    assert n_fail == 0, "BP-OSD should correct every weight-1 error on bb72"
    n_s, n_f = ss.mc_stratum(ctx, 8, 150, [3, 1])
    assert n_s == 150 and 0 <= n_f <= 150
    assert ss.verify_against_repo(ctx, n_shots=40, seed=3, w=6) is True


# ── CSV schema ─────────────────────────────────────────────────────────────
def test_rows_for_p_grid_columns_match_csv_header():
    fw_rows = [
        dict(code_type="toric", d="3", p_decoder="", n=18, w=1, mode="enum",
             n_samples=18, n_fail=0, f_w=0.0, f_w_se=0.0),
        dict(code_type="toric", d="3", p_decoder="", n=18, w=2, mode="enum",
             n_samples=153, n_fail=12, f_w=12 / 153, f_w_se=0.0),
    ]
    meta = dict(noise="bit_flip", code_type="toric", d="3", decoder="mwpm",
                n_cores=4, n_target=1000, wall_clock=1.5, w_max=18)
    rows = ss.rows_for_p_grid(fw_rows, 18, [1e-2, 1e-3], meta)
    assert len(rows) == 2
    for row in rows:
        assert list(row.keys()) == direct_mc.CSV_HEADER
    assert rows[0]["n_trials"] == 18 + 153
    assert rows[0]["n_faulty"] == 12
    assert rows[0]["decoder"] == "mwpm"
    assert rows[0]["n_target"] == 1000


# ── cross-code-audit families (hgp_hamming3 / bb72 / bb_l9m6) ──────────────
# These three are the codes tab:audit reports; their reference values must be
# computed with the SAME decoder cluster_scripts/zh_rerun.py uses, i.e. BP-OSD
# with the channel prior set to p.  A missing _DISTANCE entry would silently
# switch the "zero failures below w_assert" theorem check off (assert_wmax -> 0),
# so the registry is pinned here rather than left implicit.
AUDIT_CODES = {
    "hgp_hamming3": dict(n=58, k=16, distance=3, w_assert=1),
    "bb72": dict(n=72, k=12, distance=6, w_assert=2),
    "bb_l9m6": dict(n=108, k=8, distance=10, w_assert=2),
    "bb144": dict(n=144, k=12, distance=12, w_assert=2),
}


def test_audit_code_registry():
    for name, exp in AUDIT_CODES.items():
        assert ss.is_p_independent(name) is False, (
            f"{name} must take the p-dependent BP-OSD path")
        assert ss.is_mwpm(name) is False, f"{name} has no pymatching fast path"
        assert ss.decoder_name(name) == "bposd"
        assert ss.code_distance(name, 0) == exp["distance"]
        assert ss.assert_wmax(name, 0) == exp["w_assert"]
    # MWPM families keep the coding-theory bound, and plain BP asserts nothing
    # (non-convergence counts as a failure, so there is no low-weight theorem).
    assert ss.assert_wmax("toric", 5) == 2
    assert ss.assert_wmax("bb72", 0, "bp") == 0


def test_audit_codes_build_with_the_zh_rerun_decoder():
    """build_ctx reproduces cluster_scripts/zh_rerun.py's code + BP-OSD decoder."""
    zh = _load_cluster_module("zh_rerun")
    for name, exp in AUDIT_CODES.items():
        ctx = ss.build_ctx("bit_flip", name, 0, 1e-2)
        assert ctx.n == exp["n"], name
        assert ctx.matching is None, f"{name} must not take the pymatching fast path"
        assert ctx.lzT.shape == (exp["n"], exp["k"]), name
        ref = zh.build_code(name)
        assert np.array_equal(ctx.code.hz, ref.hz), f"{name}: hz differs from zh_rerun"
        assert np.array_equal(ctx.code.lz, ref.lz), f"{name}: lz differs from zh_rerun"
        assert type(ctx.decoder).__name__ == "_BPOSDDecoder", name


def test_cli_parse_codes_accepts_the_new_names():
    cli = _load_cli()
    assert cli.parse_codes("hgp_hamming3:0,bb72:0,bb_l9m6:0") == [
        ("hgp_hamming3", "0"), ("bb72", "0"), ("bb_l9m6", "0")]
    # the CLI's task loop must emit one run PER p for every BP-OSD code
    assert all(not ss.is_p_independent(c) for c, _ in cli.parse_codes("hgp_hamming3:0,bb_l9m6:0"))
    assert cli.WMAX_BPOSD >= 8


def _load_cluster_module(stem):
    """Import cluster_scripts/<stem>.py under a name that cannot collide.

    cluster_scripts/subset_sampling.py shadows src/subset_sampling.py by name, so
    a plain import would silently return the wrong module.
    """
    import importlib.util

    path = os.path.join(ROOT, "cluster_scripts", f"{stem}.py")
    spec = importlib.util.spec_from_file_location(f"_cluster_{stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_cli():
    return _load_cluster_module("subset_sampling")


def _zh_truth(code):
    """(truth_ew, truth_se, p_L, p_L_se) for one code at p=1e-2 from zh_rerun.csv.

    zh_rerun.csv carries no p_L SE, so it is reconstructed from the direct-MC
    trial count implied by n_faulty_truth / p_L (binomial).
    """
    import csv

    path = os.path.join(ROOT, "cluster_results", "zh_rerun.csv")
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["code"] == code and float(r["p"]) == 0.01:
                p_l = float(r["p_L"])
                n_trials = float(r["n_faulty_truth"]) / p_l
                return (float(r["truth_ew"]), float(r["truth_se"]), p_l,
                        math.sqrt(p_l * (1.0 - p_l) / n_trials))
    raise AssertionError(f"no zh_rerun.csv truth row for {code} at p=1e-2")


def test_bb72_reference_matches_the_zh_rerun_direct_mc_truth():
    """Weight-stratified P_L / E[w|F] vs the independent direct-MC truth (3 sigma).

    Tiny budget (~4 s serial): w<=2 enumerated exactly, w=3..8 sampled.  The SE
    ceilings keep the 3-sigma band from growing wide enough to pass vacuously.
    """
    p = 1e-2
    res = ss.run_code(
        "bit_flip", "bb72", 0, p, w_max=8, enum_cap=3000, n_max=12_000,
        target_fail=200, pilot_n=500, workers=1, seed=5, cache_cap=0,
        verify=True, verify_shots=100, progress=False)
    assert res["n"] == 72
    assert res["decoder"] == "bposd"
    assert res["w_assert"] == 2
    assert [r["f_w"] for r in res["fw_rows"] if r["w"] <= 2] == [0.0, 0.0]

    n = res["n"]
    est = ss.synthesize(n, res["fw_rows"], p)
    # truncation at w_max must be negligible next to the statistical error
    assert ss.tail_bound(n, 8, p) < 0.01 * est["p_L"]

    truth_ew, truth_se, truth_pl, truth_pl_se = _zh_truth("bb72")
    assert est["ew_se"] < 0.10, "budget too small: the 3-sigma band would be vacuous"
    assert est["p_L_se"] < 0.15 * truth_pl
    for name, val, se, ref, ref_se in (
            ("E[w|F]", est["ew_mean"], est["ew_se"], truth_ew, truth_se),
            ("P_L", est["p_L"], est["p_L_se"], truth_pl, truth_pl_se)):
        sig = (val - ref) / math.hypot(se, ref_se)
        assert abs(sig) < 3.0, (
            f"{name}: stratified {val:.6g} +/- {se:.2g} vs zh_rerun direct-MC "
            f"{ref:.6g} +/- {ref_se:.2g} -> {sig:+.2f} sigma")


def test_hgp_hamming3_low_weight_theorem_and_generic_path():
    """d=3: BP-OSD corrects every weight-1 error, and w<=2 is fully enumerable."""
    ctx = ss.build_ctx("bit_flip", "hgp_hamming3", 0, 1e-2)
    n_seen, n_fail = ss.enum_stratum(ctx, 1, 0, ctx.n)
    assert (n_seen, n_fail) == (58, 0)
    assert ss.verify_against_repo(ctx, n_shots=60, seed=4, w=3) is True
    # weight 2 CAN fail on a d=3 code, so it must not be asserted away
    assert ss.assert_wmax("hgp_hamming3", 0) == 1


# ── concat_steane: p-independent code on the GENERIC (non-pymatching) path ──
# The level-2 concatenated Steane code [[49,1,9]] is decoded by a hierarchical
# hard-decision decoder that carries no channel prior, so f_w is p-independent
# (one run synthesizes the whole p grid) even though pymatching's decode_batch
# fast path does not apply.  That is the third regime is_p_independent/is_mwpm
# have to keep apart.
def test_concat_steane_is_p_independent_but_not_mwpm():
    assert ss.is_p_independent("concat_steane") is True
    assert ss.is_mwpm("concat_steane") is False
    assert ss.is_mwpm("concat_steane", "mwpm") is False  # no pymatching object either
    for name in AUDIT_CODES:
        assert ss.is_p_independent(name) is False, f"{name} f_w depends on the BP-OSD prior"
        assert ss.is_mwpm(name) is False
    for code_type, d in (("toric", 5), ("surface", 7), ("asymmetric_toric", "3x11")):
        assert ss.is_p_independent(code_type) is True
        assert ss.is_mwpm(code_type) is True
    assert ss.is_p_independent("bb72", "mwpm") is True
    assert ss.is_p_independent("bb72", "bp") is False


def test_concat_steane_distance_and_assert_wmax():
    # d(L_ell) = 3**ell
    assert ss.code_distance("concat_steane", "2") == 9
    assert ss.code_distance("concat_steane", 1) == 3
    assert ss.code_distance("concat_steane", 3) == 27
    # ... but the hard decoder is NOT a minimum-weight decoder: t_ell = 2**ell - 1
    assert ss.assert_wmax("concat_steane", "2") == 3
    assert ss.assert_wmax("concat_steane", 1) == 1
    assert ss.assert_wmax("concat_steane", 3) == 7
    # the CSV row keeps direct_mc's decoder label so the rows join like-for-like
    assert ss.decoder_name("concat_steane") == "mwpm"


def test_concat_steane_build_ctx_uses_the_hard_decoder():
    ctx = ss.build_ctx("bit_flip", "concat_steane", "2", 1e-3)
    assert ctx.n == 49
    assert ctx.matching is None, "concat_steane must not take the pymatching fast path"
    code_ref, dec_ref = direct_mc._make_code_and_decoder("bit_flip", "concat_steane", 2, 1e-3)
    assert np.array_equal(ctx.code.hz, code_ref.hz)
    assert type(ctx.decoder).__name__ == type(dec_ref).__name__


def test_concat_steane_run_code_and_p_grid_synthesis():
    """w<=3 enumerated exactly (theorem: zero failures), w=4 the first failures.

    C(49,3) = 18424 <= enum_cap, so f_1..f_3 are EXACT.  A weight-4 error with two
    errors in each of two inner Steane blocks defeats the hierarchical decoder, so
    f_4 > 0 and E[w|F] -> 4 as p -> 0.
    """
    res = ss.run_code(
        "bit_flip", "concat_steane", "2", 1e-3, w_max=6, enum_cap=20_000,
        n_max=2000, target_fail=20, pilot_n=200, workers=1, seed=1,
        cache_cap=50_000, verify=True, verify_shots=50, progress=False)
    assert res["n"] == 49
    assert res["decoder"] == "mwpm"
    assert res["w_assert"] == 3
    fw = {r["w"]: r for r in res["fw_rows"]}
    assert set(fw) == {1, 2, 3, 4, 5, 6}
    for w in (1, 2, 3):
        assert fw[w]["mode"] == "enum" and fw[w]["f_w"] == 0.0
    assert fw[3]["n_samples"] == math.comb(49, 3)
    assert fw[4]["f_w"] > 0.0
    # f_w is p-independent -> the p_decoder column is blank, like the MWPM codes
    assert all(r["p_decoder"] == "" for r in res["fw_rows"])

    meta = dict(noise="bit_flip", code_type="concat_steane", d="2",
                decoder=res["decoder"], n_cores=1, n_target=20,
                wall_clock=res["wall"], w_max=res["w_max"])
    ps = [1e-2, 1e-3, 1e-4]
    rows = ss.rows_for_p_grid(res["fw_rows"], res["n"], ps, meta)
    assert [r["p"] for r in rows] == ps
    assert all(list(r.keys()) == direct_mc.CSV_HEADER for r in rows)
    low = rows[-1]
    assert 4.0 <= low["ew_mean"] <= 4.05, low["ew_mean"]
    assert low["p_L"] > 0.0 and low["feasible"]


def test_cli_treats_concat_steane_as_one_run_for_the_whole_p_grid():
    cli = _load_cli()
    assert cli.parse_codes("concat_steane:2") == [("concat_steane", "2")]
    assert ss.is_p_independent("concat_steane") is True   # -> one run, all p
    assert cli.WMAX_MWPM == 16                            # ... and the MWPM w_max


def test_concat_steane_hard_decoder_bound_is_tight_at_level_3():
    """t_ell = 2**ell - 1 is the bound assert_wmax claims: correct at 7, tight at 8.

    Level 3 (n = 343, d = 27): random weight-7 errors are all corrected, while the
    constructed weight-8 error that puts two 2-error Steane blocks inside each of
    two level-2 blocks defeats the decoder -- so (d-1)//2 = 13 would be a false
    theorem while 2**3 - 1 = 7 is the true one.
    """
    ctx = ss.build_ctx("bit_flip", "concat_steane", "3", 1e-3, cache_cap=0)
    assert ctx.n == 343
    assert ss.assert_wmax("concat_steane", "3") == 7 < (ss.code_distance("concat_steane", "3") - 1) // 2
    assert ss.mc_stratum(ctx, 7, 200, [17, 3]) == (200, 0)
    idx = np.array([[blk * 49 + sub * 7 + q
                     for blk in (0, 1) for sub in (0, 1) for q in (0, 1)]], dtype=np.int32)
    assert ss.fail_mask(idx, ctx).tolist() == [True]


def test_cli_noise_flag_defaults():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ss_cli", os.path.join(ROOT, "cluster_scripts", "subset_sampling.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    a = mod.parse_args(["--noise", "depolarizing"])
    assert a.noise == "depolarizing"
    assert a.out_prefix.endswith("subset_sampling_depol")
    assert mod.parse_codes(a.codes) == [
        ("toric", "5"), ("toric", "7"), ("surface", "5"), ("surface", "7"),
        ("asymmetric_toric", "3x11"), ("asymmetric_toric", "3x15"),
        ("hgp_hamming3", "0"), ("bb72", "0"), ("bb144", "0"), ("bb_l9m6", "0")]
    assert [float(x) for x in a.p_list.split(",")] == [1e-2, 1e-3, 1e-4]
    b = mod.parse_args([])
    assert b.noise == "bit_flip" and b.out_prefix.endswith("subset_sampling")
