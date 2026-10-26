"""Unit tests for the convergence-flag rule functions (compare/convergence_flags.py).

Only the pure decision functions are exercised, on synthetic inputs: the loaders
are covered by running the CLI against cluster_results/.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "compare"))

from convergence_flags import (  # noqa: E402
    classify, classify_primary, combine_se, drift_verdict, late_drift_verdict,
    multistart_verdict, rhat_critical, rhat_verdict, scaling_verdict,
    select_within, wilson_hilferty_chi2,
)


# ------------------------------------------------------------------ chi2 cut --

def test_wilson_hilferty_matches_known_quantiles():
    # chi^2_{49} upper quantiles: 74.92 at 0.99, 85.35 at 0.999.
    assert wilson_hilferty_chi2(49, 0.99) == pytest.approx(74.92, rel=0.01)
    assert wilson_hilferty_chi2(49, 0.999) == pytest.approx(85.35, rel=0.01)


def test_rhat_critical_is_k_aware_and_monotone():
    assert rhat_critical(50, 1e-2) == pytest.approx(1.24, abs=0.02)
    assert rhat_critical(50, 1e-3) == pytest.approx(1.32, abs=0.02)
    # fewer chains -> a noisier s_B -> a looser cut
    assert rhat_critical(8, 1e-3) > rhat_critical(50, 1e-3) > 1.0
    # a stricter false-alarm rate -> a looser cut
    assert rhat_critical(50, 1e-4) > rhat_critical(50, 1e-2)
    assert rhat_critical(1, 1e-3) == float("inf")


# ------------------------------------------------------------------- scaling --

def test_scaling_passes_on_exact_one_over_sqrt_n():
    v = scaling_verdict([50000, 100000, 200000], [0.4, 0.4 / math.sqrt(2), 0.2])
    assert v["ok"] is True
    assert v["ratio"] == pytest.approx(1.0)
    assert v["sd_expected"] == pytest.approx(0.2)


def test_scaling_fails_when_spread_does_not_shrink():
    # a frozen inter-sector offset: the SD stays put as N grows
    v = scaling_verdict([50000, 200000], [0.4, 0.4])
    assert v["ok"] is False
    assert v["ratio"] == pytest.approx(2.0)


def test_scaling_tolerance_is_the_factor():
    ns, base = [50000, 200000], 0.4
    expect = base / 2.0
    assert scaling_verdict(ns, [base, 1.4 * expect], scale_factor=1.5)["ok"] is True
    assert scaling_verdict(ns, [base, 1.6 * expect], scale_factor=1.5)["ok"] is False
    assert scaling_verdict(ns, [base, 1.6 * expect], scale_factor=2.0)["ok"] is True


def test_scaling_undecidable_with_one_checkpoint():
    v = scaling_verdict([200000], [0.2])
    assert v["ok"] is None
    assert math.isnan(v["ratio"])
    assert scaling_verdict([], [])["ok"] is None


# --------------------------------------------------------------------- drift --

def test_drift_passes_within_k_sigma_and_fails_beyond():
    assert drift_verdict(3.0, 3.0 + 2.9 * 0.01, 0.01, drift_k=3.0)["ok"] is True
    assert drift_verdict(3.0, 3.0 - 3.1 * 0.01, 0.01, drift_k=3.0)["ok"] is False
    assert drift_verdict(3.0, 2.97, 0.01)["drift"] == pytest.approx(-0.03)
    assert drift_verdict(3.0, 2.97, 0.01)["drift_z"] == pytest.approx(3.0)


def test_drift_with_zero_se_never_passes():
    assert drift_verdict(3.0, 3.0001, 0.0)["ok"] is False


# ---------------------------------------------------------------------- rhat --

def test_rhat_is_the_dispersion_ratio():
    v = rhat_verdict(0.02, 0.01, 1.32)
    assert v["rhat"] == pytest.approx(2.0)
    assert v["ok"] is False
    assert rhat_verdict(0.0105, 0.01, 1.32)["ok"] is True


def test_rhat_undefined_without_a_within_chain_se():
    for bad in (float("nan"), 0.0, None):
        v = rhat_verdict(0.02, bad, 1.32)
        assert v["ok"] is None and math.isnan(v["rhat"])


# ------------------------------------------------------------------ classify --

BASE = dict(scaling_ok=None, rhat_ok=True, rhat_circular=False,
            has_variance_ref=True, drift_ok=True, aux_a_ok=True, aux_b_ok=True)


def _c(**kw):
    return classify(**{**BASE, **kw})


def test_scaling_takes_precedence_over_rhat():
    ok, src, _ = _c(scaling_ok=True, rhat_ok=False)
    assert (ok, src) == (True, "scaling")
    ok, src, why = _c(scaling_ok=False, rhat_ok=True)
    assert (ok, src, why) == (False, "scaling", "scaling")


def test_drift_is_an_and_on_every_branch():
    assert _c(drift_ok=False) == (False, "rhat", "drift")
    assert _c(scaling_ok=True, drift_ok=False) == (False, "scaling", "drift")
    assert _c(rhat_circular=True, drift_ok=False)[0] is False


def test_missing_drift_data_is_treated_as_passing():
    assert _c(drift_ok=None)[0] is True


def test_no_variance_reference_is_never_certified():
    # even with a perfect burn-in and a very long chain: this is the
    # p6_depol bb144 1e-3 minw OBF case (capped5=0, n/tau=251, bias +2.99)
    ok, src, why = _c(has_variance_ref=False, rhat_ok=None,
                      aux_a_ok=True, aux_b_ok=True)
    assert (ok, src, why) == (False, "none", "no_variance_reference")


def test_circular_rhat_row_falls_back_to_auxiliary_criteria():
    ok, src, _ = _c(rhat_circular=True, rhat_ok=True)
    assert (ok, src) == (True, "aux(circular-Rhat)")
    assert _c(rhat_circular=True, aux_a_ok=False) == (
        False, "aux(circular-Rhat)", "fixedpoint")
    assert _c(rhat_circular=True, aux_b_ok=False) == (
        False, "aux(circular-Rhat)", "short")


def test_auxiliary_criteria_do_not_override_a_passing_rhat():
    # (a)/(b) are reported but not part of the AND on Rhat-decided rows
    assert _c(aux_a_ok=False, aux_b_ok=False) == (True, "rhat", "ok")


def test_rhat_failure_reason_is_named():
    assert _c(rhat_ok=False) == (False, "rhat", "rhat")
    assert _c(rhat_ok=False, drift_ok=False) == (False, "rhat", "rhat+drift")


# ------------------------------------------------------------------ combine --

def test_combine_se_is_quadrature_and_skips_nan():
    assert combine_se(3.0, 4.0) == pytest.approx(5.0)
    assert combine_se(3.0, float("nan")) == pytest.approx(3.0)
    assert combine_se() == 0.0


# ------------------------------------------ primary rule: scaling reference --

CKPTS = [12500.0, 25000.0, 50000.0, 100000.0, 200000.0]


def _sqrt_n_sds(sd_final=0.05):
    """Between-chain SDs that shrink exactly like 1/sqrt(n)."""
    N = CKPTS[-1]
    return [sd_final * math.sqrt(N / n) for n in CKPTS]


def test_scaling_ref_frac_picks_the_quarter_checkpoint():
    v = scaling_verdict(CKPTS, _sqrt_n_sds(), ref_frac=0.25)
    assert v["n_ref"] == 50000.0
    assert v["ratio"] == pytest.approx(1.0)
    assert v["ratio_first"] == pytest.approx(1.0)
    assert v["ok"] is True


def test_scaling_ref_frac_default_is_the_earliest_checkpoint():
    assert scaling_verdict(CKPTS, _sqrt_n_sds())["n_ref"] == CKPTS[0]


def test_scaling_n16_reference_is_fooled_by_a_transient_the_n4_one_is_not():
    """UT(3,15) p=1e-2 long OBF in miniature.

    A decaying initialization transient inflates the EARLIEST between-chain SD, so
    the N/16 reference reports a spuriously healthy ratio while the N/4 reference,
    measured after the transient, sees the spread stall.
    """
    sds = [0.90, 0.45, 0.20, 0.18, 0.17]   # transient decays, then the spread stalls
    v = scaling_verdict(CKPTS, sds, scale_factor=1.5, ref_frac=0.25)
    assert v["ok"] is False               # 0.17 vs 0.20/2 = 0.10
    assert v["ratio"] == pytest.approx(1.7, abs=0.05)
    assert v["ratio_first"] < 1.0         # the N/16 reference would have passed


def test_scaling_flags_a_frozen_sector_offset_but_not_a_shrinking_ensemble():
    frozen = scaling_verdict(CKPTS, [0.4] * 5, ref_frac=0.25)
    assert frozen["ok"] is False and frozen["ratio"] == pytest.approx(2.0)
    assert scaling_verdict(CKPTS, _sqrt_n_sds(), ref_frac=0.25)["ok"] is True


# ------------------------------------------------ primary rule: late drift --


def test_late_drift_is_zero_on_a_flat_running_mean():
    v = late_drift_verdict(CKPTS, [3.0] * 5, se_final=0.01)
    assert v["drift"] == pytest.approx(0.0)
    assert v["ok"] is True
    assert v["n_split"] == 100000.0


def test_late_drift_recovers_the_segment_means():
    # segment means 4.0 on (0, N/4], 3.0 on (N/4, N/2], 2.0 on (N/2, N]
    N = 200000.0
    m1 = 4.0
    m2 = (0.25 * 4.0 + 0.25 * 3.0) / 0.5
    mf = (0.25 * 4.0 + 0.25 * 3.0 + 0.5 * 2.0) / 1.0
    v = late_drift_verdict([N / 4, N / 2, N], [m1, m2, mf], se_final=1.0)
    assert v["seg_early"] == pytest.approx(3.0)
    assert v["seg_late"] == pytest.approx(2.0)
    assert v["drift"] == pytest.approx(-1.0)
    assert v["drift_se"] == pytest.approx(math.sqrt(6.0))


def test_late_drift_ignores_a_transient_confined_to_the_first_quarter():
    """The long-initialization case the cumulative test punishes unfairly.

    A chain that starts high and settles by N/4 has a large CUMULATIVE drift but no
    late-segment drift; the burn-in the table quotes has already removed it.
    """
    N = 200000.0
    seg = {0.25: 9.0, 0.5: 2.0, 1.0: 2.0}   # first quarter high, then flat
    cum = {}
    acc = 0.0
    prev = 0.0
    for frac in (0.25, 0.5, 1.0):
        acc += (frac - prev) * seg[frac]
        cum[frac] = acc / frac
        prev = frac
    ns = [N / 4, N / 2, N]
    ms = [cum[0.25], cum[0.5], cum[1.0]]
    assert late_drift_verdict(ns, ms, se_final=0.01)["ok"] is True
    # the cumulative m(N) vs m(N/2) test on the same series is far from passing
    assert abs(ms[-1] - ms[-2]) / 0.01 > 3.0


def test_late_drift_needs_three_checkpoints_and_a_positive_se():
    assert late_drift_verdict(CKPTS[-2:], [3.0, 3.0], 0.01)["ok"] is None
    assert late_drift_verdict(CKPTS, [3.0] * 5, 0.0)["ok"] is None


def test_late_drift_fails_a_still_moving_chain():
    N = 200000.0
    v = late_drift_verdict([N / 4, N / 2, N], [6.0, 5.0, 4.0], se_final=0.01)
    assert v["ok"] is False and v["drift_z"] > 3.0


# ------------------------------------------------ primary rule: multi-start --


def test_multistart_passes_when_the_two_starts_agree():
    v = multistart_verdict(2.0097, 0.0004, 2.0099, 0.0004)
    assert v["ok"] is True and v["ms_z"] < 1.0


def test_multistart_catches_a_frozen_sector_that_every_other_test_misses():
    # UT(3,15) at p=1e-4: short start 2.0090(4), long start frozen at 8.0079(4)
    v = multistart_verdict(8.0079, 0.0004, 2.0090, 0.0004)
    assert v["ok"] is False
    assert v["ms_diff"] == pytest.approx(5.9989, abs=1e-3)
    # ... while its within-ensemble statistics are all healthy
    assert scaling_verdict(CKPTS, _sqrt_n_sds(0.0026), ref_frac=0.25)["ok"] is True
    assert late_drift_verdict(CKPTS, [8.008] * 5, se_final=0.0004)["ok"] is True
    assert rhat_verdict(0.0026, 0.0026, 3.0)["ok"] is True


def test_multistart_is_vacuous_with_a_single_initialization():
    v = multistart_verdict(3.0, 0.01, float("nan"), float("nan"))
    assert v["ok"] is None


# --------------------------------------------------- primary rule: classify --

PBASE = dict(scaling_ok=True, drift_ok=True, ms_ok=True, rhat_ok=True)


def _p(**kw):
    return classify_primary(**{**PBASE, **kw})


def test_classify_primary_is_an_and_of_the_four_tests():
    assert _p() == (True, "primary", "ok")
    for name in ("scaling", "drift", "multistart", "rhat"):
        key = {"multistart": "ms_ok"}.get(name, name + "_ok")
        ok, src, why = _p(**{key: False})
        assert (ok, src, why) == (False, "primary", name)


def test_classify_primary_names_every_failing_test_in_order():
    ok, _, why = _p(scaling_ok=False, ms_ok=False, rhat_ok=False)
    assert (ok, why) == (False, "scaling+multistart+rhat")


def test_classify_primary_treats_an_uncomputable_test_as_passing():
    assert _p(ms_ok=None) == (True, "primary", "ok")
    assert _p(ms_ok=None, scaling_ok=False)[2] == "scaling"


# ------------------------------------------- primary rule: which s_W feeds Rhat --

NAN = float("nan")


def test_select_within_prefers_the_chain_form_by_default():
    """The 2026-09 default: the run's own pooled Gelman-Rubin W decides."""
    assert select_within(0.02, 0.01) == (0.02, "chain")
    assert select_within(0.02, 0.01, "chain") == (0.02, "chain")


def test_select_within_falls_back_to_the_reference_on_a_pre_2026_09_csv():
    """`w_var_within_mean` is absent there, so the chain form is NaN and the row
    keeps its old reference-based verdict instead of losing the test."""
    s, used = select_within(NAN, 0.01, "chain")
    assert (s, used) == (0.01, "ref")


def test_select_within_treats_a_nonpositive_chain_variance_as_absent():
    for bad in (0.0, -1.0, None):
        assert select_within(bad, 0.01, "chain") == (0.01, "ref")


def test_select_within_ref_mode_ignores_an_available_chain_form():
    """--rhat-within ref must reproduce the 2026-08 verdicts bit for bit."""
    assert select_within(0.02, 0.01, "ref") == (0.01, "ref")
    assert select_within(0.02, NAN, "ref")[1] == "none"


def test_select_within_is_none_when_neither_form_exists():
    s, used = select_within(NAN, NAN, "chain")
    assert math.isnan(s) and used == "none"
    # ... and Rhat is then undefined, which classify_primary reads as "not decided"
    assert rhat_verdict(0.02, s, 3.0)["ok"] is None


def test_select_within_reproduces_the_new_bb108_dagger():
    """BB[[108,8,10]] p=1e-3 Z+H long, the one row the switch re-flags.

    s_B / s_W_ref = 2.01 passes the loose cut of 3; the chains' own within
    variance is smaller, so s_B / s_W_chain = 3.48 fails it.
    """
    s_b, s_ref, s_chain = 1.0, 1.0 / 2.01, 1.0 / 3.48
    s_w, used = select_within(s_chain, s_ref, "chain")
    assert used == "chain"
    assert rhat_verdict(s_b, s_w, 3.0)["ok"] is False
    assert rhat_verdict(s_b, select_within(s_chain, s_ref, "ref")[0], 3.0)["ok"] is True


def test_load_references_keys_by_noise(tmp_path):
    from convergence_flags import load_references, lookup_reference
    hdr = ("noise,code_type,d,p,decoder,n_cores,n_target,n_trials,n_faulty,p_L,p_L_se,"
           "ew_mean,ew_se,vw,vw_se,wall_clock,feasible\n")
    (tmp_path / "subset_sampling.csv").write_text(
        hdr + "bit_flip,bb144,0,0.001,bposd,1,1,1,1,1e-6,1e-7,3.0,0.01,0.2,0.01,1,True\n")
    (tmp_path / "subset_sampling_depol.csv").write_text(
        hdr + "depolarizing,bb144,0,0.001,bposd,1,1,1,1,2e-6,1e-7,3.4,0.01,0.3,0.01,1,True\n")
    refs = load_references(str(tmp_path))
    row = {"code_type": "bb144", "d": "0"}
    assert lookup_reference(refs, "bb144", 1e-3, "bit_flip", row)["ew"] == 3.0
    assert lookup_reference(refs, "bb144", 1e-3, "depolarizing", row)["ew"] == 3.4
    assert lookup_reference(refs, "bb144", 1e-2, "depolarizing", row) is None


def test_depol_meanw_dataset_registered():
    from convergence_flags import DATASETS
    ds = {d["name"]: d for d in DATASETS}
    assert ds["depol_meanw"]["csv"] == "depol_meanw_2e6.csv"
    assert ds["depol_meanw"]["noise"] == "depolarizing"
    assert ds["depol_meanw"].get("key_d", True) is False
    assert ds["depol_meanw"]["tables"] == ds["lowp_audit_gr"]["tables"]


def test_new_datasets_registered():
    from convergence_flags import DATASETS

    names = {d["name"]: d for d in DATASETS}
    assert names["lowp_audit_k1"]["csv"] == "lowp_audit_2e6_k1.csv"
    assert names["lowp_audit_k1"]["tables"] == ("tab:audit3",)
    assert names["lowp_bb108_p1e2_4e7"]["csv"] == "lowp_bb108_p1e2_4e7.csv"
    assert names["lowp_bb108_p1e2_4e7"]["tables"] == ("tab:lowp",)
    for name in ("lowp_audit_k1", "lowp_bb108_p1e2_4e7"):
        assert names[name]["key_d"] is False
        assert names[name]["noise"] == "bit_flip"
