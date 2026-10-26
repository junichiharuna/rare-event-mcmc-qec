"""Fast CLI test for cluster_scripts/lowp_audit.py (the low-p cross-code witness audit).

Runs the real driver end to end on the cheapest condition (toric5, MWPM, p=1e-2) at a toy
budget, and checks the three contracts a 6-hour cluster job must not get wrong:
the CSV schema, the flat-basis init handling, and resume.
"""
import csv
import math
import pathlib
import statistics
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "cluster_scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import lowp_audit as LA  # noqa: E402

_SCRIPT = _ROOT / "cluster_scripts" / "lowp_audit.py"


def _run(out_csv, *extra):
    cmd = [sys.executable, str(_SCRIPT),
           "--codes", "toric5", "--p-list", "1e-2", "--methods", "OBF",
           "--inits", "short", "--mcmc-target", "2000", "--n-seeds", "2",
           "--n-cores", "2", "--out-csv", str(out_csv), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT), timeout=300)


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_cli_writes_expected_schema_and_chain_count(tmp_path):
    out = tmp_path / "lowp.csv"
    r = _run(out)
    assert r.returncode == 0, r.stderr

    rows = _rows(out)
    # toric5 has a FLAT logical-weight basis ([5, 5]), so "short" collapses to the single
    # "uniform" sector: one row, not zero (a naive tag filter would emit none).
    assert len(rows) == 1
    row = rows[0]
    assert list(row) == LA.FIELDS
    assert row["code"] == "toric5" and row["method"] == "OBF"
    assert row["init"] == "uniform"
    assert int(row["n_chains"]) == 2
    assert int(row["n_seeds"]) == 2
    assert int(row["n_steps"]) == 2000
    assert int(row["thin_stride"]) == 1          # no striding at the default budget
    for col in ("tau", "tau_max", "ew_burn5", "ew_burn10", "ew_chain_sd",
                "ew_within_se", "accept_rate", "wall_per_chain"):
        assert float(row[col]) == float(row[col])          # not NaN
    assert int(row["n_capped5"]) == int(row["capped5"])
    assert int(row["n_capped10"]) == int(row["capped10"])
    assert 2.0 < float(row["ew_mean"]) < 12.0

    # Per-chain checkpoint CSV: 2 chains x the geometric grid, ending at the full length.
    chains = _rows(tmp_path / "lowp_chains.csv")
    assert list(chains[0]) == LA.CHAIN_FIELDS
    ckpts = LA._checkpoints(2000)
    assert len(chains) == 2 * len(ckpts)
    assert {int(c["checkpoint_n"]) for c in chains} == set(ckpts)
    assert max(int(c["checkpoint_n"]) for c in chains) == 2000
    assert len({c["seed"] for c in chains}) == 2

    conv = _rows(tmp_path / "lowp_conv.csv")
    assert list(conv[0]) == LA.CONV_FIELDS and len(conv) == len(ckpts)

    # Per-seed CSV: the UNAGGREGATED chain summaries, one row per (condition, seed).
    seeds = _rows(tmp_path / "lowp_seeds.csv")
    assert list(seeds[0]) == LA.SEED_FIELDS
    assert len(seeds) == 2
    assert {s["seed"] for s in seeds} == {c["seed"] for c in chains}
    for s in seeds:
        assert s["code"] == "toric5" and s["init"] == "uniform" and s["method"] == "OBF"
        for col in ("ew_full", "tau_full", "ew_burn5", "var_burn5", "tau_burn5",
                    "accept_rate", "wall"):
            assert float(s[col]) == float(s[col])          # not NaN
        assert 0 < int(s["n_used5"]) <= 2000
        assert int(s["capped5"]) in (0, 1)

    # The summary's Gelman-Rubin W is exactly the MEAN of the per-chain burn-5 variances,
    # and the median/mean twins agree with the same reductions over the seeds file.
    var5 = [float(s["var_burn5"]) for s in seeds]
    assert float(row["w_var_within_mean"]) == pytest.approx(sum(var5) / len(var5))
    assert float(row["w_var_within_median"]) == pytest.approx(statistics.median(var5))
    assert float(row["n_used5_mean"]) == pytest.approx(
        statistics.fmean(float(s["n_used5"]) for s in seeds))
    assert float(row["tau_burn5_mean"]) == pytest.approx(
        statistics.fmean(float(s["tau_burn5"]) for s in seeds))


def test_resume_skips_completed_condition(tmp_path):
    out = tmp_path / "lowp.csv"
    assert _run(out).returncode == 0
    before = _rows(out)

    r = _run(out)                                  # second call, same arguments
    assert r.returncode == 0, r.stderr
    assert "skip (already in" in r.stdout
    assert "nothing to do" in r.stdout
    assert _rows(out) == before                    # untouched, not appended or recomputed


def test_inits_for_respects_requested_tags_on_a_spread_basis():
    meta = LA._meta("ut3x11")                      # logical weights [3, 11]
    assert [t for t, _i in LA._inits_for(meta, ["short", "long"])] == ["short", "long"]
    assert [t for t, _i in LA._inits_for(meta, ["long"])] == ["long"]
    assert [t for t, _i in LA._inits_for(meta, ["maxw"])] == ["long"]
    # A flat basis has one sector however many inits are asked for.
    assert LA._inits_for(LA._meta("toric5"), ["short", "long"]) == [("uniform", 0)]


def test_truth_lookup_is_total_when_the_reference_is_missing():
    assert LA._load_truth("no/such/file.csv") == {}
    assert LA._truth_for({}, "bb144", 1e-3)[3] == "missing"
    truth = {("bb144", "0", "0.001"): {"ew_mean": "3.07", "ew_se": "0.005", "p_L": "5.8e-8"}}
    ew, se, p_l, src = LA._truth_for(truth, "bb144", 1e-3)
    assert (ew, se) == (3.07, 0.005) and p_l == 5.8e-8 and src.startswith("subset_sampling")
    # A code absent from the reference must not raise; it gets NaN and is still written.
    assert LA._truth_for(truth, "hgp_hamming3", 1e-3)[3] == "missing"


def _fake_stats(triples):
    """[(var5, tau5, n_used5)] -> the minimal _chain_stats shape _gr_cols/_burn_cols read.

    Burn tuple layout (cluster_scripts/lowp_audit.py:_chain_stats):
    (mean, var, tau, capped, n_used). c=10 is filled with the same numbers; only c=5 is
    read by _gr_cols and by the within-chain SE of the summary.
    """
    return [{"burn": {c: (0.0, v, t, False, n) for c in LA._BURN_CS}}
            for (v, t, n) in triples]


def _median_form_tau1(stats):
    """ew_within_se_tau1 recomputed exactly as the summary block of main() does it:
    the MEDIAN over chains of sqrt(var * 2 tau / n_used), divided by sqrt(2)."""
    within = [math.sqrt(s["burn"][5][1] * 2.0 * max(s["burn"][5][2], 1.0)
                        / max(s["burn"][5][4], 1)) for s in stats]
    return statistics.median(within) / math.sqrt(2.0)


def _pooled_form(stats, gr):
    """s_W as compare/convergence_flags.py will form it: sqrt(W * tau_med / n_med), with
    the MEDIAN tau_burn5 / n_used5 the summary already carries (via _burn_cols)."""
    b = LA._burn_cols(stats)
    return math.sqrt(gr["w_var_within_mean"] * max(b["tau_burn5"], 1.0)
                     / max(b["n_used5"], 1))


def test_gr_cols_aggregates_the_per_chain_burn5_ingredients():
    stats = _fake_stats([(2.0, 10.0, 1000), (8.0, 20.0, 2000), (2.0, 30.0, 3000)])
    gr = LA._gr_cols(stats)
    assert gr["w_var_within_mean"] == pytest.approx(4.0)        # (2 + 8 + 2) / 3
    assert gr["w_var_within_median"] == pytest.approx(2.0)      # median(2, 2, 8)
    assert gr["n_used5_mean"] == pytest.approx(2000.0)
    assert gr["tau_burn5_mean"] == pytest.approx(20.0)
    # ...and W is genuinely the mean, not the median, when the variances are heterogeneous.
    assert gr["w_var_within_mean"] != pytest.approx(gr["w_var_within_median"])


def test_pooled_W_differs_from_the_median_form_only_when_variances_are_heterogeneous():
    # Heterogeneous variances, tau/n identical so ONLY the variance aggregation can differ:
    # sqrt is concave, so median(sqrt(var)) != sqrt(mean(var)) (Jensen).
    het = _fake_stats([(1.0, 10.0, 1000), (1.0, 10.0, 1000), (25.0, 10.0, 1000)])
    assert _pooled_form(het, LA._gr_cols(het)) != pytest.approx(_median_form_tau1(het))
    assert _pooled_form(het, LA._gr_cols(het)) > _median_form_tau1(het)

    # All three factors identical across chains -> mean == median chainwise and the two
    # forms coincide exactly. (tau >= 1 matters: the summary clamps max(tau, 1.0).)
    hom = _fake_stats([(4.0, 10.0, 1000)] * 3)
    assert _pooled_form(hom, LA._gr_cols(hom)) == pytest.approx(_median_form_tau1(hom))


def test_tau_striding_is_inert_at_the_default_budget():
    assert LA._stride(200_000) == 1                # p=1e-2/1e-3/1e-4 audit rows
    assert LA._stride(2_000_000) == 10             # the bb144 long run


def test_load_truth_filters_noise(tmp_path):
    # Review focus 2: a depolarizing run must never read bit-flip reference rows.
    csv_path = tmp_path / "ref.csv"
    csv_path.write_text(
        "noise,code_type,d,p,decoder,n_cores,n_target,n_trials,n_faulty,p_L,p_L_se,"
        "ew_mean,ew_se,vw,vw_se,wall_clock,feasible\n"
        "bit_flip,bb144,0,0.0001,bposd,1,1,1,1,1e-9,1e-10,3.0,0.01,0.1,0.01,1,True\n"
        "depolarizing,bb144,0,0.0001,bposd,1,1,1,1,2e-9,1e-10,3.5,0.01,0.1,0.01,1,True\n")
    bf = LA._load_truth(str(csv_path))                      # default: bit_flip
    dp = LA._load_truth(str(csv_path), noise="depolarizing")
    assert float(bf[("bb144", "0", "0.0001")]["ew_mean"]) == 3.0
    assert float(dp[("bb144", "0", "0.0001")]["ew_mean"]) == 3.5
    assert LA._truth_for(dp, "bb144", 1e-4)[0] == 3.5


def test_cli_depolarizing_runs_toric5(tmp_path):
    out = tmp_path / "depol.csv"
    r = _run(out, "--noise", "depolarizing")
    assert r.returncode == 0, r.stderr
    rows = _rows(out)
    assert len(rows) == 1 and rows[0]["init"] == "uniform"
    assert rows[0]["h_accept_rate"] == "nan"
    assert 2.0 < float(rows[0]["ew_mean"]) < 12.0
    seeds = _rows(tmp_path / "depol_seeds.csv")
    assert len(seeds) == 2


def test_k1_codes_registered_with_lookup_or_mwpm():
    assert LA.CODES["surface3"][1:] == ("mwpm", "surface", ("3",))
    assert LA.CODES["concat_steane2"][1:] == ("lookup", "concat_steane", ("2",))


@pytest.mark.parametrize("name", ["surface3", "concat_steane2"])
def test_k1_codes_single_uniform_init(name):
    meta = LA._meta(name)
    assert meta["k"] == 1
    assert LA._inits_for(meta, ["short", "long"]) == [("uniform", 0)]


def test_lookup_decoder_matches_paper_codes_and_rejects_depolarizing():
    import paper_codes as PC

    code = LA.build_code("concat_steane2")
    dec = LA.build_decoder("concat_steane2", code, 1e-3)
    _, paper_dec = PC.build("concat_steane2", "bit_flip", 1e-3)
    assert type(dec).__name__ == type(paper_dec).__name__
    with pytest.raises(ValueError, match="lookup decoder has no depolarizing variant"):
        LA.build_decoder("concat_steane2", code, 1e-3, noise="depolarizing")
