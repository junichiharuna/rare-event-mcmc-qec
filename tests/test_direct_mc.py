import math
import os
import sys

import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "compare"))


def test_exact_ew_d3_reference():
    from exact_ew import exact_ew_d3

    ew, vw, z = exact_ew_d3(1e-2)
    assert abs(ew - 2.2610) < 1e-3
    assert abs(vw - 0.2150) < 1e-3
    assert 0.0 < z < 1.0


def test_estimate_matches_numpy():
    from direct_mc import Stats, estimate

    # 5 faulty samples with known weights, out of M=20 trials
    ws = np.array([2, 3, 2, 4, 3], dtype=float)
    F = len(ws)
    M = 20
    st = Stats(
        n_trials=M,
        n_faulty=F,
        s1=float(ws.sum()),
        s2=float((ws**2).sum()),
        s3=float((ws**3).sum()),
        s4=float((ws**4).sum()),
    )
    est = estimate(st)

    assert est["p_L"] == pytest.approx(F / M)
    assert est["p_L_se"] == pytest.approx(math.sqrt((F / M) * (1 - F / M) / M))
    assert est["ew_mean"] == pytest.approx(ws.mean())
    assert est["vw"] == pytest.approx(ws.var(ddof=1))  # Bessel-corrected
    assert est["ew_se"] == pytest.approx(math.sqrt(ws.var(ddof=1) / F))


def test_pool_stats_sums_fields():
    from direct_mc import Stats, pool_stats

    a = Stats(10, 3, 6.0, 14.0, 36.0, 98.0)
    b = Stats(5, 2, 5.0, 13.0, 35.0, 97.0)
    p = pool_stats([a, b])
    assert p == Stats(15, 5, 11.0, 27.0, 71.0, 195.0)


def test_estimate_handles_no_faulty():
    from direct_mc import Stats, estimate

    est = estimate(Stats(100, 0, 0.0, 0.0, 0.0, 0.0))
    assert est["p_L"] == pytest.approx(0.0)
    assert math.isnan(est["ew_mean"])
    assert math.isnan(est["vw"])


class _FakeAdapter:
    """Deterministic adapter yielding a fixed (faulty, w) sequence, cycled."""

    def __init__(self, seq):
        self.seq = seq
        self.i = 0

    def sample(self, rng):
        return None

    def decode_and_check(self, _state):
        item = self.seq[self.i % len(self.seq)]
        self.i += 1
        return item


def test_run_core_stops_at_target_and_counts():
    from direct_mc import run_core

    # pattern: faulty(w=2), not, faulty(w=4)  -> 2 faulty per 3 trials
    seq = [(True, 2), (False, 0), (True, 4)]
    st = run_core(_FakeAdapter(seq), rng=None, target=4, m_max=10_000)
    assert st.n_faulty == 4
    assert st.n_trials == 6  # 2 faulty every 3 trials
    assert st.s1 == 2 + 4 + 2 + 4
    assert st.s2 == 4 + 16 + 4 + 16


def test_run_core_respects_m_max():
    from direct_mc import run_core

    seq = [(False, 0)]  # never faulty
    st = run_core(_FakeAdapter(seq), rng=None, target=5, m_max=50)
    assert st.n_trials == 50
    assert st.n_faulty == 0


class _PassThroughDecoder:
    """Bit-flip decoder stub: correction = 0, faulty iff weight is odd.

    Records how many real decode calls happen so we can check caching.
    """

    def __init__(self, n):
        self.n = n
        self.calls = 0

    def decode_syndrome(self, s):
        self.calls += 1
        return np.zeros(self.n, dtype=np.int8)

    def is_faulty(self, error, correction):
        return bool(int(error.sum()) % 2 == 1)


def test_bitflip_adapter_weight_and_faulty():
    from css_code import make_toric_code
    from direct_mc import BitFlipAdapter

    code = make_toric_code(3)
    dec = _PassThroughDecoder(code.n)
    ad = BitFlipAdapter(code, dec, p=0.2, use_cache=True)
    rng = np.random.default_rng(0)
    e = ad.sample(rng)
    faulty, w = ad.decode_and_check(e)
    assert w == int(e.sum())
    assert faulty == (w % 2 == 1)


def test_bitflip_adapter_cache_avoids_redecode():
    from css_code import make_toric_code
    from direct_mc import BitFlipAdapter

    code = make_toric_code(3)
    dec = _PassThroughDecoder(code.n)
    ad = BitFlipAdapter(code, dec, p=0.2, use_cache=True)
    e = np.zeros(code.n, dtype=np.int8)
    e[0] = 1  # fixed NON-zero error -> non-trivial syndrome (zero error short-circuits decode)
    for _ in range(5):
        ad.decode_and_check(e)
    assert dec.calls == 1  # same syndrome -> decoded once, then cached


def test_make_code_and_decoder_selects_decoder():
    from direct_mc import _make_code_and_decoder

    code, dec = _make_code_and_decoder("bit_flip", "toric", 3, 1e-2)
    assert hasattr(dec, "decode_syndrome") and hasattr(dec, "is_faulty")
    code, dec = _make_code_and_decoder("depolarizing", "toric", 3, 1e-2)
    assert hasattr(dec, "decode_x") and hasattr(dec, "decode_z")


@pytest.mark.slow
def test_run_condition_d3_matches_exact():
    """End-to-end: direct MC on d=3 toric bit-flip agrees with exact reference."""
    from exact_ew import exact_ew_d3
    from direct_mc import run_condition

    ew_x, vw_x, z_x = exact_ew_d3(1e-2)
    row = run_condition(
        "bit_flip",
        "toric",
        3,
        1e-2,
        n_target=1000,
        m_max=2_000_000,
        n_cores=1,
        base_seed=7,
    )
    assert row["feasible"] is True
    assert row["n_faulty"] >= 1000
    assert abs(row["p_L"] - z_x) < 5 * row["p_L_se"]
    assert abs(row["ew_mean"] - ew_x) < 5 * row["ew_se"]
    assert abs(row["vw"] - vw_x) < 5 * row["vw_se"]


@pytest.mark.slow
def test_run_condition_cache_consistency():
    from direct_mc import run_condition

    common = dict(
        noise="bit_flip",
        code_type="toric",
        d=3,
        p=1e-2,
        n_target=300,
        m_max=1_000_000,
        n_cores=1,
        base_seed=11,
    )
    a = run_condition(use_cache=True, **common)
    b = run_condition(use_cache=False, **common)
    assert a["n_trials"] == b["n_trials"]
    assert a["n_faulty"] == b["n_faulty"]
    assert a["ew_mean"] == pytest.approx(b["ew_mean"])
    assert a["vw"] == pytest.approx(b["vw"])


@pytest.mark.slow
def test_run_condition_reproducible():
    from direct_mc import run_condition

    common = dict(
        noise="bit_flip",
        code_type="toric",
        d=3,
        p=1e-2,
        n_target=200,
        m_max=1_000_000,
        n_cores=1,
        base_seed=3,
    )
    a = run_condition(**common)
    b = run_condition(**common)
    assert a["n_trials"] == b["n_trials"]
    assert a["ew_mean"] == pytest.approx(b["ew_mean"])


def test_run_condition_infeasible_is_graceful():
    from direct_mc import run_condition

    row = run_condition(
        "bit_flip",
        "toric",
        3,
        1e-2,
        n_target=10_000_000,
        m_max=200,
        n_cores=1,
        base_seed=1,
    )
    assert row["feasible"] is False
    assert row["n_trials"] == 200


def test_run_condition_depolarizing_smoke():
    from direct_mc import run_condition

    row = run_condition(
        "depolarizing",
        "toric",
        3,
        0.05,
        n_target=200,
        m_max=1_000_000,
        n_cores=1,
        base_seed=5,
    )
    assert 0.0 <= row["p_L"] <= 1.0
    assert row["ew_mean"] >= 0.0
    assert row["decoder"] == "mwpm"


@pytest.mark.slow
def test_run_condition_pool_path_runs():
    from direct_mc import run_condition

    row = run_condition(
        "bit_flip",
        "toric",
        3,
        1e-2,
        n_target=400,
        m_max=2_000_000,
        n_cores=2,
        base_seed=9,
    )
    assert row["n_cores"] == 2
    assert row["n_faulty"] >= 400


@pytest.mark.slow
def test_local_driver_cli_writes_csv(tmp_path):
    import subprocess

    out = tmp_path / "dmc.csv"
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(BASE, "compare", "direct_mc_ew.py"),
            "--noise",
            "bit_flip",
            "--code-type",
            "toric",
            "--code-size",
            "3",
            "--p",
            "1e-2",
            "--n-target",
            "200",
            "--m-max",
            "1000000",
            "--n-cores",
            "1",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=BASE,
    )
    assert r.returncode == 0, r.stderr
    assert out.exists()
    import csv as _csv

    rows = list(_csv.DictReader(open(out)))
    assert len(rows) == 1
    assert rows[0]["code_type"] == "toric"
    assert int(rows[0]["n_faulty"]) >= 200


def test_faulty_weight_counts_matches_exact_d3():
    # The general enumerator must reproduce the known toric d=3 exact values.
    from css_code import make_toric_code, make_mwpm_decoder
    from exact_ew import faulty_weight_counts, exact_ew_from_counts

    code = make_toric_code(3)
    dec = make_mwpm_decoder(code)
    c_faulty, n, uniq, corr = faulty_weight_counts(code, dec)
    ew, vw, z = exact_ew_from_counts(c_faulty, n, 1e-2)
    assert abs(ew - 2.2610) < 1e-3
    assert abs(vw - 0.2150) < 1e-3
    assert corr.shape == (len(uniq), n)
    assert 0.0 < z < 1.0


def test_binomial_sampler_marginal_count_matches_p():
    from css_code import make_toric_code
    from direct_mc import BitFlipAdapter

    code = make_toric_code(5)
    ad = BitFlipAdapter(
        code, _PassThroughDecoder(code.n), p=0.1, use_cache=False, sampler="binomial"
    )
    rng = np.random.default_rng(0)
    counts = [int(ad.sample(rng).sum()) for _ in range(2000)]
    # mean count ~ n*p = 50*0.1 = 5; check within a few SE
    assert abs(np.mean(counts) - 5.0) < 0.5


def test_binomial_and_bernoulli_agree_distributionally():
    from css_code import make_toric_code
    from direct_mc import BitFlipAdapter

    code = make_toric_code(5)
    # Same RNG seed won't give identical draws (different algorithms), but the
    # faulty fraction and mean weight must agree within sampling error.
    ad_bern = BitFlipAdapter(
        code, _PassThroughDecoder(code.n), p=0.1, use_cache=False, sampler="bernoulli"
    )
    ad_bino = BitFlipAdapter(
        code, _PassThroughDecoder(code.n), p=0.1, use_cache=False, sampler="binomial"
    )
    rng = np.random.default_rng(1)
    w_bern = [int(ad_bern.sample(rng).sum()) for _ in range(3000)]
    rng = np.random.default_rng(1)
    w_bino = [int(ad_bino.sample(rng).sum()) for _ in range(3000)]
    assert abs(np.mean(w_bern) - np.mean(w_bino)) < 0.3


@pytest.mark.slow
def test_run_condition_binomial_matches_exact_d3():
    from exact_ew import exact_ew_d3
    from direct_mc import run_condition

    ew_x, vw_x, z_x = exact_ew_d3(1e-2)
    row = run_condition(
        "bit_flip",
        "toric",
        3,
        1e-2,
        n_target=1500,
        m_max=5_000_000,
        n_cores=1,
        base_seed=4,
        sampler="binomial",
    )
    assert row["feasible"] is True
    assert abs(row["p_L"] - z_x) < 5 * row["p_L_se"]
    assert abs(row["ew_mean"] - ew_x) < 5 * row["ew_se"]


def stats_fields(st):
    return {
        "n_trials": st.n_trials,
        "n_faulty": st.n_faulty,
        "s1": st.s1,
        "s2": st.s2,
        "s3": st.s3,
        "s4": st.s4,
    }


def test_stats_jsonl_roundtrip(tmp_path):
    from direct_mc import Stats, stats_to_dict, dict_to_stats

    st = Stats(1000, 7, 21.0, 71.0, 255.0, 953.0)
    d = stats_to_dict(
        st, meta={"noise": "bit_flip", "code_type": "bb144", "d": 12, "p": 1e-3}
    )
    assert d["n_trials"] == 1000 and d["n_faulty"] == 7
    assert d["p"] == 1e-3 and d["code_type"] == "bb144"
    st2 = dict_to_stats(d)
    assert st2 == st


def test_write_read_stats_jsonl(tmp_path):
    from direct_mc import Stats, write_stats_jsonl, read_stats_jsonl

    recs = [
        {
            "noise": "bit_flip",
            "code_type": "bb144",
            "d": 12,
            "p": 1e-3,
            **stats_fields(Stats(500, 3, 9.0, 29.0, 99.0, 353.0)),
        },
        {
            "noise": "bit_flip",
            "code_type": "bb144",
            "d": 12,
            "p": 1e-3,
            **stats_fields(Stats(400, 2, 6.0, 20.0, 72.0, 272.0)),
        },
    ]
    path = tmp_path / "stats.jsonl"
    write_stats_jsonl(recs, str(path))
    got = read_stats_jsonl(str(path))
    assert len(got) == 2
    assert got[0]["n_trials"] == 500 and got[1]["n_faulty"] == 2


@pytest.mark.slow
def test_run_condition_stats_poolable(tmp_path):
    from direct_mc import run_condition_stats, dict_to_stats, estimate

    rec = run_condition_stats(
        "bit_flip",
        "toric",
        3,
        1e-2,
        n_target=500,
        m_max=2_000_000,
        n_cores=2,
        base_seed=8,
        sampler="binomial",
    )
    # returns a flat record with metadata + pooled sufficient statistics
    assert rec["noise"] == "bit_flip" and rec["code_type"] == "toric"
    assert rec["d"] == 3 and rec["p"] == 1e-2
    assert rec["n_faulty"] >= 500
    # the pooled stats must be self-consistent through estimate()
    est = estimate(dict_to_stats(rec))
    assert 0.0 < est["p_L"] < 1.0
    assert est["ew_mean"] > 0.0


@pytest.mark.slow
def test_sweep_stats_cli_writes_jsonl(tmp_path):
    import subprocess

    out = tmp_path / "stats_sub.jsonl"
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(BASE, "cluster_scripts", "sweep_direct_mc_stats.py"),
            "--noise",
            "bit_flip",
            "--code-type",
            "toric",
            "--code-size",
            "3",
            "--p",
            "1e-2",
            "--n-target",
            "100",
            "--m-max",
            "1000000",
            "--n-cores",
            "1",
            "--base-seed",
            "5",
            "--sampler",
            "binomial",
            "--out-jsonl",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=BASE,
    )
    assert r.returncode == 0, r.stderr
    recs = __import__("json").loads
    lines = open(out).read().strip().splitlines()
    assert len(lines) == 1
    rec = recs(lines[0])
    assert rec["noise"] == "bit_flip" and rec["n_faulty"] >= 1
    assert rec["code_type"] == "toric" and rec["d"] == 3


def test_collate_stats_pools_by_condition(tmp_path):
    import sys as _sys

    _sys.path.insert(0, os.path.join(BASE, "cluster_scripts"))
    from merge_direct_mc_stats import collate_stats
    from direct_mc import write_stats_jsonl

    # two shards of the SAME condition -> must pool into one row
    write_stats_jsonl(
        [
            {
                "noise": "bit_flip",
                "code_type": "bb144",
                "d": 12,
                "p": 1e-3,
                "n_trials": 1000,
                "n_faulty": 4,
                "s1": 12.0,
                "s2": 40.0,
                "s3": 140.0,
                "s4": 520.0,
            },
        ],
        str(tmp_path / "a.jsonl"),
    )
    write_stats_jsonl(
        [
            {
                "noise": "bit_flip",
                "code_type": "bb144",
                "d": 12,
                "p": 1e-3,
                "n_trials": 2000,
                "n_faulty": 6,
                "s1": 18.0,
                "s2": 60.0,
                "s3": 210.0,
                "s4": 780.0,
            },
        ],
        str(tmp_path / "b.jsonl"),
    )

    rows = collate_stats(str(tmp_path / "*.jsonl"))
    assert len(rows) == 1
    row = rows[0]
    assert row["n_trials"] == 3000 and row["n_faulty"] == 10
    # pooled p_L = 10/3000
    assert abs(row["p_L"] - 10 / 3000) < 1e-12
    assert row["ew_mean"] == (12.0 + 18.0) / 10
    assert row["feasible"] is True


def test_bb144_lowp_shell_script_is_valid(tmp_path):
    import subprocess, shutil

    script = os.path.join(BASE, "cluster_scripts", "direct_mc_bb144_lowp.sh")
    assert os.path.exists(script)
    bash = shutil.which("bash")
    if bash:  # bash -n syntax check when available; relative path + cwd so the
        # check also works when `bash` resolves to WSL (backslash paths break there)
        r = subprocess.run([bash, "-n", "cluster_scripts/direct_mc_bb144_lowp.sh"],
                           capture_output=True, text=True, cwd=BASE)
        assert r.returncode == 0, r.stderr
    body = open(script).read()
    # must declare a PBS array and contain the pooling-merge guard
    assert "#PBS -J" in body
    assert "merge_direct_mc_stats.py" in body
    assert "ntfy" in body.lower()


def test_run_condition_decoder_kind_bp():
    from direct_mc import run_condition
    row = run_condition(
        "bit_flip", "toric", 3, 0.08,
        n_target=20, m_max=200_000, n_cores=1, decoder_kind="bp",
    )
    assert row["decoder"] == "bp"
    assert row["n_faulty"] >= 1          # plain BP fails plenty at p=0.08, d=3
    assert 0.0 < row["p_L"] <= 1.0


def test_bitflip_adapter_zero_error_shortcircuit():
    from css_code import make_toric_code, make_mwpm_decoder
    from direct_mc import BitFlipAdapter
    code = make_toric_code(3)
    ad = BitFlipAdapter(code, make_mwpm_decoder(code), 0.1, True, "binomial")
    z = np.zeros(code.n, dtype=np.int8)
    assert ad.decode_and_check(z) == (False, 0)
