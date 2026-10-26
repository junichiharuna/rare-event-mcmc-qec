import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cluster_scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import p_common as PC
from css_code import make_toric_code, make_asymmetric_toric_code


def test_minmax_inits_flat_code():
    # toric lx rows all have weight d -> single uniform init
    code = make_toric_code(3)
    inits = PC.minmax_inits(code)
    assert inits == [("uniform", int(np.argmin([r.sum() for r in code.lx])))]


def test_minmax_inits_spread_code():
    code = make_asymmetric_toric_code(3, 11)
    inits = dict(PC.minmax_inits(code))
    assert set(inits) == {"minw", "maxw"}
    lwts = [int(r.sum()) for r in code.lx]
    assert lwts[inits["minw"]] == min(lwts)
    assert lwts[inits["maxw"]] == max(lwts)


def test_reorder_logicals_puts_requested_row_first():
    code = make_asymmetric_toric_code(3, 11)
    idx = dict(PC.minmax_inits(code))["maxw"]
    re = PC.reorder_logicals(code, idx)
    assert np.array_equal(re.lx[0], code.lx[idx])
    assert re.lx.shape == code.lx.shape


def test_append_rows_and_load_done_roundtrip(tmp_path):
    path = str(tmp_path / "out.csv")
    fields = ["code", "p", "seed", "x"]
    PC.append_rows(path, fields, [{"code": "a", "p": 0.01, "seed": 5, "x": 1.5}])
    PC.append_rows(path, fields, [{"code": "b", "p": 0.001, "seed": 6, "x": 2.5}])
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and rows[0]["code"] == "a"
    done = PC.load_done_keys(path, ("code", "p", "seed"))
    assert ("a", "0.01", "5") in done and ("b", "0.001", "6") in done
    assert PC.load_done_keys(str(tmp_path / "missing.csv"), ("code",)) == set()


def test_mwpm_truth_chunk_sane():
    code = make_toric_code(3)
    nt, nf, s1, s2 = PC.mwpm_truth_chunk(code, p=0.05, m_max=200_000, seed=1)
    assert nt == 200_000 and nf > 0
    ew = s1 / nf
    assert 1.5 < ew < 8.0          # faulty configs have weight >= ~2 on toric d=3
    assert s2 >= s1                # w >= 1 for every faulty sample => sum w^2 >= sum w


def test_persample_truth_chunk_matches_mwpm_stats():
    from css_code import make_mwpm_decoder
    code = make_toric_code(3)
    dec = make_mwpm_decoder(code)
    nt, nf, s1, s2 = PC.persample_truth_chunk(code, dec, p=0.05, m_max=20_000, seed=2)
    assert nt == 20_000 and nf > 0
    assert 1.5 < s1 / nf < 8.0


class _AllFaultyDecoder:
    """Stub: never corrects, so every nonzero error counts as faulty.

    Turns binom_truth_chunk into a pure SAMPLER test: n_faulty must be the number of
    samples with w >= 1 and s1 the total weight, both analytic under iid Bernoulli(p).
    """

    def __init__(self, n):
        self.n = n

    def decode_syndrome(self, syndrome):
        return np.zeros(self.n, dtype=np.int8)

    def is_faulty(self, error, correction):
        return bool(((error ^ correction) % 2).any())


def test_binom_truth_chunk_reproduces_bernoulli_marginals():
    code = make_toric_code(3)          # n = 18
    n, p, m = code.n, 0.01, 100_000
    nt, nf, s1, s2 = PC.binom_truth_chunk(code, _AllFaultyDecoder(n), p, m, seed=7)
    assert nt == m
    # P(w >= 1) = 1 - (1-p)^n and E[sum w] = m n p, both to well within 5 SE
    q = 1.0 - (1.0 - p) ** n
    assert abs(nf - m * q) < 5.0 * np.sqrt(m * q * (1 - q))
    assert abs(s1 - m * n * p) < 5.0 * np.sqrt(m * n * p * (1 - p))
    assert s2 >= s1                    # w >= 1 on every faulty sample


def test_binom_truth_chunk_zero_p_never_faulty():
    code = make_toric_code(3)
    assert PC.binom_truth_chunk(code, _AllFaultyDecoder(code.n), 0.0, 5_000,
                                seed=1) == (5_000, 0, 0.0, 0.0)


def test_binom_truth_chunk_agrees_with_persample_under_a_real_decoder():
    from css_code import make_mwpm_decoder
    code = make_toric_code(3)
    dec = make_mwpm_decoder(code)
    a = PC.binom_truth_chunk(code, dec, p=0.05, m_max=20_000, seed=3)
    b = PC.persample_truth_chunk(code, dec, p=0.05, m_max=20_000, seed=4)
    assert a[0] == b[0] == 20_000
    # same estimand, independent streams: p_L within ~4 SE of each other
    pa, pb = a[1] / a[0], b[1] / b[0]
    se = np.sqrt(pa * (1 - pa) / a[0] + pb * (1 - pb) / b[0])
    assert abs(pa - pb) < 4.0 * se
