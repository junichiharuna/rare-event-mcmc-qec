"""Synthetic bit-flip audit rows and the pre-k=1 table-body snapshot."""
import argparse
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "compare"))
import build_audit_tables as B  # noqa: E402


LEGACY_CODES = [
    ("toric5", "5", ["uniform"]), ("toric7", "7", ["uniform"]),
    ("surface5", "5", ["uniform"]), ("surface7", "7", ["uniform"]),
    ("hgp_hamming3", "0", ["short", "long"]),
    ("ut3x11", "3x11", ["short", "long"]), ("ut3x15", "3x15", ["short", "long"]),
    ("bb144", "0", ["short", "long"]), ("bb_l9m6", "0", ["short", "long"]),
    ("bb72", "0", ["short", "long"]),
]
K1_CODES = [("surface3", "3", ["uniform"]), ("concat_steane2", "2", ["uniform"])]
PS = [(1e-4, r"10^{-4}")]


def _entries(dataset, codes, truth=3.0):
    return [
        dict(dataset=dataset, code=code, d=d, p=1e-4, init=ini, method=method,
             bias=0.5 if method == "OBF" else 0.0, bias_se=0.01,
             truth=truth, truth_se=0.001, converged=True)
        for code, d, inits in codes for ini in inits for method in ("OBF", "Z+H")
    ]


@pytest.fixture
def args():
    return argparse.Namespace(
        scale_factor=1.5, scale_ref_frac=0.25, drift_k=3.0, ms_k=3.0,
        chain_rhat_max=3.0, rhat_within="chain", bold_abs_min=0.005, p_only=None,
    )


@pytest.fixture
def idx_legacy():
    return B.index(_entries(B.LOWP_DATASET, LEGACY_CODES))


@pytest.fixture
def idx_with_k1():
    # Decoy rows expose an accidental fallback to the old dataset for k=1 codes.
    return B.index(_entries(B.LOWP_DATASET, LEGACY_CODES)
                   + _entries("lowp_audit_k1", K1_CODES, truth=2.0)
                   + _entries(B.LOWP_DATASET, K1_CODES, truth=99.0))


# Captured once from HEAD f37aa3d, before the k=1 rows were added.
EXPECTED_BEFORE = r"""toric $d=5$                      & $0$  & $3.0000(10)$ & \multicolumn{2}{c}{$\mathbf{+0.500(10)}$} & \multicolumn{2}{c}{$0.000(10)$} \\
toric $d=7$                      & $0$  & $3.0000(10)$ & \multicolumn{2}{c}{$\mathbf{+0.500(10)}$} & \multicolumn{2}{c}{$0.000(10)$} \\
surface $d=5$                    & $0$  & $3.0000(10)$ & \multicolumn{2}{c}{$\mathbf{+0.500(10)}$} & \multicolumn{2}{c}{$0.000(10)$} \\
surface $d=7$                    & $0$  & $3.0000(10)$ & \multicolumn{2}{c}{$\mathbf{+0.500(10)}$} & \multicolumn{2}{c}{$0.000(10)$} \\
\midrule
HGP(Hamming) $[[58,16,3]]$       & $1$  & $3.0000(10)$ & $\mathbf{+0.500(10)}$ & $\mathbf{+0.500(10)}$ & $0.000(10)$ & $0.000(10)$ \\
\midrule
$\mathrm{AT}(3,11)$              & $8$  & $3.0000(10)$ & $\mathbf{+0.500(10)}$ & $\mathbf{+0.500(10)}$ & $0.000(10)$ & $0.000(10)$ \\
$\mathrm{AT}(3,15)$              & $12$ & $3.0000(10)$ & $\mathbf{+0.500(10)}$ & $\mathbf{+0.500(10)}$ & $0.000(10)$ & $0.000(10)$ \\
\midrule
BB $[[144,12,12]]$               & $12$ & $3.0000(10)$ & $\mathbf{+0.500(10)}$ & $\mathbf{+0.500(10)}$ & $0.000(10)$ & $0.000(10)$ \\
BB $[[108,8,10]]$                & $18$ & $3.0000(10)$ & $\mathbf{+0.500(10)}$ & $\mathbf{+0.500(10)}$ & $0.000(10)$ & $0.000(10)$ \\
BB $[[72,12,6]]$                 & $8$  & $3.0000(10)$ & $\mathbf{+0.500(10)}$ & $\mathbf{+0.500(10)}$ & $0.000(10)$ & $0.000(10)$ \\
"""


def test_audit3_body_has_twelve_codes_and_k1_rows_use_their_dataset(idx_with_k1, args):
    body = B._audit3_body(idx_with_k1, args, ps=PS)
    assert body.count(r"\\") == 12
    assert body.count(r"\multicolumn{2}{c}") == 2 * (4 + 2)
    assert r"surface $d=3$" in body and r"concat.\ Steane ($\ell=2$)" in body
    assert B.AUDIT3_GROUPS[0][0] == "single-logical codes"
    lines = body.splitlines()
    assert lines[0].startswith(r"surface $d=3$")
    assert lines[1].startswith(r"concat.\ Steane ($\ell=2$)")
    assert all("$2.0000(10)$" in line for line in lines[:2])
    assert "99.000" not in body
    assert B.build_audit_1e4(idx_with_k1, args).endswith(body)


def test_audit3_body_without_dataset_field_is_unchanged(idx_legacy, args):
    codes = {code for code, _, _ in LEGACY_CODES}
    assert all(len(row) == 5 for _, rows in B.AUDIT3_GROUPS
               for row in rows if row[1] in codes)
    assert B._audit3_body(idx_legacy, args, codes=codes, ps=PS) == EXPECTED_BEFORE


def test_audit3_body_legacy_rows_honor_explicit_dataset(args):
    codes = {code for code, _, _ in LEGACY_CODES}
    idx = B.index(_entries("custom", LEGACY_CODES))
    assert B._audit3_body(idx, args, codes=codes, ps=PS,
                          dataset="custom") == EXPECTED_BEFORE


def test_depol_meanw_keeps_ten_codes_even_when_k1_entries_are_present(args):
    idx = B.index(_entries("depol_meanw", LEGACY_CODES)
                  + _entries("lowp_audit_k1", K1_CODES, truth=99.0))
    body = B.build_depol_meanw(idx, args)
    assert body.endswith(EXPECTED_BEFORE)
    assert body.count(r"\\") == 10
    assert r"surface $d=3$" not in body and r"concat.\ Steane" not in body


def test_audit3_header_records_both_production_sources(idx_with_k1, args):
    header = B.build_audit_1e4(idx_with_k1, args).splitlines()[1]
    assert "lowp_audit_2e6_gr.csv" in header
    assert "lowp_audit_2e6_k1.csv" in header

