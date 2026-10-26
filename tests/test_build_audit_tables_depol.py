# tests/test_build_audit_tables_depol.py
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compare"))
import build_audit_tables as B  # noqa: E402


def _entry(dataset, code, d, p, init, method, bias, truth=3.0):
    return dict(dataset=dataset, code=code, d=d, p=p, init=init, method=method,
                bias=bias, bias_se=0.01, truth=truth, truth_se=0.001, converged=True)


def _args():
    return argparse.Namespace(scale_factor=1.5, scale_ref_frac=0.25, drift_k=3.0, ms_k=3.0,
                              chain_rhat_max=3.0, rhat_within="chain", bold_abs_min=0.005,
                              rhat_max=None, rhat_alpha=1e-3, tau_factor=1.0, drift_mode="cum",
                              cap_frac=0.1, n_tau=50.0, p_only=None)


def test_depol_meanw_body_reads_the_depol_dataset():
    entries = []
    for code, d, inits in [("toric5", "5", ["uniform"]), ("toric7", "7", ["uniform"]),
                           ("surface5", "5", ["uniform"]), ("surface7", "7", ["uniform"]),
                           ("hgp_hamming3", "0", ["short", "long"]),
                           ("ut3x11", "3x11", ["short", "long"]), ("ut3x15", "3x15", ["short", "long"]),
                           ("bb144", "0", ["short", "long"]), ("bb_l9m6", "0", ["short", "long"]),
                           ("bb72", "0", ["short", "long"])]:
        for ini in inits:
            for m in ("OBF", "Z+H"):
                entries.append(_entry("depol_meanw", code, d, 1e-4, ini, m, 0.5 if m == "OBF" else 0.0))
    body = B.build_depol_meanw(B.index(entries), _args())
    assert "depol_meanw" in body.splitlines()[1] or "depolarizing" in body
    assert body.count(r"\\") == 10 and r"\mathbf{+0.500(10)}" in body   # fmt_pm: 3 decimals when se < 0.05


def test_depol_lowp_body_six_rows():
    entries = []
    for code in ("bb144", "bb_l9m6"):
        for p in (1e-2, 1e-3, 1e-4):
            for ini in ("short", "long"):
                for m in ("OBF", "Z+H"):
                    entries.append(_entry("depol_meanw", code, "0", p, ini, m, 0.0))
    body = B.build_depol_lowp(B.index(entries), _args())
    assert body.count(r"\\") == 6 and "BB $[[108,8,10]]$" in body
