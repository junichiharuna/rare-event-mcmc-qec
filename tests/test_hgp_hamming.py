import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from css_code import make_hgp_hamming  # noqa: E402


def test_hgp_hamming_shape_and_k():
    code = make_hgp_hamming()
    assert code.n == 58                     # 7*7 + 3*3
    assert code.lx.shape[0] == 16           # k = 4^2 + 0^2
    assert code.lz.shape[0] == 16
    assert code.lx.shape[1] == 58
    assert code.hz.shape[1] == 58


def test_hgp_hamming_is_valid_css():
    code = make_hgp_hamming()
    # CSS commutation: every X-stabilizer commutes with every Z-stabilizer.
    assert not ((code.hx @ code.hz.T) % 2).any()
    # logicals commute with the opposite-type stabilizers.
    assert not ((code.lx @ code.hz.T) % 2).any()
    assert not ((code.lz @ code.hx.T) % 2).any()
