"""Analytic hard-decision logical-error recursion for the concatenated Steane code.

P_L(level L, p) = Q^L(p), where Q is the exact single-block Steane logical-error
probability under i.i.d. bit-flip noise with the Hamming lookup decoder. This is a
sampling-free truth anchor for auditing rare-event splitting (it reaches arbitrarily
low p, where direct MC is infeasible).
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from css_code import make_steane_code, _steane_lookup  # noqa: E402

_CODE = make_steane_code()
# Precompute, per error pattern, (weight, is_logical) once.
_PATTERNS = []
for _bits in itertools.product([0, 1], repeat=7):
    _e = np.array(_bits, dtype=np.int8)
    _w = int(_e.sum())
    _synd = (_CODE.hz @ _e) % 2
    _corr = _steane_lookup(_synd)
    _res = (_e ^ _corr) % 2
    _is_log = bool(((_CODE.lz @ _res) % 2).any())
    _PATTERNS.append((_w, _is_log))


def single_block_logical_prob(x: float) -> float:
    """Exact logical-error probability Q(x) of one Steane block at bit-flip prob x."""
    total = 0.0
    for w, is_log in _PATTERNS:
        if is_log:
            total += (x ** w) * ((1.0 - x) ** (7 - w))
    return total


def concat_pl(p: float, levels: int) -> float:
    """P_L for the level-`levels` concatenated Steane code at physical bit-flip prob p."""
    q = p
    for _ in range(levels):
        q = single_block_logical_prob(q)
    return q


if __name__ == "__main__":
    for p in (1e-1, 1e-2, 1e-3, 1e-4):
        print(p, [f"{concat_pl(p, L):.3e}" for L in (1, 2, 3, 4)])
