"""R-2: opt-in decode-call counting on run_rimc_depolarizing.

The returned decode count must equal the number of unique syndromes decoded,
which (since decode_x/decode_z are called only on a lookup-cache miss) equals
the number of decoder.decode_x + decoder.decode_z calls. We verify against an
independent call counter wrapped around the real decoder, and confirm the
default call signature is unchanged.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from css_code import make_toric_code, make_depolarizing_mwpm_decoder
from runners import run_rimc_depolarizing


class _CountingDecoder:
    """Delegates to a real depolarizing decoder, counting decode calls."""

    def __init__(self, inner):
        self._inner = inner
        self.n_decode_x = 0
        self.n_decode_z = 0

    def decode_x(self, s_z):
        self.n_decode_x += 1
        return self._inner.decode_x(s_z)

    def decode_z(self, s_x):
        self.n_decode_z += 1
        return self._inner.decode_z(s_x)

    def is_faulty(self, e_x, e_z, c_x, c_z):
        return self._inner.is_faulty(e_x, e_z, c_x, c_z)


def test_decode_count_matches_unique_syndromes():
    code = make_toric_code(3)
    dec = _CountingDecoder(make_depolarizing_mwpm_decoder(code))
    dist, trace, decode_calls = run_rimc_depolarizing(
        1e-2,
        500,
        code=code,
        decoder=dec,
        n_replicas=2,
        seed=7,
        return_decode_count=True,
    )
    assert isinstance(dist, dict)
    assert isinstance(trace, np.ndarray)
    assert isinstance(decode_calls, int)
    # decode is invoked exactly once per unique syndrome (cache miss), so the
    # returned count equals the total number of decode_x + decode_z calls.
    assert decode_calls == dec.n_decode_x + dec.n_decode_z
    assert decode_calls > 0


def test_decode_count_default_signature_unchanged():
    out = run_rimc_depolarizing(1e-2, 200, n_replicas=2, seed=42)
    assert len(out) == 2


def test_decode_count_with_swap_stats_appends_last():
    dist, trace, swaps, decode_calls = run_rimc_depolarizing(
        1e-2,
        300,
        n_replicas=4,
        seed=3,
        return_swap_stats=True,
        return_decode_count=True,
    )
    assert isinstance(dist, dict)
    swaps = np.asarray(swaps, dtype=float)
    assert swaps.shape == (3,)
    assert isinstance(decode_calls, int) and decode_calls > 0
