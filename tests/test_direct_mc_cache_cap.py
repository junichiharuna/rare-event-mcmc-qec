import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from css_code import make_surface_code, make_mwpm_decoder
from direct_mc import _make_adapter


def test_bitflip_adapter_cache_is_bounded():
    code = make_surface_code(3)
    dec = make_mwpm_decoder(code)
    ad = _make_adapter("bit_flip", code, dec, 0.2, use_cache=True, cache_cap=5)
    rng = np.random.default_rng(0)
    for _ in range(500):
        e = (rng.random(code.n) < 0.4).astype(np.int8)  # high-weight -> many distinct syndromes
        ad.decode_and_check(e)
    assert ad.lookup is not None
    assert len(ad.lookup) <= 5 + 1   # clear-on-full: size never exceeds cap+1


def test_bitflip_adapter_uncapped_by_default():
    code = make_surface_code(3)
    dec = make_mwpm_decoder(code)
    ad = _make_adapter("bit_flip", code, dec, 0.2, use_cache=True)  # no cache_cap -> unbounded
    rng = np.random.default_rng(0)
    for _ in range(300):
        e = (rng.random(code.n) < 0.4).astype(np.int8)
        ad.decode_and_check(e)
    assert len(ad.lookup) > 5   # grows past a small cap (default is unbounded)
