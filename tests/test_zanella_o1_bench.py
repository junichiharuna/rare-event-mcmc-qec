import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from css_code import make_toric_code, make_mwpm_decoder
from runners import run_rimc_mixed


def test_zanella_o1_bench_d9():
    """Smoke test: d=9 run completes in reasonable time."""
    code = make_toric_code(9)
    decoder = make_mwpm_decoder(code)
    t0 = time.perf_counter()
    dist, trace = run_rimc_mixed(
        0.01, 2000, code=code, decoder=decoder, n_replicas=4, seed=42
    )
    elapsed = time.perf_counter() - t0
    ew = sum(w * p for w, p in dist.items())
    assert 0 < ew < 100  # sanity
    print(f"\nd=9 N=2000: {elapsed:.2f}s, E[w]={ew:.3f}")
