import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from css_code import make_toric_code, make_mwpm_decoder
from runners import run_rimc_mixed


def test_cache_cap_preserves_results():
    """Capping/clearing the decode cache must not change results (decode is
    deterministic, so re-decoding after a clear yields the same correction)."""
    code = make_toric_code(3)
    dec = make_mwpm_decoder(code)
    capped = run_rimc_mixed(0.08, 600, p_zanella=1.0, n_replicas=2,
                            code=code, decoder=dec, seed=7, cache_cap=2)
    full = run_rimc_mixed(0.08, 600, p_zanella=1.0, n_replicas=2,
                          code=code, decoder=dec, seed=7, cache_cap=None)
    assert capped[0] == full[0]                  # faulty_error_distribution
    assert np.array_equal(capped[1], full[1])    # weight_trace


def test_cache_cap_evicts_forcing_redecodes():
    """A tiny cap must actually evict, forcing more decode calls than the
    unbounded cache (proves the cap bounds memory rather than being ignored)."""
    code = make_toric_code(3)
    base = make_mwpm_decoder(code)

    class Counting:
        def __init__(self):
            self.n = 0

        def decode_syndrome(self, syn):
            self.n += 1
            return base.decode_syndrome(syn)

        def is_faulty(self, e, c):
            return base.is_faulty(e, c)

    d_full = Counting()
    run_rimc_mixed(0.08, 3000, p_zanella=1.0, n_replicas=1,
                   code=code, decoder=d_full, seed=3, cache_cap=None)
    d_cap = Counting()
    run_rimc_mixed(0.08, 3000, p_zanella=1.0, n_replicas=1,
                   code=code, decoder=d_cap, seed=3, cache_cap=1)
    assert d_cap.n > d_full.n


def test_record_trace_false_preserves_distribution():
    """record_trace=False must give the same faulty_error_distribution (the E[w] source)
    and return an empty trace."""
    code = make_toric_code(3)
    dec = make_mwpm_decoder(code)
    full = run_rimc_mixed(0.08, 600, p_zanella=1.0, n_replicas=2,
                          code=code, decoder=dec, seed=11)
    notrace = run_rimc_mixed(0.08, 600, p_zanella=1.0, n_replicas=2,
                             code=code, decoder=dec, seed=11, record_trace=False)
    assert notrace[0] == full[0]          # identical distribution
    assert len(notrace[1]) == 0           # trace not recorded
    assert len(full[1]) > 0
