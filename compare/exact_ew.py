### Compute exact E[w], V[w] and Z_faulty for d=3 toric code by exhaustive enumeration.
### Loads syndrome table built by build_syndrome_table.py.
### Usage: python compare/exact_ew.py
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from css_code import make_toric_code

TABLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "Data", "syndrome_table_d3.npz"
)
CODE_SIZE = 3
P_VALUES = [1e-2, 5e-3, 2e-3, 1e-3]


def _faulty_weight_counts(table_path=None):
    """Return C_faulty[w] = number of faulty error configs of weight w (d=3)."""
    table_path = table_path or TABLE_PATH
    data = np.load(table_path)
    unique_syndromes = data["unique_syndromes"]
    corrections = data["corrections"]
    n_check = unique_syndromes.shape[1]

    code = make_toric_code(CODE_SIZE)
    n = code.n
    Hz = code.hz
    Lz = code.lz

    pow2 = (1 << np.arange(n_check)).astype(np.int32)
    syndrome_ints = unique_syndromes.astype(np.int32) @ pow2
    correction_lookup = np.zeros((int(syndrome_ints.max()) + 1, n), dtype=np.int8)
    correction_lookup[syndrome_ints] = corrections

    idx = np.arange(2**n, dtype=np.int32)
    bit_shifts = np.arange(n, dtype=np.int32)
    all_errors = ((idx[:, None] >> bit_shifts[None, :]) & 1).astype(np.uint8)

    syndromes_all = ((all_errors.astype(np.int32) @ Hz.T.astype(np.int32)) % 2).astype(
        np.uint8
    )
    syndrome_ints_all = syndromes_all.astype(np.int32) @ pow2
    corrections_for_all = correction_lookup[syndrome_ints_all]

    residuals = (all_errors ^ corrections_for_all.astype(np.uint8)) % 2
    logical_syndromes = (
        (residuals.astype(np.int32) @ Lz.T.astype(np.int32)) % 2
    ).astype(np.uint8)
    faulty_mask = logical_syndromes.any(axis=1)

    weights = all_errors.sum(axis=1).astype(np.int32)
    C_faulty = np.bincount(weights[faulty_mask], minlength=n + 1)
    return C_faulty, n


def exact_ew_from_counts(c_faulty, n, p):
    """Exact (E[w], V[w], Z_faulty) from a faulty weight-count array C_faulty[w]
    at physical rate p. Z_faulty = sum_{faulty e} p^w (1-p)^(n-w) = P(faulty)."""
    ws = np.arange(n + 1, dtype=float)
    valid = c_faulty > 0
    log_boltzmann = ws * np.log(p) + (n - ws) * np.log(1.0 - p)
    log_terms = np.full(n + 1, -np.inf)
    log_terms[valid] = np.log(c_faulty[valid].astype(float)) + log_boltzmann[valid]
    log_Z = np.logaddexp.reduce(log_terms[valid])
    probs = np.where(valid, np.exp(log_terms - log_Z), 0.0)
    e_w = float(np.dot(ws, probs))
    v_w = float(np.dot(ws**2, probs)) - e_w**2
    return e_w, v_w, float(np.exp(log_Z))


def faulty_weight_counts(code, decoder):
    """C_faulty[w] = number of faulty weight-w error configs for any small code
    (n <= ~22) by exhaustive enumeration; decodes once per unique syndrome.

    Returns (c_faulty, n, unique_syndromes, corrections) — the latter two form the
    syndrome->correction lookup table (uint8 / int8) for reuse.
    """
    n = code.n
    Hz = code.hz
    Lz = code.lz
    idx = np.arange(2**n, dtype=np.int64)
    bit_shifts = np.arange(n, dtype=np.int64)
    all_errors = ((idx[:, None] >> bit_shifts[None, :]) & 1).astype(np.uint8)
    syndromes = ((all_errors.astype(np.int32) @ Hz.T.astype(np.int32)) % 2).astype(
        np.uint8
    )
    unique_syndromes, inv = np.unique(syndromes, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    corrections = np.empty((len(unique_syndromes), n), dtype=np.int8)
    for i, s in enumerate(unique_syndromes):
        corrections[i] = decoder.decode_syndrome(s.astype(np.uint8))
    corr_all = corrections[inv]
    residual = (all_errors ^ corr_all.astype(np.uint8)) % 2
    logical = ((residual.astype(np.int32) @ Lz.T.astype(np.int32)) % 2).astype(np.uint8)
    faulty = logical.any(axis=1)
    weights = all_errors.sum(axis=1).astype(np.int32)
    c_faulty = np.bincount(weights[faulty], minlength=n + 1)
    return c_faulty, n, unique_syndromes, corrections


def exact_ew_d3(p, table_path=None):
    """Exact (E[w], V[w], Z_faulty) for the d=3 toric code at physical rate p."""
    c_faulty, n = _faulty_weight_counts(table_path)
    return exact_ew_from_counts(c_faulty, n, p)


if __name__ == "__main__":
    print(f"{'p':<12} {'E[w]_exact':>12} {'V[w]_exact':>12} {'Z_faulty':>16}")
    print("-" * 56)
    for p in P_VALUES:
        e_w, v_w, z = exact_ew_d3(p)
        print(f"  {p:<10.4e}  {e_w:>12.6f}  {v_w:>12.6f}  {z:>16.6e}")
