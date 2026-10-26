### Precompute syndrome -> correction table for d=3 toric code.
### Runs MWPM once per unique syndrome (~64 calls), saves compact .npz file.
### Usage: python compare/build_syndrome_table.py
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from css_code import make_toric_code, make_mwpm_decoder

CODE_SIZE = 3
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "Data", "syndrome_table_d3.npz"
)


if __name__ == "__main__":
    code = make_toric_code(CODE_SIZE)
    decoder = make_mwpm_decoder(code)
    n = code.n  # 18
    Hz = code.hz  # shape (n_check, 18), int8

    print(
        f"Building syndrome table for d={CODE_SIZE} "
        f"(n={n}, 2^{n} = {2**n:,} configs) ..."
    )

    # Generate all 2^n error configs as (262144, 18) uint8
    idx = np.arange(2**n, dtype=np.int32)
    bit_shifts = np.arange(n, dtype=np.int32)
    all_errors = ((idx[:, None] >> bit_shifts[None, :]) & 1).astype(np.uint8)

    # Compute syndromes: (262144, n_check), values in {0,1}
    syndromes_all = ((all_errors.astype(np.int32) @ Hz.T.astype(np.int32)) % 2).astype(
        np.uint8
    )

    # Find unique syndromes
    unique_syndromes, _ = np.unique(syndromes_all, axis=0, return_inverse=True)
    n_unique = len(unique_syndromes)
    print(f"Unique syndromes found: {n_unique}")

    # Run MWPM once per unique syndrome
    corrections = np.empty((n_unique, n), dtype=np.int8)
    for i, s in enumerate(unique_syndromes):
        corrections[i] = decoder.decode_syndrome(s.astype(np.uint8))
    print(f"MWPM calls completed: {n_unique}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH, unique_syndromes=unique_syndromes, corrections=corrections
    )
    size = os.path.getsize(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")
    print(
        f"File size: {size} bytes  "
        f"({n_unique} syndromes x {n} qubits = "
        f"{n_unique * n} entries)"
    )
