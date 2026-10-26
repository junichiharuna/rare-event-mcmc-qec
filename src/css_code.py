from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass
class CSSCode:
    """CSS quantum error correction code as dense NumPy arrays (int8)."""

    hx: np.ndarray  # shape (n_stab_x, n) — X-type stabilizers (Z-syndrome)
    hz: np.ndarray  # shape (n_stab_z, n) — Z-type stabilizers (X-syndrome)
    lx: np.ndarray  # shape (n_logical, n) — X-type logical operators
    lz: np.ndarray  # shape (n_logical, n) — Z-type logical operators
    n: int  # number of physical qubits


class Decoder(Protocol):
    def decode_syndrome(self, syndrome: np.ndarray) -> np.ndarray: ...
    def is_faulty(self, error: np.ndarray, correction: np.ndarray) -> bool: ...


def make_toric_code(d: int) -> CSSCode:
    """Return a CSSCode for the d×d toric code (n = 2d²)."""
    from ldpc.codes import ring_code
    from bposd.hgp import hgp

    q = hgp(ring_code(d))
    return CSSCode(
        hx=q.hx.toarray().astype(np.int8),
        hz=q.hz.toarray().astype(np.int8),
        lx=q.lx.toarray().astype(np.int8),
        lz=q.lz.toarray().astype(np.int8),
        n=q.N,
    )


def make_mwpm_decoder(code: CSSCode) -> Decoder:
    """Return a pymatching MWPM decoder using code.hz; is_faulty uses code.lz."""
    import pymatching

    matching = pymatching.Matching(code.hz)
    lz = code.lz  # captured for is_faulty

    class _MWPMDecoder:
        def decode_syndrome(self, syndrome: np.ndarray) -> np.ndarray:
            return matching.decode(syndrome)

        def is_faulty(self, error: np.ndarray, correction: np.ndarray) -> bool:
            residual = (error ^ correction) % 2
            return bool(((lz @ residual) % 2).any())

    return _MWPMDecoder()


class DepolarizingDecoder(Protocol):
    def decode_x(self, s_z: np.ndarray) -> np.ndarray: ...
    def decode_z(self, s_x: np.ndarray) -> np.ndarray: ...
    def is_faulty(
        self, e_x: np.ndarray, e_z: np.ndarray, c_x: np.ndarray, c_z: np.ndarray
    ) -> bool: ...


def make_depolarizing_mwpm_decoder(code: CSSCode) -> DepolarizingDecoder:
    """Independent MWPM decoders for X and Z errors under depolarizing noise."""
    import pymatching

    matching_x = pymatching.Matching(code.hz)  # X errors decoded from Z-syndrome
    matching_z = pymatching.Matching(code.hx)  # Z errors decoded from X-syndrome
    lz = code.lz
    lx = code.lx

    class _DepolarizingMWPMDecoder:
        def decode_x(self, s_z: np.ndarray) -> np.ndarray:
            return matching_x.decode(s_z)

        def decode_z(self, s_x: np.ndarray) -> np.ndarray:
            return matching_z.decode(s_x)

        def is_faulty(
            self, e_x: np.ndarray, e_z: np.ndarray, c_x: np.ndarray, c_z: np.ndarray
        ) -> bool:
            residual_x = (e_x ^ c_x) % 2
            residual_z = (e_z ^ c_z) % 2
            return bool(((lz @ residual_x) % 2).any() or ((lx @ residual_z) % 2).any())

    return _DepolarizingMWPMDecoder()


def make_bposd_decoder(code: CSSCode, error_rate: float = 0.01) -> Decoder:
    """BP-OSD decoder for CSS codes (recommended for LDPC codes like BB).

    Parameters from [2511.15177]: 100 BP iterations, combination sweep order 10.
    error_rate: physical error rate (used as BP channel prior).
    """
    from ldpc import BpOsdDecoder as _BpOsdDecoder

    _dec = _BpOsdDecoder(
        code.hz,
        error_rate=error_rate,
        max_iter=100,
        bp_method="min_sum",
        osd_method="osd_cs",
        osd_order=10,
    )
    lz = code.lz

    class _BPOSDDecoder:
        def decode_syndrome(self, syndrome: np.ndarray) -> np.ndarray:
            return _dec.decode(syndrome).astype(np.int8)

        def is_faulty(self, error: np.ndarray, correction: np.ndarray) -> bool:
            residual = (error ^ correction) % 2
            return bool(((lz @ residual) % 2).any())

    return _BPOSDDecoder()


def make_bp_decoder(code: CSSCode, error_rate: float = 0.01) -> Decoder:
    """Plain belief-propagation decoder (NO OSD) for CSS codes.

    Exposes the BP error floor: plain BP can fail to converge on low-weight
    degenerate configurations (Poulin-Chung), returning a hard decision that
    does not satisfy the syndrome. Non-convergence is counted as faulty.
    error_rate: physical error rate used as the BP channel prior.
    """
    try:
        from ldpc import BpDecoder as _BpDecoder
    except ImportError:  # older ldpc layout
        from ldpc.bp_decoder import BpDecoder as _BpDecoder

    _dec = _BpDecoder(
        code.hz,
        error_rate=error_rate,
        max_iter=100,
        bp_method="min_sum",
    )
    hz = code.hz
    lz = code.lz

    class _BPDecoder:
        def decode_syndrome(self, syndrome: np.ndarray) -> np.ndarray:
            return _dec.decode(syndrome).astype(np.int8)

        def is_faulty(self, error: np.ndarray, correction: np.ndarray) -> bool:
            residual = (error ^ correction) % 2
            unconverged = bool(((hz @ residual) % 2).any())  # syndrome not reproduced
            wrong_coset = bool(((lz @ residual) % 2).any())
            return unconverged or wrong_coset

    return _BPDecoder()


def make_depolarizing_bposd_decoder(
    code: CSSCode, error_rate: float = 0.01
) -> DepolarizingDecoder:
    """Independent BP-OSD decoders for X and Z errors under depolarizing noise."""
    from ldpc import BpOsdDecoder as _BpOsdDecoder

    _dec_x = _BpOsdDecoder(
        code.hz,
        error_rate=error_rate,
        max_iter=100,
        bp_method="min_sum",
        osd_method="osd_cs",
        osd_order=10,
    )
    _dec_z = _BpOsdDecoder(
        code.hx,
        error_rate=error_rate,
        max_iter=100,
        bp_method="min_sum",
        osd_method="osd_cs",
        osd_order=10,
    )
    lz = code.lz
    lx = code.lx

    class _DepolarizingBPOSDDecoder:
        def decode_x(self, s_z: np.ndarray) -> np.ndarray:
            return _dec_x.decode(s_z).astype(np.int8)

        def decode_z(self, s_x: np.ndarray) -> np.ndarray:
            return _dec_z.decode(s_x).astype(np.int8)

        def is_faulty(
            self, e_x: np.ndarray, e_z: np.ndarray, c_x: np.ndarray, c_z: np.ndarray
        ) -> bool:
            residual_x = (e_x ^ c_x) % 2
            residual_z = (e_z ^ c_z) % 2
            return bool(((lz @ residual_x) % 2).any() or ((lx @ residual_z) % 2).any())

    return _DepolarizingBPOSDDecoder()


# ── GF(2) linear algebra helpers ──────────────────────────────────────────


def _gf2_row_reduce(H: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row echelon form over GF(2). Returns (rref, pivot_cols)."""
    H = H.astype(np.int8).copy()
    m, n = H.shape
    pivots: list[int] = []
    row = 0
    for col in range(n):
        found = next((r for r in range(row, m) if H[r, col]), -1)
        if found == -1:
            continue
        H[[row, found]] = H[[found, row]]
        for r in range(m):
            if r != row and H[r, col]:
                H[r] = (H[r] + H[row]) % 2
        pivots.append(col)
        row += 1
    return H, pivots


def _gf2_rank(H: np.ndarray) -> int:
    _, pivots = _gf2_row_reduce(H)
    return len(pivots)


def _gf2_nullspace(H: np.ndarray) -> np.ndarray:
    """Rows span ker(H) over GF(2). Returns array shape (dim_ker, n)."""
    m, n = H.shape
    rref, pivots = _gf2_row_reduce(H)
    pivot_set = set(pivots)
    free_cols = [c for c in range(n) if c not in pivot_set]
    null_vecs = []
    for f in free_cols:
        v = np.zeros(n, dtype=np.int8)
        v[f] = 1
        for i, p in enumerate(pivots):
            v[p] = rref[i, f]
        null_vecs.append(v)
    if not null_vecs:
        return np.zeros((0, n), dtype=np.int8)
    return np.array(null_vecs, dtype=np.int8)


def _css_logicals(HX: np.ndarray, HZ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute LX, LZ for a CSS code with HX @ HZ.T = 0 mod 2."""
    n = HX.shape[1]
    rank_HX = _gf2_rank(HX)
    rank_HZ = _gf2_rank(HZ)
    k = n - rank_HX - rank_HZ

    def _logicals_for(H_stab: np.ndarray, H_other: np.ndarray) -> np.ndarray:
        # Step 1: basis of ker(H_stab)
        ker = _gf2_nullspace(H_stab)  # shape (n - rank_stab, n)
        if k == 0 or ker.shape[0] == 0:
            return np.zeros((0, n), dtype=np.int8)
        # Step 2: project out rowspan(H_other) from ker
        # Get pivot columns of H_other in RREF
        rref_other, pivots_other = _gf2_row_reduce(H_other)
        # For each pivot column, eliminate that column in ker using rref_other
        K = ker.copy()
        for rank_idx, pivot_col in enumerate(pivots_other):
            # Zero out pivot_col in every row of K using rref_other[rank_idx]
            for r in range(K.shape[0]):
                if K[r, pivot_col]:
                    K[r] = (K[r] + rref_other[rank_idx]) % 2
        # Step 3: row-reduce K; non-zero rows are the logical operators
        rref_K, _ = _gf2_row_reduce(K)
        logicals = rref_K[rref_K.any(axis=1)]
        return logicals.astype(np.int8)

    return _logicals_for(HZ, HX), _logicals_for(HX, HZ)


# ── Code factories ─────────────────────────────────────────────────────────


def make_asymmetric_toric_code(d1: int, d2: int) -> CSSCode:
    """Rectangular toric code UT(d1, d2) with n = 2*d1*d2 qubits."""
    from ldpc.codes import ring_code
    from bposd.hgp import hgp

    q = hgp(ring_code(d1), ring_code(d2))
    return CSSCode(
        hx=q.hx.toarray().astype(np.int8),
        hz=q.hz.toarray().astype(np.int8),
        lx=q.lx.toarray().astype(np.int8),
        lz=q.lz.toarray().astype(np.int8),
        n=q.N,
    )


def make_surface_code(d: int) -> CSSCode:
    """CSS surface code [[2d²-2d+1, 1, d]] via HGP of repetition codes."""
    from ldpc.codes import rep_code
    from bposd.hgp import hgp

    q = hgp(rep_code(d), rep_code(d))
    return CSSCode(
        hx=q.hx.toarray().astype(np.int8),
        hz=q.hz.toarray().astype(np.int8),
        lx=q.lx.toarray().astype(np.int8),
        lz=q.lz.toarray().astype(np.int8),
        n=q.N,
    )


def make_hgp_hamming() -> CSSCode:
    """HGP of the [7,4,3] Hamming code with itself → [[58, 16, 3]] (k>1).

    No-trap control for the trapping study: k=16 > 1 but (empirically) the logical
    weight structure is uniform, so OBF does not trap. n = 7*7 + 3*3 = 58,
    k = nullity(H)^2 + nullity(H^T)^2 = 4^2 + 0^2 = 16.
    """
    from ldpc.codes import hamming_code
    from bposd.hgp import hgp

    H = hamming_code(3)   # scipy sparse [3,7] Hamming parity check (uint8)
    q = hgp(H, H)
    return CSSCode(
        hx=q.hx.toarray().astype(np.int8),
        hz=q.hz.toarray().astype(np.int8),
        lx=q.lx.toarray().astype(np.int8),
        lz=q.lz.toarray().astype(np.int8),
        n=q.N,
    )


def make_bb_code(
    l: int,
    m: int,
    a_terms: list[tuple[int, int]],
    b_terms: list[tuple[int, int]],
) -> CSSCode:
    """Bivariate Bicycle code over Z_l × Z_m.

    a_terms / b_terms: list of (i, j) exponent pairs for polynomials A, B
    in the group ring Z_2[Z_l × Z_m].  HX = [A|B], HZ = [B^T|A^T].
    """
    n_block = l * m

    def _shift(di: int, dj: int) -> np.ndarray:
        S = np.zeros((n_block, n_block), dtype=np.int8)
        for i in range(l):
            for j in range(m):
                S[((i + di) % l) * m + ((j + dj) % m), i * m + j] = 1
        return S

    A = (
        sum(
            (_shift(di, dj) for di, dj in a_terms),
            np.zeros((n_block, n_block), dtype=np.int8),
        )
        % 2
    )
    B = (
        sum(
            (_shift(di, dj) for di, dj in b_terms),
            np.zeros((n_block, n_block), dtype=np.int8),
        )
        % 2
    )

    HX = np.hstack([A, B]).astype(np.int8)
    HZ = np.hstack([B.T, A.T]).astype(np.int8)
    LX, LZ = _css_logicals(HX, HZ)

    return CSSCode(hx=HX, hz=HZ, lx=LX, lz=LZ, n=2 * n_block)


def make_bb_code_72() -> CSSCode:
    """[[72, 12, 6]] bivariate bicycle code (Bravyi et al. 2024, l=6 m=6)."""
    return make_bb_code(
        l=6,
        m=6,
        a_terms=[(3, 0), (0, 1), (0, 2)],
        b_terms=[(0, 3), (1, 0), (2, 0)],
    )


def make_bb_code_144() -> CSSCode:
    """[[144, 12, 12]] bivariate bicycle code (IBM, l=12 m=6)."""
    return make_bb_code(
        l=12,
        m=6,
        a_terms=[(3, 0), (0, 1), (0, 2)],
        b_terms=[(0, 3), (1, 0), (2, 0)],
    )


def _lift_rows(checks: np.ndarray, inner_logical: np.ndarray, n_in: int) -> np.ndarray:
    """Lift outer check rows: for each outer-qubit in a row's support, place the
    inner logical operator on that block."""
    n_out = checks.shape[1]
    rows = []
    for s in checks:
        row = np.zeros(n_out * n_in, dtype=np.int8)
        for b in range(n_out):
            if s[b]:
                row[b * n_in:(b + 1) * n_in] = inner_logical
        rows.append(row)
    return np.array(rows, dtype=np.int8).reshape(checks.shape[0], n_out * n_in)


def _block_diag_copies(check: np.ndarray, n_blocks: int) -> np.ndarray:
    """Place `check` (r x n_in) on each of n_blocks blocks; rows ordered block-by-block."""
    r, n_in = check.shape
    out = np.zeros((n_blocks * r, n_blocks * n_in), dtype=np.int8)
    for b in range(n_blocks):
        out[b * r:(b + 1) * r, b * n_in:(b + 1) * n_in] = check
    return out


def concatenate_css(outer: CSSCode, inner: CSSCode) -> CSSCode:
    """Concatenate: each outer qubit is encoded into one `inner` block (inner.k == 1).

    Row order is [7 inner blocks (block-by-block), then lifted outer rows] — the
    hard-decision decoder relies on this layout.
    """
    if inner.lx.shape[0] != 1 or inner.lz.shape[0] != 1:
        raise ValueError("concatenate_css requires an inner code with k == 1")
    n_in = inner.n
    n_out = outer.n
    lx_in = inner.lx[0]
    lz_in = inner.lz[0]

    hx = np.vstack([
        _block_diag_copies(inner.hx, n_out),
        _lift_rows(outer.hx, lx_in, n_in),
    ]).astype(np.int8)
    hz = np.vstack([
        _block_diag_copies(inner.hz, n_out),
        _lift_rows(outer.hz, lz_in, n_in),
    ]).astype(np.int8)
    lx = _lift_rows(outer.lx, lx_in, n_in).astype(np.int8)
    lz = _lift_rows(outer.lz, lz_in, n_in).astype(np.int8)
    return CSSCode(hx=hx, hz=hz, lx=lx, lz=lz, n=n_out * n_in)


def make_concatenated_steane_code(levels: int) -> CSSCode:
    """Level-`levels` self-concatenation of the Steane code: n = 7**levels.

    L_1 = Steane; L_ell = concatenate_css(outer=Steane, inner=L_{ell-1}), so each
    of the 7 outer blocks is a full L_{ell-1} code.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")
    code = make_steane_code()
    for _ in range(levels - 1):
        code = concatenate_css(make_steane_code(), code)
    return code


# Steane Z-syndrome (3 bits) -> single-qubit X correction position; index = binary value.
_STEANE_HZ = np.array(
    [[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=np.int8
)


def _steane_lookup(synd3: np.ndarray) -> np.ndarray:
    """3-bit Z-syndrome -> 7-bit min-weight X correction (Hamming lookup)."""
    idx = int(synd3[0]) + 2 * int(synd3[1]) + 4 * int(synd3[2])
    c = np.zeros(7, dtype=np.int8)
    if idx != 0:
        c[idx - 1] = 1
    return c


def make_concat_steane_hard_decoder(levels: int):
    """Hard-decision level-by-level decoder for the level-`levels` concatenated Steane code.

    Corrects X-errors from the Z-syndrome; `is_faulty` uses the code's lz.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")
    code = make_concatenated_steane_code(levels)
    lz_full = code.lz
    # Per-level inner data, index l in [1..levels-1] describes the inner code at that step.
    inner_codes = [make_concatenated_steane_code(l) for l in range(1, levels)]  # L_1..L_{levels-1}

    def _decode(synd: np.ndarray, level: int) -> np.ndarray:
        if level == 1:
            return _steane_lookup(synd)
        inner = inner_codes[level - 2]  # L_{level-1}
        n_in = inner.n
        r_in = inner.hz.shape[0]
        lx_in = inner.lx[0]
        lz_in = inner.lz[0]
        c = np.zeros(7 * n_in, dtype=np.int8)
        lvec = np.zeros(7, dtype=np.int8)
        for b in range(7):
            sub = synd[b * r_in:(b + 1) * r_in]
            cb = _decode(sub, level - 1)
            c[b * n_in:(b + 1) * n_in] = cb
            lvec[b] = int((lz_in @ cb) % 2)
        outer_synd_lifted = synd[7 * r_in:7 * r_in + 3]
        outer_synd = (outer_synd_lifted + (_STEANE_HZ @ lvec)) % 2
        outer_corr = _steane_lookup(outer_synd)  # which outer qubits to flip
        for b in range(7):
            if outer_corr[b]:
                c[b * n_in:(b + 1) * n_in] = (c[b * n_in:(b + 1) * n_in] + lx_in) % 2
        return c.astype(np.int8)

    class _ConcatSteaneHardDecoder:
        def decode_syndrome(self, syndrome: np.ndarray) -> np.ndarray:
            return _decode(np.asarray(syndrome, dtype=np.int8), levels)

        def is_faulty(self, error: np.ndarray, correction: np.ndarray) -> bool:
            residual = (error ^ correction) % 2
            return bool(((lz_full @ residual) % 2).any())

    return _ConcatSteaneHardDecoder()


def make_steane_code() -> CSSCode:
    """Self-dual CSS Steane code [[7,1,3]] from the [7,4,3] Hamming code.

    Column c (0-indexed) of H is the binary representation of (c+1), so the
    Z-syndrome read as an integer equals the index (1..7) of a single-qubit
    X error (0 = no error) — used by the hard-decision lookup decoder.
    """
    H = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1],  # LSB
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],  # MSB
        ],
        dtype=np.int8,
    )
    LX, LZ = _css_logicals(H, H)
    return CSSCode(hx=H.copy(), hz=H.copy(), lx=LX, lz=LZ, n=7)
