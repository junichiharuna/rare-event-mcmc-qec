"""Weight-stratified ("subset sampling" / exact importance sampling) reference
values P_L(p) and E[w | faulty](p) for a general CSS code under bit-flip noise.

Exact decomposition (no approximation beyond per-stratum MC and an explicit
truncation bound):

    P_L(p)    = sum_w B(w; n, p) f_w
    E[w|F](p) = sum_w w B(w; n, p) f_w / sum_w B(w; n, p) f_w

with B the binomial pmf and f_w = P(decoder fails | |e| = w).  Per stratum:

  * C(n, w) <= enum_cap  ->  enumerate every weight-w pattern; f_w is EXACT (SE 0).
  * otherwise            ->  MC over uniform weight-w patterns; a short pilot sizes
    the run for ~target_fail failures, capped at n_max.  f_w = k/N with an
    Agresti-Coull SE (so k = 0 still gets a meaningful SE rather than 0).

Truncation: strata above w_max are dropped; the neglected mass is bounded above by
sum_{w>w_max} B(w; n, p) (using f_w <= 1).  The reported p_L_se is floored at that
bound, so a fully enumerated row still carries its only remaining error.

Three regimes, kept apart by is_mwpm (does the pymatching decode_batch fast path
apply?) and is_p_independent (does one run synthesize the whole p grid?):

  * MWPM codes (toric / surface / asymmetric_toric) -- fast path, and f_w carries
    no channel prior, so ONE run covers an arbitrary p grid.
  * BP-OSD codes (bb*/hgp_hamming3) -- generic per-syndrome path, and the prior IS
    p, so f_w depends on p and one run PER p is required; the caller is
    responsible for that loop (see cluster_scripts/subset_sampling.py).
  * hard-decoder codes (concat_steane) -- generic path like BP-OSD, but the
    hierarchical hard-decision decoder has no prior, so f_w is p-independent and
    one run again covers the whole grid.

Depolarizing noise (noise="depolarizing"): each qubit is non-identity with
probability p and then X, Y or Z uniformly, so the support weight is still
Binomial(n, p) and, given the support, the Pauli types are i.i.d. uniform.  The
same formulas hold with f_w = P(faulty | |supp| = w), where "faulty" is
direct_mc.DepolarizingAdapter's definition (X part decoded from H_Z e_X, Z part
from H_X e_Z, faulty if either logical test fails).  A weight class then holds
C(n, w) 3**w patterns, and n_samples counts (support, type) patterns.

This is the code-agnostic generalization of the toric-only compare/stratified_pl.py.
Codes and decoders come from direct_mc._make_code_and_decoder wherever it knows the
family, so the reference value computed here is the same quantity the direct-MC truth
grid estimates.  The families it does not know -- rectangular asymmetric toric, and
the cross-code-audit codes hgp_hamming3 / bb_l9m6 -- are built here from the same
css_code factories and the same decoder constructor that the MCMC audit driver
cluster_scripts/zh_rerun.py uses, so reference and chain share a failure definition.
"""

from __future__ import annotations

import itertools
import math
import os
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any

import numpy as np

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import direct_mc  # noqa: E402
from css_code import (  # noqa: E402
    make_asymmetric_toric_code, make_bb_code, make_bp_decoder, make_bposd_decoder,
    make_depolarizing_bposd_decoder, make_depolarizing_mwpm_decoder,
    make_hgp_hamming, make_mwpm_decoder,
)

# ── defaults ───────────────────────────────────────────────────────────────
ENUM_CAP = 5_000_000
N_MAX = 5_000_000
TARGET_FAIL = 1000
PILOT_N = 20_000
ENUM_BATCH = 100_000       # inner vectorized batch (memory), enumeration
MC_BATCH = 25_000          # inner vectorized batch (memory), MC
# Smallest work unit handed to a pool worker.  These bound how far a stratum can
# be split: the node must stay saturated (>= n_cores concurrent tasks) even for a
# small stratum, but a chunk starting at offset `start` must first skip `start`
# combinations, so chunks cannot be made arbitrarily tiny either.
ENUM_CHUNK_MIN = 5_000
MC_CHUNK_MIN = 2_000
CACHE_CAP = 200_000

FW_HEADER = [
    "code_type", "d", "p_decoder", "n", "w", "mode",
    "n_samples", "n_fail", "f_w", "f_w_se",
]

# ── code registry ──────────────────────────────────────────────────────────
# Codes whose decoder is BP-OSD (channel prior = p, so f_w is p-DEPENDENT and one
# run per p is required) rather than MWPM.  This is the single membership test
# read by is_mwpm / is_p_independent / decoder_name / assert_wmax and by the CLI's
# task loop.
_BPOSD_CODES = ("bb72", "bb144", "bb_l9m6", "hgp_hamming3")

# Codes decoded by a p-INDEPENDENT decoder that is nevertheless not pymatching:
# concat_steane's hierarchical hard-decision decoder (css_code.
# make_concat_steane_hard_decoder) takes no channel prior, so it runs on the
# generic per-syndrome path yet still synthesizes the whole p grid from one run.
# This is the pymatching-EXCLUSION list only (read by is_mwpm).  It carries no
# correctable-weight bound: assert_wmax names concat_steane explicitly because
# its 2**ell - 1 is family-specific, so a new entry here needs its own branch
# there too, or its theorem check silently switches off.
_HARD_CODES = ("concat_steane",)


def _make_bb_l9m6():
    """BB [[108, 8, 10]] over Z_9 x Z_6 -- same polynomials as bb72 / bb144.

    Identical to cluster_scripts/zh_rerun.py's build_code("bb_l9m6").
    """
    return make_bb_code(9, 6, a_terms=[(3, 0), (0, 1), (0, 2)],
                        b_terms=[(0, 3), (1, 0), (2, 0)])


# Builders for the paper's cross-code-audit families that
# direct_mc._make_code_and_decoder does not know.  Every entry must build the
# SAME code object as cluster_scripts/zh_rerun.py's build_code(), so that the
# reference value here and the MCMC chain share one failure definition.
_EXTRA_BUILDERS = {
    "hgp_hamming3": make_hgp_hamming,   # HGP(Hamming(3)) [[58, 16, 3]]
    "bb_l9m6": _make_bb_l9m6,           # BB [[108, 8, 10]]
}

# Known code distances, used ONLY for the "zero failures below w_assert" sanity
# assert.  A MISSING entry silently disables that assert (assert_wmax falls to 0),
# so every registered code is covered either here or by an explicit branch of
# code_distance() (toric / surface / asymmetric_toric / concat_steane, whose
# distance follows from the size spec rather than the family name).
# bb_l9m6 uses the literature distance 10;
# cluster_results/zh_rerun.csv's lwt_min=12 is the minimum weight over the
# COMPUTED lz basis (an upper bound on d), not d itself.  The value only places
# the assert, never the estimate.
_DISTANCE = {"bb72": 6, "bb144": 12, "bb_l9m6": 10, "hgp_hamming3": 3}

# Largest weight at which failures are asserted to be impossible.  For a
# minimum-weight decoder this is the coding-theory bound (d-1)//2.  BP-OSD is NOT
# a minimum-weight decoder, so for the BP-OSD families only the trivially
# correctable weights are asserted and everything above is merely reported.
_BPOSD_ASSERT_WMAX = 2


# ── d-spec handling ────────────────────────────────────────────────────────
def parse_d(d) -> str:
    """Canonical string form of a code-size spec: 5 -> "5", "3x11" -> "3x11"."""
    s = str(d).strip()
    if "x" in s:
        parts = tuple(int(x) for x in s.split("x"))
        return "x".join(str(x) for x in parts)
    return str(int(s))


def d_parts(d) -> tuple:
    """Integer components of a d-spec: "3x11" -> (3, 11), 7 -> (7,)."""
    s = str(d).strip()
    if "x" in s:
        return tuple(int(x) for x in s.split("x"))
    return (int(s),)


def code_distance(code_type: str, d) -> int | None:
    """Code distance if it is known analytically, else None.

    Used only to place the theorem assert; a wrong entry here shows up as a loud
    assertion failure rather than a silent bias.
    """
    if code_type in _DISTANCE:
        return _DISTANCE[code_type]
    parts = d_parts(d)
    if code_type in ("toric", "surface"):
        return parts[0]
    if code_type == "asymmetric_toric":
        return min(parts)  # UT(d1, d2) = [[2 d1 d2, 2, min(d1, d2)]]
    if code_type == "concat_steane":
        return 3 ** parts[0]  # L_ell = [[7**ell, 1, 3**ell]]; d is the LEVEL count
    return None


def is_mwpm(code_type: str, decoder_kind: str = "auto") -> bool:
    """True iff build_code_and_decoder returns a pymatching MWPM decoder.

    This is the decode_batch FAST-PATH test only.  For "does one run synthesize
    the whole p grid?" use is_p_independent, which is also true for the
    hard-decoder family concat_steane.
    """
    if code_type in _HARD_CODES:
        return False  # hierarchical hard decoder; there is no pymatching object
    if decoder_kind == "mwpm":
        return True
    if decoder_kind == "auto":
        return code_type not in _BPOSD_CODES
    return False


def is_p_independent(code_type: str, decoder_kind: str = "auto") -> bool:
    """True iff the f_w table is valid at every p, so ONE run covers the p grid.

    False only when the decoder carries a channel prior fixed at p: BP-OSD (the
    _BPOSD_CODES families under decoder_kind="auto") and plain BP.  MWPM and
    concat_steane's hard-decision decoder have no prior.
    """
    if decoder_kind == "mwpm":
        return True
    if decoder_kind == "bp":
        return False
    if decoder_kind == "auto":
        return code_type not in _BPOSD_CODES
    return False


def decoder_name(code_type: str, decoder_kind: str = "auto") -> str:
    """The `decoder` column value, matching direct_mc.run_condition's own label.

    NOTE concat_steane inherits "mwpm" from direct_mc, which labels everything
    that is not a BB code that way; the decoder is really the hierarchical hard
    decoder.  The point of reusing the string is that these rows join
    like-for-like against the direct-MC truth grid.
    """
    if decoder_kind in ("mwpm", "bp"):
        return decoder_kind
    return "bposd" if code_type in _BPOSD_CODES else "mwpm"


def assert_wmax(code_type: str, d, decoder_kind: str = "auto") -> int:
    """Highest weight at which zero decoder failures is a THEOREM, not a measurement.

    MWPM: the coding-theory bound (d-1)//2.  BP-OSD: only the trivially
    correctable weights (capped at _BPOSD_ASSERT_WMAX = 2), and never above
    (d-1)//2 -- so hgp_hamming3 (d=3) asserts w<=1 while the BB codes assert
    w<=2.  Plain BP counts non-convergence as a failure and has no such theorem,
    hence 0.  0 means "assert nothing"; it is also what an unknown distance
    yields, which is why every registered code has a code_distance().

    concat_steane is NOT a minimum-weight decoder, so (d-1)//2 = (3**ell-1)//2 is
    WRONG for it (a weight-4 error with two errors in each of two inner Steane
    blocks already defeats the level-2 decoder, although d = 9).  The bound that
    does hold is t_ell = 2*t_{ell-1} + 1 = 2**ell - 1, since the outer Steane
    miscorrects only once TWO sub-blocks each carry more than t_{ell-1} errors:
    t_1 = 1 (Steane), t_2 = 3, t_3 = 7 -- strictly below (d-1)//2 everywhere.
    """
    if decoder_kind == "bp":
        return 0
    # Before the is_mwpm / _BPOSD_CODES branches: falling through would silently
    # return 0 and switch the theorem check off.
    if code_type == "concat_steane":
        return 2 ** d_parts(d)[0] - 1
    d_code = code_distance(code_type, d)
    if not d_code:
        return 0
    if is_mwpm(code_type, decoder_kind):
        return (d_code - 1) // 2
    if code_type in _BPOSD_CODES:
        return min(_BPOSD_ASSERT_WMAX, (d_code - 1) // 2)
    return 0


# ── per-process context ────────────────────────────────────────────────────
@dataclass
class Ctx:
    """Everything a worker needs to turn support indices into a failure mask."""

    code: Any
    decoder: Any
    n: int
    hzT: np.ndarray          # (n, m) uint8 -- hz transposed, for the XOR-gather
    lzT: np.ndarray          # (n, k) uint8 -- lz transposed, for the logical test
    matching: Any            # pymatching.Matching, or None (generic path)
    cache: dict = field(default_factory=dict)
    cache_cap: int = CACHE_CAP
    code_type: str = ""
    d: str = ""
    p_decoder: float = 0.0
    decoder_kind: str = "auto"
    noise: str = "bit_flip"
    hxT: np.ndarray | None = None     # (n, m_x) uint8, Z-part syndromes (depolarizing)
    lxT: np.ndarray | None = None     # (n, k) uint8, logical test of the Z part
    matching_z: Any = None            # pymatching.Matching(hx), or None
    cache_z: dict = field(default_factory=dict)


def build_code_and_decoder(noise, code_type, d, p, decoder_kind="auto"):
    """direct_mc._make_code_and_decoder, extended to the codes it does not know.

    Two extensions:
      * rectangular asymmetric toric -- direct_mc hard-codes
        make_asymmetric_toric_code(d, d); a "d1xd2" spec is built here instead
        (same construction as cluster_scripts/rect_audit_zh.py);
      * the cross-code-audit families hgp_hamming3 / bb_l9m6 -- built from the
        same css_code factories, and decoded by make_bposd_decoder(code, p),
        which is EXACTLY what cluster_scripts/zh_rerun.py uses for both its MCMC
        chains and its own direct-MC truth.  Sharing the decoder is the point:
        the reference value and the chain must agree on what "faulty" means.
    Under depolarizing noise these become make_depolarizing_mwpm_decoder /
    make_depolarizing_bposd_decoder(code, p), as in cluster_scripts/paper_codes.py.

    Everything else falls through to direct_mc, including concat_steane, whose
    d-spec is a LEVEL count: "2" -> make_concatenated_steane_code(2) [[49,1,9]]
    plus make_concat_steane_hard_decoder(2).
    """
    parts = d_parts(d)
    if code_type == "asymmetric_toric" and len(parts) == 2:
        if decoder_kind not in ("auto", "mwpm"):
            raise ValueError(f"asymmetric_toric wants an MWPM decoder, got {decoder_kind!r}")
        code = make_asymmetric_toric_code(parts[0], parts[1])
        if noise == "depolarizing":
            return code, make_depolarizing_mwpm_decoder(code)
        if noise != "bit_flip":
            raise ValueError(f"unknown noise {noise!r}")
        return code, make_mwpm_decoder(code)
    if code_type in _EXTRA_BUILDERS:
        code = _EXTRA_BUILDERS[code_type]()
        if noise == "depolarizing":
            if decoder_kind != "auto":
                raise ValueError("depolarizing supports decoder_kind='auto' only")
            return code, make_depolarizing_bposd_decoder(code, p)
        if noise != "bit_flip":
            raise ValueError(f"unknown noise {noise!r}")
        if decoder_kind == "mwpm":
            return code, make_mwpm_decoder(code)
        if decoder_kind == "bp":
            return code, make_bp_decoder(code, p)
        if decoder_kind != "auto":
            raise ValueError(f"Unknown decoder_kind: {decoder_kind}")
        return code, make_bposd_decoder(code, p)
    return direct_mc._make_code_and_decoder(noise, code_type, parts[0], p, decoder_kind)


def build_ctx(noise, code_type, d, p, decoder_kind="auto", cache_cap=CACHE_CAP,
              force_generic=False) -> Ctx:
    """Build the code, the decoder and the vectorized helpers for one condition.

    `force_generic=True` suppresses the pymatching decode_batch fast path so the
    generic per-syndrome path can be exercised (and compared) on an MWPM code.
    """
    code, decoder = build_code_and_decoder(noise, code_type, d, p, decoder_kind)
    matching = None
    if is_mwpm(code_type, decoder_kind) and not force_generic:
        import pymatching

        matching = pymatching.Matching(code.hz)
    matching_z = None
    if noise == "depolarizing" and is_mwpm(code_type, decoder_kind) and not force_generic:
        import pymatching

        matching_z = pymatching.Matching(code.hx)
    return Ctx(
        code=code,
        decoder=decoder,
        n=code.n,
        hzT=np.ascontiguousarray(code.hz.T.astype(np.uint8)),
        lzT=np.ascontiguousarray(code.lz.T.astype(np.uint8)),
        matching=matching,
        cache_cap=int(cache_cap),
        code_type=code_type,
        d=parse_d(d),
        p_decoder=float(p),
        decoder_kind=decoder_kind,
        noise=noise,
        hxT=np.ascontiguousarray(code.hx.T.astype(np.uint8)),
        lxT=np.ascontiguousarray(code.lx.T.astype(np.uint8)),
        matching_z=matching_z,
    )


# ── the hot path ───────────────────────────────────────────────────────────
def _syndromes(idx: np.ndarray, ctx: Ctx) -> np.ndarray:
    """(b, m) uint8 syndromes for a batch of weight-w supports (XOR of hz columns)."""
    return np.bitwise_xor.reduce(ctx.hzT[idx], axis=1)


def _errors(idx: np.ndarray, ctx: Ctx, dtype=np.uint8) -> np.ndarray:
    b = idx.shape[0]
    e = np.zeros((b, ctx.n), dtype=dtype)
    e[np.arange(b)[:, None], idx] = 1
    return e


def _cached(cache, s, decode, cap):
    """decode(s), memoized in a bounded dict keyed by the syndrome bytes (cap 0 = off)."""
    if not cap:
        return decode(s)
    key = s.tobytes()
    c = cache.get(key)
    if c is None:
        c = decode(s)
        cache[key] = c
        if len(cache) > cap:
            cache.clear()
    return c


def _depol_parts(idx: np.ndarray, types: np.ndarray, n: int) -> tuple:
    """(e_X, e_Z) uint8 (b, n) arrays from support indices and Pauli types 0=X, 1=Y, 2=Z."""
    b = idx.shape[0]
    sel_x = (types <= 1).astype(np.uint8)          # X or Y
    sel_z = (types >= 1).astype(np.uint8)          # Y or Z
    ex = np.zeros((b, n), dtype=np.uint8)
    ez = np.zeros_like(ex)
    rows = np.arange(b)
    for j in range(idx.shape[1]):                  # one column of the support at a time
        ex[rows, idx[:, j]] = sel_x[:, j]
        ez[rows, idx[:, j]] = sel_z[:, j]
    return ex, ez


def _fail_mask_depol(idx: np.ndarray, ctx: Ctx, types: np.ndarray) -> np.ndarray:
    """Depolarizing failure mask: X part decoded from H_Z e_X, Z part from H_X e_Z,
    faulty if either logical test fails (the direct_mc.DepolarizingAdapter definition)."""
    if types is None:
        raise ValueError("depolarizing fail_mask needs the Pauli types (b, w)")
    b = idx.shape[0]
    ex, ez = _depol_parts(idx, types, ctx.n)
    s_z = (ex @ ctx.hzT) & 1        # (b, m_z)  X-part syndrome
    s_x = (ez @ ctx.hxT) & 1        # (b, m_x)  Z-part syndrome
    if ctx.matching is not None:
        c_x = ctx.matching.decode_batch(s_z.astype(np.uint8)).astype(np.uint8)
        c_z = ctx.matching_z.decode_batch(s_x.astype(np.uint8)).astype(np.uint8)
        return (((ex ^ c_x) @ ctx.lzT & 1).any(axis=1)
                | ((ez ^ c_z) @ ctx.lxT & 1).any(axis=1))
    out = np.zeros(b, dtype=bool)
    for i in range(b):
        c_x = _cached(ctx.cache, s_z[i].astype(np.uint8), ctx.decoder.decode_x, ctx.cache_cap)
        c_z = _cached(ctx.cache_z, s_x[i].astype(np.uint8), ctx.decoder.decode_z, ctx.cache_cap)
        out[i] = bool(ctx.decoder.is_faulty(ex[i].astype(np.int8), ez[i].astype(np.int8),
                                            c_x, c_z))
    return out


def fail_mask(idx: np.ndarray, ctx: Ctx, types: np.ndarray | None = None) -> np.ndarray:
    """Boolean failure mask for a batch of weight-w patterns given as support indices.

    Fast path (MWPM): one pymatching decode_batch + a vectorized logical test.
    Generic path: decoder.decode_syndrome per distinct syndrome, memoized in a
    bounded dict keyed by the syndrome bytes, then decoder.is_faulty per shot --
    which keeps decoder-specific failure definitions (e.g. plain BP counts
    non-convergence as a failure) intact.

    Under depolarizing noise (ctx.noise) `types` (b, w) gives the Pauli on each
    support qubit, 0=X, 1=Y, 2=Z; see _fail_mask_depol.
    """
    b = idx.shape[0]
    if b == 0:
        return np.zeros(0, dtype=bool)
    if ctx.noise != "bit_flip":
        return _fail_mask_depol(idx, ctx, types)
    synd = _syndromes(idx, ctx)
    if ctx.matching is not None:
        e = _errors(idx, ctx)
        corr = ctx.matching.decode_batch(synd).astype(np.uint8)
        return ((e ^ corr) @ ctx.lzT & 1).any(axis=1)

    e = _errors(idx, ctx, dtype=np.int8)
    cache = ctx.cache if ctx.cache_cap else None
    out = np.zeros(b, dtype=bool)
    decode = ctx.decoder.decode_syndrome
    faulty = ctx.decoder.is_faulty
    for i in range(b):
        s = synd[i]
        if cache is None:
            c = decode(s)
        else:
            key = s.tobytes()
            c = cache.get(key)
            if c is None:
                c = decode(s)
                cache[key] = c
                if len(cache) > ctx.cache_cap:
                    cache.clear()
        out[i] = bool(faulty(e[i], c))
    return out


def fail_count(idx: np.ndarray, ctx: Ctx, types: np.ndarray | None = None) -> int:
    """Number of weight-w patterns in the batch that the decoder fails on."""
    return int(fail_mask(idx, ctx, types).sum())


# ── strata ─────────────────────────────────────────────────────────────────
def plan_strata(n: int, w_lo: int, w_max: int, enum_cap: int, type_mult: int = 1) -> list:
    """[(w, mode, class size)] for w in [w_lo, min(w_max, n)], mode in {enum, mc}.

    The class size is C(n, w) * type_mult**w: type_mult = 1 for bit flips, 3 for
    depolarizing noise (each support qubit carries X, Y or Z).
    """
    out = []
    for w in range(max(0, w_lo), min(w_max, n) + 1):
        c = math.comb(n, w) * type_mult ** w
        out.append((w, "enum" if c <= enum_cap else "mc", c))
    return out


def enum_stratum(ctx: Ctx, w: int, start: int, stop: int) -> tuple:
    """Enumerate combinations [start, stop) of C(n, w); return (n_seen, n_fail).

    Under depolarizing noise every combination is expanded into its 3**w Pauli
    assignments, so n_seen counts (support, type) patterns; start/stop still
    index supports.
    """
    it = itertools.islice(itertools.combinations(range(ctx.n), w), start, stop)
    n_seen = n_fail = 0
    if ctx.noise != "bit_flip":
        types_all = np.array(list(itertools.product(range(3), repeat=w)), dtype=np.int8)
        per = max(1, ENUM_BATCH // len(types_all))
        while True:
            flat = np.fromiter(
                itertools.chain.from_iterable(itertools.islice(it, per)), dtype=np.int32)
            if flat.size == 0:
                break
            idx = flat.reshape(-1, w)
            idx_rep = np.repeat(idx, len(types_all), axis=0)
            typ_rep = np.tile(types_all, (idx.shape[0], 1))
            n_seen += idx_rep.shape[0]
            n_fail += fail_count(idx_rep, ctx, typ_rep)
        return n_seen, n_fail
    while True:
        flat = np.fromiter(
            itertools.chain.from_iterable(itertools.islice(it, ENUM_BATCH)),
            dtype=np.int32,
        )
        if flat.size == 0:
            break
        idx = flat.reshape(-1, w)
        n_seen += idx.shape[0]
        n_fail += fail_count(idx, ctx)
    return n_seen, n_fail


def mc_stratum(ctx: Ctx, w: int, n_samples: int, seed_key) -> tuple:
    """Draw n_samples uniform weight-w patterns; return (n_samples, n_fail).

    Under depolarizing noise each support qubit also draws a uniform Pauli type.
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed_key))
    n_fail = 0
    left = int(n_samples)
    while left > 0:
        b = min(MC_BATCH, left)
        idx = np.argpartition(rng.random((b, ctx.n)), w - 1, axis=1)[:, :w]
        idx = np.ascontiguousarray(idx.astype(np.int32))
        if ctx.noise != "bit_flip":
            types = rng.integers(0, 3, size=(b, w)).astype(np.int8)
            n_fail += fail_count(idx, ctx, types)
        else:
            n_fail += fail_count(idx, ctx)
        left -= b
    return int(n_samples), n_fail


# ── statistics ─────────────────────────────────────────────────────────────
def ac_se(k: int, n: int, z: float = 1.959963984540054) -> float:
    """Agresti-Coull standard error for a binomial proportion (k = 0 safe)."""
    if n <= 0:
        return float("nan")
    n_t = n + z * z
    f_t = (k + z * z / 2.0) / n_t
    return math.sqrt(f_t * (1.0 - f_t) / n_t)


def _log_binom_pmf(n: int, w: int, p: float) -> float:
    return (
        math.lgamma(n + 1) - math.lgamma(w + 1) - math.lgamma(n - w + 1)
        + w * math.log(p) + (n - w) * math.log1p(-p)
    )


def binom_pmf(n: int, w: int, p: float) -> float:
    return math.exp(_log_binom_pmf(n, w, p))


def tail_bound(n: int, w_max: int, p: float) -> float:
    """sum_{w>w_max} B(w; n, p), term-by-term in log space (no cancellation)."""
    return math.fsum(binom_pmf(n, w, p) for w in range(w_max + 1, n + 1))


def synthesize(n: int, strata: list, p: float) -> dict:
    """P_L, E[w|F], V[w|F] and delta-method SEs at physical rate p.

    `strata` = list of dicts with keys w, mode, f_w, f_w_se.
    """
    b = np.array([binom_pmf(n, s["w"], p) for s in strata])
    f = np.array([s["f_w"] for s in strata])
    sf = np.array([s["f_w_se"] for s in strata])
    ws = np.array([float(s["w"]) for s in strata])

    bf = b * f
    s0 = bf.sum()
    s1 = (ws * bf).sum()
    s2 = (ws * ws * bf).sum()
    p_l_se = math.sqrt(float(((b * sf) ** 2).sum()))
    if s0 <= 0:
        nan = float("nan")
        return dict(p_L=0.0, p_L_se=p_l_se, ew_mean=nan, ew_se=nan, vw=nan, vw_se=nan,
                    frac_mc=nan)

    ew = s1 / s0
    e_w2 = s2 / s0
    vw = e_w2 - ew * ew
    d_ew = b * (ws - ew) / s0
    ew_se = math.sqrt(float(((d_ew * sf) ** 2).sum()))
    d_vw = (b / s0) * (ws * ws - e_w2 - 2.0 * ew * (ws - ew))
    vw_se = math.sqrt(float(((d_vw * sf) ** 2).sum()))

    is_mc = np.array([s["mode"] == "mc" for s in strata])
    frac_mc = float(bf[is_mc].sum() / s0)
    return dict(p_L=float(s0), p_L_se=p_l_se, ew_mean=float(ew), ew_se=ew_se,
                vw=float(vw), vw_se=vw_se, frac_mc=frac_mc)


# ── independent cross-check of the vectorized pipeline ─────────────────────
def verify_against_repo(ctx: Ctx, n_shots: int = 2000, seed: int = 0, w: int | None = None) -> bool:
    """Assert fail_mask reproduces the repo's per-shot, UNCACHED reference path.

    For MWPM this is decode_batch vs decoder.decode_syndrome (incl. tie-breaking).
    For the generic path the reference is decoded fresh every shot, so this also
    catches a broken syndrome-bytes cache key (the failure mode that a
    self-comparison of the generic path against itself would miss).
    """
    if w is None:
        d_code = code_distance(ctx.code_type, ctx.d)
        w = (d_code + 1) // 2 if d_code else 3
    w = max(1, min(int(w), ctx.n))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 99, ctx.n, w]))
    idx = np.argpartition(rng.random((n_shots, ctx.n)), w - 1, axis=1)[:, :w]
    idx = np.ascontiguousarray(idx.astype(np.int32))

    if ctx.noise != "bit_flip":
        # Reference: the direct-MC depolarizing adapter, uncached, shot by shot.
        types = rng.integers(0, 3, size=(n_shots, w)).astype(np.int8)
        vec = fail_mask(idx, ctx, types)
        ex, ez = _depol_parts(idx, types, ctx.n)
        adapter = direct_mc.DepolarizingAdapter(ctx.code, ctx.decoder, ctx.p_decoder,
                                                use_cache=False)
        for i in range(n_shots):
            ref, w_ref = adapter.decode_and_check((ex[i].astype(np.int8), ez[i].astype(np.int8)))
            assert w_ref == w, f"support weight mismatch at shot {i}"
            assert bool(ref) == bool(vec[i]), (
                f"failure verdict differs at shot {i} ({ctx.code_type} d={ctx.d}, w={w}, "
                "depolarizing) -- vectorized/cached path disagrees with the repo per-shot path")
        return True

    synd = _syndromes(idx, ctx)
    e = _errors(idx, ctx, dtype=np.int8)
    vec = fail_mask(idx, ctx)

    hz = ctx.code.hz
    for i in range(n_shots):
        s_ref = (hz @ e[i] % 2).astype(np.uint8)
        assert np.array_equal(s_ref, synd[i]), (
            f"syndrome mismatch at shot {i} ({ctx.code_type} d={ctx.d})")
        c_ref = ctx.decoder.decode_syndrome(s_ref)
        assert bool(ctx.decoder.is_faulty(e[i], c_ref)) == bool(vec[i]), (
            f"failure verdict differs at shot {i} ({ctx.code_type} d={ctx.d}, w={w}) "
            "-- vectorized/cached path disagrees with the repo per-shot path")
    return True


# ── multiprocessing plumbing ───────────────────────────────────────────────
_CTX: Ctx | None = None


def _init_worker(spec):
    global _CTX
    _CTX = build_ctx(*spec)


def _enum_job(args):
    w, start, stop = args
    n_seen, n_fail = enum_stratum(_CTX, w, start, stop)
    return w, n_seen, n_fail


def _mc_job(args):
    w, n_samples, seed_key = args
    n_s, n_fail = mc_stratum(_CTX, w, n_samples, seed_key)
    return w, n_s, n_fail


def _verify_job(args):
    n_shots, seed, w = args
    return verify_against_repo(_CTX, n_shots, seed, w)


def _chunk_ranges(total: int, n_chunks: int):
    step = max(1, math.ceil(total / n_chunks))
    return [(s, min(s + step, total)) for s in range(0, total, step)]


# ── driver ─────────────────────────────────────────────────────────────────
def run_code(noise, code_type, d, p_decoder, *, w_max=16, enum_cap=ENUM_CAP,
             n_max=N_MAX, target_fail=TARGET_FAIL, pilot_n=PILOT_N, workers=1,
             seed=20260828, cache_cap=CACHE_CAP, decoder_kind="auto", w_lo=None,
             verify=True, verify_shots=500, progress=True) -> dict:
    """Measure the f_w table for one (code, decoder-prior) condition.

    Returns dict(fw_rows, n, w_lo, w_max, wall, code_type, d, decoder,
    n_samples_total, n_fail_total).  `d` may be an int or a "d1xd2" string.
    """
    t0 = time.perf_counter()
    d_str = parse_d(d)
    spec = (noise, code_type, d_str, float(p_decoder), decoder_kind, int(cache_cap), False)

    ctx0 = build_ctx(*spec)
    n = ctx0.n
    d_code = code_distance(code_type, d_str)
    mwpm = is_mwpm(code_type, decoder_kind)
    p_indep = is_p_independent(code_type, decoder_kind)

    if w_lo is None:
        # Start one stratum BELOW the first weight that can fail, so every run
        # exercises the "provably zero failures" assert.  Only MWPM has a
        # min-weight guarantee to skip up to; every other decoder (BP-OSD, and
        # concat_steane's hard decoder, whose correctable weight is well below
        # (d-1)//2) starts at w=1 and enumerates the cheap low strata.
        w_lo = max(1, (d_code - 1) // 2) if (mwpm and d_code) else 1
    type_mult = 3 if noise == "depolarizing" else 1
    plan = plan_strata(n, w_lo, w_max, enum_cap, type_mult)
    if not plan:
        raise ValueError(
            f"empty stratum plan for {code_type} d={d_str}: w_max={w_max} is below "
            f"w_lo={w_lo} (n={n}), so every stratum with f_w > 0 would be truncated "
            "away and P_L would come out 0 with no warning")

    enum_tasks = []
    for w, mode, c in plan:
        if mode != "enum":
            continue
        # Chunks index SUPPORTS (enum_stratum expands each into type_mult**w
        # patterns), so the range is C(n, w) while the work is the class size c.
        n_comb = c // type_mult ** w
        n_chunks = min(4 * max(workers, 1), max(1, math.ceil(c / ENUM_CHUNK_MIN)), n_comb)
        enum_tasks += [(w, a, b) for a, b in _chunk_ranges(n_comb, n_chunks)]
    pilot_tasks = [(w, pilot_n, [seed, 1, n, w]) for w, mode, _ in plan if mode == "mc"]

    if progress:
        n_enum = sum(1 for x in plan if x[1] == "enum")
        print(f"  [{code_type} d={d_str} p_dec={p_decoder:g}] n={n} "
              f"w={w_lo}..{min(w_max, n)}  strata {len(plan)} "
              f"(enum {n_enum}, mc {len(plan) - n_enum})  "
              f"enum chunks {len(enum_tasks)}  workers {workers}", flush=True)

    acc: dict = {}
    timings = dict(t_verify=0.0, t_enum=0.0, t_pilot=0.0, t_mc=0.0)

    def _acc(results):
        for w, ns, nf in results:
            s, f = acc.get(w, (0, 0))
            acc[w] = (s + ns, f + nf)

    use_pool = workers and workers > 1
    if use_pool:
        pool = Pool(workers, initializer=_init_worker, initargs=(spec,))
        _map = lambda fn, it: pool.map(fn, it, chunksize=1)  # noqa: E731
    else:
        _init_worker(spec)
        pool = None
        _map = lambda fn, it: [fn(x) for x in it]  # noqa: E731

    try:
        if verify:
            t_v = time.perf_counter()
            assert _map(_verify_job, [(verify_shots, seed, None)])[0]
            timings["t_verify"] = time.perf_counter() - t_v
            if progress:
                print(f"    verify: vectorized/cached mask == repo per-shot path "
                      f"({verify_shots} shots, {time.perf_counter() - t_v:.1f}s)", flush=True)

        if enum_tasks:
            t_e = time.perf_counter()
            _acc(_map(_enum_job, enum_tasks))
            timings["t_enum"] = time.perf_counter() - t_e
            if progress:
                print(f"    enum done in {time.perf_counter() - t_e:.1f}s", flush=True)

        pilots = {}
        if pilot_tasks:
            t_p = time.perf_counter()
            for w, ns, nf in _map(_mc_job, pilot_tasks):
                pilots[w] = (ns, nf)
            timings["t_pilot"] = time.perf_counter() - t_p
            if progress:
                print(f"    pilots done in {time.perf_counter() - t_p:.1f}s", flush=True)

        mc_tasks = []
        for w, mode, _c in plan:
            if mode != "mc":
                continue
            ns, nf = pilots[w]
            f_hat = (nf + 1.0) / (ns + 1.0)
            n_run = int(min(n_max, max(pilot_n, math.ceil(1.5 * target_fail / f_hat))))
            n_chunks = min(4 * max(workers, 1), max(1, math.ceil(n_run / MC_CHUNK_MIN)))
            per = math.ceil(n_run / n_chunks)
            for j in range(n_chunks):
                take = min(per, n_run - j * per)
                if take > 0:
                    mc_tasks.append((w, take, [seed, 2, n, w, j]))
        if mc_tasks:
            t_m = time.perf_counter()
            _acc(_map(_mc_job, mc_tasks))
            timings["t_mc"] = time.perf_counter() - t_m
            if progress:
                print(f"    mc done in {time.perf_counter() - t_m:.1f}s "
                      f"({sum(t[1] for t in mc_tasks):,} samples)", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # ── assemble the f_w table ────────────────────────────────────────────
    w_assert = assert_wmax(code_type, d_str, decoder_kind)

    fw_rows = []
    # Blank whenever the table is valid at every p; the CLI resumes on this key.
    p_dec_col = "" if p_indep else f"{float(p_decoder):g}"
    for w, mode, c in plan:
        ns, nf = acc[w]
        if mode == "enum":
            assert ns == c, (
                f"enum coverage mismatch {code_type} d={d_str} w={w}: {ns} != "
                f"C({n},{w})*{type_mult}^{w}={c}")
            f_w, se = nf / ns, 0.0
        else:
            f_w, se = nf / ns, ac_se(nf, ns)
        fw_rows.append(dict(code_type=code_type, d=d_str, p_decoder=p_dec_col, n=n,
                            w=w, mode=mode, n_samples=ns, n_fail=nf,
                            f_w=f_w, f_w_se=se))
        if w <= w_assert:
            assert nf == 0, (
                f"THEOREM VIOLATED: {code_type} d={d_str} p_dec={p_decoder:g} has {nf} "
                f"failure(s) at w={w} <= {w_assert} (n={n}, assumed distance "
                f"{d_code}). Check the code_distance() map for this family first, "
                "then the hz/lz pairing, residual and index alignment.")

    wall = time.perf_counter() - t0
    return dict(fw_rows=fw_rows, n=n, w_lo=w_lo, w_max=min(w_max, n), wall=wall,
                code_type=code_type, d=d_str, p_decoder=float(p_decoder),
                decoder=decoder_name(code_type, decoder_kind),
                n_samples_total=sum(r["n_samples"] for r in fw_rows),
                n_fail_total=sum(r["n_fail"] for r in fw_rows),
                n_enum_samples=sum(r["n_samples"] for r in fw_rows if r["mode"] == "enum"),
                n_mc_samples=sum(r["n_samples"] for r in fw_rows if r["mode"] == "mc"),
                workers=max(int(workers or 1), 1), w_assert=w_assert, **timings)


def rows_for_p_grid(fw_rows: list, n: int, ps, meta: dict) -> list:
    """direct_mc.CSV_HEADER rows synthesized from an f_w table over a p grid.

    meta keys: noise, code_type, d, decoder, n_cores, n_target, wall_clock, w_max.
    n_trials / n_faulty are the TOTAL stratified sample and failure counts (they
    are not direct-MC trial counts; they document the cost of the row).
    p_L_se is floored at the truncation bound sum_{w>w_max} B(w; n, p).
    """
    w_max = int(meta.get("w_max", max(r["w"] for r in fw_rows)))
    tot_s = sum(r["n_samples"] for r in fw_rows)
    tot_f = sum(r["n_fail"] for r in fw_rows)
    rows = []
    for p in ps:
        est = synthesize(n, fw_rows, float(p))
        tb = tail_bound(n, min(w_max, n), float(p))
        p_l_se = max(est["p_L_se"], tb)
        row = {
            "noise": meta.get("noise", "bit_flip"),
            "code_type": meta["code_type"],
            "d": meta["d"],
            "p": float(p),
            "decoder": meta["decoder"],
            "n_cores": int(meta.get("n_cores", 1)),
            "n_target": int(meta.get("n_target", 0)),
            "n_trials": tot_s,
            "n_faulty": tot_f,
            "p_L": est["p_L"],
            "p_L_se": p_l_se,
            "ew_mean": est["ew_mean"],
            "ew_se": est["ew_se"],
            "vw": est["vw"],
            "vw_se": est["vw_se"],
            "wall_clock": float(meta.get("wall_clock", 0.0)),
            "feasible": bool(est["p_L"] > 0.0 and math.isfinite(est["ew_mean"])),
        }
        rows.append({k: row[k] for k in direct_mc.CSV_HEADER})
    return rows
