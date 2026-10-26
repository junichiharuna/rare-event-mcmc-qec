"""Shared helpers for the P1-P6 numerics package (cluster-side; imports src/ only).

Used by cluster_scripts/p{1,2,3,4,5}_*.py runners. Keep this module free of
compare/ or docs/ imports: the cluster sparse checkout only has src/,
cluster_scripts/ and cluster_results/.
"""
import csv
import io
import os

import numpy as np

from css_code import CSSCode


def reorder_logicals(code, first_idx):
    """Return a CSSCode whose logical rows are permuted so row `first_idx` is first.

    run_rimc_mixed initialises every replica at lx[0], so this selects the
    starting logical sector (same helper as general_code_audit / depol_audit).
    """
    order = [first_idx] + [i for i in range(code.lx.shape[0]) if i != first_idx]
    return CSSCode(hx=code.hx, hz=code.hz,
                   lx=code.lx[order].copy(), lz=code.lz[order].copy(), n=code.n)


def minmax_inits(code):
    """[("minw", argmin), ("maxw", argmax)] over logical-X row weights,
    or [("uniform", argmin)] when all rows have equal weight."""
    lwts = [int(r.sum()) for r in code.lx]
    lo, hi = int(np.argmin(lwts)), int(np.argmax(lwts))
    if min(lwts) == max(lwts):
        return [("uniform", lo)]
    return [("minw", lo), ("maxw", hi)]


def append_rows(path, fields, rows):
    """Append dict rows to a CSV, writing the header iff the file is new/empty."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()


def load_done_keys(path, key_cols):
    """Set of stringified key tuples already present in a CSV (resume support).

    Values are compared as str(v) of what the task generator produces, matching
    csv round-tripping: use str(p) / str(seed) when building task keys.
    """
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {tuple(r[c] for c in key_cols) for r in csv.DictReader(f)}


def read_complete_rows(path, fields, int_cols=(), float_cols=()):
    """Complete rows of an append-only resume CSV, as the raw string dicts.

    A row is complete iff it has exactly the columns of `fields`, every value is
    non-empty, every `int_cols` value parses with int() and every `float_cols` value
    with float(). A final line without its newline (a kill mid-write) is dropped
    before parsing, since its last field may be a truncated but parseable number.
    Returns (rows, n_dropped). Raises ValueError if the header is not `fields`.
    """
    with open(path, newline="") as f:
        text = f.read()
    if text and not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None:
        return [], 0
    if list(reader.fieldnames) != list(fields):
        raise ValueError(f"{path}: header {reader.fieldnames} != expected {list(fields)}")
    rows, dropped = [], 0
    for r in reader:
        ok = None not in r and all(r.get(c) not in (None, "") for c in fields)
        try:
            if ok:
                for c in int_cols:
                    int(r[c])
                for c in float_cols:
                    float(r[c])
        except ValueError:
            ok = False
        if ok:
            rows.append(r)
        else:
            dropped += 1
    return rows, dropped


def rewrite_rows_atomic(path, fields, rows):
    """Rewrite `path` as header + rows via path.tmp, flush + fsync, os.replace."""
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({c: r[c] for c in fields} for r in rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)          # a kill mid-rewrite never loses finished rows


def mwpm_truth_chunk(code, p, m_max, seed, batch=65536):
    """Direct-MC chunk with batched MWPM decoding (fast path for matchable codes).

    Samples m_max iid bit-flip errors at rate p, decodes with pymatching
    decode_batch, and returns (n_trials, n_faulty, s1, s2) where s1/s2 are the
    sums of w and w^2 over faulty samples (sufficient stats for E[w|F], V[w|F],
    p_L = n_faulty/n_trials).
    """
    import pymatching
    m = pymatching.Matching(code.hz)
    lz, hz, n = code.lz, code.hz, code.n
    rng = np.random.default_rng(seed)
    nf, s1, s2, done = 0, 0.0, 0.0, 0
    while done < m_max:
        b = int(min(batch, m_max - done))
        e = (rng.random((b, n)) < p).astype(np.uint8)
        corr = m.decode_batch((e @ hz.T) % 2).astype(np.uint8)
        faulty = (((e ^ corr) % 2) @ lz.T % 2).any(axis=1)
        w = e.sum(axis=1)[faulty].astype(np.float64)
        nf += int(faulty.sum())
        s1 += float(w.sum())
        s2 += float((w * w).sum())
        done += b
    return int(m_max), nf, s1, s2


def binom_truth_chunk(code, decoder, p, m_max, seed, cache_cap=200_000, batch=1_000_000):
    """Direct-MC chunk for generic decoders, sampling the WEIGHT first (rare-event fast path).

    Statistically identical to persample_truth_chunk: k ~ Binomial(n, p) followed by a
    uniformly random support of size k IS an iid Bernoulli(p) error. The point is that the
    zero-weight samples -- 99.3% of them at bb72/p=1e-4 -- are never materialised: e = 0 has
    syndrome 0 and can never be faulty, so it only increments n_trials. Same construction as
    direct_mc.BitFlipAdapter(sampler="binomial") ("exact Bernoulli(p) marginal"), batched so
    the weights come from one vectorised numpy draw.

    Measured on bb72/BP-OSD at p=1e-4: ~5e6 trials/s/core against ~5e4 for
    persample_truth_chunk, i.e. the difference between a 2e12-trial truth point costing
    ~1 h and ~116 h on one 96-core node.

    Same return contract as mwpm_truth_chunk: (n_trials, n_faulty, s1, s2).
    """
    hz, n, m_max = code.hz, code.n, int(m_max)
    rng = np.random.default_rng(seed)
    lookup = {}
    nf, s1, s2, done = 0, 0.0, 0.0, 0
    while done < m_max:
        b = int(min(batch, m_max - done))
        ks = rng.binomial(n, p, size=b)
        for i in np.flatnonzero(ks):
            w = int(ks[i])
            e = np.zeros(n, dtype=np.int8)
            e[rng.choice(n, size=w, replace=False)] = 1
            s = (hz @ e % 2).astype(np.uint8)
            key = s.tobytes()                 # bytes as the KEY only; decode gets the array
            c = lookup.get(key)
            if c is None:
                if len(lookup) > cache_cap:
                    lookup.clear()
                c = decoder.decode_syndrome(s)
                lookup[key] = c
            if decoder.is_faulty(e, c):
                nf += 1
                s1 += w
                s2 += w * w
        done += b
    return m_max, nf, s1, s2


def persample_truth_chunk(code, decoder, p, m_max, seed, cache_cap=200_000):
    """Direct-MC chunk decoding one sample at a time (BP-OSD / generic decoders).

    Same return contract as mwpm_truth_chunk. A syndrome->correction cache with
    clear-on-full eviction keeps repeated low-weight syndromes cheap.
    """
    hz, lz, n = code.hz, code.lz, code.n
    rng = np.random.default_rng(seed)
    lookup = {}
    nf, s1, s2 = 0, 0.0, 0.0
    for _ in range(int(m_max)):
        e = (rng.random(n) < p).astype(np.int8)
        key = tuple((hz @ e) % 2)
        if key not in lookup:
            if len(lookup) > cache_cap:
                lookup.clear()
            lookup[key] = decoder.decode_syndrome(np.array(key, dtype=np.uint8))
        if decoder.is_faulty(e, lookup[key]):
            w = float(e.sum())
            nf += 1
            s1 += w
            s2 += w * w
    return int(m_max), nf, s1, s2
