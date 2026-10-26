"""Direct (rejection) Monte Carlo for the faulty-error manifold.

Estimates logical error probability p_L = P(faulty), E[w|faulty] and V[w|faulty]
for a CSS code by drawing i.i.d. physical errors, decoding, and keeping faulty
samples. Unbiased and free of MCMC autocorrelation/trapping; feasible only where
P(faulty) is not astronomically small (high p / multimodal-hard codes).
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import namedtuple
from multiprocessing import Pool

import numpy as np

from css_code import (
    make_toric_code,
    make_surface_code,
    make_asymmetric_toric_code,
    make_bb_code_72,
    make_bb_code_144,
    make_mwpm_decoder,
    make_bposd_decoder,
    make_bp_decoder,
    make_depolarizing_mwpm_decoder,
    make_depolarizing_bposd_decoder,
    make_concatenated_steane_code,
    make_concat_steane_hard_decoder,
)

# Sufficient statistics accumulated over a sampling run (weight moments over the
# faulty subset). Pooling across cores is field-wise addition.
Stats = namedtuple("Stats", "n_trials n_faulty s1 s2 s3 s4")


def pool_stats(stats_list):
    """Sum a list of Stats field-wise."""
    acc = [0, 0, 0.0, 0.0, 0.0, 0.0]
    for s in stats_list:
        for i in range(6):
            acc[i] += s[i]
    return Stats(*acc)


def stats_to_dict(st, meta):
    """Merge a Stats tuple with a metadata dict into one JSON-serialisable record."""
    rec = dict(meta)
    rec.update(
        n_trials=int(st.n_trials),
        n_faulty=int(st.n_faulty),
        s1=float(st.s1),
        s2=float(st.s2),
        s3=float(st.s3),
        s4=float(st.s4),
    )
    return rec


def dict_to_stats(rec):
    """Reconstruct a Stats tuple from a record produced by stats_to_dict."""
    return Stats(
        int(rec["n_trials"]),
        int(rec["n_faulty"]),
        float(rec["s1"]),
        float(rec["s2"]),
        float(rec["s3"]),
        float(rec["s4"]),
    )


def write_stats_jsonl(records, path):
    """Append records (a list of dicts, or a single dict) as JSON lines.

    Safe to call repeatedly; each call appends. Callers that want a fresh file
    delete it first (the PBS subjob does `rm -f` before running).
    """
    if isinstance(records, dict):
        records = [records]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def read_stats_jsonl(path):
    """Read all JSON-line records from a file into a list of dicts."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def estimate(st):
    """Estimators + standard errors from pooled Stats.

    Returns dict(p_L, p_L_se, ew_mean, ew_se, vw, vw_se). E[w]/V[w] are NaN when
    fewer than 2 faulty samples were collected.
    """
    M, F = st.n_trials, st.n_faulty
    p_L = F / M if M > 0 else float("nan")
    p_L_se = math.sqrt(p_L * (1.0 - p_L) / M) if M > 0 else float("nan")

    if F < 2:
        nan = float("nan")
        return dict(p_L=p_L, p_L_se=p_L_se, ew_mean=nan, ew_se=nan, vw=nan, vw_se=nan)

    mean = st.s1 / F
    m2 = max(st.s2 / F - mean**2, 0.0)  # population 2nd central moment
    var = (F / (F - 1)) * m2  # Bessel-corrected sample variance
    ew_se = math.sqrt(var / F)

    mu4 = st.s4 / F - 4 * mean * (st.s3 / F) + 6 * mean**2 * (st.s2 / F) - 3 * mean**4
    vw_se = math.sqrt(max(mu4 - ((F - 3) / (F - 1)) * m2**2, 0.0) / F)

    return dict(p_L=p_L, p_L_se=p_L_se, ew_mean=mean, ew_se=ew_se, vw=var, vw_se=vw_se)


def run_core(adapter, rng, target, m_max, time_budget=None):
    """Sample until `target` faulty configs collected, `m_max` trials reached,
    or (if set) `time_budget` seconds elapsed.

    `adapter` provides sample(rng) and decode_and_check(state) -> (faulty, w).
    The wall-clock check is throttled to every 65536 trials so it adds no
    measurable overhead to the hot loop. Returns Stats over the faulty subset.
    """
    n_trials = 0
    n_faulty = 0
    s1 = s2 = s3 = s4 = 0.0
    sample = adapter.sample
    check = adapter.decode_and_check
    t0 = time.perf_counter() if time_budget is not None else 0.0
    while n_faulty < target and n_trials < m_max:
        faulty, w = check(sample(rng))
        n_trials += 1
        if faulty:
            n_faulty += 1
            w2 = w * w
            s1 += w
            s2 += w2
            s3 += w2 * w
            s4 += w2 * w2
        if time_budget is not None and (n_trials & 0xFFFF) == 0:
            if time.perf_counter() - t0 > time_budget:
                break
    return Stats(n_trials, n_faulty, s1, s2, s3, s4)


_BB = ("bb72", "bb144")


class BitFlipAdapter:
    """Bit-flip (X-error) noise: e ~ Bernoulli(p), decode Z-syndrome, w = |e|."""

    def __init__(self, code, decoder, p, use_cache=True, sampler="bernoulli", cache_cap=None):
        self.n = code.n
        self.Hz = code.hz.astype(np.int8)
        self.decoder = decoder
        self.p = p
        self.sampler = sampler
        self.cache_cap = cache_cap
        self._zero = np.zeros(self.n, dtype=np.int8)  # shared (read-only) for k=0
        self.lookup = {} if use_cache else None

    def sample(self, rng):
        if self.sampler == "binomial":
            k = int(rng.binomial(self.n, self.p))  # exact Bernoulli(p) marginal
            if k == 0:
                return self._zero  # syndrome 0 -> not faulty (cached)
            e = np.zeros(self.n, dtype=np.int8)
            e[rng.choice(self.n, size=k, replace=False)] = 1
            return e
        return (rng.random(self.n) < self.p).astype(np.int8)

    def _decode(self, s):
        if self.lookup is None:
            return self.decoder.decode_syndrome(s)
        key = s.tobytes()
        c = self.lookup.get(key)
        if c is None:
            c = self.decoder.decode_syndrome(s)
            self.lookup[key] = c
            if self.cache_cap is not None and len(self.lookup) > self.cache_cap:
                self.lookup.clear()
        return c

    def decode_and_check(self, e):
        if not e.any():                 # zero error -> syndrome 0, never faulty; skip matmul+decode
            return False, 0
        s = (self.Hz @ e % 2).astype(np.uint8)
        c = self._decode(s)
        return self.decoder.is_faulty(e, c), int(e.sum())


class DepolarizingAdapter:
    """Depolarizing noise: each qubit I/X/Y/Z = (1-p, p/3, p/3, p/3).

    ex=1 iff X or Y, ez=1 iff Z or Y, w = |ex OR ez|. Intervals on u~U[0,1):
    I=[0,1-p), X=[1-p,1-2p/3), Z=[1-2p/3,1-p/3), Y=[1-p/3,1).
    """

    def __init__(self, code, decoder, p, use_cache=True):
        self.n = code.n
        self.Hz = code.hz.astype(np.int8)
        self.Hx = code.hx.astype(np.int8)
        self.decoder = decoder
        self.p = p
        self.lookup_x = {} if use_cache else None
        self.lookup_z = {} if use_cache else None

    def sample(self, rng):
        u = rng.random(self.n)
        p = self.p
        ex = (((u >= 1 - p) & (u < 1 - 2 * p / 3)) | (u >= 1 - p / 3)).astype(np.int8)
        ez = (u >= 1 - 2 * p / 3).astype(np.int8)
        return ex, ez

    @staticmethod
    def _decode(lookup, s, fn):
        if lookup is None:
            return fn(s)
        key = s.tobytes()
        c = lookup.get(key)
        if c is None:
            c = fn(s)
            lookup[key] = c
        return c

    def decode_and_check(self, state):
        ex, ez = state
        s_z = (self.Hz @ ex % 2).astype(np.uint8)
        s_x = (self.Hx @ ez % 2).astype(np.uint8)
        c_x = self._decode(self.lookup_x, s_z, self.decoder.decode_x)
        c_z = self._decode(self.lookup_z, s_x, self.decoder.decode_z)
        w = int((ex | ez).sum())
        return self.decoder.is_faulty(ex, ez, c_x, c_z), w


def _make_code_and_decoder(noise, code_type, d, p, decoder_kind="auto"):
    if code_type == "toric":
        code = make_toric_code(d)
    elif code_type == "surface":
        code = make_surface_code(d)
    elif code_type == "asymmetric_toric":
        code = make_asymmetric_toric_code(d, d)
    elif code_type == "bb72":
        code = make_bb_code_72()
    elif code_type == "bb144":
        code = make_bb_code_144()
    elif code_type == "concat_steane":
        code = make_concatenated_steane_code(int(d))  # d = number of levels
    else:
        raise ValueError(f"Unknown code_type: {code_type}")

    if code_type == "concat_steane":
        if noise != "bit_flip":
            raise ValueError("concat_steane currently supports only bit_flip noise")
        return code, make_concat_steane_hard_decoder(int(d))

    is_bb = code_type in _BB
    if decoder_kind == "bp":
        if noise != "bit_flip":
            raise ValueError("decoder_kind='bp' is only supported for bit_flip noise")
        decoder = make_bp_decoder(code, p)
    elif decoder_kind == "mwpm":
        if noise != "bit_flip":
            raise ValueError("decoder_kind='mwpm' is only supported for bit_flip noise")
        decoder = make_mwpm_decoder(code)
    elif decoder_kind == "auto":
        if noise == "bit_flip":
            decoder = make_bposd_decoder(code, p) if is_bb else make_mwpm_decoder(code)
        elif noise == "depolarizing":
            decoder = (
                make_depolarizing_bposd_decoder(code, p)
                if is_bb
                else make_depolarizing_mwpm_decoder(code)
            )
        else:
            raise ValueError(f"Unknown noise: {noise}")
    else:
        raise ValueError(f"Unknown decoder_kind: {decoder_kind}")
    return code, decoder


def _make_adapter(noise, code, decoder, p, use_cache, sampler="bernoulli", cache_cap=None):
    if noise == "bit_flip":
        return BitFlipAdapter(code, decoder, p, use_cache, sampler, cache_cap)
    if noise == "depolarizing":
        return DepolarizingAdapter(code, decoder, p, use_cache)
    raise ValueError(f"Unknown noise: {noise}")


CSV_HEADER = [
    "noise",
    "code_type",
    "d",
    "p",
    "decoder",
    "n_cores",
    "n_target",
    "n_trials",
    "n_faulty",
    "p_L",
    "p_L_se",
    "ew_mean",
    "ew_se",
    "vw",
    "vw_se",
    "wall_clock",
    "feasible",
]

_NOISE_ID = {"bit_flip": 0, "depolarizing": 1}
_CODE_ID = {"toric": 0, "surface": 1, "asymmetric_toric": 2, "bb72": 3, "bb144": 4, "concat_steane": 5}


def _worker(args):
    (
        noise,
        code_type,
        d,
        p,
        target,
        m_max,
        seed_seq,
        use_cache,
        time_budget,
        sampler,
        decoder_kind,
    ) = args
    code, decoder = _make_code_and_decoder(noise, code_type, d, p, decoder_kind)
    adapter = _make_adapter(noise, code, decoder, p, use_cache, sampler)
    rng = np.random.default_rng(seed_seq)
    return run_core(adapter, rng, target, m_max, time_budget)


def run_condition(
    noise,
    code_type,
    d,
    p,
    n_target,
    m_max,
    n_cores=1,
    use_cache=True,
    base_seed=42,
    sampler="bernoulli",
    decoder_kind="auto",
):
    """Run direct MC for one (noise, code, p) condition; return a CSV row dict.

    Splits n_target/m_max across n_cores; each core has its own decoder+cache and
    an independent RNG stream. Pools the sufficient statistics and estimates.
    """
    seeds = np.random.SeedSequence(
        [base_seed, _NOISE_ID[noise], _CODE_ID[code_type], int(d), int(round(p * 1e6))]
    ).spawn(n_cores)
    target_pc = math.ceil(n_target / n_cores)
    mmax_pc = math.ceil(m_max / n_cores)
    args = [
        (noise, code_type, d, p, target_pc, mmax_pc, ss, use_cache, None, sampler,
         decoder_kind)
        for ss in seeds
    ]

    t0 = time.perf_counter()
    if n_cores == 1:
        results = [_worker(args[0])]
    else:
        with Pool(n_cores) as pool:
            results = pool.map(_worker, args)
    wall = time.perf_counter() - t0

    st = pool_stats(results)
    est = estimate(st)
    if decoder_kind == "bp":
        decoder_name = "bp"
    elif decoder_kind == "mwpm":
        decoder_name = "mwpm"
    else:
        decoder_name = "bposd" if code_type in _BB else "mwpm"
    return {
        "noise": noise,
        "code_type": code_type,
        "d": d,
        "p": p,
        "decoder": decoder_name,
        "n_cores": n_cores,
        "n_target": n_target,
        "n_trials": st.n_trials,
        "n_faulty": st.n_faulty,
        "p_L": est["p_L"],
        "p_L_se": est["p_L_se"],
        "ew_mean": est["ew_mean"],
        "ew_se": est["ew_se"],
        "vw": est["vw"],
        "vw_se": est["vw_se"],
        "wall_clock": wall,
        "feasible": st.n_faulty >= max(n_target, 2),
    }


def run_condition_stats(
    noise,
    code_type,
    d,
    p,
    n_target,
    m_max,
    n_cores=1,
    time_budget=None,
    sampler="bernoulli",
    subjob_idx=0,
    use_cache=True,
    base_seed=42,
    decoder_kind="auto",
):
    """Run one direct-MC condition and return *pooled raw sufficient statistics*
    (not a finished estimate), as a JSON-serialisable record.

    Identical sampling machinery to run_condition, but: (a) the seed includes
    subjob_idx so independent PBS subjobs draw disjoint streams, (b) each core
    may stop early on time_budget, and (c) the return value is the pooled Stats
    plus metadata — many such records are pooled and estimated once at merge time.
    """
    seeds = np.random.SeedSequence(
        [
            base_seed,
            _NOISE_ID[noise],
            _CODE_ID[code_type],
            int(d),
            int(round(p * 1e6)),
            int(subjob_idx),
        ]
    ).spawn(n_cores)
    target_pc = math.ceil(n_target / n_cores)
    mmax_pc = math.ceil(m_max / n_cores)
    args = [
        (
            noise,
            code_type,
            d,
            p,
            target_pc,
            mmax_pc,
            ss,
            use_cache,
            time_budget,
            sampler,
            decoder_kind,
        )
        for ss in seeds
    ]

    t0 = time.perf_counter()
    if n_cores == 1:
        results = [_worker(args[0])]
    else:
        with Pool(n_cores) as pool:
            results = pool.map(_worker, args)
    wall = time.perf_counter() - t0

    st = pool_stats(results)
    return {
        "noise": noise,
        "code_type": code_type,
        "d": int(d),
        "p": float(p),
        "sampler": sampler,
        "subjob_idx": int(subjob_idx),
        "decoder_kind": decoder_kind,
        "n_cores": int(n_cores),
        "n_target": int(n_target),
        "n_trials": int(st.n_trials),
        "n_faulty": int(st.n_faulty),
        "s1": float(st.s1),
        "s2": float(st.s2),
        "s3": float(st.s3),
        "s4": float(st.s4),
        "wall_clock": float(wall),
    }


def write_rows(rows, out_csv, append=False):
    """Write/append rows (list of dicts) to a CSV with CSV_HEADER columns."""
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    exists = os.path.exists(out_csv)
    mode = "a" if append and exists else "w"
    with open(out_csv, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def format_console(row):
    """Human-readable one-condition summary string."""
    return (
        f"[{row['noise']}] {row['code_type']} d={row['d']} p={row['p']:g}  "
        f"decoder={row['decoder']}\n"
        f"  p_L   = {row['p_L']:.3e} ± {row['p_L_se']:.1e}     "
        f"(F={row['n_faulty']} / M={row['n_trials']:,})\n"
        f"  E[w]  = {row['ew_mean']:.4g} ± {row['ew_se']:.2g}\n"
        f"  V[w]  = {row['vw']:.4g} ± {row['vw_se']:.2g}     "
        f"wall={row['wall_clock']:.0f}s  feasible={row['feasible']}"
    )
