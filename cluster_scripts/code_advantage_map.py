"""OBF vs Z+H advantage map across codes (paper money figure).

Two phases per (code, p), both written as raw per-chain rows; all classification
(consensus sigma/cross_gap, cell state) is done downstream in
compare/plot_code_advantage_map.py from these raw rows.

  phase=truth      : direct-MC unbiased E[w] (MWPM codes only: rect, surface).
  phase=consensus  : full-length chains from RANDOM inits, OBF & Z+H, M seeds each.
                     downstream: sigma_zh, sigma_obf, cross_gap, classify_truth.
  phase=advantage  : long-init (worst trap) fixed-grid chains (L0,2L0,...) per method;
                     downstream: wall-time-to-precision and speedup.

Flat 96-core Pool over all (phase, code, p, method, seed, length) units. Memory
bounded via run_rimc_mixed(record_trace=False) + cache_cap.
"""
import argparse
import csv
import os
import sys
import time
from collections import namedtuple
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from css_code import (  # noqa: E402
    CSSCode, make_asymmetric_toric_code, make_surface_code, make_toric_code, make_bb_code,
    make_bb_code_72, make_bb_code_144, make_hgp_hamming, make_concatenated_steane_code,
    make_mwpm_decoder, make_bposd_decoder, make_concat_steane_hard_decoder,
)
from runners import run_rimc_mixed  # noqa: E402

METHODS = {
    "OBF": (dict(p_single=1.0, p_zanella=0.0, n_replicas=1), 1.0),
    "Z+H": (dict(p_zanella=0.7, p_cluster=0.3, cluster_k=2, n_replicas=1), 1.0),
}

CodeEntry = namedtuple("CodeEntry", "make_code make_decoder group truth_kind")

CODE_REGISTRY = {
    "rect:3x11": CodeEntry(lambda: make_asymmetric_toric_code(3, 11),
                           lambda c, p: make_mwpm_decoder(c), "trap", "directmc"),
    "rect:3x15": CodeEntry(lambda: make_asymmetric_toric_code(3, 15),
                           lambda c, p: make_mwpm_decoder(c), "trap", "directmc"),
    "bb72": CodeEntry(make_bb_code_72,
                      lambda c, p: make_bposd_decoder(c, p), "trap", "consensus"),
    "bb144": CodeEntry(make_bb_code_144,
                       lambda c, p: make_bposd_decoder(c, p), "trap", "consensus"),
    # BB [[108,8,10]] (l=9, m=6); builder copied verbatim from lowp_audit._bb_l9m6
    "bb_l9m6": CodeEntry(lambda: make_bb_code(9, 6, a_terms=[(3, 0), (0, 1), (0, 2)],
                                              b_terms=[(0, 3), (1, 0), (2, 0)]),
                         lambda c, p: make_bposd_decoder(c, p), "trap", "consensus"),
    "hgp_hamming": CodeEntry(make_hgp_hamming,
                             lambda c, p: make_bposd_decoder(c, p), "ctrl_uniform", "consensus"),
    "surface:3": CodeEntry(lambda: make_surface_code(3),
                           lambda c, p: make_mwpm_decoder(c), "ctrl_k1", "directmc"),
    "toric:5": CodeEntry(lambda: make_toric_code(5),
                         lambda c, p: make_mwpm_decoder(c), "ctrl_uniform", "directmc"),
    "toric:7": CodeEntry(lambda: make_toric_code(7),
                         lambda c, p: make_mwpm_decoder(c), "ctrl_uniform", "directmc"),
    "surface:5": CodeEntry(lambda: make_surface_code(5),
                           lambda c, p: make_mwpm_decoder(c), "ctrl_k1", "directmc"),
    "surface:7": CodeEntry(lambda: make_surface_code(7),
                           lambda c, p: make_mwpm_decoder(c), "ctrl_k1", "directmc"),
    "concat_steane:2": CodeEntry(lambda: make_concatenated_steane_code(2),
                                 lambda c, p: make_concat_steane_hard_decoder(2),
                                 "ctrl_k1", "consensus"),
}

FIELDS = ["phase", "code", "group", "p", "method", "init", "seed", "length",
          "ew", "wall", "truth_kind"]


def build_code_decoder(spec, p):
    e = CODE_REGISTRY[spec]
    code = e.make_code()
    dec = e.make_decoder(code, p)
    return code, dec


def _reorder_logicals(code, first_idx):
    order = [first_idx] + [i for i in range(code.lx.shape[0]) if i != first_idx]
    return CSSCode(hx=code.hx, hz=code.hz,
                   lx=code.lx[order].copy(), lz=code.lz[order].copy(), n=code.n)


def _long_idx(code):
    return int(np.argmax([int(r.sum()) for r in code.lx]))


def _ew(dist):
    return sum(w * pr for w, pr in dist.items())


def _consensus_worker(payload):
    spec, p, method, seed, length, cache_cap = payload
    code, dec = build_code_decoder(spec, p)
    kw, r = METHODS[method]
    t0 = time.perf_counter()
    dist, _ = run_rimc_mixed(p, length, max_error_rate=r * p, code=code, decoder=dec,
                             seed=seed, cache_cap=cache_cap, record_trace=False,
                             randomize_init=True, **kw)
    wall = time.perf_counter() - t0
    g = CODE_REGISTRY[spec]
    return ("consensus", spec, g.group, p, method, "random", seed, length, _ew(dist), wall, g.truth_kind)


def _advantage_worker(payload):
    spec, p, method, seed, length, cache_cap = payload
    code, _ = build_code_decoder(spec, p)
    code = _reorder_logicals(code, _long_idx(code))
    dec = CODE_REGISTRY[spec].make_decoder(code, p)
    kw, r = METHODS[method]
    t0 = time.perf_counter()
    dist, _ = run_rimc_mixed(p, length, max_error_rate=r * p, code=code, decoder=dec,
                             seed=seed, cache_cap=cache_cap, record_trace=False,
                             randomize_init=False, **kw)
    wall = time.perf_counter() - t0
    g = CODE_REGISTRY[spec]
    return ("advantage", spec, g.group, p, method, "long", seed, length, _ew(dist), wall, g.truth_kind)


def _truth_worker(payload):
    spec, p, m_max, batch, seed = payload
    import pymatching
    code = CODE_REGISTRY[spec].make_code()
    m = pymatching.Matching(code.hz)
    lz, n = code.lz, code.n
    rng = np.random.default_rng(seed)
    nf, s1, done = 0, 0.0, 0
    while done < m_max:
        b = min(batch, m_max - done)
        e = (rng.random((b, n)) < p).astype(np.uint8)
        corr = m.decode_batch((e @ code.hz.T) % 2).astype(np.uint8)
        faulty = (((e ^ corr) % 2) @ lz.T % 2).any(axis=1)
        w = e.sum(axis=1)[faulty]
        nf += int(faulty.sum())
        s1 += float(w.sum())
        done += b
    ew = s1 / nf if nf else float("nan")
    g = CODE_REGISTRY[spec]
    return ("truth", spec, g.group, p, "directmc", "-", seed, 0, ew, float("nan"), g.truth_kind)


_DISPATCH = {"truth": _truth_worker, "consensus": _consensus_worker, "advantage": _advantage_worker}


def _run_task(t):
    kind, payload = t
    return _DISPATCH[kind](payload)


# --- checkpoint / resume: identity key per task, matched against rows already on disk ---
_KEY_COLS = ("phase", "code", "p", "method", "seed", "length")


def _task_key(task):
    """Stable identity (phase, code, p, method, seed, length) as strings, matching the CSV."""
    kind, payload = task
    if kind == "truth":
        spec, p, _m_max, _batch, seed = payload
        return ("truth", spec, str(p), "directmc", str(seed), "0")
    spec, p, method, seed, length, _cap = payload   # consensus / advantage share this shape
    return (kind, spec, str(p), method, str(seed), str(length))


def _row_key(row):
    return tuple(row[c] for c in _KEY_COLS)


def _load_done(out_csv):
    """Return (done_keys, valid_rows) from a prior (possibly killed) run.

    A run killed mid-flush can leave a truncated final line; rows missing any key
    field are dropped so resume recomputes them (no duplicate, no corruption).
    """
    if not os.path.exists(out_csv):
        return set(), []
    rows = []
    with open(out_csv, newline="") as f:
        for row in csv.DictReader(f):
            # a complete row has every field non-empty (ew/wall are "nan" strings, never "");
            # a truncated final line from a kill is missing trailing fields -> drop & recompute.
            if any(row.get(c) in (None, "") for c in FIELDS):
                continue
            rows.append(row)
    return {_row_key(r) for r in rows}, rows


def _fixed_grid_lengths(L0, L_max):
    out, L = [], L0
    while L <= L_max:
        out.append(L)
        L *= 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default=",".join(CODE_REGISTRY))
    ap.add_argument("--p-list", default="1e-2,1e-3")
    ap.add_argument("--m-consensus", type=int, default=4, help="random inits per kernel (consensus)")
    ap.add_argument("--consensus-len-obf", type=int, default=2_000_000)
    ap.add_argument("--consensus-len-zh", type=int, default=200_000)
    ap.add_argument("--n-seeds", type=int, default=24, help="seeds per advantage (code,p,method,length)")
    ap.add_argument("--L0", type=int, default=1000)
    ap.add_argument("--L-max", type=int, default=33_554_432,
                    help="OBF fixed-grid cap for TRAP codes (trace the plateau / unreachable)")
    ap.add_argument("--ctrl-obf-L-max", type=int, default=512_000,
                    help="OBF fixed-grid cap for CONTROL codes (converge early; no need for huge L)")
    ap.add_argument("--kernel-L-max", type=int, default=512_000)    # Z+H cap (fixed-grid)
    ap.add_argument("--cache-cap", type=int, default=200_000)
    ap.add_argument("--truth-mmax", type=int, default=2_000_000_000)
    ap.add_argument("--truth-chunks", type=int, default=24,
                    help="split direct-MC truth across this many cores (tail saturation)")
    ap.add_argument("--truth-batch", type=int, default=200_000)
    ap.add_argument("--n-cores", type=int, default=96)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    specs = args.specs.split(",")
    ps = [float(x) for x in args.p_list.split(",")]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    tasks = []
    # truth (direct-MC) tasks — MWPM codes only, split into chunks for tail saturation
    mper = (args.truth_mmax + args.truth_chunks - 1) // args.truth_chunks
    for spec in specs:
        if CODE_REGISTRY[spec].truth_kind == "directmc":
            for p in ps:
                for k in range(args.truth_chunks):
                    tasks.append(("truth", (spec, p, mper, args.truth_batch, 12345 + k)))
    # consensus tasks — all codes, both kernels, M random-init seeds
    for spec in specs:
        for p in ps:
            for method in METHODS:
                clen = args.consensus_len_obf if method == "OBF" else args.consensus_len_zh
                for s in range(args.m_consensus):
                    tasks.append(("consensus", (spec, p, method, 1000 + s, clen, args.cache_cap)))
    # advantage tasks — long init fixed-grid, both kernels, n_seeds.
    # OBF cap is group-aware: TRAP codes need the huge L (plateau/unreachable); CONTROL
    # codes converge early, so capping them small avoids wasted compute + tail starvation.
    for spec in specs:
        is_trap = CODE_REGISTRY[spec].group == "trap"
        for p in ps:
            for method in METHODS:
                if method == "OBF":
                    cap = args.L_max if is_trap else args.ctrl_obf_L_max
                else:
                    cap = args.kernel_L_max
                for L in _fixed_grid_lengths(args.L0, cap):
                    for s in range(args.n_seeds):
                        tasks.append(("advantage", (spec, p, method, 7000 + s, L, args.cache_cap)))

    # Resume: skip tasks already on disk from a prior (possibly killed) run.
    done, existing = _load_done(args.out_csv)
    pending = [t for t in tasks if _task_key(t) not in done]
    print(f"{len(existing)} done, {len(pending)} pending (of {len(tasks)})", flush=True)

    pool = Pool(args.n_cores) if args.n_cores > 1 else None
    results_iter = (pool.imap_unordered(_run_task, pending) if pool
                    else (_run_task(t) for t in pending))

    # Incremental write: each completed chain is flushed immediately, so a walltime
    # kill loses at most the in-flight chains, not the whole run. The "w" rewrite
    # also sanitizes any truncated final line from the prior run.
    n_new = 0
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in existing:
            w.writerow({k: r[k] for k in FIELDS})
        f.flush()
        for res in results_iter:
            w.writerow(dict(zip(FIELDS, res)))
            f.flush()
            n_new += 1
    if pool:
        pool.close(); pool.join()
    print(f"wrote {n_new} new rows ({len(existing) + n_new} total) -> {args.out_csv}")


if __name__ == "__main__":
    main()
