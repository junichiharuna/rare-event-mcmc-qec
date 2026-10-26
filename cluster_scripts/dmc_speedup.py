"""Direct-MC vs OBF speedup run: dmc_cutoff + advantage phases over the codes named by
--specs (the published table uses 7, plus bb_l9m6 as a later per-code top-up) x
{1e-1,1e-3,1e-5}, one unified CSV. Reuses code_advantage_map (advantage worker,
code registry) and direct_mc (cutoff via run_core). Bit-flip channel.
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from direct_mc import run_core, _make_adapter  # noqa: E402
from code_advantage_map import (  # noqa: E402
    CODE_REGISTRY, build_code_decoder, _advantage_worker, _fixed_grid_lengths,
)

FIELDS = ["phase", "code", "group", "p", "method", "init", "seed", "length",
          "ew", "wall", "truth_kind", "n_trials", "n_faulty", "s1", "s2", "n_cores"]

_EMPTY = {"n_trials": "", "n_faulty": "", "s1": "", "s2": "", "n_cores": ""}

# Hardest first, with the topological controls ordered by decreasing code size.
DIFFICULTY = ["bb144", "bb_l9m6", "rect:3x15", "rect:3x11", "bb72", "hgp_hamming",
              "toric:7", "surface:7", "toric:5", "surface:5", "concat_steane:2", "surface:3"]


def dmc_cutoff_worker(payload):
    """One single-core direct-MC cutoff chunk -> a dmc_cutoff row dict."""
    spec, p, seed, time_budget, cache_cap = payload
    code, dec = build_code_decoder(spec, p)
    # Bounded cache (Part A): keep the syndrome->correction cache (a big win at low/moderate p
    # where low-weight syndromes repeat, especially for expensive BP-OSD) while capping memory
    # at high p via clear-on-full eviction, mirroring runners.py's cache_cap.
    adapter = _make_adapter("bit_flip", code, dec, p, use_cache=True, cache_cap=cache_cap)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    st = run_core(adapter, rng, target=10 ** 18, m_max=10 ** 18, time_budget=time_budget)
    wall = time.perf_counter() - t0
    g = CODE_REGISTRY[spec]
    return {"phase": "dmc_cutoff", "code": spec, "group": g.group, "p": repr(p),
            "method": "directmc", "init": "-", "seed": seed, "length": 0,
            "ew": "", "wall": wall, "truth_kind": "directmc",
            "n_trials": st.n_trials, "n_faulty": st.n_faulty, "s1": st.s1,
            "s2": st.s2, "n_cores": 1}


def advantage_row(payload):
    """Wrap code_advantage_map._advantage_worker's 11-tuple into the extended dict."""
    tup = _advantage_worker(payload)  # (phase, code, group, p, method, init, seed, length, ew, wall, truth_kind)
    keys = ["phase", "code", "group", "p", "method", "init", "seed", "length", "ew", "wall", "truth_kind"]
    row = dict(zip(keys, tup))
    row.update(_EMPTY)
    return row


def _run(task):
    kind, payload = task
    return dmc_cutoff_worker(payload) if kind == "dmc" else advantage_row(payload)


# --- resume: skip (phase,code,p,method,seed,length) already on disk (mirrors code_advantage_map) ---
_KEY = ("phase", "code", "p", "method", "seed", "length")


def _task_key(task):
    kind, payload = task
    if kind == "dmc":
        spec, p, seed, _b, _c = payload
        return ("dmc_cutoff", spec, repr(p), "directmc", str(seed), "0")
    spec, p, method, seed, L, _c = payload
    return ("advantage", spec, repr(p), method, str(seed), str(L))


def _load_done(out_csv):
    if not os.path.exists(out_csv):
        return set(), []
    rows = []
    with open(out_csv, newline="") as f:
        for r in csv.DictReader(f):
            if any(r.get(c) in (None, "") for c in _KEY):  # drop truncated final line from a kill
                continue
            rows.append(r)
    return {tuple(r[c] for c in _KEY) for r in rows}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default=",".join(CODE_REGISTRY))
    ap.add_argument("--p-list", default="1e-1,1e-3,1e-5")
    ap.add_argument("--dmc-chunks", type=int, default=96, help="single-core cutoff chunks per cell")
    ap.add_argument("--dmc-budget", type=float, default=300.0, help="wall seconds per cutoff chunk")
    ap.add_argument("--n-seeds", type=int, default=24)
    ap.add_argument("--L0", type=int, default=1000)
    ap.add_argument("--L-max", type=int, default=33_554_432, help="OBF fixed-grid cap for TRAP codes")
    ap.add_argument("--ctrl-obf-L-max", type=int, default=512_000, help="OBF cap for CONTROL codes")
    ap.add_argument("--kernel-L-max", type=int, default=512_000, help="Z+H fixed-grid cap")
    ap.add_argument("--cache-cap", type=int, default=200_000)
    ap.add_argument("--n-cores", type=int, default=96)
    ap.add_argument("--run-id", type=int, default=1,
                    help="independent repeat index; offsets every RNG seed by 1e6*(run_id-1), "
                         "so run 1 reproduces the original seeds and runs 2,3,... are disjoint")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()
    seed_off = 1_000_000 * (args.run_id - 1)

    specs = args.specs.split(",")
    ps = [float(x) for x in args.p_list.split(",")]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    # Hardest-first ordering (single node): process the deepest-trap, lowest-p cells first, so
    # if the walltime is hit the scientifically hardest cells (esp. bb144 at low p, the
    # formerly-blank Fig-3 row) are already done and only the easy control tail remains (which a
    # resume run finishes). imap_unordered dispatches in list order, so list order = priority.
    # Difficulty: trap codes (bb144 hardest) before controls, then p ascending.
    specs = sorted(specs, key=lambda s: DIFFICULTY.index(s) if s in DIFFICULTY else 99)
    ps = sorted(ps)  # ascending -> lowest p (hardest) first

    # Group-aware caps mirror code_advantage_map: TRAP-OBF needs huge L (plateau / unreachable
    # lower bound); CONTROL-OBF and Z+H converge early, so cap them small to avoid walltime blow-up.
    tasks = []
    for spec in specs:
        is_trap = CODE_REGISTRY[spec].group == "trap"
        for p in ps:
            for c in range(args.dmc_chunks):
                tasks.append(("dmc", (spec, p, 90000 + c + seed_off, args.dmc_budget, args.cache_cap)))
            for method in ("OBF", "Z+H"):
                cap = (args.L_max if is_trap else args.ctrl_obf_L_max) if method == "OBF" \
                    else args.kernel_L_max
                for s in range(args.n_seeds):
                    for L in _fixed_grid_lengths(args.L0, cap):
                        tasks.append(("adv", (spec, p, method, 7000 + s + seed_off, L, args.cache_cap)))

    done, existing = _load_done(args.out_csv)   # resume: skip completed tasks
    pending = [t for t in tasks if _task_key(t) not in done]
    print(f"{len(existing)} done, {len(pending)} pending (of {len(tasks)})", flush=True)

    with Pool(args.n_cores) as pool, open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in existing:               # re-write completed rows (also sanitizes a truncated line)
            w.writerow({k: r.get(k, "") for k in FIELDS})
        f.flush()
        for row in pool.imap_unordered(_run, pending, chunksize=1):
            w.writerow(row)
            f.flush()
    print(f"wrote {args.out_csv} ({len(existing)} kept + {len(pending)} new)")


if __name__ == "__main__":
    main()
