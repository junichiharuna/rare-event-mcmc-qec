# cluster_scripts/kernel_param_tta.py
"""Cluster-fraction and exchange-size sweeps measured EXACTLY as the second benchmark
(sec:bench2): long init, no burn-in, independent chains at L = L0*2^k, sustained
T_2% against the weight-stratified reference (computed offline by
compare/analyze_kernel_param_tta.py), repeated over --run-id with disjoint seed streams.

Arms: p_cluster in {0,0.1,0.2,0.3,0.5,0.7,0.9} at m=2, and m in {3,4,6} at p_cluster=0.3
(the (0.3, 2) arm is the production kernel and is shared by both sweeps). With
run_id r and seed 7000+s+1e6(r-1) the (0.3, 2) arm reproduces the Z+H chains of
cluster_results/dmc_speedup_run{r}.csv bit for bit up to its length cap, which the
analyzer uses as a cross-check.

One flat pool over (spec, p, arm, seed, L); longest-expected-first; memory bounded by
record_trace=False + cache_cap; incremental append + resume on (code, p, p_cluster,
cluster_k, seed, length). Wall AND cpu time per chain are recorded (cpu/wall ~ 1 is the
no-contention certificate of docs/cluster_job_design_guide.md).
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from runners import run_rimc_mixed  # noqa: E402
from code_advantage_map import (  # noqa: E402
    CODE_REGISTRY, build_code_decoder, _reorder_logicals, _long_idx, _ew, _fixed_grid_lengths,
)

ARMS = [(pc, 2) for pc in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)] + [(0.3, k) for k in (3, 4, 6)]
FIELDS = ["code", "group", "p", "p_cluster", "cluster_k", "init", "seed", "length",
          "ew", "wall", "cpu"]
SEED_BASE = 7000                     # = dmc_speedup's advantage seeds
RUN_STRIDE = 1_000_000


def kernel_kwargs(pc, k):
    return dict(p_single=0.0, p_zanella=round(1.0 - pc, 10), p_cluster=float(pc),
                cluster_k=int(k), n_replicas=1)


def _worker(payload):
    spec, p, pc, k, seed, L, cache_cap = payload
    code, _ = build_code_decoder(spec, p)
    code = _reorder_logicals(code, _long_idx(code))
    dec = CODE_REGISTRY[spec].make_decoder(code, p)
    t0, c0 = time.perf_counter(), time.process_time()
    dist, _ = run_rimc_mixed(p, L, max_error_rate=p, code=code, decoder=dec, seed=seed,
                             cache_cap=cache_cap, record_trace=False, randomize_init=False,
                             **kernel_kwargs(pc, k))
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    return dict(code=spec, group=CODE_REGISTRY[spec].group, p=repr(p), p_cluster=repr(pc),
                cluster_k=k, init="long", seed=seed, length=L, ew=_ew(dist), wall=wall, cpu=cpu)


def _task_key(t):
    spec, p, pc, k, seed, L, _c = t
    return (spec, repr(p), repr(pc), str(k), str(seed), str(L))


def _row_key(r):
    return (r["code"], repr(float(r["p"])), repr(float(r["p_cluster"])), str(int(r["cluster_k"])),
            str(int(r["seed"])), str(int(r["length"])))


def _load_done(out_csv):
    if not os.path.exists(out_csv):
        return set(), []
    rows = []
    with open(out_csv, newline="") as f:
        for r in csv.DictReader(f):
            if any(r.get(c) in (None, "") for c in FIELDS):
                continue
            rows.append(r)
    return {_row_key(r) for r in rows}, rows


def build_tasks(specs, ps, arms, n_seeds, L0, L_max, cache_cap, run_id):
    off = RUN_STRIDE * (run_id - 1)
    tasks = []
    for spec in specs:
        for p in sorted(ps):
            for pc, k in arms:
                for s in range(n_seeds):
                    for L in _fixed_grid_lengths(L0, L_max):
                        tasks.append((spec, float(p), float(pc), int(k), SEED_BASE + s + off,
                                      int(L), cache_cap))
    return tasks


BPOSD_PREFIXES = ("bb", "hgp_hamming")   # decoders ~20x slower per step than MWPM


def _cost(t, n_of):
    """Expected chain cost: steps x (BP-OSD penalty) x code length."""
    spec, _p, _pc, _k, _s, L, _c = t
    return L * (20 if spec.startswith(BPOSD_PREFIXES) else 1) * n_of[spec]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="rect:3x11,bb144,bb_l9m6")
    ap.add_argument("--p-list", default="1e-3,1e-4,1e-5")
    ap.add_argument("--arms", default="all", help="'all' or '0.3,2;0.5,2' style list (p_cluster,m pairs)")
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--L0", type=int, default=1000)
    ap.add_argument("--L-max", type=int, default=2_048_000)
    ap.add_argument("--cache-cap", type=int, default=200_000)
    ap.add_argument("--n-cores", type=int, default=96)
    ap.add_argument("--run-id", type=int, default=1)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()
    specs = args.specs.split(",")
    ps = [float(x) for x in args.p_list.split(",")]
    arms = ARMS if args.arms == "all" else [
        (float(a.split(",")[0]), int(a.split(",")[1])) for a in args.arms.split(";")]
    n_of = {spec: CODE_REGISTRY[spec].make_code().n for spec in specs}
    tasks = build_tasks(specs, ps, arms, args.n_seeds, args.L0, args.L_max, args.cache_cap,
                        args.run_id)
    done, existing = _load_done(args.out_csv)
    pending = sorted([t for t in tasks if _task_key(t) not in done],
                     key=lambda t: -_cost(t, n_of))
    print(f"{len(existing)} done, {len(pending)} pending (of {len(tasks)})", flush=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    tmp = args.out_csv + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in existing:
            w.writerow({c: r.get(c, "") for c in FIELDS})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, args.out_csv)          # a kill mid-rewrite never loses finished rows
    with open(args.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        t_start = time.perf_counter()
        with Pool(args.n_cores) as pool:
            for i, row in enumerate(pool.imap_unordered(_worker, pending, chunksize=1), 1):
                w.writerow(row)
                f.flush()
                if i % 200 == 0 or i == len(pending):
                    print(f"  [{time.perf_counter() - t_start:7.0f}s] {i}/{len(pending)}", flush=True)
    print(f"done -> {args.out_csv}")


if __name__ == "__main__":
    main()
