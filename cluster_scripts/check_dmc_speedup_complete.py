"""Completeness check for a dmc_speedup output CSV, run at the end of a job.

Prints one word on stdout -- COMPLETE or INCOMPLETE -- so a job script can put it in the
commit message and the ntfy notification; the details go to stderr.

Why it exists: compare/analyze_dmc_speedup.py::n_seeds_of() raises
"seed count differs across cells" if ANY cell of ANY merged run carries a seed count
different from the rest, so a file truncated by a walltime kill would block the whole
table until its RUN_ID is resubmitted.  Better to say so at job time.

Usage:
    python cluster_scripts/check_dmc_speedup_complete.py OUT.csv \
        [--n-seeds 50] [--expect OBF:16,Z+H:10] [--p-list 1e-1,1e-3,1e-5]

--expect gives the number of fixed-grid checkpoints each kernel must have, i.e.
len(_fixed_grid_lengths(L0, cap)): 16 for a cap of 33,554,432 and 10 for 512,000 at
L0=1000.  Every (code, p, method) cell is required to have exactly that many distinct
lengths and exactly --n-seeds distinct seeds at each of them.
"""
import argparse
import collections
import csv
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--expect", default="OBF:16,Z+H:10",
                    help="checkpoints per kernel, e.g. 'OBF:16,Z+H:10'")
    ap.add_argument("--p-list", default="1e-1,1e-3,1e-5")
    args = ap.parse_args()

    expect = {k: int(v) for k, v in (kv.split(":") for kv in args.expect.split(","))}
    ps = [float(x) for x in args.p_list.split(",")]

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    seeds = collections.defaultdict(set)     # (code, p, method, length) -> seeds
    faulty = collections.Counter()           # p -> direct-MC faulty draws
    codes = set()
    for r in rows:
        if r["phase"] == "advantage":
            codes.add(r["code"])
            seeds[(r["code"], float(r["p"]), r["method"], int(r["length"]))].add(r["seed"])
        elif r["phase"] == "dmc_cutoff":
            faulty[float(r["p"])] += int(r["n_faulty"])

    problems = []
    lengths = collections.defaultdict(set)   # (code, p, method) -> lengths
    for (code, p, method, L), ss in seeds.items():
        lengths[(code, p, method)].add(L)
        if len(ss) != args.n_seeds:
            problems.append(f"{code} p={p:g} {method} L={L}: {len(ss)} seeds != {args.n_seeds}")
    for code in sorted(codes):
        for p in ps:
            for method, n in expect.items():
                got = len(lengths.get((code, p, method), ()))
                if got != n:
                    problems.append(f"{code} p={p:g} {method}: {got} checkpoints != {n}")

    print(f"codes={sorted(codes)} rows={len(rows)}", file=sys.stderr)
    print("dmc_cutoff faulty draws per p: "
          + ", ".join(f"{p:g}:{faulty.get(p, 0)}" for p in ps), file=sys.stderr)
    if problems:
        for line in problems[:20]:
            print("  MISSING " + line, file=sys.stderr)
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more", file=sys.stderr)
        print("INCOMPLETE")
        return
    print("COMPLETE")


if __name__ == "__main__":
    main()
