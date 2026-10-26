"""Local CLI driver for direct (rejection) Monte Carlo E[w]/V[w]/p_L.

Runs one condition (inner multiprocessing splits the faulty-sample target),
appends a CSV row, and prints a summary.

Usage:
    python compare/direct_mc_ew.py --noise bit_flip --code-type bb144 --p 1e-2 \\
        --n-target 2000 --m-max 5e8 --n-cores 16 --out cluster_results/direct_mc_ew.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from direct_mc import run_condition, write_rows, format_console  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Direct MC E[w]/V[w]/p_L — one condition")
    ap.add_argument("--noise", required=True, choices=["bit_flip", "depolarizing"])
    ap.add_argument(
        "--code-type",
        required=True,
        choices=["toric", "surface", "asymmetric_toric", "bb72", "bb144"],
    )
    ap.add_argument("--code-size", type=int, default=0)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--n-target", type=int, default=2000)
    ap.add_argument("--m-max", type=float, default=5e8)
    ap.add_argument("--n-cores", type=int, default=1)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--sampler", default="bernoulli", choices=["bernoulli", "binomial"])
    ap.add_argument("--out", default="cluster_results/direct_mc_ew.csv")
    args = ap.parse_args()

    row = run_condition(
        args.noise,
        args.code_type,
        args.code_size,
        args.p,
        n_target=args.n_target,
        m_max=int(args.m_max),
        n_cores=args.n_cores,
        use_cache=not args.no_cache,
        base_seed=args.base_seed,
        sampler=args.sampler,
    )
    write_rows([row], args.out, append=True)
    print(format_console(row))
    print(f"Appended 1 row to {args.out}")


if __name__ == "__main__":
    main()
