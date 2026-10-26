"""Direct MC — one PBS subjob writing *raw pooled Stats* (one JSONL line).

Many identical subjobs (distinct --subjob-idx) run the same condition with
disjoint seeds; merge_direct_mc_stats.py pools all their Stats into one row.

Usage (PBS):
    python cluster_scripts/sweep_direct_mc_stats.py \\
        --noise bit_flip --code-type bb144 --code-size 0 --p 1e-3 \\
        --n-target 200 --m-max 1e15 --n-cores 20 --time-budget 84600 \\
        --sampler binomial --subjob-idx ${PBS_ARRAY_INDEX} \\
        --out-jsonl /path/to/dmc_stats_p1e-3_${PBS_ARRAY_INDEX}.jsonl
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from direct_mc import run_condition_stats, write_stats_jsonl   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Direct MC — one stats subjob")
    ap.add_argument("--noise", required=True, choices=["bit_flip", "depolarizing"])
    ap.add_argument("--code-type", required=True)
    ap.add_argument("--code-size", type=int, default=0)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--n-target", type=int, default=200)
    ap.add_argument("--m-max", type=float, default=1e15)
    ap.add_argument("--n-cores", type=int, default=20)
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--sampler", default="binomial",
                    choices=["bernoulli", "binomial"])
    ap.add_argument("--subjob-idx", type=int, default=0)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--decoder-kind", default="auto",
                    choices=["auto", "mwpm", "bp"])
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    rec = run_condition_stats(
        args.noise, args.code_type, args.code_size, args.p,
        n_target=args.n_target, m_max=int(args.m_max), n_cores=args.n_cores,
        time_budget=args.time_budget, sampler=args.sampler,
        subjob_idx=args.subjob_idx, base_seed=args.base_seed,
        use_cache=not args.no_cache, decoder_kind=args.decoder_kind,
    )
    write_stats_jsonl([rec], args.out_jsonl)
    print(f"subjob {args.subjob_idx}: M={rec['n_trials']:,} F={rec['n_faulty']} "
          f"wall={rec['wall_clock']:.0f}s -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
