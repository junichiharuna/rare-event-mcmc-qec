"""Weight-stratified ("subset sampling") reference values P_L(p) and E[w|F](p)
for the paper's general-code families (single full-node job).

Why this instead of more direct MC: the low-p reference values the paper quotes
(1e-3, 1e-4) cost 1e12+ direct-MC trials on bb144 for a few hundred faulty samples.
The exact decomposition P_L = sum_w B(w;n,p) f_w moves the whole cost into the
p-INDEPENDENT quantity f_w = P(fail | |e|=w), which is cheap: the low strata are
enumerated exactly and the rest are sampled uniformly at fixed weight, where
failures are common.  See src/subset_sampling.py for the estimator and
docs/superpowers/specs/2026-08-28-subset-sampling-design.md for the design.

Three decoder regimes (ss.is_p_independent decides one-run-per-code vs one-run-per-p;
ss.is_mwpm decides only whether the pymatching decode_batch fast path applies):
  * MWPM codes (toric / surface / asymmetric_toric) -- fast path, and f_w does not
    depend on p, so ONE run per code synthesizes the entire --p-list.
  * BP-OSD codes (bb144 / bb72 / bb_l9m6 / hgp_hamming3; channel prior = p) --
    f_w does depend on p, so one run PER p.  Their decoder is built by the same
    make_bposd_decoder(code, p) the MCMC audit (cluster_scripts/zh_rerun.py) uses,
    so these rows are a like-for-like reference for tab:audit's truth column.
  * hard-decoder codes (concat_steane) -- generic per-syndrome path like BP-OSD,
    but the hierarchical hard-decision decoder carries no prior, so f_w is again
    p-independent and ONE run synthesizes the entire --p-list (w_max = WMAX_MWPM).

Outputs (both read-modify-write, so a rerun replaces the rows for the same
condition instead of duplicating them, and a walltime kill keeps what finished):
  <prefix>_fw.csv   the f_w table   (code_type,d,p_decoder,n,w,mode,n_samples,n_fail,f_w,f_w_se)
                    (--noise depolarizing writes under its own prefix,
                    cluster_results/subset_sampling_depol*; there w is the SUPPORT
                    weight and n_samples counts (support, Pauli-type) patterns)
  <prefix>.csv      synthesized rows in the direct_mc.CSV_HEADER schema

Caveat: a condition is resumed on the f_w table alone, so if --p-list CHANGES
between runs the already-computed codes are skipped and their rows for the OLD p
grid survive in <prefix>.csv.  Pass --force (or delete the CSVs) after editing
--p-list.  Note also that n_trials / n_faulty in <prefix>.csv are STRATIFIED
sample and failure counts, not direct-MC trials: n_faulty/n_trials is NOT p_L
(P_L is the p_L column).

The `d` column carries the code-size SPEC verbatim -- "5"/"7" for toric/surface,
"5x5"/"3x11"/"3x15" for the rectangular toric family (which direct_mc's own d
column cannot express), "2" for concat_steane (a concatenation LEVEL count, not a
distance: L_2 = [[49, 1, 9]]), and "0" for the codes whose size is neither
(bb144, bb72, bb_l9m6, hgp_hamming3), matching cluster_results/*.csv.

Flag names deliberately avoid the --d* prefix, which `conda run` swallows.
"""
import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import subset_sampling as ss  # noqa: E402
from direct_mc import CSV_HEADER  # noqa: E402

BITFLIP_CODES = ("toric:5,toric:7,surface:5,surface:7,"
                 "asymmetric_toric:5x5,asymmetric_toric:3x11,"
                 "asymmetric_toric:3x15,bb144:0")
DEPOL_CODES = ("toric:5,toric:7,surface:5,surface:7,asymmetric_toric:3x11,"
               "asymmetric_toric:3x15,hgp_hamming3:0,bb72:0,bb144:0,bb_l9m6:0")

# Default truncation weight per decoder regime.  At p = 1e-2 and the largest n
# here (n = 144) the neglected mass sum_{w>w_max} B(w;n,p) is ~1e-9 at w_max=12
# and ~1e-13 at w_max=16 -- orders below the P_L being reported, and the bound is
# printed for every row so the choice is auditable rather than assumed.
WMAX_MWPM = 16
WMAX_BPOSD = 12


def parse_codes(spec: str):
    """"toric:5,bb144:0" -> [("toric", "5"), ("bb144", "0")]."""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"bad --codes entry {item!r} (want code_type:dspec)")
        code_type, d = item.split(":", 1)
        out.append((code_type.strip(), ss.parse_d(d)))
    return out


# ── CSV read-modify-write ──────────────────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(rows, path, header):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _fw_key(row):
    return (str(row["code_type"]), str(row["d"]), str(row["p_decoder"]))


def _pl_key(row):
    return (str(row["code_type"]), str(row["d"]), f"{float(row['p']):.12g}")


def merge_rows(old, new, keyfn):
    """Replace every old row whose key matches a new row's, then append the new."""
    keys = {keyfn(r) for r in new}
    return [r for r in old if keyfn(r) not in keys] + list(new)


# ── main ───────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise", choices=("bit_flip", "depolarizing"),
                    default="bit_flip",
                    help="error model; sets the default --codes, --p-list and "
                         "--out-prefix")
    ap.add_argument("--codes", default=None,
                    help="default: the bit-flip set, or the ten tab:meanw codes "
                         "under --noise depolarizing")
    ap.add_argument("--p-list", default=None,
                    help="default: 1e-2,3e-3,1e-3,1e-4 (bit_flip) or "
                         "1e-2,1e-3,1e-4 (depolarizing)")
    ap.add_argument("--wmax", type=int, default=0,
                    help="highest weight stratum kept (0 = per-code default: "
                         f"{WMAX_MWPM} p-independent / {WMAX_BPOSD} BP-OSD)")
    ap.add_argument("--enum-cap", type=int, default=ss.ENUM_CAP,
                    help="enumerate a stratum exactly when C(n,w) <= this")
    ap.add_argument("--n-max", type=int, default=ss.N_MAX,
                    help="max MC samples per sampled stratum")
    ap.add_argument("--target-fail", type=int, default=ss.TARGET_FAIL,
                    help="target failures per sampled stratum")
    ap.add_argument("--pilot-n", type=int, default=ss.PILOT_N,
                    help="pilot samples used to size each sampled stratum")
    ap.add_argument("--n-cores", type=int, default=96)
    ap.add_argument("--cache-cap", type=int, default=ss.CACHE_CAP,
                    help="per-process decode-cache entries (0 = no cache); the "
                         "cache only pays off on the BP-OSD path")
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the per-condition cross-check of the vectorized/"
                         "cached failure mask against the repo per-shot path")
    ap.add_argument("--force", action="store_true",
                    help="recompute conditions already present in the f_w CSV")
    a = ap.parse_args(argv)
    if a.noise == "depolarizing":
        a.out_prefix = a.out_prefix or "cluster_results/subset_sampling_depol"
        a.codes = a.codes or DEPOL_CODES
        a.p_list = a.p_list or "1e-2,1e-3,1e-4"
    else:
        a.out_prefix = a.out_prefix or "cluster_results/subset_sampling"
        a.codes = a.codes or BITFLIP_CODES
        a.p_list = a.p_list or "1e-2,3e-3,1e-3,1e-4"
    a._parser = ap
    return a


def main():
    args = parse_args()
    ap = args._parser

    if args.wmax < 0:
        ap.error("--wmax must be >= 1 (or 0 for the per-code default)")
    codes = parse_codes(args.codes)
    ps = [float(x) for x in args.p_list.split(",")]
    if not ps:
        ap.error("--p-list is empty")
    fw_path = f"{args.out_prefix}_fw.csv"
    out_path = f"{args.out_prefix}.csv"

    fw_all = load_csv(fw_path)
    pl_all = load_csv(out_path)
    done = {_fw_key(r) for r in fw_all}

    # Longest-first is not knowable a priori here, but the conditions are ordered
    # cheapest-first (small n, MWPM) so a walltime kill still publishes rows.
    tasks = []
    for code_type, d in codes:
        if ss.is_p_independent(code_type):
            tasks.append((code_type, d, None))          # one run, all p synthesized
        else:
            tasks += [(code_type, d, p) for p in ps]    # one run per decoder prior

    print(f"subset sampling: {len(tasks)} condition(s), p-grid "
          f"{','.join(f'{p:g}' for p in ps)}, cores {args.n_cores}, "
          f"enum-cap {args.enum_cap:,}, n-max {args.n_max:,}, "
          f"target-fail {args.target_fail}, cache-cap {args.cache_cap:,}", flush=True)

    summary = []
    t_all = time.perf_counter()
    for code_type, d, p_dec in tasks:
        p_indep = ss.is_p_independent(code_type)
        w_max = args.wmax or (WMAX_MWPM if p_indep else WMAX_BPOSD)
        p_run = p_dec if p_dec is not None else max(ps)
        key = (code_type, d, "" if p_indep else f"{p_run:g}")
        if key in done and not args.force:
            print(f"skip (already in {fw_path}): {code_type} d={d} "
                  f"p_dec={'-' if p_indep else f'{p_run:g}'}", flush=True)
            continue

        res = ss.run_code(
            args.noise, code_type, d, p_run, w_max=w_max, enum_cap=args.enum_cap,
            n_max=args.n_max, target_fail=args.target_fail, pilot_n=args.pilot_n,
            workers=args.n_cores, seed=args.seed, cache_cap=args.cache_cap,
            verify=not args.no_verify)

        n = res["n"]
        meta = dict(noise=args.noise, code_type=code_type, d=d, decoder=res["decoder"],
                    n_cores=args.n_cores, n_target=args.target_fail,
                    wall_clock=res["wall"], w_max=res["w_max"])
        pl_rows = ss.rows_for_p_grid(res["fw_rows"], n, ps if p_indep else [p_run], meta)

        fw_all = merge_rows(fw_all, res["fw_rows"], _fw_key)
        pl_all = merge_rows(pl_all, pl_rows, _pl_key)
        write_csv(fw_all, fw_path, ss.FW_HEADER)
        write_csv(pl_all, out_path, CSV_HEADER)
        done.add(key)

        # Per-decode cost, measured on the ENUMERATED strata only: there decodes
        # == samples exactly (distinct supports, no cache hits worth counting),
        # so this is the number that scales the production walltime.
        ne = res["n_enum_samples"]
        ms = (res["t_enum"] * res["workers"] * 1e3 / ne) if ne else float("nan")
        print(f"    -> {res['wall']:.1f}s wall  (verify {res['t_verify']:.1f}s, "
              f"enum {res['t_enum']:.1f}s / {ne:,} pat, "
              f"mc {res['t_mc']:.1f}s / {res['n_mc_samples']:,} pat)  "
              f"~{ms:.3f} ms/decode/core on enum", flush=True)
        for r in pl_rows:
            p = r["p"]
            tb = ss.tail_bound(n, res["w_max"], p)
            est = ss.synthesize(n, res["fw_rows"], p)
            summary.append((code_type, d, p, r["p_L"], r["p_L_se"], r["ew_mean"],
                            r["ew_se"], tb, est["frac_mc"], r["feasible"]))
            print(f"    {code_type} d={d} p={p:g}: P_L={r['p_L']:.4e} "
                  f"+/- {r['p_L_se']:.2e}  E[w|F]={r['ew_mean']:.4f} "
                  f"+/- {r['ew_se']:.4f}  tail<={tb:.2e}  "
                  f"mc-frac={est['frac_mc']:.3f}", flush=True)

    print(f"\nwrote {fw_path} ({len(fw_all)} rows)\n"
          f"wrote {out_path} ({len(pl_all)} rows)\n"
          f"total wall {time.perf_counter() - t_all:.1f}s", flush=True)

    if summary:
        print("\n=== summary ===")
        print(f"{'code':<20}{'d':<7}{'p':>9}  {'P_L':>12} {'+/-':>10}  "
              f"{'E[w|F]':>8} {'+/-':>8}  {'tail<=':>9} {'mcfrac':>7}  feas")
        for (ct, d, p, pl, pse, ew, ese, tb, frac, feas) in summary:
            frac_s = "  nan" if not math.isfinite(frac) else f"{frac:7.3f}"
            print(f"{ct:<20}{d:<7}{p:>9.1e}  {pl:>12.4e} {pse:>10.2e}  "
                  f"{ew:>8.4f} {ese:>8.4f}  {tb:>9.1e} {frac_s}  {feas}")


if __name__ == "__main__":
    main()
