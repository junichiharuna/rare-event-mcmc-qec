"""Depolarizing-channel audit: do the bit-flip pillar-B findings (OBF traps on
spread-weight codes; the cluster move H de-biases) survive on the joint X/Z manifold?

For each code and p: an MCMC-independent direct-MC truth E[w|F] (w = support size
|ex OR ez|) and the OBF / Z+H depolarizing chains started in the min-weight vs
max-weight logical sector, with tau and the E[w]-vs-budget convergence curve.
Z+H uses the support-preserving depolarizing cluster move added in B1
(run_rimc_depolarizing(..., p_cluster=0.3, n_replicas=1)).

Verdict the run answers (see docs/superpowers/specs/2026-06-23-depolarizing-run-design.md):
do the easy codes (toric, bb72) stay unbiased; is bb144 again the sole biased one;
does the cluster move close it.

Burn-in columns (2026-08-14): in ADDITION to the full-trace columns (ew_mean, ew_std,
tau -- kept byte-compatible with the June run so cluster_results/depol_audit.csv stays
reproducible from them), every row carries the tau-based burn-in statistics adopted in
docs/results_burnin_thinning_2026-08-14.md: ew_burn5 / ew_burn5_sd / tau_burn5 /
capped5 / n_used5 (primary, c=5) and the c=10 sensitivity analogues. No thinning.
See _burn_stats for the per-seed burn convention. Run parameters are UNCHANGED from
June (same conditions, seeds, kernels and budgets); output goes to a new CSV.

Scope of this driver: p=1e-2 for all codes (direct-MC truth reachable). The bb144
low-p conditions are now reachable in-driver via --codes/--p-list (P6); the truth
phase is capped by --truth-mmax.
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import direct_mc  # noqa: E402
from css_code import (  # noqa: E402
    CSSCode, make_toric_code, make_bb_code_72, make_bb_code_144,
    make_depolarizing_mwpm_decoder, make_depolarizing_bposd_decoder,
)
from runners import run_rimc_depolarizing  # noqa: E402
from metrics import iat, running_mean  # noqa: E402
from postprocess import burn_in  # noqa: E402

# Burn-in policy (docs/results_burnin_thinning_2026-08-14.md): tau-based burn-in with
# c=5 as the primary and c=10 as the sensitivity check; cross-chain SE; NO thinning.
_BURN_CS = (5, 10)

METHODS = {
    "OBF": dict(p_single=1.0, p_zanella=0.0, n_replicas=1),
    "Z+H": dict(p_zanella=0.7, p_cluster=0.3, cluster_k=2, n_replicas=1),
}

_BB = ("bb72", "bb144")

# (code_type, d, p, [methods]); d ignored for bb codes.
def conditions(smoke):
    if smoke:
        return [("toric", 3, 1e-2, ["OBF", "Z+H"]),
                ("bb72", 0, 1e-2, ["OBF", "Z+H"])]
    conds = []
    # Topological controls = small toric d=3,5 at p=1e-2: truth reachable AND OBF converges
    # (smoke d=5 @1e-2: OBF bias -0.05). Large toric (d=7,9) was dropped: at p=1e-2 its P_L is
    # too small for rejection-MC truth, and at p=3e-2 OBF is ergodic-but-slow (under-converged
    # at smoke budget, d=9 OBF -1.25) so it muddies the "control is unbiased" message.
    for d in (3, 5):
        conds.append(("toric", d, 1e-2, ["OBF", "Z+H"]))
    conds.append(("bb72", 0, 1e-2, ["OBF", "Z+H"]))   # control: k=12, narrow spread
    conds.append(("bb144", 0, 1e-2, ["OBF", "Z+H"]))  # trap: k=12, wide spread
    return conds


def override_conditions(codes_str, p_list_str):
    """--codes 'bb144' / 'toric:5,bb72' (x --p-list '1e-3,3e-3') -> conditions list.

    Every named code at every p with both methods; ordered lowest p first (LPT:
    the rare-event truth is the long pole). Default p-list: [1e-2].
    """
    codes = []
    for tok in codes_str.split(","):
        c, _, d = tok.partition(":")
        codes.append((c, int(d) if d else 0))
    ps = sorted(float(x) for x in (p_list_str or "1e-2").split(","))
    return [(c, d, p, ["OBF", "Z+H"]) for p in ps for (c, d) in codes]


FIELDS = ["code", "d", "n", "k", "lwt_min", "lwt_max", "lwt_spread", "p", "init", "method",
          "ew_mean", "ew_std", "tau", "n_seeds", "truth_ew", "truth_se",
          "bias", "p_L", "n_faulty_truth"] + [
    col for c in _BURN_CS
    for col in (f"ew_burn{c}", f"ew_burn{c}_sd", f"tau_burn{c}",
                f"capped{c}", f"n_used{c}")]
CONV_FIELDS = ["code", "d", "p", "init", "method", "n_samples", "ew_running", "truth_ew"]


def build_code(code_type, d) -> CSSCode:
    if code_type == "toric":
        return make_toric_code(int(d))
    if code_type == "bb72":
        return make_bb_code_72()
    if code_type == "bb144":
        return make_bb_code_144()
    raise ValueError(f"unknown code_type {code_type}")


def build_decoder(code, code_type, p):
    if code_type in _BB:
        return make_depolarizing_bposd_decoder(code, p)
    return make_depolarizing_mwpm_decoder(code)


def _reorder_logicals(code, first_idx):
    order = [first_idx] + [i for i in range(code.lx.shape[0]) if i != first_idx]
    return CSSCode(hx=code.hx, hz=code.hz,
                   lx=code.lx[order].copy(), lz=code.lz[order].copy(), n=code.n)


def _conv_grid(target):
    return np.unique(np.geomspace(max(target // 200, 100), target, 15).astype(int))


def _burn_stats(trace):
    """Per-seed tau-based burn-in summary. -> {c: (burned mean, tau_hat, capped, n_used)}.

    BURN CONVENTION -- PER-SEED, NOT CONDITION-LEVEL. This driver runs every seed in a
    SEPARATE Pool process, so the K traces of one condition never coexist in a single
    process and the condition-level convention of compare/analyze_burnin_thinning.py
    (one COMMON cut at the across-chain median tau, via chain_stats on a stacked (K, N)
    array) is not available here. Each chain is therefore burned with its OWN tau and
    the burned per-seed means are pooled afterwards by _burn_cols; ew_burn*_sd is the
    cross-chain std (ddof=1) of those means, exactly as ew_std is for the full trace.
    Cost is ~10 extra iat() calls per worker, parallel over the pool, and no extra
    memory: the trace is already materialised here for iat/running_mean.
    """
    out = {}
    for c in _BURN_CS:
        burned, info = burn_in(trace, method="tau", c=float(c))
        out[c] = (float(np.mean(burned)) if burned.size else float("nan"),
                  float(info["tau_hat"]), bool(info["capped"]), int(burned.size))
    return out


def _burn_cols(burn_list):
    """[per-seed _burn_stats dicts] -> flat CSV columns, for each c in _BURN_CS.

    capped{c} counts the SEEDS whose cut hit the max_frac=0.5 clamp (0..n_seeds): a
    non-zero count means the c*tau rule wanted to discard more than the trace holds,
    i.e. burn-in cannot rescue those chains and ew_burn{c} must not be quoted as a
    clean "burn-in at c*tau" number. n_used{c} is the median per-chain post-burn
    length. NOTE: for a capped chain both c=5 and c=10 fall back to the SAME cut
    (the clamp), so ew_burn5 == ew_burn10 there by construction, not by a wiring bug.
    """
    cols = {}
    for c in _BURN_CS:
        means = np.array([b[c][0] for b in burn_list], dtype=float)
        cols[f"ew_burn{c}"] = float(means.mean())
        cols[f"ew_burn{c}_sd"] = float(means.std(ddof=1)) if len(means) > 1 else 0.0
        cols[f"tau_burn{c}"] = float(np.median([b[c][1] for b in burn_list]))
        cols[f"capped{c}"] = int(sum(b[c][2] for b in burn_list))
        cols[f"n_used{c}"] = int(np.median([b[c][3] for b in burn_list]))
    return cols


def _mcmc_worker(task):
    code_type, d, p, init_idx, method, seed, target = task
    code = _reorder_logicals(build_code(code_type, d), init_idx)
    decoder = build_decoder(code, code_type, p)
    kw = METHODS[method]
    r = 1.0 if method == "OBF" else 3.0
    dist, trace = run_rimc_depolarizing(p, target, max_error_rate=r * p, code=code,
                                        decoder=decoder, seed=seed, **kw)
    ew = sum(w * pr for w, pr in dist.items())
    return ((code_type, d, p, init_idx, method), ew, iat(trace),
            running_mean(trace, _conv_grid(target)), _burn_stats(trace))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcmc-target", type=int, default=100_000)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--truth-target", type=int, default=2000)
    ap.add_argument("--truth-mmax", type=float, default=4e7)
    ap.add_argument("--n-cores", type=int, default=96)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--codes", default=None,
                    help="override condition codes: 'bb144' or 'toric:5,bb72'")
    ap.add_argument("--p-list", default=None,
                    help="override ps (with --codes): '1e-3,3e-3'")
    args = ap.parse_args()

    conds = (override_conditions(args.codes, args.p_list) if args.codes
             else conditions(args.smoke))
    target = 5000 if args.smoke else args.mcmc_target
    truth_target = 1000 if args.smoke else args.truth_target
    truth_mmax = 2e6 if args.smoke else int(args.truth_mmax)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    # per-(code_type,d) metadata + init sectors (min/max logical weight; uniform if flat)
    keys = sorted({(c, d) for c, d, _p, _m in conds})
    meta = {}
    for c, d in keys:
        code = build_code(c, d)
        lwts = [int(r.sum()) for r in code.lx]
        lo, hi = int(np.argmin(lwts)), int(np.argmax(lwts))
        inits = ([("uniform", lo)] if min(lwts) == max(lwts)
                 else [("minw", lo), ("maxw", hi)])
        meta[(c, d)] = dict(n=code.n, k=code.lx.shape[0], lwts=lwts, inits=inits)

    # Phase 1: direct-MC truth (each condition saturates the node via run_condition n_cores)
    truth = {}
    for c, d, p, _m in conds:
        row = direct_mc.run_condition(
            noise="depolarizing", code_type=c, d=int(d), p=p,
            n_target=truth_target, m_max=truth_mmax, n_cores=args.n_cores,
            decoder_kind="auto",
        )
        if row["n_faulty"] == 0:
            print(f"{c} d={d} p={p:g}: no faulty truth (raise truth-mmax)")
            continue
        truth[(c, d, p)] = dict(ew=row["ew_mean"], se=row["ew_se"],
                                p_L=row["p_L"], nf=row["n_faulty"])
        print(f"truth {c} d={d} p={p:g}: ew={row['ew_mean']:.3f}+/-{row['ew_se']:.3f} "
              f"(p_L={row['p_L']:.2e}, F={row['n_faulty']})")

    # Phase 2: MCMC, flat over cores
    mcmc_tasks = [(c, d, p, idx, mth, 1000 + s, target)
                  for c, d, p, mths in conds
                  for (_t, idx) in meta[(c, d)]["inits"]
                  for mth in mths for s in range(args.n_seeds)]
    pool = Pool(args.n_cores) if args.n_cores > 1 else None
    _map = pool.map if pool else (lambda f, it: [f(x) for x in it])
    magg = {}
    for keypart, ew, tau, curve, bstat in _map(_mcmc_worker, mcmc_tasks):
        a = magg.setdefault(keypart, dict(ew=[], tau=[], curve=[], burn=[]))
        a["ew"].append(ew); a["tau"].append(tau); a["curve"].append(curve)
        a["burn"].append(bstat)
    if pool:
        pool.close(); pool.join()

    grid = _conv_grid(target)
    rows, conv_rows = [], []
    for c, d, p, mths in conds:
        m = meta[(c, d)]
        t = truth.get((c, d, p))
        if t is None:
            continue
        for (tag, idx) in m["inits"]:
            for mth in mths:
                a = magg[(c, d, p, idx, mth)]
                ews = np.array(a["ew"])
                mean_curve = np.mean(np.array(a["curve"]), axis=0)
                bcols = _burn_cols(a["burn"])
                rows.append({
                    "code": c, "d": d, "n": m["n"], "k": m["k"],
                    "lwt_min": min(m["lwts"]), "lwt_max": max(m["lwts"]),
                    "lwt_spread": max(m["lwts"]) - min(m["lwts"]), "p": p,
                    "init": tag, "method": mth, "ew_mean": float(ews.mean()),
                    "ew_std": float(ews.std(ddof=1)) if len(ews) > 1 else 0.0,
                    "tau": float(np.median(a["tau"])), "n_seeds": len(ews),
                    "truth_ew": t["ew"], "truth_se": t["se"],
                    "bias": float(ews.mean()) - t["ew"], "p_L": t["p_L"],
                    "n_faulty_truth": t["nf"], **bcols,
                })
                print(f"  {c} d={d} p={p:g} {tag}/{mth}: ew={float(ews.mean()):.4f} "
                      f"burn5={bcols['ew_burn5']:.4f} (tau={bcols['tau_burn5']:.0f}, "
                      f"capped={bcols['capped5']}/{len(ews)}, n_used={bcols['n_used5']}) "
                      f"burn10={bcols['ew_burn10']:.4f} "
                      f"(capped={bcols['capped10']}/{len(ews)}, "
                      f"n_used={bcols['n_used10']})", flush=True)
                for nsamp, ew_n in zip(grid, mean_curve):
                    conv_rows.append({"code": c, "d": d, "p": p, "init": tag, "method": mth,
                                      "n_samples": int(nsamp), "ew_running": float(ew_n),
                                      "truth_ew": t["ew"]})
        print(f"{c} d={d} p={p:g}: "
              + ", ".join(
                  f"{tag}/{mth}:bias{np.array(magg[(c,d,p,idx,mth)]['ew']).mean()-t['ew']:+.3f}"
                  for (tag, idx) in m["inits"] for mth in mths))

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    conv_csv = args.out_csv.replace(".csv", "_conv.csv")
    with open(conv_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONV_FIELDS); w.writeheader(); w.writerows(conv_rows)
    print(f"wrote {len(rows)} rows -> {args.out_csv}; {len(conv_rows)} conv rows -> {conv_csv}")


if __name__ == "__main__":
    main()
