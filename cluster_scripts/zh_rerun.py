"""Z+H (no-PT) rerun: replace the relabeled Z+PT+H values shown as Z+H in
tab:ablation (Sec.5.1), tab:lowp (Sec.5.3), tab:audit (Sec.4.3) with the true
Z+H kernel (cluster move on, parallel tempering off, n_replicas=1), and unify the
bb144 Z+H numbers so tab:ablation and tab:lowp report ONE consistent set.

Convention (see docs/superpowers/specs/2026-06-23-zh-no-pt-rerun-design.md):
  - kernel columns use the raw chain mean ew_mean, biased against the truth;
  - the MBAR column in the paper tables is an intrinsically tempered (PT-ladder)
    estimator and is NOT recomputed here: Z+H at n_replicas=1 has no PT ladder,
    so there is no Z+H MBAR. The rerun replaces only the Z+H KERNEL column; the
    existing "MBAR" columns stay as-is (correctly labelled, an estimator
    cross-check that "reweighting de-biases nothing").

Conditions emitted (one CSV, one row per code,p,init,method):
  - bb144 at p=1e-2/1e-3/1e-4, methods {OBF, Z+H, Z, H}  -> the 2x2 factorial of
    tab:ablation AND the Z+H column of tab:lowp (shared numbers);
  - hgp_hamming3 / bb72 / bb_l9m6 at p=1e-2, methods {OBF, Z+H}  -> the Z+H
    column of tab:audit (bb144 at 1e-2 is already covered above).

Burn-in columns (2026-08-14): in ADDITION to the full-trace columns (ew_mean, ew_std,
tau -- kept byte-compatible with the June run so cluster_results/zh_rerun.csv stays
reproducible from them), every row carries the tau-based burn-in statistics adopted in
docs/results_burnin_thinning_2026-08-14.md: ew_burn5 / ew_burn5_sd / tau_burn5 /
capped5 / n_used5 (primary, c=5) and the c=10 sensitivity analogues. No thinning.
See _burn_stats for the per-seed burn convention. Run parameters are UNCHANGED from
June (same conditions, seeds, kernels and budgets); output goes to a new CSV.

Truth: rejection (direct) MC where reachable (all p=1e-2 conditions); the bb144
rare-event truth at p=1e-3/1e-4 is infeasible for rejection MC (P_L ~ 1e-8/1e-10)
and is read from TRUTH_CONST (the 400-core distributed-pool values reported in the
paper, tab:lowp: 3.073(5) and 3.053(9); 1e-2 = 4.617(35) is recomputed here).
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from css_code import (  # noqa: E402
    CSSCode, make_bb_code, make_bb_code_72, make_bb_code_144, make_bposd_decoder,
)
from runners import run_rimc_mixed  # noqa: E402
from metrics import iat, running_mean  # noqa: E402
from postprocess import burn_in  # noqa: E402

# Burn-in policy (docs/results_burnin_thinning_2026-08-14.md): tau-based burn-in with
# c=5 as the primary and c=10 as the sensitivity check; cross-chain SE; NO thinning.
_BURN_CS = (5, 10)

# All methods PT-off (n_replicas=1). The 2x2 factorial toggles {within-mode mover:
# single vs zanella} x {cluster on/off}; every arm keeps a weight-mobile base (single
# or zanella), since the cluster move alone preserves weight and would freeze the chain.
#   OBF = single, no cluster            -H  = zanella, no cluster (cluster removed)
#   -Z  = single + cluster (zanella     Z+H = zanella + cluster
#         removed)
# Bias columns are read off as: OBF/-H biased (no cluster), -Z/Z+H de-biased (cluster on).
METHODS = {
    "OBF": dict(p_single=1.0, p_zanella=0.0, n_replicas=1),
    "Z+H": dict(p_zanella=0.7, p_cluster=0.3, cluster_k=2, n_replicas=1),
    "-H":  dict(p_zanella=1.0, n_replicas=1),                                   # Z only
    "-Z":  dict(p_single=0.7, p_zanella=0.0, p_cluster=0.3, cluster_k=2, n_replicas=1),
}

# bb144 rare-event truth (paper tab:lowp; distributed direct-MC pool). (ew, se)
TRUTH_CONST = {
    ("bb144", 1e-3): (3.073, 0.005),
    ("bb144", 1e-4): (3.053, 0.009),
}

# (code, p, [methods]) conditions.
def conditions(smoke):
    if smoke:
        return [
            ("hgp_hamming3", 1e-2, ["OBF", "Z+H"]),
            ("bb72", 1e-2, ["OBF", "Z+H"]),
        ]
    conds = []
    for p in (1e-2, 1e-3, 1e-4):
        conds.append(("bb144", p, ["OBF", "Z+H", "-H", "-Z"]))  # tab:ablation 2x2 + tab:lowp
    for code in ("hgp_hamming3", "bb72", "bb_l9m6"):
        conds.append((code, 1e-2, ["OBF", "Z+H"]))            # tab:audit
    return conds


FIELDS = ["code", "n", "k", "lwt_min", "lwt_max", "lwt_spread", "p", "init", "method",
          "ew_mean", "ew_std", "tau", "n_seeds", "truth_ew", "truth_se",
          "bias", "p_L", "n_faulty_truth"] + [
    col for c in _BURN_CS
    for col in (f"ew_burn{c}", f"ew_burn{c}_sd", f"tau_burn{c}",
                f"capped{c}", f"n_used{c}")]
CONV_FIELDS = ["code", "p", "init", "method", "n_samples", "ew_running", "truth_ew"]


def build_code(name) -> CSSCode:
    if name == "bb72":
        return make_bb_code_72()
    if name == "bb144":
        return make_bb_code_144()
    if name == "bb_l9m6":
        return make_bb_code(9, 6, a_terms=[(3, 0), (0, 1), (0, 2)],
                            b_terms=[(0, 3), (1, 0), (2, 0)])
    if name == "hgp_hamming3":
        from ldpc.codes import hamming_code
        from bposd.hgp import hgp
        q = hgp(hamming_code(3), hamming_code(3))
        return CSSCode(hx=q.hx.toarray().astype(np.int8), hz=q.hz.toarray().astype(np.int8),
                       lx=q.lx.toarray().astype(np.int8), lz=q.lz.toarray().astype(np.int8),
                       n=q.N)
    raise ValueError(f"unknown code {name}")


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


def _truth_worker(task):
    name, p, n_trials, seed = task
    code = build_code(name)
    decoder = make_bposd_decoder(code, p)
    hz, n = code.hz, code.n
    rng = np.random.default_rng(seed)
    n_faulty = 0
    s1 = 0.0
    s2 = 0.0
    for _ in range(n_trials):
        e = (rng.random(n) < p).astype(np.int8)
        corr = decoder.decode_syndrome((hz @ e) % 2)
        if decoder.is_faulty(e, corr):
            w = float(e.sum())
            n_faulty += 1
            s1 += w
            s2 += w * w
    return (name, p), n_trials, n_faulty, s1, s2


def _mcmc_worker(task):
    name, p, init_idx, method, seed, target, cache_cap = task
    code = _reorder_logicals(build_code(name), init_idx)
    decoder = make_bposd_decoder(code, p)
    kw = METHODS[method]
    r = 1.0 if method == "OBF" else 3.0
    dist, trace = run_rimc_mixed(p, target, max_error_rate=r * p, code=code,
                                 decoder=decoder, seed=seed, cache_cap=cache_cap, **kw)
    ew = sum(w * pr for w, pr in dist.items())
    return ((name, p, init_idx, method), ew, iat(trace),
            running_mean(trace, _conv_grid(target)), _burn_stats(trace))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcmc-target", type=int, default=200_000)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--truth-mmax", type=float, default=4e7)
    ap.add_argument("--cache-cap", type=int, default=200_000)
    ap.add_argument("--n-cores", type=int, default=96)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    conds = conditions(args.smoke)
    target = 2000 if args.smoke else args.mcmc_target
    truth_mmax = 5e4 if args.smoke else int(args.truth_mmax)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    # per-code metadata + init sectors (min/max logical weight; uniform if flat)
    names = sorted({c for c, _p, _m in conds})
    meta = {}
    for name in names:
        c = build_code(name)
        lwts = [int(r.sum()) for r in c.lx]
        lo, hi = int(np.argmin(lwts)), int(np.argmax(lwts))
        inits = ([("uniform", lo)] if min(lwts) == max(lwts)
                 else [("minw", lo), ("maxw", hi)])
        meta[name] = dict(n=c.n, k=c.lx.shape[0], lwts=lwts, inits=inits)

    pool = Pool(args.n_cores) if args.n_cores > 1 else None
    _map = pool.map if pool else (lambda f, it: [f(x) for x in it])

    # Phase 1: truth (rejection MC, BP-OSD), flat over cores; skip TRUTH_CONST conditions
    need_truth = sorted({(c, p) for c, p, _m in conds if (c, p) not in TRUTH_CONST})
    per = int(truth_mmax) // max(args.n_cores, 1)
    truth_tasks = [(c, p, per, 7000 + i) for (c, p) in need_truth
                   for i in range(max(args.n_cores, 1))]
    tagg = {}
    for key, nt, nf, s1, s2 in _map(_truth_worker, truth_tasks):
        a = tagg.setdefault(key, [0, 0, 0.0, 0.0])
        a[0] += nt; a[1] += nf; a[2] += s1; a[3] += s2
    truth = {}
    for key, (nt, nf, s1, s2) in tagg.items():
        if nf == 0:
            continue
        ew = s1 / nf
        truth[key] = dict(ew=ew, se=(max(s2 / nf - ew ** 2, 0.0) / nf) ** 0.5,
                          p_L=nf / nt, nf=nf)
    for key, (ew, se) in TRUTH_CONST.items():
        if key in {(c, p) for c, p, _m in conds}:
            truth[key] = dict(ew=ew, se=se, p_L=float("nan"), nf=-1)

    # Phase 2: MCMC, flat over cores
    mcmc_tasks = [(c, p, idx, mth, 1000 + s, target, args.cache_cap)
                  for c, p, mths in conds
                  for (_t, idx) in meta[c]["inits"]
                  for mth in mths for s in range(args.n_seeds)]
    magg = {}
    for keypart, ew, tau, curve, bstat in _map(_mcmc_worker, mcmc_tasks):
        a = magg.setdefault(keypart, dict(ew=[], tau=[], curve=[], burn=[]))
        a["ew"].append(ew); a["tau"].append(tau); a["curve"].append(curve)
        a["burn"].append(bstat)
    if pool:
        pool.close(); pool.join()

    grid = _conv_grid(target)
    rows, conv_rows = [], []
    for c, p, mths in conds:
        m = meta[c]
        t = truth.get((c, p))
        if t is None:
            print(f"{c} p={p:g}: no faulty truth (raise truth-mmax)")
            continue
        for (tag, idx) in m["inits"]:
            for mth in mths:
                a = magg[(c, p, idx, mth)]
                ews = np.array(a["ew"])
                mean_curve = np.mean(np.array(a["curve"]), axis=0)
                bcols = _burn_cols(a["burn"])
                rows.append({
                    "code": c, "n": m["n"], "k": m["k"],
                    "lwt_min": min(m["lwts"]), "lwt_max": max(m["lwts"]),
                    "lwt_spread": max(m["lwts"]) - min(m["lwts"]), "p": p,
                    "init": tag, "method": mth, "ew_mean": float(ews.mean()),
                    "ew_std": float(ews.std(ddof=1)) if len(ews) > 1 else 0.0,
                    "tau": float(np.median(a["tau"])), "n_seeds": len(ews),
                    "truth_ew": t["ew"], "truth_se": t["se"],
                    "bias": float(ews.mean()) - t["ew"], "p_L": t["p_L"],
                    "n_faulty_truth": t["nf"], **bcols,
                })
                print(f"  {c} p={p:g} {tag}/{mth}: ew={float(ews.mean()):.4f} "
                      f"burn5={bcols['ew_burn5']:.4f} (tau={bcols['tau_burn5']:.0f}, "
                      f"capped={bcols['capped5']}/{len(ews)}, n_used={bcols['n_used5']}) "
                      f"burn10={bcols['ew_burn10']:.4f} "
                      f"(capped={bcols['capped10']}/{len(ews)}, "
                      f"n_used={bcols['n_used10']})", flush=True)
                for nsamp, ew_n in zip(grid, mean_curve):
                    conv_rows.append({"code": c, "p": p, "init": tag, "method": mth,
                                      "n_samples": int(nsamp), "ew_running": float(ew_n),
                                      "truth_ew": t["ew"]})
        print(f"{c} p={p:g}: truth_ew={t['ew']:.3f}+/-{t['se']:.3f} (p_L={t['p_L']:.2e}); "
              + ", ".join(
                  f"{tag}/{mth}:bias{np.array(magg[(c,p,idx,mth)]['ew']).mean()-t['ew']:+.3f}"
                  for (tag, idx) in m["inits"] for mth in mths))

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    conv_csv = args.out_csv.replace(".csv", "_conv.csv")
    with open(conv_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONV_FIELDS); w.writeheader(); w.writerows(conv_rows)
    print(f"wrote {len(rows)} rows -> {args.out_csv}; {len(conv_rows)} conv rows -> {conv_csv}")


if __name__ == "__main__":
    main()
