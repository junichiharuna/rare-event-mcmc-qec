"""Low-p cross-code witness audit: OBF / Z+H chains at p = 1e-3 and 1e-4 (one full node).

Purpose (2026-08-28): tab:audit (Sec.4.3) currently spans only p=1e-2. This job supplies
the SAME cross-code audit at p=1e-3 and 1e-4 so the table can become a three-tier
p=1e-2 / 1e-3 / 1e-4 block, extending the "OBF is trapped where the basis-weight spread is
broad" statement from one decade to three.

WHAT THIS JOB DOES *NOT* DO -- the reference.
  It computes NO direct-MC truth of its own. Rejection MC at p=1e-4 is a rare-event problem
  in its own right (P_L(bb144, 1e-4) ~ 4e-10) and it was the entire cost of zh_rerun.py.
  The reference now comes from the weight-stratified estimator (src/subset_sampling.py ->
  cluster_results/subset_sampling.csv), exact-decomposition based and minutes, not node-days.
  This driver only LOOKS the value UP at job time (see _load_truth): a hit fills
  truth_ew / truth_se / bias / p_L, a miss writes NaN and the row is STILL emitted. Codes
  missing from that CSV today (hgp_hamming3, bb72, bb_l9m6) get their reference from a later
  subset-sampling job; the bias column is then recomputed offline from this CSV.

PROTOCOL -- identical to cluster_scripts/rect_audit_zh.py and cluster_scripts/zh_rerun.py,
which produced the p=1e-2 rows, so the three tiers need no protocol footnote:
  - kernels OBF / Z+H, both PT-off (n_replicas=1), cluster_k=2, p_cluster=0.3;
  - K=50 chains of N=2e5, min-weight ("short") AND max-weight ("long") initialisation
    (a code with a flat logical-weight basis has ONE init, tagged "uniform");
  - full-trace columns (ew_mean, ew_std, tau) PLUS the tau-based burn-in columns
    (ew_burn5 / ew_burn10 with their capped / n_used counts), 5*tau primary and 10*tau
    sensitivity, half-trace fallback when the rule has no fixed point;
  - max_error_rate is INERT at n_replicas=1 (the PT ladder is geomspace(p, r*p, 1) = [p]),
    so passing r=1 here and r=3 in zh_rerun.py is not a protocol difference.

FOUR OUTPUT CSVs.
  <out>.csv         one row per (code, p, init, method): the summary above.
  <out>_chains.csv  one row per (condition, seed, checkpoint) on the geometric grid
                    N/16, N/8, N/4, N/2, N. This is the input to the POST-HOC convergence
                    verdict: the decision is made from how the cross-chain scatter of the
                    running mean shrinks with chain length, not from an online rule. Each
                    row carries the raw running mean, the running mean after discarding the
                    first 10% of that checkpoint, the weight-change proxy, and tau measured
                    at that checkpoint.
  <out>_conv.csv    the same grid POOLED over chains (mean of the per-chain running means),
                    i.e. the convergence curve of cluster_results/rect_audit_zh_conv.csv,
                    on the 5-point grid. Derived from the checkpoint data at no extra cost.
  <out>_seeds.csv   one row per (condition, seed): the UNAGGREGATED per-chain summary --
                    ew/tau on the full trace, and (ew, var, tau, n_used, capped) on each of
                    the 5*tau and 10*tau burned segments, plus the accept proxy and wall
                    time. See GELMAN-RUBIN COLUMNS below for why it exists.

COLUMN NAMING. The summary CSV keeps rect_audit_zh MEANINGS exactly (ew_mean = raw
full-trace chain-mean average; ew_burn5 = the 5*tau-burned one), so the two files can be
concatenated. The convergence-flag diagnostics requested for the three-tier table are ADDED
as separate columns, some aliasing a compatibility column on purpose:
    requested name   ->  column here
    ew_mean (burned) ->  ew_burn5          (ew_mean stays the RAW mean, as in rect_audit_zh)
    n_capped5/10     ->  n_capped5/10      (alias of capped5/capped10)
    ew_chain_sd      ->  ew_chain_sd       (alias of ew_burn5_sd; the SD, not the SE, of the
                                            K burned chain means)
    ew_within_se     ->  median over chains of sqrt(var_burned * 2 tau / n_used)
The convergence flag is ew_chain_sd >> ew_within_se (an R-hat-like statement); both are
computed on the SAME 5*tau-burned segments, so the comparison is meaningful.

TAU CONVENTION -- read before using ew_within_se. src.metrics.iat() returns the
Geyer-truncated tau_int = 1 + 2 sum_k rho_k, the convention in which
Var(chain mean) = tau * sigma^2 / n, i.e. WITHOUT a further factor of 2.
ew_within_se carries the factor-2 form that the job specification asked for; the
factor-1 form, which compare/convergence_flags.py verified empirically against the
reference Var[w|F] of subset_sampling.csv, is emitted alongside it as
ew_within_se_tau1 = ew_within_se / sqrt(2). Use ew_within_se_tau1 for an R-hat-style
ratio against ew_chain_sd unless you have a reason not to.

GELMAN-RUBIN COLUMNS (added 2026-09-04, for convergence rule (iv)).
  ew_within_se / ew_within_se_tau1 are a MEDIAN over chains of the per-chain SE
  sqrt(var_c * tau_c / n_c) -- a robust summary, but NOT the textbook Gelman-Rubin W, and
  the per-chain var_c it consumes used to be discarded on the spot. Rule (iv) wants to
  replace the external reference Var_F[w] by a within-chain variance, so W itself must be
  in the CSV. Added, all on the PRIMARY 5*tau-burned segments:
      w_var_within_mean    = arithmetic MEAN over chains of var_burn5  <- the Gelman-Rubin W
      w_var_within_median  = MEDIAN over chains of var_burn5           <- cross-check of the
                                                                          old median form
      n_used5_mean         = mean over chains of n_used5
      tau_burn5_mean       = mean over chains of tau_burn5
  Downstream (compare/convergence_flags.py) forms s_W = sqrt(w_var_within_mean *
  tau_burn5 / n_used5); tau_burn5 / n_used5 are the MEDIAN forms already in the file and
  the *_mean twins above let that choice be revisited offline. Both tau and n_used are in
  full-step units (see MEMORY / THINNING), and var is unit-free, so every such ratio is
  stride-invariant exactly as ew_within_se is.
  ew_within_se_tau1 is KEPT UNCHANGED for continuity with the published tables.
  <out>_seeds.csv carries the raw per-chain (var, tau, n_used, capped) ingredients, so any
  other aggregation -- pooled-with-weights W, trimmed means, per-chain R-hat -- can be
  redone offline WITHOUT a cluster rerun. That file is the whole point of the rerun.

MEMORY / THINNING. run_rimc_mixed(record_trace=False) returns a ZERO-LENGTH trace, so tau
and burn-in become uncomputable -- the trace must be recorded. It is cheap: N=2e6 int32 =
8 MB per chain, under 1 GB across 96 workers, and no chain's trace ever leaves its worker
(only the ~5 checkpoint tuples and the scalar summary are pickled back). The real cost is
the ~11 iat() calls of the two burn-in fixed-point loops, each an rfft at n=2*N. Those run
on a THINNED copy: stride k = ceil(N / _TAU_MAX_SAMPLES) keeps <= 2e5 points, and tau /
n_used are reported back in FULL-step units (multiplied by k). k == 1 for every N <= 2e5,
so the standard-budget rows are bit-for-bit the rect_audit_zh computation.

RESUME. The output CSVs are read-modify-written after every COMPLETED condition, so a
walltime kill keeps the finished rows and a resubmit continues (same convention as
cluster_scripts/subset_sampling.sh). Conditions already present are skipped unless --force.
"Present" is decided from the SUMMARY CSV alone (_row_key over code/p/init/method); the
three per-chain files are read back and appended to, never consulted. One consequence:
resuming into a summary written BEFORE the Gelman-Rubin columns existed silently produces a
partial <out>_seeds.csv and blank W columns on the old rows -- the old conditions are in the
skip set and never re-emit per-chain rows. That is why the GR rerun uses a NEW --out-csv
stem (cluster_scripts/lowp_audit_2e6_gr.sh) instead of resuming the old one.

DEPOLARIZING (2026-10). `--noise depolarizing` runs the SAME protocol with the depolarizing
sampler (runners.run_rimc_depolarizing), the depolarizing decoders of css_code, and the
reference cluster_results/subset_sampling_depol.csv (rows filtered on its `noise` column;
the default stays bit_flip and byte-identical). Start state: e_X = the chosen L_X basis
element (short/long via _reorder_logicals), e_Z = 0; the weight is |e_X OR e_Z|. That
sampler returns no move stats, so h_accept_rate is NaN in this mode.
"""
import argparse
import csv
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from css_code import (  # noqa: E402
    CSSCode, make_asymmetric_toric_code, make_bb_code, make_bb_code_72, make_bb_code_144,
    make_bposd_decoder, make_concat_steane_hard_decoder, make_concatenated_steane_code,
    make_depolarizing_bposd_decoder, make_depolarizing_mwpm_decoder,
    make_mwpm_decoder, make_surface_code, make_toric_code,
)
from runners import run_rimc_depolarizing, run_rimc_mixed  # noqa: E402
from metrics import iat  # noqa: E402
from postprocess import burn_in  # noqa: E402

# Burn-in policy (docs/results_burnin_thinning_2026-08-14.md, paper sec:setup):
# tau-based burn-in, c=5 primary and c=10 sensitivity, cross-chain SE, NO thinning of the
# reported average (the striding below is a compute shortcut for the tau estimate only).
_BURN_CS = (5, 10)

# Cap on the number of samples handed to iat() / burn_in(). Above it the trace is strided.
_TAU_MAX_SAMPLES = 200_000

# Geometric checkpoint grid for the per-chain post-hoc convergence data, as fractions of N.
_CHECKPOINT_FRACS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)

# Fraction of each checkpoint discarded for the second running mean of <out>_chains.csv.
_CKPT_BURN_FRAC = 0.10

# Kernels, identical to the zh_rerun.py / rect_audit_zh.py definitions. All PT-off.
#   OBF = uniform single-bit flip          -H  = Zanella only (cluster removed)
#   -Z  = single flip + cluster            Z+H = Zanella + cluster  (production kernel)
METHODS = {
    "OBF": dict(p_single=1.0, p_zanella=0.0, n_replicas=1),
    "Z+H": dict(p_single=0.0, p_zanella=0.7, p_cluster=0.3, cluster_k=2, n_replicas=1),
    "-H":  dict(p_single=0.0, p_zanella=1.0, n_replicas=1),
    "-Z":  dict(p_single=0.7, p_zanella=0.0, p_cluster=0.3, cluster_k=2, n_replicas=1),
}


# -- code registry ----------------------------------------------------------------------
def _hgp_hamming3() -> CSSCode:
    from ldpc.codes import hamming_code
    from bposd.hgp import hgp
    q = hgp(hamming_code(3), hamming_code(3))
    return CSSCode(hx=q.hx.toarray().astype(np.int8), hz=q.hz.toarray().astype(np.int8),
                   lx=q.lx.toarray().astype(np.int8), lz=q.lz.toarray().astype(np.int8),
                   n=q.N)


def _bb_l9m6() -> CSSCode:
    return make_bb_code(9, 6, a_terms=[(3, 0), (0, 1), (0, 2)],
                        b_terms=[(0, 3), (1, 0), (2, 0)])


# name -> (builder, decoder kind, subset_sampling code_type, subset_sampling d candidates).
# The decoder rule follows direct_mc._make_code_and_decoder and the paper (sec:setup-decoders):
# BP-OSD for LDPC families (HGP, BB), MWPM for topological codes, lookup for concat. Steane.
# The d candidates are the spellings the reference CSV may use; the lookup also falls back
# to a unique (code_type, p) match, because bb144 is written there with the placeholder
# d = 0 and the codes not yet measured may use a different placeholder.
CODES = {
    "hgp_hamming3": (_hgp_hamming3,    "bposd", "hgp_hamming3", ("0", "3")),
    "bb72":         (make_bb_code_72,  "bposd", "bb72",         ("0", "6")),
    "bb144":        (make_bb_code_144, "bposd", "bb144",        ("0", "12")),
    "bb_l9m6":      (_bb_l9m6,         "bposd", "bb_l9m6",      ("0", "10")),
    "ut5x5":   (lambda: make_asymmetric_toric_code(5, 5),  "mwpm",
                "asymmetric_toric", ("5x5",)),
    "ut3x11":  (lambda: make_asymmetric_toric_code(3, 11), "mwpm",
                "asymmetric_toric", ("3x11",)),
    "ut3x15":  (lambda: make_asymmetric_toric_code(3, 15), "mwpm",
                "asymmetric_toric", ("3x15",)),
    "toric5":   (lambda: make_toric_code(5),   "mwpm", "toric",   ("5",)),
    "toric7":   (lambda: make_toric_code(7),   "mwpm", "toric",   ("7",)),
    "surface3": (lambda: make_surface_code(3), "mwpm", "surface", ("3",)),
    "surface5": (lambda: make_surface_code(5), "mwpm", "surface", ("5",)),
    "surface7": (lambda: make_surface_code(7), "mwpm", "surface", ("7",)),
    "concat_steane2": (lambda: make_concatenated_steane_code(2), "lookup", "concat_steane", ("2",)),
}


def build_code(name) -> CSSCode:
    if name not in CODES:
        raise ValueError(f"unknown code {name!r} (known: {', '.join(CODES)})")
    return CODES[name][0]()


def build_decoder(name, code, p, noise="bit_flip"):
    kind = CODES[name][1]
    if kind == "lookup":
        if noise == "depolarizing":
            raise ValueError(f"{name}: lookup decoder has no depolarizing variant")
        return make_concat_steane_hard_decoder(int(CODES[name][3][0]))
    if noise == "depolarizing":
        return (make_depolarizing_bposd_decoder(code, p) if kind == "bposd"
                else make_depolarizing_mwpm_decoder(code))
    return make_bposd_decoder(code, p) if kind == "bposd" else make_mwpm_decoder(code)


def _reorder_logicals(code, first_idx):
    """Put logical `first_idx` first: run_rimc_mixed starts every chain in the L_X[0] coset,
    so the initialisation sector is chosen by reordering the basis (as in zh_rerun.py)."""
    order = [first_idx] + [i for i in range(code.lx.shape[0]) if i != first_idx]
    return CSSCode(hx=code.hx, hz=code.hz,
                   lx=code.lx[order].copy(), lz=code.lz[order].copy(), n=code.n)


# -- per-chain statistics ---------------------------------------------------------------
def _stride(n):
    """Sub-sampling stride that keeps at most _TAU_MAX_SAMPLES points. 1 for N <= 2e5."""
    return max(1, int(math.ceil(n / _TAU_MAX_SAMPLES)))


def _checkpoints(n):
    """Ascending, de-duplicated checkpoint lengths on _CHECKPOINT_FRACS, always ending at n."""
    return sorted({max(int(round(f * n)), 1) for f in _CHECKPOINT_FRACS})


def _accept_proxy(trace):
    """Fraction of consecutive recorded weights that differ.

    A PROXY, not an MH acceptance rate: run_rimc_mixed exposes no global acceptance counter,
    and adding one to src/kernels.py would perturb other jobs' reproducibility. It
    undercounts for Z+H, whose cluster exchange is weight-preserving by construction (the
    true cluster acceptance is reported separately as h_accept_rate). Never quote it as an
    acceptance rate in the paper.
    """
    return float(np.mean(trace[1:] != trace[:-1])) if trace.size > 1 else float("nan")


def _ckpt_rows(trace):
    """Per-chain post-hoc convergence data. -> [(m, ew_running, ew_running_burn10pct,
    accept_rate, tau)] on the geometric grid.

    Computed by slicing the recorded trace (no full-trace COPY, no cumsum array): each
    entry is a reduction over trace[:m], so the memory stays the 8 MB of the trace itself.
    tau is measured on a strided view of the same prefix, so its cost is bounded by
    _TAU_MAX_SAMPLES regardless of N.
    """
    out = []
    for m in _checkpoints(len(trace)):
        pre = trace[:m]
        cut = int(_CKPT_BURN_FRAC * m)
        k = _stride(m)
        out.append((int(m), float(np.mean(pre)),
                    float(np.mean(pre[cut:])) if m - cut > 0 else float("nan"),
                    _accept_proxy(pre),
                    float(iat(np.asarray(pre[::k], dtype=float))) * k))
    return out


def _chain_stats(trace):
    """Per-chain summary of one weight trace. All tau / length values in FULL-step units.

    -> dict(ew, tau, burn={c: (mean, var, tau, capped, n_used)})

    The tau work (iat, and the fixed-point loop inside burn_in -- up to _TAU_ITERS rffts
    per c) runs on a STRIDED copy, so an N=2e6 chain costs the same FFTs as an N=2e5 one.
    tau and n_used are converted back to full steps by multiplying by the stride, which
    leaves ew_within_se = sqrt(var * 2 tau / n_used) invariant (both factors scale by k).
    The strided tau is a faithful rescaling only while k << tau; at N=2e6 the measured tau
    is ~3e4 against k=10, comfortably so.
    """
    k = _stride(len(trace))
    sub = np.asarray(trace[::k], dtype=float)
    out = {"ew": float(np.mean(trace)), "tau": float(iat(sub)) * k, "burn": {}}
    for c in _BURN_CS:
        burned, info = burn_in(sub, method="tau", c=float(c))
        out["burn"][c] = (
            float(np.mean(burned)) if burned.size else float("nan"),
            float(np.var(burned, ddof=1)) if burned.size > 1 else 0.0,
            float(info["tau_hat"]) * k,
            bool(info["capped"]),
            int(burned.size) * k,
        )
        # n_used is in FULL-step units like tau; a stride bug here would silently halve
        # every ew_within_se (and so over-certify convergence), so pin it down.
        assert out["burn"][c][4] <= len(trace)
    return out


def _burn_cols(stats):
    """[per-chain _chain_stats dicts] -> flat CSV columns for each c in _BURN_CS.

    capped{c} counts the CHAINS whose cut hit the max_frac=0.5 clamp: a non-zero count
    means the c*tau rule wants to discard more than the trace holds, so ew_burn{c} is not a
    clean "burn-in at c*tau" number -- the row falls back to the half-trace convention and
    is certified by the reference instead (tab:audit caption). For a capped chain c=5 and
    c=10 share the SAME cut, so ew_burn5 == ew_burn10 there by construction, not by a bug.
    """
    cols = {}
    for c in _BURN_CS:
        means = np.array([s["burn"][c][0] for s in stats], dtype=float)
        cols[f"ew_burn{c}"] = float(means.mean())
        cols[f"ew_burn{c}_sd"] = float(means.std(ddof=1)) if len(means) > 1 else 0.0
        cols[f"tau_burn{c}"] = float(np.median([s["burn"][c][2] for s in stats]))
        cols[f"capped{c}"] = int(sum(s["burn"][c][3] for s in stats))
        cols[f"n_used{c}"] = int(np.median([s["burn"][c][4] for s in stats]))
    return cols


def _gr_cols(stats):
    """[per-chain _chain_stats dicts] -> the Gelman-Rubin within-chain ingredients.

    W is defined on the PRIMARY 5*tau-burned segments only (c=5 is hardcoded on purpose:
    these four column names are unsuffixed and must not drift with _BURN_CS). The burn
    tuple is (mean, var, tau, capped, n_used), so index 1 is var_burn5 and index 4 n_used5.

    w_var_within_mean is the textbook W = (1/K) sum_c var_c. The median twin is emitted
    beside it so the offline side can check how much the old MEDIAN-form ew_within_se
    differs from the pooled one, and n_used5_mean / tau_burn5_mean are the mean twins of
    the median n_used5 / tau_burn5 already in the summary. All are cheap reductions over
    the same list, and the raw per-chain values also go to <out>_seeds.csv.
    """
    var5 = np.array([s["burn"][5][1] for s in stats], dtype=float)
    return {
        "w_var_within_mean": float(var5.mean()),
        "w_var_within_median": float(np.median(var5)),
        "n_used5_mean": float(np.mean([s["burn"][5][4] for s in stats])),
        "tau_burn5_mean": float(np.mean([s["burn"][5][2] for s in stats])),
    }


# -- truth lookup (weight-stratified reference; NO direct MC is run here) ---------------
def _p_key(p):
    """The float->string convention of cluster_scripts/subset_sampling.py, so the two tools
    agree on what "the same p" means."""
    return f"{float(p):.12g}"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _load_truth(path, noise="bit_flip"):
    """-> {(code_type, d, p_key): row dict}, or {} if the file is missing/unreadable.

    Deliberately total: a missing or malformed reference CSV must never abort a job that
    has already spent hours on MCMC. Callers then get NaN truth and the row is still written.
    """
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as exc:                                     # noqa: BLE001
        print(f"[truth] cannot read {path}: {exc} -- truth columns will be NaN", flush=True)
        return {}
    out = {}
    for r in rows:
        if (r.get("noise") or "bit_flip") != noise:
            continue
        try:
            out[(str(r["code_type"]), str(r["d"]), _p_key(r["p"]))] = r
        except Exception:                                        # noqa: BLE001
            continue
    return out


def _truth_for(truth, name, p):
    """-> (ew, se, p_L, source). NaN and source="missing" when the reference is absent."""
    nan = float("nan")
    if not truth:
        return nan, nan, nan, "missing"
    code_type, d_cands = CODES[name][2], CODES[name][3]
    pk = _p_key(p)
    for d in d_cands:
        r = truth.get((code_type, d, pk))
        if r is not None:
            return (_f(r.get("ew_mean")), _f(r.get("ew_se")), _f(r.get("p_L")),
                    f"subset_sampling:{code_type}/{d}")
    # Fall back to a UNIQUE (code_type, p) hit: d is a placeholder for the non-topological
    # codes (bb144 is written as d=0) and its spelling for the not-yet-measured codes is
    # not knowable here.
    hits = [(key, r) for key, r in truth.items() if key[0] == code_type and key[2] == pk]
    if len(hits) == 1:
        key, r = hits[0]
        return (_f(r.get("ew_mean")), _f(r.get("ew_se")), _f(r.get("p_L")),
                f"subset_sampling:{key[0]}/{key[1]}")
    return nan, nan, nan, ("missing" if not hits else f"ambiguous({len(hits)})")


# -- pool worker ------------------------------------------------------------------------
def _mcmc_worker(task):
    name, p, init_idx, method, seed, target, cache_cap, noise = task
    t0 = time.perf_counter()
    code = _reorder_logicals(build_code(name), init_idx)
    decoder = build_decoder(name, code, p, noise)
    if noise == "depolarizing":
        # e_X = the chosen L_X basis element, e_Z = 0; the recorded weight is |e_X OR e_Z|.
        dist, trace = run_rimc_depolarizing(
            p, target, max_error_rate=p, code=code, decoder=decoder, seed=seed,
            cache_cap=cache_cap, **METHODS[method])
        mstats = {"h_proposed": 0, "h_accepted": 0}
    else:
        dist, trace, mstats = run_rimc_mixed(
            p, target, max_error_rate=p, code=code, decoder=decoder, seed=seed,
            cache_cap=cache_cap, return_move_stats=True, **METHODS[method])
    wall = time.perf_counter() - t0
    st = _chain_stats(trace)
    st["ckpt"] = _ckpt_rows(trace)
    st["accept"] = _accept_proxy(trace)
    st["h_prop"] = int(mstats["h_proposed"])
    st["h_acc"] = int(mstats["h_accepted"])
    st["wall"] = wall
    st["seed"] = seed
    # `dist` is the exact full-chain weight histogram; its mean must equal the trace mean
    # (both are the cold replica). Kept as a cheap self-consistency column.
    st["ew_dist"] = float(sum(w * pr for w, pr in dist.items()))
    return (name, p, init_idx, method), st


# -- CSV schemas ------------------------------------------------------------------------
# Block 1 keeps cluster_results/rect_audit_zh.csv semantics (see the docstring's naming
# table); block 2 is the added convergence-flag diagnostics.
FIELDS = (
    ["code", "code_type", "d", "n", "k", "lwt_min", "lwt_max", "lwt_spread", "lwts",
     "p", "init", "method",
     "ew_mean", "ew_std", "tau", "n_seeds",
     "truth_ew", "truth_se", "bias", "p_L", "truth_source"]
    + [col for c in _BURN_CS
       for col in (f"ew_burn{c}", f"ew_burn{c}_sd", f"tau_burn{c}",
                   f"capped{c}", f"n_used{c}")]
    + ["n_chains", "n_steps", "tau_max", "n_capped5", "n_capped10",
       "ew_chain_sd", "ew_within_se", "ew_within_se_tau1",
       "accept_rate", "h_accept_rate",
       "wall_per_chain", "thin_stride"]
    # Block 3, APPENDED 2026-09-04 (see the docstring's GELMAN-RUBIN COLUMNS): new columns
    # go at the END so every earlier column keeps its position and meaning byte-for-byte.
    + ["w_var_within_mean", "w_var_within_median", "n_used5_mean", "tau_burn5_mean"]
)
CHAIN_FIELDS = ["code", "p", "init", "method", "seed", "checkpoint_n",
                "ew_running", "ew_running_burn10pct", "accept_rate", "tau_at_checkpoint"]
CONV_FIELDS = ["code", "p", "init", "method", "n_samples", "ew_running", "truth_ew"]
# One row per (condition, seed): the per-chain ingredients of every aggregate above, so an
# aggregation choice can be revisited offline instead of by a cluster rerun.
SEED_FIELDS = (["code", "p", "init", "method", "seed", "ew_full", "tau_full"]
               + [col for c in _BURN_CS
                  for col in (f"ew_burn{c}", f"var_burn{c}", f"tau_burn{c}",
                              f"n_used{c}", f"capped{c}")]
               + ["accept_rate", "wall"])


def _row_key(row):
    return (str(row["code"]), _p_key(row["p"]), str(row["init"]), str(row["method"]))


def _read_rows(path):
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _write_rows(path, fields, rows):
    """Read-modify-write via a temp file + os.replace, so a walltime kill between
    conditions always leaves a complete, parseable CSV behind."""
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


# -- condition planning -----------------------------------------------------------------
def _meta(name):
    c = build_code(name)
    lwts = [int(r.sum()) for r in c.lx]
    return dict(n=c.n, k=c.lx.shape[0], lwts=lwts,
                lo=int(np.argmin(lwts)), hi=int(np.argmax(lwts)))


def _inits_for(meta, requested):
    """-> [(tag, logical index)]. A flat logical-weight basis has ONE sector, tagged
    "uniform": short and long would be the same chain, so asking for both must neither
    double the cost nor (as a naive tag filter would) emit zero rows."""
    if min(meta["lwts"]) == max(meta["lwts"]):
        return [("uniform", meta["lo"])]
    avail = {"short": meta["lo"], "long": meta["hi"],
             "minw": meta["lo"], "maxw": meta["hi"]}
    out, seen = [], set()
    for tag in requested:
        if tag not in avail:
            raise ValueError(f"unknown init {tag!r} (use short/long)")
        canon = "short" if avail[tag] == meta["lo"] else "long"
        if canon not in seen:
            seen.add(canon)
            out.append((canon, avail[tag]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=",".join(CODES))
    ap.add_argument("--p-list", default="1e-3,1e-4")
    ap.add_argument("--methods", default="OBF,Z+H")
    ap.add_argument("--inits", default="short,long")
    ap.add_argument("--mcmc-target", type=int, default=200_000)
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--n-cores", type=int, default=96)
    ap.add_argument("--cache-cap", type=int, default=200_000)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--noise", choices=("bit_flip", "depolarizing"), default="bit_flip")
    ap.add_argument("--truth-csv", default=None,
                    help="reference CSV; default subset_sampling.csv (bit_flip) or "
                         "subset_sampling_depol.csv (depolarizing)")
    ap.add_argument("--seed", type=int, default=1000, help="base seed; chain s uses seed+s")
    ap.add_argument("--force", action="store_true",
                    help="recompute conditions already present in --out-csv")
    args = ap.parse_args()

    names = [c.strip() for c in args.codes.split(",") if c.strip()]
    ps = [float(x) for x in args.p_list.split(",")]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    inits = [i.strip() for i in args.inits.split(",") if i.strip()]
    for m in methods:
        if m not in METHODS:
            ap.error(f"unknown method {m!r} (known: {', '.join(METHODS)})")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)

    if args.truth_csv is None:
        args.truth_csv = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "cluster_results",
            "subset_sampling_depol.csv" if args.noise == "depolarizing"
            else "subset_sampling.csv")
    truth = _load_truth(args.truth_csv, args.noise)
    print(f"[truth] {len(truth)} {args.noise} reference rows from {args.truth_csv}",
          flush=True)

    meta = {name: _meta(name) for name in names}
    chain_csv = args.out_csv.replace(".csv", "_chains.csv")
    conv_csv = args.out_csv.replace(".csv", "_conv.csv")
    seed_csv = args.out_csv.replace(".csv", "_seeds.csv")
    rows = [] if args.force else _read_rows(args.out_csv)
    chain_rows = [] if args.force else _read_rows(chain_csv)
    conv_rows = [] if args.force else _read_rows(conv_csv)
    seed_rows = [] if args.force else _read_rows(seed_csv)
    done = {_row_key(r) for r in rows}

    # ONE FLAT pool over every (code, p, init, method, seed): concurrency stays >= n_cores
    # in every phase, tail included. Tasks are ordered longest-expected-first (BP-OSD codes,
    # then by n) so the expensive chains start earliest and the cheap ones backfill.
    tasks, planned = [], []
    for name in names:
        for p in ps:
            for tag, idx in _inits_for(meta[name], inits):
                for mth in methods:
                    if (name, _p_key(p), tag, mth) in done:
                        print(f"skip (already in {args.out_csv}): "
                              f"{name} p={p:g} {tag}/{mth}", flush=True)
                        continue
                    planned.append((name, p, idx, mth, tag))
                    tasks += [(name, p, idx, mth, args.seed + s, args.mcmc_target,
                               args.cache_cap, args.noise) for s in range(args.n_seeds)]
    if not tasks:
        print("nothing to do (all conditions present; use --force to recompute)")
        return
    cost = {n: (20 if CODES[n][1] == "bposd" else 1) * meta[n]["n"] for n in names}
    tasks.sort(key=lambda t: -cost[t[0]])
    n_expected = {(n, p, i, m): args.n_seeds for (n, p, i, m, _t) in planned}
    tag_of = {(n, p, i, m): t for (n, p, i, m, t) in planned}
    print(f"noise={args.noise}: "
          f"{len(planned)} conditions, {len(tasks)} chains of N={args.mcmc_target} "
          f"over {args.n_cores} cores "
          f"(tau stride {_stride(args.mcmc_target)})", flush=True)

    agg = {}
    pool = Pool(args.n_cores) if args.n_cores > 1 else None
    results = (pool.imap_unordered(_mcmc_worker, tasks, chunksize=1) if pool
               else (_mcmc_worker(t) for t in tasks))
    t_start = time.perf_counter()
    for key, st in results:
        agg.setdefault(key, []).append(st)
        if len(agg[key]) < n_expected[key]:
            continue
        # Condition complete -> emit its rows and rewrite all three CSVs (resume-safe).
        stats = sorted(agg.pop(key), key=lambda s: s["seed"])
        name, p, _idx, mth = key
        tag = tag_of[key]
        m = meta[name]
        ews = np.array([s["ew"] for s in stats])
        bcols = _burn_cols(stats)
        gcols = _gr_cols(stats)
        t_ew, t_se, t_pl, t_src = _truth_for(truth, name, p)
        # Both factors of the within-chain SE come from the SAME 5*tau-burned segment and
        # in the same (full-step) units, so the ratio tau/n_used is stride-invariant.
        # The 2*tau here is the form the job spec asked for; ew_within_se_tau1 below is
        # the tau_int form that matches src.metrics.iat() (see the module docstring).
        within = np.array([
            math.sqrt(s["burn"][5][1] * 2.0 * max(s["burn"][5][2], 1.0)
                      / max(s["burn"][5][4], 1)) for s in stats])
        h_prop = sum(s["h_prop"] for s in stats)
        rows.append({
            "code": name, "code_type": CODES[name][2], "d": CODES[name][3][0],
            "n": m["n"], "k": m["k"], "lwt_min": min(m["lwts"]), "lwt_max": max(m["lwts"]),
            "lwt_spread": max(m["lwts"]) - min(m["lwts"]),
            "lwts": "|".join(map(str, sorted(m["lwts"]))), "p": p, "init": tag,
            "method": mth,
            "ew_mean": float(ews.mean()),
            "ew_std": float(ews.std(ddof=1)) if len(ews) > 1 else 0.0,
            "tau": float(np.median([s["tau"] for s in stats])), "n_seeds": len(ews),
            "truth_ew": t_ew, "truth_se": t_se, "bias": float(ews.mean()) - t_ew,
            "p_L": t_pl, "truth_source": t_src, **bcols,
            "n_chains": len(ews), "n_steps": args.mcmc_target,
            "tau_max": float(np.max([s["tau"] for s in stats])),
            "n_capped5": bcols["capped5"], "n_capped10": bcols["capped10"],
            "ew_chain_sd": bcols["ew_burn5_sd"],
            "ew_within_se": float(np.median(within)),
            "ew_within_se_tau1": float(np.median(within)) / math.sqrt(2.0),
            "accept_rate": float(np.mean([s["accept"] for s in stats])),
            "h_accept_rate": (sum(s["h_acc"] for s in stats) / h_prop
                              if h_prop else float("nan")),
            "wall_per_chain": float(np.mean([s["wall"] for s in stats])),
            "thin_stride": _stride(args.mcmc_target),
            **gcols,
        })
        for s in stats:
            seed_rows.append({
                "code": name, "p": p, "init": tag, "method": mth, "seed": s["seed"],
                "ew_full": s["ew"], "tau_full": s["tau"],
                "accept_rate": s["accept"], "wall": s["wall"],
                **{col: val
                   for c in _BURN_CS
                   for col, val in ((f"ew_burn{c}", s["burn"][c][0]),
                                    (f"var_burn{c}", s["burn"][c][1]),
                                    (f"tau_burn{c}", s["burn"][c][2]),
                                    (f"n_used{c}", s["burn"][c][4]),
                                    (f"capped{c}", int(s["burn"][c][3])))},
            })
        for s in stats:
            for (cn, ew_r, ew_b, acc, tau_c) in s["ckpt"]:
                chain_rows.append({
                    "code": name, "p": p, "init": tag, "method": mth, "seed": s["seed"],
                    "checkpoint_n": cn, "ew_running": ew_r,
                    "ew_running_burn10pct": ew_b, "accept_rate": acc,
                    "tau_at_checkpoint": tau_c})
        # Pooled curve on the same grid: mean over chains of the per-chain running means.
        by_ckpt = {}
        for s in stats:
            for (cn, ew_r, _b, _a, _t) in s["ckpt"]:
                by_ckpt.setdefault(cn, []).append(ew_r)
        for cn in sorted(by_ckpt):
            conv_rows.append({"code": name, "p": p, "init": tag, "method": mth,
                              "n_samples": cn,
                              "ew_running": float(np.mean(by_ckpt[cn])),
                              "truth_ew": t_ew})
        _write_rows(args.out_csv, FIELDS, rows)
        _write_rows(chain_csv, CHAIN_FIELDS, chain_rows)
        _write_rows(conv_csv, CONV_FIELDS, conv_rows)
        _write_rows(seed_csv, SEED_FIELDS, seed_rows)
        print(f"  [{time.perf_counter() - t_start:7.0f}s] {name} p={p:g} {tag}/{mth}: "
              f"ew={float(ews.mean()):.4f} burn5={bcols['ew_burn5']:.4f} "
              f"bias={float(ews.mean()) - t_ew:+.4f} "
              f"(tau={bcols['tau_burn5']:.0f}, capped={bcols['capped5']}/{len(ews)}, "
              f"chain_sd={bcols['ew_burn5_sd']:.4f}, "
              f"within_se={float(np.median(within)):.4f}, "
              f"acc={float(np.mean([s['accept'] for s in stats])):.3f}, "
              f"{float(np.mean([s['wall'] for s in stats])):.1f}s/chain)", flush=True)
    if pool:
        pool.close()
        pool.join()
    print(f"wrote {len(rows)} rows -> {args.out_csv}; "
          f"{len(chain_rows)} chain rows -> {chain_csv}; "
          f"{len(conv_rows)} conv rows -> {conv_csv}; "
          f"{len(seed_rows)} seed rows -> {seed_csv}")


if __name__ == "__main__":
    main()
