"""Direct-MC vs OBF speedup table (lower-bounded) from one or more dmc_speedup runs.

Reads cluster_results/dmc_speedup*.csv (phases: dmc_cutoff, advantage), computes
per (code,p): direct-MC time-to-2%-RMSE LOWER BOUND (cutoff extrapolation),
OBF/Z+H wall-to-accuracy, and the two lower-bound speedups. With several CSVs
(independent repeat runs, disjoint seeds) each run is analyzed separately, the times
are averaged over runs, and the speedup is the ratio of the mean times (see
combine_runs). Emits a numbers CSV, a LaTeX table on stdout, and
the same table to paper/tables/dmc_speedup_table.tex (the paper \\input's it).
Both sides are charged in single-core-equivalent core-seconds: direct MC by the
core-seconds it needs to collect N* faulty draws, the MCMC kernels by the summed
wall time of all S chains whose pooled estimate certifies the accuracy target.

CSVs are grouped by RUN_ID (see group_paths_by_run), so a per-code top-up file such as
dmc_speedup_run7_bb108.csv is folded into run 7 instead of counting as an extra run.

The reference E[w|F] of a cell comes from the first available tier of truth_for():
weight-stratified (cluster_results/subset_sampling.csv) -> direct-MC cutoff -> direct-MC
truth grid -> the Z+H chain itself. The first three are MCMC-independent, so they test
the chain's VALUE; the last only tests whether it has converged. Every tier reports its
own standard error, which enters the RMSE of wall_to_precision.

Usage: python compare/analyze_dmc_speedup.py [run1.csv run2.csv ...]
"""
import csv
import os
import re
import sys

import numpy as np

RELEPS = 0.02


def n_target(ew, vw, releps=RELEPS):
    """Faulty samples for relative precision `releps` on E[w|F]: vw/(ew^2*releps^2)."""
    return vw / (ew ** 2 * releps ** 2)


def dmc_time_lower_bound(n_faulty, wall, n_cores, ew, vw, releps=RELEPS):
    """Single-core-seconds estimate of direct MC's time to 2% precision, with flags.

    Cutoff run spent C = n_cores*wall core-seconds and observed `n_faulty` faulty
    draws. Three cases:
      converged — the cutoff run itself reached the target (n_faulty >= N* with a
        measured V[w|F]); single-core faulty rate = n_faulty/C, so the time to N*
        faulty on one core is N*·C/n_faulty, a MEASUREMENT (interpolation within
        the run), not a bound — at moderate p the budget exceeds N* by many orders.
      hard — n_faulty == 0; substitute 1 (an upper bound on the rate), so the
        time is N*·C, the budget floor for the N* faulty draws the target needs.
      partial — 0 < n_faulty < N* (finite N*): the run spent C and did not reach
        the target, so the time is charged as C itself, a strict floor.  (The
        extrapolation N*·C/n_faulty would be a point estimate, not a floor.)
    N* is floored to >=1 (>=1 faulty is needed to estimate E[w] at all); if V[w|F]
    is unavailable (vw NaN), N* is unknown and set to 1, so the time is C/n_faulty
    (time to one faulty draw), a floor, never `converged`.  Charging C there would
    not be a floor: with N* unknown the run may already have had enough draws.
    Returns (t, hard, converged).
    """
    nstar_raw = n_target(ew, vw, releps)
    converged = bool(np.isfinite(nstar_raw) and n_faulty >= max(nstar_raw, 1.0))
    nstar = nstar_raw if np.isfinite(nstar_raw) and nstar_raw >= 1.0 else 1.0
    hard = n_faulty == 0
    C = n_cores * wall
    if hard:
        return nstar * C, hard, converged
    if converged or not np.isfinite(nstar_raw):
        return nstar * C / int(n_faulty), hard, converged
    return C, hard, converged          # partial: strict floor = budget spent


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def wall_to_precision(rows, code, p, method, truth, releps=RELEPS, truth_se=0.0):
    """Return the first *sustained* cost at which
    max(|seed-pooled mean - truth|, SE across seeds) / truth is below ``releps``, in
    single-core-equivalent core-seconds.

    The SE condition is the one direct MC meets (relative SE below the target); the
    deviation condition is added because a chain, unlike direct MC, can be biased.
    The two are not added in quadrature: the realized deviation already contains the
    statistical scatter, and the earlier sqrt(bias**2 + SE**2 + truth_se**2) counted
    that variance twice.  ``truth_se`` is accepted for the callers' signature and not
    used.  The estimate
    comes from all S chains, so the cost charged is what all S chains consumed to
    reach that checkpoint: the sum of their per-chain wall times, burn-in included.
    Charging one chain's wall while crediting the S-chain standard error would
    understate the cost by a factor S and make the comparison with the single-core
    direct-MC time in ``dmc_time_lower_bound`` inconsistent.

    A checkpoint counts only if it and every later checkpoint meet the target,
    avoiding an optimistic first-hitting time from a chance crossing. If the target
    is never sustained, return the largest-grid cost and ``False``.
    """
    adv = [r for r in rows if r["phase"] == "advantage" and r["code"] == code
           and abs(_f(r["p"]) - p) < 1e-12 and r["method"] == method]
    by_len = {}
    for r in adv:
        by_len.setdefault(int(r["length"]), []).append(r)
    checkpoints = []
    for L in sorted(by_len):
        ews = np.array([_f(r["ew"]) for r in by_len[L]])
        walls = np.array([_f(r["wall"]) for r in by_len[L]])
        ews = ews[np.isfinite(ews)]
        # total core-seconds spent by all S chains up to this checkpoint
        wall = float(walls[np.isfinite(walls)].sum())
        if len(ews) < 2 or not truth:
            rel_err = float("inf")
        else:
            se = float(ews.std(ddof=1) / np.sqrt(len(ews)))
            rel_err = max(abs(float(ews.mean() - truth)), se) / abs(truth)
        checkpoints.append((wall, rel_err < releps))
    if not checkpoints:
        return float("nan"), False
    for i, (wall, _) in enumerate(checkpoints):
        if all(ok for _, ok in checkpoints[i:]):
            return wall, True
    return checkpoints[-1][0], False


def wall_to_precision_strict(rows, code, p, method, truth, releps=RELEPS, truth_se=0.0):
    """``wall_to_precision``, except that a target first met at the FINAL checkpoint does
    not count as reached.

    The longest length has no successor, so ``wall_to_precision`` calls any crossing there
    "sustained" by construction, and a chance crossing is then indistinguishable from a
    real one.  Such a cell carries no more information than an unreached one and is
    reported the same way: the cost of the final checkpoint with ``False`` (a lower bound).
    A cell counts as reached only if the sustained stretch already starts on the grid with
    this cell's longest length removed and yields the same cost.  The longest length is
    that of the (code, p, method) cell itself, not of the whole file, so a cell extended to
    longer chains does not mask the final checkpoint of another.
    """
    t, ok = wall_to_precision(rows, code, p, method, truth, releps=releps, truth_se=truth_se)
    if not ok:
        return t, False
    sel = [r for r in rows if r["phase"] == "advantage" and r["code"] == code
           and abs(_f(r["p"]) - p) < 1e-12 and r["method"] == method]
    lmax = max(int(r["length"]) for r in sel)
    shorter = [r for r in sel if int(r["length"]) < lmax]
    t2, ok2 = wall_to_precision(shorter, code, p, method, truth, releps=releps, truth_se=truth_se)
    return t, bool(ok2 and np.isfinite(t2) and abs(t - t2) <= 1e-9 * max(abs(t), 1.0))


def speedup(t_ref, t_zh, ref_reached):
    """(ratio, is_lower_bound). is_lower_bound True when the reference (OBF or dMC)
    did not actually converge, so t_ref is itself a floor."""
    if not (t_zh and np.isfinite(t_zh)):
        return float("nan"), False
    return t_ref / t_zh, (not ref_reached)


ROW_ORDER = ["surface:3", "surface:5", "surface:7", "toric:5", "toric:7",
             "concat_steane:2", "hgp_hamming",
             "rect:3x11", "rect:3x15", "bb72", "bb144", "bb_l9m6"]  # LDPC set by basis-weight spread 8 < 12 < 18, as in tab:meanw
# row labels matching the paper's tab:codes naming
ROW_LABEL = {
    "surface:3": r"surface $d=3$",
    "surface:5": r"surface $d=5$",
    "surface:7": r"surface $d=7$",
    "toric:5": r"toric $d=5$",
    "toric:7": r"toric $d=7$",
    "concat_steane:2": r"concat.\ Steane ($\ell=2$)",
    "hgp_hamming": r"$\mathrm{HGP}(\text{Hamming})$",
    "rect:3x11": r"$\mathrm{AT}(3,11)$",
    "rect:3x15": r"$\mathrm{AT}(3,15)$",
    "bb72": r"BB $[[72,12,6]]$",
    "bb_l9m6": r"BB $[[108,8,10]]$",
    "bb144": r"BB $[[144,12,12]]$",
}
P_ORDER = [1e-1, 1e-3, 1e-5]
MIN_FAULTY_FOR_TRUTH = 30

CSV = os.path.join(os.path.dirname(__file__), "..", "cluster_results", "dmc_speedup.csv")
GRID = os.path.join(os.path.dirname(__file__), "..", "cluster_results", "direct_mc_ew.csv")
STRAT = os.path.join(os.path.dirname(__file__), "..", "cluster_results", "subset_sampling.csv")
OUT = os.path.join(os.path.dirname(__file__), "..", "cluster_results", "dmc_speedup_table.csv")
TEX = os.path.join(os.path.dirname(__file__), "..", "paper", "tables", "dmc_speedup_table.tex")

# map dmc_speedup code specs -> direct_mc_ew.csv code_type (grid fallback)
GRID_CODE = {"bb144": "bb144", "bb72": "bb72"}
# map dmc_speedup code specs -> (code_type, d) as spelled in subset_sampling.csv
# (the preferred reference tier).  The stratified driver names codes by family and
# writes the distance-free families with d="0" and the rectangular ones as "3x11".
STRAT_CODE = {
    "surface:3": ("surface", "3"),
    "surface:5": ("surface", "5"),
    "surface:7": ("surface", "7"),
    "toric:5": ("toric", "5"),
    "toric:7": ("toric", "7"),
    "concat_steane:2": ("concat_steane", "2"),
    "hgp_hamming": ("hgp_hamming3", "0"),
    "rect:3x11": ("asymmetric_toric", "3x11"),
    "rect:3x15": ("asymmetric_toric", "3x15"),
    "bb72": ("bb72", "0"),
    "bb144": ("bb144", "0"),
    "bb_l9m6": ("bb_l9m6", "0"),
}


def _cutoff_stats(rows, code, p):
    """Pooled (n_trials, n_faulty, s1, s2, wall, n_cores) for the dmc_cutoff phase."""
    cut = [r for r in rows if r["phase"] == "dmc_cutoff" and r["code"] == code
           and abs(_f(r["p"]) - p) < 1e-12]
    if not cut:
        return None
    nf = sum(int(r["n_faulty"]) for r in cut)
    s1 = sum(_f(r["s1"]) for r in cut)
    s2 = sum(_f(r["s2"]) for r in cut)
    wall = max(_f(r["wall"]) for r in cut)
    ncores = sum(int(r["n_cores"]) for r in cut)
    nt = sum(int(r["n_trials"]) for r in cut)
    return dict(n_trials=nt, n_faulty=nf, s1=s1, s2=s2, wall=wall, n_cores=ncores)


def load_grid(path):
    """{(code_type, p): (ew, vw, ew_se)} from direct_mc_ew.csv (independent truth grid).

    ``ew_se`` is the SE of the grid's own mean; it is 0 when the column is absent
    or unparsable, which makes the reference behave as exact (the old behaviour).
    """
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                ew_se = _f(r.get("ew_se"))
                if not np.isfinite(ew_se):
                    ew_se = 0.0
                out[(r["code_type"], float(r["p"]))] = (float(r["ew_mean"]),
                                                        float(r["vw"]), float(ew_se))
            except (KeyError, ValueError, TypeError):
                continue
    return out


def load_stratified(path):
    """{(code_type, d, p): dict(ew, vw, ew_se)} from subset_sampling.csv.

    The weight-stratified ("subset sampling") reference of src/subset_sampling.py:
    P_L and E[w|F] synthesized from per-weight failure rates f_w, MCMC-independent
    and available where direct MC collects too few faulty draws.  ``d`` is kept as
    the raw string of the CSV ("5", "0", "3x11"), matching STRAT_CODE.  Rows flagged
    infeasible or carrying a non-finite E[w|F] are skipped.  NOTE: n_trials/n_faulty
    in this file are per-stratum sample counts, NOT direct-MC trials, and are
    deliberately not read here.
    """
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                if str(r.get("feasible", "True")).strip().lower() in ("false", "0"):
                    continue
                ew = float(r["ew_mean"])
                p0 = float(r["p"])
            except (KeyError, ValueError, TypeError):
                continue
            if not np.isfinite(ew):
                continue
            ew_se = _f(r.get("ew_se"))
            vw = _f(r.get("vw"))
            out[(r["code_type"], str(r["d"]).strip(), p0)] = dict(
                ew=ew, vw=vw, ew_se=ew_se if np.isfinite(ew_se) else 0.0)
    return out


def _strat_get(strat, code, p):
    """Stratified entry for a dmc_speedup code spec at `p`, or None.

    p is matched with a tolerance because both sides round-trip through a CSV.
    """
    key = STRAT_CODE.get(code)
    if not key or not strat:
        return None
    ct, d = key
    for (ct0, d0, p0), v in strat.items():
        if ct0 == ct and d0 == d and abs(p0 - p) <= 1e-9 * max(abs(p), abs(p0)):
            return v
    return None


def truth_for(rows, code, p, dmc_grid, strat=None):
    """(ew, vw, source, truth_se) for one (code, p) cell.

    Tier order: weight-stratified reference (subset_sampling.csv) -> direct-MC
    cutoff (>= MIN_FAULTY_FOR_TRUTH faulty in this run) -> direct-MC truth grid ->
    the Z+H longest chain itself.  The first three are MCMC-independent; the last
    is not, and only tests convergence.  ``truth_se`` is the standard error of the
    reference value, folded into the accuracy test by ``wall_to_precision``
    (stratified and grid: their ew_se column; cutoff: sqrt(vw/n_faulty); zh: 0,
    the pooled chain mean has no meaningful independent error here).

    The stratified estimate outranks the cutoff because it is the sharper of two
    references that agree: at p=1e-3 its ew_se is more than an order of magnitude
    below the cutoff's per-run SE (bb_l9m6: 0.07% vs 1.08% of E[w|F]) while the two
    agree to |z| < 1.7 on all six codes where both exist, so it stops a noisy
    reference from spending the 2% accuracy budget.  It is also the only tier
    extensible to the p columns where direct MC collects too few faulty draws to
    certify anything: it now covers p=1e-3 and p=1e-5 for all eight TTA codes, so
    the low-p end no longer falls back on the chain itself and the cells it wins at
    1e-3 displace a healthy cutoff rather than nothing.  p=1e-1 is still served by
    the cutoff, so a row does mix tiers across p, but both tiers there are
    MCMC-independent.
    """
    st = _strat_get(strat, code, p)
    if st is not None:
        # the real V[w|F] comes along, so N* in dmc_time_lower_bound is a genuine
        # sample-size target rather than the single-faulty floor of the zh tier
        return st["ew"], st["vw"], "stratified", float(st["ew_se"])
    cut = _cutoff_stats(rows, code, p)
    if cut and cut["n_faulty"] >= MIN_FAULTY_FOR_TRUTH:
        f = cut["n_faulty"]
        ew = cut["s1"] / f
        vw = max(cut["s2"] / f - ew ** 2, 0.0) * (f / (f - 1))
        return ew, vw, "directmc_cutoff", float(np.sqrt(vw / f))
    gk = (GRID_CODE.get(code), p)
    if gk in dmc_grid:
        ew, vw, ew_se = dmc_grid[gk]
        return ew, vw, "directmc_grid", float(ew_se)
    # Z+H longest-length seed-pooled estimate. We have the per-seed mean E[w] but NOT a
    # reliable V[w|F] here (cross-seed scatter is the estimator variance ~V/ESS, not V[w|F]),
    # so return vw=NaN -> N* floors to the single-faulty bound (see dmc_time_lower_bound).
    adv = [r for r in rows if r["phase"] == "advantage" and r["code"] == code
           and abs(_f(r["p"]) - p) < 1e-12 and r["method"] == "Z+H"]
    if adv:
        lmax = max(int(r["length"]) for r in adv)
        ews = np.array([_f(r["ew"]) for r in adv if int(r["length"]) == lmax])
        ews = ews[np.isfinite(ews)]
        if len(ews):
            return float(ews.mean()), float("nan"), "zh", 0.0
    return float("nan"), float("nan"), "none", 0.0


def _unc_step(se):
    """Last displayed digit of a value with SE `se` under the PDG rule: the SE keeps
    two significant figures when its leading three digits are 100-354, one for
    355-949, and 950-999 round up to 1000 and keep two (0.098 -> 0.10)."""
    e = int(np.floor(np.log10(se)))
    lead = int(round(se / 10.0 ** (e - 2)))
    if lead >= 1000:
        e, lead = e + 1, 100
    return e - 1 if lead <= 354 else e


def _val_unc(su, se):
    """Compact value(uncertainty) under the PDG rule of `_unc_step`, the value
    rounded to the same decimal place (631.8, 82.6 -> '630(80)'; 3.62, 0.16 ->
    '3.62(16)')."""
    e = _unc_step(se)
    if e >= 0:
        step = 10.0 ** e
        return f"{round(su / step) * step:.0f}({round(se / step) * step:.0f})"
    d = -e
    return f"{su:.{d}f}({round(se * 10 ** d):d})"


def _floor2(x):
    """`x` rounded down to two significant figures, for a lower bound."""
    step = 10.0 ** (int(np.floor(np.log10(x))) - 1)
    return np.floor(x / step + 1e-9) * step


def _fmt(su, lb, mark="", se=None):
    """Two significant figures everywhere; scientific notation from 10^3 up.
    `mark` is a string of footnote letters rendered as a superscript.
    `se` (one-sigma run-to-run scatter) is appended as a parenthetical
    last-digit uncertainty on measured cells; floors keep the bare \\geq value,
    whose scatter is not meaningful and which rounds down."""
    if not np.isfinite(su):
        return "--"
    if lb:
        su = _floor2(su)
    pre = r"\geq " if lb else ""
    # {}^{...}: the empty group avoids a LaTeX double-superscript after 10^{e}
    post = rf"{{}}^{{\mathrm{{{mark}}}}}" if mark else ""
    show_se = se is not None and np.isfinite(se) and se > 0 and not lb
    if su >= 1000:
        e = int(np.floor(np.log10(su)))
        m = su / 10 ** e
        if round(m, 1) >= 10:  # 9.96 -> 10.0: carry into the exponent
            m, e = m / 10, e + 1
        if show_se and se / 10 ** e >= 0.05:
            return f"${_val_unc(m, se / 10 ** e)}\\times10^{{{e}}}{post}$"
        return f"${pre}{m:.1f}\\times10^{{{e}}}{post}$"
    if show_se:
        return f"${pre}{_val_unc(su, se)}{post}$"
    if su >= 100:
        body = f"{round(su, -1):.0f}"
    elif su >= 10:
        body = f"{su:.0f}"
    else:
        body = f"{su:.1f}"
    return f"${pre}{body}{post}$"


# budget-saturation footnotes: (a) the dMC cutoff run saw no faulty draw, so its
# faulty rate is only bounded above (<= 1/C) and it is charged the budget for the
# N* = Var[w|F]/(E[w|F]^2 releps^2) faulty draws the target needs, with Var and mean
# taken from the stratified reference and N* floored to 1; the entry is therefore a
# floor on the ratio.  (b) OBF did not certify 2% within its length budget; (c) Z+H
# itself hit its length cap, so the whole cell is a ratio of budgets.  When the dMC
# cutoff run drew some faulty samples but fewer than N*, it is charged the budget
# C it actually spent (a strict floor).  It assumes N* is finite: with V[w|F]
# unavailable the time is C/n_faulty, which no current cell reaches.  Since
# 2026-10-15 the lower-bound cases (a), (b), (d) print no letter, only the >=
# sign (the captions say what >= means); only (c) keeps its letter.  Only marks
# actually used are explained.
_MARK_TEXT = {
    "c": r"$(\mathrm{c})$ \zh{} itself reached the largest checkpoint, so the entry is "
         r"neither an upper nor a lower bound",
}


# prose for each truth tier, used to name the reference in the caption
_SOURCE_TEXT = {
    "stratified": r"the weight-stratified reference of \Cref{sec:setup-stratified}",
    "directmc_cutoff": r"the direct-MC cutoff run",
    "directmc_grid": r"the direct-MC truth grid",
    "zh": r"the longest \zh{} chain itself",
}


def _p_tex(p, with_p=True):
    """``$p=10^{-3}$`` (with_p) or the bare ``$10^{-3}$`` for a continued list."""
    e = int(round(np.log10(p)))
    lhs = "p=" if with_p else ""
    if abs(p - 10 ** e) < 1e-12 * p:
        return rf"${lhs}10^{{{e}}}$"
    return rf"${lhs}{p:g}$"


def _reference_sentence(table):
    r"""Caption sentence naming the reference behind $\mathbb{E}_{\mathrm{ref}}$.

    Read off the table's own ``truth_source`` column and grouped by p, so it cannot
    drift from the tiers actually used; a column whose codes disagree is reported
    as the list of tiers that occur in it."""
    groups = []          # [(prose, [p, ...])], consecutive equal prose merged
    for p in P_ORDER:
        # aggregate_runs joins disagreeing per-run tiers with "+"; split them so
        # a Python identifier never reaches the caption
        srcs = sorted({s for t in table if t["p"] == p
                       for s in str(t.get("truth_source") or "").split("+") if s})
        if not srcs:
            continue
        text = " and ".join(_SOURCE_TEXT.get(s, s) for s in srcs)
        if groups and groups[-1][0] == text:
            groups[-1][1].append(p)
        else:
            groups.append((text, [p]))
    if not groups:
        return ""
    chunks = []
    for text, ps in groups:
        labs = [_p_tex(ps[0])] + [_p_tex(q, False) for q in ps[1:]]
        if len(labs) > 2:
            joined = ", ".join(labs[:-1]) + " and " + labs[-1]
        else:
            joined = " and ".join(labs)
        chunks.append(f"{text} at {joined}")
    return (r" The reference $\mathbb{E}_{\mathrm{ref}}$ is "
            + ", and ".join(chunks) + ".")


def _cell_marks(t):
    """(dmc_mark, obf_mark) footnote letters for one (code,p) cell pair."""
    zh_capped = not t.get("zh_reached", True)
    # A lower bound is marked by the >= sign alone (the author's call,
    # 2026-10-15): the competitor did not reach the target within its budget,
    # whether direct MC drew no or too few faulty configurations, or \obf{}
    # ran out of checkpoints.  Only the Z+H-capped case (c) keeps a letter,
    # since there the >= sign is dropped and the cell needs an explanation.
    c = "c" if zh_capped else ""
    return c, c


def _fmt_p(p):
    e = int(round(np.log10(p)))
    if abs(p - 10 ** e) < 1e-12 * p:
        return rf"$p=10^{{{e}}}$"
    return rf"$p={p:g}$"


N_BOOT = 20000      # bootstrap resamples of the run set for the speedup SE
BOOT_SEED = 0       # fixed so the table is reproducible


def ratio_of_means(t_ref, t_zh, rng=None, n_boot=N_BOOT):
    """Speedup over repeat runs as the ratio of mean times, mean(t_ref)/mean(t_zh),
    with its standard error from a bootstrap over runs.

    This is the ratio of the total cost of the reference method to the total cost
    of Z+H over the whole repeat set, i.e. the ratio of the entries of the wall-time
    table.  The mean of per-run ratios is NOT used: by Jensen, E[X/Y] >= E[X]/E[Y]
    when the denominator varies between runs (here mainly through the doubling grid
    of checkpoints), and on the paper's data that inflated the speedups by up to
    about 20%, always in Z+H's favour.  Runs where either time is non-finite are
    dropped from both means.  Returns (ratio, se); se is 0 for a single run.
    """
    num = np.asarray(t_ref, dtype=float)
    den = np.asarray(t_zh, dtype=float)
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    if len(num) == 0 or den.mean() <= 0:
        return float("nan"), float("nan")
    ratio = float(num.mean() / den.mean())
    if len(num) < 2:
        return ratio, 0.0
    rng = np.random.default_rng(BOOT_SEED) if rng is None else rng
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    boots = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    return ratio, float(boots.std(ddof=1))


def mean_se(t, rng=None, n_boot=N_BOOT):
    """Bootstrap-over-runs standard error of the mean of `t`, non-finite runs dropped;
    0 for a single run, NaN for none."""
    v = np.asarray(t, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan")
    if len(v) < 2:
        return 0.0
    rng = np.random.default_rng(BOOT_SEED) if rng is None else rng
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    return float(v[idx].mean(axis=1).std(ddof=1))


def combine_runs(tables):
    """Combine per-run tables into one table over independent repeat runs.

    Times (and ew/vw/truth_se/p_L) are averaged over runs; each speedup is the
    ratio of the mean reference time to the mean Z+H time (ratio_of_means), with a
    bootstrap-over-runs standard error in `su_*_se`, and the mean times carry the
    bootstrap standard error of the mean in `t_dmc_se` and `t_zh_se`.  A cell is marked a lower
    bound if it is one in ANY run: each per-run reference time is then either the
    true time or a floor of it, so the mean time, and the ratio, is itself a floor.
    Truth sources are unioned.
    """
    by = {}
    for tab in tables:
        for t in tab:
            by.setdefault((t["code"], t["p"]), []).append(t)
    rng = np.random.default_rng(BOOT_SEED)
    out = []
    for code in ROW_ORDER:
        for p in P_ORDER:
            ts = by.get((code, p))
            if not ts:
                continue
            def m(k):
                v = np.array([t[k] for t in ts], dtype=float)
                return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            t_zh = [t["t_zh"] for t in ts]
            su_dmc, su_dmc_se = ratio_of_means([t["t_dmc"] for t in ts], t_zh, rng)
            su_obf, su_obf_se = ratio_of_means([t["t_obf"] for t in ts], t_zh, rng)
            out.append(dict(
                code=code, p=p, n_runs=len(ts),
                truth_source="+".join(sorted({t["truth_source"] for t in ts})),
                ew=m("ew"), vw=m("vw"), truth_se=m("truth_se"),
                p_L=m("p_L"), n_faulty=sum(t["n_faulty"] for t in ts),
                t_dmc=m("t_dmc"), t_obf=m("t_obf"), t_zh=m("t_zh"),
                t_dmc_se=mean_se([t["t_dmc"] for t in ts]),
                t_zh_se=mean_se(t_zh),   # own rng: su_*_se stay as before
                su_dmc=su_dmc, su_dmc_se=su_dmc_se, dmc_lb=any(t["dmc_lb"] for t in ts),
                dmc_hard=any(t["dmc_hard"] for t in ts),
                su_obf=su_obf, su_obf_se=su_obf_se, obf_lb=any(t["obf_lb"] for t in ts),
                zh_reached=all(t["zh_reached"] for t in ts)))
    return out


def _panel_tabular(table, by, ref_key):
    """One `lccc` tabular (code x P_ORDER) for a single reference column (dMC or OBF).

    `ref_key` selects which per-cell value/lb/se fields to use ("dmc" or "obf").
    """
    lines = [r"\begin{tabular}{lccc}",
             r"\toprule",
             "code & " + " & ".join(_fmt_p(p) for p in P_ORDER) + r" \\",
             r"\midrule"]
    for code in ROW_ORDER:
        if all(by.get((code, p)) is None for p in P_ORDER):
            continue        # code absent from these runs (e.g. a top-up not yet merged)
        cells = []
        for p in P_ORDER:
            t = by.get((code, p))
            if t is None:
                cells.append("--")
            else:
                dmc_m, obf_m = _cell_marks(t)
                # Z+H capped -> the denominator is a floor, so the ratio is not a
                # bound in either direction: drop the >= and let mark (c) speak
                zh_capped = not t.get("zh_reached", True)
                if ref_key == "dmc":
                    cells.append(_fmt(t["su_dmc"], t["dmc_lb"] and not zh_capped, dmc_m,
                                      se=t.get("su_dmc_se", float("nan"))))
                else:
                    cells.append(_fmt(t["su_obf"], t["obf_lb"] and not zh_capped, obf_m,
                                      se=t.get("su_obf_se", float("nan"))))
        lines.append(ROW_LABEL.get(code, code.replace("_", r"\_")) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def render_latex(table, n_runs, n_seeds):
    r"""LaTeX: two separate `table` floats, `tab:dmcspeedup` (vs dMC) and
    `tab:obfspeedup` (vs \obf{}), each one `lccc` tabular with caption below.
    Footnote marks are restricted per-table to the marks that actually occur
    in that table's own cells."""
    by = {(t["code"], t["p"]): t for t in table}
    # The run count and the reference of each column are stated in sec:bench2 (ten
    # runs; direct MC at p=1e-1, weight-stratified otherwise), so the captions keep
    # only the definition, the error bars, and the lower-bound marks (KM, 2026-10-14).
    used_dmc = sorted({ch for t in table for ch in _cell_marks(t)[0]})
    used_obf = sorted({ch for t in table for ch in _cell_marks(t)[1]})
    notes_dmc = (" " + ". ".join(_MARK_TEXT[ch] for ch in used_dmc) + ".") if used_dmc else ""
    notes_obf = (" " + ". ".join(_MARK_TEXT[ch] for ch in used_obf) + ".") if used_obf else ""

    cap_dmc = (r"Time-to-accuracy speedup of \zh{} over direct Monte Carlo, the "
               r"$T_{2\%}$ of direct Monte Carlo divided by that of \zh{}, where "
               r"$T_{2\%}$ (\Cref{eq:tta}) is the computation time from which the "
               r"relative error stays below $2\%$. "
               r"Parentheses give the standard error of \Cref{sec:bench2}, and $\geq$ "
               r"marks a lower bound, where direct Monte Carlo did not reach the $2\%$ "
               r"target within its budget."
               + notes_dmc)
    cap_obf = (r"Same speedups as \Cref{tab:dmcspeedup}, but against the one-bit-flip "
               r"baseline (\obf). "
               r"Parentheses give the standard error of \Cref{sec:bench2}, and $\geq$ "
               r"marks a lower bound, where \obf{} did not reach the $2\%$ target "
               r"within the largest checkpoint."
               + notes_obf)

    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{6pt}"]
    lines += _panel_tabular(table, by, "dmc")
    lines += [r"\caption{" + cap_dmc + r"}", r"\label{tab:dmcspeedup}", r"\end{table*}", ""]
    lines += [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}"]
    lines += _panel_tabular(table, by, "obf")
    lines += [r"\caption{" + cap_obf + r"}", r"\label{tab:obfspeedup}", r"\end{table}"]
    return "\n".join(lines)

def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


_RUN_RE = re.compile(r"dmc_speedup_run(\d+)")


def group_paths_by_run(paths):
    """Group input CSVs by RUN_ID, so a per-code top-up file belongs to its run.

    A single independent run may be spread over several CSVs: the 7-code base file
    ``dmc_speedup_run7.csv`` plus per-code additions written by a later job, e.g.
    ``dmc_speedup_run7_bb108.csv``.  Those share the seed stream of run 7 (the worker
    offsets every seed by 1e6*(run_id-1)) and must be analysed as ONE run, otherwise
    the caption's run count and the per-cell run-to-run scatter are wrong.  Paths whose
    name carries no run id (e.g. the legacy ``dmc_speedup.csv``) each form their own
    group.  Order of first appearance is preserved.
    """
    groups, order = {}, []
    for path in paths:
        m = _RUN_RE.search(os.path.basename(path))
        key = m.group(1) if m else "file:" + path
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(path)
    return [groups[k] for k in order]


def build_table(rows, dmc_grid, strat=None):
    table = []
    for code in ROW_ORDER:
        for p in P_ORDER:
            if not any(r["code"] == code and abs(_f(r["p"]) - p) < 1e-12 for r in rows):
                continue
            ew, vw, tsrc, tse = truth_for(rows, code, p, dmc_grid, strat)
            cut = _cutoff_stats(rows, code, p)
            if not (np.isfinite(ew) and cut):
                continue
            t_dmc, dmc_hard, dmc_conv = dmc_time_lower_bound(cut["n_faulty"], cut["wall"],
                                                             cut["n_cores"], ew, vw)
            t_zh, zh_ok = wall_to_precision_strict(rows, code, p, "Z+H", ew, truth_se=tse)
            t_obf, obf_ok = wall_to_precision_strict(rows, code, p, "OBF", ew, truth_se=tse)
            # dMC entry is a measurement where the cutoff run itself certified 2%
            su_dmc, dmc_lb = speedup(t_dmc, t_zh, ref_reached=dmc_conv)
            su_obf, obf_lb = speedup(t_obf, t_zh, ref_reached=obf_ok)
            table.append(dict(code=code, p=p, truth_source=tsrc, ew=ew, vw=vw, truth_se=tse,
                              p_L=(cut["n_faulty"] / cut["n_trials"] if cut["n_trials"] else float("nan")),
                              n_faulty=cut["n_faulty"], t_dmc=t_dmc, t_obf=t_obf, t_zh=t_zh,
                              su_dmc=su_dmc, dmc_lb=dmc_lb, dmc_hard=dmc_hard,
                              su_obf=su_obf, obf_lb=obf_lb, zh_reached=zh_ok))
    return table


def n_seeds_of(rows):
    """Chains per (code, p, kernel) in the advantage phase, i.e. S in the caption."""
    per_cell = {}
    for r in rows:
        if r["phase"] == "advantage":
            per_cell.setdefault((r["code"], r["p"], r["method"]), set()).add(r["seed"])
    counts = {len(v) for v in per_cell.values()}
    if len(counts) != 1:
        raise ValueError(f"seed count differs across cells: {sorted(counts)}")
    return counts.pop()


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else [CSV]
    grid = load_grid(GRID)
    strat = load_stratified(STRAT)
    groups = group_paths_by_run(paths)
    runs = [[r for path in g for r in load(path)] for g in groups]
    seeds = {n_seeds_of(rows) for rows in runs}
    if len(seeds) != 1:
        raise ValueError(f"seed count differs across runs: {sorted(seeds)}")
    n_seeds = seeds.pop()
    tables = [build_table(rows, grid, strat) for rows in runs]
    combined = combine_runs(tables)
    # A code merged for only some of the runs (e.g. half the bb108 top-up files present)
    # still gets a correct per-cell mean and SE, but the caption's single run count would
    # not describe it; say so rather than publishing a silently mixed table.
    counts = {t["n_runs"] for t in combined}
    if len(counts) > 1:
        short = sorted({t["code"] for t in combined if t["n_runs"] != len(groups)})
        print(f"WARNING: per-cell run counts differ {sorted(counts)}; the caption will say "
              f"{len(groups)}. Codes present in fewer runs: {short}", file=sys.stderr)
    cols = ["code", "p", "n_runs", "truth_source", "ew", "vw", "p_L", "n_faulty",
            "t_dmc", "t_obf", "t_zh", "su_dmc", "su_dmc_se", "dmc_lb", "dmc_hard",
            "su_obf", "su_obf_se", "obf_lb", "zh_reached",
            "truth_se", "t_dmc_se", "t_zh_se"]     # appended last: keeps the existing column order stable
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in combined:
            w.writerow({c: t[c] for c in cols})
    print(f"wrote {OUT} ({len(groups)} run(s) from {len(paths)} file(s), S={n_seeds} seeds)")
    tex = render_latex(combined, len(groups), n_seeds)
    os.makedirs(os.path.dirname(TEX), exist_ok=True)
    with open(TEX, "w", newline="\n") as f:
        f.write("% generated by compare/analyze_dmc_speedup.py -- do not edit by hand\n")
        f.write(tex + "\n")
    print(f"wrote {TEX}")
    print(tex)


if __name__ == "__main__":
    main()
