# compare/analyze_kernel_param_tta.py
"""Sustained T_2% of every (code, p, p_cluster, m) arm of kernel_param_tta, averaged over
runs with a bootstrap SE, against the weight-stratified reference.

Outputs: five codes (surface:5, hgp_hamming, rect:3x11, bb144, bb_l9m6):
         cluster_results/kernel_param_tta_table.csv,
         docs/figures/fig_pcluster_tta.png   (T_2% vs p_cluster, m=2),
         paper/tables/clusterk_tta_table.tex (T_2% at m=2 and T(m)/T(m=2) for m=3,4,6 at p_cluster=0.3, p=TABLE_P,
                                              rows whose m=2 arm reached the target only).
Usage:   python compare/analyze_kernel_param_tta.py [--check-dmc] [cluster_results/kernel_param_tta_run*.csv]
run{r}.csv, run{r}_ext.csv and run{r}_add.csv are concatenated into one run.
--check-dmc compares the (0.3, 2) arm (every length present in both files) with the Z+H advantage rows of
cluster_results/dmc_speedup_run{r}.csv (+ dmc_speedup_run{r}_bb108.csv, where bb_l9m6 lives,
and dmc_speedup_run{r}_topo.csv for surface/toric)
of the same run id r (identical chains); it exits non-zero, before writing any output, if
no row matched or any row mismatched.
"""
import csv
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_dmc_speedup import (wall_to_precision, wall_to_precision_strict,  # noqa: E402,F401
                                 load_stratified, _strat_get, _fmt, _val_unc, _floor2, BOOT_SEED, N_BOOT, STRAT_CODE)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CODES = ["surface:5", "hgp_hamming", "rect:3x11", "bb144", "bb_l9m6"]
PS = [1e-3, 1e-4, 1e-5]
PCS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
KS = [2, 3, 4, 6]
LABEL = {"surface:5": r"surface $d=5$", "hgp_hamming": r"HGP(Hamming)",
         "rect:3x11": r"$\mathrm{AT}(3,11)$", "bb144": r"BB $[[144,12,12]]$",
         "bb_l9m6": r"BB $[[108,8,10]]$"}
NCOL = 3
CHECK_LMAX = 512000


def _near(a, b):
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b))


def _arm_rows(rows, pc, k):
    """Rows of one arm re-labelled so analyze_dmc_speedup.wall_to_precision can read them."""
    out = []
    for r in rows:
        if abs(float(r["p_cluster"]) - pc) < 1e-12 and int(r["cluster_k"]) == k:
            out.append(dict(r, phase="advantage", method="arm"))
    return out


def arm_times(rows_by_run, code, p, pc, k, truth, truth_se):
    """(T mean over runs, bootstrap SE, every run reached, number of runs with a finite T).

    "Reached" uses the rule of ``analyze_dmc_speedup.wall_to_precision_strict``, the same as
    the main-text speedup table: the 2% target must be sustained from a checkpoint before
    the last one of the arm.
    """
    ts, reached = [], []
    for rows in rows_by_run:
        t, ok = wall_to_precision_strict(_arm_rows(rows, pc, k), code, p, "arm", truth,
                                         truth_se=truth_se)
        if np.isfinite(t):
            ts.append(t)
            reached.append(ok)
    if not ts:
        return float("nan"), float("nan"), False, 0
    ts = np.asarray(ts)
    if len(ts) < 2:
        return float(ts.mean()), 0.0, all(reached), 1
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(ts), size=(N_BOOT, len(ts)))
    return float(ts.mean()), float(ts[idx].mean(axis=1).std(ddof=1)), all(reached), len(ts)


def build_table_from_rows(rows_by_run, strat):
    table = []
    codes = sorted({r["code"] for rows in rows_by_run for r in rows},
                   key=lambda c: CODES.index(c) if c in CODES else 99)
    for code in codes:
        for p in sorted({float(r["p"]) for rows in rows_by_run for r in rows if r["code"] == code},
                        reverse=True):
            ref = _strat_get(strat, code, p)
            if ref is None:
                continue
            arms = sorted({(float(r["p_cluster"]), int(r["cluster_k"])) for rows in rows_by_run
                           for r in rows if r["code"] == code and _near(float(r["p"]), p)})
            for pc, k in arms:
                t, se, ok, n_runs = arm_times(rows_by_run, code, p, pc, k, ref["ew"], ref["ew_se"])
                table.append(dict(code=code, p=p, p_cluster=pc, cluster_k=k, n_runs=n_runs,
                                  t_mean=t, t_se=se, reached_all=ok, truth=ref["ew"],
                                  truth_se=ref["ew_se"]))
    return table


def _fmt_ratio(r, se, lb):
    """T(m)/T(m=2): value(uncertainty) on measured cells, '\\geq' with two significant
    figures, rounded down, when the m arm is a lower bound (its scatter is not meaningful)."""
    if not np.isfinite(r):
        return "--"
    if lb:
        e = int(np.floor(np.log10(r)))
        v = _floor2(r)
        body = f"{v:.0f}" if e >= 1 else f"{v:.{1 - e}f}"
        return rf"$\geq {body}$"
    if se is not None and np.isfinite(se) and se > 0:
        return f"${_val_unc(r, se)}$"
    return f"${r:.2f}$"


TABLE_P = 1e-4      # the error rate tab:clusterk shows


def render_clusterk_tex(table, p=TABLE_P):
    """T_2% at m=2 (core-seconds) and the relative time T(m)/T(m=2) for the other m, one
    row per code at p_cluster=0.3 and error rate p. A code is emitted only when its m=2
    arm reached the target, since a ratio of two lower bounds says nothing."""
    rows = [t for t in table if abs(t["p_cluster"] - 0.3) < 1e-12 and _near(t["p"], p)]
    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r"code & $T_{2\%}(m=2)$ & " + " & ".join(f"$m={k}$" for k in KS if k != 2) + r" \\",
           r"\midrule"]
    for code in CODES:
        arm = {t["cluster_k"]: t for t in rows if t["code"] == code}
        ref = arm.get(2)
        if ref is None or not ref["reached_all"]:
            continue
        cells = [_fmt(ref["t_mean"], False, se=ref["t_se"])]
        for k in KS:
            if k == 2:
                continue
            t = arm.get(k)
            if t is None:
                cells.append("--")
                continue
            r = t["t_mean"] / ref["t_mean"]
            se = r * np.hypot(t["t_se"] / t["t_mean"], ref["t_se"] / ref["t_mean"])
            cells.append(_fmt_ratio(r, se, not t["reached_all"]))
        out.append(f"{LABEL[code]} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def _select_arm_points(table, code, p, k):
    """Sorted (p_cluster, t_mean, t_se, reached_all) of one (code, p, m) curve; p is matched
    with a relative tolerance (table p values come from a CSV round trip)."""
    return sorted((t["p_cluster"], t["t_mean"], t["t_se"], t["reached_all"]) for t in table
                  if t["code"] == code and _near(t["p"], p) and t["cluster_k"] == k)


CELL_FS = 7        # cell-label font size
REL_MAX = 30.0      # top of the colour scale; a value above it is drawn at the top colour


def _rel_times(table, code, p):
    """Per p_cluster of one (code, p, m=2) curve: (p_cluster, ratio, ratio_se, reached) with
    ratio = T_2% / smallest reached T_2% of the curve (all values if none reached) and
    ratio_se the SEs of both times in quadrature (0 for the reference cell itself); [] if no arm."""
    pts = _select_arm_points(table, code, p, 2)
    if not pts:
        return []
    xs, ys, es, ok = zip(*pts)
    ref = [(y, e) for y, e, o in zip(ys, es, ok) if o] or list(zip(ys, es))
    t0, e0 = min(ref)
    out = []
    for x, y, e, o in zip(xs, ys, es, ok):
        r = y / t0
        se = 0.0 if y == t0 else r * float(np.hypot(e / y, e0 / t0))
        out.append((x, r, se, bool(o)))
    return out


def _fmt_rel(r, reached, se=0.0):
    """Cell label: value(SE) under the PDG rule for a measured ratio with an SE, else
    two significant figures, with a leading >= on a lower bound."""
    if reached and se is not None and np.isfinite(se) and se > 0:
        return _val_unc(r, se)
    if reached and r == 1.0:
        return "1"                      # the reference cell of its row
    txt = f"{r:.1f}" if r < 10 else f"{r:.0f}"
    return txt if reached else "\u2265" + txt


FIG_ROWS = [(c, 1e-4) for c in CODES]


def plot_pcluster(table, path, rows=FIG_ROWS):
    """Write the p_cluster figure (m=2) as one small grid: a row per (code, p) in `rows`
    (default FIG_ROWS: every code at p=1e-4), a column per p_cluster in
    PCS, each cell coloured and labelled by T_2% relative to the smallest reached T_2% of
    its row. A lower bound (a run did not reach the target) is hatched and labelled with >=.
    The default column 0.3 is outlined. Returns (fig, ax) with fig already closed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    ncol = len(PCS)
    grid = np.full((len(rows), ncol), np.nan)
    reached = np.ones((len(rows), ncol), dtype=bool)
    sematrix = np.zeros((len(rows), ncol))
    for i, (c, p) in enumerate(rows):
        for x, r, se, ok in _rel_times(table, c, p):
            j = PCS.index(x)
            grid[i, j] = r
            reached[i, j] = ok
            sematrix[i, j] = se
    fig, ax = plt.subplots(figsize=(6.0, 0.34 * len(rows) + 0.55))   # full width: value(SE) labels
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("0.92")
    ax.imshow(np.clip(grid, 1.0, REL_MAX), cmap=cmap, norm=LogNorm(1.0, REL_MAX), aspect="auto")
    for i in range(len(rows)):
        for j in range(ncol):
            if np.isnan(grid[i, j]):
                continue
            r = grid[i, j]
            if not reached[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="//",
                                       edgecolor="0.35", lw=0))
            dark = r > 4.0 and reached[i, j]
            ax.text(j, i, _fmt_rel(r, reached[i, j], sematrix[i, j]), ha="center", va="center", fontsize=CELL_FS,
                    color="white" if dark else "black",
                    bbox=None if reached[i, j] else dict(facecolor="white", alpha=0.85, lw=0, pad=1.2))
    j3 = PCS.index(0.3)
    ax.add_patch(Rectangle((j3 - 0.5, -0.5), 1, len(rows), fill=False, edgecolor="black",
                           lw=1.5))
    for k in range(1, len(rows)):
        if rows[k][0] != rows[k - 1][0]:
            ax.axhline(k - 0.5, color="black", lw=0.6)
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([f"{x:g}" for x in PCS], fontsize=7)
    ax.set_xlabel(r"$p_{\mathrm{cluster}}$", fontsize=8)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    one_p = len({p for _, p in rows}) == 1          # p goes to the caption, not the labels
    ax.set_yticklabels([LABEL[c] if one_p else rf"{LABEL[c]}, $p=10^{{{int(round(np.log10(p)))}}}$"
                        for c, p in rows], fontsize=7)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return fig, ax


def cross_check_dmc(run_rows, dmc_rows):
    """Compare the (0.3, 2) arm of one kernel_param_tta run with the Z+H advantage rows of the
    dmc_speedup run of the same id, on every key present in both (any length). Returns
    (n_matched, mismatches, n_unmatched), n_unmatched = arm rows with L <= CHECK_LMAX (the dmc
    base cap) and no dmc counterpart."""
    dmc = {}
    for r in dmc_rows:
        if r.get("phase") == "advantage" and r.get("method") == "Z+H":
            dmc[(r["code"], round(float(r["p"]), 12), int(r["seed"]), int(r["length"]))] = float(r["ew"])
    n, bad, unmatched = 0, [], 0
    for r in _arm_rows(run_rows, 0.3, 2):
        L = int(r["length"])
        key = (r["code"], round(float(r["p"]), 12), int(r["seed"]), L)
        if key not in dmc:
            if L <= CHECK_LMAX:
                unmatched += 1
            continue
        n += 1
        ew = float(r["ew"])
        if not np.isclose(ew, dmc[key], rtol=1e-12, atol=0.0):
            bad.append((key, ew, dmc[key]))
    return n, bad, unmatched


def _run_id(path):
    m = re.search(r"run(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def group_paths_by_run(paths):
    """{run_id: [paths]} with run{r}.csv, run{r}_ext.csv and run{r}_add.csv in one group, sorted by run id;
    a file without a run id is its own group keyed by its filename."""
    groups = {}
    for p in paths:
        m = re.search(r"kernel_param_tta_run(\d+)(_ext|_add)?\.csv$", os.path.basename(p))
        key = int(m.group(1)) if m else os.path.basename(p)
        groups.setdefault(key, []).append(p)
    ints = sorted(k for k in groups if isinstance(k, int))
    strs = sorted(k for k in groups if isinstance(k, str))
    return {k: groups[k] for k in ints + strs}


def load_rows_by_run(paths):
    """{run_id: rows} with the rows of every file of a group concatenated."""
    return {rid: [r for p in ps for r in _read(p)] for rid, ps in group_paths_by_run(paths).items()}


def load_dmc_rows(rid, results_dir):
    """Rows of dmc_speedup_run{rid}.csv merged with dmc_speedup_run{rid}_bb108.csv (bb_l9m6)
    and dmc_speedup_run{rid}_topo.csv (surface/toric), whichever exist; None if none does."""
    if rid is None:
        return None
    paths = [os.path.join(results_dir, f"dmc_speedup_run{rid}{suf}.csv")
             for suf in ("", "_bb108", "_topo")]
    found = [p for p in paths if os.path.exists(p)]
    if not found:
        return None
    return [r for p in found for r in _read(p)]


def main(argv=None):
    argv = list(argv or [])
    check = "--check-dmc" in argv
    paths = [a for a in argv if a != "--check-dmc"] or sorted(
        glob.glob(os.path.join(ROOT, "cluster_results", "kernel_param_tta_run*.csv")))
    run_rows = load_rows_by_run(paths)
    rows_by_run = list(run_rows.values())
    if check:
        total, nbad, nunm = 0, 0, 0
        for rid, rows in run_rows.items():
            rid = rid if isinstance(rid, int) else None
            dmc_rows = load_dmc_rows(rid, os.path.join(ROOT, "cluster_results"))
            if dmc_rows is None:
                print(f"check-dmc: run {rid}: no dmc_speedup_run{rid}[_bb108].csv, skipped")
                continue
            n, bad, unm = cross_check_dmc(rows, dmc_rows)
            total += n
            nbad += len(bad)
            nunm += unm
            print(f"check-dmc: run {rid}: {n} rows matched, {len(bad)} mismatches, "
                  f"{unm} eligible rows without a dmc counterpart")
            for key, a, b in bad:
                print(f"  MISMATCH {key}: kernel_param_tta ew={a!r} dmc_speedup ew={b!r}")
        print(f"check-dmc: total {total} matched, {nbad} mismatches, {nunm} unmatched")
        if total == 0 or nbad:
            print("check-dmc: FAILED (no row matched, or a mismatch); no output written")
            sys.exit(1)
    strat = load_stratified(os.path.join(ROOT, "cluster_results", "subset_sampling.csv"))
    table = build_table_from_rows(rows_by_run, strat)
    if not table:
        print("no arms in table; nothing written")
        return
    outs = [os.path.join(ROOT, "cluster_results", "kernel_param_tta_table.csv"),
            os.path.join(ROOT, "docs", "figures", "fig_pcluster_tta.png"),
            os.path.join(ROOT, "paper", "tables", "clusterk_tta_table.tex")]
    for o in outs:
        os.makedirs(os.path.dirname(o), exist_ok=True)
    with open(outs[0], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    plot_pcluster(table, outs[1])
    with open(outs[2], "w", encoding="utf-8", newline="\n") as f:
        f.write(render_clusterk_tex(table))
    print(f"{len(table)} arms from {len(rows_by_run)} runs ({len(paths)} files)")


if __name__ == "__main__":
    main(sys.argv[1:])
