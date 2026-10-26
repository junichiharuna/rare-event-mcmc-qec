"""Introduction figure for the paper (fig:intro).

Reads the numbers CSV written by compare/analyze_dmc_speedup.py
(cluster_results/dmc_speedup_table.csv) and writes
docs/figures/fig_intro_tta.png: the time-to-accuracy speedup of Z-OBF+CE over
direct Monte Carlo and over the one-bit-flip kernel OBF (the entries of
tab:dmcspeedup and tab:obfspeedup) at the lowest physical error rate
p = 1e-5 on all twelve codes, codes in rows separated into families by thin
rules. A lower bound (the slower method did not reach the target within its
budget) is drawn as an open marker with a faint line to the right edge.

Usage: python compare/plot_intro_tta.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from analyze_dmc_speedup import ROW_ORDER  # noqa: E402
from plot_tta_walltime import TITLE, GROUPS, C_DMC, C_OBF  # noqa: E402

IN = os.path.join(os.path.dirname(__file__), "..", "cluster_results", "dmc_speedup_table.csv")
OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures", "fig_intro_tta.png")

P = 1e-5
XMIN, XMAX = 5e-2, 3e5


def _b(s):
    return str(s).strip().lower() in ("true", "1")


def _point(ax, x, se, y, color, marker, lower_bound, zorder):
    if lower_bound:
        ax.plot([x], [y], marker, color=color, ms=4.5, mfc="white", zorder=zorder)
        ax.plot([x, XMAX], [y, y], color=color, lw=1.4, alpha=0.3,
                solid_capstyle="butt", zorder=1)
    else:
        ax.errorbar([x], [y], xerr=[[min(se, 0.999 * x)], [se]], fmt=marker, color=color,
                    ms=4.5, elinewidth=0.8, capsize=0, zorder=zorder)


def main():
    with open(IN, newline="") as f:
        rows = {r["code"]: r for r in csv.DictReader(f) if abs(float(r["p"]) - P) < 1e-12}
    missing = [c for c in ROW_ORDER if c not in rows]
    if missing:
        raise ValueError(f"no p={P} row for {missing}")
    if not all(_b(rows[c]["zh_reached"]) for c in ROW_ORDER):
        raise ValueError("a Z-OBF+CE cell did not reach the target; the speedups would be upper bounds")

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7.5})
    n = len(ROW_ORDER)
    ypos = dict(zip(ROW_ORDER, reversed(range(n))))
    fig, ax = plt.subplots(figsize=(3.4, 3.1))

    ax.axvline(1.0, color="0.5", lw=0.7, ls="--", zorder=0)
    for code in ROW_ORDER:
        r, y = rows[code], ypos[code]
        _point(ax, float(r["su_dmc"]), float(r["su_dmc_se"] or 0), y, C_DMC, "s",
               _b(r["dmc_lb"]), 3)
        _point(ax, float(r["su_obf"]), float(r["su_obf_se"] or 0), y, C_OBF, "^",
               _b(r["obf_lb"]), 4)
    for _, codes in GROUPS:
        bottom = min(ypos[code] for code in codes)
        if bottom > 0:
            ax.axhline(bottom - 0.5, color="0.75", lw=0.5)

    ax.set_xscale("log")
    ax.set_xlim(XMIN, XMAX)
    ax.set_xticks([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    ax.set_xlabel(r"speedup of Z-OBF+CE at $p=10^{-5}$")
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([ypos[code] for code in ROW_ORDER])
    ax.set_yticklabels([TITLE[code] for code in ROW_ORDER])
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", which="major", alpha=0.3, lw=0.5)
    ax.plot([], [], "s", color=C_DMC, ms=4.5, ls="none", label="over direct MC")
    ax.plot([], [], "^", color=C_OBF, ms=4.5, ls="none", label="over OBF")
    ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, handletextpad=0.4, borderaxespad=0.0, columnspacing=1.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, dpi=300)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
