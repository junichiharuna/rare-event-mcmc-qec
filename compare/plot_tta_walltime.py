"""Time-to-accuracy figure for the paper (fig:ttawalltime).

Reads the numbers CSV written by compare/analyze_dmc_speedup.py
(cluster_results/dmc_speedup_table.csv) and writes
docs/figures/fig_tta_walltime.png: a two-series dot chart of T_2% in
core-seconds for Z-OBF+CE and direct Monte Carlo. Each physical error rate has
one panel, with codes in rows separated into families by thin rules. Direct-MC
lower bounds (dmc_lb, including partially sampled cells) use open squares with a
faint line extending to the right edge of the panel. Measured times carry horizontal
bars of one bootstrap-over-runs standard error (t_zh_se, t_dmc_se). OBF times are not drawn.

Usage: python compare/plot_tta_walltime.py
"""
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from analyze_dmc_speedup import P_ORDER, ROW_ORDER  # noqa: E402

IN = os.path.join(os.path.dirname(__file__), "..", "cluster_results", "dmc_speedup_table.csv")
OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures", "fig_tta_walltime.png")

# mathtext versions of analyze_dmc_speedup.ROW_LABEL (which uses \text, not in mathtext)
TITLE = {
    "surface:3": r"surface $d=3$",
    "surface:5": r"surface $d=5$",
    "surface:7": r"surface $d=7$",
    "toric:5": r"toric $d=5$",
    "toric:7": r"toric $d=7$",
    "concat_steane:2": r"concat. Steane ($\ell=2$)",
    "hgp_hamming": r"HGP(Hamming)",
    "rect:3x11": r"AT$(3,11)$",
    "rect:3x15": r"AT$(3,15)$",
    "bb72": r"BB $[[72,12,6]]$",
    "bb144": r"BB $[[144,12,12]]$",
    "bb_l9m6": r"BB $[[108,8,10]]$",
}
C_ZH, C_DMC, C_OBF = "C0", "C3", "0.45"
XMIN, XMAX = 3e-3, 3e7
GROUPS = [
    ("zero spread", ["surface:3", "surface:5", "surface:7", "toric:5", "toric:7",
                     "concat_steane:2"]),
    ("spread 1", ["hgp_hamming"]),
    ("asymmetric\ntoric", ["rect:3x11", "rect:3x15"]),
    ("BB", ["bb72", "bb144", "bb_l9m6"]),
]


def _b(s):
    return str(s).strip().lower() in ("true", "1")


def _point(ax, x, se, y, color, marker, zorder):
    """A measured time with its one-SE horizontal error bar (clipped above zero on the log axis)."""
    ax.errorbar([x], [y], xerr=[[min(se, 0.999 * x)], [se]], fmt=marker, color=color,
                ms=4.5, elinewidth=0.8, capsize=0, zorder=zorder)


def main():
    with open(IN, newline="") as f:
        rows = list(csv.DictReader(f))
    by = {(r["code"], float(r["p"])): r for r in rows}
    if not all(_b(r["zh_reached"]) for r in rows):
        raise ValueError("a Z+H cell did not reach the target; add a floor mark before publishing")

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7.5})
    n = len(ROW_ORDER)
    ypos = dict(zip(ROW_ORDER, reversed(range(n))))
    groups = [(label, codes) for label, codes in GROUPS
              if all(code in ypos for code in codes)]
    fig, axes = plt.subplots(1, len(P_ORDER), figsize=(7.0, 3.2),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, p in zip(axes, P_ORDER):
        for code in ROW_ORDER:
            r, y = by[(code, p)], ypos[code]
            t_dmc, t_zh = float(r["t_dmc"]), float(r["t_zh"])
            floor = _b(r["dmc_lb"])
            if floor:
                ax.plot([t_dmc], [y], "s", color=C_DMC, ms=4.5, mfc="white", zorder=3)
                ax.plot([t_dmc, XMAX], [y, y], color=C_DMC, lw=1.4, alpha=0.3,
                        solid_capstyle="butt", zorder=1)
            else:
                _point(ax, t_dmc, float(r["t_dmc_se"]), y, C_DMC, "s", 3)
            _point(ax, t_zh, float(r["t_zh_se"]), y, C_ZH, "o", 4)
        for _, codes in groups:
            bottom = min(ypos[code] for code in codes)
            if bottom > 0:
                ax.axhline(bottom - 0.5, color="0.75", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlim(XMIN, XMAX)
        ax.set_xticks([1e-2, 1e0, 1e2, 1e4, 1e6])
        ax.set_title(rf"$p=10^{{{round(math.log10(p))}}}$")
        ax.set_xlabel(r"$T_{2\%}$ (core-seconds)")
        ax.tick_params(axis="y", length=0)
        ax.grid(False, axis="y", which="both")
        ax.grid(True, axis="x", which="major", alpha=0.3, lw=0.5)
    ax0 = axes[0]
    ax0.set_ylim(-0.7, n - 0.3)
    ax0.set_yticks([ypos[code] for code in ROW_ORDER])
    ax0.set_yticklabels([TITLE[code] for code in ROW_ORDER])
    ax0.plot([], [], "o", color=C_ZH, ms=4.5, ls="none", label="Z-OBF+CE")
    ax0.plot([], [], "s", color=C_DMC, ms=4.5, ls="none", label="direct MC")
    ax0.plot([], [], "s", color=C_DMC, ms=4.5, mfc="white", ls="none",
             label="lower bound")
    ax0.legend(fontsize=6.5, loc="upper right", frameon=False, handletextpad=0.4)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig(OUT, dpi=300)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
