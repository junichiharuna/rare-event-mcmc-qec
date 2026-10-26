#!/usr/bin/env python
"""Regenerate every table and figure of the paper from the CSV files in cluster_results/.

    python reproduce.py          # rewrite paper/tables/*.tex and docs/figures/*.png in place
    python reproduce.py --check  # regenerate in a temporary copy and require the six
                                 # .tex files to match the committed ones byte for byte

The inputs are the CSV files only; no Monte Carlo runs. About one minute.
The figures are regenerated but not compared, since matplotlib's raster output
changes between versions.
"""
from __future__ import annotations

import argparse
import difflib
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
TABLES = ["meanw_table.tex", "lowp_table.tex", "depol_table.tex", "dmc_speedup_table.tex",
          "plcost_estimate_table.tex", "clusterk_tta_table.tex"]
FIGURES = ["fig_tta_walltime.png", "fig_pcluster_tta.png", "fig_intro_tta.png"]
# What --check copies: everything the generators read or write.
# check() removes the committed outputs from the copy before regenerating, so a
# generator that stops writing a table or figure is caught.
COPY_DIRS = ["src", "compare", "cluster_results", "paper/tables", "docs/figures"]


def _natural(path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(path))]


def dmc_run_files(root):
    """The dmc_speedup run files (run4..13, each with _bb108 and _topo) in natural order."""
    return sorted(glob.glob(os.path.join(root, "cluster_results", "dmc_speedup_run*.csv")),
                  key=_natural)


def steps(root):
    """(label, argv) in dependency order: the dmc_speedup step writes the CSV that the
    plcost and walltime steps read. argv paths are relative to root (cwd)."""
    py = sys.executable
    return [
        ("tab:meanw tab:lowp tab:depol",
         [py, "compare/build_audit_tables.py", "--paper-tables"]),
        ("tab:dmcspeedup",
         [py, "compare/analyze_dmc_speedup.py"] + dmc_run_files(root)),
        ("tab:plcost", [py, "compare/make_plcost_table.py"]),
        ("tab:clusterk fig:pcluster", [py, "compare/analyze_kernel_param_tta.py"]),
        ("fig:ttawalltime", [py, "compare/plot_tta_walltime.py"]),
        ("fig:intro", [py, "compare/plot_intro_tta.py"]),
    ]


def run(root):
    env = dict(os.environ, PYTHONUTF8="1", MPLBACKEND="Agg")
    for label, argv in steps(root):
        print("==", label, flush=True)
        subprocess.run(argv, cwd=root, env=env, check=True, stdout=subprocess.DEVNULL)
    missing = [t for t in TABLES if not os.path.isfile(os.path.join(root, "paper", "tables", t))]
    missing += [f for f in FIGURES if not os.path.isfile(os.path.join(root, "docs", "figures", f))]
    if missing:
        raise SystemExit("not produced: %s" % missing)


def compare_tables(ref_dir, new_dir, names=TABLES):
    """[(name, unified diff)] for the names whose bytes differ between the two dirs."""
    bad = []
    for n in names:
        with open(os.path.join(ref_dir, n), "rb") as f:
            a = f.read()
        with open(os.path.join(new_dir, n), "rb") as f:
            b = f.read()
        if a == b:
            continue
        if a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n"):
            bad.append((n, "line endings differ (CRLF vs LF); content identical"))
            continue
        diff = difflib.unified_diff(a.decode("utf-8", "replace").splitlines(),
                                    b.decode("utf-8", "replace").splitlines(),
                                    "committed/" + n, "regenerated/" + n, lineterm="")
        bad.append((n, "\n".join(diff)))
    return bad


def remove_outputs(root):
    """Delete the six tables and three figures under root; return the paths removed."""
    paths = [os.path.join(root, "paper", "tables", t) for t in TABLES]
    paths += [os.path.join(root, "docs", "figures", f) for f in FIGURES]
    removed = []
    for p in paths:
        if os.path.isfile(p):
            os.remove(p)
            removed.append(p)
    return removed


def check(root):
    with tempfile.TemporaryDirectory() as tmp:
        for d in COPY_DIRS:
            shutil.copytree(os.path.join(root, d), os.path.join(tmp, d),
                            ignore=shutil.ignore_patterns("__pycache__", "out", "plots"))
        remove_outputs(tmp)
        run(tmp)
        bad = compare_tables(os.path.join(root, "paper", "tables"),
                             os.path.join(tmp, "paper", "tables"))
    for n, diff in bad:
        print("MISMATCH", n)
        print(diff)
    print("reproduce --check:", "OK" if not bad else "%d table(s) differ" % len(bad))
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="regenerate in a temporary copy and compare the tables")
    a = ap.parse_args(argv)
    if a.check:
        return check(ROOT)
    run(ROOT)
    print("regenerated", TABLES, FIGURES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
