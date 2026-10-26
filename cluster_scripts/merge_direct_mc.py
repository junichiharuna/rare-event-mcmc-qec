#!/usr/bin/env python3
"""Merge per-subjob row CSVs into cluster_results/direct_mc_ew.csv."""
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from direct_mc import CSV_HEADER   # noqa: E402

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VALID_NOISE = {"bit_flip", "depolarizing"}
_IDX_RE = re.compile(r"dmc_row_(\d+)\.csv$")


def _file_index(path):
    m = _IDX_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else 1 << 30


def collate(input_glob, output):
    files = sorted(glob.glob(input_glob), key=_file_index)
    rows = []
    n_skipped = 0
    for path in files:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("noise") not in _VALID_NOISE:
                    n_skipped += 1
                    continue
                rows.append(r)
    if n_skipped:
        print(f"Skipped {n_skipped} corrupt rows", file=sys.stderr)
    rows.sort(key=lambda r: (r["noise"], r["code_type"], int(r["d"]), float(r["p"])))
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    logs_dir = os.path.join(WORKDIR, "cluster_scripts", "logs")
    input_glob = os.path.join(logs_dir, "dmc_row_*.csv")
    output = os.path.join(WORKDIR, "cluster_results", "direct_mc_ew.csv")
    files = sorted(glob.glob(input_glob), key=_file_index)
    if not files:
        print(f"No files matching {input_glob}", file=sys.stderr)
        sys.exit(1)
    rows = collate(input_glob, output)
    print(f"Merged {len(rows)} rows from {len(files)} files -> {output}")


if __name__ == "__main__":
    main()
