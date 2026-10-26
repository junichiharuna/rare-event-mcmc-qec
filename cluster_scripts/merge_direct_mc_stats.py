#!/usr/bin/env python3
"""Pool raw-Stats JSONL across all subjobs (per condition) and merge into
cluster_results/direct_mc_ew.csv.

Each (noise, code_type, d, p) group is pooled with pool_stats, estimated once,
and written as a single CSV row. Subjobs that died without writing simply do
not contribute. Merging is idempotent per condition: an existing row for the
same condition is replaced, unrelated rows are preserved.
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from direct_mc import (   # noqa: E402
    CSV_HEADER, _BB, read_stats_jsonl, dict_to_stats, pool_stats, estimate,
)

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collate_stats(input_glob):
    """Pool all JSONL records per condition; return a list of CSV row dicts."""
    groups = {}
    meta = {}
    for path in sorted(glob.glob(input_glob)):
        for rec in read_stats_jsonl(path):
            key = (rec["noise"], rec["code_type"], int(rec["d"]), float(rec["p"]))
            groups.setdefault(key, []).append(dict_to_stats(rec))
            m = meta.setdefault(key, {"n_target": 0, "n_cores": 0, "wall_clock": 0.0})
            m["n_target"] = max(m["n_target"], int(rec.get("n_target", 0)))
            m["n_cores"] += int(rec.get("n_cores", 0))
            m["wall_clock"] = max(m["wall_clock"], float(rec.get("wall_clock", 0.0)))

    rows = []
    for key, stats_list in groups.items():
        noise, code_type, d, p = key
        st = pool_stats(stats_list)
        est = estimate(st)
        m = meta[key]
        rows.append({
            "noise": noise, "code_type": code_type, "d": d, "p": p,
            "decoder": "bposd" if code_type in _BB else "mwpm",
            "n_cores": m["n_cores"], "n_target": m["n_target"],
            "n_trials": st.n_trials, "n_faulty": st.n_faulty,
            "p_L": est["p_L"], "p_L_se": est["p_L_se"],
            "ew_mean": est["ew_mean"], "ew_se": est["ew_se"],
            "vw": est["vw"], "vw_se": est["vw_se"],
            "wall_clock": m["wall_clock"],
            "feasible": st.n_faulty >= max(m["n_target"], 2),
        })
    rows.sort(key=lambda r: (r["noise"], r["code_type"], int(r["d"]), float(r["p"])))
    return rows


def merge_into_csv(new_rows, output):
    """Replace rows for the new conditions, keep all others; rewrite sorted."""
    existing = []
    if os.path.exists(output):
        with open(output, newline="") as f:
            existing = list(csv.DictReader(f))
    new_keys = {(r["noise"], r["code_type"], int(r["d"]), float(r["p"]))
                for r in new_rows}
    kept = [r for r in existing
            if (r["noise"], r["code_type"], int(r["d"]), float(r["p"])) not in new_keys]
    allrows = kept + [{k: r.get(k, "") for k in CSV_HEADER} for r in new_rows]
    allrows.sort(key=lambda r: (r["noise"], r["code_type"], int(r["d"]), float(r["p"])))
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerows(allrows)
    return allrows


def main():
    logs_dir = os.path.join(WORKDIR, "cluster_scripts", "logs")
    input_glob = os.path.join(logs_dir, "dmc_stats_*.jsonl")
    output = os.path.join(WORKDIR, "cluster_results", "direct_mc_ew.csv")
    new_rows = collate_stats(input_glob)
    if not new_rows:
        print(f"No stats files matching {input_glob}", file=sys.stderr)
        sys.exit(1)
    allrows = merge_into_csv(new_rows, output)
    for r in new_rows:
        print(f"{r['code_type']} p={r['p']:g}: F={r['n_faulty']} / M={r['n_trials']:,}"
              f"  p_L={r['p_L']:.3e}  E[w]={r['ew_mean']}")
    print(f"Merged {len(new_rows)} condition(s) -> {output} ({len(allrows)} rows total)")


if __name__ == "__main__":
    main()
