#!/bin/bash
#PBS -N dmc10_smoke
#PBS -l select=1:ncpus=8
#PBS -l walltime=00:30:00
#PBS -j oe
#
# Smoke for the 50-seed x 10-run TTA rerun (dmc_speedup10.sh): validates the NEW part of
# that job -- the two-phase (base -> bb144 top-up) structure inside ONE job, with phase B
# resume-appending to phase A's CSV -- on real cluster CPUs, at toy scale (2 seeds, small
# caps, run-id 99 so every seed stream is disjoint from runs 1-13). Exercises both decoder
# types (bb144 BP-OSD, rect MWPM), the dmc_cutoff branch, and the resume dedup.
#
# EXTRAPOLATION to production (per run, 96 cores, 50 seeds):
#   phase A = dMC part (analytic: 7 codes x 3 p x 96 chunks x 300 s / 96 cores = 1.75 h)
#             + advantage part (run1 2026-07 actual: ~2.25 h at 24 seeds -> x50/24 ~ 4.7 h)
#             => ~6.5 h.  The smoke's job here is to confirm env + per-step wall rates
#             still match the July run (compare bb144 OBF/Z+H s/step from the smoke rows).
#   phase B = bb144-only cap extension (OBF 33.5M->268M, Z+H 512k->33.5M), cost ~ per-step
#             wall x new grid steps x 50 seeds x 2 p / 96 cores => ~3.5 h at July rates.
#   Estimated production total ~10 h => walltime 16:00:00 (~1.6x headroom); revisit if the
#   smoke's per-step rates disagree with July by more than ~1.5x.
set -euo pipefail
cd "${PBS_O_WORKDIR:-$(dirname "$0")/..}"
source cluster_scripts/lib/notify.sh

OUT=cluster_results/dmc_speedup10_smoke.csv
rm -f "$OUT"

echo "=== phase A smoke (base caps, 2 seeds): start $(date -Is) ==="
SECONDS=0
conda run -n py311 python cluster_scripts/dmc_speedup.py \
  --specs bb144,rect:3x11 --p-list 1e-3 \
  --dmc-chunks 8 --dmc-budget 10 --n-seeds 2 --L0 1000 \
  --L-max 131072 --ctrl-obf-L-max 16384 --kernel-L-max 16384 \
  --n-cores 8 --run-id 99 --out-csv "$OUT"
echo "PHASE_A_SECONDS=${SECONDS}"

echo "=== phase B smoke (bb144 cap extension, resume append): start $(date -Is) ==="
SECONDS=0
conda run -n py311 python cluster_scripts/dmc_speedup.py \
  --specs bb144 --p-list 1e-3 \
  --dmc-chunks 8 --dmc-budget 10 --n-seeds 2 --L0 1000 \
  --L-max 1048576 --ctrl-obf-L-max 16384 --kernel-L-max 131072 \
  --n-cores 8 --run-id 99 --out-csv "$OUT"
echo "PHASE_B_SECONDS=${SECONDS}"

# Sanity: resume must have deduplicated (no repeated task keys), and phase B must have
# added ONLY the extended-cap bb144 rows (OBF > 131072, Z+H > 16384), no rect rows.
conda run -n py311 python - <<'EOF'
import csv, collections, sys
rows = list(csv.DictReader(open("cluster_results/dmc_speedup10_smoke.csv")))
keys = [(r["phase"], r["code"], r["p"], r["method"], r["seed"], r["length"]) for r in rows]
dup = [k for k, c in collections.Counter(keys).items() if c > 1]
assert not dup, f"duplicate task keys (resume failed): {dup[:5]}"
mx = {}
for r in rows:
    if r["phase"] == "advantage":
        k = (r["code"], r["method"])
        mx[k] = max(mx.get(k, 0), int(r["length"]))
assert mx[("bb144", "OBF")] == 1024000, mx
assert mx[("bb144", "Z+H")] == 128000, mx
assert mx[("rect:3x11", "OBF")] == 16000, mx
n_dmc = sum(1 for r in rows if r["phase"] == "dmc_cutoff")
assert n_dmc == 16, n_dmc  # 2 codes x 8 chunks, NOT re-run by phase B
print(f"sanity OK: {len(rows)} rows, max lengths {mx}, {n_dmc} dmc chunks")
EOF

echo "SMOKE_OK"
ntfy_done "dmc10 smoke done"
