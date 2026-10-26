#!/bin/bash
#PBS -N dmc_speedup10
#PBS -l select=1:ncpus=96
#PBS -l walltime=16:00:00
#PBS -j oe
#
# TTA rerun at 50 seeds x 10 independent runs (2026-09-02 design; replaces the published
# 24-seed x 3-run tab:dmcspeedup / tab:obfspeedup numbers, see the \todo marker in §5).
# One job = one full independent run: submit with
#     qsub -v RUN_ID=<4..13> cluster_scripts/dmc_speedup10.sh
# RUN_ID 4..13 keeps every RNG stream disjoint from the retired July runs 1-3 (the worker
# offsets all seeds by 1e6*(run_id-1)); the old CSVs stay untouched as history and the new
# table is built from dmc_speedup_run{4..13}.csv only.  Max 4 nodes concurrent: submit
# runs 4-7 plain and chain 8-13 with -W depend=afterany:<job of RUN_ID-4>.
#
# Two SERIAL phases inside one job (replaces the July base+topup two-job orchestration;
# the worker's _load_done resume makes phase B skip everything phase A already wrote,
# including the dmc_cutoff chunks):
#   A. the full 7-code x {1e-1,1e-3,1e-5} base run at published caps, 50 seeds;
#   B. the bb144-only cap extension (OBF 33.5M -> 268M for the (b) floors, Z+H
#      512k -> 33.5M for the (c) cell), same CSV.
# Protocol is otherwise identical to the published benchmark (same caps, same grid,
# same dmc budget), so the only deltas vs July are n_seeds and the number of runs.
#
# SATURATION: the worker orders tasks hardest-first (bb144 low-p first) and dispatches
# via imap_unordered; phase A has O(10^4) tasks, phase B ~900 (the ~1-h 268M OBF chains
# lead), so both phases fill 96 cores and the tail is short cheap tasks.
# MEMORY: cache_cap=200000 per worker, no traces; well under the 340 gb cap.
# WALLTIME 16:00:00 from the 2026-09-02 smoke + July run1 actuals: phase A ~= 1.75 h dMC
# (analytic) + ~2.25 h advantage (run1 actual) x 50/24 ~= 6.5 h; phase B ~= per-step
# rates x new grid steps x 50 seeds x 2 p / 96 ~= 3.5 h; total ~10 h, 1.6x headroom.
# If a run is killed at the wall, resubmit the same RUN_ID: the resume logic continues.
set -euo pipefail
cd "$PBS_O_WORKDIR"
source cluster_scripts/lib/notify.sh
git pull --ff-only origin main || true

RUN_ID="${RUN_ID:?submit with qsub -v RUN_ID=<4..13>}"
OUT="cluster_results/dmc_speedup_run${RUN_ID}.csv"

echo "=== phase A (base caps, 50 seeds, run ${RUN_ID}): start $(date -Is) ==="
SECONDS=0
conda run -n py311 python cluster_scripts/dmc_speedup.py \
  --specs surface:3,concat_steane:2,hgp_hamming,rect:3x11,rect:3x15,bb72,bb144 \
  --p-list 1e-1,1e-3,1e-5 \
  --dmc-chunks 96 --dmc-budget 300 --n-seeds 50 --L0 1000 \
  --L-max 33554432 --ctrl-obf-L-max 512000 --kernel-L-max 512000 \
  --n-cores 96 --run-id "$RUN_ID" --out-csv "$OUT"
echo "=== phase A: end $(date -Is) after ${SECONDS}s ==="

echo "=== phase B (bb144 top-up caps, run ${RUN_ID}): start $(date -Is) ==="
SECONDS=0
conda run -n py311 python cluster_scripts/dmc_speedup.py \
  --specs bb144 --p-list 1e-3,1e-5 \
  --dmc-chunks 96 --dmc-budget 300 --n-seeds 50 --L0 1000 \
  --L-max 268435456 --ctrl-obf-L-max 512000 --kernel-L-max 33554432 \
  --n-cores 96 --run-id "$RUN_ID" --out-csv "$OUT"
echo "=== phase B: end $(date -Is) after ${SECONDS}s ==="

git add "$OUT"
git commit -m "results(dmc-speedup): 50-seed run ${RUN_ID} (base + bb144 top-up; analysis local)" || true
git pull --rebase origin main || true
git push origin main || true
ntfy_done "dmc_speedup10 run${RUN_ID} done"
