#!/bin/bash
#PBS -N kpt_run
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=64gb
#PBS -l walltime=08:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/kpt_run.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/kpt_run.err
# WHAT: kernel-parameter sweeps under the TTA protocol. p_cluster in {0,0.1,0.2,0.3,0.5,0.7,0.9}
# at m=2 and m in {3,4,6} at p_cluster=0.3 (10 arms), on rect:3x11, bb144, bb_l9m6 at
# p = 1e-3, 1e-4, 1e-5; K=50 seeds; L = 1000*2^k <= 2,048,000. One run per RUN_ID (4..8) with
# disjoint seed streams. The (0.3,2) arm reproduces dmc_speedup's Z+H chains bit for bit.
# WALLTIME: MEASURED by the smoke (job 264992, 2026-10-08): bb144 6.9e-5 s/step, rect:3x11
# 3.0e-5 s/step at p=1e-3, cpu/wall = 0.997 -> 285 core-hours = 2.97 node-hours per run
# (the smoke's own extrapolation; bb_l9m6 assumed at the bb144 rate). walltime 08:00:00 ~ 2.7x.
# SATURATION: one flat pool over ~54k (arm, seed, L) tasks sorted longest-first; all 96
# cores stay busy through the tail.
# MEMORY: record_trace=False, one 200k-entry cache per worker ~ 0.2 GB x 96 ~ 20 GB; 64gb ample.
# WALL-TIME MEASUREMENT: this job measures wall time. It occupies the whole node (ncpus=96),
# runs all conditions inside one pool (no co-tenants) and records wall and cpu per chain.
# RESUME: resubmit with the same RUN_ID; the CSV is rewritten atomically and appended per chain.
# SUBMISSION: qsub -v RUN_ID=4 cluster_scripts/kernel_param_tta.sh  ... RUN_ID=8
# (at most 4 concurrent; submit the 5th after the first finishes). The production run ids
# are 4..8 so that every run's (0.3,2) arm can be cross-checked bit for bit against the
# existing cluster_results/dmc_speedup_run{4..8}.csv (analyze_kernel_param_tta.py --check-dmc).
set -uo pipefail
WORKDIR=/home/user/workspace/RIMC
cd "$WORKDIR" || exit 1
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96
RUN_ID="${RUN_ID:?submit with qsub -v RUN_ID=<4..8>}"
git pull --ff-only origin main || true
OUT=cluster_results/kernel_param_tta_run${RUN_ID}.csv
FAIL=0
SECONDS=0
conda run -n py311 --no-capture-output python cluster_scripts/kernel_param_tta.py \
    --specs rect:3x11,bb144,bb_l9m6 --p-list 1e-3,1e-4,1e-5 --arms all \
    --n-seeds 50 --L0 1000 --L-max 2048000 --cache-cap 200000 --n-cores 96 \
    --run-id "$RUN_ID" --out-csv "$WORKDIR/$OUT" || FAIL=1
echo "[kpt] elapsed ${SECONDS}s (fail=$FAIL)"
if [ "$FAIL" -eq 0 ]; then
    git add "$OUT" || FAIL=1
    git commit -m "data(kernel_param_tta): run ${RUN_ID} (p_cluster and m sweeps, TTA protocol) (job ${PBS_JOBID})" || FAIL=1
    for _try in 1 2 3; do
        { git pull --rebase origin main && git push origin HEAD:main; } && break
        echo "[publish] push attempt $_try failed - retrying"; sleep 20
        [ "$_try" -eq 3 ] && FAIL=1
    done
fi
ntfy_done "RIMC kernel_param_tta run${RUN_ID} done" "job ${PBS_JOBID} finished (fail=$FAIL) after ${SECONDS}s. Run: git pull"
exit $FAIL
