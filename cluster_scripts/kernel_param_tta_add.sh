#!/bin/bash
#PBS -N kpt_add
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=64gb
#PBS -l walltime=03:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/kpt_add.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/kpt_add.err
# WHAT: kernel-parameter sweeps under the TTA protocol on the two codes missing from
# kernel_param_tta.sh: surface:5 (zero spread, MWPM) and hgp_hamming (spread 1, BP-OSD).
# Same arms (p_cluster in {0,0.1,0.2,0.3,0.5,0.7,0.9} at m=2 and m in {3,4,6} at 0.3), same
# p = 1e-3, 1e-4, 1e-5, K=50 seeds, L = 1000*2^k <= 2,048,000, one run per RUN_ID (4..8) with
# the same seed streams, so the (0.3,2) arm reproduces the Z+H chains of
# dmc_speedup_run{r}_topo.csv (surface:5) and dmc_speedup_run{r}.csv (hgp_hamming) bit for bit.
# WALLTIME: MEASURED by the smoke (job 265047, 2026-10-10): surface:5 3.83e-5 s/step and
# hgp_hamming 4.10e-5 s/step on the slowest arm (0.3,2) at p=1e-3, cpu/wall = 0.997 ->
# 135 core-hours = 1.41 node-hours per run (the smoke's own extrapolation); the earlier
# kernel_param_tta runs overran their estimate by ~10%. walltime 03:00:00 ~ 2.1x.
# SATURATION: one flat pool over 2 codes x 3 p x 10 arms x 50 seeds x 12 lengths = 36,000
# (arm, seed, L) tasks sorted longest-first; all 96 cores stay busy through the tail.
# MEMORY: record_trace=False, one 200k-entry cache per worker ~ 0.2 GB x 96 ~ 20 GB; 64gb ample.
# WALL-TIME MEASUREMENT: this job measures wall time. It occupies the whole node (ncpus=96),
# runs all conditions inside one pool (no co-tenants) and records wall and cpu per chain.
# RESUME: resubmit with the same RUN_ID; the CSV is rewritten atomically and appended per chain. A re-run of an already published RUN_ID is a no-op (0 pending, nothing to commit, fail=0).
# CONCURRENCY: never submit a RUN_ID while a job with the same RUN_ID is queued or running; the four concurrent jobs share one checkout and each commits only its own CSV.
# SUBMISSION: qsub -v RUN_ID=4 -o .../logs/kpt_add4.out -e .../logs/kpt_add4.err \
#   cluster_scripts/kernel_param_tta_add.sh  ... RUN_ID=8 (at most 4 concurrent; chain the 5th
#   with -W depend=afterany:<jobid of run 4>). New stem *_add.csv: the existing
#   kernel_param_tta_run{r}.csv and _ext.csv are never touched.
set -uo pipefail
: "${PBS_JOBID:?submit this script with qsub}"
WORKDIR=/home/user/workspace/RIMC
cd "$WORKDIR" || exit 1
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96 PYTHONUTF8=1
RUN_ID="${RUN_ID:?submit with qsub -v RUN_ID=<4..8>}"
case "$RUN_ID" in
    4|5|6|7|8) ;;
    *) echo "RUN_ID must be 4..8" >&2; exit 1 ;;
esac
git pull --ff-only origin main || true
exec > >(tee -a "cluster_scripts/logs/kpt_add_run${RUN_ID}.log") 2>&1
OUT=cluster_results/kernel_param_tta_run${RUN_ID}_add.csv
FAIL=0
SECONDS=0
conda run -n py311 --no-capture-output python cluster_scripts/kernel_param_tta.py \
    --specs surface:5,hgp_hamming --p-list 1e-3,1e-4,1e-5 --arms all \
    --n-seeds 50 --L0 1000 --L-max 2048000 --cache-cap 200000 --n-cores 96 \
    --run-id "$RUN_ID" --out-csv "$WORKDIR/$OUT" || FAIL=1
echo "[kpt_add] elapsed ${SECONDS}s (fail=$FAIL)"
if [ "$FAIL" -eq 0 ] && [ -f "$OUT" ]; then
    # Shared checkout: up to four RUN_IDs publish from the same index. Commit only this
    # run's CSV (pathspec-limited), skip the commit when it is already published, and
    # retry the whole add+commit+pull+push sequence (index.lock collisions, push races).
    PUB=1
    for _try in 1 2 3 4 5; do
        if git add -- "$OUT" \
           && { git diff --cached --quiet -- "$OUT" \
                || git commit -m "data(kernel_param_tta): run ${RUN_ID} added codes surface:5, hgp_hamming (job ${PBS_JOBID})" -- "$OUT"; } \
           && git pull --rebase origin main \
           && git push origin HEAD:main; then
            PUB=0
            break
        fi
        echo "[publish] attempt $_try failed - retrying"
        sleep $((20 * _try))
    done
    [ "$PUB" -eq 0 ] || FAIL=1
fi
ntfy_done "RIMC kernel_param_tta_add run${RUN_ID} done" "job ${PBS_JOBID} finished (fail=$FAIL) after ${SECONDS}s. Run: git pull"
exit $FAIL
