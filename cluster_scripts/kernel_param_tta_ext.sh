#!/bin/bash
#PBS -N kpt_ext
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=64gb
#PBS -l walltime=12:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/kpt_ext.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/kpt_ext.err
# WHAT: extension of the BB arms (bb144, bb_l9m6) at p = 1e-3, 1e-5 to L = 8.2M. No arm reaches the
# 2% target within 2,048,000 steps there, so only the two new lengths 4,096,000 and 8,192,000 are run
# (--L0 4096000 --L-max 8192000), all 10 arms, K=50 seeds. The seeds are the same as in
# kernel_param_tta.sh (7000 + s + 1_000_000*(run_id-1), independent of L), so each chain is the
# continuation of the corresponding shorter chain. One run per RUN_ID (4..8), written to
# cluster_results/kernel_param_tta_run${RUN_ID}_ext.csv (a new file; analyze_kernel_param_tta.py
# treats run{r}.csv + run{r}_ext.csv as one run).
# WALLTIME: per run 2 codes x 2 p x 10 arms x 50 seeds x (4.096M + 8.192M) = 2.46e10 steps at the
# measured 6.9e-5 s/step (smoke 264992) -> 471 core-hours = 4.9 node-hours; the 2M-cap runs took
# ~3.3 h against a 2.97 h estimate, so expect ~5.5 h; walltime 12:00:00 ~ 2.2x.
# SATURATION: one flat pool over 4000 (arm, seed, L) chains sorted longest-first; all 96 cores stay
# busy through the tail.
# MEMORY: record_trace=False, one 200k-entry cache per worker ~ 0.2 GB x 96 ~ 20 GB; 64gb ample.
# RESUME: resubmit with the same RUN_ID; the _ext CSV is rewritten atomically and appended per chain.
# SUBMISSION: qsub -v RUN_ID=4 -o /home/user/workspace/RIMC/cluster_scripts/logs/kpt_ext4.out \
#   -e /home/user/workspace/RIMC/cluster_scripts/logs/kpt_ext4.err cluster_scripts/kernel_param_tta_ext.sh
#   ... RUN_ID=8 (at most 4 concurrent; submit the 5th after the first finishes).
set -uo pipefail
WORKDIR=/home/user/workspace/RIMC
cd "$WORKDIR" || exit 1
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96
RUN_ID="${RUN_ID:?submit with qsub -v RUN_ID=<4..8>}"
git pull --ff-only origin main || true
OUT=cluster_results/kernel_param_tta_run${RUN_ID}_ext.csv
FAIL=0
SECONDS=0
conda run -n py311 --no-capture-output python cluster_scripts/kernel_param_tta.py \
    --specs bb144,bb_l9m6 --p-list 1e-3,1e-5 --arms all \
    --n-seeds 50 --L0 4096000 --L-max 8192000 --cache-cap 200000 --n-cores 96 \
    --run-id "$RUN_ID" --out-csv "$WORKDIR/$OUT" || FAIL=1
echo "[kpt_ext] elapsed ${SECONDS}s (fail=$FAIL)"
if [ "$FAIL" -eq 0 ]; then
    git add "$OUT" || FAIL=1
    git commit -m "data(kernel_param_tta): run ${RUN_ID} extension of the BB arms at p=1e-3,1e-5 to L=8.2M (job ${PBS_JOBID})" || FAIL=1
    for _try in 1 2 3; do
        { git pull --rebase origin main && git push origin HEAD:main; } && break
        echo "[publish] push attempt $_try failed - retrying"; sleep 20
        [ "$_try" -eq 3 ] && FAIL=1
    done
fi
ntfy_done "RIMC kernel_param_tta_ext run${RUN_ID} done" "job ${PBS_JOBID} finished (fail=$FAIL) after ${SECONDS}s. Run: git pull"
exit $FAIL
