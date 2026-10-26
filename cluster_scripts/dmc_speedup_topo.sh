#!/bin/bash
#PBS -N dmc_speedup_topo
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=96gb
#PBS -l walltime=03:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/dmc_speedup_topo.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/dmc_speedup_topo.err
#
# WHAT. Add toric:5,toric:7,surface:5,surface:7 to the second benchmark, p=1e-1,
# 1e-3,1e-5; 50 seeds x10 independent RUN_IDs 4..13. OBF/Z+H caps 512000,
# L0=1000 (10 checkpoints); direct MC 96 chunks x300 s/cell. One base phase.
# WALLTIME. MEASURED by the smoke (job 265026, 2026-10-08): OBF 1.2-2.4e-5 and Z+H
# 5.0-8.2e-5 s/step -> MCMC 12.2 core-h; dMC fixed 12 cells x96 x300 s =96 core-h;
# 1.13 node-h/run, smoke's 2x suggestion 02:15:12 -> 03:00:00 (2.7x). dMC rates
# 2.1e4-1.8e5 trials/core-s; at 1e-5 the cutoff gives 0 faulty draws (floor, as planned).
# SATURATION. One flat Pool(96): 1152 dMC tasks (300 s each) +12000 MCMC chains,
# hardest-first with imap_unordered; the tail consists of short MCMC tasks.
# MEMORY. Driver default cache_cap=200000, record_trace=False, n<=98;
# mem=96gb <340gb. Bounded caches; rows are streamed to disk.
# RESUME. New dmc_speedup_run${RUN_ID}_topo.csv only; existing main/bb108 stems
# stay untouched. Resubmit SAME RUN_ID after interruption to retain finished rows.
# Completeness status is included in commit/ntfy; do not analyse INCOMPLETE runs.
# Per-run output appends to cluster_scripts/logs/dmc_speedup_topo_run${RUN_ID}.log.
# REFERENCE. dMC cutoff at p=1e-1; analysis uses subset_sampling.csv at p=1e-3
# and the four synthesized p=1e-5 rows from existing f_w. Never synthesize p=1e-1.
#
# Submit only after all three smokes pass and measured 2x walltimes are set.
# Spec section 4: <=4 nodes; longest chains first, B: topo 4->7->10->13,
# C: topo 5->8->11, D: topo 6->9->12, then A: BB108->k1.
# Chain successors with qsub -W depend=afterany:<previous-jobid>; git pull first.
# Never run this script or its driver on the login node.
set -uo pipefail
: "${PBS_JOBID:?submit this script with qsub}"
WORKDIR=/home/user/workspace/RIMC
cd "$WORKDIR" || exit 1
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96 PYTHONUTF8=1
git pull --ff-only origin main || true
RUN_ID="${RUN_ID:?submit with qsub -v RUN_ID=<4..13>}"
case "$RUN_ID" in
    4|5|6|7|8|9|10|11|12|13) ;;
    *) echo "RUN_ID must be 4..13" >&2; exit 1 ;;
esac
exec > >(tee -a "cluster_scripts/logs/dmc_speedup_topo_run${RUN_ID}.log") 2>&1
OUT="cluster_results/dmc_speedup_run${RUN_ID}_topo.csv"
FAIL=0
conda run -n py311 --no-capture-output python cluster_scripts/dmc_speedup.py \
    --specs toric:5,toric:7,surface:5,surface:7 --p-list 1e-1,1e-3,1e-5 \
    --dmc-chunks 96 --dmc-budget 300 --n-seeds 50 --L0 1000 \
    --L-max 33554432 --ctrl-obf-L-max 512000 --kernel-L-max 512000 \
    --n-cores 96 --run-id "$RUN_ID" --out-csv "$OUT" || FAIL=1
STATUS=$(conda run -n py311 --no-capture-output python cluster_scripts/check_dmc_speedup_complete.py "$OUT" --n-seeds 50 --expect OBF:10,Z+H:10 || echo CHECK_FAILED)
echo "completeness: $STATUS"
[ "$STATUS" = COMPLETE ] || FAIL=1
# Preserve completed rows even when the driver/check reports an incomplete run.
if [ -f "$OUT" ]; then
    git add "$OUT" || FAIL=1
    git diff --cached --quiet || git commit -m "data(dmc_speedup_topo): run ${RUN_ID}, $STATUS (job ${PBS_JOBID})" || FAIL=1
    for _try in 1 2 3; do
        { git pull --rebase origin main && git push origin HEAD:main; } && break
        echo "[publish] push attempt $_try failed - retrying"
        sleep 20
        [ "$_try" -eq 3 ] && FAIL=1
    done
else
    FAIL=1
fi
ntfy_done "RIMC dmc_speedup_topo run${RUN_ID} done" "job ${PBS_JOBID}; $STATUS; Run: git pull (fail=$FAIL)"
exit $FAIL
