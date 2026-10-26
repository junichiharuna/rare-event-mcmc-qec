#!/bin/bash
#PBS -N lowp_bb108_p1e2_4e7
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=96gb
#PBS -l walltime=05:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/lowp_bb108_p1e2_4e7.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/lowp_bb108_p1e2_4e7.err
#
# WHAT. BB[[108,8,10]] (bb_l9m6), p=1e-2, Z+H, short/long, N=4e7, K=48,
# default base seed 1000. Check the +0.095(26) (3.6 sigma) long-init residual;
# keep separate evidence for a footnote, without replacing tab:lowp rows.
# WALLTIME. MEASURED by the smoke (job 265027, 2026-10-08): 55.2 / 56.1 s per
# 4e5 steps (long / short) -> 5520 / 5608 s = 1.56 h per production chain, one wave;
# smoke's 2x suggestion 03:06:57. 05:00:00 kept (3.2x; 2.9x against the full-load
# 314 s/2e6 = 1.75 h of lowp_audit_2e6_gr.csv), since a wall kill loses all work.
# Rows appear only on condition completion: a wall kill can lose all work.
# SATURATION. One flat Pool(96): 2 inits x48 seeds =96 chains, exactly one wave;
# no serial init pools. Longest-first task dispatch, no extra wave of idle cores.
# MEMORY. int32 trace 4e7 x4 =160 MB +40 MB accept-proxy temporary +~0.2 GB
# BP-OSD cache (cache_cap=200000) ~0.4 GB/worker x96 ~40 GB; mem=96gb <340gb.
# Traces stay in workers and tau processing is bounded.
# RESUME. New lowp_bb108_p1e2_4e7 stem; SUMMARY CSV alone decides completion.
# Never reuse an older CSV stem to extend length: its conditions would be skipped
# and companions would not be regenerated. Same-protocol resubmit skips finished
# conditions, but unfinished conditions must restart all 48 chains.
# REFERENCE. Look up cluster_results/subset_sampling.csv at job time; the existing
# BB108 p=1e-2 reference has E[w|faulty] ~4.961. Smoke checks reference lookup.
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
STEM=cluster_results/lowp_bb108_p1e2_4e7
OUT="${STEM}.csv"
FAIL=0
conda run -n py311 --no-capture-output python cluster_scripts/lowp_audit.py \
    --codes bb_l9m6 --p-list 1e-2 --methods Z+H --inits short,long \
    --mcmc-target 40000000 --n-seeds 48 --n-cores 96 --cache-cap 200000 \
    --out-csv "$OUT" || FAIL=1
if [ "$FAIL" -eq 0 ]; then
    git add "${STEM}.csv" "${STEM}_chains.csv" "${STEM}_conv.csv" "${STEM}_seeds.csv" || FAIL=1
    git diff --cached --quiet || git commit -m "data(lowp_bb108_p1e2_4e7): BB108 p=1e-2 Z+H short/long, K=48 N=4e7 (job ${PBS_JOBID})" || FAIL=1
    for _try in 1 2 3; do
        { git pull --rebase origin main && git push origin HEAD:main; } && break
        echo "[publish] push attempt $_try failed - retrying"
        sleep 20
        [ "$_try" -eq 3 ] && FAIL=1
    done
fi
ntfy_done "RIMC lowp_bb108_p1e2_4e7 done" "job ${PBS_JOBID}; Run: git pull (fail=$FAIL)"
exit $FAIL
