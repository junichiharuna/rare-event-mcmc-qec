#!/bin/bash
#PBS -N lowp_audit_k1
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=64gb
#PBS -l walltime=01:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/lowp_audit_k1.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/lowp_audit_k1.err
#
# WHAT. Add surface3 and concat_steane2 to the first benchmark: p=1e-2,1e-3,1e-4,
# OBF/Z+H, K=50, N=2e6, default base seed 1000; flat bases collapse to uniform init.
# WALLTIME. MEASURED by the smoke (job 265025, 2026-10-08): OBF 22-26 s and Z+H
# 75-122 s per production chain (x100 of the N/100 smoke); sum x50 /96 = 0.10 node-h,
# smoke's 2x suggestion 00:12:27. 01:00:00 keeps the spec's 1 h floor (~10x) because
# the smoke ran 12 chains on 96 cores and full-load per-chain wall is higher.
# SATURATION. One flat Pool(96): 2 codes x3 p x2 methods x1 init x50 =600 chains,
# 6.25 waves. K=50 is required; the final 24-chain tail (~2 min) is the spec's
# allowed exception. Longest-first within the driver; no serial condition pools.
# MEMORY. int32 trace 2e6 x4 =8 MB/worker, cache_cap=200000 <0.2 GB/worker;
# mem=64gb, below 340gb. Traces stay in workers; tau processing is bounded.
# RESUME. Use this NEW lowp_audit_2e6_k1 stem only. The driver checks the SUMMARY
# CSV alone: never point it at an older stem (companions would stay incomplete).
# Resubmit this same protocol to skip completed conditions; no --force.
# REFERENCE. cluster_results/subset_sampling.csv: all six code/p cells must resolve
# to subset_sampling:surface/3 or subset_sampling:concat_steane/2 in the smoke.
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
STEM=cluster_results/lowp_audit_2e6_k1
OUT="${STEM}.csv"
FAIL=0
conda run -n py311 --no-capture-output python cluster_scripts/lowp_audit.py \
    --codes surface3,concat_steane2 --p-list 1e-2,1e-3,1e-4 --methods OBF,Z+H --inits short,long \
    --mcmc-target 2000000 --n-seeds 50 --n-cores 96 --cache-cap 200000 \
    --out-csv "$OUT" || FAIL=1
if [ "$FAIL" -eq 0 ]; then
    git add "${STEM}.csv" "${STEM}_chains.csv" "${STEM}_conv.csv" "${STEM}_seeds.csv" || FAIL=1
    git diff --cached --quiet || git commit -m "data(lowp_audit_k1): first benchmark, K=50 N=2e6 (job ${PBS_JOBID})" || FAIL=1
    for _try in 1 2 3; do
        { git pull --rebase origin main && git push origin HEAD:main; } && break
        echo "[publish] push attempt $_try failed - retrying"
        sleep 20
        [ "$_try" -eq 3 ] && FAIL=1
    done
fi
ntfy_done "RIMC lowp_audit_k1 done" "job ${PBS_JOBID}; Run: git pull (fail=$FAIL)"
exit $FAIL
