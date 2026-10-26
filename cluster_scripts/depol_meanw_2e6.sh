#!/bin/bash
#PBS -N depol_meanw_2e6
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=128gb
#PBS -l walltime=04:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/depol_meanw_2e6.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/depol_meanw_2e6.err
# Depolarizing first benchmark: the mirror of tab:meanw / tab:lowp under depolarizing
# noise (spec 2026-10-07). Protocol identical to lowp_audit_2e6_gr.sh: K=50 seeds,
# N=2e6 steps, both inits (short/long), 5 tau burn-in, OBF and Z+H, --cache-cap 200000.
# Only the noise model changes (--noise depolarizing).
#
# CONDITIONS. Invocation 1: 10 codes at p=1e-4 -> 6 non-flat codes x 2 inits + 4 flat
# codes x 1 init = 16 code-init pairs x 2 methods = 32 conditions. Invocation 2: 2 codes
# (bb144, bb_l9m6) x 2 p (1e-2, 1e-3) x 2 inits x 2 methods = 16 conditions. Total 48
# conditions x 50 seeds = 2400 chains in one flat longest-first pool, 25 waves on 96
# cores (saturation holds through the tail).
#
# WALLTIME. MEASURED by the smoke (job 264994, 2026-10-08) at N=2e4, 1 seed: OBF chains
# 0.2-0.7 s, Z+H chains 1.2-3.1 s; summed over the 48 conditions ~51 s per seed set
# -> x100 (length) x50 (seeds) / 96 cores ~ 0.74 node-hours; the bit-flip lowp_audit_2e6_gr
# analogue (5100 chains, 1.1 node-hours) agrees in scale. walltime 04:00:00 ~ 5x.
#
# MEMORY. Two 200k-entry caches per worker ~0.4 GB x 96 ~ 40 GB, plus 8 MB traces that
# never leave the workers. 128gb is ample.
#
# RESUME. Both invocations write to the SAME stem; all four CSVs are rewritten after each
# completed condition and finished conditions are skipped (the two invocations' conditions
# are disjoint). A walltime kill keeps its rows: resubmit the same script.
#
# REFERENCE. truth_ew / truth_se / p_L come from cluster_results/subset_sampling_depol.csv
# at job time; truth_source=missing (that job not landed) is not a failure.
set -uo pipefail
WORKDIR=/home/user/workspace/RIMC
cd "$WORKDIR" || exit 1
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96
git pull --ff-only origin main || true
STEM=cluster_results/depol_meanw_2e6
FAIL=0
conda run -n py311 --no-capture-output python cluster_scripts/lowp_audit.py --noise depolarizing \
    --codes toric5,toric7,surface5,surface7,ut3x11,ut3x15,hgp_hamming3,bb72,bb144,bb_l9m6 \
    --p-list 1e-4 --methods OBF,Z+H --inits short,long \
    --mcmc-target 2000000 --n-seeds 50 --n-cores 96 --cache-cap 200000 \
    --out-csv "$WORKDIR/${STEM}.csv" || FAIL=1
conda run -n py311 --no-capture-output python cluster_scripts/lowp_audit.py --noise depolarizing \
    --codes bb144,bb_l9m6 --p-list 1e-2,1e-3 --methods OBF,Z+H --inits short,long \
    --mcmc-target 2000000 --n-seeds 50 --n-cores 96 --cache-cap 200000 \
    --out-csv "$WORKDIR/${STEM}.csv" || FAIL=1
if [ "$FAIL" -eq 0 ]; then
    git add "${STEM}.csv" "${STEM}_chains.csv" "${STEM}_conv.csv" "${STEM}_seeds.csv" || FAIL=1
    git commit -m "data: depolarizing first benchmark, tab:meanw/tab:lowp mirror, K=50 N=2e6 (job ${PBS_JOBID})" || FAIL=1
    for _try in 1 2 3; do
        { git pull --rebase origin main && git push origin HEAD:main; } && break
        echo "[publish] push attempt $_try failed - retrying"; sleep 20
        [ "$_try" -eq 3 ] && FAIL=1
    done
fi
ntfy_done "RIMC depol_meanw_2e6 done" "job ${PBS_JOBID} finished (fail=$FAIL). Run: git pull"
exit $FAIL
