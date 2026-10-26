#!/bin/bash
#PBS -N lowp_audit_gr_smoke
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=128gb
#PBS -l walltime=00:30:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/lowp_audit_gr_smoke.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/lowp_audit_gr_smoke.err

# Fast cluster smoke (~minutes) for the GELMAN-RUBIN rerun of the low-p cross-code audit
# (cluster_scripts/lowp_audit_2e6_gr.sh). Identical in spirit to lowp_audit_smoke.sh -- the
# FULL production condition set (all 11 codes at p=1e-2, 1e-3 and 1e-4, both inits,
# OBF + Z+H) at 1/50 of the chains and 1/100 of the production chain length -- but it exists
# to validate the NEW outputs the driver grew on 2026-09-04:
#   * the fourth CSV <out>_seeds.csv (one row per (condition, seed): the per-chain
#     ew/tau/var/n_used/capped ingredients that used to be discarded), and
#   * the four appended summary columns w_var_within_mean / w_var_within_median /
#     n_used5_mean / tau_burn5_mean.
# It also re-validates what the old smoke did: env/paths/pool, the BP-OSD and MWPM paths,
# the subset_sampling.csv truth lookup (hgp_hamming3 / bb72 / bb_l9m6 are expected to report
# truth_source=missing until their reference job lands -- that is a PASS, not a failure),
# the (now four) CSV writers and the resume path, and it gives the timing for the production
# walltime. Writes to a mktemp CSV; no git commit.
#
# 102 conditions x 1 seed = 102 tasks on 96 cores. That is barely over one wave ON PURPOSE
# (a smoke measures per-chain cost, not throughput); the production job runs 5100 tasks in
# one flat longest-first pool.

WORKDIR=/home/user/workspace/RIMC
cd $WORKDIR
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96

OUT=$(mktemp -d)/lowp_audit_gr_smoke.csv
conda run -n py311 python cluster_scripts/lowp_audit.py \
    --codes hgp_hamming3,bb72,bb_l9m6,bb144,ut5x5,ut3x11,ut3x15,toric5,toric7,surface5,surface7 \
    --p-list 1e-2,1e-3,1e-4 --methods OBF,Z+H --inits short,long \
    --mcmc-target 20000 --n-seeds 1 --n-cores 96 --cache-cap 200000 \
    --out-csv $OUT

echo "=== smoke lowp_audit_gr summary CSV (head) ==="
head -5 $OUT
echo "=== smoke lowp_audit_gr PER-SEED CSV (the new file) ==="
cat ${OUT%.csv}_seeds.csv
echo "=== smoke lowp_audit_gr per-chain checkpoints (head) ==="
head -20 ${OUT%.csv}_chains.csv
rm -rf $(dirname $OUT)
ntfy_done "RIMC lowp_audit_gr SMOKE done" "job ${PBS_JOBID} finished OK -> check _seeds.csv + the w_var_within_* columns, estimate walltime, submit lowp_audit_2e6_gr.sh"
