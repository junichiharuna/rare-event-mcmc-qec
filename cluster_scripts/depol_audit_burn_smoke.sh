#!/bin/bash
#PBS -N depol_burn_smoke
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=64gb:vnode=node012
#PBS -l walltime=00:40:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/depol_audit_burn_smoke.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/depol_audit_burn_smoke.err

# Fast cluster smoke (~minutes) for the burn-in-augmented depolarizing audit. Runs the
# FULL p=1e-2 conditions (toric d=3,5 + bb72 + bb144) at a small budget to validate the
# pipeline, the B1 depolarizing cluster move and the new ew_burn{5,10}* columns on a
# node, and to time the production. No git commit. (NOT --smoke, which is the local
# toric-d3 + bb72 subset.) Same budgets as the June smoke, so timings are comparable.
#
# WATCH FOR: capped5/capped10 ~ n_seeds on the bb144 rows (tau is a large fraction of N
# there, so the c*tau cut hits the max_frac=0.5 clamp and c=5/c=10 then coincide by
# construction) and 0 on the toric / bb72 Z+H rows, whose tau is O(10-100).

WORKDIR=/home/user/workspace/RIMC
cd "$WORKDIR" || exit 1
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96

OUT=$(mktemp -d)/depol_burn_smoke.csv
conda run -n py311 --no-capture-output python cluster_scripts/depol_audit.py \
    --mcmc-target 20000 --n-seeds 4 --truth-target 1500 --truth-mmax 4000000 \
    --n-cores 96 --out-csv $OUT
RC=$?

echo "=== smoke depol_audit_burn CSV ==="
cat $OUT
rm -rf $(dirname $OUT)
ntfy_done "RIMC depol_audit_burn SMOKE done" \
    "job ${PBS_JOBID} finished (rc=$RC) -> check the burn columns, then submit depol_audit_burn.sh"
exit $RC
