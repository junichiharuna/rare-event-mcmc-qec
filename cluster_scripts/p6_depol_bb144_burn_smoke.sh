#!/bin/bash
#PBS -N p6_burn_smoke
#PBS -l select=1:ncpus=96:mem=64gb:vnode=node012
#PBS -l walltime=00:40:00
#PBS -j oe
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/p6_depol_bb144_burn_smoke.out

# Fast cluster smoke (~minutes) for the burn-in-augmented p6 depol bb144 low-p audit.
# The production is ~4.5 h with NO RESUME, so the cheap pre-flight is worth it -- and
# the one code path today's other smokes did NOT exercise is depol_audit.py's
# --codes/--p-list override (override_conditions), which is exactly how p6 selects its
# conditions. Everything else (the depolarizing kernels, the burn columns) was already
# validated on a node by job 263630.
#
# Trimmed only where the cost lives: p=3e-3 alone (the cheaper of the two truths;
# p=1e-3 is the identical code path, just 2e9 trials deep), truth-mmax 2e7 instead of
# 2e9, and a small MCMC budget. At P_L = 7.5e-7 that yields ~15 faulty, enough for the
# truth row to be emitted and the pipeline to run end to end. No git commit.
#
# WATCH FOR: the ew_burn{5,10}* columns present and populated; capped5 high on these
# bb144 rows (tau is a large fraction of N, so the c*tau cut hits the max_frac=0.5
# clamp and c=5 and c=10 then coincide by construction).

set -uo pipefail
WORKDIR=/home/user/workspace/RIMC
cd "$WORKDIR" || exit 1
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96

OUT=$(mktemp -d)/p6_burn_smoke.csv
conda run -n py311 --no-capture-output python cluster_scripts/depol_audit.py \
  --codes bb144 --p-list 3e-3 \
  --truth-target 20 --truth-mmax 2e7 \
  --mcmc-target 20000 --n-seeds 4 --n-cores 96 \
  --out-csv "$OUT"
RC=$?

echo "=== smoke p6_depol_bb144_burn CSV ==="
cat "$OUT"
rm -rf "$(dirname "$OUT")"
ntfy_done "RIMC p6_depol_bb144_burn SMOKE done" \
    "job ${PBS_JOBID} finished (rc=$RC) -> check the burn columns, then submit p6_depol_bb144_burn.sh"
exit $RC
