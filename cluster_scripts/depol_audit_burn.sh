#!/bin/bash
#PBS -N depol_audit_burn
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=200gb:vnode=node012
#PBS -l walltime=12:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/depol_audit_burn.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/depol_audit_burn.err

# Single full-node job: RERUN of the June job 260849 (depolarizing-channel audit B2,
# source of Sec.6.3 + App E) with the tau-based BURN-IN statistics added
# (docs/results_burnin_thinning_2026-08-14.md: c=5 primary, c=10 sensitivity,
# cross-chain SE, no thinning). Every run parameter is IDENTICAL to June -- same
# conditions (toric d=3,5 + bb72 + bb144 at p=1e-2; note the June header comment
# claiming toric d=5/7/9 was already stale, conditions() and the June CSV are d=3,5),
# same seeds (1000+s), kernels, --mcmc-target 100000, --n-seeds 50, --truth-target
# 2000, --truth-mmax 4e7 -- so the existing full-trace columns (ew_mean / ew_std / tau)
# reproduce cluster_results/depol_audit.csv and the new ew_burn{5,10}* columns are the
# only addition.
#
# Output goes to a NEW CSV (cluster_results/depol_audit_burn.csv); the June
# cluster_results/depol_audit.csv is NEVER touched -- it stays the regression baseline.
# vnode is pinned (most nodes are state-unknown; the scheduler has repeatedly picked the
# broken node007) and deliberately DIFFERENT from zh_rerun_burn.sh's node042
# so the two independent jobs run concurrently.
# Walltime is June's 12h anchor, ~100x the June actual (~7 min); memory is June's 200gb.

WORKDIR=/home/user/workspace/RIMC
cd "$WORKDIR" || exit 1
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96

conda run -n py311 --no-capture-output python cluster_scripts/depol_audit.py \
    --mcmc-target 100000 --n-seeds 50 --truth-target 2000 --truth-mmax 4e7 \
    --n-cores 96 --out-csv $WORKDIR/cluster_results/depol_audit_burn.csv
RC=$?

echo "=== result CSV ==="
cat $WORKDIR/cluster_results/depol_audit_burn.csv

if [ "$RC" -eq 0 ]; then
    git add cluster_results/depol_audit_burn.csv cluster_results/depol_audit_burn_conv.csv
    git commit -m "data: depolarizing audit with burn-in stats (job ${PBS_JOBID})"
    # Retry loop: the zh_rerun_burn job pushes to the same branch concurrently,
    # so a single plain push can lose the race (June's scripts had one pusher).
    for _try in 1 2 3; do
        git pull --rebase origin main && git push origin HEAD:main && break
        echo "[publish] push attempt $_try failed — retrying"
        sleep 20
    done
fi

ntfy_done "RIMC depol_audit_burn done" \
    "job ${PBS_JOBID} finished (rc=$RC). Run: git pull, then swap Sec.6.3 / App E to the burned columns"
exit $RC
