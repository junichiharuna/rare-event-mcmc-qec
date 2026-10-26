#!/bin/bash
#PBS -N p6_depol_burn
#PBS -l select=1:ncpus=96:mem=200gb:vnode=node042
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/p6_depol_bb144_burn.out

# RERUN of the June job 261711 (depolarizing bb144 low-p audit, source of
# tab:depollowp) with the tau-based BURN-IN statistics added
# (docs/results_burnin_thinning_2026-08-14.md: c=5 primary, c=10 sensitivity,
# cross-chain SE, no thinning). The driver is cluster_scripts/depol_audit.py -- the
# SAME driver as depol_audit.sh, invoked through its --codes/--p-list override -- so
# the burn columns added there (commit a4b56bc) are already in place and nothing
# further changed for this job. Every run parameter is IDENTICAL to June: bb144 at
# p=1e-3/3e-3, --truth-target 500, --truth-mmax 2e9, --mcmc-target 200000,
# --n-seeds 8, seeds 1000+s.
#
# Output goes to a NEW CSV (cluster_results/p6_depol_bb144_burn.csv); the June
# cluster_results/p6_depol_bb144.csv is NEVER touched -- it stays the regression
# baseline. The June V0 columns are exactly reproducible: the same check on
# zh_rerun_burn / depol_audit_burn (jobs 263631/263632) came back byte-identical on
# all 18/19 columns, truth included, so direct_mc.run_condition is deterministic.
#
# NO RESUME (unchanged from June): depol_audit.py writes all output at the end, so a
# walltime kill loses both phases and a re-qsub restarts from scratch, truth included.
# The truth phase dominates: the 2e9-trial cap binds at p=1e-3 (P_L = 2.55e-8 gave only
# F=51 in June), and p=3e-3 needs ~7.7e8 trials for its 500 faulty -- ~2.8e9 BP-OSD
# depolarizing trials in total. Measured node throughput from job 263632 today is
# ~1.7e5 such trials/s, i.e. a ~4.5 h forecast. June fitted inside a 12 h wall; this
# asks for 24 h (~5x the forecast, 2x June's request) purely because there is no
# resume and the downside of a kill is losing the whole run.
# vnode is pinned because most nodes are state-unknown and the scheduler has
# repeatedly picked the broken node007.

set -uo pipefail
WORKDIR=/home/user/workspace/RIMC
cd "$WORKDIR" || exit 1
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96
git pull --ff-only origin main || true

OUT=cluster_results/p6_depol_bb144_burn.csv
conda run -n py311 --no-capture-output python cluster_scripts/depol_audit.py \
  --codes bb144 --p-list 1e-3,3e-3 \
  --truth-target 500 --truth-mmax 2e9 \
  --mcmc-target 200000 --n-seeds 8 --n-cores 96 \
  --out-csv "$OUT"
RC=$?

echo "=== result CSV ==="
cat "$OUT"

if [ "$RC" -eq 0 ]; then
    git add "$OUT" "${OUT%.csv}_conv.csv"
    git commit -m "data: depol bb144 low-p audit with burn-in stats (job ${PBS_JOBID})"
    for _try in 1 2 3; do
        git pull --rebase origin main && git push origin HEAD:main && break
        echo "[publish] push attempt $_try failed — retrying"
        sleep 20
    done
fi

ntfy_done "RIMC p6_depol_bb144_burn done" \
    "job ${PBS_JOBID} finished (rc=$RC). Run: git pull, then swap tab:depollowp to the burned columns"
exit $RC
