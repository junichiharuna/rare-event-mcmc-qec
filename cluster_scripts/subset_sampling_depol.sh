#!/bin/bash
#PBS -N ss_depol
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=64gb
#PBS -l walltime=02:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/ss_depol.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/ss_depol.err
# Weight-stratified reference under DEPOLARIZING noise for the ten codes of tab:meanw at
# p = 1e-2, 1e-3, 1e-4 (spec 2026-10-07 §2). Same budget as the bit-flip reference
# (enum-cap 5e6, n-max 5e6, target-fail 1000). f_w of the BP-OSD codes depends on the
# prior p -> one run per p (12 conditions); MWPM codes synthesize the grid from one run.
# WALLTIME: MEASURED by the smoke (job 264993, 2026-10-08) at 1/50 of the budget: 1.6-7 s
# per condition (bb144 slowest), 24 s in total for 8 conditions. The production budget is
# up to 50x more samples per stratum -> ~2-6 min per BP-OSD condition, 18 conditions
# -> well under 1 h. walltime 02:00:00 ~ 2-4x.
# SATURATION: run_code splits every stratum into >= 4*96 chunks, so the 96-core pool
# is full within each condition; conditions are serial (cheapest first).
# RESUME: read-modify-write per condition on the f_w table; resubmit to continue.
set -uo pipefail
WORKDIR=/home/user/workspace/RIMC
cd "$WORKDIR" || exit 1
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96
git pull --ff-only origin main || true
PREFIX=cluster_results/subset_sampling_depol
conda run -n py311 --no-capture-output python cluster_scripts/subset_sampling.py \
    --noise depolarizing \
    --enum-cap 5000000 --n-max 5000000 --target-fail 1000 --pilot-n 20000 \
    --n-cores 96 --cache-cap 200000 --out-prefix "$WORKDIR/$PREFIX"
FAIL=$?
if [ "$FAIL" -eq 0 ]; then
    git add "${PREFIX}.csv" "${PREFIX}_fw.csv" || FAIL=1
    git commit -m "data: weight-stratified reference under depolarizing noise, ten codes, p=1e-2..1e-4 (job ${PBS_JOBID})" || FAIL=1
    for _try in 1 2 3; do
        { git pull --rebase origin main && git push origin HEAD:main; } && break
        echo "[publish] push attempt $_try failed - retrying"; sleep 20
        [ "$_try" -eq 3 ] && FAIL=1
    done
fi
ntfy_done "RIMC ss_depol done" "job ${PBS_JOBID} finished (fail=$FAIL). Run: git pull"
exit $FAIL
