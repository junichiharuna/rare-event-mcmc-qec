#!/bin/bash
#PBS -N depol_bb144_2e7_lowp
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=340gb
#PBS -l walltime=04:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/depol_bb144_2e7_lowp.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/depol_bb144_2e7_lowp.err
#
# WHAT. BB144 under depolarizing noise at p=1e-3 and 1e-4: OBF and Z+H, short/long,
# K=50, N=2e7 (ten times the benchmark length of tab:meanw / tab:lowp and of the 2e6
# depolarizing run depol_meanw_2e6.csv), default base seed 1000. Together with
# depol_bb144_2e7_p1e2.sh (p=1e-2, the other node) this supplies a tab:depol at one
# uniform budget of 2e7 steps, where the 2e6 run left one Z+H entry failing rule (iv)
# and one 3.3 sigma below the reference at p=1e-3 (2026-10-30).
# Same driver and protocol as depol_meanw_2e6.sh; only --mcmc-target changes.
# WALLTIME. Extrapolated from the N=2e6 run (job 264999): Z+H 166 s and OBF 23-25 s per
# chain at p=1e-3 and 1e-4 -> x10 = 1660 s / 250 s; 200 Z+H + 200 OBF chains = 106
# core-hours, 1.1 node-hours ideal, ~1.5 h with the third Z+H wave. 04:00:00 (~2.7x)
# unless the smoke (depol_bb144_2e7_smoke.sh) says otherwise; override with
# qsub -l walltime=. Rows appear only on condition completion: a wall kill loses the
# unfinished condition, and the two p are independent conditions (resume keeps a
# finished p).
# SATURATION. One flat Pool(96) over 400 chains, longest-expected-first; the 8 Z+H
# chains of the third wave run beside the 200 short OBF chains.
# MEMORY. int32 trace 2e7 x4 = 80 MB + ~0.2-0.4 GB BP-OSD cache (cache_cap=200000) per
# worker, ~50 GB steady state over 96 workers. The first submission (job 265965,
# mem=96gb) was killed by PBS at 101 GB eleven minutes in, when the 50 long-start OBF
# chains of p=1e-3 finished together and their post-processing (float64 running means,
# cumsums and checkpoint copies of the 2e7-step trace, ~1 GB transient per chain)
# overlapped. mem=340gb (the 350gb node, queue default) covers 96 simultaneous
# transients. tau runs on a thinned copy (stride 100 at N=2e7, <= 2e5 points).
# RESUME. New depol_bb144_2e7_lowp stem; the summary CSV alone decides completion.
# Never reuse an older CSV stem to extend length. A same-protocol resubmit skips
# finished conditions; an unfinished condition restarts all 50 chains.
# REFERENCE. cluster_results/subset_sampling_depol.csv (bb144 at p=1e-3, E[w|F] ~3.097,
# and p=1e-4, ~3.049) looked up at job time; the smoke checks the lookup.
# Never run this script or its driver on the login node.
set -uo pipefail
: "${PBS_JOBID:?submit this script with qsub}"
WORKDIR=/home/user/workspace/RIMC
cd "$WORKDIR" || exit 1
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96 PYTHONUTF8=1
git pull --ff-only origin main || true
STEM=cluster_results/depol_bb144_2e7_lowp
OUT="${STEM}.csv"
FAIL=0
conda run -n py311 --no-capture-output python cluster_scripts/lowp_audit.py --noise depolarizing \
    --codes bb144 --p-list 1e-3,1e-4 --methods OBF,Z+H --inits short,long \
    --mcmc-target 20000000 --n-seeds 50 --n-cores 96 --cache-cap 200000 \
    --out-csv "$OUT" || FAIL=1
if [ "$FAIL" -eq 0 ]; then
    git add "${STEM}.csv" "${STEM}_chains.csv" "${STEM}_conv.csv" "${STEM}_seeds.csv" || FAIL=1
    git diff --cached --quiet || git commit -m "data(depol_bb144_2e7_lowp): BB144 depolarizing p=1e-3 and 1e-4, OBF and Z+H, short/long, K=50 N=2e7 (job ${PBS_JOBID})" || FAIL=1
    # The p1e2 job may push to the same branch at about the same time: retry.
    for _try in 1 2 3; do
        { git pull --rebase origin main && git push origin HEAD:main; } && break
        echo "[publish] push attempt $_try failed - retrying"
        sleep 20
        [ "$_try" -eq 3 ] && FAIL=1
    done
fi
ntfy_done "RIMC depol_bb144_2e7_lowp done" "job ${PBS_JOBID}; Run: git pull (fail=$FAIL)"
exit $FAIL
