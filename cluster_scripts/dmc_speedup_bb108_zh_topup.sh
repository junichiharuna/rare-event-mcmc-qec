#!/bin/bash
#PBS -N dmc_bb108_zh
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=64gb
#PBS -l walltime=03:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/dmc_speedup_bb108_zh.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/dmc_speedup_bb108_zh.err
#
# OPTIONAL follow-up to cluster_scripts/dmc_speedup_bb108.sh -- SUBMIT ONLY IF NEEDED.
#
# WHEN.  After the 10 base runs are merged and the table rebuilt, look at the BB[[108,8,10]]
# cells.  Footnote (c) means "Z+H itself hit its length cap", i.e. the entry is a ratio of
# budgets rather than a bound in either direction (that is what happened to bb144 at p=1e-3,
# and is why dmc_speedup10.sh grew its phase B).  If any bb_l9m6 cell carries (c), this job
# lifts the Z+H cap from 512,000 to 33,554,432 (10 -> 16 checkpoints) at p=1e-3 and 1e-5;
# if none does, do not run it.  The OBF cap is deliberately left at 33,554,432: raising it to
# 268,435,456 as bb144's phase B does costs ~163 core-h per run and only tightens a (b) floor
# that stays a floor either way, exactly as bb72's does.
#
# HOW IT MERGES.  Same OUT file and same --run-id as the base job, so dmc_speedup.py's
# _load_done resume skips every chain already on disk and only the six new checkpoints per
# seed are computed; the direct-MC cutoff chunks are not re-run.  Submit one job per RUN_ID:
#     qsub -v RUN_ID=<4..13> cluster_scripts/dmc_speedup_bb108_zh_topup.sh
# The base run for that RUN_ID must be COMPLETE and pulled first.
#
# WALLTIME 03:00:00.  Added work per run = (1000*(2^16-1) - 1000*(2^10-1)) = 6.45e7 steps
# per seed per p, x 50 seeds x 2 p, at bb_l9m6's measured Z+H rate of 113 s / 2e6 steps
# (cluster_results/lowp_audit_2e6_gr.csv, p=1e-3) = ~101 core-h = ~1.1 h on 96 cores.  The
# longest single task is the L=3.36e7 chain at ~0.5 h, so the tail is short.  ~2.7x headroom.
# SATURATION: 2 p x 50 seeds x 6 new checkpoints = 600 tasks in one flat 96-core pool.
set -euo pipefail
WORKDIR=/home/user/workspace/RIMC
cd "${PBS_O_WORKDIR:-$WORKDIR}"
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96
git pull --ff-only origin main || true

RUN_ID="${RUN_ID:?submit with qsub -v RUN_ID=<4..13>}"
OUT="cluster_results/dmc_speedup_run${RUN_ID}_bb108.csv"
[ -f "$OUT" ] || { echo "missing $OUT -- run dmc_speedup_bb108.sh for RUN_ID=$RUN_ID first" >&2; exit 1; }

echo "=== bb_l9m6 Z+H cap 512k -> 33.5M at p=1e-3,1e-5, run ${RUN_ID}: start $(date -Is) ==="
SECONDS=0
conda run -n py311 python cluster_scripts/dmc_speedup.py \
  --specs bb_l9m6 --p-list 1e-3,1e-5 \
  --dmc-chunks 96 --dmc-budget 300 --n-seeds 50 --L0 1000 \
  --L-max 33554432 --ctrl-obf-L-max 512000 --kernel-L-max 33554432 \
  --cache-cap 200000 --n-cores 96 --run-id "$RUN_ID" --out-csv "$OUT"
echo "=== end $(date -Is) after ${SECONDS}s ==="

# p=1e-1 keeps the base Z+H grid (10 checkpoints); 1e-3 and 1e-5 must now have 16.
STATUS_HIGH=$(conda run -n py311 python cluster_scripts/check_dmc_speedup_complete.py "$OUT" \
                --p-list 1e-1 --expect OBF:16,Z+H:10 || echo CHECK_FAILED)
STATUS_LOW=$(conda run -n py311 python cluster_scripts/check_dmc_speedup_complete.py "$OUT" \
                --p-list 1e-3,1e-5 --expect OBF:16,Z+H:16 || echo CHECK_FAILED)
STATUS="${STATUS_HIGH}/${STATUS_LOW}"
echo "completeness (p=1e-1 / p=1e-3,1e-5): ${STATUS}"

git add "$OUT"
git commit -m "data: dmc_speedup BB[[108,8,10]] Z+H cap top-up run ${RUN_ID} (job ${PBS_JOBID})

Extends the Z+H fixed-grid cap from 512,000 to 33,554,432 at p=1e-3 and p=1e-5 for
bb_l9m6, so those cells stop being budget-limited ratios (footnote (c) in tab:dmcspeedup).
Appended to the run-${RUN_ID} bb108 CSV through the worker's resume logic: same seeds, same
protocol, nothing recomputed.  The OBF cap stays at 33,554,432, as for bb72.

Status: ${STATUS}." || true
git pull --rebase origin main || true
git push origin main || true
ntfy_done "dmc_speedup bb108 Z+H top-up run${RUN_ID} done" "job ${PBS_JOBID}, ${STATUS}"
