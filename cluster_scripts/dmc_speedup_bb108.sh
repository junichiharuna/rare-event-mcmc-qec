#!/bin/bash
#PBS -N dmc_bb108
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=64gb
#PBS -l walltime=06:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/dmc_speedup_bb108.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/dmc_speedup_bb108.err
#
# BB[[108,8,10]] (bb_l9m6) TOP-UP for the time-to-accuracy benchmark of tab:dmcspeedup /
# tab:obfspeedup.  The published table covers 7 codes at p in {1e-1,1e-3,1e-5} over 10
# independent repeat runs (RUN_ID 4..13, cluster_scripts/dmc_speedup10.sh); this job adds
# the EIGHTH code and nothing else, one job per RUN_ID:
#     qsub -v RUN_ID=<4..13> cluster_scripts/dmc_speedup_bb108.sh
#
# WHY A SEPARATE FILE, AND WHY IT MERGES.  The 7-code runs are finished and committed as
# cluster_results/dmc_speedup_run{4..13}.csv; they must not be touched.  This job writes
# cluster_results/dmc_speedup_run${RUN_ID}_bb108.csv instead, and
# compare/analyze_dmc_speedup.py::group_paths_by_run groups the two files of the same
# RUN_ID into ONE run (they share that run's seed stream), so the per-cell run-to-run
# scatter and the caption's run count stay correct.  Analyse with
#     python compare/analyze_dmc_speedup.py cluster_results/dmc_speedup_run{4..13}.csv \
#                                           cluster_results/dmc_speedup_run{4..13}_bb108.csv
#
# SEED EQUIVALENCE.  dmc_speedup.py derives every seed from --run-id alone: advantage
# chains get 7000+s+1e6*(RUN_ID-1), direct-MC cutoff chunks 90000+c+1e6*(RUN_ID-1).  With
# --n-seeds 50 and the same --run-id, bb_l9m6's streams come from the same disjoint block
# as the other seven codes in that run, so its rows are statistically equivalent to theirs.
# Do NOT change --n-seeds: analyze_dmc_speedup.n_seeds_of() rejects the whole analysis if
# any cell has a seed count differing from the rest.
#
# PROTOCOL = the published base caps (dmc_speedup10.sh phase A), not the bb144 extension:
# OBF checkpoints 1e3..3.36e7 (16, trap group), Z+H 1e3..5.12e5 (10), direct-MC cutoff
# 96 chunks x 300 s per p.  The bb144-only phase-B extension is deliberately NOT applied
# here; if the finished table marks the bb_l9m6 cells with footnote (c) -- Z+H itself hit
# its length cap -- run cluster_scripts/dmc_speedup_bb108_zh_topup.sh afterwards, which
# appends to this same CSV via the worker's resume logic.
#
# REFERENCE.  No external truth file is needed or used: the dmc_cutoff phase in this job IS
# the reference.  bb_l9m6 has no rows in cluster_results/direct_mc_ew.csv (and no GRID_CODE
# alias), and cluster_results/subset_sampling.csv covers it only at p=1e-2..1e-4, which the
# analysis never reads.  Expect truth_source=directmc_cutoff at p=1e-1 and p=1e-3 (p_L is
# 4.2e-8 at 1e-3 per subset sampling, and bb144 at p_L=5.8e-8 collected ~170 faulty draws
# per run, comfortably over MIN_FAULTY_FOR_TRUTH=30 -- CHECK THIS in run 4's output before
# submitting 5..13) and the zh fallback at p=1e-5, exactly as for bb72 and bb144.
#
# WALLTIME 06:00:00, from run 13's measured per-step rates and lowp_audit_2e6_gr.csv:
#   p=1e-3/1e-5 are n-independent for BP-OSD (bb_l9m6 OBF 25.5 s / Z+H 113 s per 2e6 steps,
#   vs bb144's 26.0 / 119 and bb72's 25.1 / 107), giving ~12 core-h per p for OBF and
#   ~0.8 core-h per p for Z+H.  p=1e-1 is the expensive cell and has no cluster measurement
#   for bb_l9m6; a local single-core probe (this driver, p=1e-1, L=4000) puts it at 1.76x
#   bb72 per step for OBF and 1.86x for Z+H, and the same probe reproduces the cluster's
#   bb144/bb72 ratio of 2.36 as 2.39, so the scaling is trustworthy: ~156 core-h for OBF
#   and ~11 for Z+H.  Plus 24 core-h of direct-MC cutoff.  Total ~214 core-h = 2.2 h on
#   96 cores, ~2.9 h makespan with the tail; if p=1e-1 is really as expensive per step as
#   bb144's it is ~267 core-h and ~3.2 h.  RUN THE SMOKE FIRST
#   (dmc_speedup_bb108_smoke.sh) -- it measures the p=1e-1 rate on the node and prints
#   this projection, so the walltime can be raised before run 4 if the rate disagrees.
#
# SATURATION.  One flat 96-core pool: 3 p x (96 cutoff chunks + 50 seeds x 16 OBF lengths
# + 50 seeds x 10 Z+H lengths) = 4188 tasks, ~44 waves.  Within each p the worker emits the
# checkpoints seed-major and length-ascending, so the 50 long OBF chains (~1.6 h each at the
# probed rate, the only tasks comparable to the makespan) are dispatched in the first moments
# of their block and finish inside it rather than forming a tail.
# MEMORY.  cache_cap=200000 per worker (~0.2 GB) x 96 and no traces (record_trace=False);
# 64gb is ample and far under the 340gb cap.
# RESUME.  A walltime kill keeps every completed row; resubmit the SAME RUN_ID and
# _load_done skips what is on disk.  An incomplete file must NOT be fed to the analysis --
# the completeness check below labels the commit and the notification accordingly.
set -euo pipefail
WORKDIR=/home/user/workspace/RIMC
cd "${PBS_O_WORKDIR:-$WORKDIR}"
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96
git pull --ff-only origin main || true

RUN_ID="${RUN_ID:?submit with qsub -v RUN_ID=<4..13>}"
OUT="cluster_results/dmc_speedup_run${RUN_ID}_bb108.csv"

echo "=== bb_l9m6 base caps, 50 seeds, run ${RUN_ID}: start $(date -Is) ==="
SECONDS=0
conda run -n py311 python cluster_scripts/dmc_speedup.py \
  --specs bb_l9m6 --p-list 1e-1,1e-3,1e-5 \
  --dmc-chunks 96 --dmc-budget 300 --n-seeds 50 --L0 1000 \
  --L-max 33554432 --ctrl-obf-L-max 512000 --kernel-L-max 512000 \
  --cache-cap 200000 --n-cores 96 --run-id "$RUN_ID" --out-csv "$OUT"
echo "=== end $(date -Is) after ${SECONDS}s ==="

# Completeness: every (p, method) cell must carry all 50 seeds at every checkpoint, else the
# merged analysis aborts in n_seeds_of().  Also report the faulty counts that decide whether
# p=1e-3 gets a direct-MC reference or falls back to the circular Z+H one.
# (|| echo: never let a hiccup in the check abort the commit of a multi-hour run)
STATUS=$(conda run -n py311 python cluster_scripts/check_dmc_speedup_complete.py "$OUT" || echo CHECK_FAILED)
echo "completeness: ${STATUS}"

git add "$OUT"
git commit -m "data: dmc_speedup BB[[108,8,10]] top-up run ${RUN_ID} (job ${PBS_JOBID})

Adds the eighth code (bb_l9m6, BP-OSD) to the time-to-accuracy benchmark at the published
protocol: p in {1e-1,1e-3,1e-5}, 50 seeds, OBF checkpoints 1e3..3.36e7 and Z+H 1e3..5.12e5,
direct-MC cutoff 96 chunks x 300 s per p.  Seeds come from --run-id ${RUN_ID} exactly as for
the seven codes in cluster_results/dmc_speedup_run${RUN_ID}.csv, which is left untouched;
compare/analyze_dmc_speedup.py groups the two files into that one run.

Status: ${STATUS}." || true
git pull --rebase origin main || true
git push origin main || true
ntfy_done "dmc_speedup bb108 run${RUN_ID} done" "job ${PBS_JOBID}, ${STATUS}. Locally: git pull, then rerun compare/analyze_dmc_speedup.py over run{4..13}.csv + run{4..13}_bb108.csv"
