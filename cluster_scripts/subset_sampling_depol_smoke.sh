#!/bin/bash
#PBS -N ss_depol_smoke
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=64gb
#PBS -l walltime=00:30:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/ss_depol_smoke.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/ss_depol_smoke.err
# Smoke for subset_sampling_depol.sh: the two decoder paths (toric:5 MWPM fast path,
# bb72 BP-OSD generic path) under depolarizing noise at a 1/50 budget, plus bb144 and
# hgp_hamming3 (BP-OSD) at the same budget so the depolarizing w<=2 theorem assert runs
# on the other BP-OSD codes before production. Prints the per-
# condition wall so the production walltime can be set (bit-flip reference: bb144 24 s,
# bb_l9m6 16 s per p on this pipeline; depolarizing doubles the decodes and triples the
# enumerated classes, so expect 2-4x). Writes under mktemp; no git commit.
set -uo pipefail
WORKDIR=/home/user/workspace/RIMC
cd "$WORKDIR" || exit 1
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96
git pull --ff-only origin main || true
TMP=$(mktemp -d) || exit 1
OUT="$TMP/subset_sampling_depol_smoke"
conda run -n py311 --no-capture-output python cluster_scripts/subset_sampling.py \
    --noise depolarizing --codes toric:5,bb72:0,bb144:0,hgp_hamming3:0 --p-list 1e-2,1e-4 \
    --enum-cap 100000 --n-max 100000 --target-fail 20 --pilot-n 2000 \
    --n-cores 96 --cache-cap 200000 --out-prefix "$OUT"
RC=$?
echo "=== smoke rows ==="; cat "${OUT}.csv"; cat "${OUT}_fw.csv" | head -30
rm -rf "$TMP"
ntfy_done "RIMC ss_depol SMOKE done" "job ${PBS_JOBID} rc=$RC. Read wall_clock, then submit subset_sampling_depol.sh"
exit $RC
