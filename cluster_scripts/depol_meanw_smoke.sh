#!/bin/bash
#PBS -N depol_meanw_smoke
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=128gb
#PBS -l walltime=00:30:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/depol_meanw_smoke.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/depol_meanw_smoke.err
# Smoke for depol_meanw_2e6.sh: every production condition (10 codes x p=1e-4, plus
# bb144/bb_l9m6 x 1e-2,1e-3; OBF + Z+H; short/long) at 1 seed and N=2e4 (1/100). It
# validates the depolarizing sampler path of lowp_audit.py, the truth lookup against
# subset_sampling_depol.csv (truth_source=missing is a PASS if that job has not landed),
# and gives wall_per_chain for the production walltime. 48 conditions x 1 seed = 48
# tasks: under one wave ON PURPOSE (a smoke measures per-chain cost).
set -uo pipefail
WORKDIR=/home/user/workspace/RIMC
cd "$WORKDIR" || exit 1
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96
git pull --ff-only origin main || true
TMP=$(mktemp -d) || exit 1
OUT="$TMP/depol_meanw_smoke.csv"
conda run -n py311 --no-capture-output python cluster_scripts/lowp_audit.py --noise depolarizing \
    --codes toric5,toric7,surface5,surface7,ut3x11,ut3x15,hgp_hamming3,bb72,bb144,bb_l9m6 \
    --p-list 1e-4 --methods OBF,Z+H --inits short,long \
    --mcmc-target 20000 --n-seeds 1 --n-cores 96 --cache-cap 200000 --out-csv "$OUT"
RC=$?
conda run -n py311 --no-capture-output python cluster_scripts/lowp_audit.py --noise depolarizing \
    --codes bb144,bb_l9m6 --p-list 1e-2,1e-3 --methods OBF,Z+H --inits short,long \
    --mcmc-target 20000 --n-seeds 1 --n-cores 96 --cache-cap 200000 --out-csv "$OUT"
RC=$(( RC + $? ))
echo "=== wall_per_chain by condition (s, at N=2e4; x100 for production) ==="
conda run -n py311 --no-capture-output python - "$OUT" <<'EOF' || RC=1
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    print(f"{r['code']:14s} p={float(r['p']):.0e} {r['init']:7s} {r['method']:4s} "
          f"wall={float(r['wall_per_chain']):8.2f}s truth={r['truth_source']}")
EOF
rm -rf "$TMP"
ntfy_done "RIMC depol_meanw SMOKE done" "job ${PBS_JOBID} rc=$RC. Sum wall x100 x50 / 96 -> walltime of depol_meanw_2e6.sh"
exit $RC
