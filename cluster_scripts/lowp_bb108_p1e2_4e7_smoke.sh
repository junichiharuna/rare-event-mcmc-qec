#!/bin/bash
#PBS -N lowp_bb108_p1e2_4e7_smoke
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=32gb
#PBS -l walltime=00:30:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/lowp_bb108_p1e2_4e7_smoke.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/lowp_bb108_p1e2_4e7_smoke.err
#
# WHAT. Smoke BB108 p=1e-2 Z+H short/long, N=400000 (production 4e7 /100),
# one seed (base seed 1000), to validate environment, reference and per-chain cost.
# WALLTIME. PLACEHOLDER until smoke job <id> for production; smoke cap 00:30:00.
# Production prior 314 s /2e6 x20 ~6300 s =1.75 h/chain, placeholder 05:00:00
# (~2.8x). Print wall_per_chain x100, node-hours and 2x longest-chain walltime.
# SATURATION. One flat Pool(96); two smoke chains intentionally under one wave.
# Production 2 inits x48 seeds =96 chains, exactly one wave; no serial init pools.
# MEMORY. mem=32gb; smoke int32 trace 1.6 MB/worker, cache_cap=200000 ~0.2 GB.
# Production 160 MB trace +40 MB proxy +~0.2 GB cache ~0.4 GB/worker x96 ~40 GB,
# mem=96gb. Traces remain in workers, bounded tau processing.
# RESUME. Fresh mktemp stem, never reuse older summary (summary alone decides skip).
# Second identical invocation must say nothing to do and change none of four CSVs.
# Production rows appear only on condition completion; wall kill can lose all work.
# REFERENCE. Require finite truth_ew and subset_sampling: source from the existing
# BB108 p=1e-2 reference (~4.961) in cluster_results/subset_sampling.csv. No commit.
#
# Submit only after all three smokes pass and measured 2x walltimes are set.
# Spec section 4: <=4 nodes; longest chains first, B: topo 4->7->10->13,
# C: topo 5->8->11, D: topo 6->9->12, then A: BB108->k1.
# Chain successors with qsub -W depend=afterany:<previous-jobid>; git pull first.
# Never run this script or its driver on the login node.
set -uo pipefail
: "${PBS_JOBID:?submit this script with qsub}"
WORKDIR=/home/user/workspace/RIMC
cd "$WORKDIR" || exit 1
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96 PYTHONUTF8=1
git pull --ff-only origin main || true
TMP=$(mktemp -d) || exit 1
OUT="$TMP/lowp_bb108_p1e2_4e7_smoke.csv"
# Remove only this smoke's four outputs; leave any unexpected files for inspection.
trap 'rm -f -- "$OUT" "${OUT%.csv}_chains.csv" "${OUT%.csv}_conv.csv" "${OUT%.csv}_seeds.csv"; rmdir -- "$TMP"' EXIT
conda run -n py311 --no-capture-output python cluster_scripts/lowp_audit.py \
    --codes bb_l9m6 --p-list 1e-2 --methods Z+H --inits short,long \
    --mcmc-target 400000 --n-seeds 1 --n-cores 96 --cache-cap 200000 \
    --out-csv "$OUT"
RC=$?
echo "[smoke] first run rc=$RC"
if [ "$RC" -eq 0 ]; then
    BEFORE=$(sha256sum "$OUT" "${OUT%.csv}_chains.csv" "${OUT%.csv}_conv.csv" "${OUT%.csv}_seeds.csv") || RC=1
    SECOND=$(conda run -n py311 --no-capture-output python cluster_scripts/lowp_audit.py \
    --codes bb_l9m6 --p-list 1e-2 --methods Z+H --inits short,long \
    --mcmc-target 400000 --n-seeds 1 --n-cores 96 --cache-cap 200000 \
    --out-csv "$OUT" 2>&1)
    RC2=$?
    printf '%s\n' "$SECOND"
    [ "$RC2" -eq 0 ] || RC=1
    if ! grep -q "nothing to do" <<< "$SECOND"; then
        echo "[smoke] FAIL: resume did not report nothing to do"
        RC=1
    fi
    AFTER=$(sha256sum "$OUT" "${OUT%.csv}_chains.csv" "${OUT%.csv}_conv.csv" "${OUT%.csv}_seeds.csv") || RC=1
    if [ "$BEFORE" = "$AFTER" ]; then
        echo "[smoke] resume: all four CSVs unchanged (0 added rows)"
    else
        echo "[smoke] FAIL: resume changed output"
        RC=1
    fi
fi
conda run -n py311 --no-capture-output python - "$OUT" "bb" <<'EOF' || RC=1
import csv
import itertools
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
is_k1 = sys.argv[2] == "k1"
codes = ("surface3", "concat_steane2") if is_k1 else ("bb_l9m6",)
ps = (1e-2, 1e-3, 1e-4) if is_k1 else (1e-2,)
methods = ("OBF", "Z+H") if is_k1 else ("Z+H",)
inits = ("uniform",) if is_k1 else ("short", "long")
expected = set(itertools.product(codes, ps, inits, methods))
with path.open(newline="") as f:
    rows = list(csv.DictReader(f))
keys = {(r["code"], float(r["p"]), r["init"], r["method"]) for r in rows}
assert keys == expected and len(rows) == len(expected), "missing/duplicate conditions"
walls = []
for r in rows:
    assert int(r["n_seeds"]) == 1
    assert int(r["n_steps"]) == (20000 if is_k1 else 400000)
    source = r["truth_source"]
    assert source.startswith("subset_sampling:"), f"missing reference: {r}"
    if is_k1:
        assert source == {"surface3": "subset_sampling:surface/3",
                          "concat_steane2": "subset_sampling:concat_steane/2"}[r["code"]]
    assert math.isfinite(float(r["truth_ew"]))
    wall = float(r["wall_per_chain"])
    assert math.isfinite(wall) and wall > 0
    walls.append(wall)
    print(f"{r['code']:14s} p={float(r['p']):.0e} {r['init']:7s} {r['method']:4s} "
          f"wall_per_chain={wall:.3f}s truth_source={source} production_chain={wall * 100:.1f}s")
for suffix, count in (("_chains", len(expected) * 5),
                      ("_conv", len(expected) * 5), ("_seeds", len(expected))):
    with path.with_name(path.stem + suffix + ".csv").open(newline="") as f:
        assert sum(1 for _ in csv.DictReader(f)) == count, f"bad companion: {suffix}"
if is_k1:
    node_s = sum(walls) * 100 * 50 / 96
    print("extrapolation: sum(wall_per_chain) x100 x50 /96")
else:
    node_s = sum(walls) * 100 * 48 / 96
    print("extrapolation: 2 init x48 seeds /96 cores; longest wall_per_chain x100 sets makespan")
makespan_s = max(node_s, max(walls) * 100)
print(f"estimated production node-hours = {node_s / 3600:.4f}")
print(f"estimated production wall-hours (longest-chain floor) = {makespan_s / 3600:.4f}")
seconds = math.ceil(2 * makespan_s)
print(f"suggested walltime (2x): {seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}")
EOF
ntfy_done "RIMC lowp_bb108_p1e2_4e7 SMOKE done" "job ${PBS_JOBID}; read extrapolation and set production walltime before submission; rc=$RC"
exit $RC
