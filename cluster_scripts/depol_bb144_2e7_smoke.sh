#!/bin/bash
#PBS -N depol_bb144_2e7_smoke
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=32gb
#PBS -l walltime=00:30:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/depol_bb144_2e7_smoke.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/depol_bb144_2e7_smoke.err
#
# WHAT. Smoke for the two production jobs depol_bb144_2e7_p1e2.sh (p=1e-2) and
# depol_bb144_2e7_lowp.sh (p=1e-3, 1e-4): BB144 under depolarizing noise, OBF and
# Z+H, short/long, N=2e5 (production 2e7 / 100), one seed (base seed 1000), all
# twelve production conditions in one invocation. Validates the environment, the
# depolarizing sampler path of lowp_audit.py, the reference lookup against
# cluster_results/subset_sampling_depol.csv, the resume behaviour, and measures
# wall_per_chain for the production walltimes. The two production jobs split the
# conditions by p so that two nodes run concurrently (2026-10-30, author request).
# WALLTIME. Smoke cap 00:30:00. Measured at N=2e6 (job 264999): Z+H 229 s (1e-2) and
# 166 s (1e-3, 1e-4), OBF 23-28 s per chain -> smoke chains of 2-25 s, one wave.
# MEMORY. mem=32gb; int32 trace 0.8 MB/worker, BP-OSD cache (cache_cap=200000) ~0.2 GB.
# RESUME. Fresh mktemp stem; the second identical invocation must say "nothing to do"
# and change none of the four CSVs. Never reuse an older summary stem.
# Never run this script or its driver on the login node.
set -uo pipefail
: "${PBS_JOBID:?submit this script with qsub}"
WORKDIR=/home/user/workspace/RIMC
cd "$WORKDIR" || exit 1
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96 PYTHONUTF8=1
git pull --ff-only origin main || true
TMP=$(mktemp -d) || exit 1
OUT="$TMP/depol_bb144_2e7_smoke.csv"
# Remove only this smoke's four outputs; leave any unexpected files for inspection.
trap 'rm -f -- "$OUT" "${OUT%.csv}_chains.csv" "${OUT%.csv}_conv.csv" "${OUT%.csv}_seeds.csv"; rmdir -- "$TMP"' EXIT
conda run -n py311 --no-capture-output python cluster_scripts/lowp_audit.py --noise depolarizing \
    --codes bb144 --p-list 1e-2,1e-3,1e-4 --methods OBF,Z+H --inits short,long \
    --mcmc-target 200000 --n-seeds 1 --n-cores 96 --cache-cap 200000 \
    --out-csv "$OUT"
RC=$?
echo "[smoke] first run rc=$RC"
if [ "$RC" -eq 0 ]; then
    BEFORE=$(sha256sum "$OUT" "${OUT%.csv}_chains.csv" "${OUT%.csv}_conv.csv" "${OUT%.csv}_seeds.csv") || RC=1
    SECOND=$(conda run -n py311 --no-capture-output python cluster_scripts/lowp_audit.py --noise depolarizing \
    --codes bb144 --p-list 1e-2,1e-3,1e-4 --methods OBF,Z+H --inits short,long \
    --mcmc-target 200000 --n-seeds 1 --n-cores 96 --cache-cap 200000 \
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
conda run -n py311 --no-capture-output python - "$OUT" <<'EOF' || RC=1
import csv
import itertools
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
ps = (1e-2, 1e-3, 1e-4)
expected = set(itertools.product(("bb144",), ps, ("short", "long"), ("OBF", "Z+H")))
with path.open(newline="") as f:
    rows = list(csv.DictReader(f))
keys = {(r["code"], float(r["p"]), r["init"], r["method"]) for r in rows}
assert keys == expected and len(rows) == len(expected), "missing/duplicate conditions"
wall = {}
for r in rows:
    assert int(r["n_seeds"]) == 1
    assert int(r["n_steps"]) == 200000
    source = r["truth_source"]
    assert source.startswith("subset_sampling:"), f"missing reference: {r}"
    assert math.isfinite(float(r["truth_ew"]))
    w = float(r["wall_per_chain"])
    assert math.isfinite(w) and w > 0
    wall[(float(r["p"]), r["init"], r["method"])] = w
    print(f"bb144 p={float(r['p']):.0e} {r['init']:5s} {r['method']:4s} "
          f"wall_per_chain={w:.3f}s truth_source={source} production_chain={w * 100:.1f}s")
for suffix, count in (("_chains", len(expected) * 5),
                      ("_conv", len(expected) * 5), ("_seeds", len(expected))):
    with path.with_name(path.stem + suffix + ".csv").open(newline="") as f:
        assert sum(1 for _ in csv.DictReader(f)) == count, f"bad companion: {suffix}"
print("extrapolation per job: sum(wall_per_chain over its conditions) x100 x50 /96; "
      "longest chain x100 is the makespan floor")
for job, job_ps in (("depol_bb144_2e7_p1e2", (1e-2,)), ("depol_bb144_2e7_lowp", (1e-3, 1e-4))):
    ws = [w for (p, _i, _m), w in wall.items() if p in job_ps]
    node_s = sum(ws) * 100 * 50 / 96
    makespan_s = max(node_s, max(ws) * 100)
    seconds = math.ceil(2 * makespan_s)
    print(f"{job}: node-hours = {node_s / 3600:.3f}, wall-hours (floor) = {makespan_s / 3600:.3f}, "
          f"suggested walltime (2x): {seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}")
EOF
ntfy_done "RIMC depol_bb144_2e7 SMOKE done" "job ${PBS_JOBID}; read the two extrapolations and set the production walltimes before submission; rc=$RC"
exit $RC
