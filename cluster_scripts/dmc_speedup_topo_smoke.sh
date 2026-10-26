#!/bin/bash
#PBS -N dmc_speedup_topo_smoke
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=32gb
#PBS -l walltime=00:30:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/dmc_speedup_topo_smoke.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/dmc_speedup_topo_smoke.err
#
# WHAT. Smoke toric:5,toric:7,surface:5,surface:7 x p=1e-1,1e-3,1e-5,
# RUN_ID=99, one seed, 96 dMC chunks x10 s/cell; OBF/Z+H 10 checkpoints each,
# L0=1000, caps 512000 as production. Results stay under cluster_results/smoke/.
# WALLTIME. PLACEHOLDER until smoke job <id> for production; smoke cap 00:30:00.
# Print per-cell s/step and dMC trials/core-s; MCMC production core-s =sum of
# per-cell rates x1.023e6 x50. Add fixed 96 core-h dMC, divide by96, then 2x.
# Prior ~10 core-h MCMC +96 core-h dMC =~1.1-1.2 node-h/run; placeholder 03:00:00.
# SATURATION. One flat Pool(96), 1152 dMC +240 MCMC =1392 smoke tasks;
# hardest-first imap_unordered. Production 1152 dMC +12000 MCMC, short MCMC tail.
# MEMORY. mem=32gb; driver default cache_cap=200000, record_trace=False, n<=98.
# Bounded caches and streamed rows; production mem=96gb, all below 340gb.
# RESUME. Delete only the dedicated run99 smoke CSV first. The second identical
# invocation must report 0 pending and add 0 rows. Never commit smoke results or
# point this smoke at a production stem; resubmit production with SAME RUN_ID.
# REFERENCE. Check COMPLETE for OBF:10,Z+H:10 at one seed; also assert all four
# codes x3 p and all 96 dMC chunks exist. Analysis uses cutoff truth at p=1e-1,
# subset_sampling.csv at p=1e-3 and synthesized p=1e-5 (never synthesize p=1e-1).
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
OUT=cluster_results/smoke/dmc_speedup_run99_topo.csv
mkdir -p cluster_results/smoke || exit 1
rm -f -- "$OUT" || exit 1
conda run -n py311 --no-capture-output python cluster_scripts/dmc_speedup.py \
    --specs toric:5,toric:7,surface:5,surface:7 --p-list 1e-1,1e-3,1e-5 \
    --dmc-chunks 96 --dmc-budget 10 --n-seeds 1 --L0 1000 \
    --L-max 33554432 --ctrl-obf-L-max 512000 --kernel-L-max 512000 \
    --n-cores 96 --run-id 99 --out-csv "$OUT"
RC=$?
echo "[smoke] first run rc=$RC"
STATUS=$(conda run -n py311 --no-capture-output python cluster_scripts/check_dmc_speedup_complete.py "$OUT" --n-seeds 1 --expect OBF:10,Z+H:10 || echo CHECK_FAILED)
echo "completeness: $STATUS"
[ "$STATUS" = COMPLETE ] || RC=1
if [ "$RC" -eq 0 ]; then
    BEFORE=$(wc -l < "$OUT") || RC=1
    SECOND=$(conda run -n py311 --no-capture-output python cluster_scripts/dmc_speedup.py \
    --specs toric:5,toric:7,surface:5,surface:7 --p-list 1e-1,1e-3,1e-5 \
    --dmc-chunks 96 --dmc-budget 10 --n-seeds 1 --L0 1000 \
    --L-max 33554432 --ctrl-obf-L-max 512000 --kernel-L-max 512000 \
    --n-cores 96 --run-id 99 --out-csv "$OUT" 2>&1)
    RC2=$?
    printf '%s\n' "$SECOND"
    [ "$RC2" -eq 0 ] || RC=1
    if ! grep -q " 0 pending" <<< "$SECOND"; then
        echo "[smoke] FAIL: resume did not report 0 pending"
        RC=1
    fi
    AFTER=$(wc -l < "$OUT") || RC=1
    if [ "$BEFORE" = "$AFTER" ]; then
        echo "[smoke] resume: 0 added rows"
    else
        echo "[smoke] FAIL: resume changed row count"
        RC=1
    fi
fi
conda run -n py311 --no-capture-output python - "$OUT" <<'EOF' || RC=1
import collections
import csv
import itertools
import math
import sys

with open(sys.argv[1], newline="") as f:
    rows = list(csv.DictReader(f))
codes = ("toric:5", "toric:7", "surface:5", "surface:7")
ps = (1e-1, 1e-3, 1e-5)
lengths = {1000 * 2**i for i in range(10)}
expected_adv = set(itertools.product(codes, ps, ("OBF", "Z+H")))
expected_dmc = set(itertools.product(codes, ps))
adv, dmc = collections.defaultdict(list), collections.defaultdict(list)
for r in rows:
    if r["phase"] == "advantage":
        adv[(r["code"], float(r["p"]), r["method"])].append(r)
    else:
        assert r["phase"] == "dmc_cutoff", f"unexpected phase: {r['phase']}"
        dmc[(r["code"], float(r["p"]))].append(r)
assert set(adv) == expected_adv and set(dmc) == expected_dmc, "missing cells"
core_s = 0.0
max_chain_s = 0.0
for key, group in sorted(adv.items()):
    assert len(group) == 10 and {int(r["length"]) for r in group} == lengths
    assert {int(r["seed"]) for r in group} == {7000 + 1_000_000 * 98}
    longest = next(r for r in group if int(r["length"]) == 512000)
    rate = float(longest["wall"]) / 512000
    assert math.isfinite(rate) and rate > 0
    seconds = rate * sum(lengths) * 50
    core_s += seconds
    max_chain_s = max(max_chain_s, rate * 512000)
    print(f"{key[0]:10s} p={key[1]:.0e} {key[2]:4s}: {rate:.6e} s/step; "
          f"production core-hours={seconds / 3600:.4f}")
for key, group in sorted(dmc.items()):
    assert len(group) == 96
    assert {int(r["seed"]) for r in group} == set(range(90000 + 1_000_000 * 98,
                                                       90096 + 1_000_000 * 98))
    assert all(int(r["n_cores"]) == 1 for r in group)
    seconds = sum(float(r["wall"]) for r in group)
    trials = sum(int(r["n_trials"]) for r in group)
    assert math.isfinite(seconds) and seconds > 0 and trials > 0
    print(f"{key[0]:10s} p={key[1]:.0e} directmc: {trials / seconds:.3f} trials/core-s")
print(f"estimated production MCMC core-hours = {core_s / 3600:.4f}")
core_s += 96 * 3600  # 12 cells x96 chunks x300 seconds
node_s = core_s / 96
print("production dMC core-hours = 96.0000")
print(f"estimated production node-hours per run = {node_s / 3600:.4f}")
seconds = math.ceil(2 * max(node_s, max_chain_s, 300))
print(f"suggested walltime (2x): {seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}")
EOF
ntfy_done "RIMC dmc_speedup_topo SMOKE done" "job ${PBS_JOBID}; $STATUS; read extrapolation and set production walltime; rc=$RC"
exit $RC
