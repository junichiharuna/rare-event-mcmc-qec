#!/bin/bash
#PBS -N kpt_smoke
#PBS -q workq
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=00:30:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/kpt_smoke.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/kpt_smoke.err
# Smoke test of cluster_scripts/kernel_param_tta.py before the production sweeps.
# VALIDATES: (i) both decoder types run end to end (bb144 = BP-OSD, rect:3x11 = MWPM);
# (ii) the pc=0 (0.0,2), pc=0.3 (0.3,2) and m=6 (0.3,6) arms; (iii) resume dedup (a second
# identical invocation must report "0 pending"); (iv) cpu/wall ~ 1 on every chain (no
# contention, so the wall-time measurements of the production job are meaningful).
# EXTRAPOLATION: s/step is read from the longest rows per (code, arm); production steps
# per run = 3 p x 10 arms x 50 seeds x 4,095,000 per code (sum of the doubling grid
# 1000..2,048,000); core-seconds = steps x s/step of that code (bb144 rate for bb_l9m6);
# node-hours = core-seconds / 96 / 3600. Use the printed "suggested walltime (2x)" to
# replace the walltime in kernel_param_tta.sh.
set -uo pipefail
WORKDIR=/home/user/workspace/RIMC
cd "$WORKDIR" || exit 1
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96
git pull --ff-only origin main || true
OUT=cluster_results/smoke/kernel_param_tta_smoke.csv
mkdir -p cluster_results/smoke
rm -f "$OUT"
conda run -n py311 --no-capture-output python cluster_scripts/kernel_param_tta.py \
    --specs bb144,rect:3x11 --p-list 1e-3 --arms "0.0,2;0.3,2;0.3,6" \
    --n-seeds 2 --L0 1000 --L-max 32000 --cache-cap 200000 --n-cores 8 \
    --run-id 99 --out-csv "$WORKDIR/$OUT"
RC=$?
echo "[smoke] first run rc=$RC"
# Second identical invocation: every task is already in the CSV -> must print "0 pending".
SECOND=$(conda run -n py311 --no-capture-output python cluster_scripts/kernel_param_tta.py \
    --specs bb144,rect:3x11 --p-list 1e-3 --arms "0.0,2;0.3,2;0.3,6" \
    --n-seeds 2 --L0 1000 --L-max 32000 --cache-cap 200000 --n-cores 8 \
    --run-id 99 --out-csv "$WORKDIR/$OUT" 2>&1)
RC2=$?
echo "$SECOND"
if echo "$SECOND" | grep -q " 0 pending"; then
    echo "[smoke] resume dedup OK (0 pending)"
else
    echo "[smoke] WARNING: resume dedup FAILED (second run did not report 0 pending)"
    RC=1
fi
[ "$RC2" -ne 0 ] && RC=1
conda run -n py311 --no-capture-output python - "$WORKDIR/$OUT" <<'EOF' || RC=1
import csv, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("WARNING: empty CSV")
    sys.exit(1)
for r in rows:
    r["length"] = int(r["length"])
    for k in ("wall", "cpu", "p_cluster"):
        r[k] = float(r[k])
    r["cluster_k"] = int(r["cluster_k"])

groups = defaultdict(list)
for r in rows:
    groups[(r["code"], r["p_cluster"], r["cluster_k"])].append(r)

rate = defaultdict(list)   # code -> list of s/step over arms
bad = 0
print("code, p_cluster, m : L_longest  s/step  cpu/wall")
for key in sorted(groups):
    code, pc, m = key
    Lmax = max(r["length"] for r in groups[key])
    top = [r for r in groups[key] if r["length"] == Lmax]
    sps = sum(r["wall"] for r in top) / sum(r["length"] for r in top)
    cw = sum(r["cpu"] for r in top) / max(sum(r["wall"] for r in top), 1e-12)
    rate[code].append(sps)
    print(f"{code:10s} pc={pc:<4g} m={m}: L={Lmax:<7d} {sps:.3e} s/step  cpu/wall={cw:.3f}")
for r in rows:
    cw = r["cpu"] / max(r["wall"], 1e-12)
    if cw <= 0.9:
        bad += 1
        print(f"WARNING: cpu/wall={cw:.3f} <= 0.9 for {r['code']} pc={r['p_cluster']} m={r['cluster_k']} L={r['length']} seed={r['seed']}")
print("cpu/wall check:", "OK" if bad == 0 else f"{bad} row(s) below 0.9")

STEPS = 3 * 10 * 50 * 4_095_000          # per code, per run
mean = {c: sum(v) / len(v) for c, v in rate.items()}
bb = mean.get("bb144")
core_s = 0.0
for code in ("rect:3x11", "bb144", "bb_l9m6"):
    s = mean.get(code, bb if code == "bb_l9m6" else None)
    if s is None:
        print(f"WARNING: no rate for {code}; skipped")
        continue
    cs = STEPS * s
    core_s += cs
    print(f"{code:10s} steps={STEPS:.3e} s/step={s:.3e} core-hours={cs/3600:.1f}")
node_h = core_s / 96 / 3600
print(f"estimated node-hours per run = {node_h:.2f}")
print(f"suggested walltime (2x) = {2 * node_h:.1f} h")
EOF
ntfy_done "RIMC kernel_param_tta SMOKE done" "job ${PBS_JOBID} rc=$RC. Read the extrapolation, set walltime in kernel_param_tta.sh, then qsub -v RUN_ID=4..8"
exit $RC
