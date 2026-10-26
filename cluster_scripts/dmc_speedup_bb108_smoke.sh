#!/bin/bash
#PBS -N dmc_bb108_smoke
#PBS -q workq
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=00:30:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/dmc_speedup_bb108_smoke.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/dmc_speedup_bb108_smoke.err
#
# SMOKE for the BB[[108,8,10]] (bb_l9m6) addition to the time-to-accuracy benchmark
# (cluster_scripts/dmc_speedup_bb108.sh).  Validates, on real cluster CPUs and at toy
# scale, the only things that are NEW relative to the published 7-code job:
#   * the bb_l9m6 entry just added to code_advantage_map.CODE_REGISTRY builds
#     (n=108, k=8, logical X weights 12..30) and its BP-OSD decoder is constructed
#     per p, exactly as bb72/bb144 are;
#   * both phases of dmc_speedup.py run for it (dmc_cutoff + advantage), the resume
#     dedup holds, and the CSV is written where the analysis expects it;
#   * MOST IMPORTANTLY, it MEASURES the per-step wall rate at p=1e-1, which is the one
#     number the production walltime rests on and for which no bb_l9m6 data exists.
#     cluster_results/lowp_audit_2e6_gr.csv covers only p<=1e-2, and the p=1e-1 cost of
#     BP-OSD grows fast with n (run 13: bb144 OBF@1e-1 is 2.36x bb72's per-step cost,
#     while at p=1e-3 the two are within 5%), so the p=1e-1 rate was EXTRAPOLATED for
#     the production estimate by interpolating that ratio in n (108 vs 72 and 144).
#     This job replaces the extrapolation with a measurement.
#
# run-id 99 keeps every seed stream disjoint from the production runs 4..13.
# Cost: ~2 min of 8 cores (longest single task = OBF, p=1e-1, L=1.05e6 ~ 3 min).
set -euo pipefail
WORKDIR=/home/user/workspace/RIMC
cd "${PBS_O_WORKDIR:-$WORKDIR}"
source cluster_scripts/lib/notify.sh
git pull --ff-only origin main || true

OUT=cluster_results/dmc_speedup_bb108_smoke.csv
rm -f "$OUT"

SECONDS=0
conda run -n py311 python cluster_scripts/dmc_speedup.py \
  --specs bb_l9m6 --p-list 1e-1,1e-3 \
  --dmc-chunks 8 --dmc-budget 10 --n-seeds 2 --L0 1000 \
  --L-max 1048576 --ctrl-obf-L-max 1048576 --kernel-L-max 65536 \
  --n-cores 8 --run-id 99 --out-csv "$OUT"
echo "SMOKE_SECONDS=${SECONDS}"

# Sanity + the production extrapolation the walltime is set from.
conda run -n py311 python - <<'PY'
import collections, csv
rows = list(csv.DictReader(open("cluster_results/dmc_speedup_bb108_smoke.csv")))
keys = [(r["phase"], r["code"], r["p"], r["method"], r["seed"], r["length"]) for r in rows]
dup = [k for k, c in collections.Counter(keys).items() if c > 1]
assert not dup, f"duplicate task keys (resume failed): {dup[:5]}"
assert {r["code"] for r in rows} == {"bb_l9m6"}, {r["code"] for r in rows}
assert {r["group"] for r in rows} == {"trap"}, "bb_l9m6 must be in the trap group"
n_dmc = sum(1 for r in rows if r["phase"] == "dmc_cutoff")
assert n_dmc == 16, n_dmc                      # 2 p x 8 chunks
mx = {}
for r in rows:
    if r["phase"] == "advantage":
        mx[(r["p"], r["method"])] = max(mx.get((r["p"], r["method"]), 0), int(r["length"]))
assert mx[("0.1", "OBF")] == 1024000, mx     # 1000*2^10 <= 1048576 < 1000*2^11
assert mx[("0.1", "Z+H")] == 64000, mx      # 1000*2^6  <=   65536 < 1000*2^7

# s/step per (p, method), then the production cost per RUN_ID at the base caps:
#   OBF  cap 33,554,432 -> 16 checkpoints, sum of lengths = 1000*(2^16-1) = 6.5535e7 steps/seed
#   Z+H  cap    512,000 -> 10 checkpoints, sum of lengths = 1000*(2^10-1) = 1.023e6  steps/seed
# times 50 seeds, plus 3 p x 96 chunks x 300 s = 24 core-h of direct-MC cutoff.
rate = collections.defaultdict(lambda: [0.0, 0])
for r in rows:
    if r["phase"] == "advantage":
        rate[(float(r["p"]), r["method"])][0] += float(r["wall"])
        rate[(float(r["p"]), r["method"])][1] += int(r["length"])
print("\n  p        method   s/step")
for k in sorted(rate):
    print(f"  {k[0]:<8g} {k[1]:<8s} {rate[k][0] / rate[k][1]:.3e}")
STEPS = {"OBF": 1000 * (2 ** 16 - 1), "Z+H": 1000 * (2 ** 10 - 1)}
core_h, longest = 3 * 96 * 300 / 3600.0, 0.0
for (p, m), (w, L) in rate.items():
    for pp in ([0.1] if p == 0.1 else [1e-3, 1e-5]):   # 1e-5 costs ~ the same as 1e-3
        core_h += (w / L) * STEPS[m] * 50 / 3600.0
        longest = max(longest, (w / L) * (STEPS[m] + 1000) / 2 / 3600.0)
print(f"\n  projected production: {core_h:.0f} core-h per RUN_ID = {core_h / 96:.2f} h on 96 cores")
print(f"  longest single task ~ {longest:.2f} h (OBF, p=1e-1, L=3.36e7)")
print(f"  => makespan ~ {max(core_h / 96, longest) * 1.15:.2f} h; walltime 06:00:00 is fine "
      f"if this is under ~4 h, otherwise raise it before submitting run 4.")
print("  NOTE: conservative. The BP-OSD syndrome cache warms over a chain, so the s/step of")
print("        these short smoke chains exceeds that of a 3.36e7-step production chain.")
PY

echo "SMOKE_OK"
ntfy_done "dmc_speedup bb108 smoke done" "check the printed s/step and the projected per-RUN_ID makespan before qsub-ing dmc_speedup_bb108.sh"
