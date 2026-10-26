#!/bin/bash
#PBS -N subset_sampling_smoke
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=64gb
#PBS -l walltime=00:30:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/subset_sampling_smoke.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/subset_sampling_smoke.err

# Fast cluster smoke (~minutes) for the weight-stratified reference values.
# Runs the FULL production condition set (7 MWPM codes + bb144 at 4 p) at a small
# budget: validates env/paths/pool/pymatching-fast-path/BP-OSD-generic-path, the
# "zero failures below d/2" assert, the resume read-modify-write, and the two CSV
# writes on a node -- and gives the ms/decode timing that sets the production
# walltime.  The low-w bb144 strata are deliberately starved here (enum-cap 2e5
# pushes C(144,3)=487k into MC), so the smoke checks the code path, not precision.
#
# Local reference timings (4-8 cores, laptop): MWPM ~0.006 ms/decode/core;
# bb144 BP-OSD 0.05 ms/decode at w=3 rising to ~0.21 ms at w=5 (harder syndromes
# -> more BP iterations + OSD-CS), essentially independent of the p prior.
# The decode cache hit rate on bb144 is ~0 (weight-w errors below d/2 have
# distinct syndromes), so --cache-cap here only bounds memory.
# No git commit.

WORKDIR=/home/user/workspace/RIMC
cd $WORKDIR
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96

OUTDIR=$(mktemp -d)
conda run -n py311 python cluster_scripts/subset_sampling.py \
    --codes toric:5,toric:7,surface:5,surface:7,asymmetric_toric:5x5,asymmetric_toric:3x11,asymmetric_toric:3x15,bb144:0 \
    --p-list 1e-2,3e-3,1e-3,1e-4 \
    --enum-cap 200000 --n-max 200000 --target-fail 50 --pilot-n 5000 \
    --n-cores 96 --cache-cap 200000 \
    --out-prefix $OUTDIR/subset_sampling_smoke

echo "=== smoke f_w table ==="
cat $OUTDIR/subset_sampling_smoke_fw.csv
echo "=== smoke synthesized rows (direct_mc CSV_HEADER schema) ==="
cat $OUTDIR/subset_sampling_smoke.csv
rm -rf $OUTDIR

ntfy_done "RIMC subset_sampling SMOKE done" "job ${PBS_JOBID} finished OK -> check ms/decode, then submit subset_sampling.sh"
