#!/bin/bash
#PBS -N subset_ext_smoke
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=64gb
#PBS -l walltime=00:30:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/subset_sampling_ext_smoke.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/subset_sampling_ext_smoke.err

# Fast cluster smoke (~minutes) for the cross-code-audit extension.  Runs the FULL
# production condition set (hgp_hamming3 / bb72 / bb_l9m6 at 4 p) at a small budget:
# validates env/paths/pool/BP-OSD generic path, the low-weight theorem assert
# (w<=1 for hgp_hamming3 at d=3, w<=2 for the two BB codes), the resume
# read-modify-write and the two CSV writes on a node -- and gives the ms/decode
# timing that sets the production walltime.  Precision is deliberately starved
# (enum-cap 2e5 pushes C(58,4)=424k and C(108,3)=204k into MC), so the smoke
# checks the code path, not the numbers.
#
# Local reference (serial per-shot direct MC, laptop): hgp_hamming3 0.066 ms/decode,
# bb72 0.051, bb_l9m6 0.067 -- all cheaper than bb144's ~0.2 ms.  The decode cache
# hit rate is ~0 (distinct low-weight syndromes), so --cache-cap only bounds memory.
#
# hgp_hamming3 is BP-OSD-VERSION SENSITIVE: its f_w (and hence P_L / E[w|F]) differs
# by ~4% between the cluster env and a laptop with ldpc 2.4.1 / bposd 2.1, while bb72
# and bb_l9m6 agree to <0.3 sigma.  Compare its reference value only against a
# zh_rerun.csv produced in the SAME environment.
# Writes to a temp prefix, so it cannot touch cluster_results/. No git commit.

WORKDIR=/home/user/workspace/RIMC
cd $WORKDIR
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96

OUTDIR=$(mktemp -d)
conda run -n py311 python cluster_scripts/subset_sampling.py \
    --codes hgp_hamming3:0,bb72:0,bb_l9m6:0 \
    --p-list 1e-2,3e-3,1e-3,1e-4 \
    --enum-cap 200000 --n-max 200000 --target-fail 50 --pilot-n 5000 \
    --n-cores 96 --cache-cap 200000 \
    --out-prefix $OUTDIR/subset_sampling_ext_smoke

echo "=== smoke f_w table ==="
cat $OUTDIR/subset_sampling_ext_smoke_fw.csv
echo "=== smoke synthesized rows (direct_mc CSV_HEADER schema) ==="
cat $OUTDIR/subset_sampling_ext_smoke.csv
rm -rf $OUTDIR

ntfy_done "RIMC subset_sampling_ext SMOKE done" "job ${PBS_JOBID} finished OK -> check ms/decode, then submit subset_sampling_ext.sh"
