#!/bin/bash
#PBS -N subset_sampling
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=128gb
#PBS -l walltime=04:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/subset_sampling.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/subset_sampling.err

# Production weight-stratified reference values P_L(p) and E[w|F](p) at
# p = 1e-2, 3e-3, 1e-3, 1e-4 for the paper's general-code families.
#
# This replaces direct MC as the source of the LOW-p reference values: the exact
# decomposition P_L = sum_w B(w;n,p) f_w moves the cost into the p-independent
# f_w = P(fail | |e|=w), which is enumerated exactly at low w and sampled at
# fixed weight (where failures are common) above the enum cap.  bb144 at 1e-4
# cost 1.5e12 direct-MC trials for 681 faulty samples; here it is minutes.
#
# MWPM codes: f_w is p-independent, so ONE run per code synthesizes all four p.
# bb144: BP-OSD's channel prior is p, so it runs once per p (4 runs).
#
# Budget note: --n-max 5e6 is a CAP, not a target; the pilot sizes each sampled
# stratum for --target-fail 1000, and only the rarest strata approach the cap.
# Expected runtime from the smoke timings is well under an hour; the 04:00:00
# walltime is deliberate headroom for the worst case in which several bb144
# strata do saturate the cap at ~0.2 ms/decode (4.5e7 decodes/p / 96 cores).
# --cache-cap must stay bounded: 96 procs x 200k entries x ~0.4 kB ~ 8 GB.  It is
# passed explicitly (the cache hit rate is ~0 anyway; it is a memory bound here).
#
# RESUME: both CSVs are read-modify-written after EVERY condition, and a
# condition already present in the f_w CSV is skipped, so a walltime kill keeps
# the finished rows and a resubmit continues instead of restarting.

WORKDIR=/home/user/workspace/RIMC
cd $WORKDIR
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96

conda run -n py311 python cluster_scripts/subset_sampling.py \
    --codes toric:5,toric:7,surface:5,surface:7,asymmetric_toric:5x5,asymmetric_toric:3x11,asymmetric_toric:3x15,bb144:0 \
    --p-list 1e-2,3e-3,1e-3,1e-4 \
    --enum-cap 5000000 --n-max 5000000 --target-fail 1000 --pilot-n 20000 \
    --n-cores 96 --cache-cap 200000 \
    --out-prefix $WORKDIR/cluster_results/subset_sampling

git add cluster_results/subset_sampling_fw.csv cluster_results/subset_sampling.csv
git commit -m "data: weight-stratified reference P_L / E[w|F] at p=1e-2..1e-4 (toric/surface/UT/bb144) (job ${PBS_JOBID})

Exact decomposition P_L = sum_w B(w;n,p) f_w (src/subset_sampling.py), NOT direct MC.
In subset_sampling.csv the n_trials / n_faulty columns are stratified sample and
failure counts, not MC trials -- n_faulty/n_trials is not p_L; read the p_L column.
p_L_se is floored at the truncation bound sum_{w>w_max} B(w;n,p), so a fully
enumerated row is exact and its SE is a systematic bound, not a statistical one."
git push origin main
ntfy_done "RIMC subset_sampling done" "job ${PBS_JOBID} finished. Run: git pull, then use cluster_results/subset_sampling.csv for tab:lowp / tab:rect / tab:endtoend"
