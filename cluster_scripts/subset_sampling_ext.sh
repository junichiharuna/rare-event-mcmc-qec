#!/bin/bash
#PBS -N subset_ext
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=128gb
#PBS -l walltime=02:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/subset_sampling_ext.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/subset_sampling_ext.err

# Extension of subset_sampling.sh to the three cross-code-audit families that the
# first production run did not cover: hgp_hamming3 [[58,16,3]], bb72 [[72,12,6]]
# and bb_l9m6 [[108,8,10]].  Same estimator, same budgets, same output prefix.
#
# All three are decoded by BP-OSD with the channel prior set to p -- exactly the
# decoder cluster_scripts/zh_rerun.py builds for its MCMC chains and its own
# direct-MC truth -- so f_w is p-DEPENDENT and each code runs once per p
# (3 codes x 4 p = 12 conditions).  Sharing the decoder is the point: these rows
# are the MCMC-independent reference for tab:audit's truth column, at the low p
# where rejection MC cannot reach (bb_l9m6 has P_L ~ 8e-5 already at p=1e-2).
#
# Reproducibility caveat: hgp_hamming3's f_w depends on the BP-OSD IMPLEMENTATION,
# not just its parameters -- the same code and the same make_bposd_decoder(code, p)
# give P_L = 1.643e-2 on the cluster and 1.573e-2 on a laptop with ldpc 2.4.1 /
# bposd 2.1 (verified against an independent local direct MC; the code construction
# is byte-identical, see tests/test_subset_sampling.py).  bb72 and bb_l9m6 agree
# across both environments to <0.3 sigma.  So this job must run in the SAME env that
# produced cluster_results/zh_rerun.csv, and its hgp_hamming3 row is only comparable
# to that file's truth_ew, never to a locally computed one.
#
# Budget note: --n-max 5e6 is a CAP, not a target; the pilot sizes each sampled
# stratum for --target-fail 1000.  Enumeration reaches w<=5 (n=58), w<=4 (n=72)
# and w<=3 (n=108) at --enum-cap 5e6.  Local timing is ~0.03-0.1 ms/decode/core
# (these codes are smaller than bb144), so the worst case in which several strata
# saturate the cap is ~4.5e7 decodes/condition / 96 cores ~ 1 min; 02:00:00 is
# headroom.  --cache-cap 200000 bounds memory (96 procs x 200k x ~0.4 kB ~ 8 GB);
# the hit rate is ~0 because low-weight errors have distinct syndromes.
#
# RESUME: the SAME --out-prefix as subset_sampling.sh.  Both CSVs are read-
# modify-written per condition and a condition already present in the f_w CSV is
# skipped, so the toric/surface/UT/bb144 rows written by the first job are left
# untouched and only the 12 new conditions run.  --p-list is unchanged from that
# job, so the "stale p grid" caveat in subset_sampling.py does not apply.

WORKDIR=/home/user/workspace/RIMC
cd $WORKDIR
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96

conda run -n py311 python cluster_scripts/subset_sampling.py \
    --codes hgp_hamming3:0,bb72:0,bb_l9m6:0 \
    --p-list 1e-2,3e-3,1e-3,1e-4 \
    --enum-cap 5000000 --n-max 5000000 --target-fail 1000 --pilot-n 20000 \
    --n-cores 96 --cache-cap 200000 \
    --out-prefix $WORKDIR/cluster_results/subset_sampling

git add cluster_results/subset_sampling_fw.csv cluster_results/subset_sampling.csv
git commit -m "data: weight-stratified reference P_L / E[w|F] at p=1e-2..1e-4 (hgp_hamming3/bb72/bb_l9m6) (job ${PBS_JOBID})

Extends the subset_sampling CSVs to the three cross-code-audit families, using the
same BP-OSD(prior=p) decoder as cluster_scripts/zh_rerun.py so the reference and the
MCMC chains share one failure definition.  Exact decomposition P_L = sum_w B(w;n,p) f_w
(src/subset_sampling.py), NOT direct MC: the n_trials / n_faulty columns are stratified
sample and failure counts, so n_faulty/n_trials is not p_L -- read the p_L column.
p_L_se is floored at the truncation bound sum_{w>w_max} B(w;n,p)."

# This job appends to CSVs an earlier subset_sampling job already pushed, so the
# checkout can be behind origin; rebase-and-retry instead of losing the race.
for _try in 1 2 3; do
    git pull --rebase origin main && git push origin HEAD:main && break
    echo "[publish] push attempt $_try failed - retrying"
    sleep 20
done

ntfy_done "RIMC subset_sampling_ext done" "job ${PBS_JOBID} finished. Run: git pull, then use cluster_results/subset_sampling.csv for the tab:audit reference values"
