#!/bin/bash
#PBS -N subset_p1e5
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=128gb
#PBS -l walltime=01:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/subset_sampling_p1e5.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/subset_sampling_p1e5.err

# Weight-stratified reference values P_L(p) and E[w|F](p) for the EIGHT codes of the
# time-to-accuracy benchmark (tab:dmcspeedup / tab:obfspeedup), extended to p = 1e-5.
#
# Why: at p=1e-5 direct MC sees too few faulty draws (<30 per run) on seven of the
# eight codes, so compare/analyze_dmc_speedup.py fell back to the longest Z+H chain
# itself as the reference -- which tests convergence, not accuracy.  These rows give
# every (code, p) cell an MCMC-independent reference, and the analysis script's new
# "stratified" tier (cutoff -> stratified -> grid -> zh) picks them up.
#
# Three decoder regimes (see src/subset_sampling.py):
#   * MWPM (surface:3, asymmetric_toric 3x11 / 3x15) -- f_w is p-independent; ONE run
#     per code synthesizes the whole --p-list.  surface:3 (n=13) is new; the two
#     rectangular codes are recomputed only because --p-list changed (see below).
#   * concat_steane:2 (n=49, hierarchical hard decoder) -- also p-independent, generic
#     per-syndrome path (no pymatching); new in this job.
#   * BP-OSD (hgp_hamming3, bb72, bb144, bb_l9m6; channel prior = p) -- f_w depends on
#     p, one run per p.  Their 1e-2..1e-4 rows are recomputed too (same seed, same
#     budget, so they reproduce the published values) because --force is on.
#
# --force is REQUIRED here: the driver resumes on the f_w table alone, so with a
# changed --p-list an already-computed code would be skipped and keep only its OLD p
# grid in subset_sampling.csv (driver docstring).  The rows of codes NOT listed in
# --codes (toric 5/7, surface 5/7, asymmetric_toric 5x5) are untouched.
#
# Budget: identical to subset_sampling.sh / subset_sampling_ext.sh (enum-cap 5e6,
# n-max 5e6 cap, target-fail 1000).  Measured 1e-4 wall on this pipeline: bb144 24 s,
# bb_l9m6 16 s, bb72 16 s, hgp_hamming3 10 s per condition; MWPM codes ~1 s.  So the
# 8 codes x (5 p for BP-OSD, 1 run for the rest) are ~6 min; 01:00:00 is headroom.
# --cache-cap 200000 bounds memory (96 procs x 200k x ~0.4 kB ~ 8 GB).
#
# hgp_hamming3 is BP-OSD-VERSION SENSITIVE (see subset_sampling_ext.sh): this job must
# run in the same env that produced cluster_results/dmc_speedup_run*.csv, which it does.

WORKDIR=/home/user/workspace/RIMC
cd $WORKDIR
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96

conda run -n py311 python cluster_scripts/subset_sampling.py \
    --codes surface:3,asymmetric_toric:3x11,asymmetric_toric:3x15,concat_steane:2,hgp_hamming3:0,bb72:0,bb144:0,bb_l9m6:0 \
    --p-list 1e-2,3e-3,1e-3,1e-4,1e-5 \
    --enum-cap 5000000 --n-max 5000000 --target-fail 1000 --pilot-n 20000 \
    --n-cores 96 --cache-cap 200000 --force \
    --out-prefix $WORKDIR/cluster_results/subset_sampling

git add cluster_results/subset_sampling_fw.csv cluster_results/subset_sampling.csv
git commit -m "data: weight-stratified reference extended to p=1e-5 for the 8 TTA codes (job ${PBS_JOBID})

Adds p=1e-5 rows (and new surface d=3 / concat_steane l=2 rows) to the weight-stratified
reference so that every cell of tab:dmcspeedup / tab:obfspeedup has an MCMC-independent
E[w|F] reference; compare/analyze_dmc_speedup.py's stratified tier reads them.
Exact decomposition P_L = sum_w B(w;n,p) f_w (src/subset_sampling.py), NOT direct MC:
n_trials / n_faulty are stratified sample and failure counts, read the p_L column.
Recomputed with --force (same seed and budget) because --p-list changed."

for _try in 1 2 3; do
    git pull --rebase origin main && git push origin HEAD:main && break
    echo "[publish] push attempt $_try failed - retrying"
    sleep 20
done

ntfy_done "RIMC subset_sampling_p1e5 done" "job ${PBS_JOBID} finished. Run: git pull, then python compare/analyze_dmc_speedup.py"
