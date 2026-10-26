#!/bin/bash
#PBS -N subset_at3x7
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=128gb
#PBS -l walltime=00:20:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/subset_sampling_at3x7.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/subset_sampling_at3x7.err

# Weight-stratified reference values P_L(p) and E[w|F](p) for the asymmetric toric
# code AT(3,7), the last code of the p2 diagnostic benchmark without one.
#
# Why: tab:diag covers six codes, and cluster_results/subset_sampling.csv already
# carries stratified rows for five of them; AT(3,7) ("ut3x7" in cluster_results/
# p2_diag) is the only one still missing.  The paper is moving to the weight-
# stratified estimate as the universal bit-flip reference, so this last cell has to
# come from the same MCMC-independent estimator as the others rather than from a
# chain or from direct MC.
#
# One condition, MWPM (src/subset_sampling.py builds the rectangular
# asymmetric_toric spec "3x7" with make_asymmetric_toric_code(3, 7) plus
# make_mwpm_decoder): f_w carries no channel prior, so it is p-INDEPENDENT and a
# single run synthesizes the whole --p-list at once.
#
# Budget: identical to subset_sampling.sh / subset_sampling_p1e5.sh (enum-cap 5e6,
# n-max 5e6 cap, target-fail 1000).  n = 42 is the smallest code in the family and
# the measured MWPM wall on this pipeline is ~1 s per condition, so the job is over
# almost as soon as it starts; the 96-core request just follows the driver's pool
# convention (--n-cores 96) and 00:20:00 is pure headroom.
#
# RESUME, and NO --force: the driver resumes on the f_w table alone, and --p-list
# here is exactly the grid already in the CSVs (1e-2,3e-3,1e-3,1e-4,1e-5), so the
# "stale p grid" caveat in subset_sampling.py does not apply.  Every code already
# present -- toric/surface, the other asymmetric_toric sizes, concat_steane and the
# BP-OSD families -- is skipped and its rows are left untouched; only the one new
# AT(3,7) condition runs.

WORKDIR=/home/user/workspace/RIMC
cd $WORKDIR
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96

conda run -n py311 python cluster_scripts/subset_sampling.py \
    --codes asymmetric_toric:3x7 \
    --p-list 1e-2,3e-3,1e-3,1e-4,1e-5 \
    --enum-cap 5000000 --n-max 5000000 --target-fail 1000 --pilot-n 20000 \
    --n-cores 96 --cache-cap 200000 \
    --out-prefix $WORKDIR/cluster_results/subset_sampling

git add cluster_results/subset_sampling_fw.csv cluster_results/subset_sampling.csv
git commit -m "data: weight-stratified reference for AT(3,7) (asymmetric_toric 3x7), all 5 p (job ${PBS_JOBID})

Completes the stratified reference for the six codes of the p2 diagnostic benchmark
(tab:diag): AT(3,7) was the only one without a row.  MWPM decoder, so f_w is
p-independent and one run synthesizes p = 1e-2..1e-5.  Resume mode, no --force, and
--p-list unchanged, so all previously written rows are untouched.
Exact decomposition P_L = sum_w B(w;n,p) f_w (src/subset_sampling.py), NOT direct MC:
n_trials / n_faulty are stratified sample and failure counts, read the p_L column."

# This job appends to CSVs earlier subset_sampling jobs already pushed, so the
# checkout can be behind origin; rebase-and-retry instead of losing the race.
for _try in 1 2 3; do
    git pull --rebase origin main && git push origin HEAD:main && break
    echo "[publish] push attempt $_try failed - retrying"
    sleep 20
done

ntfy_done "RIMC subset_sampling_at3x7 done" "job ${PBS_JOBID} finished. Run: git pull, then python compare/analyze_p2_diag_table.py"
