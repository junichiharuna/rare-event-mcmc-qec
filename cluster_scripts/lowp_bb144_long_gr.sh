#!/bin/bash
#PBS -N lowp_bb144_long_gr
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=128gb
#PBS -l walltime=01:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/lowp_bb144_long_gr.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/lowp_bb144_long_gr.err

# The GELMAN-RUBIN rerun of cluster_scripts/lowp_bb144_long.sh -- the bb144 companion of
# lowp_audit_2e6_gr.sh, submitted with it so the corner condition of tab:lowp gets its
# within-chain variance too. See lowp_audit_2e6_gr.sh for the full rationale; in short,
# convergence rule (iv) is being internalised (within-chain W instead of the external
# reference Var_F[w]) and the finished CSVs never recorded the per-chain var_c that W needs.
#
# Long-chain Z+H run on [[144,12,12]] at p=1e-3, the one condition where Z+H still sits
# ~+0.5 above the reference at the default budget (tab:lowp). N=2e6 is ~70 tau at the
# measured tau ~ 3e4, i.e. a 10x budget increase over the N=2e5 default, run from BOTH
# initialisations so the multi-start test of Sec.recipe applies at the longer budget.
# Same driver, same protocol, same conditions, same base seed as lowp_bb144_long.sh: the run
# is SEED-IDENTICAL and every existing column is expected back bit-for-bit. Only the OUTPUT
# grew -- a fourth CSV lowp_bb144_long_gr_seeds.csv (one row per (condition, seed) with the
# raw per-chain ew/tau/var/n_used/capped on the 5*tau and 10*tau burned segments) and four
# summary columns appended at the END of the header (w_var_within_mean = the Gelman-Rubin W,
# w_var_within_median, n_used5_mean, tau_burn5_mean).
#
# THE OLD FILES ARE LEFT UNTOUCHED: cluster_results/lowp_bb144_long.csv and its _chains /
# _conv companions keep their names and content; this job writes the *_gr.csv family beside
# them. Do NOT point this script at the old stem to "resume" it -- the driver decides
# "already done" from the SUMMARY CSV alone, so the finished conditions would be skipped and
# would never emit per-chain rows (see the RESUME paragraph of the driver docstring).
#
# WALLTIME. Measured, not extrapolated: one bb144 Z+H chain at p=1e-3 and N=2e6 takes 97 s
# on a single local core (16.9 s at N=2e5, so the marginal cost FALLS with length -- the
# early steps miss the syndrome cache constantly, the settled chain does not). 2 inits x 50
# seeds = 100 chains = 2.7 core-hours. Ideal makespan on 96 cores is ~1.7 min; see the
# SATURATION note below for why it is really ~2x that, and allow 3x for a slower cluster core
# and 96-way memory contention -- call it 10-15 min. The original 06:00:00 request was smoke
# headroom that the finished lowp_bb144_long.sh run has now retired; 01:00:00 is still ~4x.
# The new per-chain bookkeeping is four float reductions and one extra CSV write, and does
# not move this number.
#
# SATURATION CAVEAT (CLAUDE.md: concurrent tasks >= 96 in EVERY phase, tail included).
# 2 inits x 50 seeds = 100 tasks on 96 cores: one full wave of 96, then a 4-task tail with
# 92 cores idle for a whole chain length, so the makespan is ~2x the ideal. The absolute
# cost is small enough that this is affordable, but if the job is ever grown (e.g. by
# adding OBF, which makes it 200 tasks) fix it by choosing --n-seeds 48 (exactly 96 tasks,
# one clean wave; the cross-chain SE grows by sqrt(50/48) = 2%) or --n-seeds 96 (192 tasks,
# two clean waves and better statistics at 2x the time). 50 is kept here to match the
# K=50 convention of app:conventions -- and to stay seed-identical to lowp_bb144_long.csv.
#
# MEMORY. Chains of N=2e6 record an 8 MB int32 trace each (< 1 GB over 96 workers) and the
# trace never leaves its worker -- only the 5 checkpoint tuples and the scalar per-chain
# summary are pickled back, and the new _seeds.csv row is built from that same summary, so
# it adds no transfer. The tau / burn-in FFT work runs on a stride-10 copy capped at 2e5
# points, so it costs the same as an N=2e5 chain and tau / n_used are rescaled to full steps.
# Peak is dominated by the 200k-entry BP-OSD decode caches, ~20 GB over the node.
#
# The per-chain CSVs are the point of this job: lowp_bb144_long_gr_chains.csv holds the
# running means at N/16 .. N = 1.25e5 .. 2e6 for the post-hoc convergence verdict (made from
# how the cross-chain scatter shrinks with chain length), and lowp_bb144_long_gr_seeds.csv
# holds the within-chain variances that rule (iv) now needs.

WORKDIR=/home/user/workspace/RIMC
cd $WORKDIR
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96

conda run -n py311 python cluster_scripts/lowp_audit.py \
    --codes bb144 --p-list 1e-3 --methods Z+H --inits short,long \
    --mcmc-target 2000000 --n-seeds 50 --n-cores 96 --cache-cap 200000 \
    --out-csv $WORKDIR/cluster_results/lowp_bb144_long_gr.csv

git add cluster_results/lowp_bb144_long_gr.csv cluster_results/lowp_bb144_long_gr_chains.csv \
        cluster_results/lowp_bb144_long_gr_conv.csv cluster_results/lowp_bb144_long_gr_seeds.csv
git commit -m "data: rerun of lowp_bb144_long with per-chain variances for Gelman-Rubin W (job ${PBS_JOBID})

Convergence rule (iv) is being internalised: the within-chain variance W replaces the
external reference Var_F[w]. The finished cluster_results/lowp_bb144_long.csv carries only
ew_within_se_tau1, a MEDIAN over chains of sqrt(var_c tau_c / n_c), and the per-chain var_c
was never written, so W = (1/K) sum_c var_c could not be reconstructed offline.

This run is SEED-IDENTICAL to lowp_bb144_long.csv -- same driver, protocol, conditions,
budget and base seed -- and only ADDS output: the new lowp_bb144_long_gr_seeds.csv holds the
raw per-chain (ew, tau, var, n_used, capped) on the 5*tau and 10*tau burned segments, and
four columns are appended at the END of the summary header (w_var_within_mean = W,
w_var_within_median, n_used5_mean, tau_burn5_mean). Every pre-existing column keeps its
meaning and position. cluster_results/lowp_bb144_long.csv and its _chains / _conv companions
are left untouched under their old names.

Reference is looked up in cluster_results/subset_sampling.csv, not recomputed."
git push origin main
ntfy_done "RIMC lowp_bb144_long_gr done" "job ${PBS_JOBID} finished. Run: git pull, then diff lowp_bb144_long_gr.csv against lowp_bb144_long.csv (shared columns must match) and feed w_var_within_mean to compare/convergence_flags.py"
