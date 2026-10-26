#!/bin/bash
#PBS -N lowp_audit_2e6_gr
#PBS -q workq
#PBS -l select=1:ncpus=96:mem=128gb
#PBS -l walltime=03:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/lowp_audit_2e6_gr.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/lowp_audit_2e6_gr.err

# WHY THIS RERUN EXISTS. Convergence rule (iv) is being INTERNALISED: the rule currently
# compares the cross-chain scatter of the witness against a reference variance Var_F[w]
# taken from the weight-stratified estimator, which makes an MCMC convergence criterion
# depend on an external truth file. The replacement is a textbook Gelman-Rubin statement --
# cross-chain B against WITHIN-chain W, both measured inside the run. The audit driver never
# emitted W: cluster_results/lowp_audit_2e6.csv carries only ew_within_se_tau1, a MEDIAN
# over chains of sqrt(var_c tau_c / n_c), and the per-chain var_c was discarded on the spot,
# so W = (1/K) sum_c var_c cannot be reconstructed offline from the finished data. Hence
# this rerun, which re-executes the SAME chains and additionally records them.
#
# WHAT CHANGES vs cluster_scripts/lowp_audit_long.sh: NOTHING in the physics. Same driver,
# same protocol, same 11 codes / 3 decades of p / OBF+Z+H / short+long init, same
# --mcmc-target 2000000 --n-seeds 50 --cache-cap 200000, same base --seed default (1000).
# The run is SEED-IDENTICAL to lowp_audit_2e6.csv -- a local rerun was verified to reproduce
# the cluster _chains.csv values exactly -- so every existing column is expected to come back
# bit-for-bit. Only the OUTPUT grew:
#   * a fourth CSV, lowp_audit_2e6_gr_seeds.csv, one row per (condition, seed), with the raw
#     per-chain ew_full / tau_full / (ew, var, tau, n_used, capped) on the 5*tau and 10*tau
#     burned segments, plus the accept proxy and wall time. Any aggregation -- pooled W,
#     trimmed means, per-chain R-hat -- can then be redone offline with no more cluster time.
#   * four summary columns APPENDED at the end of the header (every earlier column keeps its
#     position and meaning): w_var_within_mean (= the Gelman-Rubin W on the 5*tau segments),
#     w_var_within_median, n_used5_mean, tau_burn5_mean. compare/convergence_flags.py forms
#     s_W = sqrt(w_var_within_mean * tau_burn5 / n_used5) from these.
#
# THE OLD FILES ARE LEFT UNTOUCHED under their old names. cluster_results/lowp_audit_2e6.csv
# and its _chains / _conv companions are what the current tables were built from and must
# survive verbatim; this job writes the *_gr.csv family alongside them. Do NOT point this
# script at the old stem to "resume" it: the driver decides "already done" from the SUMMARY
# CSV alone, so the pre-existing conditions would be skipped and would never emit per-chain
# rows, leaving a partial _seeds.csv and blank W columns (see the RESUME paragraph of the
# driver docstring).
#
# NO DIRECT MC IS RUN HERE, exactly as in lowp_audit_long.sh: truth_ew / truth_se / p_L are
# looked up in cluster_results/subset_sampling.csv at job time, and a code missing from that
# file still gets its row, with truth_source=missing and NaN truth/bias.
#
# WALLTIME. MEASURED, not extrapolated: the identical lowp_audit_long.sh run took 107
# core-hours ~ 1.1 node-hours on 96 cores. The added work is four floating-point reductions
# and one extra CSV rewrite per condition -- unmeasurable next to 5100 chains of N=2e6 -- so
# the same cost is expected. 03:00:00 is ~2.7x headroom.
#
# SATURATION (CLAUDE.md: concurrent tasks >= 96 in EVERY phase, the tail included). 11 codes,
# of which the 6 with a non-flat logical-weight basis (hgp_hamming3, bb72, bb_l9m6, bb144,
# ut3x11, ut3x15) carry both a min-weight ("short") and a max-weight ("long") init and the 5
# flat ones (ut5x5, toric5, toric7, surface5, surface7) collapse to a single "uniform" init:
# 17 code-init pairs x 3 p x 2 methods = 102 conditions x 50 seeds = 5100 chains in ONE FLAT
# pool, ordered longest-expected-first (BP-OSD codes then by n), so all 96 cores stay busy
# through the tail. 5100 tasks over 96 cores is ~53 waves, so the tail costs at most one
# chain and the ideal makespan is essentially achieved.
#
# MEMORY. Dominated by the 200k-entry BP-OSD syndrome->correction cache, one per worker,
# ~0.2 GB x 96 ~ 20 GB. Each N=2e6 chain also records an 8 MB int32 trace, < 1 GB over 96
# workers, and the trace never leaves its worker (only the 5 checkpoint tuples and the scalar
# per-chain summary are pickled back -- the new _seeds.csv row is built from that same
# summary, so it adds no transfer). The tau / burn-in FFT runs on a stride-10 copy capped at
# 2e5 points, so it costs the same as at N=2e5 and tau / n_used are rescaled to full steps by
# the driver. 128gb is ample.
#
# RESUME. All FOUR CSVs are rewritten after every completed condition and conditions already
# present in the summary are skipped, so a walltime kill keeps its finished rows and a plain
# resubmit of THIS script continues instead of restarting (use --force only to deliberately
# recompute; note that --force discards all four files, not just the summary).

WORKDIR=/home/user/workspace/RIMC
cd $WORKDIR
source cluster_scripts/lib/notify.sh
export NUMEXPR_MAX_THREADS=96

conda run -n py311 python cluster_scripts/lowp_audit.py \
    --codes hgp_hamming3,bb72,bb_l9m6,bb144,ut5x5,ut3x11,ut3x15,toric5,toric7,surface5,surface7 \
    --p-list 1e-2,1e-3,1e-4 --methods OBF,Z+H --inits short,long \
    --mcmc-target 2000000 --n-seeds 50 --n-cores 96 --cache-cap 200000 \
    --out-csv $WORKDIR/cluster_results/lowp_audit_2e6_gr.csv

git add cluster_results/lowp_audit_2e6_gr.csv cluster_results/lowp_audit_2e6_gr_chains.csv \
        cluster_results/lowp_audit_2e6_gr_conv.csv cluster_results/lowp_audit_2e6_gr_seeds.csv
git commit -m "data: rerun of lowp_audit_2e6 with per-chain variances for Gelman-Rubin W (job ${PBS_JOBID})

Convergence rule (iv) is being internalised: the within-chain variance W replaces the
external reference Var_F[w]. The finished cluster_results/lowp_audit_2e6.csv carries only
ew_within_se_tau1, a MEDIAN over chains of sqrt(var_c tau_c / n_c), and the per-chain var_c
was never written, so W = (1/K) sum_c var_c could not be reconstructed offline.

This run is SEED-IDENTICAL to lowp_audit_2e6.csv -- same driver, protocol, conditions,
budget and base seed -- and only ADDS output: the new lowp_audit_2e6_gr_seeds.csv holds the
raw per-chain (ew, tau, var, n_used, capped) on the 5*tau and 10*tau burned segments, and
four columns are appended at the END of the summary header (w_var_within_mean = W,
w_var_within_median, n_used5_mean, tau_burn5_mean). Every pre-existing column keeps its
meaning and position. cluster_results/lowp_audit_2e6.csv and its _chains / _conv companions
are left untouched under their old names.

Reference values are NOT computed here -- truth_ew/truth_se/p_L are looked up in
cluster_results/subset_sampling.csv and rows whose code is not yet in that file carry
truth_source=missing with NaN truth/bias."
git push origin main
ntfy_done "RIMC lowp_audit_2e6_gr done" "job ${PBS_JOBID} finished. Run: git pull, then diff lowp_audit_2e6_gr.csv against lowp_audit_2e6.csv (shared columns must match) and feed w_var_within_mean to compare/convergence_flags.py"
