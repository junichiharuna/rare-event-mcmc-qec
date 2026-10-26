#!/bin/bash
#PBS -N dmc_bb144_lowp
#PBS -q workq
#PBS -l select=1:ncpus=20:mem=80gb
#PBS -l walltime=24:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/dmclp_^array_index^.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/dmclp_^array_index^.err
#PBS -J 1-20

# bb144 low-p brute-force rejection MC: 20 subjobs x 20 cores = 400 cores, all
# running the SAME condition with disjoint seeds; their raw Stats are pooled.
# Edit P / NTARGET / MMAX / TBUDGET below, then qsub. Phase 1: P=0.001. After
# reading off the exponent t, Phase 2: P=0.0001 with the gated NTARGET.

P=0.001
PTAG=p1e-3            # filename tag (no dots; keep distinct per campaign)
NTARGET=200
MMAX=1e16
TBUDGET=84600        # seconds; < walltime (24h=86400) so Stats are flushed before kill
NSUB=20

WORKDIR=/home/user/workspace/RIMC
cd $WORKDIR
source cluster_scripts/lib/notify.sh

OUT_JSONL=$WORKDIR/cluster_scripts/logs/dmc_stats_${PTAG}_${PBS_ARRAY_INDEX}.jsonl
MARKER=$WORKDIR/cluster_scripts/logs/dmclp_${PTAG}_${PBS_ARRAY_INDEX}.done
rm -f $MARKER $OUT_JSONL

conda run -n py311 python cluster_scripts/sweep_direct_mc_stats.py \
    --noise bit_flip \
    --code-type bb144 \
    --code-size 0 \
    --p $P \
    --n-target $NTARGET \
    --m-max $MMAX \
    --n-cores 20 \
    --time-budget $TBUDGET \
    --sampler binomial \
    --subjob-idx $PBS_ARRAY_INDEX \
    --out-jsonl $OUT_JSONL

PYEXIT=$?
echo "exit=$PYEXIT" > $MARKER

N_DONE=$(grep -l 'exit=0' $WORKDIR/cluster_scripts/logs/dmclp_${PTAG}_*.done 2>/dev/null | wc -l)
LOCK=$WORKDIR/cluster_scripts/logs/dmclp_${PTAG}.lock
if [ "$N_DONE" -ge "$NSUB" ] && mkdir "$LOCK" 2>/dev/null; then
    conda run -n py311 python cluster_scripts/merge_direct_mc_stats.py
    git add cluster_results/direct_mc_ew.csv
    git commit -m "data: bb144 low-p direct MC (${PTAG}, job ${PBS_JOBID}) - pooled ${NSUB} subjobs"
    git push origin main
    ntfy_done "RIMC bb144 low-p done" \
        "job ${PBS_JOBID} (${PTAG}) finished - ${N_DONE}/${NSUB} subjobs. Run: git pull"
fi
