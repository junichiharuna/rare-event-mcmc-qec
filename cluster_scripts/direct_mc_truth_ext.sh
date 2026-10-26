#!/bin/bash
#PBS -N dmc_truth_ext
#PBS -q workq
#PBS -l select=1:ncpus=20:mem=80gb
#PBS -l walltime=24:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/dmcext_^array_index^.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/dmcext_^array_index^.err
#PBS -J 1-60

# Truth-grid extension: d=7, p=1e-3 for toric / surface / asymmetric_toric.
# 60 subjobs = 3 codes x 20 subjobs (20 cores each) -> 400 cores per code, pooled.
# Feasibility: p_faulty ~1-2e-9 -> ~1.5-3e11 trials for ~300 faulty (< bb144 1e-4).

P=0.001
D=7
NTARGET=300
MMAX=1e15
TBUDGET=84600        # seconds; < walltime (24h=86400) so Stats are flushed before kill
NSUB=60              # total subjobs across all 3 codes

WORKDIR=/home/user/workspace/RIMC
cd $WORKDIR
source cluster_scripts/lib/notify.sh

IDX=$(( PBS_ARRAY_INDEX - 1 ))
CODES=(toric surface asymmetric_toric)
CODE=${CODES[$(( IDX / 20 ))]}
SUB=$(( IDX % 20 ))

OUT_JSONL=$WORKDIR/cluster_scripts/logs/dmc_stats_${CODE}_d${D}_p1e-3_${PBS_ARRAY_INDEX}.jsonl
MARKER=$WORKDIR/cluster_scripts/logs/dmcext_${PBS_ARRAY_INDEX}.done
rm -f $MARKER $OUT_JSONL

conda run -n py311 python cluster_scripts/sweep_direct_mc_stats.py \
    --noise bit_flip \
    --code-type $CODE \
    --code-size $D \
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

N_DONE=$(grep -l 'exit=0' $WORKDIR/cluster_scripts/logs/dmcext_*.done 2>/dev/null | wc -l)
LOCK=$WORKDIR/cluster_scripts/logs/dmcext.lock
if [ "$N_DONE" -ge "$NSUB" ] && mkdir "$LOCK" 2>/dev/null; then
    conda run -n py311 python cluster_scripts/merge_direct_mc_stats.py
    git add cluster_results/direct_mc_ew.csv
    git commit -m "data: truth-grid extension d=7 p=1e-3 toric/surface/asym (job ${PBS_JOBID}) - pooled ${NSUB} subjobs"
    git push origin main
    ntfy_done "RIMC truth-ext done" \
        "job ${PBS_JOBID} finished - ${N_DONE}/${NSUB} subjobs. Run: git pull"
fi
