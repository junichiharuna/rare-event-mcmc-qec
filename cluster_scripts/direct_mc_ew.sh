#!/bin/bash
#PBS -N direct_mc_ew
#PBS -q workq
#PBS -l select=1:ncpus=20:mem=80gb
#PBS -l walltime=24:00:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/dmc_^array_index^.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/dmc_^array_index^.err
#PBS -J 1-14

# Set -J 1-<number of lines in direct_mc_conditions.txt> before qsub.

WORKDIR=/home/user/workspace/RIMC
cd $WORKDIR
source cluster_scripts/lib/notify.sh

CONDITIONS=$WORKDIR/cluster_scripts/direct_mc_conditions.txt
read NOISE CODE D P NTARGET MMAX < <(sed -n "${PBS_ARRAY_INDEX}p" "$CONDITIONS")

OUT_CSV=$WORKDIR/cluster_scripts/logs/dmc_row_${PBS_ARRAY_INDEX}.csv
MARKER=$WORKDIR/cluster_scripts/logs/dmc_${PBS_ARRAY_INDEX}.done
rm -f $MARKER $OUT_CSV

conda run -n py311 python cluster_scripts/sweep_direct_mc_ew.py \
    --noise $NOISE \
    --code-type $CODE \
    --code-size $D \
    --p $P \
    --n-target $NTARGET \
    --m-max $MMAX \
    --n-cores 20 \
    --out-csv $OUT_CSV

PYEXIT=$?
echo "exit=$PYEXIT" > $MARKER

N_COND=$(grep -cve '^[[:space:]]*$' "$CONDITIONS")
N_DONE=$(grep -l 'exit=0' $WORKDIR/cluster_scripts/logs/dmc_*.done 2>/dev/null | wc -l)
LOCK=$WORKDIR/cluster_scripts/logs/dmc.lock
if [ "$N_DONE" -ge "$N_COND" ] && mkdir "$LOCK" 2>/dev/null; then
    conda run -n py311 python cluster_scripts/merge_direct_mc.py
    git add cluster_results/direct_mc_ew.csv
    git commit -m "data: direct MC E[w]/V[w]/p_L (job ${PBS_JOBID}) — auto-push by last subjob"
    git push origin main
    ntfy_done "RIMC direct-MC done" \
        "job ${PBS_JOBID} finished — ${N_DONE}/${N_COND} subjobs. Run: git pull"
fi
