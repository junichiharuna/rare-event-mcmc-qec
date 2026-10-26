#!/bin/bash
#PBS -N test_ntfy
#PBS -q workq
#PBS -l select=1:ncpus=1:mem=1gb
#PBS -l walltime=00:05:00
#PBS -o /home/user/workspace/RIMC/cluster_scripts/logs/test_ntfy_^array_index^.out
#PBS -e /home/user/workspace/RIMC/cluster_scripts/logs/test_ntfy_^array_index^.err
#PBS -J 1-3

WORKDIR=/home/user/workspace/RIMC
cd $WORKDIR
source cluster_scripts/lib/notify.sh

MARKER=$WORKDIR/cluster_scripts/logs/test_ntfy_${PBS_ARRAY_INDEX}.done
rm -f $MARKER

echo "subjob ${PBS_ARRAY_INDEX} sleeping..."
sleep 5
echo "exit=0" > $MARKER

N_COND=3
N_DONE=$(ls $WORKDIR/cluster_scripts/logs/test_ntfy_*.done 2>/dev/null | wc -l)
LOCK=$WORKDIR/cluster_scripts/logs/test_ntfy.lock
if [ "$N_DONE" -ge "$N_COND" ] && mkdir "$LOCK" 2>/dev/null; then
    ntfy_done "RIMC test_ntfy done" \
        "All 3 subjobs finished (job ${PBS_JOBID}). Notification pipeline OK."
fi
