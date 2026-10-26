#!/bin/bash
# Generic PBS array-job watcher with HANG detection.
#
# Polls every 30s and logs "done=N/EXPECTED running=R" to a log file.
# Exits (waking any controller waiting on this background task) when EITHER:
#   - the .done marker count reaches EXPECTED (normal completion), OR
#   - a running subjob's cput (qstat "Time Use") stays frozen for >=STALL_LIMIT
#     consecutive polls (default 6 * 30s = 3 min) while still in state R.
#     A frozen cput with advancing walltime means the process is hung
#     (e.g. memory-contention I/O stall), not merely slow.
#
# Usage:
#   jobs/watch_array.sh <marker_glob> <expected_count> <log_path> <jobid> [jobid ...]
# Example:
#   jobs/watch_array.sh 'cluster_scripts/logs/q3k15_*.done' 125 cluster_scripts/logs/q3_watch.log '257965[]' '257968[]'

set -u
MARKER_GLOB="$1"; EXPECTED="$2"; LOG="$3"; shift 3
JOBS=("$@")
STALL_LIMIT=6   # consecutive frozen-cput polls before flagging a hang (6*30s = 3 min)

declare -A prev_cput
declare -A stall
: > "$LOG"

while :; do
  N=$(ls $MARKER_GLOB 2>/dev/null | wc -l)
  # running subjobs as "subjobid cput(TimeUse)"; field 4 = Time Use, field 5 = state
  mapfile -t lines < <(qstat -t "${JOBS[@]}" 2>/dev/null | awk 'NR>2 && $5=="R"{print $1, $4}')
  nrun=${#lines[@]}
  hangmsg=""
  for ln in "${lines[@]}"; do
    jid=${ln%% *}; cput=${ln##* }
    if [ -n "$cput" ] && [ "${prev_cput[$jid]:-}" = "$cput" ]; then
      stall[$jid]=$(( ${stall[$jid]:-0} + 1 ))
    else
      stall[$jid]=0
    fi
    prev_cput[$jid]=$cput
    if [ "${stall[$jid]:-0}" -ge "$STALL_LIMIT" ]; then
      hangmsg="$hangmsg ${jid}(cput=${cput} frozen ~$(( STALL_LIMIT * 30 ))s)"
    fi
  done
  ts=$(date +%H:%M:%S)
  if [ -n "$hangmsg" ]; then
    echo "$ts  done=$N/$EXPECTED  running=$nrun  ** HANG SUSPECTED:$hangmsg **" | tee -a "$LOG"
    break
  fi
  echo "$ts  done=$N/$EXPECTED  running=$nrun" | tee -a "$LOG"
  [ "$N" -ge "$EXPECTED" ] && break
  sleep 30
done

NFAIL=$(grep -L 'exit=0' $MARKER_GLOB 2>/dev/null | wc -l)
echo "$(date +%H:%M:%S)  WATCH_EXIT done=$N/$EXPECTED non_zero_exit=$NFAIL hang='${hangmsg# }'" | tee -a "$LOG"
