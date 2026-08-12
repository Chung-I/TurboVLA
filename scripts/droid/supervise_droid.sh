#!/usr/bin/env bash
# Supervisor for the long DROID run on cml18.
#
# The loader has its own stall watchdog, but if the whole job ever hard-hangs
# (log silent > STALL_MIN) or crashes, this kills it and relaunches with
# --resume_mode all (checkpoints every 1000 steps, so <=1000 steps are lost).
#
# Usage (detached):
#   setsid nohup scripts/droid/supervise_droid.sh full > /abs/path/supervisor.log 2>&1 &

set -u

MODE="${1:-full}"
DROID_ROOT="${DROID_ROOT:-/tmp2/chungyili/droid}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="$DROID_ROOT/logs/$MODE.log"
STALL_MIN="${STALL_MIN:-30}"
MAX_STEPS_FILE="$DROID_ROOT/outputs/$MODE/turbovla_droid_100000.pth"
[ "$MODE" = smoke ] && MAX_STEPS_FILE="$DROID_ROOT/outputs/$MODE/turbovla_droid_2000.pth"

ts() { date "+%Y-%m-%d %H:%M:%S"; }

train_pids() {
  # Bracket trick: this supervisor's own cmdline never matches.
  ps -eo pid,cmd | grep "[e]xperiments/droid/train.py" | awk '{print $1}'
}

launch() {
  echo "$(ts) launching $MODE run"
  setsid nohup bash -c "cd $REPO_ROOT && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} DROID_ROOT=$DROID_ROOT HF_HOME=${HF_HOME:-/tmp2/chungyili/hf_home} $REPO_ROOT/scripts/droid/train_cml18.sh $MODE >> $LOG 2>&1" &
  sleep 30
}

kill_run() {
  local pids
  pids="$(train_pids)"
  [ -z "$pids" ] && return
  echo "$(ts) killing stalled/stuck run: $pids"
  kill -TERM $pids 2>/dev/null
  sleep 10
  pids="$(train_pids)"
  [ -n "$pids" ] && kill -KILL $pids 2>/dev/null
  # Also reap torchrun parents left behind.
  local tr_pids
  tr_pids="$(ps -eo pid,cmd | grep "[t]orchrun .*experiments/droid" | awk '{print $1}')"
  [ -n "$tr_pids" ] && kill -KILL $tr_pids 2>/dev/null
  sleep 5
}

while true; do
  if [ -f "$MAX_STEPS_FILE" ]; then
    echo "$(ts) final checkpoint present: $MAX_STEPS_FILE; supervisor exiting"
    exit 0
  fi

  if [ -z "$(train_pids)" ]; then
    echo "$(ts) no training process found"
    launch
  elif [ -f "$LOG" ]; then
    age_min=$(( ($(date +%s) - $(stat -c %Y "$LOG")) / 60 ))
    if [ "$age_min" -ge "$STALL_MIN" ]; then
      echo "$(ts) log silent for ${age_min}min (>= ${STALL_MIN})"
      kill_run
      launch
    fi
  fi
  sleep 300
done
