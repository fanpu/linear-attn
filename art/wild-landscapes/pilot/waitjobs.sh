#!/bin/bash
# usage: waitjobs.sh id...   exits when all are terminal, prints states
while true; do
  done_all=1; out=""
  for j in "$@"; do
    s=$(pasar show $j --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])' 2>/dev/null)
    out="$out $j:$s"
    case "$s" in completed|failed|cancelled|oom|gpu_oom|gpu_xid|preempted) ;; *) done_all=0;; esac
  done
  [ $done_all = 1 ] && { echo "$out"; exit 0; }
  sleep 10
done
