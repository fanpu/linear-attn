#!/usr/bin/env bash
# Part II campaign, sequential (one GPU job at a time: running invariance + divergence together OOM'd the
# unified pool on 2026-09-14 05:28). Launch: setsid nohup ./run_part2.sh > logs/part2.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
PY=/home/fzeng/ml/research/art/.venv/bin/python
mkdir -p logs
log() { echo "$(date '+%F %T') $*" | tee -a logs/part2.log; }
idle() { while [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)" -gt 5 ]; do sleep 10; done; sleep 5; }
export OMP_NUM_THREADS=8

log "== fingerprint invariance (resume)"
( cd fingerprint && $PY invariance.py --bmax 512 --out cache/inv.npz 2>&1 | grep -v USDT ) >> fingerprint/logs/inv_full2.log 2>&1
log "invariance exit $?"
for p in feynman story sky; do
  idle; log "== fingerprint divergence $p"
  ( cd fingerprint && $PY divergence.py --prompt $p --L 320 --out cache/div_$p.npz 2>&1 | grep -v USDT ) >> fingerprint/logs/div_$p.log 2>&1
  log "divergence $p exit $?"
done
touch fingerprint/logs/div_done.flag

idle; log "== roofline measure"
( cd roofline && $PY measure.py --out cache/roofline.npz 2>&1 | grep -v USDT ) >> roofline/logs/measure.log 2>&1
log "roofline exit $?"

idle; log "== staircase campaign"
( cd staircase && ./run_all.sh ) >> staircase/logs/run_all.out 2>&1
log "staircase exit $?"

idle; log "== pulse phases"
( cd pulse && $PY record.py --tag phases 2>&1 | grep -v USDT ) >> pulse/logs/phases.log 2>&1
log "pulse phases exit $?"
idle; log "== pulse soak"
( cd pulse && $PY record.py --tag soak --soak 900 2>&1 | grep -v USDT ) >> pulse/logs/soak.log 2>&1
log "pulse soak exit $?"
log "== ALL DONE"
