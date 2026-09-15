#!/usr/bin/env bash
# M2 driver (CPU side; launch with setsid nohup). Controller rule: no gpu1.sh job longer than ~20 min.
#  1. bf16 cube in segments of <= 18 min (cube.py --budget-min 18), one gpu1.sh job each, resuming per k-slab;
#  2. the timing volume as one exclusive job (GPU1_EXCLUSIVE=1, ~35 min);
#  3. optional fp32 cube in segments, which writes SKIPPED.json and stops if its projection exceeds 1 h.
D=/home/fzeng/ml/research/hardware/crystal; P=/home/fzeng/ml/research/art/.venv/bin/python; Q=/home/fzeng/ml/research/art/_shared/gpu1.sh
log() { echo "[run_m2 $(date '+%F %T')] $*" >> $D/logs/run_m2.log; }
segments() {   # $1 dtype, $2 extra args
  local dt=$1; shift; local seg=0 stall=0 before after
  while :; do
    before=$(ls $D/cache/cube_$dt/k*.npz 2>/dev/null | wc -l)
    [ "$before" -ge 256 ] && { log "$dt cube complete"; return 0; }
    [ -e $D/cache/cube_$dt/SKIPPED.json ] && { log "$dt cube skipped (SKIPPED.json)"; return 0; }
    seg=$((seg + 1)); log "$dt segment $seg queued ($before/256 slabs done)"
    $Q env OMP_NUM_THREADS=4 $P -u $D/cube.py --dtype $dt --budget-min 18 "$@" > $D/logs/cube_${dt}_seg$(printf %02d $seg).log 2>&1 < /dev/null
    rc=$?; after=$(ls $D/cache/cube_$dt/k*.npz 2>/dev/null | wc -l)
    log "$dt segment $seg exit $rc, $after/256 slabs"
    if [ "$after" -le "$before" ]; then stall=$((stall + 1)); [ $stall -ge 2 ] && { log "$dt: no progress twice, stopping"; return 1; }
    else stall=0; fi
  done
}
log "start"
segments bf16
log "timing queued (GPU1_EXCLUSIVE=1)"
GPU1_EXCLUSIVE=1 $Q env OMP_NUM_THREADS=4 $P -u $D/timing.py --dtype bf16 > $D/logs/timing_bf16.log 2>&1 < /dev/null
log "timing exit $?"
segments fp32 --max-hours 1
log "done"
