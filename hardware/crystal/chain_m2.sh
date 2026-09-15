#!/usr/bin/env bash
# M2 chain: wait for the bf16 cube job to exit, queue the exclusive timing job, and once timing has started
# (it then holds the art lock) queue the optional fp32 cube behind it (share mode, skips itself if > 1 h).
D=/home/fzeng/ml/research/hardware/crystal; P=/home/fzeng/ml/research/art/.venv/bin/python; Q=/home/fzeng/ml/research/art/_shared/gpu1.sh
until grep -q "^\[gpu1 .*\] exit " $D/logs/cube_bf16.log; do sleep 30; done
echo "[chain $(date '+%F %T')] cube exited; queueing timing (exclusive)" >> $D/logs/chain_m2.log
GPU1_EXCLUSIVE=1 setsid nohup $Q env OMP_NUM_THREADS=4 $P -u $D/timing.py --dtype bf16 > $D/logs/timing_bf16.log 2>&1 < /dev/null &
until grep -q "^\[gpu1 .*\] start: " $D/logs/timing_bf16.log; do sleep 30; done
echo "[chain $(date '+%F %T')] timing started; queueing fp32 cube (--max-hours 1)" >> $D/logs/chain_m2.log
setsid nohup $Q env OMP_NUM_THREADS=4 $P -u $D/cube.py --dtype fp32 --max-hours 1 > $D/logs/cube_fp32.log 2>&1 < /dev/null &
echo "[chain $(date '+%F %T')] done" >> $D/logs/chain_m2.log
