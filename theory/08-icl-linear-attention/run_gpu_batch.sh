#!/usr/bin/env bash
# One GPU slot, two small jobs side by side: the task-diversity sweep and the rest of the sequence-model grid.
cd "$(dirname "$0")"
rm -f cache/taskdiv_main_s0.json cache/taskdiv_main_s0.ckpt 2>/dev/null || true
../.venv/bin/python taskdiv.py --logM 0,2,4,6,7,8,9,10,11,12,14,16,inf --batch 256 --steps 8000 --eval_every 2000 --tag main > logs/taskdiv_main.log 2>&1 &
./run_seq_grid.sh "gdelta delta softmax linear" > logs/seq_grid_C.log 2>&1
wait
