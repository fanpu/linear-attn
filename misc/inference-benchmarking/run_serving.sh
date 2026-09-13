#!/bin/bash
# Phase 2: serving latency, one model at a time.
cd "$(dirname "$0")"
for m in Qwen/Qwen3-8B Qwen/Qwen3-30B-A3B; do
  echo "=== $m $(date) ==="
  /usr/bin/python3 -u serve_bench.py --model "$m"
done
echo "SERVING DONE $(date)"
