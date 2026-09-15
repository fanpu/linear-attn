#!/usr/bin/env bash
# Run every (config, seed) pair that sweep.py lists, in order.  [AI-owned]
# after karpathy/nanochat speedrun.sh: one shell script runs the whole study end to end.
#
#     nohup experiments/day3_seed_variance/run.sh > run.log 2>&1 &
#
# Each run writes runs/<mixer>_<size>_s<seed>/ (config.toml, env.json,
# metrics.jsonl, stdout.log, checkpoints, summary.json). A run whose
# summary.json exists is skipped and a run whose ckpt_latest.pt exists resumes,
# so the same command is safe to re-launch after a crash or reboot.
set -u
cd "$(dirname "$0")/../.."
export TORCH_CUDA_ARCH_LIST="12.1a"          # GB10 (sm_121)
# export TRITON_PTXAS_PATH=...               # set it here if your box needs it
python experiments/day3_seed_variance/sweep.py | while read -r config seed; do
  name="$(basename "$config" .toml)_s${seed}"
  out="runs/${name}"
  if [ -f "$out/summary.json" ]; then echo "[run.sh] $out done, skipping"; continue; fi
  mkdir -p "$out"
  echo "[run.sh] $(date '+%F %T') starting $out"
  python scripts/train.py --config "$config" --seed "$seed" --out "$out" 2>&1 | tee -a "$out/stdout.log"
  echo "[run.sh] $(date '+%F %T') finished $out (exit ${PIPESTATUS[0]})"
done
echo "[run.sh] $(date '+%F %T') all runs finished"
