#!/usr/bin/env bash
# Run one stage of the day's sweep, one run after another. [AI]
# after nanochat's speedrun.sh: one script runs the list end to end.
#
#     nohup experiments/day4_mqar/run.sh lr > run_lr.log 2>&1 &
#
# Finished runs (summary.json present) are skipped by train_mqar.py, so the
# same command can be re-launched after an interruption.
set -euo pipefail
cd "$(dirname "$0")/../.."

export TORCH_CUDA_ARCH_LIST="12.1a"          # GB10 (sm_121); the environment fla's kernels were verified in
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-$(dirname "$(which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)")/ptxas}"

STAGE="${1:?usage: run.sh <smoke|lr|scan|scan512|seeds|gate|pivot>}"
python experiments/day4_mqar/sweep.py "$STAGE" | while read -r cfg seed out; do
  echo "=== $(date '+%F %T') $out"
  mkdir -p "$out"
  python scripts/train_mqar.py --config "$cfg" --seed "$seed" --out "$out" 2>&1 | tee -a "$out/stdout.log"
done
echo "=== $(date '+%F %T') stage $STAGE done"
