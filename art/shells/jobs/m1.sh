#!/usr/bin/env bash
# M1 GPU chain (one gpu1.sh job): GPU checks -> pick K -> random-direction 27^3 -> PCA-direction 27^3.
# Rerunning resumes: volume.py skips finished z-slabs; check.py is skipped once cache/check_cuda.json exists.
#   setsid nohup /home/fzeng/ml/research/art/_shared/gpu1.sh bash jobs/m1.sh > logs/m1.log 2>&1 < /dev/null &
set -euo pipefail
cd /home/fzeng/ml/research/art/shells
export OMP_NUM_THREADS=4
PY=/home/fzeng/ml/research/art/.venv/bin/python
if [ ! -f cache/check_cuda.json ]; then
  $PY check.py --device cuda --n 50 --nslice 50 --K 16 --Ks 1 4 8 16 32 64
fi
read K IB < <($PY - <<'PYEOF'
import json
r = json.load(open("cache/check_cuda.json"))
assert r["equiv_pass"], "vmap equivalence failed at K=16"
ok = {int(k): v for k, v in r["throughput_vs_K"].items() if not v.get("oom") and v["equiv_max_rel"] <= 1e-5}
k = max(ok, key=lambda k: ok[k]["pts_per_s"])
print(k, ok[k]["img_batch"])
PYEOF
)
echo "chosen K=$K img_batch=$IB"
$PY volume.py --dirs random --K "$K" --img-batch "$IB"
$PY volume.py --dirs pca --K "$K" --img-batch "$IB"
