#!/usr/bin/env bash
# M2 final renders (GPU, via gpu1.sh). Each plate is one process, so a rerun skips finished plates.
set -e
cd /home/fzeng/ml/research/art/ribbon
export OMP_NUM_THREADS=4
P=/home/fzeng/ml/research/art/.venv/bin/python
$P render_ribbon.py hero --size 256 --device cuda --out-suffix _gputest
rm -f gallery/hero_glow_gputest.png
[ -e gallery/hero_glow.png ] || $P render_ribbon.py hero --size 2400 --device cuda
[ -e gallery/honesty_fixed_vs_moving.png ] || $P render_ribbon.py honesty --size 1600 --device cuda
[ -e gallery/stereo_crosseye.png ] || $P render_ribbon.py stereo --size 1400 --device cuda
[ -e gallery/context_eos_glow.png ] || $P render_ribbon.py context --size 3000 --device cuda
echo "m2 renders done"
