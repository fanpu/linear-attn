#!/usr/bin/env bash
# M2 hero batch. CPU part (splats, AO) runs directly at 4 threads; the volume renders run inside the serial GPU
# queue with render_volume on CUDA (splatting stays on CPU, see render_lib.dev()).
set -euo pipefail
cd /home/fzeng/ml/research/art/invariant-tori
P=/home/fzeng/ml/research/art/.venv/bin/python
export OMP_NUM_THREADS=4
$P render_tori.py plaster glow plotter --size 2400 --R 0.008
$P render_plates.py --size 2000
R3D_VOL_DEVICE=cuda /home/fzeng/ml/research/art/_shared/gpu1.sh $P render_sea.py hero cutaway stereo --size 2400
echo M2 BATCH DONE
