#!/usr/bin/env bash
# M3 batch: CPU stills first (tori re-render for the shared frame, slice, plates), then one GPU-queue job for the
# sea stills (enlarged box, sheet fade, 1.5 deg stereo), the eps-sweep film and the turntable (frames checkpointed).
set -euo pipefail
cd /home/fzeng/ml/research/art/invariant-tori
P=/home/fzeng/ml/research/art/.venv/bin/python
export OMP_NUM_THREADS=4
$P render_tori.py plaster glow plotter slice --size 2400 --R 0.008
convert -density 72 gallery/tori_plotter.svg gallery/tori_plotter_raster.png
$P render_plates.py --size 2000
$P render_secondpole.py --size 1600
R3D_VOL_DEVICE=cuda R3D_DEVICE=cuda /home/fzeng/ml/research/art/_shared/gpu1.sh bash -c \
  "$P render_sea.py hero cutaway stereo --size 2400 && $P render_film.py sweep --size 1080 && $P render_film.py turntable --size 1080"
echo M3 BATCH DONE
