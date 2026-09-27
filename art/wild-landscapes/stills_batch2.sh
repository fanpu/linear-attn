#!/bin/bash
# second pass of gallery stills (after the critique of pass 1).  Run from wild-landscapes/.
set -e
PY=../.venv/bin/python
R="--device cuda --ss 2"
O=cache/renders
# three minima: square, steep camera, low key light (relief reads as a massif with three wells)
for al in 0 1; do
  $PY stills.py --scene three_a$al --path cache/three/plane_a${al}_r512.npz --K 0.4 --up 2 --width 3072 --aspect 1 $R \
      --exposure 1.1 --key 200,30 --kw '{"lo": -0.89, "hi": 1.16}' --orbit "0,0.0,0.15,2.1,-90,68,48" --out $O/three_sq_a$al.png
done
# ResNet-56 with / without skips, +-3 filter-normalised units, same 128^2 grid, same clip [-1, 1.5] (L <= 32)
K='{"lo": -1.0, "hi": 1.5, "pal": "tam", "gam": 1.0}'
$PY stills.py --scene field_ns --path cache/resnet/rand_resnet56_noshort_s3_r256.npz --sub 2 --up 4 --K 0.3 --width 2048 $R \
    --kw "$K" --orbit "0,0,0,2.4,-100,50,45" --out $O/resnet56_noshort_s3_r128.png
$PY stills.py --scene field_sk --path cache/resnet/rand_resnet56_s3_r128.npz --up 4 --K 0.3 --width 2048 $R \
    --kw "$K" --orbit "0,0,0,2.4,-100,50,45" --out $O/resnet56_s3_r128.png
# no-skip crater at its full 256^2, close
$PY stills.py --scene field_ns --path cache/resnet/rand_resnet56_noshort_s3_r256.npz --up 4 --K 0.3 --width 4096 $R \
    --kw "$K" --orbit "0,0.05,0,1.3,-105,55,50" --out $O/resnet56_noshort_s3_r256_close.png
# RNN hero, second composition: low raking key across the highlands
$PY stills.py --scene rnn_T30 --up 2 --width 4096 $R --orbit "0,0.2,0,1.25,-90,38,55" --key 160,18 --exposure 1.5 \
    --out $O/rnn_T30_hero_b.png
