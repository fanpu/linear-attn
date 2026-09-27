#!/bin/bash
# All gallery stills (GPU, one pasar job).  Run from wild-landscapes/.
set -e
PY=../.venv/bin/python
R="--device cuda --ss 2"
O=cache/renders
# unrolled RNN: horizon triptych, same camera / light / mapping as the hero
for T in 10 30 100; do
  $PY stills.py --scene rnn_T$T --path cache/rnn/plane_T${T}_r2049_s2.5.npz --K 0.25 --up 1 --width 2048 $R \
      --orbit "0,0,0,1.5,-120,55,55" --exposure 1.4 --out $O/rnn_horizon_T$T.png
done
# tiny nets: parity-4, 4 hidden units, whole plane (+-5) and the knot (+-0.8)
$PY stills.py --scene tinysign_p4 --path cache/tiny/p4_h4_rand_s5_r4096.npz --K 0.12 --width 4096 $R --exposure 1.2 \
    --kw '{"chroma": 0.09, "Llo": 0.14, "Lhi": 0.9}' --orbit "0,0,0,1.5,-100,52,50" --out $O/tiny_p4_s5.png
$PY stills.py --scene tinysign_p4 --path cache/tiny/p4_h4_rand_s0.8_r4096.npz --K 0.12 --width 4096 $R --exposure 1.2 \
    --kw '{"chroma": 0.09, "Llo": 0.14, "Lhi": 0.9}' --orbit "0,0,0,1.8,-100,50,45" --out $O/tiny_p4_s0.8.png
# three minima, raw and permutation-aligned, identical mapping
for al in 0 1; do
  $PY stills.py --scene three_a$al --path cache/three/plane_a${al}_r512.npz --K 0.3 --up 2 --width 4096 $R --exposure 1.0 \
      --kw '{"lo": -0.89, "hi": 1.16}' --orbit "0,0.15,0,2.9,-90,48,42" --out $O/three_a$al.png
done
# Hessian vs random directions (ResNet-56, same 128 images, same clip)
for f in rand_resnet56_s1_r128 hess_resnet56_s0.05_r128; do
  $PY stills.py --scene field_$f --path cache/resnet/$f.npz --up 4 --K 0.25 --width 2048 $R \
      --kw '{"lo": -1.0, "hi": 2.2, "pal": "aurora", "gam": 0.6}' --orbit "0,0,0,2.6,-100,42,45" --out $O/$f.png
done
# transformer vs convnet (same K per decade, own clip = own range)
$PY stills.py --scene field_gpt --path cache/gpt/plane_adamw_s1_r128.npz --up 4 --K 0.25 --width 2048 $R \
    --kw '{"key": "acc", "pal": "aurora", "clo": 0.0, "chi": 0.63, "gam": 0.6}' --orbit "0,0,0,2.6,-100,42,45" --out $O/gpt_adamw_s1_r128.png
