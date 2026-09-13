#!/usr/bin/env bash
# Production compute for piece A ("Four Roots", depth-3 scalar factorisation). Run via gpu_run.sh.
set -u
cd /home/fzeng/ml/research/art/outcome-basins
PY=/home/fzeng/ml/research/art/.venv/bin/python
C="-1.1986301369863015 -1.7054794520547945 -1.1429794520547947 -1.7936643835616437 -1.1709118150684934 -1.7764340753424657 -1.1722629494863017 -1.772996040239726 -1.1723381983090757 -1.7725729746361301 -1.1723438419707837 -1.7725213455827269 -1.1723416210853892 -1.7725280082389101 -1.1723414479869687 -1.772528788814806 -1.1723414614592513 -1.772528692875823"
# heroes (4096^2)
$PY compute_map.py fact3 --s 0.5 --eta 1.1 --win -3.5 3.5 -3.5 3.5 --res 4096 --tag hero_f3_s0.5_e1.1
$PY compute_map.py fact3 --s 0.0 --eta 1.25 --win -3.5 3.5 -3.5 3.5 --res 4096 --tag hero_f3_s0_e1.25
$PY compute_map.py fact3 --s 1.5 --eta 0.8 --win -3.5 3.5 -3.5 3.5 --res 4096 --tag hero_f3_s1.5_e0.8
$PY compute_map.py fact3 --s 0.0 --eta 0.3 --win -3.5 3.5 -3.5 3.5 --res 4096 --tag hero_f3_s0_e0.3
# eta small multiples (2048^2) and null (gradient-flow-like) regime
$PY compute_map.py fact3 --s 0.5 --eta 0.005 0.1 0.4 0.7 0.9 1.0 1.1 1.2 1.3 --win -3.5 3.5 -3.5 3.5 --res 2048 --tag eta_f3
$PY compute_map.py fact3 --s 0.5 --eta 0.005 --win -1.5 1.5 -1.5 1.5 --res 4097 --tag null_f3_res4097
# zoom levels (2048^2) on fixed boundary centres; null zoom at small eta (auto centres)
$PY compute_zoom.py fact3 --s 0.5 --eta 1.1 --win -3.5 3.5 -3.5 3.5 --levels 10 --factor 8 --res 2048 --centers $C --tag zA
$PY compute_zoom.py fact3 --s 0.5 --eta 0.005 --win -1.5 1.5 -1.5 1.5 --levels 5 --factor 8 --res 2048 --tag znull
# resolution check: level-3 window at 1025 / 2049 / 4097 (coarse samples coincide with fine ones)
for R in 1025 2049 4097; do
  $PY compute_map.py fact3 --s 0.5 --eta 1.1 --win -1.1777477525684934 -1.1640758775684934 -1.7832700128424657 -1.7695981378424657 --res $R --tag res_f3_L3_R$R
done
# uncertainty exponent: fringe window, deep stripe window, null
$PY compute_uncert.py fact3 --s 0.5 --eta 1.1 --win -3.5 3.5 -3.5 3.5 --eps -1 -13 25 --tag f3_e1.1_full
$PY compute_uncert.py fact3 --s 0.5 --eta 1.1 --win -1.6361 -0.7611 -2.1430 -1.2680 --eps -1 -13 25 --tag f3_e1.1_L1
$PY compute_uncert.py fact3 --s 0.5 --eta 0.005 --win -1.5 1.5 -1.5 1.5 --eps -1 -13 25 --tag f3_null
# films
$PY compute_frames.py fact3 eta --s 0.5 --eta0 0.02 --eta1 1.32 --n 240 --win -3.5 3.5 -3.5 3.5 --res 1080 --tag f3_etafilm
$PY compute_frames.py fact3 zoom --s 0.5 --eta 1.1 --center -1.1723414614592513 -1.772528692875823 --start_center 0 0 --half0 3.5 --half1 2.5e-8 --n 420 --res 1080 --tag f3_zoomfilm
echo PROD_FACT3_DONE
