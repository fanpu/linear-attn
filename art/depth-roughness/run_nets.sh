#!/bin/bash
cd /home/fzeng/ml/research/art/depth-roughness
PY=/home/fzeng/ml/research/art/.venv/bin/python
$PY compute_nets.py B 256,1024,4096,16384 2
$PY compute_nets.py B 64,256,1024,4096,16384 1
$PY compute_nets.py A
$PY compute_nets.py B 65536 2
