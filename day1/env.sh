# Day 1 environment — source this before anything else
export TORCH_CUDA_ARCH_LIST="12.1a"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export MAX_JOBS=4          # cap any from-source build parallelism (OOM guard)
export TRITON_CACHE_DIR=$HOME/.triton/day1
