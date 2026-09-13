# neural-collapse — NOTES (handoff)

## State
- `train_nc.py`: CIFAR ResNet18 (CIFAR stem), no aug, SGD m0.9 wd5e-4 lr0.05 bs128, lr/10 at E/3, 2E/3.
  Per checkpoint: stats (NC1-4, grams, SW eig) on a fixed balanced train subset + full test set,
  fp32 extraction, float64 stats; full-train acc/NC1/NC4 every 10 epochs. Stores features (f16) of
  train subset + full test set. Fractional checkpoints at iters {5..240} of epoch 1 (`itXXXX.npz`).
- Runs launched (background, via gpu_run.sh) with `./run_train.sh`:
  - `cache/c10`: 10 classes, width 32, 200 epochs, 1000/class train subset. log `logs/c10.log`
  - `cache/c4`: classes 0-3 (airplane, automobile, bird, cat), width 32, 250 epochs, 2500/class. log `logs/c4.log`
- Timing (contended GPU): 60k-image fp32 extraction ~35 s; expect ~40 s/epoch for c10.
- `cache/toy10`: old 3-epoch width-64 toy (500/class), dev only.

## Decisions
- Width 32 + 200/250 epochs to fit a few GPU-hours on a shared GPU (Papyan used width 64, 350 ep).
- C=10 projection: Procrustes of centred class means onto discrete-Fourier basis of the ideal ETF;
  plane k shows {10/k}. C=4: same basis gives a regular tetrahedron in 3D (faithful, rank 3).

## Next
- nclib.py (loading/alignment), render scripts, STL, films, README.

## Resume
- check progress: `tail -3 logs/c10.log logs/c4.log`
- if runs died: re-run `./run_train.sh` (overwrites cache/<tag>).
