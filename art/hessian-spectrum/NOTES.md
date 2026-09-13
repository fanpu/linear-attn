# hessian-spectrum NOTES (living handoff)

## State
- Session 2 started 2026-09-13. GPU saturated -> everything on CPU (OMP/torch threads <= 8).
- Existing from session 1: common.py (HVP/GN-VP/Lanczos/SLQ/Papyan), train.py, analyze.py (GPU-oriented),
  toy run cache/runs/toy_mnist_mlp (3 epochs) + cache/spectra/toy_mnist_mlp/step_001404.npz.
  Toy H top Ritz: 6.03 4.59 3.90 3.48 3.21 2.72 2.32 1.99 1.51 | 1.04 0.99 0.94 ... (9 above a gap, C=10).

## Plan
1. CPU-ify train/analyze (device arg), class-count series C=2,3,4,5,7,10 MNIST MLP (+FashionMNIST check).
2. Outlier count: log-gap criterion on top Ritz values of H and G, cross-check vs Papyan G1 eigenvalues and
   overlap of top eigenvectors with span of class-mean gradients.
3. Pieces: class-count plate stack (emission / absorption), barcode, spectrograph-over-training (waterfall + mp4),
   Spectral signed split variant, palettes.py variant.
