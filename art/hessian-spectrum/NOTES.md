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

## Key numbers so far (final checkpoints, per_class=500 Hessian subset)
- exact (mlps, 10x10 MNIST, P~4.4k): top eigvecs in span{d_c} (overlap>0.5): C=2:1, 3:2, 4:3, 5:4, 7:6  => C-1 lines.
  widest-log-gap count agrees for C<=5, ambiguous at C=7 (k=7 gap 1.47 vs k=2 1.46). So plates use the
  eigenvector-overlap count ("structural lines"), gap count reported in verify table.
- Lanczos (mlp 784-128-128, P~118k, m=200): C=2:1, 3:2, 4:3, 5:4 (gap count says 1), 7:6 by overlap.
- Papyan G1 eigenvalues are ~10-30x smaller than the H outliers (directions match, magnitudes don't:
  within-class gradient variance adds along the same directions at low loss).
## Decisions
- Plates: symlog tau=2e-3 (render_plates.py --tau), photographic exposure model 1-exp(-density/0.7), every
  eigenvalue a Gaussian line sigma=1.3px. Hue by x-position (declared spectroscope idiom).
## Jobs (nohup, logs/): run_exact_series.sh, run_lanczos_series.sh, run_film.sh (waits for exact series)
