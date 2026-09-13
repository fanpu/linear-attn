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

## HANDOFF (session 2 end, context budget) -- what exists and what to do next
Background jobs STILL RUNNING (nohup, safe to leave; check with `pgrep -af "run_|analyze"`):
- run_exact_series.sh -> cache/exact/s_mlps_C10 (last one, N=5000, ~15 min), log logs/exact_series.log
- run_lanczos_series.sh -> cache/spectra/s_mlp_C10, log logs/lanczos_series.log
- run_film.sh -> starts when exact series done: cache/exact/film_mlps_C10_pc200/step_*.npz (49 ckpts, H only,
  N=2000, ~2-3 min each => ~2 h), log logs/film.log
Written but NOT yet run/viewed:
- render_barcode.py (barcode_paper/night, gel) -- run after C=10 exists; view downscaled; fix layout.
- render_film.py --stills --film (waterfall stills in 5 styles incl. Spectral split + 1080p mp4/gif).
  Note: it defines its own islog_ with tau=1e-4 matching render_common default TAU; if you change tau, keep consistent.
  The 'split' colorize branch has dead lines (neg=..., first t=) -- clean up; check the look.
- verify.py -> prints markdown table for README (count by gap, by eigenvector overlap, chance C/P).
Rendered + viewed: gallery/plates_exact_emission.png (good; C-1 red lines clearly visible). Re-run once C=10 done:
  `OMP_NUM_THREADS=2 .venv/bin/python render_plates.py` (all 4 styles) and `--src lanczos`.
TODO after: view absorption/silver/negative; maybe a riso 2-ink plate colouring lines by ov_dc (lines in class-mean
  subspace = red ink, bulk = blue ink) -- meaningful; a palettes.py variant (e.g. P.render_split on film sheet,
  pairings 'aurora_ember'/'cyanotype_vandyke'); a Papyan hierarchy plate (H vs G vs E_eig, G1/G12 eigs in npz);
  README.md (full per BRIEF) with verify table; optional FashionMNIST C=10 exact check
  (`train.py --ds FashionMNIST --arch mlps --down 10 --C 10 --epochs 10 --lr 0.05 --name f_mlps_C10` then exact_analyze).
Headline so far: outliers = C-1 (not C), identified by eigenvectors lying in span of class-mean gradients
(overlap 0.84-0.96 vs chance C/P ~1e-3); widest-gap count fragile for C>=5-7. Wall: exact ~3-12 min/ckpt on
contended CPU (4 threads); Lanczos m=200 ~2-9 min. GPU used: 0.
Python: /home/fzeng/ml/research/art/.venv/bin/python ; run scripts from the project dir.
