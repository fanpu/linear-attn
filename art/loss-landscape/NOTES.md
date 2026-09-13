# NOTES: loss-landscape (living handoff)

## State (Sep 13 ~18:50 EDT): round COMPLETE, nothing running
- Training (40 epochs, milestones 20/30/37) train/test acc: r20 95.6/90.4, r20ns 92.9/87.9, r56 96.8/90.9, r56ns 81.8/78.8.
- All surfaces in cache/surf: g51 x4, g101 r56 + r56ns (merged from shards), line x4, line_n1000/n5000, zoom0-3 x2,
  pca41 (r56), pcaf10_41 (r56ns), ckline 16 epochs x2. jobs/chainD.sh reran PCA planes (chainB failed: torch.load
  weights_only; fixed in landscape.py line 51).
- Stats: cache/surf_stats_g51.json, surf_stats_g101.json, zoom_test.json. README fully filled (no *pending*), link check clean.
- Renders (all viewed downscaled): g101 survey/hachure/hachure_dark/hillshade(new: aspect shading + Tanaka contours)/
  hypsometric/spectral/hubble pairs; g51 all four re-rendered; curvature_split_g51/g101; stl/*.stl (11.6 MB each)
  + stl_preview_g101 (fixed meshgrid bug); ridge_{dark,paper}_* (fixed zorder, inverted: low loss up, 41 rows);
  sweep_g101.mp4/gif (13.7 MB); zoom_test.png + zoom_roughness.png; pca_trail_* (grey = loss>150, dashed decades);
  ckpt_ridge.png + ckpt_slices.mp4/gif (24 blend frames between epochs).

## Key findings
- r56ns slice smooth at every scale: 101² matches cubic interp of 51² (median 4e-5, max 0.008 log10); zoom windows
  monotone at half-widths 0.05..0.0005, range scales linearly; RMS Δ² ∝ s^1.61 (r56) / s^1.49 (ns) over 3 finest
  (tiny ~1e-9 excess texture, likely ReLU kinks/noise; NOT no-shortcut specific). No fractal claim.
- Reproduces qualitatively: noshort basin 7.3% vs 17.2% area, floor 0.53 vs 0.093, non-convex in basin 5.3% vs 0.3%.
- Subset check n1000 vs n5000: median |log10 ratio| 0.0064, max 0.030.
- Likely cause of mismatch vs Li et al.: 40-epoch undertraining (81.8% train acc). Untested.
- GPU: training ~1.5 h, surfaces ~4.2 h slot wall.

## Commands (PY=/home/fzeng/ml/research/art/.venv/bin/python, OMP_NUM_THREADS=4)
- `$PY analyze_surf.py g101; $PY render_maps.py --tag g101 --models resnet56 resnet56_noshort`
- `$PY render_extra.py {stl|ridge|sweep|curv|zoom|lines|pca|ckfilm} --tag g101`

## Possible next steps (optional)
- Retrain r56ns 150-300 epochs (train.py has no resume; ~2-4 h GPU) and redo g51 + zoom to test the undertraining hypothesis.
- Better strongest-images: hachure_dark pair, spectral pair, ridge_dark, sweep film.
