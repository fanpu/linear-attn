# NOTES: loss-landscape (living handoff)

## State
- Training DONE (40 epochs, Li et al. recipe shortened, milestones 20/30/37). train/test acc:
  r20 95.6/90.4, r20ns 92.9/87.9, r56 96.8/90.9, r56ns 81.8/78.8. Checkpoints cache/ckpt/.
- Surface compute: 3 background chains (nohup setsid, each job through gpu_run.sh), logs in logs/chain{A,B,C}.log,
  scripts in jobs/chain{A,B,C}.sh. Outputs cache/surf/<model>_<final|epNNN>_<tag>.npz (resumable .partial.npz).
  - A: g51 r56ns, g51 r56, then g101 shard 0/2 for r56ns then r56 (reuses g51 points).
  - B: g51 r20, r20ns; `line` (401 pts) x4; subset check line_n5000 / line_n1000 (r56ns); zoom0..3 (α=0.5±{0.5,.05,.005,.0005},
    201 pts, zoom2/3 float64) for r56ns and r56; PCA planes pca41 (r56) and pcaf10_41 (r56ns, epochs>=10);
    then g101 shard 1/2 for r56ns, r56.
  - C: `ckline` 101-pt 1-D slices at 16 checkpoints for r56 and r56ns.
  - Merge g101 shards: `python merge_shards.py resnet56_noshort_final_g101 resnet56_final_g101` (writes only if complete).
  - Check if running: `pgrep -af landscape.py`; resume = rerun the same command (partial files resume).
- Evaluation: n=1000 fixed train subset (fixed_subset seed 0), BN eval, fp32 no TF32 (zoom2/3 fp64).
  Speed under contention: r56 ~1.65 pts/s, r20 ~4 pts/s. vmap/bf16 gave no speedup (GPU saturated).
- PCA finding: r56ns all-epoch PCA is 99.3% one PC = weight-norm collapse (epoch-1 weights at distance 412,
  WD shrinks them; BN scale-invariance). Hence PCA from epoch 10 for r56ns (92.1%/5.4%). r56 all epochs 76.6%/9.8%.
- Renderers: render_maps.py (survey, hachure, hachure_dark, hillshade, hypsometric, spectral split at chance ln10,
  hubble_sho split, atlas) — `python render_maps.py --tag g51`; render_extra.py (stl, ridge, sweep, zoom, pca, ckfilm, lines).
  Shared clip LO,HI = 0.08,150 on log10 loss.

- Surface stats: `python analyze_surf.py g51` -> cache/surf_stats_g51.json (+ cache/curv_*.npz for curvature plate).
- KEY FINDING so far: r56ns 51² slice is NOT chaotic (single elongated valley, 1 local min, basin 7% of square vs 17-22%
  for r20/r20ns; non-convex fraction inside basin ~5%). Negative vs Li et al. Fig 5; likely 40-epoch undertraining
  (81.8% train acc). README states this; zoom test + 101² hero will decide at finer spacing.
- Rendered + viewed (g51, r20/r20ns/r56ns): survey (good), hachure + hachure_dark (strong; dark crop is lovely),
  hillshade (ok after alt40/exag0.35 + faint contours), hypsometric (ok after palette fix), spectral/hubble split
  at chance ln10 (good), curvature_split_g51.png (λ_min split; red = convex), atlas_g51.png (3 of 4 so far),
  slices_1d.png (partial), ckpt_ridge/ckpt_slices smoke test only (2 epochs; rerun when chain C done).
- render_extra.py smoke-tested: ckfilm, zoom, lines, curv. NOT yet run: stl, ridge, sweep, pca (need data).
- README.md drafted with mandatory slice caveat; *pending* markers to fill.

## Running at handoff (started ~16:10 EDT Sep 13; do not relaunch while alive: `pgrep -af landscape.py`)
- chainA: r56 g51 (ETA ~17:25), then g101 shard0 r56ns, r56. chainB: 1D lines, subset check, zooms, PCA, g101 shard1.
- chainC: ckline epochs (at ep10 of 16 at 17:00; ~3 min/epoch pair).
- If a chain died: rerun its remaining lines by hand (partial files resume; finished outputs are recomputed if
  rerun, so skip those whose .npz exists).

## Next (exact commands, PY=/home/fzeng/ml/research/art/.venv/bin/python, OMP_NUM_THREADS=4)
1. When resnet56_final_g51.npz lands: `$PY analyze_surf.py g51 && $PY render_maps.py --tag g51 --models resnet56 && $PY render_maps.py --tag g51 --styles atlas && $PY render_extra.py curv --tag g51`; fill the README stats row.
2. When zooms land: `$PY render_extra.py zoom` (json in cache/zoom_test.json); `$PY render_extra.py lines` (prints subset check).
3. PCA: `$PY render_extra.py pca`. Checkpoints: `$PY render_extra.py ckfilm` after chain C.
4. Hero: `$PY merge_shards.py resnet56_noshort_final_g101 resnet56_final_g101`, then `$PY analyze_surf.py g101`,
   `$PY render_maps.py --tag g101 --models resnet56 resnet56_noshort`, `$PY render_extra.py stl|ridge|sweep --tag g101`.
   (STL files ~? MB: 241² grid -> ~116k tris ~5.8 MB each, under 20 MB.) View each downscaled; fix.
5. README: fill pending, hero -> g101 images, zoom-test roughness table, GPU time (sum meta wall_s), commit.
6. Optional if budget: retrain r56ns longer (e.g. 150 epochs) to test whether chaos appears with full training.
