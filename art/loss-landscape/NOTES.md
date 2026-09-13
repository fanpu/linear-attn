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

## Next
1. Render g51 for all 4 + atlas, view, fix. 2. lines/zoom/pca/ckfilm when data lands. 3. Merge g101, render heroes,
stl, ridge, sweep. 4. README (mandatory slice caveat, Dinh et al. 2017), commit.
