# neural-collapse: NOTES (handoff)

## State (2026-09-13, session 2 end): ALL PIECES RENDERED, README written
- Training done. c10: width 32, 200 ep, 6664 s, NC1 0.044 train / 0.387 test, test acc 86.7 %.
  c4: width 32, 250 ep, 3774 s, NC1 0.0070 / 0.212, test acc 90.2 %. No background processes running.
- `cache/misfit_{c10,c4}.npz` (from `misfit.py compute`) = misfit series (train/test/W) + random-null quantiles.
- Gallery done: star plates (night/riso/spectral, k1234 at ep 0/10/200, k3 hero 3000 px, k3 8-epoch sequence),
  plotter star sheets (ink, spectral), star film (720 MP4 10 MB + GIF 11 MB; 1080 master 75 MB not committed),
  tetra series I–V (epochs 0 2 16 60 250; brass/plotter/cyanotype/spectral, view az 20 el -20) + 5 STL,
  tetra rotation film (brass MP4 16.6 MB + GIF 15.2 MB), misfit plot (paper/night), NC curves with null bands
  (paper/night), gram grids (c10 spectral, c10 means_W riso + pal_indigo_madder, c4 spectral + pal_aurora_ember).
- Film frame PNGs remain in scratch/film_c10 and scratch/filmtetra_c4_brass (gitignored; delete if disk needed).

## Key findings (in README §4)
- **Corrected null:** at the real width d=256, random Gaussian means give misfit 0.097 median (5–95 %: 0.081–0.114)
  for C=10, 0.053 (0.028–0.085) for C=4. Earlier 0.18 was for d=64.
- c10 train-means misfit 0.112 NEVER beats the d=256 null; cos-std 0.082 is worse than random 0.061. W rows 0.037 do.
- c4 train means beat the null (below median from epoch 76, final 0.029, min 0.016 @ 225); max angle error 2.07°.
- Test means never beat even the d=64 null. NC3 plateaus at ~0.18 (self-duality not reached).

## Possible next steps (optional polish)
- Tetra GIF is 15.2 MB (slightly over ~15): re-encode at scale 420 if needed (`film_tetra.py` encode args).
- Plotter/cyanotype rotation film (`python film_tetra.py c4 --style plotter`).
- Longer c10 run (350 ep, width 64) would be needed for a star that beats the random null.
- Commit: `/home/fzeng/ml/research/art/_shared/commit.sh neural-collapse "..."`.

## Code map
train_nc.py (compute), nclib.py (load_run, fourier_etf, Aligner Procrustes), artlib.py (splat/glow/save+preview),
misfit.py, render_curves.py, render_stars.py (auto exposure; `d["norm"]` fixes it), plotter_stars.py,
film_stars.py, render_tetra.py (+ --stl), film_tetra.py, render_gram.py (styles spectral|riso|pal_<pairing>).
