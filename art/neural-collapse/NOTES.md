# neural-collapse — NOTES (handoff)

## State (2026-09-13, session 1 end)
- **Training runs launched but still QUEUED** behind other agents' jobs (all 8 gpu_run slots were busy).
  Launched with `./run_train.sh` (background; two `gpu_run.sh` waiters). Logs stay empty until a slot
  is acquired (first line `[gpu_run] slot N acquired`).
  - `cache/c10`: all 10 CIFAR classes, ResNet18 width 32, 200 epochs, stats on 1000/class train subset. `logs/c10.log`
  - `cache/c4`: classes 0-3 (airplane, automobile, bird, cat), width 32, 250 epochs, 2500/class subset. `logs/c4.log`
  - Check: `ps aux | grep train_nc | grep -v grep`; `tail -n 3 logs/c4.log`. If the waiters died
    (no processes, empty logs): re-run `nohup ./run_train.sh &` (overwrites cache/<tag>).
- Expected cost on the contended GPU: ~40 s/epoch c10 (+~12 s per checkpoint measure), ~16 s/epoch c4.
  c10 ≈ 2.5-3 h, c4 ≈ 1.2 h. Record actual wall time from `metrics.jsonl` "time" field for README.
- `cache/toy10`: old 3-epoch width-64 toy (500/class, 512-d), dev only; all renderers ran on it OK.

## Code (all CPU renderers read cache/, write gallery/ + a downscaled preview to scratch/prev_*.png)
- `train_nc.py` compute. Per checkpoint npz: mu, muG, mu_test, W, b, gram_M/gram_W/gram_Mtest, sv_M,
  cov_sub, SW_eig, h_train (subset, f16), h_test (full test set, f16). metrics.jsonl: nc1, nc1_test,
  {M,W,Mtest}_{equinorm,cos_std,cos_dev}, nc3, acc/loss/nc4 train+test, full_* every 10 epochs.
  Checkpoints: every epoch 0-30, every 2 to 100, every 5 after, + it0005..it0240 inside epoch 1.
  Model state saved at E/3, 2E/3, E.
- `nclib.py`: `load_run(tag)`, `fourier_etf(C)` (orthonormal DFT basis of 1-perp; rows = ideal ETF
  vertices; `planes[k]` column idx), `Aligner(mu, muG)` = SVD basis of centred means + orthogonal
  Procrustes onto the Fourier ETF + one global scale (isometry; `.residual` = relative Procrustes misfit,
  0 for exact ETF, verified 8e-16), `pick_ckpts`.
- `artlib.py`: splat / glow / tonemap / exact-pixel canvas / save+preview.
- `render_stars.py tag --epoch E --style night|riso|spectral [--planes 1 2 3 4] [--res]`:
  2x2 plate of Fourier planes k=1..4 ({10/1},{10/2}=2 pentagons,{10/3},{10/4}=2 pentagrams; k=5 is a
  line, not drawn), or single plane hero with `--planes 3`. night = class hue (colorcet cyclic, declared)
  additive glow + measured star (bright) over ideal star (faint). riso = train fluo-pink vs test blue,
  4px/1200 declared misregistration. spectral = per-sample logit margin (correct − best other) split
  Sohl-Dickstein (red=misclassified, purple=correct, dark=near boundary). Planned: `plotter` style
  (line-only stars of means over ~40 checkpoints, Molnár-like) — NOT yet written.
- `render_tetra.py c4 --epochs e1..e5 --style brass|plotter|cyanotype|spectral` (series I–V plate with
  captions: 6 pairwise angles, max dev from 109.47°); `--stl` writes gallery/stl/*.stl (rods+spheres+hub,
  ideal circumradius 50 mm). spectral colours edges by signed angle deviation. UNTESTED (needs c4 data).
- `render_curves.py c10 c4 --style paper|night`: NC1–NC4 + error small multiples, log epoch. Tested on toy.
- `render_gram.py tag --pair train_test|means_W --style spectral|riso`: grid of CxC cosine tiles over
  log-spaced checkpoints; upper triangle / lower triangle = the pair; split at −1/(C−1), global per-side
  rank normalisation. Tested on toy (looks good; late tiles will darken toward the seam = ETF).

## Findings so far
- Null model for the star projection: 10 random Gaussian means in 64-d already give Procrustes
  residual 0.18 (random high-d vectors are near-equiangular). Toy epoch 3: 0.54; init: 0.93.
  README must show residual vs epoch against this null (and init) so stars aren't over-claimed.
- Toy init gram: cos_std 0.62 (features at init share a direction), so init is far from ETF.

## Next steps
1. When checkpoints appear: render stars/curves/gram on intermediate data; view previews; tune gain/EXT.
2. c4: choose I–V epochs (e.g. init, ~it40, 1, 10, final, or by residual quantiles); render all 4 tetra
   styles + STL; rotation MP4/GIF of final frame (brass) and a morph-through-epochs rotation film.
3. `film_stars.py`: 1080² 2x2 night planes, per-sample linear tween between consecutive checkpoints
   (declared), ~8 frames/ckpt, 30 fps, ffmpeg H.264 yuv420p + GIF (palettegen). Use <=4 workers.
4. Plotter star style; train-vs-test diptych; residual-vs-epoch plot with null band.
5. README (BRIEF structure), verification numbers, caveats (width 32 not 64, 200/250 epochs, one lr, bf16
   training forward, subset stats, projection of samples is lossy for C=4 clouds, Procrustes allows
   reflection, class order in Fourier planes is a declared choice — ETF is S_C-symmetric).
6. Commit via `/home/fzeng/ml/research/art/_shared/commit.sh neural-collapse "..."`.
