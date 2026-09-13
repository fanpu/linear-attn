# weight-spectrum NOTES (living handoff)

## State (2026-09-13 ~14:40, agent 1 handed off at context budget)
- GPU saturated -> all training on CPU, single-threaded processes (1 thread is FASTER than 6 under load ~35).
- Model: MLP 784-1024-1024-1024-10 ReLU, Glorot normal, FashionMNIST (global mean/std norm), SGD lr .01 mom .9, no wd, 30 epochs.
- `train.py` additions: `--device cpu --threads`, `--n_lin` (extra linearly spaced ckpts), shuffled-entries null ESD
  (`<layer>/lam_shuf`), fine loss trace. `cache/<run>.npz` is written only at the END of a run; full W in `cache/<run>_W/`.

### Runs (logs/*.log; all nohup, safe to leave running; check `tail -qn1 logs/*.log`)
| run | status | notes |
|---|---|---|
| mlp_bs32/64/128/256_s0 | done | 60 log ckpts, 6 full W |
| mlp_bs16_s0 | running (ETA ~15:20) | 120 log ckpts, 24 full W |
| mlp_bs8_s0 | running (ETA ~16:00) | 80 log ckpts, 12 full W |
| mlp_bs512_s0, mlp_bs1024_s0 | running (short) | from `run_series.sh` |
| mlp_bs16_s1 | running (ETA ~16:15) | **MAIN ART RUN**: 150 linear + 50 log ckpts (197), 12 full W |
| mlp_bs{1024,512,256,128,64,32}_s1 | queued by `run_seed1_queue.sh` (keeps <=5-6 procs) | replicates for error bars |
- A test render `render_ridge_film.py mlp_bs32_s0 FC1 joy` was running in background (log logs/render_ridgefilm_bs32.log) — check
  gallery/ridge_film_mlp_bs32_s0_FC1_joy.mp4/gif; re-render with the main run later.

## Key numbers so far (seed 0, final ckpt; `python render_verify.py` prints them + writes cache/metrics_table.json)
| bs | test acc | alpha FC1/FC2/FC3 | lmax/MP edge FC1/FC2/FC3 | eigs above null edge FC1/FC2/FC3 |
|---|---|---|---|---|
| 32 | .8975 | 2.26 / 2.98 / 3.57 | 6.2 / 4.7 / 3.7 | 46 / 39 / 25 |
| 64 | .8955 | 2.84 / 4.01 / 4.05 | 4.2 / 2.1 / 2.8 | 37 / 28 / 16 |
| 128 | .8973 | 4.33 / 5.51 / 4.40 | 2.5 / 1.6 / 2.6 | 25 / 18 / 11 |
| 256 | .8951 | 7.42 / 6.76 / 5.08 | 1.5 / 1.4 / 2.4 | 14 / 11 / 9 |
- Init: alpha fit on MP matrices = 7-13, shuffled null 5-19: alpha is meaningless in the random phase (fits jump 5-30 early).
  Films/plots only show alpha once >=5 eigenvalues exceed the null edge.
- Tail monotone with batch size; test acc flat (~0.895-0.898) -> no alpha-generalization relation visible here (caveat piece).
- GOTCHA found: shuffled null keeps the global entry mean; FC2/FC3 weights drift negative (FC2 bs32 mean -3.6e-3) -> rank-1
  spike mu^2 M in the null (2.7x MP edge). Fixed: `common.null_edge` = null's SECOND largest eigenvalue. Mention in README.
- Entry kurtosis of trained W is small (<=1.2), dW kurtosis 2-4: departure is correlation structure, not heavy-tailed entries.
- Most spectral change happens in the last decade of log-steps -> linear-time rows for art (bs16_s1).
- dW = W_T - W_0 is noise-dominated (SGD random walk) -> weave/fields textures look like noise (weak pieces);
  FC1 top right singular vectors become garment-like after ~1k steps (garments piece is the good eigenvector piece).
- Critique refs verified to exist: Kothapalli et al. 2024 arXiv:2406.04657 (HT ESD without gradient noise, via large lr);
  "Eigenspectrum Analysis of Neural Networks without Aspect Ratio Bias" arXiv:2506.06280 (alpha biased by Q). MM 2018 arXiv:1810.01075
  (5+1 phases: random-like, bleeding-out, bulk+spikes, bulk-decay, heavy-tailed, rank-collapse; batch-size series with MiniAlexNet).

## Code (render from cache; each prints output paths; previews at cache/preview/)
- `common.py` loader, MP law, Clauset fit, `metrics(run, layer)` (lru_cached).
- `render_common.py` styles, save+preview, `frames_to_video`, `riso_composite`, palettes import (`R.P`).
- `render_ridgeline.py run layer [joy ink gold spectral riso]` — AXIS='sv' (x = sqrt(lambda) linear) is the good one, use FC1.
  Ridges picked linear in step if run has n_lin else log from step 100. Looked great on bs128 FC1 (tail bumps).
- `render_ridge_film.py run layer [joy gold ink]` — live bottom ridge, copies drift up; verified frame looks strong.
- `render_plate.py run layer [magma paper riso bio spectral]` — spectrograph plate, outlier streamers; good.
- `render_esd_film.py run layer [night ink]` (--only k for one frame) — chart-like explanatory film, works.
- `render_riso_plate.py run [inkA inkB]` — MP (inkA) vs ESD bars (inkB), 3 layers x 5 ckpts; works.
- `render_departure.py run layer` — log rho_ESD - log rho_null field; rank split (sd_spectral, labelled: amplifies frozen noise
  stripes) + linear aurora_ember (honest). 
- `render_texture.py run [weave fields garments ipr]` — garments good; weave/fields noise-dominated.
- `render_verify.py [main_run]` — over-training sheet, CCDF+fit, batch series, caveat scatter.
- NB: editing these files with sed made tool noise; use the Edit tool.

## Next steps
1. When bs16_s0 / bs8_s0 / bs16_s1 finish: `render_verify.py mlp_bs16_s1`; stills for FC1 (and FC2/FC3 plates):
   ridgeline (all 5 styles) on mlp_bs16_s1 and mlp_bs8_s0; plate magma/paper/riso/bio; riso plate 3 ink pairs; departure; garments; ipr.
2. Films (use <=3 workers while training runs): `render_ridge_film.py mlp_bs16_s1 FC1 joy` (+gold), `render_esd_film.py mlp_bs16_s1 FC1 night` (+ink).
   GIFs must be <15 MB (reduce gif_width/fps if not).
3. After seed-1 replicates finish: batch sheet with mean±range over seeds (edit sheet_batch to group by bs).
4. Review every image downscaled; iterate on ridgeline + plate (the strongest).
5. Finish README.md (skeleton exists with __HERO__/__GALLERY__ placeholders): gallery with captions (measured vs declared),
   what was computed + exact commands (`run_series.sh`, `run_seed1_queue.sh`, bs8/bs16_s1 commands in this file's run table:
   `train.py --name mlp_bs16_s1 --data fmnist --widths 1024,1024,1024 --bs 16 --lr 0.01 --momentum 0.9 --epochs 30 --n_ckpt 50 --n_lin 150 --n_full 12 --k_vec 32 --seed 1 --device cpu --threads 1`;
   bs8: same with `--bs 8 --n_ckpt 80 --n_full 12 --seed 0`, no n_lin), verification table, null-mean gotcha, caveats
   (contested generalization, confounded batch series, alpha estimator bias with Q and n_tail, fixed lr), references. No fractal claims.
6. Commit: `/home/fzeng/ml/research/art/_shared/commit.sh weight-spectrum "..."`. GPU time used: 0 (all CPU).
