# weight-spectrum NOTES (living handoff)

## State (2026-09-13 ~16:30, agent 2): ESSENTIALLY COMPLETE
- ALL training runs finished (bs8..1024 seed 0; bs16..1024 seed 1; main art run mlp_bs16_s1). No background processes left.
  (The `train.py --depth 56` processes visible in ps belong to another agent.)
- All CPU; GPU time 0. Total CPU wall ~20,000 s.
- Final pieces rendered from mlp_bs16_s1 FC1 (38 files in gallery/), README.md fully written (hero, gallery, table, verification, caveats, refs).
- Model: MLP 784-1024-1024-1024-10 ReLU, Glorot normal, FashionMNIST, SGD lr .01 mom .9, no wd, 30 epochs.

### Agent-2 changes
- `render_verify.py`: batch sheet grouped by bs (dots = seeds, line = mean, hollow alpha where n_out<5); caveat scatter with Spearman
  (rho(mean alpha, test acc) = -0.16 over 13 reliable runs, 0.00 for bs16-256; gap rho 0.45); CCDF xlim clipped.
- `render_ridgeline.py`: style suffix `_log` (e.g. `joy_log`) forces log-spaced rows -> `..._<style>_logtime.png`. joy_logtime = HERO
  (58 unique ckpts, cleaner than 80 linear rows). Caption now states linear/log row spacing.
- `render_plate.py`: title glyph fix (serif font lacked U+1D40; now mathtext $W^{T}W/N$).
- GIF re-encode (frames kept in cache/frames/ridge_mlp_bs16_s1_FC1_{joy,gold}, esd_...night):
  `ffmpeg -framerate 30 -i cache/frames/ridge_mlp_bs16_s1_FC1_joy/%05d.png -vf "fps=10,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=8:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle"`
  (max_colors=32 for gold). joy 8.6 MB, gold 13.7 MB. Default frames_to_video GIFs were 42 MB.
- Bigger picture (bs8_s0): alpha FC1 1.70 but test acc .886 / train .947 (too noisy to fit); bs1024 under-trained (.860/.884), FC1 lmax at MP edge.

## Optional next steps (not required)
- seed-1 for bs8 (60 min CPU) to complete the replicate table; ink/gold ridge film styles for the ESD film (`render_esd_film.py ... ink` if the style exists).
- After committing, frames in cache/frames can be deleted (~2k PNGs).

## Key numbers (seed 0, final ckpt; full table with seeds + bs8/512/1024 in README; `python render_verify.py` prints them + writes cache/metrics_table.json)
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

