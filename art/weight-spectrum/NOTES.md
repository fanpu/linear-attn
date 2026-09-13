# weight-spectrum NOTES (living handoff)

## State
- GPU saturated -> all training on CPU, single-threaded processes (1 thread is FASTER than 6 under load ~40).
- `train.py`: added `--device cpu --threads`, shuffled-entries null ESD (`<layer>/lam_shuf`), fine loss trace.
- Batch-size series running (`run_series.sh`, logs in `logs/`): MLP 784-1024-1024-1024-10, FashionMNIST, SGD lr .01 mom .9,
  30 epochs, Glorot normal, seed 0, bs in {16 (120 ckpts, 24 full W), 32, 64, 128, 256, 512, 1024 (60 ckpts)} + bs8 (80 ckpts) launched separately.
  Outputs: `cache/mlp_bs{bs}_s0.npz` (only written at the END of a run) + `cache/mlp_bs{bs}_s0_W/step*.npz` full W.
- `common.py`: loader, MP law, Clauset power-law fit (xmin by KS), `metrics(run, layer)`.
- `render_common.py`: styles, save+preview (cache/preview), frames->mp4/gif, riso composite.
- `render_ridgeline.py` (joy/ink/gold/spectral/riso) and `render_esd_film.py` (night/ink) written, layout-tested on toy run.

## Next
- When runs finish: metrics table (alpha vs bs), tune ridgeline x-range/bandwidth on real data, films,
  riso MP-vs-ESD plate, W texture / FC1 singular vectors (28x28) / IPR, departure spectrogram (Spectral split), verification plots, README.
