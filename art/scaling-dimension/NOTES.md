# scaling-dimension NOTES (living handoff)

Python: `/home/fzeng/ml/research/art/.venv/bin/python` (call it PY). Run everything from this directory.

## State (2026-09-13 16:15, agent 2): the project is essentially complete
- No background jobs are running. Both sweeps are finished:
  - `cache/main/ts_main.npz`: 10 widths × 8 d × 2 families × 3 seeds, 72 min.
  - `cache/real/real.json`: cifar10, fmnist, mnist × 8 widths, 1 seed.
- Analysis: `cache/main/ts_main_analysis.{json,npz}` (ts_analyze.py). The summary table is in `cache/main/summary.json` and `summary_table.md`. That table was built by an inline snippet, and the same numbers are in the README.
- The README is complete: 8 sections, the full results table, real-data table, disagreements, reproduce commands.
- Gallery, all committed:
  - `diptych_`, `fan_`, `agree_` × {paper, dark, riso, spectral}.png
  - `zoom_{dark,paper,riso}.mp4` (720², CRF 32, 9–15 MB) and `.gif` (420 px, 10 fps, 8–9 MB).
  - The 1080² masters `zoom_*_1080.mp4` (85–96 MB) stay local and are NOT committed (over 20 MB). The README says so.

## Key numbers
- relu0, 4/α vs student TwoNN ID:
  - d=2: 2.42 / 2.01
  - d=3: 3.09 / 2.97
  - d=4: 4.02 / 3.94
  - d=5: 4.41 / 4.86
  - d=6: 5.16 / 5.84
  - d=8: 6.63 / 7.55
  - d=10: 7.30 / 9.36
  - d=12: 8.46 / 10.73
- relub is similar except at low d (d=2: 3.42, d=3: 3.68).
- Slope through the origin: 4/α ≈ 0.84·ID.
- Fit-range sensitivity: the small-N half and the large-N half differ by ±30–40% (the L(N) curves are concave). The seed-bootstrap CIs are too narrow with 3 seeds.
- Real data, 4/α (raw CE power law) vs hidden-layer TwoNN:
  - MNIST: 7.3 vs 8.9
  - FMNIST: 19.7 vs 8.9
  - CIFAR-10: 27.5 vs 11.8 (the paper reports a CIFAR match and FMNIST 5.95 vs 9.4)
  - Only MNIST agrees. The README lists the likely causes: CE floor, 12 epochs, 1 seed, 2c bottleneck.
- GPT-2 doc claim: checked, see the README (4/α ≈ 53; first-layer ID 50–80; other layers > 90).

## Scripts
- `ts_train.py`, `ts_analyze.py`, `idlib.py` (+ `test_idlib.py`), `real_train.py`
- `render_plates.py --pieces diptych,fan,agree --styles paper,dark,riso,spectral --idw 45 --dpi 180` (renders from cache, about 1 min)
- `zoom_compute.py` then `render_zoom.py` (commands are in the README). Re-encode for commit:
  - MP4: `ffmpeg -i zoom_S_1080.mp4 -vf scale=720:720 -crf 32 -tune grain`
  - GIF: fps 10, 420 px, 64 colours, bayer 5
- `style.py`: the spectral style now uses a charcoal ground (#18191f), so the pale Spectral midtones stay visible.

## Optional next steps (only if more budget)
1. Diptych right panel: the d = 8/10/12 labels crowd at the top. Place the labels by angle instead of at count = 2000.
2. Real data: longer training (50 epochs, 3+ seeds), and fit L − L∞ or use the error rate over the clean range; then re-render agree.
3. Longer T/S training (240k steps, as in the paper) to test whether the large-N concavity is an optimisation floor.
4. Zoom film: an L(N) inset whose markers light up as each student peels off.
