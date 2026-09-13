# PAUSED (wind-down requested)

## Done
- **Core math and tests.** `dd_core.py` holds the closed forms and simulators; `test_core.py` passes.
- **Reproductions, rendered in `figures/`:**
  - linear model: the family, sizes, bias/variance and ridge plates;
  - random features vs Mei–Montanari (`rf_mm.png`);
  - anisotropy (`aniso.png`);
  - frozen vs trained two-layer net (`twolayer.png`).
- **Animations:** `hero.mp4` and `mp_edge.mp4`.
- **Widgets:** `widgets/rf.js` and `widgets/risk.js`. Their self-tests pass at `post.html#selftest`.
- **Post draft:** `post.md` is complete except for §6.1 (the CNN), marked `CNN_SECTION_PLACEHOLDER`, and `CNN_HOURS` in "Reproduce it".

## Interrupted
The CNN sweep was killed.
- Widths 5, 12, 24, 32 and 48 finished all 500 epochs.
- The other widths (1, 2, 3, 4, 6, 8, 10, 16, 64) stopped at epoch 427. Their last checkpoint is at epoch 425.
- Resume with `_shared/gpu_run.sh bash 01-double-descent/run_cnn.sh main 10000 0.2 500`. It skips finished widths quickly.
- About 1.75 GPU-slot-hours have been used so far.

## Next steps
1. Resume the CNN sweep to 500 epochs.
2. Render the CNN figures: `python render_cnn.py all` and `python render_cnn_anim.py`.
3. Fix the peak detector.
   - The epoch-to-epoch test-error jitter for wide nets is about 0.028.
   - The model-wise bump at epoch 268 is only about 0.026, at k≈8 (the EMC=n width there is 7.4). It grows with training.
   - Use `smooth_epochs(..., epochs=E)` together with a prominence rule. Consider extending widths 4–24 to about 1000 epochs.
4. Write §6.1, using the heatmap, slices, EMC scatter and animation. Report honestly that the bump is weak at 500 epochs.
5. Replace ★ in `render_cnn.py`; the font lacks that glyph.
6. Add `compute_twolayer.py` / `render_twolayer.py` to "Reproduce it".
7. Do a final render and screenshot check, then write the final report.
