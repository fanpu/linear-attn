# PAUSED (2026-09-13, ~03:55)

## Done
- Core math + tests (`saxe_core.py`, `test_core.py`, 5 tests pass).
- All compute finished and cached: `cache/reproduce.npz`, `semantic.npz`, `break_{init,imbalance,whiten,depth}.npz`,
  `attn_{showcase,sweep_rank1,sweep_mode3}.npz`, `attn_reduced_*.npz`. GPU used so far: ~16 min.
- Figures: `modes_overlay.png`, `landscape.png`, `breaks.png`, `depth.png`, `attn_showcase.png`, `attn_scaling.png`;
  animations `hero_tree.mp4`, `matrix_sharpening.mp4`, `attention_saddles.mp4`, `landscape_swarm.mp4` (previous version).
- Widgets `widgets/saddle.js`, `widgets/modes.js` (self-tests pass, zero console errors at last check).
- Full post draft `post.md` (no TODOs left), `README.md`.

## Interrupted
- Re-render of the swarm animation (text backdrop + loop fade-in/out; code change already in `render_landscape.py`).
  The committed `figures/landscape_swarm.mp4` is the older, valid version. Rerun:
  `cd theory && .venv/bin/python 03-saxe-dynamics/render_landscape.py --anim`
- Final page render + screenshots + console check (post.html predates the attention-animation insertion). Rerun:
  `cd theory && .venv/bin/python _shared/render_post.py 03-saxe-dynamics/post.md --shot --slices 20`
  then the headless-Chrome console check with `post.html#selftest`.

## Next steps (in order)
1. Rerun the swarm animation (above); look at 3 extracted frames.
2. `.venv/bin/python 03-saxe-dynamics/render_breaks.py` (picks up the last y-limit tweak on panel c).
3. Render post with `--slices 20`, read every `_preview/page_*.png`, fix layout issues; run console/self-test check.
4. Second widget pass: exercise depth 4 + random init + extreme sliders in the modes widget; check the saddle demo button.
5. Proofread post numbers against caches (all numbers in text were computed; recheck after any re-render).
6. Delete `PAUSED.md`, final commit, final report (citation notes: Saxe 2014 uses sums over P examples and tau = 1/lambda,
   depth eq. counts N_l layers of neurons = N_l - 1 matrices; Saxe 2019 8-item data came from a hierarchical diffusion
   process, ours is hand-built; Boix-Adsera theory assumes diagonal weights).
