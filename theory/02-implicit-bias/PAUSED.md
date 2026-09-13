# PAUSED (clean checkpoint)

## Done
- Everything in `post.md` is written: hero, §1–7, the build-on sections (Adam, Muon), "Reproduce it" and References. `post.html` renders, and both widgets pass `#selftest` with 0 console errors.
- All compute is finished and cached in `cache/`:
  - Soudry: 1e8 GD steps, NGD / sign GD variants
  - diagonal nets
  - matrix completion and Razin–Cohen
  - Adam (geom, gauss, β₂ sweep)
  - spectral/Muon: `compute_spectral.py 1e6`, `compute_ns.py 1e6`
- All figures are rendered: 4 animations, 9 static plates, 2 widgets.
- Commits: e799058, 8ac3311, ab06627, 9852d53, 9ade214 (+ this one).

## Interrupted
- Nothing was running. The last edit was the staggered depth labels in `widgets/diagnet.js`. It is committed but its screenshot hasn't been checked. To check it:
  `../.venv/bin/python ../_shared/render_post.py post.md --shot --slices 16` and look at `_preview/page_05.png`.

## Next steps (in order)
1. Look at the diagnet side-panel labels in the fresh screenshot.
2. Do a final proofread pass on `post.md`, checking the numbers against `render_*.py` printouts.
3. Optionally trim `hero.gif` (6 MB; the post uses the MP4).
4. Write the final report (≤250 words) described in `_shared/BRIEF.md`.
