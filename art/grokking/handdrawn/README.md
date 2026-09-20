# The Winder: grokking, drawn by hand

<video src="../gallery/handdrawn_winder.mp4" controls width="720"></video>

A 46.5 s pencil film (1080², 24 fps, with a generated score). A small winder strings one thread through 113 nails in
counting order, 0 → 1 → … → 112. It is a tangle, so every answer goes on a paper tag (memorisation). Through a long
night the nails creep (circuit formation). Then a wind (weight decay) takes the tags, and the same thread pulls
taut into the star polygon {113/36} (cleanup).

## Measured
- **Nails:** seed 0's W_E(t) projected on the final (cos, sin) plane of key frequency k = 36, centred and scaled to
  unit RMS per checkpoint, exactly as in `../render_film.py`. 283 checkpoints, linear interpolation between them.
- **Pinned chart and step counter:** measured train/test loss for seed 0 (log scale), drawn only up to the current step.
- **Nail numbers:** the token a. In the coda, neighbours round the ring differ by 22 = 36⁻¹ mod 113.

## Declared (metaphor, not measurement)
- The film-time → training-step map: winding and tagging sit at step 1,500 (the declared end of pure memorisation,
  `phases.py` t_circ); night runs 1,500 → 19,600; the wind 19,600 → 27,000; the coda 27,000 → 40,000.
- Thread slack, the 34 tags (30% of nails, echoing the 30% train fraction), the wind, the winder, and the score.
- k = 36 only: its ring is already R ≈ 0.92 by step 19k, so the star is half-there under the tags before the wind.
  That is honest to the mechanism (the circuit forms during the plateau; cleanup only reveals it).

## Files
`winder-film.html` (brief, beat sheet, shots, score) · `winder.js` (whole-pose character builder) · `poses.js`
(authored keys) · `board.js` (nails, thread, tags, wind, chart) · `lettering.js` (single-stroke path alphabet, no
fonts needed) · `export_data.py` → `data.js` · `sheet.html` (pose sheet) · `core.js`, `studio.js`, `cels.js`,
`materials.js`, `render.mjs` copied unchanged from the hand-drawn-canvas-animation skill.

```
../../.venv/bin/python export_data.py          # needs ../cache/analysis.npz
npm i --no-audit --no-fund
node render.mjs winder-film.html --grid 48     # overview; --strip START,COUNT for motion QA
node render.mjs winder-film.html               # out/winder-film-final.mp4 (+ score wav, contact sheet)
```

## Known limitations
- Motion was reviewed from consecutive-frame strips and stills, not normal-speed playback (headless session).
- Drawings are programmatically authored poses (a rig locates elbows and knees; contours are built per pose),
  not scans of hand animation. The winder only exists in one right-facing view.
- Nails, thread and tags run on ones while the figure is on twos; nail numbers collide where the real embedding
  clusters points in pairs (nothing is nudged).
