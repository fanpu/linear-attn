# The Last House: grokking, drawn by hand

<video src="../gallery/handdrawn_lasthouse.mp4" controls width="720"></video>

74.5 s, 1920×1080, 24 fps, generated score. A postbird learns every door round a pond by heart and keeps a note for each
route. One letter is for a house it has never found; an old mole waits there each evening with a lamp. All autumn and
winter it knocks on wrong doors. A storm takes the notes. Alone on the ice it sees that the doors are a ring, and
flies straight to the last house. The lamp is out. In spring it can deliver anything, and the letter is still on the step.

**Top-right corner (measured):** seed 0's W_E(t) projected on the final k = 36 plane, unit-RMS per checkpoint, thread
a → a+1, drawn straight on the sky (pencil by day, starlight by night). **Declared:** which training step each beat
shows (first day 0–1.5k, the letter 1.5–2.5k, seasons 2.5–19.6k, storm 19.6–22k, seeing 22–27k, after 27–40k).
The pond, the bird, the mole and the ending are story, not measurement. Why it fits: on the plateau the ring is already
forming (R ≈ 0.92 by step 19k) while every unseen question is still answered wrong; cleanup removes what was memorised.

Files: `lasthouse.html` (brief, shots, score) · `cast.js` (bird, mole: whole-pose builders) · `set.js` (pond, houses,
seasons, night, weather, constellation) · `lettering.js` · `data.js` (from `../handdrawn/export_data.py`) · `sheet.html`
(cast sheet) · engine files copied unchanged from the hand-drawn-canvas-animation skill; `node_modules` is a symlink
to `../handdrawn/node_modules`. Render: `node render.mjs lasthouse.html` (`--grid 48` for an overview).

Known limitations: reviewed from stills and frame strips, not normal-speed playback. Flight is a two-pose flap on a
parabola; the bird has one view, mirrored. Houses are flat elevations on a three-quarter pond. Night close-ups are dim.
