# NOTES: mode-connectivity (One Basin)

Living handoff. cache/ and logs/ are gitignored (local only). Python: /home/fzeng/ml/research/art/.venv/bin/python

## Pipeline (compute -> cache -> render)
- `common.py`: data, MLP (functional dicts), Adam training, weight matching (Git Re-Basin Alg.1, scipy LAP, records sweep history), REPAIR for MLPs, Bezier training, batched eval (`eval_stack`/`eval_list`, per-class loss).
- `compute_hero.py <ds>`: width-512 pair (seeds 0,1), 20 ep Adam 1e-3; weights -> cache/hero_<ds>_weights.pt; paths (201 pts, train/test, per-class) + predictions + emergence (28 ckpts) -> cache/hero_<ds>.npz.
- `compute_width.py <ds>`: widths 32..2048, 3 pairs, 25-pt lerps naive/matched/+REPAIR -> cache/width_<ds>.npz (resumable) + pair-0 weights.
- `compute_planes.py hero|width <ds>`: 2-D loss planes -> cache/planes_{hero,width}_<ds>.npz. Toy timing: 41² width128 = 21 s with 10k test.
- `analyze_units.py <ds>` (CPU): similarity matrices, matched cosines, per-sweep test barrier -> cache/units_<ds>.npz.
- Renderers: `render_triptych.py`, `render_planes.py`, `render_units.py` (all tested on a toy `--tag _toy`, width 128/2 epochs). Previews in gallery/preview (gitignored).

## State (15:05, session 1 HANDOFF)
Everything below is DONE, rendered, viewed and committed unless listed under "Running" / "Next".
- Hero MNIST + FMNIST paths, width series 32..2048 (3 pairs; 1024/2048 in cache/width_mnist_{1024,2048}.npz, merged by render_width), units_{mnist,fmnist}.npz, hero perm plane (planes_hero_mnist.npz, 141^2, key perm_*).
- Gallery (all viewed): triptych_{mnist,fmnist}_{survey,night,spectral,riso}; plane_mnist_perm_{spectral_basin,aurora_basin,topo,night}; width_ridge_mnist_{night,paper}, width_barrier_mnist; emergence_mnist_strata_{paper,night} + emergence_mnist.mp4/.gif; perm_mnist_{ink,riso}; similarity_{mnist,fmnist}_{night,paper}; quilt_{mnist,fmnist}_{vik,riso,spectral}; twins_mnist; walk_mnist.mp4/.gif; sorting_mnist.mp4/.gif.
- README.md is essentially complete (sections 1-6 with verified numbers). Remaining tokens: BEZIER_PLANE_TBD, WIDTH_PLANES_TBD, PLANE_TIME_TBD, WPLANE_TIME_TBD.
- Key numbers: MNIST naive barrier 1.399 train, matched 0.0055, Bezier 0 (mid-def 0.0012), matched+REPAIR 0.0014, B->pi(B) 1.355. Width matched: 0.644/0.267/0.081/0.012/0.003/0.001/0.000. Emergence: matched 0.92 (ep .05) -> 0.006; pi_final on early ckpts lower than pi_k. FMNIST: naive 1.427, matched 0.050.
- GPU time so far: width1024 521 s, width2048 1821 s, hero perm plane 1590 s (+ bezier/bezm planes running). CPU: hero mnist 3061 s, width 32-512 ~45 min, fmnist ~1 h+.

## Running (left running on purpose; check, do not relaunch)
- GPU `compute_planes.py hero mnist --res 141 ... --planes perm,bezier,bezm` (logs/planes_hero_mnist.out): perm done; bezier (+curve) and bezm (train only) pending; npz rewritten after each plane.
- GPU queued `compute_planes.py width mnist --res 97 --ntrain 5000` (logs/planes_width_mnist.out -> cache/planes_width_mnist.npz, resumable).
- CPU `compute_hero.py fmnist` in emergence stage (cache/hero_fmnist.log; writes full npz at "done hero_fmnist").
Beware: `pkill -f <pattern>` in a Bash call whose command line contains the pattern kills the shell itself; use pgrep, then kill PID in a separate call.

## Next (resume commands; PY=/home/fzeng/ml/research/art/.venv/bin/python, OMP_NUM_THREADS=2, run in this dir)
1. After "plane bezier" in logs/planes_hero_mnist.out: `$PY render_planes.py mnist --plane bezier --styles spectral,topo` (and `--plane bezm` if done). View previews in gallery/preview/. Replace README token BEZIER_PLANE_TBD with an <img> table + caption (seam = highest loss on the curve), fill PLANE_TIME_TBD from the log.
2. After planes_width finishes: `$PY render_width.py mnist --pieces planes` -> width_planes_mnist_{spectral,topo}.png; view; replace WIDTH_PLANES_TBD; fill WPLANE_TIME_TBD.
3. After "done hero_fmnist": `$PY render_emergence.py fmnist --pieces strata` (optional film), `$PY render_walk.py fmnist` (optional); add links in README.
4. `grep TBD README.md` must be empty; check every gallery link exists; commit via _shared/commit.sh; final report (<=200 words: key numbers, 3 strongest: plane_mnist_perm_spectral_basin.png, triptych_mnist_spectral.png, emergence_mnist_strata_night.png / walk_mnist.mp4; negatives: width 32-64 not LMC, REPAIR hurts naive at w64, chance seam useless, spectral quilt confetti, FMNIST matched-Bezier test barrier 0.17, no CIFAR).
