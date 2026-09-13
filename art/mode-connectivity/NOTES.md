# NOTES: mode-connectivity (One Basin)

Living handoff. cache/ and logs/ are gitignored (local only). Python: /home/fzeng/ml/research/art/.venv/bin/python

## Pipeline (compute -> cache -> render)
- `common.py`: data, MLP (functional dicts), Adam training, weight matching (Git Re-Basin Alg.1, scipy LAP, records sweep history), REPAIR for MLPs, Bezier training, batched eval (`eval_stack`/`eval_list`, per-class loss).
- `compute_hero.py <ds>`: width-512 pair (seeds 0,1), 20 ep Adam 1e-3; weights -> cache/hero_<ds>_weights.pt; paths (201 pts, train/test, per-class) + predictions + emergence (28 ckpts) -> cache/hero_<ds>.npz.
- `compute_width.py <ds>`: widths 32..2048, 3 pairs, 25-pt lerps naive/matched/+REPAIR -> cache/width_<ds>.npz (resumable) + pair-0 weights.
- `compute_planes.py hero|width <ds>`: 2-D loss planes -> cache/planes_{hero,width}_<ds>.npz. Toy timing: 41² width128 = 21 s with 10k test.
- `analyze_units.py <ds>` (CPU): similarity matrices, matched cosines, per-sweep test barrier -> cache/units_<ds>.npz.
- Renderers: `render_triptych.py`, `render_planes.py`, `render_units.py` (all tested on a toy `--tag _toy`, width 128/2 epochs). Previews in gallery/preview (gitignored).

## State (14:00)
- GPU queue was full for 25+ min; the Grace CPU is FAST for these MLPs, so most compute runs on CPU: `MC_DEV=cpu OMP_NUM_THREADS=4 python ...` (common.DEV reads MC_DEV).
- DONE hero mnist (cache/hero_mnist.npz, 3061 s CPU): naive barrier 1.399 train / 1.328 test; matched 0.0055/0; Bezier 0/0 (mid-def 0.0012); matched+REPAIR 0.0014; naive+REPAIR 0.106; self B->pi(B) 1.355. WM 13 sweeps; sweep 1 already gives test barrier 0.007. Emergence: ~0 at init, naive peaks 1.45, matched falls 0.28 (ep .5) -> 0.006 (ep 20).
- DONE units_mnist.npz; renders done+viewed: triptych (4 styles, x100 exaggeration strip), perm ink/riso, similarity night/paper, quilt spectral/vik/riso, walk film, sorting film (vik), emergence strata paper/night. Emergence film rendering (logs/r_emerg.out).
- RUNNING: width mnist CPU (32..1024 done/doing; KILL it if it starts 2048 - `grep "width 2048" cache/width_mnist.log`); width 2048 GPU job via gpu_run (tag _2048 -> cache/width_mnist_2048.npz, render_width merges); planes hero mnist CPU res 121, 5k train/5k test (logs/planes_hero_mnist.out); hero fmnist CPU.
- Width matched barrier (train, mean): 32: 0.64, 64: 0.27, 128: 0.08, 256: 0.012, 512: 0.003, 1024 pair0: 0.0000.
- README.md drafted with GALLERY_TBD / COMPUTED_TBD / VERIFY_TBD tokens.

## Next
1. planes done -> `python render_planes.py mnist --plane perm --styles spectral,aurora,topo,night` (+ `--plane bezier`), view.
2. width done -> `compute_planes.py width mnist --res 65 --ntrain 3000` (CPU), then `render_width.py mnist`.
3. fmnist: analyze_units, triptych, planes (optional), walk.
4. Fill README tokens; commit; final report.
