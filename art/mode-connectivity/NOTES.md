# NOTES: mode-connectivity (One Basin)

Living handoff. cache/ and logs/ are gitignored (local only). Python: /home/fzeng/ml/research/art/.venv/bin/python

## Pipeline (compute -> cache -> render)
- `common.py`: data, MLP (functional dicts), Adam training, weight matching (Git Re-Basin Alg.1, scipy LAP, records sweep history), REPAIR for MLPs, Bezier training, batched eval (`eval_stack`/`eval_list`, per-class loss).
- `compute_hero.py <ds>`: width-512 pair (seeds 0,1), 20 ep Adam 1e-3; weights -> cache/hero_<ds>_weights.pt; paths (201 pts, train/test, per-class) + predictions + emergence (28 ckpts) -> cache/hero_<ds>.npz.
- `compute_width.py <ds>`: widths 32..2048, 3 pairs, 25-pt lerps naive/matched/+REPAIR -> cache/width_<ds>.npz (resumable) + pair-0 weights.
- `compute_planes.py hero|width <ds>`: 2-D loss planes -> cache/planes_{hero,width}_<ds>.npz. Toy timing: 41² width128 = 21 s with 10k test.
- `analyze_units.py <ds>` (CPU): similarity matrices, matched cosines, per-sweep test barrier -> cache/units_<ds>.npz.
- Renderers: `render_triptych.py`, `render_planes.py`, `render_units.py` (all tested on a toy `--tag _toy`, width 128/2 epochs). Previews in gallery/preview (gitignored).

## State
- GPU jobs queued via gpu_run.sh since 12:50 (slots full of other agents): hero mnist, hero fmnist, width mnist. Logs: logs/*.out, cache/*.log.
- Toy result (w128, 2 ep): naive barrier 1.38, matched 0.15, bezier 0.08, matched+REPAIR 0.057.

## Next
1. When hero done: `compute_planes.py hero mnist --res 161 --ntrain 10000` (bg, gpu_run), analyze_units, renders.
2. Width planes: `compute_planes.py width mnist --res 81 --ntrain 5000`.
3. Renders still to write: width series (ridgeline + plane grid), emergence film, path-walk film, sorting film.
