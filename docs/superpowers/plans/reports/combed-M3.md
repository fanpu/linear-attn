# §8 Combed — M3 report (film, caption fixes, README)

**Status:** DONE.

## What was done

1. **Caption re-flow.** `render_common.caption_strip` gained a `min_frac` floor (default 0.016, a 1.6% safety margin above the controller's 1.4% ruling). Font size is solved by a 4-iteration fixed point against the *final* (panel + caption-strip) image height, since the strip's height depends on the font size in turn.
   - Checked numerically before touching any renders: the hair diptych (2048 px panel, scale 1.0) already sat at 1.57% of final height under the OLD formula, and the stereo pairs (1600 px, scale 0.9) at 1.47% — both already compliant, so they were left untouched.
   - The tube images (2400 px, scale 0.75) and basin images (2400 px, scale 0.8) were at 0.61–0.65% — well under the ruling. Re-ran `render_tubes.py tubes --size 2400 --ns 64 1024` (16 images, 35 s) and `render_basin.py --res 128|256 --size 2400` (10 images, 13 s + 21 s). All now sit at ≥1.6% (spot-checked: tubes fs=44/2699≈1.63%). No code change was needed in the callers — the new floor is automatic.
   - `basin_slices_R{128,256}.png` (scale 1.6) was already far above the floor (≈3.3%) and unchanged.
2. **Film.** New `render_film.py`: N-sweep diptych (closed-form left, trained MLP right), N = 16, 64, 256, 1024, 4096, hard cuts between segments (no flow interpolation), camera continuously rotating az −60° → +60° over the whole film (same VIEW parameters/pattern for every N, satisfying "same camera"). Reuses `render_hair.glow_panel` unchanged, per-frame camera override. For CPU budget each frame draws 4,000 of the shared 20,000 seeds (declared in-frame; render cost scales linearly in seed count — 14–18 s/panel at 20k vs. 3.2 s/panel at 4k at film resolution). 50 frames/N × 5 N = 250 frames, panel 520 px (1064×584 total with a 24 px gap and caption strip), 10 fps → 25 s. Each frame prints N and the memorised fraction for both fields with the fresh-knot null (reused `R.memo_numbers`, itself reading `cache/summary.json` and the dense-run `end_d1`/`end_d2` at t = 1 − 10⁻⁶).
   - `--test` mode (4 frames, 2 N values) verified the pipeline end-to-end before the full run.
   - Full run: 1712 s (28.5 min) CPU, 4 threads — under the ~45 min budget.
   - `r3d.write_film` produced `gallery/film_N_sweep.mp4` (17.5 MB, under the 20 MB commit cap) at fps=10. The default `gif_width=480` GIF came out at 22.4 MB (over the 20 MB cap commit.sh enforces); re-encoded the GIF only (no frame re-render) at `gif_width=360` → 12.3 MB, and swapped it in.
3. **README.** Written per the controller's shape: hero (`hair_diptych_N256.png` + the film), "The phenomenon" (Bertrand/Gu/Scarvelis vs. Kadkhodaie), "The pieces" (one caption block per image group, each stating measured vs. declared), "What was computed" (wall-clock table spanning M1–M3), "Verification" (continuity equation, step doubling, t→1 check, memorised-fraction-vs-N table with the null, basin box counting D=2.13 vs. planar null 2.10 over 1.2 decades, no fractal claim), "Caveats" (Gu criterion misfires on 1D manifolds; width/budget fixed not swept; trained fraction below null explained; ortho stereo is rotation stereo), "References" (all four papers).
   - Link check: `grep -o 'gallery/[^")> ]*' README.md | sort -u | while read f; do [ -e "$f" ] || echo MISSING $f; done` → clean. Note for future editors: the check does not stop on backtick/comma/period, only on `"`, `)`, `>`, and space — every gallery path in the README is therefore wrapped in `<img src="...">`/`<video src="...">` or `[text](gallery/...)` markdown-link syntax, never bare backtick-quoted prose, to avoid false MISSING hits.
   - `NOTES.md` updated with the M3 decision log and resume commands.

## Numbers (unchanged from M1/M2, reproduced in the README verification table)

| N | closed-form | trained MLP | null (fresh knot) |
|---|---|---|---|
| 16 | 1.000 | 0.704 | 0.263 |
| 64 | 1.000 | 0.434 | 0.367 |
| 256 | 1.000 | 0.183 | 0.313 |
| 1024 | 0.998 | 0.049 | 0.333 |
| 4096 | 0.998 | 0.005 | 0.345 |

Basin box counting (N = 16, closed-form): D = 2.13 (256³, ε 1–16 vox) / 2.16 (128³, ε 1–8 vox); planar-Voronoi null 2.10 / 2.13; boundary-voxel ratio 256³/128³ = 4.11 (surface ≈ 4). Fit spans 1.2 decades — declared, no fractal claim.

## Decisions (logged in `NOTES.md`)

1. Caption floor = 1.6% of final image height (controller ruling ≥1.4%), solved by fixed point in `caption_strip`; hair diptych/stereo left as-is since already compliant.
2. Film shows 4,000 of 20,000 shared seeds per frame for CPU budget (declared); static diptychs unaffected (still 20,000).
3. Hard cuts + continuous camera rotation across the whole film, rather than cross-fades, to avoid any appearance of interpolating between measured flows at different N.
4. GIF re-encoded at `gif_width=360` (not the 480 used elsewhere in the piece) to stay under the 20 MB commit cap; the MP4 needed no re-encode (17.5 MB at CRF 18, the `write_film` default).

## Risks / open items

- The film's per-frame 4,000-seed subsample is a declared speed compromise; at N = 16 with only 4,000 of 20,000 seeds some tufts read slightly sparser than in the 20,000-seed static diptych. The caption states the seed count so this is not silently smoothed over.
- `cache/frames_film/` (209 MB, 250 PNGs) is gitignored, local only, like the rest of `cache/`.
- Box-counting fit range is still 1.2 decades (a pre-existing M2 item, carried into the README's Verification section with the same declaration; extending it would need a 512³ basin grid, ~90 min GPU, not done here since the controller's M1/M2 gates already accepted the 1.2-decade result with the planar-null comparison).

## Files

- New: `art/combed/render_film.py`, `art/combed/README.md`, `art/combed/gallery/film_N_sweep.mp4`, `art/combed/gallery/film_N_sweep.gif`.
- Modified: `art/combed/render_common.py` (caption floor), `art/combed/NOTES.md` (M3 log), 8 basin PNGs + 16 tube PNGs (re-rendered with the new caption floor; pixel content otherwise identical).
- Cache (gitignored, not committed): `art/combed/cache/frames_film/` (209 MB).

## Commands (from `art/combed/`)
```
OMP_NUM_THREADS=4 ../.venv/bin/python render_tubes.py tubes --size 2400 --ns 64 1024
OMP_NUM_THREADS=4 ../.venv/bin/python render_basin.py --res 128 --size 2400
OMP_NUM_THREADS=4 ../.venv/bin/python render_basin.py --res 256 --size 2400
OMP_NUM_THREADS=4 ../.venv/bin/python render_film.py --test          # smoke test, 4 frames
OMP_NUM_THREADS=4 ../.venv/bin/python render_film.py --device cpu    # full film, ~29 min
../.venv/bin/python -m pytest -q test_combed.py                      # 7 pass, unaffected by M3
```
