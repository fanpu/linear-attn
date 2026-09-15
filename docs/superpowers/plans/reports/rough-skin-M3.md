# Rough Skin (§7) — M3 report: film, objects, README

**Status: DONE.** Film, STL objects and README are complete, the README's link check is clean, and everything is committed. Directory: `/home/fzeng/ml/research/art/rough-skin/` (code, README.md, NOTES.md; `cache/` gitignored, 2.7 GB).

Commit: `7d6ac34` — `art/rough-skin: M3 -- depth-dial + turntable film, Heaviside/ReLU zero-set STLs, measured-D diptych captions, README`.

## What was done (commands, from the piece directory, `PY=art/.venv/bin/python`)

```bash
$PY render_diptych.py exterior ; $PY render_diptych.py cutaway     # re-render with measured D beside theory
$PY make_stl.py                                                    # 6.1 s
setsid nohup ../_shared/gpu1.sh $PY render_film.py all --size 1080 --device cuda > logs/film.log 2>&1 < /dev/null &
# queued ~45 min behind solid-edge's B256 volume; ran 08:04:32-08:12:57 (8.4 min GPU)
ffmpeg -y -c:v libx264 -crf 23 -pix_fmt yuv420p gallery/film_rough_skin.mp4   # CRF 18 -> 23, 21.34 -> 13.13 MB
cd art/rough-skin && grep -o 'gallery/[^")> ]*' README.md | sort -u | while read f; do [ -e "$f" ] || echo MISSING $f; done   # clean
../_shared/commit.sh rough-skin "art/rough-skin: M3 -- depth-dial + turntable film, ..."
```

## 1. Diptych re-render (controller ruling)

`render_diptych.py`'s Heaviside caption now reads "D = 2.426 ± 0.031 (theory 2.5)" (previously "theory dim 3 − 1/2 = 2.5" only), with the slice value (2.448 ± 0.067) and the 3-draw/calibration protocol moved into the italic subtitle. Shortened to fit the half-canvas width (checked with `ImageDraw.textlength`); the original, shorter wording had fit without checking. Re-rendered both `diptych_heaviside_L1_vs_relu.png` and its cutaway twin; viewed both, no overlap.

## 2. STL objects

`make_stl.py`, 6.1 s total:

| name | grid | why | faces | watertight | size (mm) | file |
|---|---|---|---|---|---|---|
| Heaviside L = 1 | **96³, native re-evaluation** | the 128³ cache-subsample mesh is 584,426 faces / ~29.2 MB, over the 20 MB budget; no mesh-decimation library is installed (`trimesh`/`open3d`/`pyvista` all absent from `art/.venv`), so per the M3 ruling's fallback this uses a fresh 96³ evaluation of the same weights (seed 7) instead of decimating | 291,782 | True | 80 × 80 × 80 | 14.59 MB |
| ReLU L = 1 | 128³, cache subsample (even nodes of the cached 256³ field) | fits the budget as-is (smooth surface, far fewer faces) | 153,872 | True | 80 × 80 × 71.3 | 7.69 MB |

Both via `r3d.marching_cubes(..., closed=True)`, `r3d.is_watertight` (True for both), `r3d.scale_to_mm(80)`. Level = volume median at each mesh's own grid (the piece's declared level throughout — the literal T = 0 level can miss the 0.5 rad patch, per the M1 level decision). "Zero-set" in the spec's phrasing means this declared level set, not literal T = 0; the README's Caveats section says so, and that the two objects were meshed at different native resolutions.

## 3. Film

`render_film.py all --size 1080 --device cuda`, one MP4 + GIF via `r3d.write_film`, two hard-cut parts (declared: no depth is ever interpolated into another):

- **Depth dial** (seed 7, one draw): L = 1, L = 2 are the exact 3D crisp-voxel cast (same camera/level/light as the M2 casts), captioned with the measured D beside theory (L1: 2.426 ± 0.031 / theory 2.5; L2: 2.774 ± 0.076 / theory 2.75, both 3-draw mean ± sd, window-calibrated k = 1). L = 3, L = 4 switch to the plotter idiom — one of the 12 slice-atlas coastlines (`render_atlas.py`'s `coast_tile`, duplicated inline), enlarged, captioned "saturated, no measured D" with the reason (the 3D box-count estimator saturates above D ≈ 2.97 at 256³, and the isosurface itself is unreadable foam past L = 2). Each depth held 3 s (72 frames at 24 fps), rendered once and replicated via hardlink.
- **Turntable**: 360° of the L = 1 Heaviside cast, 240 frames (1.5°/frame), fixed world-space light — faces visibly darken as they rotate out of the raking light, a real property of a turntable under one lamp, not a bug.

Final: 1080×1080, 24 fps, 528 frames, 22.0 s. First-pass MP4 (CRF 18, `r3d.write_film`'s hardcoded value) was 21.34 MB, over the 20 MB budget; re-encoded with `ffmpeg -c:v libx264 -crf 23 -pix_fmt yuv420p` to **13.13 MB**. GIF: 11.66 MB (left as `write_film` made it; no size budget stated for the GIF).

Timing test before committing to GPU: 1080² crisp-voxel cast, CPU 4-thread ≈ 17 s/frame vs GPU ≈ 2.5–3.7 s/frame. 240 turntable frames on CPU would have been 60–90 min, over both the ~30 min CPU budget and gpu1's ≤20 min per-segment rule; on GPU the whole film (dial + turntable) finished in one 8.4-minute segment, comfortably inside the 20-minute cap.

**Viewed**: dial frame 0 (L1, cast + caption), dial frame ~150 (L4, coastline + caption), a mid-turntable frame (rotation and caption both correct), and the frame at the dial→turntable hard cut (frame 288, which lands on the same camera angle as the dial's L1 frame, confirming the two parts read as one continuous object).

Bug fixed during testing: captions initially overflowed the frame at 1080² (a unicode superscript-minus glyph the C059 font doesn't render, plus untested line lengths) and the L3/L4 coastline tile initially filled the whole frame with no room for the caption band, making the text illegible over the coastline. Fixed with an `ImageDraw.textlength`-based auto-fit (`fit_font`), fraction notation for 2⁻ᴸ (1/2, 1/4, 1/8, 1/16) instead of the unicode superscript, and a reserved blank top margin in the coastline frame.

## 4. README

Written following the shared brief's shape (title + hook, hero media, "The phenomenon", "The pieces" with per-image captions stating what's measured vs. declared, "What was computed" table with wall-clock times, "Verification", "Caveats", "References"). Hero: the re-rendered diptych plus the film. Verification section carries forward the M1/M2 numbers (calibration systematics ±0.07/±0.04, 1.5-decade fit range, no D claim for L ≥ 3, plate reproduction 3/27 of 1,048,576 pixels, float32/64 flip rates, resolution doubling) per the controller's ruling. Caveats section states the L ≥ 3 saturation, the 1.5-decade shortfall, calibration-protocol dependence, L = 2's foam-from-outside exterior, the STL grids differing, and the "zero-set" language.

Link check (`grep -o 'gallery/[^")> ]*' README.md | sort -u | while read f; do [ -e "$f" ] || echo MISSING $f; done`) initially flagged 4 false positives — backtick-wrapped inline mentions of `gallery/` (e.g. "exist in `gallery/`") where the trailing backtick got swept into the matched path. Reworded those four spots to avoid bare backtick-wrapped `gallery/` references (two now read "the gallery folder" in prose, one pair became proper markdown links). Re-ran clean: no output.

## Decisions (logged in NOTES.md)

- STL fallback to native 96³ re-evaluation for Heaviside (not decimation — no library installed; not a coarser subsample of the same 128³ mesh, but a fresh, honest evaluation at a genuinely coarser native grid).
- Film is one MP4/GIF pair with two hard-cut internal parts, not two separate deliverables — matches the ruling's single bullet ("plus a turntable") and keeps the total inside 20–30 s.
- MP4 re-encoded CRF 18 → 23 after the fact rather than changing `r3d.write_film` (shared module, out of scope to edit for one piece's size budget).
- Diptych caption shortened rather than shrinking the font, to keep it legible at the diptych's existing size.

## Risks / open

1. **Cache is 2.7 GB** (three 512 MB fields, five width-1024 draws, 150 MB of film frames plus calibration data). All gitignored; no cleanup was done since NOTES.md's resume commands assume the fields are present.
2. **STL grids differ in resolution** between the two objects (96³ vs 128³) — declared in the README, but a viewer comparing file sizes or face counts alone could misread it as a fidelity difference beyond the actual cause (the 20 MB budget).
3. **No dedicated STL preview render**: r3d has no generic triangle-mesh rasterizer (only `marching_cubes` → STL export), so the STL's appearance is documented only via the existing voxel-cast PNGs of the same fields, not a render of the mesh itself. Noted as acceptable in NOTES.md but not literally requested by the spec.
