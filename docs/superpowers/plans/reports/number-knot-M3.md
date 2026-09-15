# Number Knot (§6) — M3 report: films, README, final commit

**Status: DONE_WITH_CONCERNS** (minor; see Risks). Commits `b63753d` (M3) and `ba30693` (README wording).
`art/number-knot/` is clean. All three films and the README are in.

## Controller rulings, applied
1. **Fit line thinner and lower-contrast.**
   - Helix radius 0.012 → 0.006; knot radius 0.022 → 0.009.
   - Colour is scaled ×0.45 toward black on the dark ground and toward white on plaster. On plaster, darkening made it the highest-contrast element, so I changed the rule after looking.
   - Re-rendered `helix_{glow,plaster,plotter}` and `knot_{glow,plaster,plotter}` at 2400 px (glow 7–10 s, plaster 51 s each, CPU). Viewed: in the helix the beads dominate now, and the fit is a faint guide.
2. **`tower_numbers_plotter.svg` dropped.** It is deleted from the gallery, and `render_m2.py` skips plotter for that scene. The knot stays as a secondary plate. The README says: "bead noise (12° median angle error plus radial scatter) hides the T=10 winding; the minor winding is readable mainly from the fit line".
3. **Films** (`films.py`, glow, CPU; r3d's CUDA fix was not needed), each MP4 + GIF via `r3d.write_film`:

   | Film | Frames | Size | Length | MP4 / GIF | Render time |
   |---|---|---|---|---|---|
   | `gallery/film_turntable_days.{mp4,gif}` | 720 | 1024² | 24 s @30 fps | 10.4 MB / 13.6 MB | ≈593 s (CPU shared with a still render) |
   | `gallery/film_turntable_months.{mp4,gif}` | 720 | 1024² | 24 s | 14.0 MB / 13.8 MB | ≈586 s |
   | `gallery/film_sweep_helix.{mp4,gif}` | 750 | 1080×720 | 25 s | 9.2 MB / 3.5 MB | 525 s |

   - Turntables: one orbit at elevation 28°, ortho height fixed to the maximum extent over 12 azimuths, title overlay.
   - Sweep: measured | shuffled-label null side by side, same fixed camera. Each layer holds for 1 s, with 0.5 s **linear morphs (declared interpolations)**. Both panels are divided by the measured layer's RMS in-plane radius. Overlay text gives the layer (fractional during morphs).
   - I checked sample frames from each, plus a crop decoded back from the months MP4.
4. **Captions.** The README "Declarations" block and the "Caveats" section both state:
   - the per-layer scaling;
   - that the in-plane RMS radius grows from 0.3 at Qwen3's embedding to about 45 at the last layers (OLMo-2: 1.1 → 4.7);
   - that radii and heights are not comparable between layers.

   This is the RMS radius in the projection plane, which I measured; the total residual norm was not measured separately.
5. **README** (`art/number-knot/README.md`), in the edge-of-stability shape:
   - Title, italic hook, month and day tower heroes, and the months turntable video.
   - The phenomenon.
   - The pieces: towers, nulls, slice atlases, films, helix, sweep, knot, numbers tower. Each caption separates measured from declared.
   - What was computed: forward passes 526 s CPU; analysis 146 s; geometry 20 s; stills 8.5 min; atlases 10 s; films ≈590/586/525 s.
   - Verification: number ΔR² tables for 0–999 and 0–99 with family-wise nulls, circle-share columns, the poly3 comparator and Fourier peaks. Day/month held-out R² with order and point nulls, cyclic flags and exact order ranks.
   - Caveats: small explained variance; T=10 = ordered clusters; 0–99 underpowered; template choice; per-layer scaling; supervised plane vs Engels' SAE; in-sample geometry; interpolated film frames.
   - References, checked today against the arXiv abstract pages:
     - K&T recipe confirmed.
     - Engels: ICLR 2025, SAE method confirmed.
     - Zhou: title and authors confirmed; NeurIPS not shown on the abstract page, so flagged.
     - Štefánik: sinusoidal number embeddings confirmed, but **the abstract does not mention OLMo 2**. The claim in `ml-art-3d.md` is flagged as unverified.
   - Link check (`grep -o 'gallery/…' | … MISSING`) is clean.

## Decisions (in NOTES.md)
- **MP4 re-encode.** Turntable MP4s were re-encoded at CRF 27 (preset slow). `write_film`'s CRF 18 gave 37–47 MB, which `commit.sh` would skip. At CRF 27 they are 10–14 MB; I checked a crop decoded back from the file.
- **Sweep framing.** The camera frames the central 99 % of beads over all layers and both panels.
- **Film style.** Glow only: plaster frames cost ~45 s each.
- **Scene sizes.** The sweep scene uses bead radius 0.045 and a hairline glow of 0.12. The first sweep pass (radius 0.02, glow 0.5) was dominated by saturated hairlines, so I killed it at 659/750 frames and re-rendered.

## Gallery (34 files, none over 20 MB)
- Heroes: `hero_tower_{days,months}_{glow,plaster}.png` and `…_plotter.svg`.
- Diptychs with nulls: `tower_{days,months}_{glow,plaster}.png`, `…_plotter.svg`, `tower_numbers_{glow,plaster}.png`.
- `helix_{glow,plaster}.png`, `helix_plotter.svg`.
- `knot_{glow,plaster}.png`, `knot_plotter.svg`.
- `atlas_{days,months,numbers}_{measured,null}.png`.
- The three films above.
- `timings_2400.json`, `timings_films.json`.

## Risks and open issues
- **Sweep, last layer.** The measured cloud at layer 16 (elongated, ΔR²_T100 0.014) and the null's tail run out of the fixed frame. The README states this. A fix would be per-layer 3-D RMS scaling or 99.9 % framing, at about 9 min of re-render.
- **Knot plate** remains visually weak by nature; this is stated.
- **commit.sh edge case.** `commit.sh` printed `[: : integer expression expected` for the deleted file (`stat` on a missing path inside the `--modified` loop). The deletion was still committed correctly (`b63753d`). The script could skip missing paths; I did not touch it, since it is shared.
- **Turntable timings** come from frame mtimes, because the first `films.py` run was stopped before writing its timing file.
- **Reports** left in `docs/superpowers/plans/reports/` for the controller to commit.
