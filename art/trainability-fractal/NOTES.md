# NOTES (handoff) - trainability-fractal

## State (end of session 3, 2026-09-13 12:40)
- Engine `tfractal.py` (batched hand-written fwd/bwd, float64, torch.compile); styles `styles.py`
  (spectral, dark_magma, riso_two_ink, line_boundary, hillshade, NEW isolines); plate pages `pages.py`.
- Deep zoom DONE to 10^6.5: zoomA kf0-5 (half-decade, 10^0..10^2.5) + zoomB kf0-3 (10^3.5..10^6.5).
  `merge_zoom.py` -> cache/zoom_zoomAB (currently 10 kf). 1-ulp flips of boundary px = 0 at every plate >=10^0.5.
- Box counting (verify.py zoomAB null_quadratic liu, b=2..32 px): D = 1.20,1.19,1.32,1.50,1.61,1.66,1.61,1.36,1.64,1.37
  at 10^0..10^6.5; quadratic null 1.01-1.17 (one 1.65 corner outlier at 10^0.5). Table: `python readme_tables.py`.
- Resolution check DONE (`rescheck.py`, cache/verify_res.json): edge px 524/1634/5034 at 1x/2x/4x -> r^1.63;
  conv frac stable 36.5%; 128^2 block D=1.94; label agreement 77-83%. Written into README.
- Gallery rendered: zoomAB plates + contact sheets in spectral, riso, isolines, aurora_ember
  (`python render_zoom.py zoomAB --plates --style S`). Weak `line` plate style dropped (chunky at 256^2).
- README: DTABLE/FLIPS/RES/ZOOM filled. Remaining placeholders: <!-- HERO -->, <!-- NKF -->, <!-- HEROT -->, <!-- ZOOM-VIDEO -->.

## GPU jobs running at handoff (shared GPU very contended, 33-80 px/s per job)
Logs in logs/s3_*.log; each script ends with `echo DONE`. Outputs in cache/windows/ unless noted.
- s3_hero64: hero_overview_tanh_1024_f64 (replaces the float32 hero; ~4 h)
- s3_relu: ov_relu_512_f64 (overview ReLU for `render_windows.py diptych OV`, which pairs it with the tanh f64 hero)
- s3_steps: steps_zoomA2_384 (1000 steps, checkpoints 10:1000:10) -> `python render_windows.py steps steps_zoomA2_384` (spectral default)
- s3_sem: sem_sigma_lr_384 then sem_wd_lr_384 -> `python render_windows.py semantic`
- s3_deep1024: deep_zoomA4_1024_f64 (native 1024^2 float64 of the 10^2 swirl plate: hero candidate + box counting b=2..256 = 2.1 decades).
  **cache/windows/deep_zoomA3_1024_f64.npz is a 0-byte DUMMY** placed to make the script skip A3; delete it (`rm`) after s3_deep1024 prints DONE.
- s3_fill: zoom_compute --tag fill --path cache/fill_path.json (4 half-decade in-betweens 10^3,10^4,10^5,10^6, centred on the next deeper centre, nesting checked).

## Session 4 changes (12:40)
- Killed the queued (never-started) s3_fill and s3_relu waiters; APPENDED them to scripts already holding slots:
  s3_sem.sh: sem_sigma -> (sem_wd skipped via 0-byte DUMMY cache/windows/sem_wd_lr_384.npz) -> DONE -> fill -> FILL_DONE -> rm dummy -> sem_wd -> SEMWD_DONE
  s3_steps.sh: steps -> DONE -> ov_relu_512_f64 -> RELU_DONE
- Preliminary video from 10 kf rendering (logs/s4_video_prelim.log); re-render after fill merge.

## Next (in order)
1. When fill DONE: `python merge_zoom.py` (-> 14 kf), `python verify.py zoomAB null_quadratic liu`, `python readme_tables.py` -> replace DTABLE block;
   re-render plates for spectral riso isolines aurora_ember; `python render_zoom.py zoomAB --video` (spectral; also --style magma) -> gallery/zoom_zoomAB_spectral.mp4/.gif;
   then delete old gallery/zoom_zoomA_spectral.{mp4,gif}; add <video> at <!-- ZOOM-VIDEO -->. Check GIF < 15 MB.
2. Hero f64: `python render_hero.py hero_overview_tanh_1024_f64 overview_tanh` (+ `--style aurora_ember`, `--style indigo_madder`); fill <!-- HERO -->, <!-- HEROT --> (seconds in npz),
   remove the float32 caveat paragraph (or keep f32-vs-f64 comparison numbers as a note).
3. deep_zoomA4_1024_f64: render spectral print (render_hero.py works for any window name; its axis labels assume overview - check),
   isolines (`S.isolines(M, px=2048)`), riso; box count over b=2..256 with boxcount.fit_dimension -> README (>=2 decades claim).
4. diptych OV, steps video, semantic plates -> README gallery sections (captions: measured vs aesthetic).
5. Link check of README (all gallery paths exist), final commit, kill no background processes left.

## Resume
cd /home/fzeng/ml/research/art/trainability-fractal; tail -n 2 logs/s3_*.log
Relaunch an unfinished job: `nohup ../_shared/gpu_run.sh bash logs/s3_<job>.sh > logs/s3_<job>.log 2>&1 &`
(window_compute skips existing outputs; zoom_compute resumes from cached keyframes).
CAUTION: don't `pkill -f <pattern>` from a shell whose own command line contains the pattern (it kills the calling shell).
GPU time so far: sessions 1-2 ~10 GPU-h wall (contended); session 3 jobs above add several more.
