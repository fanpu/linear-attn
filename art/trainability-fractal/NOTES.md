# NOTES (handoff) - trainability-fractal

## State (session 3, 2026-09-13)
- Engine `tfractal.py` (batched hand-written fwd/bwd, float64, torch.compile), styles `styles.py`, pages `pages.py`.
- zoomA: cache/zoom_zoomA kf0-5 half-decade 10^0..10^2.5. zoomB: cache/zoom_zoomB decade kf0..3 = 10^3.5..10^6.5 (kf3 running at session-3 start, job `logs/zoomB_run.sh`).
- 1-ulp flip fraction of boundary px = 0 at every network keyframe so far.
- Hero overview 1024^2 float32 exists (cache/windows/hero_overview_tanh_1024_f32.npz, 2x NN-upscaled to 2048 in gallery).
- Session-3 GPU jobs launched (logs/s3_*.sh/.log): hero_overview_tanh_1024_f64, ov_relu_1024_f64, steps_zoomA2_384 (checkpoints 10:1000:10), sem_sigma_lr_384 + sem_wd_lr_384.
- render_zoom.py now accepts --style spectral|magma|riso|line|relief|<palettes pairing e.g. aurora_ember>.

## Next
1. After zoomB kf3: `python merge_zoom.py`; compute half-decade fill keyframes between zoomB decades (centred on the next deeper centre); re-merge.
2. Render zoomAB plates (spectral, riso, line, aurora_ember) + video; hero f64; relu-vs-tanh overview diptych; steps animation; semantic plates.
3. `python verify.py zoomAB null_quadratic liu`; res-check blocks analysis; fill README placeholders.

## Resume
cd art/trainability-fractal; tail logs/s3_*.log logs/zoomB.log; relaunch any unfinished: `nohup ../_shared/gpu_run.sh bash logs/s3_<job>.sh > logs/s3_<job>.log 2>&1 &` (window_compute skips existing outputs; zoom_compute resumes from cached keyframes).
