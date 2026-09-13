# ouroboros NOTES (living handoff)

## State
- CPU only (GPU saturated). OMP_NUM_THREADS=4, torch.set_num_threads(4). OUROBOROS_DEV / OUROBOROS_DT env vars in common.py.
- common.py: targets (ring GMM 8 modes, spiral, Barnsley fern), CRN pools, batched GMM-EM (suff-stat cov), KDE LOO-CV bandwidth (golden-section, verified vs brute grid to <1%), metrics.
- chains.py: run_batch(target, model, regime, lam, n, seeds, G) -> metric histories [B,G+1].

## Plan
1. compute_film.py -> cache/film_*.npz (display samples per generation)
2. compute_phase.py -> cache/phase_*.npz ((lambda, n) grid, replace vs accumulate)
3. render_{film,spiral,phase,ridgeline,fern}.py -> gallery/
4. verify.py (collapse happens? accumulate bounded? boundary box-count + resolution + null)
5. README.md

## Key numbers
(timing G=20, B=32, n=512, pre-optimisation float64: gmm replace 16s, accumulate 93s, kde replace 89s)
