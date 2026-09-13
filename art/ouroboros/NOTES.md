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

## Running (launched 13:49)
- phase GMM ring replace 97x65 G80 n8-512 (2 workers, logs/phase_gmm_replace_w*.log); accumulate 33x33 G80 (logs/phase_gmm_acc_w*.log)
  resume: rerun the same command (parts in cache/parts/*.npz; delete stale *.lock files without matching npz first)
  R="ring gmm replace --L 97 --N 65 --nmin 8 --nmax 512 --G 80"; A="ring gmm accumulate --L 33 --N 33 --nmin 8 --nmax 512 --G 80 --bs 33"
- films: logs/film_A.log (ring gmm/kde, spiral kde/gmm), logs/film_B.log (fern gmm K64 n4096, fern kde n2048)
## Findings so far
- GMM replace lambda=0 n=128: ring collapses to 1 mode by g~120 (sw2 1.0); lambda=0.25 keeps 8 modes but squashes them into needles; accumulate stable (sw2~0.2).
- KDE (LOO bandwidth) EXPLODES instead of collapsing: variance += h^2 each gen, h grows with spread -> runaway (sw2 16 at g120, n=128). Even lambda=0.25 diffuses by g40.
- Toy phase map: at small n even lambda=1 has large sw2 (gen-0 fit error) -> use excess over gen 0 for escape metric.
