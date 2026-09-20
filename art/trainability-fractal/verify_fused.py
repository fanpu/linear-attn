"""Correctness bar for the fused engine (PERF-BRIEF s7).

  ../.venv/bin/python verify_fused.py small      # agreement vs the torch engine
  ../.venv/bin/python verify_fused.py hero       # full 1024^2 hero window vs cache
  ../.venv/bin/python verify_fused.py ulp        # 1-ulp flip statistic
"""
import json, sys, time
import numpy as np, torch
import tfractal as tf
import tfast
from boxcount import box_counts, fit_dimension
from common_render import edges

HERO = dict(c0=1.0801625442981355, c1=2.301518970647778, hw=0.045)
OVER = dict(c0=1.5, c1=1.5, hw=4.5)


def agree(a, b, label):
    fa, fb = (a < 0), (b < 0)
    flips = int((fa != fb).sum())
    Ea = edges(a.reshape(int(np.sqrt(a.size)), -1)) if a.ndim == 1 else edges(a)
    rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
    print(f'  {label:38s} labels {1-flips/a.size:.6f}  flips {flips}/{a.size}  '
          f'max|rel| {np.nanmax(rel):.3e}  conv {fa.mean():.7f}/{fb.mean():.7f}')
    return flips


def boundary_of(M):
    """Pixels adjacent to a sign change, on the full grid (edges() is (R-1)^2)."""
    E = edges(M)
    B = np.zeros(M.shape, bool)
    B[:-1, :-1] |= E; B[1:, :-1] |= E; B[:-1, 1:] |= E; B[1:, 1:] |= E
    return B


def run_small(res=128):
    prob = tf.make_problem(0, nonlin='tanh')
    for name, w in [('hero(mixed)', HERO), ('overview', OVER)]:
        h0, h1 = tf.log_grid(w['c0'], w['c1'], w['hw'], res)
        T = dict(engine='torch', verbose=False, steps=500)
        F = dict(engine='fused', verbose=False, steps=500)
        ref_ee = tf.run_grid(prob, h0, h1, **T)
        ref_ex = tf.run_grid(prob, h0, h1, early_exit=False, **T)
        fus = tf.run_grid(prob, h0, h1, **F)
        fus_x = tf.run_grid(prob, h0, h1, early_exit=False, **F)
        print(f'{name} {res}^2, P={res*res}')
        agree(ref_ee, ref_ex, 'torch early-exit vs torch exact')
        agree(fus, ref_ee, 'fused early-exit vs torch early-exit')
        agree(fus_x, ref_ex, 'fused exact      vs torch exact')
        agree(fus, fus_x, 'fused early-exit vs fused exact')
        # boundary-confinement of the disagreements
        R = res
        A, B = fus.reshape(R, R), ref_ee.reshape(R, R)
        d = (A < 0) != (B < 0)
        if d.any():
            bnd = boundary_of(B)
            print(f'    {int(d.sum())} disagreements, {int((d & bnd).sum())} on boundary pixels')


def run_window(name, w, res, out, engine='fused', steps=500):
    prob = tf.make_problem(0, nonlin='tanh')
    h0, h1 = tf.log_grid(w['c0'], w['c1'], w['hw'], res)
    torch.cuda.synchronize(); t0 = time.time()
    M = tf.run_grid(prob, h0, h1, steps=steps, verbose=False, engine=engine)
    torch.cuda.synchronize(); dt = time.time() - t0
    M = M.reshape(res, res)
    np.savez(out, measure=M, seconds=dt, res=res, steps=steps, engine=engine,
             nonlin='tanh', axes='lr_lr', dtype='float64', seed=0, minibatch=-1, **w)
    print(f'{name}: {dt:.1f}s  {res*res/dt:.1f} px/s  conv={float((M<0).mean()):.7f}')
    return M, dt


def stats(M):
    E = edges(M)
    s, c = box_counts(E)
    R = M.shape[0]
    return dict(conv_frac=float((M < 0).mean()), edge_px=int(E.sum()),
                D=fit_dimension(s, c, 2, R // 4))


def run_hero(res=1024):
    ref = np.load('cache/windows/deep_zoomA4_1024_f64.npz')['measure']
    M, dt = run_window('hero fused', HERO, res, 'cache/windows/hero_fused_1024.npz')
    sr, sm_ = stats(ref), stats(M)
    print(json.dumps(dict(reference=sr, fused=sm_), indent=1, default=float))
    print(f"conv_frac  ref {sr['conv_frac']:.10f}  fused {sm_['conv_frac']:.10f}  "
          f"|d| {abs(sr['conv_frac']-sm_['conv_frac']):.2e}")
    print(f"edge_px    ref {sr['edge_px']}  fused {sm_['edge_px']}  "
          f"rel {abs(sr['edge_px']-sm_['edge_px'])/sr['edge_px']:.4f}")
    print(f"D          ref {sr['D']['D']:.3f}+-{sr['D']['se']:.3f}  "
          f"fused {sm_['D']['D']:.3f}+-{sm_['D']['se']:.3f}")
    d = (M < 0) != (ref < 0)
    bnd = boundary_of(ref)
    print(f'label agreement {1-d.mean():.6f}  ({int(d.sum())} pixels), '
          f'{int((d & bnd).sum())} of them on reference boundary pixels '
          f'({int((d & ~bnd).sum())} off-boundary)')


def run_ulp(res=1024, engine='fused'):
    """Flip fraction of boundary pixels when both learning rates are nudged 1 ulp."""
    prob = tf.make_problem(0, nonlin='tanh')
    h0, h1 = tf.log_grid(HERO['c0'], HERO['c1'], HERO['hw'], res)
    eps = 1.0 + 2.0 ** -52
    A = tf.run_grid(prob, h0, h1, steps=500, verbose=False, engine=engine).reshape(res, res)
    B = tf.run_grid(prob, h0 * eps, h1 * eps, steps=500, verbose=False,
                    engine=engine).reshape(res, res)
    bnd = boundary_of(A)
    flip = (A < 0) != (B < 0)
    print(f'1-ulp nudge ({engine}, {res}^2): {flip.sum()} flips of {A.size} px; '
          f'boundary px {bnd.sum()}, flips on boundary {int((flip & bnd).sum())} '
          f'= {100*(flip & bnd).sum()/max(1,bnd.sum()):.3f}% of boundary, '
          f'{int((flip & ~bnd).sum())} off-boundary')


def run_chaos(res=128):
    """Is the fused-vs-torch difference bigger than the window's own 1-ulp sensitivity?
    Same grid, three runs: torch, torch with both learning rates nudged one ulp, fused.
    The point of the comparison is that most of the drift is the window's chaos, not
    the engine: only a side-by-side against the precision floor separates the two."""
    prob = tf.make_problem(0, nonlin='tanh')
    eps = 1.0 + 2.0 ** -52
    for name, w in [('overview (9 decades)', OVER), ('hero (deep zoomA4)', HERO)]:
        h0, h1 = tf.log_grid(w['c0'], w['c1'], w['hw'], res)
        A = tf.run_grid(prob, h0, h1, steps=500, verbose=False, engine='torch')
        U = tf.run_grid(prob, h0 * eps, h1 * eps, steps=500, verbose=False, engine='torch')
        F = tf.run_grid(prob, h0, h1, steps=500, verbose=False, engine='fused')
        print(f'{name}, {res}^2 = {A.size} px')
        for lbl, B in [('1-ulp nudge of both lr (torch vs torch)', U),
                       ('fused engine           (fused vs torch)', F)]:
            rel = np.abs(A - B) / np.maximum(np.abs(A), 1e-300)
            print(f'  {lbl}: sign flips {int(((A<0)!=(B<0)).sum()):3d}  '
                  f'median rel {np.median(rel):.1e}  '
                  f'>1.4e-5: {int((rel>1.4e-5).sum()):5d}  >1e-2: {int((rel>1e-2).sum()):5d}  '
                  f'max {rel.max():.1e}')


if __name__ == '__main__':
    tf.setup_gpu(0.30)
    what = sys.argv[1] if len(sys.argv) > 1 else 'small'
    res = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if what == 'small':
        run_small(res or 128)
    elif what == 'hero':
        run_hero(res or 1024)
    elif what == 'ulp':
        run_ulp(res or 1024)
    elif what == 'chaos':
        run_chaos(res or 128)
