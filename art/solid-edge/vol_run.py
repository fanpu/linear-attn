"""M2 volume pipeline: float32 volume -> float64 shell (iterated) -> 1 % float64 audit -> final
[-> float64 full plane]. Every stage checkpoints per chunk under cache/vol/<name>/.

Axes (all volumes): index [k, i, j] = [log10 sigma, log10 eta1, log10 eta0]; sigma scales the
init of every parameter block. Grids:
  --grid toyA   the M1 toy (a) chart at --res: eta at every (1024/res)-th overview pixel centre,
                log10 sigma = (k - round(26 res/64)) * 6/res  (sigma = 1 on a grid plane)
  --grid window --center c0 c1 c2 --hw h0 h1 h2 : pixel-centre log grid like tfractal.log_grid,
                10**c * 10**(off*hw), off = (m+0.5)/res*2-1, per axis (eta0, eta1, sigma)
  --grid json --grid_json F : window read from a JSON file {"center": [...], "hw": [...]}
  --grid sub --parent NAME  : the 32^3 block of NAME's float32 labels with most edge cells,
                recomputed over the identical extent at --res (resolution doubling)

  python vol_run.py --name A128 --kind net2 --grid toyA --res 128 --stages f32,shell,audit,final
"""
import argparse
import json
import math
import os
import time

import sys

import numpy as np
import torch
from scipy.ndimage import maximum_filter, minimum_filter

import se_engine as se
from se_analysis import edges3d, edges

HERE = os.path.dirname(os.path.abspath(__file__))
CH = 32768
T_START = time.time()
MAX_SECONDS = None       # segment limit: stop cleanly (exit 3) before a chunk would overrun it
EXIT_MORE = 3
STEPS = 500
BLOCK = 32


def ts():
    return time.strftime('%F %T')


def window_axes(center, hw, res):
    off = (np.arange(res, dtype=np.float64) + 0.5) / res * 2.0 - 1.0
    ten = torch.tensor(10.0, dtype=torch.float64)
    axes = []
    for c, h in zip(center, hw):
        v = (10.0 ** torch.tensor(float(c), dtype=torch.float64)) * torch.pow(ten, torch.from_numpy(off) * float(h))
        axes.append(v.numpy())
    return axes          # [eta0, eta1, sigma] values (float64)


def build_grid(args, vdir):
    gfn = os.path.join(vdir, 'grid.json')
    if os.path.exists(gfn):
        return json.load(open(gfn))
    R = args.res
    if args.grid == 'toyA':
        eta, pix = se.overview_lr_axis(R)
        sig, _, k1 = se.sigma_axis(R)
        g = dict(kind='toyA', res=R, eta0=eta.numpy().tolist(), eta1=eta.numpy().tolist(), sigma=sig.numpy().tolist(),
                 overview_pix=pix.tolist(), sigma_plane=k1)
    elif args.grid in ('window', 'json'):
        if args.grid == 'json':
            w = json.load(open(args.grid_json))
            center, hw = w['center'], w['hw']
        else:
            center, hw = args.center, args.hw
        a = window_axes(center, hw, R)
        g = dict(kind='window', res=R, center=list(center), hw=list(hw), eta0=a[0].tolist(), eta1=a[1].tolist(), sigma=a[2].tolist())
    elif args.grid == 'sub':
        par = json.load(open(os.path.join(HERE, 'cache', 'vol', args.parent, 'grid.json')))
        Mp = np.load(os.path.join(HERE, 'cache', 'vol', args.parent, 'f32.npy'))
        Rp = Mp.shape[0]
        E = edges3d(Mp < 0)
        nb = Rp // BLOCK
        cnt = E.reshape(nb, BLOCK, nb, BLOCK, nb, BLOCK).sum(axis=(1, 3, 5))
        kb, ib, jb = np.unravel_index(int(np.argmax(cnt)), cnt.shape)
        center, hw = [], []
        for ax, b in zip(['eta0', 'eta1', 'sigma'], [jb, ib, kb]):
            c, h = par['center'][['eta0', 'eta1', 'sigma'].index(ax)], par['hw'][['eta0', 'eta1', 'sigma'].index(ax)]
            dlt = 2 * h / Rp
            center.append(c - h + (b * BLOCK + BLOCK / 2) * dlt)
            hw.append(BLOCK / 2 * dlt)
        a = window_axes(center, hw, R)
        g = dict(kind='sub', res=R, parent=args.parent, block=[int(kb), int(ib), int(jb)], block_edge_cells=int(cnt.max()),
                 center=center, hw=hw, eta0=a[0].tolist(), eta1=a[1].tolist(), sigma=a[2].tolist())
    else:
        raise ValueError(args.grid)
    json.dump(g, open(gfn, 'w'))
    return g


class SegmentDone(Exception):
    pass


class Runner:
    def __init__(self, kind, grid, device):
        self.kind = kind
        self.dev = device
        self.R = grid['res']
        self.ax = [torch.tensor(grid[k], dtype=torch.float64) for k in ('eta0', 'eta1', 'sigma')]
        self.probs = {}

    def prob(self, dt):
        if dt not in self.probs:
            self.probs[dt] = se.make_problem(self.kind, device=self.dev, dtype=dt)
        return self.probs[dt]

    def run(self, flat_idx, dt):
        """flat_idx: int64 numpy array (len <= CH); padded to CH with its first entry (fixed shape)."""
        n = len(flat_idx)
        pad = np.concatenate([flat_idx, np.full(CH - n, flat_idx[0], dtype=np.int64)]) if n < CH else flat_idx
        R = self.R
        k, i, j = np.unravel_index(pad, (R, R, R))
        e0 = self.ax[0][torch.from_numpy(j)].to(self.dev)
        e1 = self.ax[1][torch.from_numpy(i)].to(self.dev)
        sg = self.ax[2][torch.from_numpy(k)].to(self.dev)
        with torch.no_grad():
            r = se.train_chunk(self.prob(dt), [e0, e1], sigma=sg, steps=STEPS, compiled=(self.dev == 'cuda'))
        return r['measure'].to(torch.float64).cpu().numpy()[:n]


def run_list(runner, idx, dt, cdir, tag, log):
    """Compute measure at flat indices idx in chunks of CH, checkpointing each chunk.
    Returns (values, total compute seconds over all chunks ever run, voxels timed)."""
    os.makedirs(cdir, exist_ok=True)
    tfn = os.path.join(cdir, f'{tag}_times.json')
    times = json.load(open(tfn)) if os.path.exists(tfn) else {}
    out = np.empty(len(idx), dtype=np.float64)
    nch = math.ceil(len(idx) / CH)
    for c in range(nch):
        fn = os.path.join(cdir, f'{tag}_{c:04d}.npy')
        s, e = c * CH, min(len(idx), (c + 1) * CH)
        if os.path.exists(fn):
            out[s:e] = np.load(fn)
            continue
        if MAX_SECONDS is not None and times:
            last = max(v[0] for v in times.values())
            if time.time() - T_START + 1.2 * last > MAX_SECONDS:
                raise SegmentDone(f'{tag} chunk {c}/{nch}')
        if runner.dev == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        m = runner.run(idx[s:e], dt)
        dtm = time.time() - t0
        np.save(fn + '.tmp.npy', m)
        os.replace(fn + '.tmp.npy', fn)
        out[s:e] = m
        times[str(c)] = [dtm, e - s]
        json.dump(times, open(tfn + '.tmp', 'w'))
        os.replace(tfn + '.tmp', tfn)
        if c % 8 == 0 or c == nch - 1:
            log(f'  {tag} chunk {c+1}/{nch}  {dtm:.1f}s  {(e-s)/dtm:.0f} px/s  conv={np.mean(m<0):.3f}  [{ts()}]')
    t_run = float(sum(v[0] for v in times.values()))
    n_run = int(sum(v[1] for v in times.values()))
    return out, t_run, n_run


def mixed3(L):
    """Voxels whose 3x3x3 neighbourhood holds both labels (within one voxel of a label change)."""
    return maximum_filter(L, size=3, mode='nearest') != minimum_filter(L, size=3, mode='nearest')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--kind', default='net2', choices=['net2', 'quad2'])
    p.add_argument('--grid', default='toyA', choices=['toyA', 'window', 'json', 'sub'])
    p.add_argument('--grid_json')
    p.add_argument('--parent')
    p.add_argument('--center', type=float, nargs=3)
    p.add_argument('--hw', type=float, nargs=3)
    p.add_argument('--res', type=int, default=64)
    p.add_argument('--stages', default='f32')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max_rounds', type=int, default=6)
    p.add_argument('--audit_frac', type=float, default=0.01)
    p.add_argument('--chunk', type=int, default=32768)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--block', type=int, default=32, help='sub-block side for --grid sub')
    p.add_argument('--root', default=None, help='cache root override (tests)')
    p.add_argument('--max_seconds', type=float, default=None, help='segment wall limit; exit 3 if work remains')
    args = p.parse_args()
    global CH, HERE, STEPS, BLOCK, MAX_SECONDS
    CH = args.chunk; STEPS = args.steps; BLOCK = args.block; MAX_SECONDS = args.max_seconds
    if args.root:
        HERE = args.root
    vdir = os.path.join(HERE, 'cache', 'vol', args.name)
    os.makedirs(vdir, exist_ok=True)
    logf = open(os.path.join(vdir, 'stages.log'), 'a')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n'); logf.flush()

    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.10)
        torch.backends.cuda.matmul.allow_tf32 = False
    grid = build_grid(args, vdir)
    R = grid['res']
    runner = Runner(args.kind, grid, args.device)
    metaf = os.path.join(vdir, 'meta.json')
    meta = json.load(open(metaf)) if os.path.exists(metaf) else dict(name=args.name, kind=args.kind, res=R, grid=grid['kind'])

    def save_meta():
        json.dump(meta, open(metaf, 'w'), indent=1)

    f32fn = os.path.join(vdir, 'f32.npy')
    try:
        run_stages(args, vdir, grid, R, runner, meta, save_meta, f32fn, log)
    except SegmentDone as ex:
        log(f'[{ts()}] {args.name} segment limit reached at {ex}; exit {EXIT_MORE} (rerun to continue)')
        sys.exit(EXIT_MORE)
    log(f'[{ts()}] {args.name} done')


def run_stages(args, vdir, grid, R, runner, meta, save_meta, f32fn, log):
    for stage in args.stages.split(','):
        log(f'[{ts()}] {args.name} stage {stage}')
        if stage == 'f32':
            if os.path.exists(f32fn):
                log('  exists'); continue
            idx = np.arange(R ** 3, dtype=np.int64)
            m, t, n = run_list(runner, idx, torch.float32, os.path.join(vdir, 'f32_chunks'), 'f32', log)
            np.save(f32fn, m.astype(np.float32).reshape(R, R, R))
            meta['f32'] = dict(voxels=R ** 3, seconds=t, px_per_s=(n / t if t else None), conv=float(np.mean(m < 0)))
            save_meta()
            log(f'  f32 done conv={np.mean(m<0):.4f} {t:.0f}s')
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        elif stage == 'shell':
            M32 = np.load(f32fn)
            M = M32.astype(np.float64).ravel()
            done = np.zeros(R ** 3, bool)
            rounds = meta.get('shell', {}).get('rounds', [])
            for r in range(args.max_rounds):
                ifn = os.path.join(vdir, f'shell_r{r}_idx.npy')
                if os.path.exists(ifn):
                    idx = np.load(ifn)
                else:
                    S = mixed3((M < 0).reshape(R, R, R)).ravel() & ~done
                    idx = np.flatnonzero(S).astype(np.int64)
                    np.save(ifn, idx)
                if len(idx) == 0:
                    break
                m64, t, n = run_list(runner, idx, torch.float64, os.path.join(vdir, 'shell_chunks'), f'r{r}', log)
                flips = int(np.sum(np.sign(m64) != np.sign(M32.ravel()[idx])))
                M[idx] = m64; done[idx] = True
                rec = dict(round=r, voxels=int(len(idx)), flips_vs_f32=flips, flip_rate=flips / len(idx),
                           seconds=t, px_per_s=(n / t if t else None))
                rounds = [x for x in rounds if x['round'] != r] + [rec]
                log(f'  shell round {r}: {len(idx)} voxels, {flips} flips ({100*flips/len(idx):.2f} %)')
            np.save(os.path.join(vdir, 'shell_measure.npy'), M.astype(np.float32).reshape(R, R, R))
            np.save(os.path.join(vdir, 'f64_mask.npy'), done.reshape(R, R, R))
            meta['shell'] = dict(rounds=rounds, total_voxels=int(done.sum()), frac=float(done.mean()))
            save_meta()
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        elif stage == 'audit':
            M32 = np.load(f32fn).ravel()
            done = np.load(os.path.join(vdir, 'f64_mask.npy')).ravel()
            rest = np.flatnonzero(~done)
            nA = int(math.ceil(args.audit_frac * len(rest)))
            ifn = os.path.join(vdir, 'audit_idx.npy')
            if os.path.exists(ifn):
                idx = np.load(ifn)
            else:
                idx = np.sort(np.random.default_rng(0).choice(rest, size=nA, replace=False)).astype(np.int64)
                np.save(ifn, idx)
            m64, t, n = run_list(runner, idx, torch.float64, os.path.join(vdir, 'audit_chunks'), 'audit', log)
            flip = np.sign(m64) != np.sign(M32[idx])
            np.save(os.path.join(vdir, 'audit_measure.npy'), m64)
            meta['audit'] = dict(voxels=int(len(idx)), of_rest=int(len(rest)), flips=int(flip.sum()),
                                 flip_rate=float(flip.mean()), seconds=t, px_per_s=(n / t if t else None),
                                 flip_indices=np.flatnonzero(flip).tolist()[:50])
            save_meta()
            log(f'  audit {len(idx)} voxels, {int(flip.sum())} flips ({100*flip.mean():.3f} %)')
        elif stage == 'final':
            M = np.load(os.path.join(vdir, 'shell_measure.npy')).ravel().copy()
            mask = np.load(os.path.join(vdir, 'f64_mask.npy')).ravel().copy()
            aidx = np.load(os.path.join(vdir, 'audit_idx.npy'))
            M[aidx] = np.load(os.path.join(vdir, 'audit_measure.npy')); mask[aidx] = True
            np.savez(os.path.join(HERE, 'cache', 'vol', f'{args.name}_final.npz'), measure=M.reshape(R, R, R),
                     f64_mask=mask.reshape(R, R, R), eta0=np.array(grid['eta0']), eta1=np.array(grid['eta1']),
                     sigma=np.array(grid['sigma']), meta=json.dumps(meta), grid=json.dumps({k: v for k, v in grid.items() if k not in ('eta0', 'eta1', 'sigma')}))
            meta['final'] = dict(conv=float(np.mean(M < 0)), f64_frac=float(mask.mean()))
            save_meta()
            log(f'  final conv={np.mean(M<0):.4f} f64 voxels {mask.mean()*100:.2f} %')
        elif stage == 'plane':
            F = np.load(os.path.join(HERE, 'cache', 'vol', f'{args.name}_final.npz'))
            L = F['measure'] < 0
            E = edges3d(L)
            kstar = int(np.argmax(E.sum(axis=(1, 2))))
            idx = np.ravel_multi_index(np.meshgrid([kstar], np.arange(R), np.arange(R), indexing='ij'), (R, R, R)).ravel().astype(np.int64)
            m64, t, n = run_list(runner, idx, torch.float64, os.path.join(vdir, 'plane_chunks'), f'plane{kstar}', log)
            P64 = m64.reshape(R, R) < 0
            Pfin = L[kstar]
            P32 = np.load(f32fn)[kstar] < 0
            near = mixed3(P64.astype(np.uint8)[None].repeat(3, 0))[1]
            res = dict(sigma_plane_index=kstar, log10_sigma=float(np.log10(grid['sigma'][kstar])), voxels=R * R,
                       agree_final=float(np.mean(P64 == Pfin)), agree_f32=float(np.mean(P64 == P32)),
                       near_def='3x3 neighbourhood of the float64 plane holds both labels', n_near=int(near.sum()),
                       disagree_near_final=int(np.sum((P64 != Pfin) & near)), disagree_far_final=int(np.sum((P64 != Pfin) & ~near)),
                       disagree_near_f32=int(np.sum((P64 != P32) & near)), disagree_far_f32=int(np.sum((P64 != P32) & ~near)),
                       seconds=t, px_per_s=(n / t if t else None))
            res['disagree_rate_near_final'] = res['disagree_near_final'] / max(1, res['n_near'])
            res['disagree_rate_near_f32'] = res['disagree_near_f32'] / max(1, res['n_near'])
            np.save(os.path.join(vdir, 'plane_f64.npy'), m64.reshape(R, R))
            meta['plane'] = res
            save_meta()
            log('  plane ' + json.dumps(res))
        else:
            raise ValueError(stage)


if __name__ == '__main__':
    main()
