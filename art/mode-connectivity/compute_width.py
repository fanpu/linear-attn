"""Width series: does linear mode connectivity after weight matching depend on width?

For widths 32..2048 (3 hidden ReLU layers), n_pairs independent pairs (seeds 2i, 2i+1), train with
Adam 1e-3 for --epochs, weight-match, and evaluate 25-point lerps (full train + test):
naive, matched, naive+REPAIR, matched+REPAIR.  Weights of pair 0 are kept for planes / quilts.
Writes cache/width_<ds>.npz after every width (resumable: skips widths already present).

usage: python compute_width.py mnist [--widths 32,64,...] [--pairs 3]
"""
import sys, argparse
sys.path.insert(0, __import__('os').path.dirname(__file__))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('ds')
ap.add_argument('--widths', default='32,64,128,256,512,1024,2048')
ap.add_argument('--pairs', type=int, default=3)
ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--tag', default='')
args = ap.parse_args()
gpu_setup()
tag = f'width_{args.ds}{args.tag}'
log = Logger(os.path.join(CACHE, tag + '.log'))
path = os.path.join(CACHE, tag + '.npz')
out = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
data = load_data(args.ds)
Xtr, ytr, Xte, yte = data
Xrep = Xtr[:10000]
lams = np.linspace(0, 1, 25)
out['lams'] = lams
t0 = time.time()


def eval_path(fn, G=25):
    pl = [fn(t) for t in lams]
    a = eval_list(pl, Xtr, ytr, G)
    b = eval_list(pl, Xte, yte, G)
    return np.stack([a[0], a[1], b[0], b[1]], 1)


for w in [int(x) for x in args.widths.split(',')]:
    if f'w{w}_naive' in out:
        log(f'width {w} cached, skip')
        continue
    R = {k: [] for k in ['naive', 'matched', 'naive_repair', 'matched_repair']}
    extra = {'dist_AB': [], 'dist_ABp': [], 'wm_sweeps': [], 'wm_time': []}
    for i in range(args.pairs):
        tw = time.time()
        pA, _ = train_mlp(data, w, 2 * i, epochs=args.epochs, log=log)
        pB, _ = train_mlp(data, w, 2 * i + 1, epochs=args.epochs, log=log)
        tm = time.time()
        perms, nit = mlp_weight_matching(pA, pB)
        extra['wm_time'].append(time.time() - tm)
        extra['wm_sweeps'].append(nit)
        pBp = mlp_apply_perm(pB, perms)
        extra['dist_AB'].append(math.sqrt(sum(((pA[k] - pB[k]) ** 2).sum().item() for k in pA)))
        extra['dist_ABp'].append(math.sqrt(sum(((pA[k] - pBp[k]) ** 2).sum().item() for k in pA)))
        fns = {'naive': lambda l: lerp(pA, pB, l), 'matched': lambda l: lerp(pA, pBp, l),
               'naive_repair': lambda l: mlp_repair(pA, pB, l, Xrep),
               'matched_repair': lambda l: mlp_repair(pA, pBp, l, Xrep)}
        msg = []
        for k, fn in fns.items():
            r = eval_path(fn)
            R[k].append(r)
            msg.append(f'{k} {barrier(r[:, 0], lams)[1]:.4f}/{barrier(r[:, 2], lams)[1]:.4f}')
        log(f'width {w} pair {i}: train/test barrier ' + ' | '.join(msg) + f' | {nit} sweeps, {time.time() - tw:.0f}s')
        if i == 0:
            torch.save(dict(A=to_cpu(pA), B=to_cpu(pB), Bp=to_cpu(pBp), perms=[np.asarray(p) for p in perms]),
                       os.path.join(CACHE, f'width_{args.ds}_w{w}_weights.pt'))
    for k in R:
        out[f'w{w}_{k}'] = np.array(R[k])
    for k in extra:
        out[f'w{w}_{k}'] = np.array(extra[k])
    out['widths_done'] = np.array(sorted(int(k[1:].split('_')[0]) for k in out if k.endswith('_naive') and k.startswith('w')))
    np.savez_compressed(path, **out)
log(f'done {tag} in {time.time() - t0:.0f}s')
