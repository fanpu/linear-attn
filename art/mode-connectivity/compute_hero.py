"""Hero experiment: one pair of width-W MLPs (3 hidden layers, ReLU) on MNIST / FashionMNIST.

Saves cache/hero_<ds>_weights.pt first (A, B, pi(B), Bezier controls, perms, WM sweep history),
then cache/hero_<ds>.npz with
  * 1-D paths (201 points): naive lerp A->B, matched lerp A->pi(B), Bezier A->C->B,
    Bezier A->C'->pi(B), lerp B->pi(B) (same function at both ends), REPAIR variants.
    columns: train_loss, train_acc, test_loss, test_acc (full 60k / 10k sets, float64 summed CE)
  * predictions of 900 class-sorted test images along naive / matched / bezier (121 points)
  * emergence: at each of ~28 checkpoints, match A_k to B_k and evaluate naive/matched lerps,
    plus pi_final applied to B_k.
2-D planes are in compute_planes.py (reads the weights file).

usage: python compute_hero.py mnist [--width 512 --epochs 20]
"""
import sys, argparse
sys.path.insert(0, __import__('os').path.dirname(__file__))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('ds')
ap.add_argument('--width', type=int, default=512)
ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--bezier_epochs', type=int, default=20)
ap.add_argument('--tag', default='')
args = ap.parse_args()
gpu_setup()
tag = f'hero_{args.ds}{args.tag}'
log = Logger(os.path.join(CACHE, tag + '.log'))
t0 = time.time()
data = load_data(args.ds)
Xtr, ytr, Xte, yte = data

CK = [0, .01, .02, .035, .05, .075, .1, .15, .2, .3, .4, .5, .65, .8, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30]
CK = [c for c in CK if c <= args.epochs]
log(f'== {tag}: training A,B width {args.width} epochs {args.epochs}')
pA, ckA = train_mlp(data, args.width, 0, epochs=args.epochs, ckpt_epochs=CK, log=log)
pB, ckB = train_mlp(data, args.width, 1, epochs=args.epochs, ckpt_epochs=CK, log=log)

hist = []
tw = time.time()
perms, nit = mlp_weight_matching(pA, pB, history=hist)
pBp = mlp_apply_perm(pB, perms)
log(f'weight matching: {nit} sweeps in {time.time() - tw:.1f}s; <A,B>={perm_objective(pA, pB):.2f} -> <A,pi(B)>={perm_objective(pA, pBp):.2f}')
hist_obj = [perm_objective(pA, mlp_apply_perm(pB, h[0])) for h in hist]
log(f'  function check test (B, pi(B)): {mlp_eval(pB, Xte, yte)} {mlp_eval(pBp, Xte, yte)}')
dist_AB = math.sqrt(sum(((pA[k] - pB[k]) ** 2).sum().item() for k in pA))
dist_ABp = math.sqrt(sum(((pA[k] - pBp[k]) ** 2).sum().item() for k in pA))
log(f'  ||A-B|| = {dist_AB:.3f}, ||A-pi(B)|| = {dist_ABp:.3f}')

log('training Bezier control points')
C = train_bezier_mlp(data, pA, pB, epochs=args.bezier_epochs)
Cp = train_bezier_mlp(data, pA, pBp, epochs=args.bezier_epochs, seed=1)
torch.save(dict(A=to_cpu(pA), B=to_cpu(pB), Bp=to_cpu(pBp), C=to_cpu(C), Cp=to_cpu(Cp),
                perms=[np.asarray(p) for p in perms], wm_hist=[[np.asarray(q) for q in h[0]] for h in hist],
                wm_hist_obj=hist_obj), os.path.join(CACHE, tag + '_weights.pt'))
log(f'weights saved ({time.time() - t0:.0f}s)')

out = dict(width=args.width, epochs=args.epochs, dist_AB=dist_AB, dist_ABp=dist_ABp, wm_hist_obj=np.array(hist_obj),
           obj_AB=perm_objective(pA, pB))

# ---------------------------------------------------------------- 1-D paths
lams = np.linspace(0, 1, 201)
out['lams'] = lams
Xrep = Xtr[:10000]
paths = {
    'naive': lambda l: lerp(pA, pB, l),
    'matched': lambda l: lerp(pA, pBp, l),
    'bezier': lambda l: bezier(pA, C, pB, l),
    'bezier_matched': lambda l: bezier(pA, Cp, pBp, l),
    'self': lambda l: lerp(pB, pBp, l),
    'naive_repair': lambda l: mlp_repair(pA, pB, l, Xrep),
    'matched_repair': lambda l: mlp_repair(pA, pBp, l, Xrep),
}


def eval_path(fn, ts, G=16, cls_store=None):
    rows, cl = [], []
    for s in range(0, len(ts), G):
        pl = [fn(t) for t in ts[s:s + G]]
        a = eval_list(pl, Xtr, ytr, G, per_class=True)
        b = eval_list(pl, Xte, yte, G, per_class=True)
        rows.append(np.stack([a[0], a[1], b[0], b[1]], 1))
        cl.append(np.stack([a[2], b[2]], 1))
    if cls_store is not None:
        cls_store.append(np.concatenate(cl))  # [n, 2 (train,test), 10] per-class loss contributions
    return np.concatenate(rows)


for name, fn in paths.items():
    cs = []
    r = eval_path(fn, lams, cls_store=cs)
    out['path_' + name] = r
    out['cls_' + name] = cs[0]
    log(f'  path {name:15s}: barrier(lin) train {barrier(r[:, 0], lams)[1]:.4f} test {barrier(r[:, 2], lams)[1]:.4f} | '
        f'barrier(mid) train {barrier(r[:, 0], lams)[0]:.4f} | mid train loss {r[100, 0]:.4f} mid test acc {r[100, 3]:.4f} | '
        f'min test acc {r[:, 3].min():.4f}')

# ---------------------------------------------------------------- predictions for animation
sel = torch.cat([torch.nonzero(yte == c)[:90, 0] for c in range(10)])
out['pred_idx'] = sel.cpu().numpy()
out['pred_labels'] = yte[sel].cpu().numpy()
tan = np.linspace(0, 1, 121)
out['pred_t'] = tan
for name in ['naive', 'matched', 'bezier']:
    P, Q = [], []
    for t in tan:
        with torch.no_grad():
            pr = F.softmax(mlp_forward(paths[name](t), Xte[sel]).double(), 1)
        P.append(pr.argmax(1).cpu().numpy().astype(np.uint8))
        Q.append(pr.max(1).values.cpu().numpy().astype(np.float32))
    out['pred_' + name] = np.stack(P)
    out['conf_' + name] = np.stack(Q)
np.savez_compressed(os.path.join(CACHE, tag + '.npz'), **out)
log(f'paths saved ({time.time() - t0:.0f}s)')

# ---------------------------------------------------------------- emergence: barrier vs epoch
log('barrier vs epoch')
le = np.linspace(0, 1, 25)
out['ep_lams'] = le
out['ep_epochs'] = np.array(CK, dtype=float)
E = {k: [] for k in ['naive', 'match', 'final']}
agree = []
for e in CK:
    a, b = ckA[e], ckB[e]
    pk, _ = mlp_weight_matching(a, b)
    bp = mlp_apply_perm(b, pk)
    bfin = mlp_apply_perm(b, perms)
    for key, fn in [('naive', lambda l: lerp(a, b, l)), ('match', lambda l: lerp(a, bp, l)), ('final', lambda l: lerp(a, bfin, l))]:
        E[key].append(eval_path(fn, le, G=25))
    agree.append([float(np.mean(pk[i] == perms[i])) for i in range(len(perms))])
    bl = lambda k: barrier(E[k][-1][:, 0], le)[1]
    log(f'  epoch {e:5}: naive {bl("naive"):.4f} matched {bl("match"):.4f} pi_final {bl("final"):.4f} '
        f'endpoint loss {E["match"][-1][0, 0]:.4f} perm agreement {np.round(agree[-1], 3)}')
for k in E:
    out['ep_' + k] = np.array(E[k])
out['ep_agree'] = np.array(agree)
np.savez_compressed(os.path.join(CACHE, tag + '.npz'), **out)
log(f'done {tag} in {time.time() - t0:.0f}s')
