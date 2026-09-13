"""Hero experiment: one pair of width-512 MLPs (3 hidden layers) on MNIST and FashionMNIST.

Saves to cache/hero_<ds>.npz (+ cache/hero_<ds>_weights.pt):
  * 1-D paths (101 points): naive lerp A->B, weight-matched lerp A->pi(B), Bezier A->C->B,
    Bezier A->C'->pi(B), lerp B->pi(B); train/test loss+acc; REPAIR variants of the lerps.
  * predictions of 900 class-sorted test images along naive / bezier / matched (121 points).
  * barrier-vs-epoch: at each checkpoint match A_k,B_k; and apply pi_k to final B.
  * 2-D loss planes through {A,B,pi(B)} and {A,B,C}, train and test loss.

usage: python compute_hero.py mnist [--res 201]
"""
import sys, argparse
sys.path.insert(0, __import__('os').path.dirname(__file__))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('ds')
ap.add_argument('--width', type=int, default=512)
ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--res', type=int, default=201)
ap.add_argument('--tag', default='')
args = ap.parse_args()
gpu_setup()
tag = f'hero_{args.ds}{args.tag}'
log = Logger(os.path.join(CACHE, tag + '.log'))
t0 = time.time()
data = load_data(args.ds)
Xtr, ytr, Xte, yte = data

CK = [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 11, 15, 20, 25, 30]
CK = [c for c in CK if c <= args.epochs]
log(f'== {tag}: training A,B width {args.width} epochs {args.epochs}')
pA, ckA = train_mlp(data, args.width, 0, epochs=args.epochs, ckpt_epochs=CK, log=log)
pB, ckB = train_mlp(data, args.width, 1, epochs=args.epochs, ckpt_epochs=CK, log=log)

perms, nit = mlp_weight_matching(pA, pB)
pBp = mlp_apply_perm(pB, perms)
log(f'weight matching: {nit} sweeps; obj <A,B>={perm_objective(pA, pB):.2f} -> <A,pi(B)>={perm_objective(pA, pBp):.2f}')
log(f'  function check test (B, pi(B)): {mlp_eval(pB, Xte, yte)} {mlp_eval(pBp, Xte, yte)}')
dist_AB = math.sqrt(sum(((pA[k] - pB[k]) ** 2).sum().item() for k in pA))
dist_ABp = math.sqrt(sum(((pA[k] - pBp[k]) ** 2).sum().item() for k in pA))
log(f'  ||A-B|| = {dist_AB:.3f}, ||A-pi(B)|| = {dist_ABp:.3f}')

log('training Bezier control points')
C = train_bezier_mlp(data, pA, pB, epochs=20, log=log)
Cp = train_bezier_mlp(data, pA, pBp, epochs=20, seed=1, log=log)

out = dict(width=args.width, epochs=args.epochs, dist_AB=dist_AB, dist_ABp=dist_ABp)

# ---------------------------------------------------------------- 1-D paths
lams = np.linspace(0, 1, 101)
out['lams'] = lams
Xrep = Xtr[:10000]


def path_eval(fn):
    r = np.array([mlp_eval(fn(l), Xtr, ytr) + mlp_eval(fn(l), Xte, yte) for l in lams])
    return r  # columns: train_loss, train_acc, test_loss, test_acc


paths = {
    'naive': lambda l: lerp(pA, pB, l),
    'matched': lambda l: lerp(pA, pBp, l),
    'bezier': lambda l: bezier(pA, C, pB, l),
    'bezier_matched': lambda l: bezier(pA, Cp, pBp, l),
    'self': lambda l: lerp(pB, pBp, l),
    'naive_repair': lambda l: mlp_repair(pA, pB, l, Xrep),
    'matched_repair': lambda l: mlp_repair(pA, pBp, l, Xrep),
}
for name, fn in paths.items():
    r = path_eval(fn)
    out['path_' + name] = r
    btr, bte = barrier(r[:, 0], lams)[1], barrier(r[:, 2], lams)[1]
    log(f'  path {name:15s}: barrier train {btr:.4f} test {bte:.4f} | mid train loss {r[50, 0]:.4f} '
        f'test acc {r[50, 3]:.4f} | paper-def barrier train {barrier(r[:, 0], lams)[0]:.4f}')

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
    # fine-resolution loss along the animation parameter
    out['anim_' + name] = np.array([mlp_eval(paths[name](t), Xtr, ytr)[0] for t in tan] )
    out['anim_test_' + name] = np.array([mlp_eval(paths[name](t), Xte, yte)[0] for t in tan])

# ---------------------------------------------------------------- barrier vs epoch
log('barrier vs epoch')
le = np.linspace(0, 1, 25)
out['ep_lams'] = le
out['ep_epochs'] = np.array(CK, dtype=float)
ep_naive, ep_match, ep_final, ep_agree = [], [], [], []
for e in CK:
    a, b = ckA[e], ckB[e]
    pk, _ = mlp_weight_matching(a, b)
    bp = mlp_apply_perm(b, pk)
    bfin = mlp_apply_perm(pB, pk)
    ep_naive.append([[*mlp_eval(lerp(a, b, l), Xtr, ytr), *mlp_eval(lerp(a, b, l), Xte, yte)] for l in le])
    ep_match.append([[*mlp_eval(lerp(a, bp, l), Xtr, ytr), *mlp_eval(lerp(a, bp, l), Xte, yte)] for l in le])
    ep_final.append([[*mlp_eval(lerp(pA, bfin, l), Xtr, ytr), *mlp_eval(lerp(pA, bfin, l), Xte, yte)] for l in le])
    ep_agree.append([float(np.mean(pk[i] == perms[i])) for i in range(len(perms))])
    log(f'  epoch {e:5}: naive {barrier(np.array(ep_naive[-1])[:, 0], le)[1]:.4f} '
        f'matched {barrier(np.array(ep_match[-1])[:, 0], le)[1]:.4f} '
        f'pi_k on final {barrier(np.array(ep_final[-1])[:, 0], le)[1]:.4f} '
        f'endpoint loss {ep_match[-1][0][0]:.4f} perm agreement {np.round(ep_agree[-1], 3)}')
out['ep_naive'] = np.array(ep_naive)
out['ep_match'] = np.array(ep_match)
out['ep_final'] = np.array(ep_final)
out['ep_agree'] = np.array(ep_agree)

# ---------------------------------------------------------------- 2-D planes
@torch.no_grad()
def plane_eval(P0, u, v, xs, ys, X, y, G=24, nb=10000):
    """loss on grid theta = P0 + x*u_hat + y*v_hat (u,v orthonormal dicts). batched with bmm."""
    L = mlp_depth(P0)
    coords = [(xx, yy) for yy in ys for xx in xs]
    res = np.zeros(len(coords))
    for s in range(0, len(coords), G):
        cc = torch.tensor(coords[s:s + G], device=DEV, dtype=torch.float32)
        g = len(cc)
        Ws = {k: P0[k][None] + cc[:, 0].view(-1, *[1] * P0[k].dim()) * u[k][None]
                 + cc[:, 1].view(-1, *[1] * P0[k].dim()) * v[k][None] for k in P0}
        tot = torch.zeros(g, device=DEV, dtype=torch.float64)
        for i in range(0, len(X), nb):
            h = X[i:i + nb][None].expand(g, -1, -1)
            for l in range(L + 1):
                h = torch.baddbmm(Ws[f'b{l}'][:, None, :], h, Ws[f'W{l}'].transpose(1, 2))
                if l < L:
                    h = F.relu(h)
            yy = y[i:i + nb]
            tot += F.cross_entropy(h.reshape(-1, h.shape[-1]).double(), yy.repeat(g), reduction='none').view(g, -1).sum(1)
        res[s:s + g] = (tot / len(X)).cpu().numpy()
    return res.reshape(len(ys), len(xs))


def dot(p, q):
    return sum((p[k].double() * q[k].double()).sum().item() for k in p)


def plane(P0, P1, P2, name, pad=0.3):
    u = {k: P1[k] - P0[k] for k in P0}
    nu = math.sqrt(dot(u, u))
    u = {k: x / nu for k, x in u.items()}
    w = {k: P2[k] - P0[k] for k in P0}
    c = dot(w, u)
    v = {k: w[k] - c * u[k] for k in P0}
    nv = math.sqrt(dot(v, v))
    v = {k: x / nv for k, x in v.items()}
    pts = np.array([[0, 0], [nu, 0], [c, nv]])
    lo, hi = pts.min(0), pts.max(0)
    span = (hi - lo).max()
    xs = np.linspace(lo[0] - pad * span, hi[0] + pad * span, args.res)
    ys = np.linspace(lo[1] - pad * span, hi[1] + pad * span, args.res)
    t = time.time()
    Ltr = plane_eval(P0, u, v, xs, ys, Xtr, ytr)
    Lte = plane_eval(P0, u, v, xs, ys, Xte, yte)
    log(f'  plane {name}: {args.res}^2 in {time.time() - t:.1f}s; min train {Ltr.min():.4f} max {Ltr.max():.3f}')
    out[f'plane_{name}_xs'] = xs
    out[f'plane_{name}_ys'] = ys
    out[f'plane_{name}_pts'] = pts
    out[f'plane_{name}_train'] = Ltr
    out[f'plane_{name}_test'] = Lte
    return u, v


log('2-D planes')
plane(pA, pB, pBp, 'perm')
u, v = plane(pA, pB, C, 'bezier')
# project Bezier curve into its plane (it lies exactly in it)
cur = []
for t in np.linspace(0, 1, 101):
    q = bezier(pA, C, pB, t)
    d = {k: q[k] - pA[k] for k in pA}
    cur.append([dot(d, u), dot(d, v)])
out['plane_bezier_curve'] = np.array(cur)

np.savez_compressed(os.path.join(CACHE, tag + '.npz'), **out)
torch.save(dict(A=to_cpu(pA), B=to_cpu(pB), Bp=to_cpu(pBp), C=to_cpu(C), Cp=to_cpu(Cp),
                perms=[np.asarray(p) for p in perms]), os.path.join(CACHE, tag + '_weights.pt'))
log(f'done {tag} in {time.time() - t0:.0f}s')
