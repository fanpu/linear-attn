"""CPU analysis of cached hero weights (no training): unit similarity matrices, match quality,
and the test-loss barrier after every weight-matching sweep.  -> cache/units_<ds>.npz

sim_l[i, j] = cosine(incoming weights+bias of A unit i, of B unit j), where B's incoming columns are
already permuted by the final layer-(l-1) permutation, so that sim_l[i, perm_l[i]] is the matched pair.
usage: python analyze_units.py mnist
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '4')
import common
common.DEV = 'cpu'
from common import *

ds = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else ''
Wt = torch.load(os.path.join(CACHE, f'hero_{ds}{tag}_weights.pt'), weights_only=False)
A, B, perms, hist = Wt['A'], Wt['B'], Wt['perms'], Wt['wm_hist']
L = mlp_depth(A)
out = {}
for l in range(L):
    WA = torch.cat([A[f'W{l}'], A[f'b{l}'][:, None]], 1).double()
    WBm = B[f'W{l}'] if l == 0 else B[f'W{l}'][:, torch.as_tensor(perms[l - 1])]
    WB = torch.cat([WBm, B[f'b{l}'][:, None]], 1).double()
    WA = WA / WA.norm(dim=1, keepdim=True)
    WB = WB / WB.norm(dim=1, keepdim=True)
    S = (WA @ WB.T).numpy().astype(np.float32)
    out[f'sim{l}'] = S
    out[f'perm{l}'] = np.asarray(perms[l])
    out[f'matchcos{l}'] = S[np.arange(len(S)), perms[l]]
    print(f'layer {l}: mean matched cos {out[f"matchcos{l}"].mean():.3f}, mean |cos| unmatched diag {np.abs(np.diag(S)).mean():.3f}')
_, _, Xte, yte = load_data(ds)
lams = np.linspace(0, 1, 13)
bar, objs = [], []
for k, pk in enumerate([[np.arange(len(p)) for p in perms]] + list(hist)):
    Bk = mlp_apply_perm(B, pk)
    ls = eval_list([lerp(A, Bk, t) for t in lams], Xte, yte, G=13)[0]
    bar.append(ls)
    objs.append(perm_objective(A, Bk))
    print(f'sweep {k}: obj {objs[-1]:.2f} test barrier {barrier(ls, lams)[1]:.4f}')
out['sweep_lams'] = lams
out['sweep_loss'] = np.array(bar)
out['sweep_obj'] = np.array(objs)
out['sweep_perm0'] = np.array([np.arange(len(perms[0]))] + [h[0] for h in hist])
np.savez_compressed(os.path.join(CACHE, f'units_{ds}{tag}.npz'), **out)
