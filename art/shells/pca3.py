"""Top-3 PCA directions of a checkpoint trajectory (CPU).  Follows
art/loss-landscape/pca_dirs.py (Li et al. 2018 §7): rows = w_epoch - w_final over all
cached epochs (ep000..ep040, 17 checkpoints), conv/linear weights only (dim<=1 entries
zeroed, 'biasbn'), unit-norm right singular vectors (NOT filter-normalised), plus each
checkpoint's coordinates and its residual outside the 3-plane.
  python pca3.py resnet20
writes cache/dirs/<model>_pca3.pt {d1,d2,d3, coords (E,3), epochs, explained, resid_frac}"""
import os, sys, glob, re, json
import numpy as np, torch
from lib import CACHE, LL, LL_ROOT

os.makedirs(os.path.join(CACHE, "dirs"), exist_ok=True)
for name in sys.argv[1:]:
    files = sorted(glob.glob(os.path.join(LL_ROOT, "cache", "ckpt", f"{name}_ep*.pt")))
    eps = [int(re.search(r"_ep(\d+)\.pt", f).group(1)) for f in files]
    net = LL.ResNetCifar(*LL.parse_name(name))
    keys = [k for k, _ in net.named_parameters()]
    shapes = [p.shape for _, p in net.named_parameters()]

    def vec(path):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        return torch.cat([sd[k].double().flatten() * (1.0 if len(s) > 1 else 0.0) for k, s in zip(keys, shapes)])

    wf = vec(os.path.join(LL_ROOT, "cache", "ckpt", f"{name}_final.pt"))
    M = torch.stack([vec(f) - wf for f in files])
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    ex = (S ** 2 / (S ** 2).sum()).numpy()
    D = Vh[:3]
    # sign convention: ep000 has negative coordinates (matches loss-landscape pca_dirs.py signs)
    sg = torch.sign(M[0] @ D.T); sg[sg == 0] = 1; D = -D * sg[:, None]
    coords = (M @ D.T).numpy()
    resid = (M - torch.tensor(coords) @ D).norm(dim=1) / M.norm(dim=1).clamp_min(1e-30)

    def unflat(v):
        out, i = [], 0
        for s in shapes:
            n = int(np.prod(s)); out.append(v[i:i + n].view(s).float().clone()); i += n
        return out

    # consistency with loss-landscape's 2-direction PCA (same recipe): |cos| should be ~1
    ll = torch.load(os.path.join(LL_ROOT, "cache", "dirs", f"{name}_pca.pt"), weights_only=False)
    cos = [abs(float(torch.cat([t.double().flatten() for t in ll[k]]) @ D[i])) for i, k in enumerate(("dx", "dy"))]
    torch.save(dict(d1=unflat(D[0]), d2=unflat(D[1]), d3=unflat(D[2]), coords=coords, epochs=eps,
                    explained=ex[:6], resid_frac=resid.numpy(), cos_vs_loss_landscape=cos),
               os.path.join(CACHE, "dirs", f"{name}_pca3.pt"))
    print(name, "explained", np.round(ex[:5], 4), "top3 sum", round(float(ex[:3].sum()), 4))
    print("|cos| vs loss-landscape pca dx,dy:", cos)
    print("coord min", coords.min(0), "max", coords.max(0))
    print(json.dumps({int(e): [round(float(v), 3) for v in c] + [round(float(r), 3)] for e, c, r in zip(eps, coords, resid)}))
