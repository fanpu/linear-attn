"""PCA directions of the training trajectory (Li et al. 2018 §7).  CPU only.
rows = w_epoch - w_final over saved checkpoints (conv/linear weights only; BN and bias
entries are zero, 'biasbn'), top-2 right singular vectors -> dx, dy (unit norm, NOT
filter-normalized, as in Li et al.), plus each checkpoint's projection coordinates.
  python pca_dirs.py resnet56 resnet56_noshort
writes cache/dirs/<model>_pca.pt  {dx, dy, coords (E,2), epochs, explained}"""
import os, sys, glob, re, json
import numpy as np, torch
from common import CACHE, ResNetCifar, parse_name

os.makedirs(os.path.join(CACHE, "dirs"), exist_ok=True)
args = [a for a in sys.argv[1:] if not a.startswith("--min-epoch=")]
emin = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--min-epoch=")), 0))
for name in args:
    files = sorted(glob.glob(os.path.join(CACHE, "ckpt", f"{name}_ep*.pt")))
    eps = [int(re.search(r"_ep(\d+)\.pt", f).group(1)) for f in files]
    files = [f for f, e in zip(files, eps) if e >= emin]; eps = [e for e in eps if e >= emin]
    net = ResNetCifar(*parse_name(name))
    keys = [k for k, _ in net.named_parameters()]
    shapes = [p.shape for _, p in net.named_parameters()]
    def vec(path):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        return torch.cat([sd[k].double().flatten() * (1.0 if len(s) > 1 else 0.0) for k, s in zip(keys, shapes)])
    wf = vec(os.path.join(CACHE, "ckpt", f"{name}_final.pt"))
    M = torch.stack([vec(f) - wf for f in files])  # (E, P)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    ex = (S ** 2 / (S ** 2).sum()).numpy()
    d1, d2 = Vh[0], Vh[1]
    coords = torch.stack([M @ d1, M @ d2], 1).numpy()
    def unflat(v):
        out, i = [], 0
        for s in shapes:
            n = int(np.prod(s)); out.append(v[i:i + n].view(s).float().clone()); i += n
        return out
    torch.save(dict(dx=unflat(d1), dy=unflat(d2), coords=coords, epochs=eps, explained=ex[:5]),
               os.path.join(CACHE, "dirs", f"{name}_pca" + (f"_from{emin}" if emin else "") + ".pt"))
    print(name, "explained", np.round(ex[:4], 3), "coord range x", coords[:, 0].min(), coords[:, 0].max(),
          "y", coords[:, 1].min(), coords[:, 1].max())
    print(json.dumps({int(e): [round(float(a), 3), round(float(b), 3)] for e, (a, b) in zip(eps, coords)}))
