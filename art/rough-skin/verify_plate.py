"""GPU: plate reproduction (spec §0.7 / §11.3). The great 2-sphere {x4 = 0} of S^3 is S^2, and a width-n net on S^3
restricted to it is a width-n net on S^2 with input weights W0[:, :3]. So this piece's forward code
(common.forward_all, float32 hidden layers) must reproduce depth-roughness's finite-network patch plate
(art/depth-roughness/cache/nets_patch.npz, L = 1, 2 at n = 4096, seed 11, 1024^2 Lambert patch) when fed depth-roughness's
own weights with the input embedded as (x, y, z, 0).

    verify_plate.py   -> cache/verify_plate.json, gallery/verify_plate_depth_roughness.png
"""
import importlib.util, json, sys
import numpy as np
import torch
from PIL import Image
torch.cuda.set_per_process_memory_fraction(0.10)
from common import forward_all, median_level

DR = "/home/fzeng/ml/research/art/depth-roughness/"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


sys.path.append(DR)                       # nets.py does `import torch` only; common.py needs scipy
drc = load("dr_common", DR + "common.py")
drn = load("dr_nets", DR + "nets.py")
ref = np.load(DR + "cache/nets_patch.npz")
C0 = np.array([0.3, -0.5, 0.8]); HW = 0.7                   # as in depth-roughness/compute_nets.py part A
v3 = drc.lambert_patch(C0, HW, 1024)
X4 = torch.as_tensor(np.concatenate([v3.reshape(-1, 3), np.zeros((1024 * 1024, 1))], 1), dtype=torch.float32, device="cuda")
out = {}
panels = []
for L in (1, 2):
    net = drn.HeavisideNet(4096, L, 11)
    W0 = torch.cat([net.W0, torch.randn(4096, 1, dtype=torch.float64, device=net.W0.device)], 1).float()   # 4th column never matters
    v = net.v.float()
    Ws = [net._full(l).float() for l in range(1, L)]
    T = torch.empty(1024 * 1024, device="cuda")
    with torch.no_grad():
        for i in range(0, T.numel(), 65536):
            T[i:i + 65536] = forward_all(X4[i:i + 65536], W0, v, Ws, "heaviside")[L - 1]
    ours = T.view(1024, 1024).double().cpu().numpy()
    theirs = ref[f"L{L}_n4096"]
    u = float(np.median(theirs))
    sa, sb = ours > u, theirs > u
    out[f"L{L}"] = dict(max_abs_diff=float(np.abs(ours - theirs).max()), sign_disagree=int((sa != sb).sum()),
                        pixels=int(sa.size), level=u)
    print(L, out[f"L{L}"], flush=True)
    diff = (sa != sb)
    panels.append(np.concatenate([np.repeat(np.where(sb, 40, 235)[..., None], 3, -1),
                                  np.full((1024, 24, 3), 255), np.repeat(np.where(sa, 40, 235)[..., None], 3, -1),
                                  np.full((1024, 24, 3), 255),
                                  np.stack([np.where(diff, 230, 250), np.where(diff, 30, 250), np.where(diff, 30, 250)], -1)], 1))
    del Ws, net
    torch.cuda.empty_cache()
json.dump(out, open("cache/verify_plate.json", "w"), indent=1)
img = np.concatenate([panels[0], np.full((24, panels[0].shape[1], 3), 255), panels[1]], 0).astype(np.uint8)
Image.fromarray(img).save("gallery/verify_plate_depth_roughness.png")
print("wrote gallery/verify_plate_depth_roughness.png")
