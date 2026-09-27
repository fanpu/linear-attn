import sys, numpy as np, torch
sys.path.insert(0, "/home/fzeng/ml/research/art/wild-landscapes")
import wl_render as R
torch.set_num_threads(16)
d = np.load(sys.argv[1]); F = np.log10(d["loss"]); ly = d["lyap"]
t = np.clip((ly + 0.4) / 0.8, 0, 1)
alb = R.cmap(t, sys.argv[3] if len(sys.argv) > 3 else "ember")
S = R.Scene(F, alb, K=float(sys.argv[4]) if len(sys.argv) > 4 else 0.35, up=1, device="cpu")
img, _ = S.render(sys.argv[5] if len(sys.argv) > 5 else "0,0,0.1,2.6,-100,38,50", width=800, ss=1, exposure=1.3)
R.save(img, sys.argv[2]); print(F.min(), F.max(), img.mean()/255)
