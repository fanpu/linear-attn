import sys, numpy as np, torch
sys.path.insert(0, "/home/fzeng/ml/research/art/wild-landscapes")
import wl_render as R
torch.set_num_threads(16)
d = np.load(sys.argv[1]); F = np.log10(d["loss"]); ly = d["lyap"]
floor = float(sys.argv[3]); F = np.maximum(F, floor)
sub = int(sys.argv[6]) if len(sys.argv) > 6 else 1
F = F[::sub, ::sub]; ly = ly[::sub, ::sub]
t = np.clip(0.5 + ly / 0.6, 0, 1)
alb = R.cmap(t, sys.argv[5])
S = R.Scene(F, alb, K=float(sys.argv[4]), Z0=floor, up=1, device="cpu")
img, _ = S.render(sys.argv[7] if len(sys.argv) > 7 else "0,0,0.1,2.7,-100,40,45", width=900, ss=1, exposure=1.4, key=(200, 30))
R.save(img, sys.argv[2]); print(F.min(), F.max(), img.mean()/255)
