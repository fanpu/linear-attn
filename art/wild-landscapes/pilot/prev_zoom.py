import sys, numpy as np, torch
sys.path.insert(0, "/home/fzeng/ml/research/art/wild-landscapes")
import wl_render as R
from PIL import Image
torch.set_num_threads(16)
orbit = sys.argv[1]; ks = [int(k) for k in sys.argv[2].split(",")]
ims = []
for k in ks:
    d = np.load(f"cache/rnn/zoom/z{k:04d}.npz"); s = float(d["s"]); F = np.maximum(d["logloss"], -1.5)
    rng = np.percentile(F, 98) - np.percentile(F, 2); K = min(0.25 * (2.5 / s) ** 0.35, 0.55 / rng)
    col = R.cmap(np.clip(0.5 + d["lyap"] / 0.8, 0, 1), "crameri_berlin")
    S = R.Scene(F[::2, ::2], col[::2, ::2], K=K, Z0=float(np.percentile(F, 50)), up=1, device="cpu")
    img, _ = S.render(orbit, width=480, ss=1, key=(200, 30), exposure=1.4, fog=0.2)
    ims.append(Image.fromarray(img))
c = Image.new("RGB", (480 * 3, 270 * ((len(ims) + 2) // 3)))
for i, im in enumerate(ims): c.paste(im, ((i % 3) * 480, (i // 3) * 270))
c.save(sys.argv[3])
