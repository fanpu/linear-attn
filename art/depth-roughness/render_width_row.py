"""Finite width vs the limit on the tile patch: rows depth 1-3, columns width 64 ... 65536 and the GP.
Each finite net is piecewise constant on the cells of its n first-layer great circles."""
import numpy as np
from render_common import *
from common import fit_dim, boxcount

nets = np.load("cache/nets_patch.npz")
gp = np.load("cache/tiles_2048.npz")
widths = [64, 256, 1024, 4096, 16384, 65536]
T, GAP, LM, TOP = 420, 30, 150, 260
cols = len(widths) + 1
W = LM + cols * T + (cols - 1) * GAP + 60
H = TOP + 3 * (T + 90) + 140
img = to_img(np.ones((H, W, 3)) * PAPER * grain((H, W), 4, 0.01)[..., None])
dr = ImageDraw.Draw(img)
dr.text((LM, 50), "Width as a zoom limit", font=font(76), fill=(27, 27, 34))
dr.text((LM, 150), "Random Heaviside networks on the same 82-degree patch of the sphere: finite width n (1024² samples) against the infinite-width limit (2048² GP sample).",
        font=font(30, "italic"), fill=(110, 105, 96))
for j, n in enumerate(widths + ["GP"]):
    dr.text((LM + j * (T + GAP), TOP - 50), f"n = {n:,}" if n != "GP" else "n = infinity", font=font(36), fill=(27, 27, 34))
for i, L in enumerate([1, 2, 3]):
    y = TOP + i * (T + 90)
    dr.text((30, y + T // 2 - 25), f"L={L}", font=font(42), fill=(27, 27, 34))
    for j, n in enumerate(widths + ["GP"]):
        x = LM + j * (T + GAP)
        if n == "GP":
            f = gp[f"heaviside_L{L}"].astype(np.float64); k = 2048
        else:
            key = f"L{L}_n{n}"
            if key not in nets.files:
                continue
            f = nets[key]; k = 1024
        c = np.median(f)
        # finite nets take few distinct values: use a level strictly between values at the median
        if n != "GP":
            vals = np.unique(f)
            hi = vals[np.searchsorted(vals, c, side="right")] if c < vals.max() else c
            c = (c + hi) / 2
        cov = ink_coverage(f, c, k // T if k // T > 1 else 1, weight=1)
        if cov.shape[0] != T:
            cov = np.asarray(to_img(np.stack([cov] * 3, -1)).resize((T, T), Image.LANCZOS))[..., 0] / 255.0
        img.paste(to_img(mix(PAPER, INK, np.clip(cov * 2.0, 0, 1) ** 0.9)), (x, y))
        s, cnt = boxcount(f, c)
        D = fit_dim(s, cnt, 2 if k == 1024 else 4, 32 if k == 1024 else 64)[0]
        dr.text((x, y + T + 12), f"D {D:.2f}", font=font(28, "mono"), fill=(27, 27, 34))
dr.text((LM, H - 110), "D = box-counting slope over the same physical range (2.7e-3 to 4.4e-2 rad). Level = patch median (nudged between the two nearest attained values for finite nets).\n"
        "Depth 3 was only run to n = 4096 (full n x n matmuls per sample).", font=font(26), fill=(110, 105, 96))
img.save("gallery/width_row.png"); print(img.size)
