"""Becher-style typology: the shadow (surviving first-layer connections per input pixel)
across every IMP round of one run. One variable changes: the pruning round.

Usage: python render_typology.py --cond mnist_adam_norm_rw0 --seed_idx 0 --norm panel
  --norm panel : dot area = count / max count in that round (declared; caption gives the max)
  --norm global: dot area = count / 300 (absolute; late rounds fade to specks)
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw
from common import *

LABEL = {
    "mnist_adam_norm_rw0": "MNIST · Adam · standardised pixels · rewind to init",
    "mnist_adam_raw_rw0": "MNIST · Adam · raw [0,1] pixels · rewind to init",
    "mnist_sgd_norm_rw0": "MNIST · SGD · standardised pixels · rewind to init",
    "fashion_adam_norm_rw0": "Fashion-MNIST · Adam · standardised pixels · rewind to init",
    "pmnist_adam_norm_rw0": "pixel-permuted MNIST (un-permuted for display) · Adam",
    "mnist_adam_norm_rw500": "MNIST · Adam · standardised pixels · rewind to step 500",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="mnist_adam_norm_rw0")
    ap.add_argument("--seed_idx", type=int, default=0)
    ap.add_argument("--norm", default="panel", choices=["panel", "global"])
    ap.add_argument("--cell", type=int, default=26)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    z = load_analysis()
    idx = imp_idx(z, a.cond)
    m = idx[a.seed_idx]
    sh = z[f"{a.cond}__shadow"][:, m]
    R = sh.shape[0]
    counts = z[f"{a.cond}__counts"]
    acc = z[f"{a.cond}__es_test"][:, m]
    seed = int(z[f"{a.cond}__seeds"][m])
    panel = 28 * a.cell
    gap, cap = int(panel * 0.16), int(panel * 0.13)
    rows = int(np.ceil(R / a.cols))
    margin = int(panel * 0.35)
    head = int(panel * 0.12)
    W = 2 * margin + a.cols * panel + (a.cols - 1) * gap
    H = head + margin + rows * (panel + cap) + (rows - 1) * gap + int(panel * 0.25)
    im = Image.new("RGB", (W, H), CREAM)
    d = ImageDraw.Draw(im)
    fs = max(12, panel // 34)
    text(d, (margin, int(margin * 0.55)), "THE TICKET'S SHADOW — SURVIVING INPUT CONNECTIONS PER PIXEL, BY PRUNING ROUND", int(fs * 1.5))
    text(d, (margin, int(margin * 0.55) + int(fs * 2.3)),
         f"LeNet-300-100 · {LABEL.get(a.cond, a.cond)} · seed {seed} · 20% of the first layer pruned per round by magnitude", fs, MUTED)
    text(d, (W - margin, int(margin * 0.55)),
         "dot area = connections / " + ("max in round" if a.norm == "panel" else "300"), fs, MUTED, anchor="ra")
    for r in range(R):
        i, j = divmod(r, a.cols)
        x = margin + j * (panel + gap); y = head + margin + i * (panel + cap + gap)
        vmax = sh[r].max() if a.norm == "panel" else 300
        im.paste(dots(sh[r], a.cell, vmax=vmax), (x, y))
        frac = counts[r, 0] / 235200
        text(d, (x, y + panel + int(cap * 0.25)),
             f"r{r:02d}  {100 * frac:5.1f}%  max {int(sh[r].max()):3d}", fs)
        text(d, (x + panel, y + panel + int(cap * 0.25)), f"test {100 * acc[r]:.1f}%", fs, MUTED, anchor="ra")
    out = a.out or f"{GALLERY}/typology_{a.cond}_s{seed}_{a.norm}.png"
    im.save(out)
    print(out, im.size)


if __name__ == "__main__":
    main()
