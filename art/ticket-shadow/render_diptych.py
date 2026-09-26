"""Diptych: two shadows of the same data. Same network, same pixels in [0,1], same pruning,
same round; only the optimiser differs (Adam vs plain SGD). Seed-mean shadows over 3 seeds.
Dot area = seed-mean connections / max in panel (declared)."""
import argparse, json
import numpy as np
from PIL import Image, ImageDraw
from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=15)
    ap.add_argument("--cell", type=int, default=64)
    a = ap.parse_args()
    z = load_analysis(); st = load_stats()
    sup = json.load(open(f"{CACHE}/support_table.json"))
    r = a.round
    conds = [("mnist_adam_raw_rw0", "ADAM"), ("mnist_sgd_raw_rw0", "SGD")]
    panel = 28 * a.cell
    fs = panel // 60
    margin = panel // 7; gap = panel // 8; head = fs * 7; foot = fs * 9
    W = 2 * margin + 2 * panel + gap
    H = margin + head + panel + foot + margin // 2
    im = Image.new("RGB", (W, H), CREAM)
    d = ImageDraw.Draw(im)
    frac = z["mnist_adam_raw_rw0__counts"][r, 0] / 235200
    text(d, (margin, margin), "TWO SHADOWS OF THE SAME DATA", int(fs * 1.8))
    text(d, (margin, margin + int(fs * 2.6)),
         f"LeNet-300-100 lottery tickets on MNIST, pixels in [0,1], round {r} of iterative magnitude pruning "
         f"({100 * frac:.1f}% of input weights left). Only the optimiser differs.", fs, MUTED)
    for k, (c, name) in enumerate(conds):
        idx = imp_idx(z, c)
        s = z[f"{c}__shadow"][r, idx].mean(0)
        x = margin + k * (panel + gap); y = margin + head
        im.paste(dots(s, a.cell), (x, y))
        dd = st[c][r]["imp"]; b = sup[f"{c}/r{r}"]["bins"]
        text(d, (x, y + panel + fs), name, int(fs * 1.5))
        text(d, (x, y + panel + int(fs * 3.2)),
             f"pixel lit in 1–9 of 55,000 images keeps {b[1]:.1f} connections; lit in >20,000 keeps {b[6]:.1f}", fs, MUTED)
        text(d, (x, y + panel + int(fs * 4.8)),
             f"r(pixel std) {dd['r_std']:.2f} · r(ever lit) {sup[f'{c}/r{r}']['r_support']:.2f} · test {100 * dd['es_test']:.1f}% · mean of {len(idx)} seeds", fs, MUTED)
    out = f"{GALLERY}/diptych_adam_vs_sgd_r{r:02d}.png"
    im.save(out); print(out, im.size)


if __name__ == "__main__":
    main()
