"""Plates. Every image is the network's measured output (or output - B) on its own sample grid,
enlarged nearest-neighbour; the declared choices live in look.py and are restated per plate.

    python render.py <run_dir> <A-run-name> [<C-run-name> <none-run-name>]
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

import look
from look import normal, pseudo, ghost_ink, to_img, up, sheet, text
from metrics import bandpass, band_edges
from pages import load

HERE = os.path.dirname(os.path.abspath(__file__))
GAL = os.path.join(HERE, "gallery")


def run(path):
    d = dict(np.load(path))
    d["args"] = json.loads(str(d["args"]))
    return d


def pick(steps, targets):
    return [int(np.argmin(np.abs(np.log1p(steps) - np.log1p(t)))) for t in targets]


def frames_strip(d, targets, out, mode="pseudo", k=2, gain=1.0, title=None):
    n = d["args"]["res"]
    P = load(n)
    idx = pick(d["snap_steps"], targets)
    pad = 24 * k // 2
    W = len(idx) * (n * k + pad) + pad
    H = n * k + 2 * pad + 60
    img = sheet(W, H)
    for j, i in enumerate(idx):
        f = d["snap_f"][i].astype(np.float32)
        rgb = pseudo(f, P["B"], gain) if mode == "pseudo" else normal(f)
        x = pad + j * (n * k + pad)
        img.paste(to_img(up(rgb, k)), (x, pad))
        text(img, (x + n * k // 2, pad + n * k + 14), f"step {int(d['snap_steps'][i]):,}", 18, italic=True, anchor="ma")
    img.save(out)
    return img


def slitscan(d, out, k=2, mode="pseudo", gain=1.0, nslices=None):
    """Time runs left to right across one page: column block j is the page at snapshot j."""
    n = d["args"]["res"]
    P = load(n)
    S = d["snap_f"].astype(np.float32)
    T = len(S)
    nslices = nslices or T
    edges = np.linspace(0, n, nslices + 1).astype(int)
    page = np.zeros((n, n), np.float32)
    for j in range(nslices):
        i = int(round(j * (T - 1) / max(nslices - 1, 1)))
        page[:, edges[j]:edges[j + 1]] = S[i][:, edges[j]:edges[j + 1]]
    rgb = pseudo(page, P["B"], gain) if mode == "pseudo" else normal(page)
    to_img(up(rgb, k)).save(out)
    return page


def band_grid(d, targets, out, crop=(0, 0, 128, 128), k=3, lo_min=4):
    """Rows: octave bands of the network's output; columns: time. Positive part printed as ink.
    A's lines run horizontally and B's vertically, so orientation says which page a band shows."""
    n = d["args"]["res"]
    e = band_edges(n)
    idx = pick(d["snap_steps"], targets)
    y0, x0, h, w = crop
    pad = 10
    lab = 150
    rows = [(lo, hi) for lo, hi in zip(e[:-1], e[1:]) if lo >= lo_min]
    W = lab + len(idx) * (w * k + pad) + pad
    H = 70 + len(rows) * (h * k + pad) + pad
    img = sheet(W, H)
    for ci, i in enumerate(idx):
        text(img, (lab + ci * (w * k + pad) + w * k // 2, 30), f"step {int(d['snap_steps'][i]):,}", 18, True, anchor="ma")
    for ri, (lo, hi) in enumerate(rows):
        yy = 70 + ri * (h * k + pad)
        text(img, (lab - 14, yy + h * k // 2), f"{lo:g}–{hi:g} c/img", 16, True, anchor="rm")
        # one scale per row: the 99.5th percentile over the row's frames
        bps = [bandpass(d["snap_f"][i].astype(np.float32), n, lo, hi)[y0:y0 + h, x0:x0 + w] for i in idx]
        s = np.percentile(np.abs(np.stack(bps)), 99.5)
        for ci, bp in enumerate(bps):
            img.paste(to_img(up(normal(np.clip(bp / s, 0, 1)), k)), (lab + ci * (w * k + pad), yy))
    img.save(out)


if __name__ == "__main__":
    root, a_name = sys.argv[1], sys.argv[2]
    d = run(os.path.join(root, a_name + ".npz"))
    tag = a_name
    os.makedirs(os.path.join(HERE, "cache", "renders"), exist_ok=True)
    o = os.path.join(HERE, "cache", "renders")
    T = d["snap_steps"][-1]
    tg = [0, 3, 10, 30, 100, 300, 1000, T]
    frames_strip(d, tg, f"{o}/{tag}_strip_pseudo.png", "pseudo")
    frames_strip(d, tg, f"{o}/{tag}_strip_normal.png", "normal")
    slitscan(d, f"{o}/{tag}_slit.png")
    band_grid(d, tg, f"{o}/{tag}_bands.png")
