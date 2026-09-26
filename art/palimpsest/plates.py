"""Gallery plates. One shared frame for every sheet: vellum ground, same margin, same title and
caption block, same type. Only the subject changes.

    python plates.py <run_dir> <group> [res]      e.g.  python plates.py cache/runs siren30_w256_adam 256
"""
import json
import os
import sys

import numpy as np
from PIL import Image

import look
from look import normal, pseudo, ghost_ink, to_img, up, sheet, text
from metrics import bandpass, band_edges
from pages import load

HERE = os.path.dirname(os.path.abspath(__file__))
GAL = os.path.join(HERE, "gallery")
M = 90          # margin, px
TITLE = 34
CAP = 19
GAP = 18


def run(path):
    d = dict(np.load(path))
    d["args"] = json.loads(str(d["args"]))
    return d


def pick(steps, targets):
    return [int(np.argmin(np.abs(np.log1p(steps) - np.log1p(t)))) for t in targets]


def frame(panels, cols, title, caption, labels=None, gap=GAP, label_size=17):
    """panels: list of HxWx3 float arrays (all same size). Returns a PIL sheet."""
    h, w = panels[0].shape[:2]
    rows = (len(panels) + cols - 1) // cols
    lab_h = 34 if labels else 0
    W = 2 * M + cols * w + (cols - 1) * gap
    f = look.font(CAP, italic=True)
    W = max(W, 2 * M + int(max(f.getlength(line) for line in caption.split("\n"))) + 4)
    H = M + TITLE + 40 + rows * (h + lab_h) + (rows - 1) * gap + 40 + CAP * (caption.count("\n") + 1) * 1.5 + M
    img = sheet(W, int(H))
    text(img, (M, M), title, TITLE)
    y0 = M + TITLE + 40
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        x = (W - (cols * w + (cols - 1) * gap)) // 2 + c * (w + gap)
        y = y0 + r * (h + lab_h + gap)
        img.paste(to_img(p), (x, y))
        if labels:
            text(img, (x + w // 2, y + h + 8), labels[i], label_size, italic=True, anchor="ma")
    yc = y0 + rows * (h + lab_h) + (rows - 1) * gap + 34
    for j, line in enumerate(caption.split("\n")):
        text(img, (M, yc + j * int(CAP * 1.5)), line, CAP, italic=True)
    return img


def palimpsest_moment(d):
    """The snapshot where the page holds most of both texts: argmax over t of
    min(B written in bands < 16 c/img, A remaining in the finest band). Measured, not chosen."""
    g = d["ghost"]
    names = [str(x) for x in d["template_names"]]
    labels = [str(x) for x in d["band_labels"]]
    e = [float(l.split("-")[0]) for l in labels]
    iA, iB = names.index("A"), names.index("B")
    coarse = [k for k, lo in enumerate(e) if 2 <= lo < 16]
    written = 1 + g[:, iB, coarse].mean(1)
    fineA = g[:, iA, -1]
    score = np.minimum(written, fineA)
    return int(np.argmax(score)), written, fineA


if __name__ == "__main__":
    root, grp = sys.argv[1], sys.argv[2]
    suffix = sys.argv[3] if len(sys.argv) > 3 else "_s0_n256"
    d = run(os.path.join(root, grp + "_AB" + suffix + ".npz"))
    n = d["args"]["res"]
    P = load(n)
    k = max(1, 768 // n)
    i_star, written, fineA = palimpsest_moment(d)
    st = d["snap_steps"]
    print("palimpsest moment: step", st[i_star], "B coarse written", written[i_star], "A finest remaining", fineA[i_star])
    out = os.path.join(HERE, "cache", "renders")
    os.makedirs(out, exist_ok=True)
    T = st[-1]
    tg = [0, 1, 2, 4, 8, 16, 64, T]
    idx = pick(st, tg)
    pan = [up(normal(d["snap_f"][i].astype(np.float32)), 1) for i in idx]
    frame(pan, 4, "The scraping", f"{grp}: page A, then trained on page B. The network's own output at phase-2 steps shown.",
          labels=[f"step {int(st[i]):,}" for i in idx]).save(f"{out}/{grp}{suffix}_scraping.png")
    f = d["snap_f"][i_star].astype(np.float32)
    frame([up(look.normal_bl(f), k), up(look.two_ink(f, P["B"]), k)], 2, "The palimpsest moment",
          f"step {int(st[i_star])}: left, the output as ink; right, the same output split into two inks: min(output, B) in sepia, max(output - B, 0) in red.\nRed is ink the network holds that page B does not have.",
          ).save(f"{out}/{grp}{suffix}_moment.png")
