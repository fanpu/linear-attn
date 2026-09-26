"""Follow-up (2026-09-26 evening): the epsilon dial, a typology of shadows in the house style.
Same network, same raw [0,1] pixels, same pruning round; only Adam's epsilon changes (then plain SGD).
Seed-mean shadows from cache/followup_dial_r15.json (followup_dial.py). Dot area = connections / max in
panel (declared, as in every other typology here); the caption line under each panel gives the numbers.

    python render_dial.py dial     # the epsilon dial
    python render_dial.py mech     # AdamW, beta2, sign-SGD, signum, SGD+momentum
"""
import json, sys
import numpy as np
from PIL import Image, ImageDraw
from common import *

SETS = {
    "dial": dict(
        title="THE ε DIAL",
        sub="LeNet-300-100 lottery tickets on MNIST, pixels in [0,1], round {r} of iterative magnitude pruning ({pct:.1f}% of input weights left). "
            "Only Adam's ε changes; the last panel is plain SGD.",
        items=[("adam_eps1e-08", "ε 1e-8"), ("adam_eps1e-06", "ε 1e-6"), ("adam_eps1e-05", "ε 1e-5"),
               ("adam_eps0.0001", "ε 1e-4"), ("adam_eps0.001", "ε 1e-3"), ("adam_eps0.01", "ε 1e-2"),
               ("adam_eps0.1", "ε 1e-1"), ("sgd_lr0.1 (job 728)", "SGD")],
        cols=8,
        foot="ε 1e-1 at the same lr is plain SGD at an effective lr of 0.012 (with EMA momentum), and is under-trained; with lr ×10 it matches SGD "
             "(rare/common {m[plateau]:.2f}, r(std) {m[r_std]:.2f}, 1 seed)."),
    "mech": dict(
        title="SAME DATA, EIGHT OPTIMISERS",
        sub="LeNet-300-100 lottery tickets on MNIST, pixels in [0,1], round {r} of iterative magnitude pruning ({pct:.1f}% of input weights left). "
            "Only the optimiser differs. signum = sign of an EMA momentum (β 0.9); sign-SGD = sign of the raw gradient; both lr 1e-4.",
        items=[("adam_eps1e-8 (job 664)", "Adam"), ("adamw_wd0.1", "AdamW wd 0.1"), ("adam_b2_0.9", "Adam β2 0.9"),
               ("adam_b2_0.9999", "Adam β2 0.9999"), ("signum_lr1e-4", "signum"), ("signsgd_lr1e-4", "sign-SGD"),
               ("sgdm0.9_lr0.01", "SGD + momentum"), ("adam_eps0.1_lr0.012", "Adam ε 0.1, lr×10")],
        cols=4),
}


def main(which="dial", r=15, cell=None):
    S = SETS[which]
    d = json.load(open(f"{CACHE}/followup_dial_r{r}.json"))
    items = [(k, lab) for k, lab in S["items"] if k in d]
    cols = S["cols"]; cell = cell or (18 if cols > 4 else 22); rows = int(np.ceil(len(items) / cols))
    panel = 28 * cell
    fs = max(14, panel // 34)
    gap = int(panel * 0.14); cap = int(fs * 5.2)
    margin = int(panel * 0.3); head = int(fs * 6)
    W = 2 * margin + cols * panel + (cols - 1) * gap
    H = margin + head + rows * (panel + cap) + (rows - 1) * gap + margin // 2 + fs * 5
    im = Image.new("RGB", (W, H), CREAM); dr = ImageDraw.Draw(im)
    pct = 100 * 0.8 ** r
    text(dr, (margin, margin), S["title"], int(fs * 1.8))
    text(dr, (margin, margin + int(fs * 2.7)), S["sub"].format(r=r, pct=pct), fs, MUTED)
    for n, (k, lab) in enumerate(items):
        i, j = divmod(n, cols)
        x = margin + j * (panel + gap); y = margin + head + i * (panel + cap + gap)
        o = d[k]
        s = np.array(o["shadow_mean"])
        im.paste(dots(s, cell), (x, y))
        b = o["bins"]
        text(dr, (x, y + panel + int(fs * 0.6)), lab, int(fs * 1.35))
        text(dr, (x + panel, y + panel + int(fs * 0.8)), f"test {100 * o['acc']:.1f}% · {o['n']} seed{'s' if o['n'] > 1 else ''}", fs, MUTED, anchor="ra")
        text(dr, (x, y + panel + int(fs * 2.6)),
             f"never lit {b[0]:.1f} · lit 10–99× {b[2]:.1f} · >20k× {b[6]:.1f}", fs, MUTED)
        text(dr, (x, y + panel + int(fs * 4.0)),
             f"rare/common {o['plateau']:.2f} · r(std) {o['r_std']:.2f} · r(ever lit) {o['r_lit']:.2f}", fs, MUTED)
    if "foot" in S:
        text(dr, (margin, H - margin // 2 - fs * 2), S["foot"].format(m=d["adam_eps0.1_lr0.012"]), fs, MUTED)
    text(dr, (margin, H - margin // 2 - fs * 3.6),
         "dot area = seed-mean connections kept / max in panel.  'lit 10–99×': mean connections kept by pixels lit in 10–99 of 55,000 training images; "
         "'rare/common' = that divided by the same for pixels lit in >20,000.", fs, MUTED)
    out = f"{GALLERY}/followup_{which}_r{r:02d}.png"
    im.save(out); print(out, im.size)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dial", int(sys.argv[2]) if len(sys.argv) > 2 else 15)
