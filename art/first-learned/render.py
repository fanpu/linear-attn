"""Render the First Learned mosaics and plates from cache/stats.npz (CPU only).

Declared aesthetic choices
  ground   cream  #F2EDE1 ; ink  #1E1C1A (warm near-black) ; madder  #A3302A
  each digit is drawn at its native 28x28 pixels, pixel value = ink opacity (a faithful glyph);
  cells are 28 px with no gutter (digits already carry a 4 px empty border in MNIST).
  reading order: left-to-right, top-to-bottom = learned earliest -> learned last.
"""
import os, struct, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(ROOT, "cache"); G = os.path.join(ROOT, "gallery")
CREAM = np.array([242, 237, 225], np.float32); INK = np.array([30, 28, 26], np.float32)
MADDER = np.array([163, 48, 42], np.float32); PALE = np.array([196, 188, 174], np.float32); GREY = np.array([150, 144, 134], np.float32)
Image.MAX_IMAGE_PIXELS = None


def idx(path):
    with open(path, "rb") as f:
        nd = struct.unpack(">I", f.read(4))[0] & 0xFF
        shape = struct.unpack(">" + "I" * nd, f.read(4 * nd))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


X = idx(os.path.join(ROOT, "..", "data", "MNIST", "raw", "train-images-idx3-ubyte"))
S = np.load(os.path.join(C, "stats.npz"))
Y = S["y"]


def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def mosaic(order, color_idx=None, cols=245, cell=28, colors=(INK, MADDER, GREY)):
    """order: example indices in reading order. color_idx: per-example colour index into colors."""
    n = len(order); rows = (n + cols - 1) // cols
    a = np.zeros((rows * cols, 28, 28), np.float32)
    a[:n] = X[order] / 255.0
    ci = np.zeros(rows * cols, np.int64)
    if color_idx is not None:
        ci[:n] = color_idx[order]
    col = np.stack(colors)[ci]  # (rows*cols, 3)
    img = CREAM[None, None, None] * (1 - a[..., None]) + col[:, None, None, :] * a[..., None]
    img = img.reshape(rows, cols, 28, 28, 3).transpose(0, 2, 1, 3, 4).reshape(rows * 28, cols * 28, 3)
    if cell != 28:
        im = Image.fromarray(img.clip(0, 255).astype(np.uint8)).resize((cols * cell, rows * cell), Image.LANCZOS)
        return np.asarray(im)
    return img.clip(0, 255).astype(np.uint8)


def order_of(key):
    return np.argsort(key, kind="stable")


def save(arr, name, margin=0.08, caption=None, fs=None, ticks=None):
    h, w = arr.shape[:2]; m = int(margin * w)
    cap_h = int(m * 1.2) if caption else 0
    canvas = Image.new("RGB", (w + 2 * m, h + 2 * m + cap_h), tuple(CREAM.astype(int)))
    canvas.paste(Image.fromarray(arr), (m, m))
    if caption:
        d = ImageDraw.Draw(canvas); f = font(fs or max(12, m // 7))
        d.multiline_text((m, h + m + m // 4), caption, fill=tuple(INK.astype(int)), font=f, spacing=f.size // 2)
    if ticks is not None:  # left-margin ledger: median (over runs) stable-learned step of the digits in that row
        o, ST = ticks; d = ImageDraw.Draw(canvas); f = font(max(10, m // 8)); rows = int(np.ceil(len(o) / 245)); rh = h / rows
        for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            r = int(frac * (rows - 1)); ex = o[r * 245:(r + 1) * 245]
            med = np.nanmedian(ST[:, ex]); nv = np.isnan(ST[:, ex]).mean()
            y0 = m + r * rh + rh / 2
            d.line([(m - m // 8, y0), (m - m // 40, y0)], fill=tuple(INK.astype(int)), width=2)
            lab = f"{frac:.0%}" + (f"\nstep {med:.0f}" if nv < 0.5 else "\nnever")
            d.multiline_text((m - m // 7, y0), lab, fill=tuple(INK.astype(int)), font=f, anchor="rm", align="right")
    p = os.path.join(G, name)
    if canvas.width > 6000:  # full resolution kept in cache/, a 5,000 px-wide copy in gallery/
        canvas.save(os.path.join(C, "full_" + name), optimize=True)
        canvas = canvas.resize((5000, int(canvas.height * 5000 / canvas.width)), Image.LANCZOS)
    canvas.save(p, optimize=True)
    print("wrote", p, canvas.size)
    return canvas


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(G, exist_ok=True)

    K = S["keys"]; true = list(S["true"]); archs = sorted(set(n.rsplit("_true_", 1)[0] for n in true))
    net = np.array([not n.startswith("logreg") for n in true])
    F = S["forgets_late"][net]; never = np.isinf(S["stables"][net]).mean(0)
    archs = [a for a in ["mlp_256", "cnn", "vit", "resnetbnb"] if a in archs]; nrun = int(net.sum())
    NICE = {"mlp_256": "MLP", "cnn": "CNN", "vit": "ViT", "resnetbnb": "ResNet-8"}
    # forgettable (declared): forgotten (correct -> wrong) after the first epoch in a majority of runs, or never stably learned in a majority
    forg_major = ((F > 0).mean(0) > 0.5) | (never > 0.5)
    print("forgettable (majority rule):", forg_major.sum())

    if which in ("all", "hero"):
        o = order_of(S["cons"])
        m = mosaic(o, forg_major.astype(int))
        Image.fromarray(m).save(os.path.join(C, "hero_full.png"))
        ST = S["stables"][net]; ST = np.where(np.isinf(ST), np.nan, ST)
        cv = save(np.asarray(Image.fromarray(m).resize((m.shape[1] // 2, m.shape[0] // 2), Image.LANCZOS)),
             "first_learned_hero.png", caption=f"First Learned. All 60,000 MNIST training digits in the order {nrun} small networks "
             f"({len(archs)} architectures x 3 seeds) learned them.\nMadder: forgotten after the first epoch in most runs.\nMargin: fraction of the set; median step (of 3,744) by which that row was stably learned.", ticks=(o, ST))

    arch_order = {a: order_of(S[f"arch_{a}"]) for a in archs}
    if which in ("all", "becher"):
        # one mosaic per architecture (3-seed mean order); madder = forgettable in that architecture's runs
        tiles = []
        for a in archs:
            ri = [i for i, n in enumerate(true) if n.startswith(a + "_true_")]
            fa = ((S["forgets_late"][ri] > 0).mean(0) > 0.5) | (np.isinf(S["stables"][ri]).mean(0) > 0.5)
            m = mosaic(arch_order[a], fa.astype(int), cell=12)
            tiles.append((a, m, fa.sum()))
        h, w = tiles[0][1].shape[:2]; g = w // 20
        big = np.full((h, len(tiles) * w + (len(tiles) - 1) * g, 3), CREAM, np.uint8)
        for i, (a, m, _) in enumerate(tiles):
            big[:, i * (w + g):i * (w + g) + w] = m
        save(big, "becher_four_architectures.png", margin=0.03,
             caption="      ".join(f"{NICE[a]}: {nf:,} forgettable" for a, _, nf in tiles) +
             "\nEach: all 60,000 digits in that architecture's 3-seed mean learning order. Madder: forgotten after epoch 1 in 2 of 3 seeds, or never stably learned.", fs=80)

    if which in ("all", "agree"):
        # agreement plates: order by run/arch A; madder = the last-learned 5% according to B
        q = int(0.05 * len(Y)); rng = np.random.default_rng(0)
        def lastset(key):
            return np.argsort(key)[-q:]
        def classmatched_random(sel):
            out = []
            for c in range(10):
                n = (Y[sel] == c).sum(); out += list(rng.choice(np.where(Y == c)[0], n, replace=False))
            return np.array(out)
        kid = {n: i for i, n in enumerate(true)}
        pan = []
        a0 = archs[0]
        RB = "resnetbnb" if "resnetbnb_true_s1" in kid else "resnet"
        pairs = [("CNN seed 0 order; madder = CNN seed 1's last 5%", K[kid["cnn_true_s0"]], K[kid["cnn_true_s1"]]),
                 ("MLP seed 0 order; madder = CNN seed 1's last 5%", K[kid["mlp_256_true_s0"]], K[kid["cnn_true_s1"]]),
                 ("ViT seed 0 order; madder = ResNet seed 1's last 5%", K[kid["vit_true_s0"]], K[kid[f"{RB}_true_s1"]]),
                 ("null: CNN seed 0 order; madder = class-matched random 5%", K[kid["cnn_true_s0"]], None)]
        for title, ka, kb in pairs:
            sel = classmatched_random(lastset(K[kid["cnn_true_s1"]])) if kb is None else lastset(kb)
            flag = np.zeros(len(Y), int); flag[sel] = 1
            m = mosaic(order_of(ka), flag, cell=10, colors=(PALE, MADDER, GREY))
            # fraction of B's last 5% that fall in A's last 5%
            pos = np.empty(len(Y), int); pos[order_of(ka)] = np.arange(len(Y))
            hit = (pos[sel] >= len(Y) - q).mean()
            pan.append((title, m, hit))
        h, w = pan[0][1].shape[:2]; g = w // 16
        big = np.full((h, len(pan) * w + (len(pan) - 1) * g, 3), CREAM, np.uint8)
        for i, (_, m, _) in enumerate(pan):
            big[:, i * (w + g):i * (w + g) + w] = m
        save(big, "agreement_plates.png", margin=0.03,
             caption="\n".join(f"{i+1}. {t}: {100*h_:.0f}% of madder lands in the last 5% of rows" for i, (t, _, h_) in enumerate(pan)), fs=70)

    if which in ("all", "specimen"):
        o = order_of(S["cons"]); z = 3; cell = 28 * z; gap = cell
        cols = 10; W = (2 * cols) * cell + gap; H = 10 * (cell + cell // 3)
        img = Image.new("RGB", (W, H), tuple(CREAM.astype(int))); d = ImageDraw.Draw(img); f = font(cell // 5)
        for c in range(10):
            oc = o[Y[o] == c]
            row = list(oc[:cols]) + [None] + list(oc[-cols:])
            x = 0
            for e in row:
                if e is None:
                    x += gap; continue
                col = MADDER if forg_major[e] else INK
                a = (X[e] / 255.0)[..., None]
                g_ = (CREAM * (1 - a) + col * a).clip(0, 255).astype(np.uint8)
                img.paste(Image.fromarray(g_).resize((cell, cell), Image.NEAREST), (x, c * (cell + cell // 3)))
                if S["mode"][e] != c:
                    d.text((x + 4, c * (cell + cell // 3) + cell), f"reads {S['mode'][e]}", fill=tuple(MADDER.astype(int)), font=f)
                x += cell
        save(np.asarray(img), "specimen_first_last.png", margin=0.05,
             caption="Per class: the ten digits learned first (left) and last (right), consensus of all runs. "
                     "'reads k' = most runs end up predicting k.", fs=28)
