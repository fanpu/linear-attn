"""Cosine-matrix grid over training. Each tile is a C x C matrix of cosines between centred
class means; upper triangle = one quantity, lower triangle = another (train/test or means/W).
The ideal simplex ETF has every off-diagonal entry = -1/(C-1); the colour is split there.

  python render_gram.py c10 --pair train_test|means_W --style spectral|riso|ink [--rows 6 --cols 8]
"""
import argparse, sys
import numpy as np
import nclib as N
import artlib as A
sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import palettes as P

PAIRS = {"train_test": ("gram_M", "gram_Mtest"), "means_W": ("gram_M", "gram_W")}


def pick(ck, n):
    """n checkpoints evenly spaced in log(1 + 20*epoch) (dense early, sparse late)."""
    E = np.array([e for e, _ in ck])
    t = np.log1p(20 * E)
    tgt = np.linspace(t.min(), t.max(), n)
    idx = []
    for g in tgt:
        order = np.argsort(np.abs(t - g))
        for i in order:
            if i not in idx:
                idx.append(i); break
    return [ck[i] for i in sorted(idx)]


def main(tag, pair, style, rows, cols, tile_px=300):
    meta, mets, ck = N.load_run(tag)
    C = len(meta["classes"]); ideal = -1 / (C - 1)
    sel = pick(ck, rows * cols)
    ku, kl = PAIRS[pair]
    G = []
    for e, p in sel:
        z = np.load(p)
        g = np.array(z[ku]); gl = np.array(z[kl])
        tri = np.tril_indices(C, -1)
        g[tri] = gl[tri]
        G.append(g)
    G = np.array(G)
    off = ~np.eye(C, dtype=bool)
    dev = G - ideal
    gap = int(tile_px * 0.16); pad = int(tile_px * 0.5)
    W = cols * tile_px + (cols - 1) * gap + 2 * pad
    H = rows * tile_px + (rows - 1) * gap + 2 * pad
    cell = tile_px // C
    if style == "spectral":
        bg = "#101014"
        v = P.signed_rank_normalize(dev[:, off].ravel(), near_boundary="small")
        cmap = P.split_cmap(P.SIDES["spectral_purple"], P.SIDES["spectral_red"])  # neg (more obtuse) purple, pos red
        colors = cmap((v + 1) / 2)[:, :3]
    elif style.startswith("pal_"):  # palettes.py split pairing, e.g. pal_aurora_ember (declared variant)
        pr = P.PAIRINGS[style[4:]]
        bg = pr.get("ground") or "#0c0c10"
        colors = P.render_split(dev[:, off].ravel(), style[4:], near_boundary="small")
    elif style == "riso":
        bg = P.RISO_PAPER
        v = P.signed_rank_normalize(dev[:, off].ravel(), near_boundary="small", pastel=1.0)
        # coverage grows with |rank|; blue ink for more-obtuse-than-ideal, fluo pink for more-acute
        cov = np.abs(v)
        ink = np.where((v < 0)[:, None], A.rgb(P.RISO["blue"]), A.rgb(P.RISO["fluo_pink"]))
        colors = A.rgb(bg) * (1 - cov[:, None] * (1 - ink))
    img = np.ones((H, W, 3)) * A.rgb(bg)
    k = 0
    for t in range(len(sel)):
        r, c = divmod(t, cols)
        y0 = pad + r * (tile_px + gap); x0 = pad + c * (tile_px + gap)
        for i in range(C):
            for j in range(C):
                if i == j:
                    continue
                img[y0 + i * cell + 1:y0 + (i + 1) * cell - 1, x0 + j * cell + 1:x0 + (j + 1) * cell - 1] = 0
        # write colours in the same row-major order as dev[:, off]
    cols_t = colors.reshape(len(sel), C * C - C, 3)
    for t in range(len(sel)):
        r, c = divmod(t, cols)
        y0 = pad + r * (tile_px + gap); x0 = pad + c * (tile_px + gap)
        q = 0
        for i in range(C):
            for j in range(C):
                if i == j:
                    continue
                img[y0 + i * cell + 1:y0 + (i + 1) * cell - 1, x0 + j * cell + 1:x0 + (j + 1) * cell - 1] = cols_t[t, q]
                q += 1
    fig = A.canvas(W, H, bg)
    ax = A.panel(fig, [0, 0, 1, 1], (0, W, H, 0))
    ax.imshow(img, extent=(0, W, H, 0), interpolation="nearest")
    tc = "#6d675c" if style == "riso" else "#b9b4a6"
    for t, (e, _) in enumerate(sel):
        r, c = divmod(t, cols)
        y0 = pad + r * (tile_px + gap); x0 = pad + c * (tile_px + gap)
        lab = f"ep {e:.0f}" if e >= 1 else (f"it {round(e * 5000 * C / meta['bs'])}" if e > 0 else "init")
        ax.text(x0, y0 + tile_px + gap * 0.12, lab, color=tc, fontsize=tile_px / 60, va="top", family="monospace")
    dmax = np.abs(dev[-1][off]).max()
    print(f"{tag} {pair}: final max|cos - ideal| (both triangles) = {dmax:.4f}; first = {np.abs(dev[0][off]).max():.3f}")
    return A.save(fig, f"gram_{tag}_{pair}_{style}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("--pair", default="train_test"); ap.add_argument("--style", default="spectral")
    ap.add_argument("--rows", type=int, default=6); ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--tile", type=int, default=300)
    a = ap.parse_args()
    print(main(a.tag, a.pair, a.style, a.rows, a.cols, a.tile))
