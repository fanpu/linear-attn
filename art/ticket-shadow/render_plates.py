"""Specimen plates with one shared treatment (ink dots on cream, area = value / panel max):
  plate_nulls.png    the shadow at one matched round under every control, with its numbers
  plate_classes.png  per-class maps: data contrast, ticket-at-init, trained ticket, path counts, null
  plate_curve.png    test accuracy vs weights remaining (is it a ticket?)
Usage: python render_plates.py --round 13
"""
import argparse, json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from common import *

P = "mnist_adam_norm_rw0"


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def grid_sheet(panels, cols, cell, title, subtitle, out, cap_lines=2, row_labels=None):
    """panels: list of (values784 or None, caption list[str], signed bool)."""
    panel = 28 * cell
    fs = max(12, panel // 30)
    gap = int(panel * 0.14); cap = int(fs * 1.5 * cap_lines + fs)
    rows = int(np.ceil(len(panels) / cols))
    lab_w = int(panel * 0.55) if row_labels else 0
    margin = int(panel * 0.3); head = int(fs * 5.5)
    W = 2 * margin + lab_w + cols * panel + (cols - 1) * gap
    H = margin + head + rows * (panel + cap) + (rows - 1) * gap + margin // 2
    im = Image.new("RGB", (W, H), CREAM)
    d = ImageDraw.Draw(im)
    text(d, (margin, margin), title, int(fs * 1.35))
    text(d, (margin, margin + int(fs * 2.1)), subtitle, fs, MUTED)
    for k, (v, caps, signed) in enumerate(panels):
        i, j = divmod(k, cols)
        x = margin + lab_w + j * (panel + gap); y = margin + head + i * (panel + cap + gap)
        if row_labels and j == 0:
            for t, line in enumerate(row_labels[i].split("\n")):
                text(d, (margin, y + t * int(fs * 1.4)), line, fs, INK if t == 0 else MUTED)
        if v is None:
            continue
        im.paste(dots(v, cell), (x, y))
        for t, line in enumerate(caps):
            text(d, (x, y + panel + int(fs * 0.5) + t * int(fs * 1.4)), line, fs, INK if t == 0 else MUTED)
    im.save(out)
    print(out, im.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=13)
    ap.add_argument("--cell", type=int, default=16)
    a = ap.parse_args()
    z = load_analysis(); st = load_stats()
    r = a.round
    ms = lambda k: z[f"data_mnist_{k}"]
    fs_ = lambda k: z[f"data_fashion_{k}"]

    def sh(cond, mode="imp", k=0):
        return z[f"{cond}__shadow"][r, imp_idx(z, cond, mode)[k]]

    def num(cond, mode="imp"):
        d = st[cond][r][mode]
        return f"r(std) {d['r_std']:.2f}  R² {d['R2_iso_std']:.2f}", f"seeds r {d['seed_r']:.2f}  test {100 * d['es_test']:.1f}%"

    def cross(cond, mode="imp"):
        return st["_cross_vs_primary"][f"{cond}/{mode}"][r]

    # residual of the primary after the best monotone function of std
    s0 = sh(P).astype(float)
    fit = IsotonicRegression(increasing=True).fit(ms("std"), s0).predict(ms("std"))
    frac = z[f"{P}__counts"][r, 0] / 235200
    panels = [
        (ms("std"), ["MNIST pixel std", "the data's own map"], False),
        (ms("mean"), ["MNIST mean image", ""], False),
        (fs_("std"), ["Fashion pixel std", ""], False),
        (z[f"{P}__w1_disp0"][0], ["|W_T - W_0|, dense run", f"r(shadow) {st[P][r]['imp']['r_disp0']:.2f}  R² {st[P][r]['imp']['R2_iso_disp0']:.2f}"], False),

        (sh(P), ["IMP · Adam · rewind 0"] + [" · ".join(num(P))], False),
        (sh("mnist_adam_norm_rw500"), ["IMP · rewind to step 500", f"r(primary) {cross('mnist_adam_norm_rw500'):.2f}"], False),
        (sh("pmnist_adam_norm_rw0"), ["permuted MNIST, un-permuted", f"r(primary) {cross('pmnist_adam_norm_rw0'):.2f}"], False),
        (sh("fashion_adam_norm_rw0"), ["IMP on Fashion-MNIST", f"r(primary) {cross('fashion_adam_norm_rw0'):.2f}  r(F-std) {st['fashion_adam_norm_rw0'][r]['imp']['r_std']:.2f}"], False),

        (sh("mnist_nulls", "random"), ["random pruning", f"r(std) {st['mnist_nulls'][r]['random']['r_std']:.2f}"], False),
        (sh("mnist_nulls", "maginit"), ["prune by |init| only", f"r(std) {st['mnist_nulls'][r]['maginit']['r_std']:.2f}"], False),
        (sh("mnist_sgd_norm_rw0"), ["IMP · plain SGD", f"r(std) {st['mnist_sgd_norm_rw0'][r]['imp']['r_std']:.2f}  r(|W_T-W_0|) {st['mnist_sgd_norm_rw0'][r]['imp']['r_disp0']:.2f}"], False),
        (sh("mnist_adam_raw_rw0"), ["IMP · Adam · raw [0,1] px", f"r(std) {st['mnist_adam_raw_rw0'][r]['imp']['r_std']:.2f}  r(|W_T-W_0|) {st['mnist_adam_raw_rw0'][r]['imp']['r_disp0']:.2f}"], False),

        (s0 - fit, ["primary − f(std), residual", f"f = best monotone fit · resid seed r {st[P][r]['imp']['resid_seed_r']:.2f}"], True),
        (z["mnist_adam_raw_rw0__w1_disp0"][0], ["|W_T - W_0|, raw-pixel run", "Adam moves rare pixels most"], False),
        (z["mnist_sgd_norm_rw0__w1_disp0"][0], ["|W_T - W_0|, SGD run", ""], False),
        (sh("mnist_adam_raw_rw0") - IsotonicRegression(increasing=True).fit(ms("std"), sh("mnist_adam_raw_rw0").astype(float)).predict(ms("std")),
         ["raw-pixel run − f(std)", f"resid seed r {st['mnist_adam_raw_rw0'][r]['imp']['resid_seed_r']:.2f}"], True),
    ]
    rl = ["DATA / MECHANISM\nwhat the shadow\ncould be", "IMP TICKETS\nsame round,\nfour data", "CONTROLS\nsame sparsity", "WHAT IS LEFT\nred = fewer\nthan std predicts"]
    grid_sheet(panels, 4, a.cell, f"THE TICKET'S SHADOW — NULL CONTROLS AT ROUND {r} ({100 * frac:.1f}% OF INPUT WEIGHTS LEFT)",
               "seed 0 shown; numbers are means over seeds · dot area = value / panel max · r = Pearson over 784 pixels · R² = best monotone function of std",
               f"{GALLERY}/plate_nulls_r{r:02d}.png", row_labels=rl)

    # ---------------- classes
    T = ms("cmean") - ms("cmean").mean(0, keepdims=True)

    def contrast(A, signed=True):
        K = A / (np.abs(A).sum(1, keepdims=True) + 1e-30)
        return K - K.mean(0, keepdims=True)
    ii = imp_idx(z, P); m0 = ii[0]
    s0avg = np.mean([contrast(z[f"{P}__signed0"][r, m]) for m in ii], 0)
    rows = [
        ("CLASS MEAN − MEAN\nof the data", T, None, None),
        ("DENSE NET, TRAINED\npath sum, ReLUs on\n(before any pruning)", contrast(z[f"{P}__signed"][0, m0]), (P, 0, "imp", "signed"), None),
        ("TRAINED TICKET\npath sum, ReLUs on", contrast(z[f"{P}__signed"][r, m0]), (P, r, "imp", "signed"), None),
        ("TICKET AT BIRTH\nmask × init weights,\npath sum", contrast(z[f"{P}__signed0"][r, m0]), (P, r, "imp", "signed0"), None),
        (f"TICKET AT BIRTH\nmean of {len(ii)} seeds' maps", s0avg, None, "avg"),
        ("PATH COUNTS\nunsigned, mask only", contrast(z[f"{P}__paths"][r, m0]), (P, r, "imp", "paths"), None),
        ("NULL: RANDOM MASK\n× init weights", contrast(z[f"mnist_nulls__signed0"][r, imp_idx(z, 'mnist_nulls', 'random')[0]]),
         ("mnist_nulls", r, "random", "signed0"), None),
    ]
    panels, lab = [], []
    for name, A, key, extra in rows:
        for c in range(10):
            cap = [f"{c}", f"r {corr(A[c], T[c]):.2f}" if (key or extra) else ""]
            panels.append((A[c], cap, True))
        if key:
            cond_, rr, mode_, k_ = key
            d = st[cond_][rr][mode_]
            lab.append(name + f"\nr {d[k_ + '_diag']:.2f}, {d[k_ + '_own']:.1f}/10 own")
        elif extra:
            dg = np.mean([corr(A[c], T[c]) for c in range(10)])
            lab.append(name + f"\nr {dg:.2f}")
        else:
            lab.append(name)
    grid_sheet(panels, 10, max(9, a.cell // 2 + 3), f"THE TICKET'S SHADOW — ONE MASK, TEN CLASSES (ROUND {r}, {100 * frac:.1f}% LEFT)",
               "each map is the class's share minus the mean over classes · ink = more, madder = less · r = corr with the data row above · "
               "'own' = classes whose best-matching data template is their own (chance 1/10), mean over seeds",
               f"{GALLERY}/plate_classes_r{r:02d}.png", row_labels=lab)

    # ---------------- accuracy curve (declared: matplotlib, same palette)
    hx = lambda c: "#%02x%02x%02x" % c
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=220, facecolor=hx(CREAM))
    ax.set_facecolor(hx(CREAM))
    series = [(P, "imp", "IMP, rewind to init (5 seeds)", INK, "-"),
              (P, "reinit", "same masks, random re-init", MADDER, "-"),
              ("mnist_adam_norm_rw500", "imp", "IMP, rewind to step 500", INK, "--"),
              ("mnist_nulls", "random", "random pruning", MUTED, "-"),
              ("mnist_nulls", "maginit", "prune by |init|", MUTED, ":"),
              ("mnist_adam_raw_rw0", "imp", "IMP, raw [0,1] pixels", INK, (0, (1, 3))),
              ("mnist_sgd_norm_rw0", "imp", "IMP, plain SGD", MADDER, "--")]
    for cond, mode, lab_, col, ls in series:
        if f"{cond}__es_test" not in z:
            continue
        acc = z[f"{cond}__es_test"][:, imp_idx(z, cond, mode)]
        fr = z[f"{cond}__counts"].sum(1) / 266200
        mu, sd = acc.mean(1) * 100, acc.std(1) * 100
        ax.plot(fr * 100, mu, ls=ls, color=hx(col), lw=1.2, label=lab_)
        ax.fill_between(fr * 100, mu - sd, mu + sd, color=hx(col), alpha=0.12, lw=0)
    ax.set_xscale("log"); ax.invert_xaxis()
    tk = [100, 50, 20, 10, 5, 2, 1]
    ax.set_xticks(tk); ax.set_xticklabels([str(t) for t in tk], family="monospace"); ax.minorticks_off()
    ax.set_xlabel("weights remaining (%)", family="monospace"); ax.set_ylabel("test accuracy at early stop (%)", family="monospace")
    ax.set_ylim(95.5, 99)
    ax.text(0.99, 0.02, "random / |init| pruning continue below the frame", transform=ax.transAxes, ha="right", fontsize=7, family="monospace", color=hx(MUTED))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, prop={"family": "monospace", "size": 8})
    ax.set_title("Is it a ticket?  LeNet-300-100 on MNIST, mean ± sd over seeds", family="monospace", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(f"{GALLERY}/plate_curve.png", facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"{GALLERY}/plate_curve.png")


if __name__ == "__main__":
    main()
