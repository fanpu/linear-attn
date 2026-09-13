"""Plotter sheet: the braid as ONE continuous pen line (render step).

usage: python render_plotter.py <cache.npz> <model> <tag> [steps_per_row] [t_start]

c_t (oscillation coordinate; windowed PCA by default, EOS_COORD=u1 for the current top eigenvector) is written as a seismograph
drum sheet: rows of `steps_per_row` steps, joined boustrophedon so the whole sheet is one polyline
with one vertex per GD step. One global gain for every row (the 99.5th percentile of |x| equals 0.9
row spacings); larger excursions cross into neighbouring rows (declared, as on a drum recorder).
Pen 2 (optional red layer): the steps at which lambda_1 > 2/eta, i.e. the GD step is locally
unstable on the quadratic model. Outputs an A2-landscape SVG in mm (plotter ready, layers as
<g inkscape:groupmode="layer">) plus a PNG proof.
"""
import sys

from matplotlib.collections import LineCollection

from eos_common import *  # noqa: F401,F403

A2 = (594.0, 420.0)
MARGIN = 30.0


def build(d, m, spr, ts):
    x = braid(d, m)
    lam = ffill(d["evals"][:, m, 0].astype(float)[:, None])[:, 0]
    inv = float(d["invs"][m])
    T = np.where(np.isfinite(x))[0].max() + 1
    x = np.nan_to_num(x[ts:T])
    above = (lam[ts:T] > inv)
    n = len(x)
    R = int(np.ceil(n / spr))
    Wd, Hd = A2[0] - 2 * MARGIN, A2[1] - 2 * MARGIN - 14
    dy = Hd / R
    gain = 0.9 * dy / (np.percentile(np.abs(x), 99.5) + 1e-12)
    i = np.arange(n)
    row, col = i // spr, i % spr
    colx = np.where(row % 2 == 0, col, spr - 1 - col)  # boustrophedon
    X = MARGIN + colx / (spr - 1) * Wd
    Y = MARGIN + (row + 0.5) * dy - gain * x  # SVG y grows downward
    return X, Y, above, R, dy, inv, T


def write_svg(path, X, Y, above, meta_txt):
    pts = " ".join(f"{a:.3f},{b:.3f}" for a, b in zip(X, Y))
    red = []
    runs = np.flatnonzero(np.diff(np.r_[0, above.astype(int), 0]))
    for s, e in zip(runs[::2], runs[1::2]):
        e = min(e + 1, len(X))
        if e - s >= 2:
            red.append(" ".join(f"{a:.3f},{b:.3f}" for a, b in zip(X[s:e], Y[s:e])))
    with open(path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
                f'width="{A2[0]}mm" height="{A2[1]}mm" viewBox="0 0 {A2[0]} {A2[1]}">\n')
        f.write(f"<!-- {meta_txt} -->\n")
        f.write('<g inkscape:groupmode="layer" inkscape:label="1 black braid" fill="none" stroke="#000" stroke-width="0.3">\n')
        f.write(f'<polyline points="{pts}"/>\n</g>\n')
        f.write('<g inkscape:groupmode="layer" inkscape:label="2 red above-edge" fill="none" stroke="#c0392b" stroke-width="0.3">\n')
        for r in red:
            f.write(f'<polyline points="{r}"/>\n')
        f.write("</g>\n</svg>\n")
    print("wrote", path)


def proof(X, Y, above, name, text, two_pen):
    fig = plt.figure(figsize=(A2[0] / 25.4, A2[1] / 25.4), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=PAPER)
    ax.set_xlim(0, A2[0])
    ax.set_ylim(A2[1], 0)
    ax.axis("off")
    ax.plot(X, Y, color=INK, lw=0.3 * 72 / 25.4, solid_joinstyle="round", alpha=0.9)
    if two_pen:
        seg = np.stack([np.stack([X[:-1], Y[:-1]], 1), np.stack([X[1:], Y[1:]], 1)], 1)[above[:-1]]
        ax.add_collection(LineCollection(seg, colors="#c0392b", linewidths=0.3 * 72 / 25.4, alpha=0.85))
    ax.text(MARGIN, A2[1] - MARGIN + 6, text, fontsize=9, family=MONO, color="#6b675e", va="top")
    save(fig, name, dpi=150, facecolor=PAPER)


if __name__ == "__main__":
    f, m, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    spr = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    ts = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    d = load(f)
    X, Y, above, R, dy, inv, T = build(d, m, spr, ts)
    txt = (f"η = 2/{inv:.0f}   steps {ts}–{T - 1}, {spr} per row, one vertex per step, one continuous line"
           f"   ·   {coord_label(short=True)}   ·   fc-tanh on CIFAR-10 5k, full-batch GD")
    write_svg(os.path.join(GAL, f"plotter_{tag}.svg"), X, Y, above, txt)
    proof(X, Y, above, f"plotter_{tag}_oneink.png", txt, False)
    proof(X, Y, above, f"plotter_{tag}_twopen.png", txt + "   ·   red pen: λ₁ > 2/η", True)
