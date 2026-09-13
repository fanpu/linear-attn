"""Static explainer: one linear-attention read-out = one step of preconditioned gradient descent (d = 2 toy prompt).

  .venv/bin/python 08-icl-linear-attention/render_mechanism.py   -> figures/mechanism.png
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import style

HERE = pathlib.Path(__file__).parent
rng = np.random.default_rng(243)  # a draw where one step lands in the typical place (~80% of the way)
d, N = 2, 8
w = np.array([1.2, -0.7])
X = rng.standard_normal((N, d))
y = X @ w
xq = np.array([0.9, 0.6])
A = np.linalg.inv((1 + 1 / N) * np.eye(d) + d / N * np.eye(d))           # ZFB optimum for Lambda = I
scores = X @ A @ xq                                                         # attention score of query on key i
contrib = scores * y / N
yhat = contrib.sum()
w1 = A @ (X.T @ y / N)

style.use_light()
cmap = style.diverging_light()
fig = plt.figure(figsize=(15, 5.2))
# (1) prompt as a matrix of tokens
ax = fig.add_axes([0.03, 0.14, 0.26, 0.66])
E = np.vstack([np.c_[X.T, xq[:, None]], np.r_[y, 0][None]])
ax.imshow(E, cmap=cmap, vmin=-2.5, vmax=2.5, aspect="auto")
for i in range(E.shape[0]):
    for j in range(E.shape[1]):
        ax.text(j, i, "?" if (i == d and j == N) else f"{E[i, j]:.1f}", ha="center", va="center", fontsize=9, color=style.INK)
ax.set_xticks(range(N + 1)); ax.set_xticklabels([f"$t_{j+1}$" for j in range(N)] + ["query"])
ax.set_yticks(range(d + 1)); ax.set_yticklabels(["$x^{(1)}$", "$x^{(2)}$", "$y$"])
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.add_patch(plt.Rectangle((N - 0.5, -0.5), 1, d + 1, fill=False, ec=style.C["linear"], lw=2.5))
ax.set_title("1  the prompt is a matrix: one column per example", pad=22)
# (2) attention scores and value-weighted contributions
bx = fig.add_axes([0.37, 0.14, 0.27, 0.66])
idx = np.arange(N)
bx.bar(idx - 0.2, scores, width=0.38, color="#c9c4ba", label="score")
bx.bar(idx + 0.2, contrib * N, width=0.38, color=style.C["linear"])
bx.axhline(0, color=style.MUTED, lw=0.8)
bx.set_xticks(idx); bx.set_xticklabels([f"$t_{j+1}$" for j in idx])
bx.set_title("2  score each example, weight its label", pad=22)
bx.text(0.0, 1.0, "gray: attention score  $x_i^\\top A\\, x_q$", transform=bx.transAxes, color=style.MUTED, fontsize=10, va="bottom")
bx.text(1.0, 1.0, "blue: score × label $y_i$", transform=bx.transAxes, color=style.C["linear"], fontsize=10, va="bottom", ha="right")
# (3) same number, read as GD in weight space
cx = fig.add_axes([0.72, 0.14, 0.25, 0.66])
g = np.linspace(-1.2, 2.2, 200)
GX, GY = np.meshgrid(g, g - 1.2)
Lw = ((GX[..., None] * X[:, 0] + GY[..., None] * X[:, 1] - y) ** 2).mean(-1) / 2
cx.contour(GX, GY, Lw, levels=np.geomspace(0.02, 6, 9), colors="#d8d2c6", linewidths=0.9)
cx.scatter(0, 0, s=40, color=style.MUTED, zorder=4)
cx.add_patch(FancyArrowPatch((0, 0), tuple(w1), arrowstyle="-|>", mutation_scale=18, color=style.C["linear"], lw=2.5, zorder=5))
cx.scatter(*w, marker="*", s=260, color="#eda100", zorder=6, edgecolor=style.PAPER)
cx.annotate("true $w$", w, xytext=(8, 6), textcoords="offset points", color="#b37800", fontweight="bold")
cx.annotate("$w_1 = A\\,\\frac{1}{N}\\sum_i y_i x_i$", w1, xytext=(-60, -26), textcoords="offset points", color=style.C["linear"], fontweight="bold", ha="center")
cx.annotate("$w_0 = 0$", (0, 0), xytext=(-8, 8), textcoords="offset points", color=style.MUTED, ha="right")
cx.set_aspect("equal"); cx.set_xlim(-1.2, 2.2); cx.set_ylim(-2.2, 0.9)
cx.set_title("3  ...which is one GD step from $w=0$", pad=22)
cx.set_xlabel("$w^{(1)}$"); cx.set_ylabel("$w^{(2)}$")
fig.text(0.03, 0.93, f"Linear attention's prediction  $\\hat y = \\sum_i \\frac{{1}}{{N}} y_i\\, x_i^\\top A\\, x_q = {yhat:.3f}$   is exactly   "
         f"$w_1^\\top x_q = {w1 @ xq:.3f}$,  one preconditioned gradient step on the in-context least-squares loss.",
         fontsize=13, color=style.INK)
fig.savefig(HERE / "figures" / "mechanism.png")
print(yhat, w1 @ xq)
