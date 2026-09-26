"""Shared drawing kit: the surveyor's register (ink on cream), map loading, road drawing."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from metrics import load

ROOT = os.path.dirname(os.path.abspath(__file__))
GAL = os.path.join(ROOT, "gallery")

# declared palette: two inks on cream, like a hand-coloured survey sheet
PAPER = "#f2ebdc"
INK = "#1c1a17"
RED = "#b3402f"      # surveyor's red: construction lines and the null
FAINT = "#8c8374"
RULE = "#cfc5b0"
FONT = "C059"        # Century Schoolbook: the face of nineteenth-century survey sheets

plt.rcParams.update({"font.family": FONT, "text.color": INK, "axes.edgecolor": INK,
                     "savefig.facecolor": PAPER, "figure.facecolor": PAPER})

TASK_TITLE = {"mnist": "MNIST", "fashion": "Fashion-MNIST", "cifar": "CIFAR-10", "modadd": "a + b mod 97"}
ARCH_NAME = {"logreg": "logistic regression", "mlp_256": "MLP 1×256", "mlp_2048": "MLP 1×2048",
             "mlp_deep": "MLP 4×512", "cnn": "CNN", "resnet": "ResNet-8", "vit": "ViT", "gru": "GRU over rows",
             "mlp": "MLP", "tf1": "transformer, 1 layer", "tf2": "transformer, 2 layers"}


class Map:
    """One InPCA embedding, oriented and scaled for drawing.

    Declared: the (PC1, PC2) plane is rotated so Ignorance -> Truth runs left to right, PC2 is flipped so
    the real roads bow upward, and the plane is scaled so |Ignorance -> Truth| = 1 on every sheet."""

    def __init__(self, task, split, mname, metric="B"):
        self.task, self.split, self.mname = task, split, mname
        suf = "" if metric == "B" else "_" + metric
        self.L = load(task, split, metric)
        z = np.load(os.path.join(ROOT, "cache", f"map_{task}_{split}_{mname}{suf}.npz"))
        self.idx, X, self.lam = z["idx"], z["X"].copy(), z["lam"]
        self.pos = {g: i for i, g in enumerate(self.idx)}
        k = self.L["kinds"]
        self.i0 = self.pos[int(np.where(k == "P0")[0][0])]; self.i1 = self.pos[int(np.where(k == "Pstar")[0][0])]
        self.scale = np.linalg.norm(X[self.i1, :2] - X[self.i0, :2])
        X = X / self.scale
        runs = self.L["runs"]
        true_pts = [self.pos[g] for g in self.idx if k[g] == "run" and runs[self.L["run"][g]]["labels"] == "true"]
        if np.mean(X[true_pts, 1]) < 0:
            X[:, 1] *= -1
        self.X = X
        self.runs = runs
        stress = json.load(open(os.path.join(ROOT, "cache", f"metrics_{task}{suf}.json")))[split][f"stress_{mname}"]
        self.stress = stress

    def traj(self, kind, ri):
        g = [x for x in self.idx if self.L["kinds"][x] == kind and self.L["run"][x] == ri]
        g = sorted(g, key=lambda x: self.L["k"][x])
        return self.X[[self.pos[x] for x in g]]

    def run_ids(self, labels="true"):
        ids = sorted(set(int(self.L["run"][g]) for g in self.idx if self.L["kinds"][g] == "run"))
        return [i for i in ids if labels is None or self.runs[i]["labels"] == labels]

    def points(self, kind):
        return self.X[[self.pos[g] for g in self.idx if self.L["kinds"][g] == kind]]

    def info(self, kind):
        return [self.L["info"][g] for g in self.idx if self.L["kinds"][g] == kind]


def trig_point(ax, xy, s=1.0, z=10):
    """Ignorance: a surveyor's triangulation pillar (triangle with a dot)."""
    ax.scatter([xy[0]], [xy[1]], marker="^", s=120 * s, facecolor=PAPER, edgecolor=INK, linewidth=1.1 * s, zorder=z)
    ax.scatter([xy[0]], [xy[1]], marker="o", s=6 * s, color=INK, zorder=z + 1)


def city(ax, xy, s=1.0, z=10):
    """Truth: a town symbol (ringed dot)."""
    ax.scatter([xy[0]], [xy[1]], marker="o", s=110 * s, facecolor=PAPER, edgecolor=INK, linewidth=1.1 * s, zorder=z)
    ax.scatter([xy[0]], [xy[1]], marker="o", s=26 * s, color=INK, zorder=z + 1)


def road(ax, P, color=INK, lw=0.6, alpha=0.85, z=5, end=True):
    ax.plot(P[:, 0], P[:, 1], color=color, lw=lw, alpha=alpha, solid_capstyle="round", solid_joinstyle="round", zorder=z)
    if end:
        ax.scatter([P[-1, 0]], [P[-1, 1]], s=4 * lw, color=color, zorder=z, linewidths=0)


def survey_line(ax, color=RED, lw=0.6, z=3):
    """The straight road: the geodesic Ignorance -> Truth on the sphere of sqrt-probabilities."""
    return None


def neat_line(ax, xlim, ylim, tick=0.1, lw=0.8, labels=True, fs=6):
    """Map border: a double neat line with graduated ticks in map units."""
    x0, x1 = xlim; y0, y1 = ylim
    for pad, w in ((0.0, lw), (0.012 * (x1 - x0), lw * 0.4)):
        ax.plot([x0 - pad, x1 + pad, x1 + pad, x0 - pad, x0 - pad], [y0 - pad, y0 - pad, y1 + pad, y1 + pad, y0 - pad],
                color=INK, lw=w, zorder=20, clip_on=False)
    t = tick
    for xv in np.arange(np.ceil(x0 / t) * t, x1 + 1e-9, t):
        for yy, dy in ((y0, 1), (y1, -1)):
            ax.plot([xv, xv], [yy, yy + dy * 0.012 * (x1 - x0)], color=INK, lw=lw * 0.5, zorder=20)
    for yv in np.arange(np.ceil(y0 / t) * t, y1 + 1e-9, t):
        for xx, dx in ((x0, 1), (x1, -1)):
            ax.plot([xx, xx + dx * 0.012 * (x1 - x0)], [yv, yv], color=INK, lw=lw * 0.5, zorder=20)
    ax.set_xlim(x0 - 0.02 * (x1 - x0), x1 + 0.02 * (x1 - x0)); ax.set_ylim(y0 - 0.02 * (x1 - x0), y1 + 0.02 * (x1 - x0))
    ax.set_aspect("equal"); ax.axis("off")


def save(fig, name, dpi=300):
    os.makedirs(GAL, exist_ok=True)
    p = os.path.join(GAL, name)
    fig.savefig(p, dpi=dpi, facecolor=PAPER)
    plt.close(fig)
    print("wrote", p)
    return p
