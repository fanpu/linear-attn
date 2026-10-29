"""One matplotlib style for every figure in the course, and a phase-portrait helper.  [AI]"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: scripts only write files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Colour-blind-safe categorical palette (Okabe-Ito), used in this order everywhere.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def set_style() -> None:
    """Apply the course style: serif text sized for a letter-page PDF, light grid, no top/right spines."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "lines.linewidth": 1.4, "mathtext.fontset": "cm",
    })


def savefig(fig, path: str | Path) -> Path:
    """Save and close; creates the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def phase_portrait(vector_field: Callable[[np.ndarray], np.ndarray],
                   trajectories: Sequence[np.ndarray] = (),
                   *, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), ax=None, density: float = 1.1,
                   traj_colors: Sequence[str] | None = None, stream_color: str = "0.75",
                   xlabel: str = "$a$", ylabel: str = "$b$", start_markers: bool = True):
    """Streamlines of a 2-D autonomous field plus given trajectories.

    vector_field(Y) maps an array (..., 2) to (..., 2) (vectorized, like tdyn.models.anchor_flow).
    trajectories: list of (n, 2) arrays, drawn on top; traj_colors one colour per trajectory.
    Returns the axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.2))
    xs = np.linspace(*xlim, 241)
    ys = np.linspace(*ylim, 241)
    X, Y = np.meshgrid(xs, ys)
    V = vector_field(np.stack([X, Y], axis=-1))
    ax.streamplot(X, Y, V[..., 0], V[..., 1], color=stream_color, density=density,
                  linewidth=0.6, arrowsize=0.7, zorder=1)
    for i, tr in enumerate(trajectories):
        c = traj_colors[i] if traj_colors is not None else PALETTE[i % len(PALETTE)]
        ax.plot(tr[:, 0], tr[:, 1], color=c, lw=0.9, zorder=3)
        if start_markers:
            ax.plot(tr[0, 0], tr[0, 1], "o", ms=2.2, color=c, zorder=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax
