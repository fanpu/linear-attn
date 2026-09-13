"""Render a real optimization landscape and its changing classifier boundary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
import torch
from torch import nn


ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 19
STEPS = 600
FRAMES = 48


class SpiralNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 40), nn.Tanh(), nn.Linear(40, 40), nn.Tanh(), nn.Linear(40, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def make_spirals(n: int = 900) -> tuple[torch.Tensor, torch.Tensor]:
    """Two interlocked noisy spirals, a small but genuinely nonlinear task."""
    gen = torch.Generator().manual_seed(SEED)
    radius = torch.sqrt(torch.rand(n, generator=gen)) * 1.15
    angle = 2.0 * torch.pi * radius + torch.randn(n, generator=gen) * 0.16
    a = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1)
    b = -a + torch.randn(n, 2, generator=gen) * 0.045
    x = torch.cat((a, b)).float()
    y = torch.cat((torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)))
    return x, y


def flattened(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in model.parameters()])


def assign_flat(model: nn.Module, values: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(values[offset : offset + count].view_as(parameter).to(DEVICE))
            offset += count


def unit_direction(reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    direction = torch.randn(reference.numel(), generator=generator)
    # Filter-wise normalization makes a useful local slice instead of one layer dominating.
    return direction / direction.norm()


def train() -> tuple[SpiralNet, torch.Tensor, torch.Tensor, list[dict[str, object]], np.ndarray]:
    torch.manual_seed(SEED)
    x, y = make_spirals()
    x, y = x.to(DEVICE), y.to(DEVICE)
    model = SpiralNet().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.22, momentum=0.86)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, object]] = []
    losses = []
    frame_steps = set(np.linspace(0, STEPS - 1, FRAMES, dtype=int).tolist())

    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step in frame_steps:
            history.append({"step": step + 1, "loss": losses[-1], "state": flattened(model).numpy()})
    return model, x, y, history, np.asarray(losses)


def loss_surface(model: SpiralNet, x: torch.Tensor, y: torch.Tensor, center: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measure loss rather than invent a landscape: evaluate a two-direction weight plane."""
    current = torch.from_numpy(center)
    gen = torch.Generator().manual_seed(7)
    direction_a = unit_direction(current, gen)
    direction_b = unit_direction(current, gen)
    direction_b = direction_b - direction_a * torch.dot(direction_a, direction_b)
    direction_b = direction_b / direction_b.norm()
    grid = np.linspace(-1.65, 1.65, 61)
    terrain = np.empty((len(grid), len(grid)))
    criterion = nn.CrossEntropyLoss()
    for iy, vertical in enumerate(grid):
        for ix, horizontal in enumerate(grid):
            assign_flat(model, current + horizontal * direction_a + vertical * direction_b)
            with torch.no_grad():
                terrain[iy, ix] = criterion(model(x), y).item()
    assign_flat(model, current)
    return grid, terrain, direction_a.numpy(), direction_b.numpy(), current.numpy()


def field(model: SpiralNet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.mgrid[-1.38:1.38:180j, -1.38:1.38:180j]
    points = np.c_[xy[0].ravel(), xy[1].ravel()]
    with torch.no_grad():
        logits = model(torch.from_numpy(points).float().to(DEVICE))
        probability = logits.softmax(1)[:, 1].cpu().numpy().reshape(xy[0].shape)
    return xy[0], xy[1], probability


def render() -> None:
    OUT.mkdir(exist_ok=True)
    model, x, y, history, losses = train()
    center = history[-1]["state"]
    assert isinstance(center, np.ndarray)
    grid, terrain, dx, dy, origin = loss_surface(model, x, y, center)
    x_cpu, y_cpu = x.cpu().numpy(), y.cpu().numpy()
    mint_rose = LinearSegmentedColormap.from_list("mint_rose", ["#09202c", "#2a7495", "#d8f0c3", "#f8c7a7", "#b63c58"])
    space = np.linspace(-1.35, 1.35, 160)
    xx, yy = np.meshgrid(space, space)
    frames: list[Image.Image] = []

    for index, entry in enumerate(history):
        state = entry["state"]
        assert isinstance(state, np.ndarray)
        assign_flat(model, torch.from_numpy(state))
        fx, fy, prob = field(model)
        delta = torch.from_numpy(state) - torch.from_numpy(origin)
        px = float(torch.dot(delta, torch.from_numpy(dx)))
        py = float(torch.dot(delta, torch.from_numpy(dy)))

        fig = plt.figure(figsize=(12, 6.8), facecolor="#07161e")
        left = fig.add_axes((0.06, 0.12, 0.42, 0.78), facecolor="#07161e")
        right = fig.add_axes((0.55, 0.12, 0.39, 0.78), facecolor="#07161e")
        left.contourf(fx, fy, prob, levels=26, cmap=mint_rose, vmin=0, vmax=1)
        left.contour(fx, fy, prob, levels=[0.5], colors="#fff6df", linewidths=1.1, alpha=0.9)
        left.scatter(x_cpu[y_cpu == 0, 0], x_cpu[y_cpu == 0, 1], s=6, c="#bdf0e4", alpha=0.6, linewidths=0)
        left.scatter(x_cpu[y_cpu == 1, 0], x_cpu[y_cpu == 1, 1], s=6, c="#ffbe94", alpha=0.6, linewidths=0)
        left.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), aspect="equal", xticks=[], yticks=[])
        left.set_title("DECISION BOUNDARY", color="#e8f2e9", loc="left", fontsize=12, pad=12)

        clipped = np.log1p(terrain)
        right.contourf(grid, grid, clipped, levels=32, cmap="magma_r")
        right.contour(grid, grid, clipped, colors="#ffd9a6", levels=12, linewidths=0.35, alpha=0.55)
        route_x, route_y = [], []
        for earlier in history[: index + 1]:
            dv = torch.from_numpy(earlier["state"]) - torch.from_numpy(origin)
            route_x.append(float(torch.dot(dv, torch.from_numpy(dx))))
            route_y.append(float(torch.dot(dv, torch.from_numpy(dy))))
        right.plot(route_x, route_y, color="#71e4c6", linewidth=1.6, alpha=0.85)
        right.scatter(route_x[-1], route_y[-1], s=55, c="#f8f0b6", edgecolors="#1d2532", linewidths=1.1, zorder=3)
        right.set(xlim=(grid.min(), grid.max()), ylim=(grid.min(), grid.max()), aspect="equal", xticks=[], yticks=[])
        right.set_title("MEASURED LOSS SLICE", color="#e8f2e9", loc="left", fontsize=12, pad=12)
        for ax in (left, right):
            for spine in ax.spines.values():
                spine.set_color("#4e6870")
        fig.text(0.06, 0.955, "THE DESCENT GARDEN", color="#f6f0dc", fontsize=22, weight="bold")
        fig.text(0.06, 0.915, f"spiral classifier  /  SGD + momentum  /  step {entry['step']:03d}  /  loss {entry['loss']:.4f}", color="#b8c9c0", fontsize=10)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba).convert("RGB"))
        if index == len(history) - 1:
            frames[-1].save(OUT / "descent-garden-final.png", quality=95)
        plt.close(fig)

    frames[0].save(OUT / "descent-garden.gif", save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=False)
    np.savez_compressed(OUT / "experiment-data.npz", losses=losses, terrain=terrain, trajectory=np.array([h["state"] for h in history]))
    metadata = {"device": str(DEVICE), "seed": SEED, "steps": STEPS, "frames": FRAMES, "final_loss": float(losses[-1])}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    render()
