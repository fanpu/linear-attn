"""Learn a 2D diffusion score model and render the reverse denoising process."""

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


ROOT, OUT = Path(__file__).parent, Path(__file__).parent / "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED, T = 43, 56


def moons(n: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(SEED)
    angle = torch.rand(n, generator=gen) * torch.pi
    top = torch.stack((torch.cos(angle), torch.sin(angle)), 1)
    bottom = torch.stack((1 - torch.cos(angle), 0.45 - torch.sin(angle)), 1)
    x = torch.cat((top, bottom)) + 0.055 * torch.randn(2 * n, 2, generator=gen)
    return (x - x.mean(0)) / x.std(0)


class ScoreNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 2))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((x, t[:, None]), 1))


def render() -> None:
    torch.manual_seed(SEED)
    OUT.mkdir(exist_ok=True)
    data = moons(1500).to(DEVICE)
    beta = torch.linspace(0.0005, 0.075, T, device=DEVICE)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, 0)
    model = ScoreNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    losses = []
    for step in range(2600):
        choice = torch.randint(len(data), (320,), device=DEVICE)
        t = torch.randint(0, T, (320,), device=DEVICE)
        noise = torch.randn(320, 2, device=DEVICE)
        a_bar = alpha_bar[t, None]
        noisy = a_bar.sqrt() * data[choice] + (1 - a_bar).sqrt() * noise
        predicted = model(noisy, t.float() / (T - 1))
        loss = (predicted - noise).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    with torch.no_grad():
        sample = torch.randn(1450, 2, device=DEVICE)
        states = [sample.cpu().numpy()]
        for t_value in range(T - 1, -1, -1):
            times = torch.full((len(sample),), t_value / (T - 1), device=DEVICE)
            eps = model(sample, times)
            a, ab, b = alpha[t_value], alpha_bar[t_value], beta[t_value]
            sample = (sample - b / (1 - ab).sqrt() * eps) / a.sqrt()
            if t_value > 0:
                sample = sample + b.sqrt() * torch.randn_like(sample)
            states.append(sample.cpu().numpy())

    cmap = LinearSegmentedColormap.from_list("excavation", ["#10152c", "#255c8b", "#4fb6a7", "#f0d98a", "#ee806c"])
    frames: list[Image.Image] = []
    chosen = np.linspace(0, len(states) - 1, 38, dtype=int)
    for frame_no, state_id in enumerate(chosen):
        points = states[state_id]
        progress = state_id / (len(states) - 1)
        fig = plt.figure(figsize=(10.6, 7), facecolor="#0a1021")
        ax = fig.add_axes((0.1, 0.14, 0.8, 0.72), facecolor="#0a1021")
        value = np.linalg.norm(points, axis=1)
        ax.scatter(points[:, 0], points[:, 1], c=value, cmap=cmap, s=9, alpha=0.58, linewidths=0)
        reference = data[::6].cpu().numpy()
        ax.scatter(reference[:, 0], reference[:, 1], s=4, c="#dce3e3", alpha=0.12, linewidths=0)
        ax.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), aspect="equal", xticks=[], yticks=[])
        for spine in ax.spines.values(): spine.set_color("#3a5265")
        reverse_t = T - 1 - int(progress * T)
        fig.text(0.1, 0.925, "DIFFUSION ARCHAEOLOGY", color="#f5ecd4", fontsize=23, weight="bold")
        fig.text(0.1, 0.885, "a learned score model excavates structure from a Gaussian field", color="#acc7c8", fontsize=10)
        fig.text(0.1, 0.07, f"reverse diffusion step {max(reverse_t, 0):02d} / {T - 1}     training noise-prediction loss {losses[-1]:.4f}", color="#acc7c8", fontsize=10)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
        if frame_no == len(chosen) - 1: frames[-1].save(OUT / "diffusion-archaeology-final.png", quality=95)
        plt.close(fig)
    frames[0].save(OUT / "diffusion-archaeology.gif", save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=False)
    np.savez_compressed(OUT / "diffusion-data.npz", final_samples=states[-1], trajectory=np.asarray(states)[chosen], losses=np.asarray(losses))
    (OUT / "metadata.json").write_text(json.dumps({"device": str(DEVICE), "seed": SEED, "steps": 2600, "diffusion_steps": T, "final_loss": losses[-1]}, indent=2) + "\n")


if __name__ == "__main__":
    render()
