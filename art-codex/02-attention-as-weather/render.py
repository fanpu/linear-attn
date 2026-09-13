"""Train a small character transformer and render its real attention as weather."""

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
import torch.nn.functional as F


ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 31
CONTEXT = 24

TEXT = (
    "attention is a moving field of relevance. a token listens backward through time, "
    "finding echoes, repetitions, and distant structure. learning gathers the scattered "
    "signal into patterns. a model attends where context becomes useful. "
) * 80


class WeatherAttention(nn.Module):
    def __init__(self, vocab: int, width: int = 48) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.pos = nn.Parameter(torch.zeros(1, CONTEXT, width))
        self.qkv = nn.Linear(width, width * 3)
        self.project = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, vocab))
        self.width = width

    def forward(self, tokens: torch.Tensor, return_attention: bool = False):
        x = self.embed(tokens) + self.pos[:, : tokens.shape[1]]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = q @ k.transpose(-2, -1) / self.width**0.5
        causal = torch.triu(torch.ones_like(scores[0], dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attention = scores.softmax(-1)
        x = self.norm(x + self.project(attention @ v))
        logits = self.head(x)
        return (logits, attention) if return_attention else logits


def render() -> None:
    torch.manual_seed(SEED)
    OUT.mkdir(exist_ok=True)
    alphabet = sorted(set(TEXT))
    to_id = {char: i for i, char in enumerate(alphabet)}
    values = torch.tensor([to_id[c] for c in TEXT], dtype=torch.long)
    model = WeatherAttention(len(alphabet)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.3e-3, weight_decay=0.02)
    checkpoint_steps = set(np.linspace(0, 1499, 36, dtype=int).tolist())
    probe = "attention finds echoes "
    probe_ids = torch.tensor([to_id[c] for c in probe], dtype=torch.long)[None].to(DEVICE)
    snapshots: list[tuple[int, float, np.ndarray]] = []

    for step in range(1500):
        starts = torch.randint(0, len(values) - CONTEXT - 1, (80,))
        batch = torch.stack([values[s : s + CONTEXT] for s in starts]).to(DEVICE)
        target = torch.stack([values[s + 1 : s + CONTEXT + 1] for s in starts]).to(DEVICE)
        loss = F.cross_entropy(model(batch).flatten(0, 1), target.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in checkpoint_steps:
            with torch.no_grad():
                _, attention = model(probe_ids, return_attention=True)
            snapshots.append((step + 1, float(loss.detach().cpu()), attention[0].cpu().numpy()))

    colors = LinearSegmentedColormap.from_list("aurora", ["#07172d", "#124b78", "#00aab4", "#f1e6a2", "#fb9268"])
    frames: list[Image.Image] = []
    labels = ["space" if char == " " else char for char in probe]
    for index, (step, loss, matrix) in enumerate(snapshots):
        fig = plt.figure(figsize=(11.4, 7.1), facecolor="#06121e")
        main = fig.add_axes((0.13, 0.15, 0.76, 0.68), facecolor="#06121e")
        image = main.imshow(matrix, cmap=colors, vmin=0, vmax=max(0.2, matrix.max()), interpolation="bilinear", aspect="auto")
        main.set_xticks(range(len(labels)), labels, fontsize=9, color="#dbe8e5")
        main.set_yticks(range(len(labels)), labels, fontsize=9, color="#dbe8e5")
        main.set_xlabel("keys: where a character looks", color="#b7cbc7", labelpad=12)
        main.set_ylabel("queries: the character doing the looking", color="#b7cbc7", labelpad=12)
        for spine in main.spines.values():
            spine.set_color("#385367")
        cbar = fig.colorbar(image, ax=main, fraction=0.032, pad=0.025)
        cbar.ax.tick_params(colors="#b7cbc7", labelsize=8)
        cbar.outline.set_edgecolor("#385367")
        cbar.set_label("attention weight", color="#b7cbc7", fontsize=9)
        fig.text(0.13, 0.93, "ATTENTION AS WEATHER", color="#f2f0d8", fontsize=23, weight="bold")
        fig.text(0.13, 0.887, "a one-layer character transformer learns which earlier characters matter", color="#aec7c4", fontsize=10)
        fig.text(0.13, 0.075, f"training step {step:04d}    cross-entropy {loss:.3f}    probe: \"{probe}\"", color="#aec7c4", fontsize=10)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
        if index == len(snapshots) - 1:
            frames[-1].save(OUT / "attention-weather-final.png", quality=95)
        plt.close(fig)
    frames[0].save(OUT / "attention-weather.gif", save_all=True, append_images=frames[1:], duration=105, loop=0, optimize=False)
    np.savez_compressed(OUT / "attention-data.npz", attention=np.array([x[2] for x in snapshots]), steps=np.array([x[0] for x in snapshots]))
    (OUT / "metadata.json").write_text(json.dumps({"device": str(DEVICE), "seed": SEED, "training_steps": 1500, "probe": probe}, indent=2) + "\n")


if __name__ == "__main__":
    render()
