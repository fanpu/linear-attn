"""Production render for The Descent Garden: data-rich, chart-free ML landscape."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED, STEPS, FRAMES = 71, 1400, 32


class GardenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 2))
    def forward(self, x): return self.net(x)


def spirals(n=1800):
    g = torch.Generator().manual_seed(SEED)
    r = torch.sqrt(torch.rand(n, generator=g)) * 1.28
    a = 2.35 * torch.pi * r + torch.randn(n, generator=g) * .105
    first = torch.stack((r * torch.cos(a), r * torch.sin(a)), 1)
    second = -first + .042 * torch.randn(n, 2, generator=g)
    return torch.cat((first, second)).float(), torch.cat((torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)))


def palette(values):
    stops = np.array([[5,17,27], [7,59,73], [17,133,110], [223,205,122], [237,84,75]], dtype=float) / 255
    at = np.clip(values, 0, 1) * (len(stops)-1)
    lo = np.floor(at).astype(int); hi = np.clip(lo+1, 0, len(stops)-1)
    return stops[lo] * (1-(at-lo))[...,None] + stops[hi] * (at-lo)[...,None]


def evaluate(model, side=560):
    axis = np.linspace(-1.52, 1.52, side, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis)
    points = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().to(DEVICE).requires_grad_(True)
    logits = model(points)
    prob = logits.softmax(1)[:, 1]
    grad = torch.autograd.grad(prob.sum(), points)[0]
    p = prob.detach().cpu().numpy().reshape(side, side)
    vector = grad.detach().cpu().numpy().reshape(side, side, 2)
    entropy = -(p*np.log(p+1e-7)+(1-p)*np.log(1-p+1e-7))/np.log(2)
    return axis, p, entropy, vector


def state(model): return torch.cat([p.detach().flatten().cpu() for p in model.parameters()])


def train():
    torch.manual_seed(SEED)
    x, y = spirals(); x, y = x.to(DEVICE), y.to(DEVICE)
    model = GardenNet().to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=2e-4)
    shots, losses, trail = [], [], []
    capture = set(np.linspace(0, STEPS-1, FRAMES, dtype=int).tolist())
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y); loss.backward(); opt.step()
        losses.append(float(loss.detach().cpu()))
        if step in capture:
            axis, p, entropy, vector = evaluate(model, 380)
            shots.append((axis, p, entropy, vector)); trail.append(state(model).numpy())
    return model, x.cpu().numpy(), y.cpu().numpy(), shots, np.asarray(losses), np.asarray(trail)


def draw(axis, probability, entropy, vector, x, y, frame, trail, master=False):
    size = (16, 9) if master else (12.8, 7.2)
    dpi = 160 if master else 100
    fig = plt.figure(figsize=size, dpi=dpi, facecolor="#05111a")
    ax = fig.add_axes((0,0,1,1), facecolor="#05111a")
    # Saturation follows certainty; cool and warm sides are the model's class probability.
    rgb = palette(probability)
    rgb = rgb * (0.34 + .66*(1-entropy[...,None])) + np.array([.025,.045,.06]) * entropy[...,None]
    ax.imshow(rgb, extent=[axis.min(), axis.max(), axis.min(), axis.max()], origin="lower", interpolation="bicubic")
    X, Y = np.meshgrid(axis, axis)
    # Fine topographic structure: probability isolines are actual decision-level sets.
    ax.contour(X, Y, probability, levels=np.linspace(.08,.92,14), colors="#e8dbb8", linewidths=.20 if master else .28, alpha=.34)
    ax.contour(X, Y, probability, levels=[.5], colors="#fff0c4", linewidths=1.35 if master else 1.1, alpha=.93)
    # Input-gradient streamlines are roots of the learned decision field.
    speed = np.linalg.norm(vector, axis=-1)
    starts = np.c_[np.linspace(-1.32,1.32,25), np.full(25,-1.32)]
    ax.streamplot(axis, axis, vector[...,0].T, vector[...,1].T, color=np.sqrt(speed.T), cmap="cividis", density=1.25, linewidth=.25, arrowsize=.1, start_points=starts, integration_direction="both", maxlength=2.2, broken_streamlines=True, zorder=2)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(x), 1500, replace=False)
    colors = np.where(y[idx,None] == 0, np.array([[144,243,208]])/255, np.array([[255,168,120]])/255)
    ax.scatter(x[idx,0], x[idx,1], s=1.3 if master else 1.8, c=colors, alpha=.44, linewidths=0, zorder=4)
    # A genuine trajectory in a two-dimensional PCA projection of parameter space.
    centered = trail - trail.mean(0); _, _, vt = np.linalg.svd(centered, full_matrices=False)
    route = centered @ vt[:2].T; route = (route-route.min(0))/(np.ptp(route, axis=0)+1e-9)
    route = route * np.array([.62,.20]) + np.array([.06,.07])
    progress = min(frame+1, len(route))
    ax.plot(route[:progress,0]*3.04-1.52, route[:progress,1]*3.04-1.52, color="#fff4cf", alpha=.68, linewidth=.75, zorder=6)
    ax.scatter(route[progress-1,0]*3.04-1.52, route[progress-1,1]*3.04-1.52, s=24, c="#ffdf89", edgecolors="none", zorder=7)
    ax.set(xlim=(-1.52,1.52), ylim=(-1.52,1.52), aspect="equal", xticks=[], yticks=[])
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.canvas.draw(); image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"); plt.close(fig)
    return image


def main():
    OUT.mkdir(exist_ok=True)
    model, x, y, shots, losses, trail = train()
    frames = [draw(*shot, x, y, i, trail) for i, shot in enumerate(shots)]
    master = draw(*shots[-1], x, y, len(shots)-1, trail, master=True)
    master.save(OUT / "master.png", quality=96)
    frames[0].save(OUT / "loop-production.gif", save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=False)
    np.savez_compressed(OUT / "production-data.npz", losses=losses, probability=shots[-1][1], uncertainty=shots[-1][2], gradient=shots[-1][3], parameter_trajectory=trail)
    (OUT / "production-metadata.json").write_text(json.dumps({"device":str(DEVICE), "seed":SEED, "training_steps":STEPS, "training_examples":len(x), "frames":FRAMES, "master_pixels":[2560,1440], "visual_mapping":{"hue":"class probability","brightness":"certainty","contours":"probability level sets","streamlines":"input-probability gradient","points":"training samples","route":"PCA projection of weight trajectory"}}, indent=2)+"\n")

if __name__ == "__main__": main()
