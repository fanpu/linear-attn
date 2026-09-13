"""Production render: learned multi-head attention as a data-driven weather system."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED, CONTEXT, STEPS, FRAMES = 113, 64, 3400, 32
CORPUS = (
    "the model gathers distant echoes. attention is weather crossing a field of language. "
    "each mark listens backward, finding rhythm, return, and structure in the moving air. "
    "a learned path is not a line; it is a changing relation between what was and what comes next. "
) * 120
HEAD_COLORS = np.array([[61, 194, 184], [243, 184, 93], [231, 104, 103], [115, 157, 231]], dtype=np.uint8)


class WeatherTransformer(nn.Module):
    def __init__(self, vocab: int, width: int = 128, heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.pos = nn.Parameter(torch.randn(1, CONTEXT, width) * 0.01)
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 3), nn.GELU(), nn.Linear(width * 3, width))
        self.head = nn.Linear(width, vocab)
        self.heads, self.head_size = heads, width // heads

    def forward(self, ids: torch.Tensor, return_attention: bool = False):
        x = self.embed(ids) + self.pos[:, : ids.shape[1]]
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_size).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_size).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_size).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / self.head_size**0.5
        mask = torch.triu(torch.ones(t, t, device=ids.device, dtype=torch.bool), diagonal=1)
        attention = scores.masked_fill(mask, float("-inf")).softmax(-1)
        mixed = (attention @ v).transpose(1, 2).reshape(b, t, c)
        x = x + self.out(mixed)
        x = x + self.mlp(self.norm2(x))
        logits = self.head(x)
        return (logits, attention) if return_attention else logits


def cubic(points: np.ndarray, p0, p1, p2, p3):
    t = points[:, None]
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3


def train():
    torch.manual_seed(SEED)
    alphabet = sorted(set(CORPUS)); lookup = {char: index for index, char in enumerate(alphabet)}
    encoded = torch.tensor([lookup[c] for c in CORPUS], dtype=torch.long)
    # Deliberately chosen from the corpus, not handwritten after training.
    probe_text = "attention is weather crossing a field of language. each mark listens "
    probe_text = probe_text[:CONTEXT]
    probe = torch.tensor([lookup[c] for c in probe_text], dtype=torch.long)[None].to(DEVICE)
    model = WeatherTransformer(len(alphabet)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=.04)
    capture = set(np.linspace(0, STEPS - 1, FRAMES, dtype=int).tolist())
    shots, losses = [], []
    for step in range(STEPS):
        starts = torch.randint(0, len(encoded) - CONTEXT - 1, (72,))
        source = torch.stack([encoded[s:s+CONTEXT] for s in starts]).to(DEVICE)
        target = torch.stack([encoded[s+1:s+CONTEXT+1] for s in starts]).to(DEVICE)
        loss = F.cross_entropy(model(source).reshape(-1, len(alphabet)), target.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(float(loss.detach().cpu()))
        if step in capture:
            with torch.no_grad(): _, attn = model(probe, return_attention=True)
            shots.append(attn[0].cpu().numpy())
    return probe_text, np.asarray(shots), np.asarray(losses)


def positions(count: int, width: int, height: int):
    u = np.linspace(0, 1, count)
    x = width * (.075 + .85*u)
    y = height * (.50 + .18*np.sin(u*2.2*np.pi + .45) + .065*np.sin(u*6.4*np.pi))
    return np.c_[x,y]


def compose(attn: np.ndarray, tokens: str, frame: int, total: int, size: tuple[int,int]):
    width, height = size; scale = width / 1280; pos = positions(len(tokens), width, height)
    rng = np.random.default_rng(SEED + frame)
    base = Image.new("RGB", size, "#061722")
    pix = np.array(base)
    # Atmospheric vertical gradient. Its intensity follows total attention concentration.
    incoming = attn.sum(axis=(0, 1))
    concentration = float(np.std(incoming) / (incoming.mean()+1e-8))
    y = np.linspace(0, 1, height)[:,None]
    pix[...,0] = 5 + 8*(1-y)
    pix[...,1] = 20 + 23*(1-y) + 5*concentration
    pix[...,2] = 31 + 22*(1-y)
    base = Image.fromarray(pix.astype(np.uint8), "RGB")
    glow = Image.new("RGBA", size, (0,0,0,0)); gd = ImageDraw.Draw(glow)
    # Attention received at each position produces literal pressure centers.
    for key, strength in enumerate(incoming):
        radius = (12 + 55 * strength / incoming.max()) * scale
        x,y0 = pos[key]
        for r, alpha in ((radius*2.4, 4), (radius*1.35, 10), (radius*.65, 23)):
            gd.ellipse((x-r, y0-r, x+r, y0+r), fill=(45, 196, 184, alpha))
    base = Image.alpha_composite(base.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(14*scale)))
    paths = Image.new("RGBA", size, (0,0,0,0)); pd = ImageDraw.Draw(paths)
    # Only the strongest causal relations become visible filaments; their width/opacity encode attention.
    turns = np.linspace(0, 1, 22)
    for head in range(attn.shape[0]):
        color = tuple(int(v) for v in HEAD_COLORS[head])
        for query in range(4, len(tokens)):
            weights = attn[head, query, :query+1]
            best = np.argpartition(weights, -3)[-3:]
            for key in best:
                strength = float(weights[key])
                if strength < .035: continue
                p0, p3 = pos[key], pos[query]
                distance = abs(query-key) / len(tokens)
                lift = (head - 1.5) * 48 * scale + (1 if (query+key)%2 else -1) * distance * 260 * scale
                p1 = p0 + np.array([(p3[0]-p0[0])*.30, lift])
                p2 = p0 + np.array([(p3[0]-p0[0])*.72, lift])
                curve = cubic(turns, p0, p1, p2, p3)
                alpha = int(min(150, 25 + 310*strength))
                pd.line([tuple(p) for p in curve], fill=(*color, alpha), width=max(1, int((.4+5*strength)*scale)), joint="curve")
    # Long-exposure afterimage: a previous state lingers under the current field.
    base = Image.alpha_composite(base, paths.filter(ImageFilter.GaussianBlur(.7*scale)))
    nodes = Image.new("RGBA", size, (0,0,0,0)); nd = ImageDraw.Draw(nodes)
    for index, (x,y0) in enumerate(pos):
        importance = incoming[index] / incoming.max()
        radius = (1.4 + 6*importance) * scale
        char = tokens[index]
        token_color = (242,222,163, 210) if char == " " else (202,238,225, int(90+150*importance))
        nd.ellipse((x-radius,y0-radius,x+radius,y0+radius), fill=token_color)
    base = Image.alpha_composite(base, nodes)
    # A quiet temporal progress indicator that is also the data state: captured training position.
    mark = Image.new("RGBA", size, (0,0,0,0)); md = ImageDraw.Draw(mark)
    progress = (frame+1)/total
    md.line((width*.075, height*.915, width*.925, height*.915), fill=(107,151,154,38), width=max(1,int(scale)))
    md.line((width*.075, height*.915, width*(.075+.85*progress), height*.915), fill=(239,208,126,170), width=max(2,int(2*scale)))
    return base.convert("RGB")


def main():
    OUT.mkdir(exist_ok=True)
    tokens, snapshots, losses = train()
    loop = [compose(attn, tokens, i, len(snapshots), (1280,720)) for i, attn in enumerate(snapshots)]
    master = compose(snapshots[-1], tokens, len(snapshots)-1, len(snapshots), (2560,1440))
    master.save(OUT / "master.png", quality=96)
    loop[0].save(OUT / "loop-production.gif", save_all=True, append_images=loop[1:], duration=100, loop=0, optimize=False)
    np.savez_compressed(OUT / "production-data.npz", attention=snapshots, losses=losses)
    metadata = {"device":str(DEVICE), "seed":SEED, "training_steps":STEPS, "heads":4, "context":CONTEXT, "master_pixels":[2560,1440], "visual_mapping":{"filaments":"three strongest causal attention links per query/head", "filament_color":"attention head", "filament_opacity_and_width":"attention weight", "glow":"attention received by token", "node_size":"attention received by token", "progress_line":"training checkpoint"}}
    (OUT / "production-metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")

if __name__ == "__main__": main()
