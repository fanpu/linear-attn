"""Shared render helpers for Combed M2 (reads cache/ only; draws with art/_shared/r3d)."""
from __future__ import annotations

import json
import sys

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/fzeng/ml/research/art/_shared")
import r3d  # noqa: E402

import combed_common as C  # noqa: E402

GALLERY = C.ROOT / "gallery"
T_END_LABEL = "t = 1 - 1e-6"
STACK = "GB10, driver 580.173.02, CUDA 13.0, torch 2.14.0+cu130"
TRAIN_DECL = "MLP 4x256 SiLU, 20k Adam steps, batch 1024 (fixed for every N)"

# Declared chart: identity map of R^3 (data coordinates), orthographic camera.
VIEW = dict(target=(0.0, 0.0, 0.0), radius=12.0, az_deg=-60.0, el_deg=40.0)


def camera(size: int, ortho_height: float = 4.4, **kw) -> r3d.Camera:
    v = dict(VIEW); v.update(kw)
    return r3d.orbit(v["target"], v["radius"], az_deg=v["az_deg"], el_deg=v["el_deg"], width=size, height=size,
                     ortho_height=ortho_height)


def load_dense(kind: str, n: int):
    z = np.load(C.CACHE / f"dense_{kind}_N{n}.npz")
    return z


def memo_numbers(kind: str, n: int):
    """Memorised fraction at t = 1 - 1e-6 (the render stop) and the fresh-knot null for this N."""
    z = load_dense(kind, n)
    mem = float((z["end_d1"] < C.MEM_RATIO * z["end_d2"]).mean())
    rows = json.loads((C.CACHE / "summary.json").read_text())
    null = [r["mem_xhat256"] for r in rows if r["N"] == n and r["field"] == "null_fresh_knot"][0]
    return mem, null


def densify(states: torch.Tensor, t: torch.Tensor, spacing: float):
    """Linear resampling of polylines (T, B, 3) at world spacing <= `spacing`. Returns points (M,3), t (M,), traj id (M,).
    Vertices are RK4 states (257 uniform + 12 tail), so linear pieces are <= 1/256 in t."""
    a, b = states[:-1], states[1:]
    T1, B, _ = a.shape
    L = (b - a).norm(dim=-1)
    k = torch.ceil(L / spacing).clamp_min(1).long().reshape(-1)
    a, d = a.reshape(-1, 3), (b - a).reshape(-1, 3)
    t0 = t[:-1, None].expand(T1, B).reshape(-1)
    dt = (t[1:] - t[:-1])[:, None].expand(T1, B).reshape(-1)
    tid = torch.arange(B, device=states.device)[None].expand(T1, B).reshape(-1)
    S = torch.arange(k.numel(), device=states.device)
    rep = torch.repeat_interleave(S, k)
    start = torch.cumsum(k, 0) - k
    u = (torch.arange(rep.numel(), device=states.device) - start[rep]).to(states.dtype) / k[rep].to(states.dtype)
    pts = a[rep] + u[:, None] * d[rep]
    return pts, t0[rep] + u * dt[rep], tid[rep]


def cmap_lut(name: str, x: torch.Tensor) -> torch.Tensor:
    if name.startswith("cet_"):
        import colorcet
        cm = colorcet.cm[name[4:]]
    else:
        cm = matplotlib.colormaps[name]
    lut = torch.tensor(cm(np.linspace(0, 1, 1024))[:, :3], dtype=x.dtype, device=x.device)
    return lut[(x.clamp(0, 1) * 1023).round().long()]


def to_u8(img: torch.Tensor) -> np.ndarray:
    return (img.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


MIN_CAPTION_FRAC = 0.016  # controller M3 ruling: caption text >= 1.4% of final image height; 1.6% is the safety margin used here


def _wrap(lines, W, fs):
    f = font(fs)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    wrapped = []
    for line in lines:  # word-wrap to the image width
        cur = ""
        for word in line.split(" "):
            trial = (cur + " " + word) if cur else word
            if probe.textlength(trial, font=f) > W - 1.6 * fs and cur:
                wrapped.append(cur); cur = "  " + word
            else:
                cur = trial
        wrapped.append(cur)
    return wrapped, f


def caption_strip(img: np.ndarray, lines: list[str], bg=(8, 8, 10), fg=(215, 210, 200), scale: float = 1.0,
                   min_frac: float = MIN_CAPTION_FRAC) -> np.ndarray:
    """Appends a word-wrapped caption strip below `img`. Font size is the larger of the `scale`-derived size and a
    floor of `min_frac` of the FINAL (image + strip) height, solved by fixed point since the strip height depends on
    the font size in turn (M3 ruling: captions >= 1.4% of image height; default floor here is 1.6%, a safety margin)."""
    H, W = img.shape[0], img.shape[1]
    fs = max(12, int(18 * scale * W / 2048))
    for _ in range(4):
        wrapped, _ = _wrap(lines, W, fs)
        strip_h = int(len(wrapped) * fs * 1.45 + fs)
        need = int(np.ceil(min_frac * (H + strip_h)))
        if need <= fs:
            break
        fs = need
    wrapped, f = _wrap(lines, W, fs)
    h = int(len(wrapped) * fs * 1.45 + fs)
    strip = Image.new("RGB", (W, h), bg)
    d = ImageDraw.Draw(strip)
    for i, line in enumerate(wrapped):
        d.text((int(fs * 0.8), int(fs * 0.5 + i * fs * 1.45)), line, fill=fg, font=f)
    return np.concatenate([img, np.asarray(strip)], 0)


def save(path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, optimize=True)
    print("wrote", path, arr.shape, flush=True)
