"""Hero animation: the same two-layer ReLU net trained lazily (left) and richly (right) on three concentric rings.

Every hidden neuron j is drawn as its "kink line" {x : w_j.x + b_j = 0}, the line where its ReLU switches on,
with brightness proportional to |a_j| |w_j| (how steep a ramp it adds to the output) and colour by the sign of
a_j (amber: pushes toward the amber class, cyan: toward cyan). The white curve is the decision boundary.

    .venv/bin/python 04-lazy-rich-mup/render_hero.py            # -> figures/hero.mp4 + figures/hero_still.png
    .venv/bin/python 04-lazy-rich-mup/render_hero.py --test     # a few frames into _preview/
"""
import os, subprocess, sys

import numpy as np
from contourpy import contour_generator
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, zoom

import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf"
NIGHT = np.array([7, 9, 18]) / 255
AMBER = np.array([255, 180, 84]) / 255
CYAN = np.array([88, 196, 245]) / 255


def font(size, weight="Regular"):
    f = ImageFont.truetype(FONT, size)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


def line_density(w, b, a, R, res, samples=1400):
    """Accumulate the kink lines of all neurons into two density images (a>0, a<0), bilinear splatting."""
    nw = np.linalg.norm(w, axis=1) + 1e-12
    u = w / nw[:, None]
    c = -b / nw                              # signed distance of the line from the origin
    wt = np.abs(a) * nw
    wt = wt / wt.mean()
    s = np.linspace(-1.9 * R, 1.9 * R, samples)
    out = np.zeros((2, res, res))
    for k, sel in enumerate([a > 0, a <= 0]):
        if not sel.any():
            continue
        perp = np.stack([-u[sel, 1], u[sel, 0]], 1)
        P = c[sel, None, None] * u[sel, None, :] + s[None, :, None] * perp[:, None, :]
        q = (P.reshape(-1, 2) + R) / (2 * R) * res - 0.5
        W = np.repeat(wt[sel], samples)
        i0 = np.floor(q).astype(int); f = q - i0
        for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            ii, jj = i0[:, 0] + dx, i0[:, 1] + dy
            ok = (ii >= 0) & (ii < res) & (jj >= 0) & (jj < res)
            ww = W * (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
            out[k] += np.bincount(jj[ok] * res + ii[ok], ww[ok], res * res).reshape(res, res)
    return out * (res / samples) / 3.8  # density per unit length, roughly resolution independent


_VIG = {}


def vignette(res, R):
    if res not in _VIG:
        g = (np.arange(res) + 0.5) / res * 2 * R - R
        rr = np.sqrt(g[None, :] ** 2 + g[:, None] ** 2)
        _VIG[res] = np.clip((1.22 - rr) / 0.14, 0, 1) ** 1.5
    return _VIG[res]


def boundary_mask(g, res, width):
    m = Image.new("L", (res, res), 0)
    dr = ImageDraw.Draw(m)
    n = g.shape[0]
    for line in contour_generator(z=g).lines(0.0):
        pts = [((p[0] / (n - 1)) * res, (1 - p[1] / (n - 1)) * res) for p in line]
        if len(pts) > 1:
            dr.line(pts, fill=255, width=width, joint="curve")
    return np.asarray(m, dtype=float) / 255


def panel(fr, d, res, gain):
    """RGB float image (res x res) for recorded frame index fr of run d."""
    R = float(d["R"])
    img = np.ones((res, res, 3)) * NIGHT
    # decision regions: a faint wash of the class colour
    g = d["grid"][fr].reshape(int(np.sqrt(d["grid"].shape[1])), -1)
    gz = zoom(g, res / g.shape[0], order=1)
    t = np.tanh(2.5 * gz)
    img += 0.07 * np.clip(t, 0, 1)[..., None] * AMBER + 0.07 * np.clip(-t, 0, 1)[..., None] * CYAN
    dens = line_density(d["w"][fr], d["b"][fr], d["a"][fr], R, res)
    for k, col in enumerate([AMBER, CYAN]):
        v = dens[k] * gain
        core = 1 - np.exp(-v)
        glow = 1 - np.exp(-gaussian_filter(v, res / 160) * 1.4)
        img += (0.85 * core + 0.35 * glow)[..., None] * col
    bm = boundary_mask(g, res, max(2, res // 330))
    bglow = gaussian_filter(bm, res / 200)
    vig = vignette(res, R)[::-1]
    img = NIGHT + (img - NIGHT) * vig[..., None]
    img = img + (0.9 * bm + 0.6 * bglow)[..., None] * vig[..., None]
    return img, g, gz


def draw_points(canvas, x, y, R, x0, y0, res, radius):
    dr = ImageDraw.Draw(canvas, "RGBA")
    for (px, py), lab in zip(x, y):
        cx = x0 + (px + R) / (2 * R) * res
        cy = y0 + (1 - (py + R) / (2 * R)) * res
        col = (255, 196, 120, 235) if lab > 0 else (140, 214, 250, 235)
        dr.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=col)


def draw_boundary(canvas, g, R, x0, y0, res, width):
    dr = ImageDraw.Draw(canvas, "RGBA")
    n = g.shape[0]
    gen = contour_generator(z=g)
    for line in gen.lines(0.0):
        pts = [(x0 + (p[0] / (n - 1)) * res, y0 + (1 - p[1] / (n - 1)) * res) for p in line]
        if len(pts) > 1:
            dr.line(pts, fill=(255, 255, 255, 90), width=width * 3, joint="curve")
            dr.line(pts, fill=(255, 255, 255, 245), width=width, joint="curve")


def movement(d, fr):
    th0 = np.concatenate([d["w"][0].ravel(), d["b"][0], d["a"][0]])
    th = np.concatenate([d["w"][fr].ravel(), d["b"][fr], d["a"][fr]])
    return np.linalg.norm(th - th0) / np.linalg.norm(th0)


def compose(frl, frr, L, Rn, SS=2, W=1920, H=1080, fade=1.0):
    Wp, Hp = W * SS, H * SS
    res = 820 * SS
    top = 136 * SS
    xl, xr = (W // 2 - 820 - 40) * SS, (W // 2 + 40) * SS
    canvas = Image.new("RGB", (Wp, Hp), tuple((NIGHT * 255).astype(int)))
    gains = {}
    for d, fr, x0, name in [(L, frl, xl, "lazy"), (Rn, frr, xr, "rich")]:
        img, g, _ = panel(fr, d, res, gain=0.55)
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        canvas.paste(im, (x0, top))
        draw_points(canvas, d["x"], d["y"], float(d["R"]), x0, top, res, 2.6 * SS)
    dr = ImageDraw.Draw(canvas, "RGBA")
    ink, muted = (236, 235, 230), (138, 141, 156)
    for d, fr, x0, title, sub in [(L, frl, xl, "LAZY", f"output scale α = {float(L['alpha']):g}"),
                                  (Rn, frr, xr, "RICH", f"output scale α = {float(Rn['alpha']):g}")]:
        dr.text((x0, 58 * SS), title, font=font(44 * SS, "Bold"), fill=ink)
        tw = dr.textlength(title, font=font(44 * SS, "Bold"))
        dr.text((x0 + tw + 20 * SS, 78 * SS), sub, font=font(22 * SS), fill=muted)
        mv = movement(d, fr)
        lab = f"weights moved  {100 * mv:,.1f}%"
        dr.text((x0, top + res + 16 * SS), lab, font=font(25 * SS, "Medium"), fill=ink)
        acc = d["acc"][fr]
        rtxt = f"train accuracy {100 * acc:3.0f}%"
        rw = dr.textlength(rtxt, font=font(22 * SS))
        dr.text((x0 + res - rw, top + res + 19 * SS), rtxt, font=font(22 * SS), fill=muted)
        # log-scale movement bar: 0.1% ... 10,000%
        by = top + res + 62 * SS
        lo, hi = -3, 2
        frac = np.clip((np.log10(max(mv, 1e-4)) - lo) / (hi - lo), 0, 1)
        dr.rounded_rectangle([x0, by, x0 + res, by + 5 * SS], radius=3 * SS, fill=(34, 38, 58))
        dr.rounded_rectangle([x0, by, x0 + max(6 * SS, int(res * frac)), by + 5 * SS], radius=3 * SS,
                             fill=(224, 86, 31) if title == "RICH" else (95, 140, 205))
        for e, tl in zip(range(lo, hi + 1), ["0.1%", "1%", "10%", "100%", "1,000%", "10,000%"]):
            tx = x0 + res * (e - lo) / (hi - lo)
            dr.line([tx, by + 9 * SS, tx, by + 15 * SS], fill=(90, 94, 112), width=SS)
            f16 = font(15 * SS)
            dr.text((tx - dr.textlength(tl, font=f16) / 2, by + 18 * SS), tl, font=f16, fill=(110, 114, 132))
    step = int(Rn["step"][frr])
    st = f"same 1,024-neuron network  ·  same data  ·  gradient step {step:,}"
    f20 = font(20 * SS)
    dr.text((W * SS // 2 - dr.textlength(st, font=f20) // 2, 18 * SS), st, font=f20, fill=muted)
    out = canvas.resize((W, H), Image.LANCZOS)
    if fade < 1:
        out = Image.blend(Image.new("RGB", (W, H), tuple((NIGHT * 255).astype(int))), out, fade)
    return out


def main():
    L = dict(np.load(f"{HERE}/cache/toy/hero_lazy.npz"))
    Rn = dict(np.load(f"{HERE}/cache/toy/hero_rich.npz"))
    assert np.array_equal(L["step"], Rn["step"])
    n = len(L["step"])
    if "--test" in sys.argv:
        os.makedirs(f"{HERE}/_preview", exist_ok=True)
        for fr in [0, n // 4, n // 2, n - 1]:
            compose(fr, fr, L, Rn).save(f"{HERE}/_preview/hero_f{fr}.png")
            print("frame", fr)
        return
    fps = 30
    idx = list(range(n)) + [n - 1] * int(1.8 * fps)
    fades = [1.0] * len(idx)
    tail = int(0.6 * fps)
    for k in range(tail):
        fades[-tail + k] = 1 - (k + 1) / tail
    head = int(0.4 * fps)
    for k in range(head):
        fades[k] = (k + 1) / head
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = f"{HERE}/figures/hero.mp4"
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
                          "-preset", "slow", "-movflags", "+faststart", mp4], stdin=subprocess.PIPE)
    for k, (fr, fa) in enumerate(zip(idx, fades)):
        im = compose(fr, fr, L, Rn, fade=fa)
        p.stdin.write(im.tobytes())
        if fr == n - 1 and fa == 1.0 and k == n - 1:
            im.save(f"{HERE}/figures/hero_still.png")
        if k % 50 == 0:
            print(k, "/", len(idx), flush=True)
    p.stdin.close(); p.wait()
    print(mp4)


if __name__ == "__main__":
    main()
