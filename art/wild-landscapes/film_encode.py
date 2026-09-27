"""Encode film frames (cache/film/<name>/f%04d.png) to gallery/film_<name>.mp4 (H.264, 1080p, 24 fps),
burning in a small measured-scale label (declared apparatus) in the lower-left corner.

    python film_encode.py zoom      label: magnification and window half-width (plane units)
    python film_encode.py horizon   label: unroll horizon T
"""
import os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def font(sz):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def label(name, k, n):
    if name == "zoom":
        from rnn_film import window
        s, c = window(k, n)
        mag = 2.5 / s
        return f"x{mag:,.0f}   window +-{s:.2e}   T = 30"
    if name == "horizon":
        hold = int(os.environ.get("HOLD", "2")); Tmax = int(os.environ.get("TMAX", "100"))
        return f"T = {min(1 + k // hold, Tmax)} steps"
    return ""


def main():
    name = sys.argv[1]
    src = os.path.join(HERE, "cache", "film", name)
    fs = sorted(f for f in os.listdir(src) if f.endswith(".png"))
    n_all = len(fs)
    idx = list(range(n_all))
    if name == "zoom":        # re-timing (declared): 1:1 up to x1000, then every 3rd window, then the hold
        idx = list(range(0, 240)) + list(range(240, 360, 3)) + [359] * 24
    if name == "horizon":     # start at T = 4, the first step with a non-zero target (y_t = x_{t-3})
        idx = list(range(6, n_all))
    fs = [fs[i] for i in idx]
    out = os.path.join(HERE, "gallery", f"film_{name}.mp4")
    first = Image.open(os.path.join(src, fs[0])); W, H = first.size
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24", "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-maxrate", "6M", "-bufsize", "12M", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    fnt = font(max(14, H // 50))
    for k, f in zip(idx, fs):
        im = Image.open(os.path.join(src, f)).convert("RGB")
        t = label(name, k, n_all)
        if t:
            d = ImageDraw.Draw(im)
            d.text((H // 40, H - H // 40 - fnt.size), t, fill=(235, 228, 215), font=fnt)
        p.stdin.write(np.asarray(im).tobytes())
    p.stdin.close(); p.wait()
    print(out, os.path.getsize(out) / 1e6, "MB", len(fs), "frames")


if __name__ == "__main__":
    main()
