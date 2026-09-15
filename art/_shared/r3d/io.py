"""Image and film output. PNG only for data images (never JPEG)."""
import subprocess

import numpy as np
from PIL import Image


def save_png(path, img):
    a = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    a = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(a).save(path)


def glow_tonemap(x, exposure):
    import torch
    x = torch.as_tensor(x)
    return 1 - torch.exp(-exposure * x)


def write_film(pattern, mp4, fps=30, gif=None, gif_width=540):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(pattern),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    str(mp4)], check=True)
    if gif is not None:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(pattern), "-vf",
                        f"fps={min(fps, 15)},scale={gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];"
                        "[b][p]paletteuse=dither=bayer:bayer_scale=3", str(gif)], check=True)
