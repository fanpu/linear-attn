"""Render the continuous bit-depth sweep as MP4 (1920x1080, silent) and a small GIF.
Colour: magma, black point fixed at -95 dB, white at -5 dB re full-scale sine (one scale for all frames, declared)."""
import os
import subprocess
import numpy as np
from PIL import Image
import render_common as rc

D = np.load(f"{rc.CACHE}/bitsweep.npz")
Fr, ratio, sinad = D["frames"], D["ratio"], D["sinad"]
W, H = 1920, 1080
out = f"{rc.GAL}/bitsweep.mp4"
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "15", "-i", "-", "-vf", "scale=1280:720",
       "-c:v", "libx264", "-preset", "slow", "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
small = []
order = list(range(len(Fr))) + [len(Fr) - 1] * 15
for j, i in enumerate(order):
    v = rc.unit(Fr[i].astype(np.float32), -95, -5, 0.9)
    img = np.array(Image.fromarray((v * 65535).astype(np.uint16)).resize((1840, 900), Image.BOX)) / 65535
    cv = rc.canvas(W, H, (0.02, 0.018, 0.03))
    rc.paste(cv, rc.cmap_rgb(img, "magma"), 40, 90)
    r = ratio[i]
    rc.text(cv, (40, 22), f"peak = {r:7.2f} LSB    levels spanned = {2 * r + 1:7.1f}    effective bits = {np.log2(2 * r + 1):5.2f}    "
            f"SINAD = {sinad[i]:5.1f} dB", 36, (0.9, 0.87, 0.82), rc.FONT_MONO)
    rc.text(cv, (40, 1010), "undithered log chirp 20 Hz - 24 kHz, 30 s; step size shrinks smoothly (no dither); STFT 2048 Blackman-Harris; "
            "colour [-95, -5] dB re full scale, fixed.  numpy float64, 2026-09-13", 22, (0.7, 0.68, 0.64), rc.FONT_MONO)
    arr = np.array(cv)
    proc.stdin.write(arr.tobytes())
    if j % 3 == 0:
        small.append(Image.fromarray(arr).resize((480, 270), Image.LANCZOS))
proc.stdin.close(); proc.wait()
small[0].save("/tmp/bitsweep.gif", save_all=True, append_images=small[1:], duration=200, loop=0)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "/tmp/bitsweep.gif", "-vf",
                "split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3", f"{rc.GAL}/bitsweep.gif"], check=True)
print(os.path.getsize(out) / 1e6, os.path.getsize(f"{rc.GAL}/bitsweep.gif") / 1e6)
