"""Zoom sequence: nested windows around a centre (T0, p0), halving the window each level.
Each window is decoded at the same resolution with the same shared uniforms, so level k+1
is a true 2x magnification of the middle of level k.

  python zoom.py --name zA --T0 0.62 --p0 0.71 --w0 1.0 --levels 12 --res 128 --L 64
"""
import argparse
import os
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--prompt", default="story")
p.add_argument("--T0", type=float, required=True)
p.add_argument("--p0", type=float, required=True)
p.add_argument("--w0", type=float, default=1.0, help="window width in T at level 0 (height = w0 * hscale)")
p.add_argument("--hscale", type=float, default=2 / 3, help="p-extent / T-extent of the window")
p.add_argument("--levels", type=int, default=12)
p.add_argument("--start", type=int, default=0)
p.add_argument("--res", type=int, default=128)
p.add_argument("--L", type=int, default=64)
a = p.parse_args()
env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.abspath(__file__)), OMP_NUM_THREADS="4")
for k in range(a.start, a.levels):
    wT = a.w0 / 2 ** k
    wp = wT * a.hscale
    x0, x1 = a.T0 - wT / 2, a.T0 + wT / 2
    y0, y1 = a.p0 - wp / 2, a.p0 + wp / 2
    out = f"cache/zoom_{a.name}_{k:02d}.npz"
    if os.path.exists(out):
        continue
    cmd = ["../.venv/bin/python", "decode.py", "--prompt", a.prompt, "--res", str(a.res), "--L", str(a.L),
           "--x", repr(x0), repr(x1), "--y", repr(y0), repr(y1), "--out", out]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
