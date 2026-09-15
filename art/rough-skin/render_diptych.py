"""Diptych: Heaviside L = 1 against its ReLU twin (same weights, same level rule, same camera, same voxel render path
and lighting), composed from the cast PNGs made by render_casts.py.

    render_diptych.py [exterior|cutaway]  -> gallery/diptych_heaviside_L1_vs_relu[_cutaway].png
"""
import json, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.append("/home/fzeng/ml/research/art/depth-roughness")
from render_common import font      # source: art/depth-roughness/render_common.py

mode = sys.argv[1] if len(sys.argv) > 1 else "exterior"
sfx = "" if mode == "exterior" else "_cutaway"
A = Image.open(f"gallery/cast_heaviside_L1_voxel{sfx}.png")
B = Image.open(f"gallery/cast_relu_L1_voxel{sfx}.png")
S = A.size[0]
bg = tuple(int(0.975 * 255) for _ in range(3))
ink = (34, 33, 38)
top, bottom, gap = 260, 330, 40
W = 2 * S + gap; H = top + S + bottom
im = Image.new("RGB", (W, H), bg)
im.paste(A, (0, top)); im.paste(B, (S + gap, top))
dr = ImageDraw.Draw(im)
dr.text((90, 70), "Same weights, one activation apart", font=font(96), fill=ink)
dr.text((90, 185), "{T ≤ median} for a width-4096 network on a 0.5 rad exp-map cube of S³, 256³ crisp voxels, orthographic",
        font=font(46, "italic"), fill=ink)
dr.text((90, top + S + 30), "Heaviside, depth 1: D = 2.426 ± 0.031  (theory 2.5)", font=font(64), fill=ink)
dr.text((S + gap + 90, top + S + 30), "ReLU, depth 1: a piecewise-linear surface, dim 2", font=font(64), fill=ink)
dr.text((90, top + S + 140), "Measured D: 3-draw mean ± sd, calibrated 3D box counting, width 4096, 256³ (1 + slice: 2.448 ± 0.067). "
        "Plaster, one raking light, hard shadow and AO show form only.", font=font(42, "italic"), fill=ink)
dr.text((90, top + S + 205), "Darker faces are cuts (the cube boundary" + (" and the cutaway quadrant" if sfx else "") +
        "), not skin. ReLU terraces are voxel steps of a smooth surface: the null through the identical path.",
        font=font(42, "italic"), fill=ink)
out = f"gallery/diptych_heaviside_L1_vs_relu{sfx}.png"
im.save(out); print(out, im.size)
